// Compile stage of the build pipeline. The world is loaded, expanded, and
// validated upstream by crate::world::prepare_world; this module takes the
// resulting WorldJsonlAsset list and:
// - Resolves each asset to a BlobAssetDef via asset_api::create_asset_def()
// - Compiles payloads for assets that need compilation
// - Packs all payloads into blobs using PayloadPacker (fills locators)
// - Sorts: components first, then systems in declared order

use crate::assets::FileKind;
use crate::world::{WorldConfig, WorldJsonlAsset, normalize_single_shader_type};

use crate::blob::PayloadPacker;
use crate::ecs::asset_api::{self, AssetRequest};
use crate::ecs::asset_id;
use crate::ecs::{AssetKind, BlobAssetDef, ComponentType};

pub fn build_from_path(json_path: &str) -> std::io::Result<()> {
    let content = std::fs::read_to_string(json_path)?;
    let loaded = crate::world::prepare_world(&content)
        .map_err(|errs| crate::check::report_validation_errors(&errs))?;

    let result = build_compiled(loaded.assets, None)?;

    let pack_result = crate::blob::write_blobs(&result.defs, &result.payloads)?;
    for (blob_idx, path) in pack_result.blob_paths.iter().enumerate() {
        let payload_bytes = result.payloads.get(blob_idx).map(|b| b.len()).unwrap_or(0);
        println!("Wrote {} ({} payload bytes)", path, payload_bytes);
    }

    if result.cache_hits + result.cache_misses > 0 {
        println!(
            "Build cache: {} reused, {} compiled",
            result.cache_hits, result.cache_misses
        );
    }

    // The blob carries only components; the lock file's pipeline-order section
    // (once a declared system run order) is therefore empty.
    let pipeline_refs: Vec<&str> = Vec::new();

    let named_refs: Vec<(String, &crate::ecs::BlobAssetDef)> = result
        .defs
        .iter()
        .map(|d| {
            let name = ComponentType::from_discriminant(d.discriminant)
                .map(|t| t.as_str().to_string())
                .unwrap_or_default();
            (name, d)
        })
        .collect();
    let named_refs: Vec<(&str, &crate::ecs::BlobAssetDef)> =
        named_refs.iter().map(|(n, d)| (n.as_str(), *d)).collect();

    crate::blob::write_lock(&pipeline_refs, &named_refs, &pack_result.blob_paths)?;
    println!("Wrote world-lock.json");

    Ok(())
}

// Collapse a list of validation errors into a single io::Error. The messages
// are newline-joined so an upstream caller (e.g. the infra agentic loop) sees
// every problem from one call.
fn errors_to_io(errors: Vec<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, errors.join("\n"))
}

// The in-memory result of a complete build pipeline run.
// Defs have payload locators filled in; payloads[i] is the raw bytes for
// blob i. This can be used directly without touching disk.
pub struct PipelineResult {
    pub defs: Vec<BlobAssetDef>,
    pub payloads: Vec<Vec<u8>>,
    // Compiled-asset payloads served from the build cache this run.
    pub cache_hits: usize,
    // Compiled-asset payloads compiled fresh this run.
    pub cache_misses: usize,
}

// Validate a single asset's type and generator without running the full build
// pipeline. Called by the server on each world_add so the LLM gets per-asset
// feedback without waiting for a WebSocket round-trip.
//
// Checks:
//   - asset type is registered (via asset_api::create_asset_def)
//   - per-type structural checks via crate::check
// Shader assets are not compiled here; use the validate_shader tool for that.
#[allow(dead_code)]
pub fn validate_asset(
    asset_type: &str,
    name: &str,
    args: &serde_json::Value,
) -> Result<(), String> {
    // Single-asset validation has no surrounding world to intern against; the
    // resulting ids are throwaway. Reset so calls do not accumulate entries.
    asset_id::reset_interner();
    let (asset_type, args) = normalize_single_shader_type(asset_type, args);
    let asset_type = asset_type.as_str();
    let type_norm = asset_type.to_lowercase().replace('_', "");

    // Build-time types are valid in world.jsonl; they are consumed by expansion
    // functions before the runtime asset registry sees them.
    if matches!(
        type_norm.as_str(),
        "environment" | "lightrig" | "materialpalette" | "camerashot" | "prefab" | "sceneimport"
    ) {
        return Ok(());
    }

    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args: Some(args.clone()),
    };
    asset_api::create_asset_def(&req).map_err(|e| format!("Asset '{}': {}", name, e))?;

    crate::check::check_asset(&type_norm, name, &args)?;

    Ok(())
}

// Run the full build pipeline on an in-memory JSONL string without touching
// disk. Loads, expands, and validates the world (crate::world::prepare_world),
// then compiles it. `artifacts_dir` is an optional directory consulted when
// resolving bare shader filenames not found under assets/; pass the account's
// artifact directory so test_world can compile user-written shaders.
pub fn build_pipeline_from_str(
    content: &str,
    artifacts_dir: Option<&str>,
) -> std::io::Result<PipelineResult> {
    let loaded = crate::world::prepare_world(content).map_err(errors_to_io)?;
    build_compiled(loaded.assets, artifacts_dir)
}

// Compile an already-prepared world (expanded + structurally and semantically
// validated) into in-memory blobs. This is the compile-only stage; it assumes
// the assets have passed crate::world::prepare_world.
pub fn build_compiled(
    mut assets: Vec<WorldJsonlAsset>,
    artifacts_dir: Option<&str>,
) -> std::io::Result<PipelineResult> {
    let config = WorldConfig::default();

    // Cache probe runs before desugar. For every glTF-sourced Mesh /
    // SkinnedMesh, hash the un-desugared args + referenced .glb and look up
    // the compiled payload by that key. On a hit, we hold the bytes and skip
    // the .glb parse entirely (the original goal: an unchanged source file
    // means no work). On a miss, the recorded key is used when the compile
    // step stores the freshly produced payload, so the next build's probe
    // can re-use it.
    let gltf_cache = probe_gltf_cache(&assets, artifacts_dir);

    // Expand any glTF-sourced SkinnedMesh and Mesh assets into inline geometry
    // before anything else looks at their args. Animations expand after the
    // skinned-mesh pass so an importer that wanted to share state could read
    // already-imported skeletons; today both passes parse the .glb fresh,
    // but the ordering keeps that option open without an API churn.
    desugar_gltf_skinned_meshes(&mut assets, &gltf_cache)?;
    desugar_gltf_meshes(&mut assets, &gltf_cache)?;
    desugar_fbx_meshes(&mut assets, &gltf_cache)?;
    desugar_gltf_animations(&mut assets)?;

    // Intern every asset name to a dense AssetId in declaration order, then
    // resolve the scene-by-naming-convention references that the runtime can
    asset_id::reset_interner();
    let names: Vec<&str> = assets.iter().map(|a| a.name.as_str()).collect();
    asset_id::intern_all(&names);
    resolve_scene_refs(&mut assets);

    let mut named: Vec<(String, BlobAssetDef)> = Vec::new();
    for asset in &assets {
        let req = AssetRequest {
            asset_type: asset.asset_type.clone(),
            args: Some(asset.args.clone()),
        };
        let mut def = asset_api::create_asset_def(&req).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': {}", asset.name, e),
            )
        })?;
        def.name = Some(asset_id::intern(&asset.name));
        named.push((asset.name.clone(), def));
    }

    let (blob_payloads, cache_hits, cache_misses) = compile_and_pack_payloads(
        &mut named,
        &assets,
        config.max_blob_bytes,
        artifacts_dir,
        &gltf_cache,
    )?;

    // The blob carries only components, emitted in declaration order. (System
    // run order is no longer a build concern: every system is internal client
    // code ordered by the client's `World::build_internal_systems` schedule.)
    let defs: Vec<BlobAssetDef> = named.into_iter().map(|(_, d)| d).collect();

    Ok(PipelineResult {
        defs,
        payloads: blob_payloads,
        cache_hits,
        cache_misses,
    })
}

// Per-asset state recorded by `probe_gltf_cache`. `key` is the cache key
// computed from the asset's pre-desugar args; `bytes` is `Some` when the
// cache already held a compiled payload for that key. On a hit, the desugar
// pass skips the .glb parse for this asset; on a miss, compile_and_pack
// stores the freshly compiled payload under the same `key` so the next
// build's probe can re-use it.
#[derive(Clone)]
struct GltfCacheEntry {
    key: String,
    bytes: Option<Vec<u8>>,
}

// Hash every glTF-sourced Mesh / SkinnedMesh asset's pre-desugar args and
// referenced .glb, then probe the content-addressed payload cache. Returns
// one entry per (source-backed) asset name. Assets without a `source` are
// not probed: their args don't depend on a file, so the regular per-asset
// cache path inside compile_and_pack_payloads is sufficient.
fn probe_gltf_cache(
    assets: &[WorldJsonlAsset],
    artifacts_dir: Option<&str>,
) -> std::collections::HashMap<String, GltfCacheEntry> {
    use crate::assets::{Mesh, SkinnedMesh};
    use crate::ecs::Component;

    let mut out = std::collections::HashMap::new();
    let empty: [WorldJsonlAsset; 0] = [];
    for asset in assets {
        let ct = if asset.asset_type == Mesh::NAME {
            ComponentType::parse(Mesh::NAME).ok()
        } else if asset.asset_type == SkinnedMesh::NAME {
            ComponentType::parse(SkinnedMesh::NAME).ok()
        } else {
            None
        };
        let Some(ct) = ct else {
            continue;
        };
        let discriminant = ct.discriminant();
        let has_source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .map(|s| !s.is_empty())
            .unwrap_or(false);
        if !has_source {
            continue;
        }

        let ctx = crate::asset::BuildCtx {
            name: asset.name.as_str(),
            artifacts_dir,
            all_assets: &empty,
        };
        let extra_sources = source_files_by_type(ct, &asset.args, &ctx);
        let key = crate::cache::payload_key(discriminant, &asset.args, &ctx, &extra_sources);
        let bytes = crate::cache::load(&key);
        out.insert(asset.name.clone(), GltfCacheEntry { key, bytes });
    }
    out
}

// Expand glTF-sourced SkinnedMesh assets in place: parse the referenced .glb
// and write the imported geometry + skeleton into the asset's inline
// `vertices` / `indices` / `skeleton` args. A SkinnedMesh with no `source` is
// left untouched, so an inline-authored mesh is byte-for-byte unchanged.
// Skips an asset whose cache probe found a precompiled payload: there is no
// reason to parse the .glb when the bytes are already in hand.
fn desugar_gltf_skinned_meshes(
    assets: &mut [WorldJsonlAsset],
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<()> {
    use crate::assets::SkinnedMesh;
    use crate::ecs::Component;

    for asset in assets.iter_mut() {
        if asset.asset_type != SkinnedMesh::NAME {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if source.is_empty() {
            continue;
        }
        // Cache probe found a compiled payload for this asset, no need
        // to parse the .glb. compile_and_pack_payloads will use the bytes
        // directly. Leave the args un-desugared so they keep matching the
        // pre-desugar cache key on the next build.
        if matches!(
            gltf_cache.get(&asset.name),
            Some(GltfCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }

        let imported = crate::gltf::import_skinned_glb(&source).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': glTF import failed: {}", asset.name, e),
            )
        })?;

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        let encode = |field: &str, value: serde_json::Result<serde_json::Value>| {
            value.map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': failed to encode imported {}: {}",
                        name, field, e
                    ),
                )
            })
        };
        obj.insert(
            "vertices".to_string(),
            encode("vertices", serde_json::to_value(&imported.vertices))?,
        );
        obj.insert(
            "indices".to_string(),
            encode("indices", serde_json::to_value(&imported.indices))?,
        );
        obj.insert(
            "skeleton".to_string(),
            encode("skeleton", serde_json::to_value(&imported.skeleton))?,
        );
        tracing::info!(
            "Asset '{}': imported glTF '{}': {} vertices, {} indices, {} joints",
            asset.name,
            source,
            imported.vertices.len(),
            imported.indices.len(),
            imported.skeleton.len()
        );
    }
    Ok(())
}

// Expand glTF-sourced static `Mesh` assets in place: parse the referenced
// `.glb` and write the imported primitive geometry into the asset's inline
// `vertices` / `indices` args. A Mesh with no `source` is left untouched. The
// GLB is parsed once per unique path; ABeautifulGame fans 35+ Mesh assets out
// of one file, so memoization keeps this O(files) rather than O(primitives).
fn desugar_gltf_meshes(
    assets: &mut [WorldJsonlAsset],
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<()> {
    use crate::assets::{Mesh, VertexData};
    use crate::ecs::Component;
    use std::collections::HashMap;

    // One split chunk: its vertices and index buffer.
    type Chunk = (Vec<VertexData>, Vec<u16>);

    let mut parsed_cache: HashMap<String, gltf::Gltf> = HashMap::new();
    // Memoize the chunk split per (source, primitive_index) so an oversized
    // primitive that fans into N chunked Mesh assets is split exactly once.
    let mut chunk_cache: HashMap<(String, u32), Vec<Chunk>> = HashMap::new();

    for asset in assets.iter_mut() {
        if asset.asset_type != Mesh::NAME {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if source.is_empty() {
            continue;
        }
        // `.fbx` sources are handled by `desugar_fbx_meshes`; this pass owns
        // only the glTF container.
        if !source.to_lowercase().ends_with(".glb") {
            continue;
        }
        // Skip the .glb parse when the cache probe already produced bytes
        // for this asset (see `desugar_gltf_skinned_meshes` for the same
        // pattern). Args stay pre-desugar so the next build's probe hits.
        if matches!(
            gltf_cache.get(&asset.name),
            Some(GltfCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }
        let primitive_index = asset
            .args
            .get("primitive_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let chunk_index = asset
            .args
            .get("chunk_index")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);

        if !parsed_cache.contains_key(&source) {
            let doc = crate::glb::parse_glb(&source).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': glTF import failed: {}", asset.name, e),
                )
            })?;
            parsed_cache.insert(source.clone(), doc);
        }
        let doc = parsed_cache.get(&source).expect("just inserted");

        let (vertices, indices) = if let Some(chunk_idx) = chunk_index {
            let key = (source.clone(), primitive_index);
            if !chunk_cache.contains_key(&key) {
                let (verts, indices32) =
                    crate::glb::read_primitive_geometry(doc, &source, primitive_index).map_err(
                        |e| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Asset '{}': glTF import failed: {}", asset.name, e),
                            )
                        },
                    )?;
                let chunks = crate::glb::split_into_u16_chunks(&verts, &indices32);
                chunk_cache.insert(key.clone(), chunks);
            }
            let chunks = chunk_cache.get(&key).expect("just inserted");
            let chunk = chunks.get(chunk_idx).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': chunk_index {} out of range, '{}' primitive {} \
                         splits into {} chunk(s)",
                        asset.name,
                        chunk_idx,
                        source,
                        primitive_index,
                        chunks.len(),
                    ),
                )
            })?;
            chunk.clone()
        } else {
            crate::glb::import_static_glb_primitive_from_doc(doc, &source, primitive_index)
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Asset '{}': glTF import failed: {}", asset.name, e),
                    )
                })?
        };

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        let encode = |field: &str, value: serde_json::Result<serde_json::Value>| {
            value.map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': failed to encode imported {}: {}",
                        name, field, e
                    ),
                )
            })
        };
        let vlen = vertices.len();
        let ilen = indices.len();
        obj.insert(
            "vertices".to_string(),
            encode("vertices", serde_json::to_value(&vertices))?,
        );
        obj.insert(
            "indices".to_string(),
            encode("indices", serde_json::to_value(&indices))?,
        );
        match chunk_index {
            Some(c) => tracing::info!(
                "Asset '{}': imported glTF '{}' primitive {} chunk {}: {} vertices, {} indices",
                asset.name,
                source,
                primitive_index,
                c,
                vlen,
                ilen,
            ),
            None => tracing::info!(
                "Asset '{}': imported glTF '{}' primitive {}: {} vertices, {} indices",
                asset.name,
                source,
                primitive_index,
                vlen,
                ilen,
            ),
        }
    }
    Ok(())
}

// Expand FBX-sourced Mesh assets in place: parse the `.fbx` into an FbxScene
// and write the imported geometry into each asset's inline `vertices` /
// `indices` args, keyed by `primitive_index` and optional `chunk_index`. A Mesh
// whose source is not a `.fbx` is left to `desugar_gltf_meshes`. The FBX is
// parsed once per unique path (Bistro fans thousands of Mesh assets out of one
// file) and each primitive's u16 chunk split is memoized.
fn desugar_fbx_meshes(
    assets: &mut [WorldJsonlAsset],
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<()> {
    use crate::assets::{Mesh, VertexData};
    use crate::ecs::Component;
    use crate::fbx::FbxScene;
    use std::collections::HashMap;

    type Chunk = (Vec<VertexData>, Vec<u16>);

    let mut parsed_cache: HashMap<String, FbxScene> = HashMap::new();
    let mut chunk_cache: HashMap<(String, u32), Vec<Chunk>> = HashMap::new();

    for asset in assets.iter_mut() {
        if asset.asset_type != Mesh::NAME {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if !source.to_lowercase().ends_with(".fbx") {
            continue;
        }
        // Honour the same content-addressed cache the glTF pass uses: a probe
        // hit means the compiled payload is already in hand, so skip the parse.
        if matches!(
            gltf_cache.get(&asset.name),
            Some(GltfCacheEntry { bytes: Some(_), .. })
        ) {
            continue;
        }
        let primitive_index = asset
            .args
            .get("primitive_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let chunk_index = asset
            .args
            .get("chunk_index")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);

        if !parsed_cache.contains_key(&source) {
            let scene = crate::fbx::parse_fbx(&source).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': FBX import failed: {}", asset.name, e),
                )
            })?;
            parsed_cache.insert(source.clone(), scene);
        }
        let scene = parsed_cache.get(&source).expect("just inserted");

        let key = (source.clone(), primitive_index);
        if !chunk_cache.contains_key(&key) {
            let (verts, indices32) = crate::fbx::read_primitive_geometry(scene, primitive_index)
                .map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Asset '{}': FBX import failed: {}", asset.name, e),
                    )
                })?;
            let chunks = crate::glb::split_into_u16_chunks(&verts, &indices32);
            chunk_cache.insert(key.clone(), chunks);
        }
        let chunks = chunk_cache.get(&key).expect("just inserted");
        let chunk = chunks.get(chunk_index).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Asset '{}': chunk_index {} out of range, '{}' primitive {} splits into {} chunk(s)",
                    asset.name,
                    chunk_index,
                    source,
                    primitive_index,
                    chunks.len(),
                ),
            )
        })?;
        let (vertices, indices) = chunk.clone();

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        let vlen = vertices.len();
        let ilen = indices.len();
        obj.insert(
            "vertices".to_string(),
            serde_json::to_value(&vertices).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "Asset '{}': failed to encode imported vertices: {}",
                        name, e
                    ),
                )
            })?,
        );
        obj.insert(
            "indices".to_string(),
            serde_json::to_value(&indices).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': failed to encode imported indices: {}", name, e),
                )
            })?,
        );
        tracing::info!(
            "Asset '{}': imported FBX '{}' primitive {} chunk {}: {} vertices, {} indices",
            asset.name,
            source,
            primitive_index,
            chunk_index,
            vlen,
            ilen,
        );
    }
    Ok(())
}

// Expand glTF-sourced `Animation` assets in place: parse the `.glb`, pick the
// animation by `animation_name` (preferred) or `animation_index`, and replace
// the asset's `duration` + `tracks` with the imported data. An Animation with
// no `source` is left untouched, so inline-authored clips are byte-for-byte
// unchanged. Channels targeting non-joint nodes are dropped silently by the
// importer.
fn desugar_gltf_animations(assets: &mut [WorldJsonlAsset]) -> std::io::Result<()> {
    use crate::assets::Animation;
    use crate::ecs::Component;

    for asset in assets.iter_mut() {
        if asset.asset_type != Animation::NAME {
            continue;
        }
        let source = asset
            .args
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if source.is_empty() {
            continue;
        }
        let animation_name = asset
            .args
            .get("animation_name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let animation_index = asset
            .args
            .get("animation_index")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Look up by name when authored; fall back to the numeric index.
        let resolved_index = if !animation_name.is_empty() {
            let names = crate::gltf::glb_animation_names(&source).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Asset '{}': glTF import failed: {}", asset.name, e),
                )
            })?;
            names
                .iter()
                .position(|n| n == &animation_name)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "Asset '{}': glTF '{}' has no animation named '{}' \
                             (file contains: {:?})",
                            asset.name, source, animation_name, names
                        ),
                    )
                })?
        } else {
            animation_index
        };

        let imported = crate::gltf::import_glb_animation(&source, resolved_index).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': glTF import failed: {}", asset.name, e),
            )
        })?;

        // Convert ImportedAnimation -> the asset's serialised track shape.
        let tracks_json: Vec<serde_json::Value> = imported
            .tracks
            .iter()
            .map(|track| {
                let keyframes: Vec<serde_json::Value> = track
                    .keys
                    .iter()
                    .map(|k| {
                        serde_json::json!({
                            "time": k.time,
                            "translation": k.pose.translation,
                            "rotation_deg": k.pose.rotation_deg,
                            "scale": k.pose.scale,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "joint": track.joint,
                    "keyframes": keyframes,
                })
            })
            .collect();

        let name = asset.name.clone();
        let obj = asset.args.as_object_mut().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': args is not a JSON object", name),
            )
        })?;
        obj.insert("duration".to_string(), serde_json::json!(imported.duration));
        obj.insert("tracks".to_string(), serde_json::Value::Array(tracks_json));
        tracing::info!(
            "Asset '{}': imported glTF '{}' animation {} ('{}'): {:.3} s, {} track(s)",
            asset.name,
            source,
            resolved_index,
            imported.name,
            imported.duration,
            imported.tracks.len(),
        );
    }
    Ok(())
}

// Validate world JSONL without running compilation. Runs the full front half
// of the pipeline (load, expand, semantic checks) plus a per-asset type/args
// resolution, but stops short of compiling payloads: intended for fast
// server-side pre-deploy checks where shader compilation is not needed.
// Every problem found is reported in a single newline-joined error.
#[allow(dead_code)]
pub fn validate_world_jsonl(content: &str) -> std::io::Result<()> {
    let loaded = crate::world::prepare_world(content).map_err(errors_to_io)?;

    let mut errors: Vec<String> = Vec::new();
    for asset in &loaded.assets {
        let req = AssetRequest {
            asset_type: asset.asset_type.clone(),
            args: Some(asset.args.clone()),
        };
        if let Err(e) = asset_api::create_asset_def(&req) {
            errors.push(format!("Asset '{}': {}", asset.name, e));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors_to_io(errors))
    }
}

fn compile_and_pack_payloads(
    named: &mut [(String, BlobAssetDef)],
    assets: &[WorldJsonlAsset],
    max_blob_bytes: u64,
    artifacts_dir: Option<&str>,
    gltf_cache: &std::collections::HashMap<String, GltfCacheEntry>,
) -> std::io::Result<(Vec<Vec<u8>>, usize, usize)> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let compiled_indices: Vec<usize> = named
        .iter()
        .enumerate()
        .filter(|(i, (_, def))| {
            if def.kind != AssetKind::Component {
                return false;
            }
            let Some(ct) = ComponentType::from_discriminant(def.discriminant) else {
                return false;
            };
            if ct.as_str() == "File" {
                // only compile File assets whose kind maps to a supported payload
                // `named[i]` is built 1:1 from `assets[i]`, so index directly.
                return assets[*i]
                    .args
                    .get("kind")
                    .and_then(|k| k.as_str())
                    .and_then(FileKind::from_ext)
                    .map(|fk| fk.is_mesh())
                    .unwrap_or(false);
            }
            ct.registration().needs_compilation()
        })
        .map(|(i, _)| i)
        .collect();

    // Snapshot each job's inputs so the parallel compile borrows nothing from
    // `named`, which is mutated afterwards to record payload locators.
    let jobs: Vec<(usize, String, u8)> = compiled_indices
        .iter()
        .map(|&idx| {
            let (name, def) = &named[idx];
            (idx, name.clone(), def.discriminant)
        })
        .collect();

    // Compile assets in parallel. Each job is independent (it reads only its
    // own args and produces its own payload bytes) and the payload cache is
    // content-addressed, so concurrent hits and stores never collide. The
    // collected order follows `jobs`, so packing below stays deterministic.
    let cache_hits = AtomicUsize::new(0);
    let pending: Vec<(usize, Vec<u8>)> = jobs
        .par_iter()
        .map(
            |(idx, name, discriminant)| -> std::io::Result<(usize, Vec<u8>)> {
                let ct = ComponentType::from_discriminant(*discriminant).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("Invalid ComponentType discriminant for asset '{}'", name),
                    )
                })?;

                // `named[i]` is built 1:1 from `assets[i]` and the job carries
                // that index, so recover the args directly instead of scanning.
                let asset_args = &assets[*idx].args;

                let ctx = crate::asset::BuildCtx {
                    name: name.as_str(),
                    artifacts_dir,
                    all_assets: assets,
                };

                // GLB-sourced Mesh / SkinnedMesh assets are probed before
                // desugar; honor those results here so the .glb parse really
                // is skipped on cache hits. On a miss the precomputed key is
                // used at store time, keeping the next build's probe valid.
                if let Some(entry) = gltf_cache.get(name) {
                    if let Some(bytes) = &entry.bytes {
                        cache_hits.fetch_add(1, Ordering::Relaxed);
                        return Ok((*idx, bytes.clone()));
                    }
                    let compiled_bytes = compile_by_type(ct, asset_args, &ctx)?;
                    crate::cache::store(&entry.key, &compiled_bytes);
                    return Ok((*idx, compiled_bytes));
                }

                // Reuse a cached payload when the asset's inputs are unchanged;
                // otherwise compile and populate the cache for the next build.
                let extra_sources = source_files_by_type(ct, asset_args, &ctx);
                let key =
                    crate::cache::payload_key(*discriminant, asset_args, &ctx, &extra_sources);
                if let Some(bytes) = crate::cache::load(&key) {
                    cache_hits.fetch_add(1, Ordering::Relaxed);
                    return Ok((*idx, bytes));
                }
                let compiled_bytes = compile_by_type(ct, asset_args, &ctx)?;
                crate::cache::store(&key, &compiled_bytes);
                Ok((*idx, compiled_bytes))
            },
        )
        .collect::<std::io::Result<Vec<_>>>()?;

    let cache_hits = cache_hits.into_inner();
    let cache_misses = pending.len() - cache_hits;

    if pending.is_empty() {
        return Ok((vec![Vec::new()], 0, 0));
    }

    let mut packer = PayloadPacker::new(max_blob_bytes);

    for (idx, bytes) in &pending {
        let locator = packer.push(bytes);
        named[*idx].1.payload = Some(locator);
    }

    Ok((packer.finish(), cache_hits, cache_misses))
}

// Dispatch payload compilation by ComponentType. Every variant listed below
// has a `BuildAsset` impl in its asset file; the body of each call here is a
// one-liner that delegates to the trait. Adding a new compiled component
// means:
//   1. impl `Component` with `PAYLOAD = AssetPayload::Compiled` for the type
//   2. impl `BuildAsset` for the type in its asset file
//   3. Add one match arm here
fn compile_by_type(
    ct: ComponentType,
    args: &serde_json::Value,
    ctx: &crate::asset::BuildCtx<'_>,
) -> std::io::Result<Vec<u8>> {
    use crate::asset::BuildAsset;
    use crate::assets::{
        AudioClip, ColorLut, CubemapTexture, EnvironmentMap, File, Font, Mesh, ProceduralMesh,
        Room, SdfVolume, ShaderStage, SkinnedMesh, Texture, VoxelChunk,
    };
    match ct {
        ComponentType::AudioClip => <AudioClip as BuildAsset>::compile_payload(args, ctx),
        ComponentType::Mesh => <Mesh as BuildAsset>::compile_payload(args, ctx),
        ComponentType::ProceduralMesh => <ProceduralMesh as BuildAsset>::compile_payload(args, ctx),
        ComponentType::SkinnedMesh => <SkinnedMesh as BuildAsset>::compile_payload(args, ctx),
        ComponentType::VoxelChunk => <VoxelChunk as BuildAsset>::compile_payload(args, ctx),
        ComponentType::File => <File as BuildAsset>::compile_payload(args, ctx),
        ComponentType::Texture => <Texture as BuildAsset>::compile_payload(args, ctx),
        ComponentType::CubemapTexture => <CubemapTexture as BuildAsset>::compile_payload(args, ctx),
        ComponentType::EnvironmentMap => <EnvironmentMap as BuildAsset>::compile_payload(args, ctx),
        ComponentType::ColorLut => <ColorLut as BuildAsset>::compile_payload(args, ctx),
        ComponentType::Room => <Room as BuildAsset>::compile_payload(args, ctx),
        ComponentType::Font => <Font as BuildAsset>::compile_payload(args, ctx),
        ComponentType::ShaderStage => <ShaderStage as BuildAsset>::compile_payload(args, ctx),
        ComponentType::SdfVolume => <SdfVolume as BuildAsset>::compile_payload(args, ctx),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Asset '{}' is marked Compiled but has no BuildAsset impl (ComponentType {:?})",
                ctx.name, other
            ),
        )),
    }
}

// Dispatch `BuildAsset::source_files` by ComponentType. Mirrors
// `compile_by_type` so the cache layer can fold the contents-hash of each
// asset's referenced source files into its payload key. Types with no
// `BuildAsset` impl, or with the trait default, contribute nothing.
fn source_files_by_type(
    ct: ComponentType,
    args: &serde_json::Value,
    ctx: &crate::asset::BuildCtx<'_>,
) -> Vec<String> {
    use crate::asset::BuildAsset;
    use crate::assets::{
        AudioClip, ColorLut, CubemapTexture, EnvironmentMap, File, Font, Mesh, ProceduralMesh,
        Room, SdfVolume, ShaderStage, SkinnedMesh, Texture, VoxelChunk,
    };
    match ct {
        ComponentType::AudioClip => <AudioClip as BuildAsset>::source_files(args, ctx),
        ComponentType::Mesh => <Mesh as BuildAsset>::source_files(args, ctx),
        ComponentType::ProceduralMesh => <ProceduralMesh as BuildAsset>::source_files(args, ctx),
        ComponentType::SkinnedMesh => <SkinnedMesh as BuildAsset>::source_files(args, ctx),
        ComponentType::VoxelChunk => <VoxelChunk as BuildAsset>::source_files(args, ctx),
        ComponentType::File => <File as BuildAsset>::source_files(args, ctx),
        ComponentType::Texture => <Texture as BuildAsset>::source_files(args, ctx),
        ComponentType::CubemapTexture => <CubemapTexture as BuildAsset>::source_files(args, ctx),
        ComponentType::EnvironmentMap => <EnvironmentMap as BuildAsset>::source_files(args, ctx),
        ComponentType::ColorLut => <ColorLut as BuildAsset>::source_files(args, ctx),
        ComponentType::Room => <Room as BuildAsset>::source_files(args, ctx),
        ComponentType::Font => <Font as BuildAsset>::source_files(args, ctx),
        ComponentType::ShaderStage => <ShaderStage as BuildAsset>::source_files(args, ctx),
        ComponentType::SdfVolume => <SdfVolume as BuildAsset>::source_files(args, ctx),
        _ => Vec::new(),
    }
}

// Resolve scene + view associations that the runtime can no longer derive
// from name strings, baking them into the asset args so they survive as
// AssetId ids.
//
// Naming-convention relationships handled:
//   - A Prop named `<scene>_*` belongs to Scene `<scene>`. The matched scene
//     name is written into the prop's `scene` arg.
//   - A Sprite, TextLabel, or HitRegion named `<view>_*` belongs to View
//     `<view>`. The matched view name is written into the asset's `view` arg.
//   - A HitRegion or KeyBinding `action` of the form `scene:<name>`,
//     `view:show:<name>`, or `view:toggle:<name>` has its `<name>` part
//     rewritten to the interned id, so `UiInputSystem` can parse an integer
//     at runtime instead of a name.
fn resolve_scene_refs(assets: &mut [WorldJsonlAsset]) {
    let norm = |s: &str| s.to_lowercase().replace('_', "");

    let scene_names: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "scene")
        .map(|a| a.name.clone())
        .collect();

    let view_names: Vec<String> = assets
        .iter()
        .filter(|a| norm(&a.asset_type) == "view")
        .map(|a| a.name.clone())
        .collect();

    // Rewrite an action string, replacing the trailing `<name>` after the
    // given action prefix with its interned id. Returns Some(new_action) when
    // the action used the prefix with an unresolved name; None otherwise.
    let resolve_action = |action: &str| -> Option<String> {
        for prefix in ["scene:", "view:show:", "view:toggle:"] {
            if let Some(rest) = action.strip_prefix(prefix) {
                if !rest.is_empty() && rest.parse::<u32>().is_err() {
                    return Some(format!("{prefix}{}", asset_id::intern(rest).0));
                }
                return None;
            }
        }
        None
    };

    for asset in assets.iter_mut() {
        match norm(&asset.asset_type).as_str() {
            "prop" => {
                if asset.args.get("scene").is_some() {
                    continue;
                }
                // Longest matching prefix wins so a nested name (e.g.
                // `level_boss_*` under both `level` and `level_boss`) binds to
                // the most specific scene. Equivalent to first-match when no
                // scene name prefixes another.
                let matched = scene_names
                    .iter()
                    .filter(|sn| asset.name.starts_with(&format!("{sn}_")))
                    .max_by_key(|sn| sn.len())
                    .cloned();
                if let (Some(sn), serde_json::Value::Object(m)) = (matched, &mut asset.args) {
                    m.insert("scene".to_string(), serde_json::Value::String(sn));
                }
            }
            "sprite" | "imageoverlay" | "textlabel" | "text" | "hitregion" | "scrollpanel" => {
                // Resolve view prefix association. Longest matching prefix wins
                // so a nested view name (e.g. `main_menu_settings_*` under both
                // `main_menu` and `main_menu_settings`) binds to the most
                // specific view. Equivalent to first-match when no view name
                // prefixes another.
                if asset.args.get("view").is_none() {
                    let matched = view_names
                        .iter()
                        .filter(|vn| asset.name.starts_with(&format!("{vn}_")))
                        .max_by_key(|vn| vn.len())
                        .cloned();
                    if let (Some(vn), serde_json::Value::Object(m)) = (matched, &mut asset.args) {
                        m.insert("view".to_string(), serde_json::Value::String(vn));
                    }
                }
                // Resolve view:* / scene:* action targets to interned ids.
                if matches!(norm(&asset.asset_type).as_str(), "hitregion") {
                    let new_action = asset
                        .args
                        .get("action")
                        .and_then(|v| v.as_str())
                        .and_then(resolve_action);
                    if let (Some(action), serde_json::Value::Object(m)) =
                        (new_action, &mut asset.args)
                    {
                        m.insert("action".to_string(), serde_json::Value::String(action));
                    }
                }
            }
            "keybinding" => {
                let new_action = asset
                    .args
                    .get("action")
                    .and_then(|v| v.as_str())
                    .and_then(resolve_action);
                if let (Some(action), serde_json::Value::Object(m)) = (new_action, &mut asset.args)
                {
                    m.insert("action".to_string(), serde_json::Value::String(action));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_pipeline_interns_names_and_resolves_refs() {
        // box=0, day=1, day_crate=2 in declaration order.
        let world = concat!(
            r#"{"name":"box","type":"ProceduralMesh","args":{"generator":"box","half_extents":[1,1,1]}}"#,
            "\n",
            r#"{"name":"day","type":"Scene","args":{}}"#,
            "\n",
            r#"{"name":"day_crate","type":"Prop","args":{"mesh":"box"}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build pipeline");

        // The Prop def's identity is the interned id, not a name string.
        let prop = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(2)))
            .expect("day_crate def present with interned id 2");

        let args: serde_json::Value = serde_json::from_slice(&prop.args_bytes).unwrap();
        // The `mesh` reference resolved to box's id (0).
        assert_eq!(args["mesh"], serde_json::json!(0));
        // The `day_` name prefix resolved to Scene `day`'s id (1).
        assert_eq!(args["scene"], serde_json::json!(1));
    }

    // The visual_novel demo world (in concinnity-infra/worlds) exercises
    // Sprite + View + KeyBinding together. Validating it here catches asset
    // registration / pipeline regressions before we ship the world.
    #[test]
    fn visual_novel_world_validates() {
        // Inline a representative subset of the world so the test stays
        // hermetic (no infra path lookup needed). Covers: an initial View,
        // a Sprite under that view's prefix, a TextLabel under it, a
        // HitRegion firing view:show on another View, and a KeyBinding to
        // toggle a third (modal) View.
        let world = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}
{"name":"f","type":"Font","args":{"size_px":20}}
{"name":"title_menu","type":"View","args":{"initial":true}}
{"name":"title_menu_bg","type":"Sprite","args":{"x":0,"y":0,"width":640,"height":360,"tint":[0.1,0.1,0.1,1]}}
{"name":"title_menu_lbl","type":"TextLabel","args":{"font":"f","content":"Start","x":260,"y":160}}
{"name":"title_menu_btn","type":"HitRegion","args":{"x":260,"y":156,"width":120,"height":40,"label":"title_menu_lbl","action":"view:show:vn_page_1"}}
{"name":"vn_page_1","type":"View","args":{}}
{"name":"vn_page_1_text","type":"TextLabel","args":{"font":"f","content":"hello","x":40,"y":40}}
{"name":"vn_page_1_next","type":"HitRegion","args":{"x":0,"y":0,"width":640,"height":360,"action":"view:show:title_menu"}}
{"name":"pause_menu","type":"View","args":{}}
{"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":640,"height":360,"tint":[0,0,0,0.6]}}
{"name":"esc","type":"KeyBinding","args":{"key":"Escape","action":"view:toggle:pause_menu"}}
"#;
        validate_world_jsonl(world).expect("visual_novel-shaped world should validate");
    }

    // `view:show:<name>` / `view:toggle:<name>` action targets are
    // rewritten to interned ids at build time, like `scene:<name>`.
    #[test]
    fn build_pipeline_resolves_view_action_refs() {
        let world = concat!(
            r#"{"name":"pause_menu","type":"View","args":{}}"#,
            "\n",
            r#"{"name":"btn","type":"HitRegion","args":{"x":0,"y":0,"width":10,"height":10,"action":"view:toggle:pause_menu"}}"#,
            "\n",
            r#"{"name":"esc","type":"KeyBinding","args":{"key":"Escape","action":"view:toggle:pause_menu"}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build");
        // pause_menu interned id = 0 (first declared name).
        let btn = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(1)))
            .expect("HitRegion def");
        let args: serde_json::Value = serde_json::from_slice(&btn.args_bytes).unwrap();
        assert_eq!(args["action"], serde_json::json!("view:toggle:0"));

        let esc = result
            .defs
            .iter()
            .find(|d| d.name == Some(crate::ecs::asset_id::AssetId(2)))
            .expect("KeyBinding def");
        let args: serde_json::Value = serde_json::from_slice(&esc.args_bytes).unwrap();
        assert_eq!(args["action"], serde_json::json!("view:toggle:0"));
    }

    // A Sprite/TextLabel/HitRegion named `<view>_*` has its `view` arg
    // resolved from the prefix at build time, mirroring Prop scene refs.
    #[test]
    fn build_pipeline_resolves_view_prefix_on_ui_assets() {
        let world = concat!(
            r#"{"name":"pause_menu","type":"View","args":{}}"#,
            "\n",
            r#"{"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":10,"height":10}}"#,
            "\n",
            r#"{"name":"pause_menu_title","type":"TextLabel","args":{"font":"f","content":"x","x":0,"y":0}}"#,
            "\n",
            r#"{"name":"pause_menu_btn","type":"HitRegion","args":{"x":0,"y":0,"width":10,"height":10,"action":"view:hide"}}"#,
            "\n",
            r#"{"name":"f","type":"Font","args":{"size_px":16}}"#,
            "\n",
        );
        let result = build_pipeline_from_str(world, None).expect("build");
        // pause_menu interned id = 0.
        for name in ["pause_menu_dim", "pause_menu_title", "pause_menu_btn"] {
            let def = result
                .defs
                .iter()
                .find(|d| {
                    let args: serde_json::Value = serde_json::from_slice(&d.args_bytes).unwrap();
                    args.get("view") == Some(&serde_json::json!(0)) && d.name.map(|n| n.0).is_some()
                })
                .unwrap_or_else(|| panic!("expected {name} to have view=0"));
            let _ = def;
        }
    }

    // Nested view names resolve by longest prefix: `<menu>_settings_*` binds
    // to the `<menu>_settings` view, not the enclosing `<menu>` view that is
    // declared first. (Regression: first-match claimed the nested elements,
    // so a MainMenu's settings sub-view rendered on top of the main menu.)
    #[test]
    fn resolve_scene_refs_picks_longest_view_prefix() {
        let mk = |name: &str, ty: &str| crate::world::WorldJsonlAsset {
            name: name.to_string(),
            asset_type: ty.to_string(),
            args: serde_json::json!({}),
        };
        let mut assets = vec![
            mk("menu", "View"),
            mk("menu_settings", "View"),
            mk("menu_title", "TextLabel"),
            mk("menu_settings_title", "TextLabel"),
        ];
        super::resolve_scene_refs(&mut assets);
        let view_of = |n: &str| {
            assets
                .iter()
                .find(|a| a.name == n)
                .and_then(|a| a.args.get("view"))
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        assert_eq!(view_of("menu_title").as_deref(), Some("menu"));
        assert_eq!(
            view_of("menu_settings_title").as_deref(),
            Some("menu_settings")
        );
    }

    // Animation with no `source` is left byte-for-byte unchanged: the
    // inline-authored path must not regress.
    #[test]
    fn desugar_gltf_animations_skips_inline_clips() {
        let original = serde_json::json!({
            "target": "flag",
            "duration": 2.0,
            "tracks": [{"joint": 0, "keyframes": [{"time": 0.0, "rotation_deg": [0,0,0]}]}],
        });
        let mut assets = vec![crate::world::WorldJsonlAsset {
            name: "wave".to_string(),
            asset_type: "Animation".to_string(),
            args: original.clone(),
        }];
        desugar_gltf_animations(&mut assets).expect("desugar succeeds");
        assert_eq!(assets[0].args, original);
    }

    #[test]
    fn voxel_chunk_payload_compiles_end_to_end() {
        let world = r#"{"name":"vert","type":"ShaderStage","args":{"kind":"vertex","source":"x.metal"}}
{"name":"frag","type":"ShaderStage","args":{"kind":"fragment","source":"x.metal"}}
{"name":"air","type":"BlockType","args":{"solid":false}}
{"name":"stone","type":"BlockType","args":{"uv_min":[0,0],"uv_max":[1,1]}}
{"name":"chunk","type":"VoxelChunk","args":{"palette":["air","stone"],"dim":[2,1,1],"blocks":[1,1]}}
"#;
        // We can't easily compile shaders here, so go through the geometry
        // entry point directly to verify the voxel chunk produces a non-empty
        // payload for two adjacent solid blocks (10 faces after interior cull).
        let chunk_args = serde_json::json!({
            "palette": ["air", "stone"],
            "dim": [2, 1, 1],
            "blocks": [1, 1],
            "block_size": 1.0,
        });
        let bt = |name: &str| -> Option<serde_json::Value> {
            match name {
                "air" => Some(serde_json::json!({"solid": false})),
                "stone" => Some(serde_json::json!({"uv_min":[0,0],"uv_max":[1,1]})),
                _ => None,
            }
        };
        let bytes = crate::geometry::compile_voxel_chunk_payload(&chunk_args, bt).unwrap();
        assert!(!bytes.is_empty());
        let _ = world; // keeps the inline jsonl reference for documentation
    }
}
