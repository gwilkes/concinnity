// Passive source catalogues captured at `GraphicsSystem::init` (under
// `cn debug`) describing every file-backed asset the renderer can hot-reload:
// the on-disk source path plus the GPU slot / draw indices it owns. These are
// plain data: the filesystem watcher, off-thread decode, and reload passes
// that consume them live in the `cn debug` binary (`crate::debug::hot_reload`),
// out of the library. `init` fills these maps and hands them off as a
// `HotReloadSources` bundle through `GraphicsSystem::take_hot_reload_sources`.
//
// These are filled by the library (init) and read by the `cn debug` binary, so
// from `cargo check --lib`'s view every field / `watch_dirs` is write-only.
// Allow dead code module-wide: the whole module is a binary-consumed handoff.
#![allow(dead_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

// Which GPU pool a [`TextureSourceEntry`]'s slot lives in. Drives the choice
// of `update_texture_slot` vs `update_normal_map_slot` at reload time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    // Albedo texture pool; reload via `backend.update_texture_slot`.
    Albedo,
    // Normal-map pool (slot 0 is the always-resident flat-normal fallback;
    // real maps start at slot 1). Reload via `backend.update_normal_map_slot`.
    NormalMap,
}

// One reload entry: a file-backed source and the GPU slot it owns. Built once
// at `GraphicsSystem::init` from the live `Texture` assets and consulted on
// every reload event. Procedural textures (sky / plaster / etc.) carry no
// source file and are absent from the map.
#[derive(Debug, Clone)]
pub struct TextureSourceEntry {
    // The `source` field from the original `Texture` asset, identical to the
    // path the build pipeline read at compile time. Resolved relative to CWD:
    // `cn debug` runs from the client checkout root, so the path is valid
    // as-is.
    pub source: String,
    // `image_index` for `.glb`-image sources; 0 (ignored) for plain PNGs.
    pub image_index: u32,
    // GPU slot in the matching pool (`textures[slot]` for albedo,
    // `normal_map_textures[slot]` for normal maps).
    pub slot: usize,
    pub kind: TextureKind,
}

// Singleton `ColorLut` reload entry. The 3D grading LUT has no slot (the
// composite pass binds `self.color_lut` directly), so we only need the
// resolved source path (the raw asset source string is resolved once at init
// via `crate::build::color_lut::resolve_source_path` so the watcher knows
// where to subscribe and the per-frame reload knows what to re-read).
#[derive(Debug, Clone)]
pub struct ColorLutSource {
    // Resolved on-disk path the build pipeline read at compile time. Stored
    // resolved rather than raw so the watcher can subscribe to a real parent
    // directory even when the asset declaration used a bare filename.
    pub resolved_path: String,
}

// One file-backed `Mesh` reload entry. A single `Mesh` asset can be
// referenced by many `Prop`s, each of which received an independent copy of
// the mesh's geometry in the shared vertex / index buffer, so a reload has
// to overwrite N draw slots, not one. `draw_indices` lists every slot that
// carries this Mesh's geometry; the reload helper walks them all per entry.
#[derive(Debug, Clone)]
pub struct MeshSourceEntry {
    // Path string from the asset declaration. Used as-is by
    // the glTF parser in concinnity-cook, which resolves
    // bare filenames internally. For the watcher's directory subscription a separate
    // resolved path is held on the [`MeshSourceMap`].
    pub source: String,
    // Which primitive (flattened across glTF meshes) to import; mirrors the
    // asset declaration so the runtime decode matches the build pass.
    pub primitive_index: u32,
    // Total LOD count from the asset declaration (`1` for no LODs).
    // Re-applied at decode time so the recomputed payload's LOD trailer
    // matches the slot's init-time layout.
    pub lod_levels: u32,
    // Per-LOD switch distances from the asset declaration. Empty means the
    // build derived a doubling sequence from the mesh's bounding radius;
    // reload reproduces the same defaults by passing through empty.
    pub lod_distances: Vec<f32>,
    // Every draw slot that received this mesh's geometry at init.
    pub draw_indices: Vec<usize>,
}

// Catalogue of every file-backed `Mesh` asset the renderer can hot-reload.
// Owned by `GraphicsSystem` under `cn debug` only. Sourced from
// `build_draw_list` extending its return tuple with `(asset_id ->
// draw_indices)` and cross-referenced against the source / `primitive_index`
// / LOD metadata captured before drains in `load_mesh_geometry`.
#[derive(Debug, Clone, Default)]
pub struct MeshSourceMap {
    pub entries: Vec<MeshSourceEntry>,
}

impl MeshSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    // Every unique parent directory across all entries. The watcher uses
    // these to know what to subscribe to; bare-filename sources (no parent)
    // are skipped here and only reachable via the debug-WS `reload-assets`
    // command. The caller should pass *resolved* paths via the resolved
    // field in [`MeshSourceEntry`]; for now resolution lives at the call
    // site (init.rs).
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.source).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }
}

// One `ProceduralMesh` reload entry. Procedural meshes have no source file,
// so their hot-reload trigger is a `world.jsonl` save (or the debug-WS
// `reload-assets` command): the renderer captures each mesh's args at init
// and re-runs the generator when the on-disk args change. `draw_indices`
// mirrors [`MeshSourceEntry`]: one ProceduralMesh asset can map to many
// draw slots when several `Prop`s share it.
#[derive(Debug, Clone)]
pub struct ProceduralMeshSourceEntry {
    // The asset's name as declared in `world.jsonl`. The reload pass joins
    // the on-disk JSONL's `ProceduralMesh` entries by name so a Prop's
    // renamed-or-replaced mesh trips the same "unknown" log as any other
    // add; we never have to round-trip AssetIds through the interner here.
    pub name: String,
    // Last-applied generator args (the `args` object from `world.jsonl`),
    // normalised through `ProceduralMesh::deserialize → serialize` so
    // default-filled fields match what `parse_world_jsonl` produces at
    // reload time. Deep [`serde_json::Value`] equality classifies whether
    // to regenerate.
    pub args: serde_json::Value,
    // Every draw slot that received this mesh's geometry at init.
    pub draw_indices: Vec<usize>,
}

// Catalogue of every `ProceduralMesh` asset whose generator args the
// renderer can hot-reload from a live `world.jsonl`. Owned by
// `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct ProceduralMeshSourceMap {
    pub entries: Vec<ProceduralMeshSourceEntry>,
}

impl ProceduralMeshSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// One world-loaded [`crate::assets::ShaderStage`] reload entry. Captures
// the stage's kind + the resolved on-disk source path that the build
// pipeline read at compile time, so the hot-reload helper can rerun
// [`concinnity_cook::shader::compile_shader`] on the same file and feed the
// fresh metallib / SPIR-V / DXBC bytes back to the backend for a pipeline
// rebuild. Stages whose source is the embedded GLSL fallback (Vulkan-only,
// no on-disk file) have an empty `resolved_path` and are filtered by the
// caller before reaching this map.
#[derive(Debug, Clone)]
pub struct ShaderStageSourceEntry {
    pub kind: crate::assets::shader_stage::ShaderKind,
    // Resolved on-disk path the build pipeline read at compile time. Stored
    // resolved (not raw) so the watcher can subscribe to a real parent
    // directory even when the asset declaration used a bare filename.
    pub resolved_path: String,
}

// Catalogue of every world-loaded `ShaderStage` whose source the renderer
// can hot-reload. Owned by `GraphicsSystem` under `cn debug` only; consumed
// by [`reload_shader_stages`] when the asset hot-reload watcher fires on a
// captured shader-source file. The map holds at most one entry per
// [`crate::assets::shader_stage::ShaderKind`] (vertex, fragment, shadow,
// vertex_instanced): the runtime drains one stage per kind at init.
#[derive(Debug, Clone, Default)]
pub struct ShaderStageSourceMap {
    pub entries: Vec<ShaderStageSourceEntry>,
}

impl ShaderStageSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    // Every unique parent directory across all entries. The watcher uses
    // these to know what to subscribe to alongside the texture / mesh /
    // LUT / envmap / world directories. Bare filenames (no parent) are
    // skipped; those are only reachable via the debug-WS `reload-assets`
    // command, mirroring the static-Mesh + texture maps.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.resolved_path).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }
}

// One file-backed `SkinnedMesh` reload entry. Unlike static `Mesh`, a
// `SkinnedMesh` is 1:1 with its draw slot (there's no shared-instance
// fan-out across Props), so a single `skinned_index` identifies the slot to
// update. The vertex region is at `[vertex_base, vertex_base + vertex_count)`
// in the shared skinned vertex buffer; `joint_count` is snapshotted at init
// so the reload can reject skeleton-shape changes (which would require
// rebuilding the skinned pipeline state from shader-library bytes that
// `upload_skinned` consumes and drops).
#[derive(Debug, Clone)]
pub struct SkinnedMeshSourceEntry {
    // Path string from the asset declaration. Used as-is by
    // the glTF parser in concinnity-cook, which resolves
    // bare filenames internally.
    pub source: String,
    // Index into `MtlContext.skinned_draw_objects` (and the corresponding
    // `SkinnedDrawObject` slot on every backend) of the draw this entry
    // owns.
    pub skinned_index: usize,
    // Vertex offset (in vertex units, not bytes) into the shared skinned
    // vertex buffer where this slot's geometry starts.
    pub vertex_base: u16,
    // Number of vertices in this slot. Used to reject size-changing
    // reloads before pushing through to the backend.
    pub vertex_count: usize,
    // Number of indices in this slot, matches
    // `SkinnedDrawObject.index_count`. Kept here too so the size check
    // runs without indirecting through the backend.
    pub index_count: usize,
    // Init-time bind-pose joint count. Reload is rejected if the re-imported
    // skeleton has a different joint count; a different shape would need
    // a full pipeline rebuild, which `upload_skinned` does not support
    // post-init.
    pub joint_count: usize,
}

// Catalogue of every file-backed `SkinnedMesh` asset the renderer can
// hot-reload. Owned by `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct SkinnedMeshSourceMap {
    pub entries: Vec<SkinnedMeshSourceEntry>,
}

impl SkinnedMeshSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    // Every unique parent directory across all entries. The watcher uses
    // these alongside the static-Mesh watch dirs.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.source).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }
}

// Singleton `EnvironmentMap` reload entry. The two IBL cubemaps have no slot
// (the fragment shader binds `self.env_map.irradiance` and
// `self.env_map.prefilter` directly), so we only need the resolved HDR path
// plus the three sizing knobs from the asset declaration. The face sizes /
// sample count are captured at init so the runtime re-decode produces the
// same texture dimensions as the build pass (a size change would invalidate
// fragment-shader assumptions about the prefilter mip chain).
#[derive(Debug, Clone)]
pub struct EnvironmentMapSource {
    // Resolved on-disk path to the `.hdr` equirectangular. Stored resolved
    // (not raw) so the watcher can subscribe to a real parent directory even
    // when the asset declaration used a bare filename.
    pub resolved_path: String,
    // Mip-0 face size of the prefiltered radiance cubemap.
    pub prefilter_face_size: u32,
    // Face size of the irradiance cubemap.
    pub irradiance_face_size: u32,
    // Hammersley sample count for the GGX prefilter convolution.
    pub prefilter_samples: u32,
}

// Catalogue of every file-backed `Texture` slot the renderer can hot-reload.
// Owned by `GraphicsSystem` under `cn debug` only.
#[derive(Debug, Clone, Default)]
pub struct TextureSourceMap {
    pub entries: Vec<TextureSourceEntry>,
}

impl TextureSourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    // Add an albedo-pool entry. Procedural / source-less textures should be
    // filtered by the caller before calling this; every entry must have a
    // non-empty `source`.
    pub fn push_albedo(&mut self, source: String, image_index: u32, slot: usize) {
        self.entries.push(TextureSourceEntry {
            source,
            image_index,
            slot,
            kind: TextureKind::Albedo,
        });
    }

    // Add a normal-map-pool entry. Slot 0 is the flat-normal fallback and
    // should never appear here; real maps live at slot >= 1.
    pub fn push_normal_map(&mut self, source: String, image_index: u32, slot: usize) {
        self.entries.push(TextureSourceEntry {
            source,
            image_index,
            slot,
            kind: TextureKind::NormalMap,
        });
    }

    // Every unique parent directory across all entries. Used by the
    // filesystem watcher to know what to subscribe to. A `.glb` source has
    // its containing directory watched too; the whole file shows up as
    // "modified" when the user re-exports it.
    pub fn watch_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: BTreeSet<PathBuf> = BTreeSet::new();
        for e in &self.entries {
            if let Some(parent) = Path::new(&e.source).parent()
                && !parent.as_os_str().is_empty()
            {
                dirs.insert(parent.to_path_buf());
            }
        }
        dirs.into_iter().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// Bundle of every captured source catalogue, handed from `GraphicsSystem`
// init to the `cn debug` binary's hot-reload drive, which builds the
// filesystem watcher + `AssetHotReloadState` from it. Empty / `None` under
// `cn run`, which never captures sources.
pub struct HotReloadSources {
    pub map: TextureSourceMap,
    pub color_lut: Option<ColorLutSource>,
    pub environment_map: Option<EnvironmentMapSource>,
    pub meshes: MeshSourceMap,
    pub skinned_meshes: SkinnedMeshSourceMap,
    pub procedural_meshes: ProceduralMeshSourceMap,
    pub shader_stages: ShaderStageSourceMap,
    pub world_jsonl_path: Option<String>,
}
