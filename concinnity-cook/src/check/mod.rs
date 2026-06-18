// src/check.rs
// Semantic validation of an expanded world: per-asset arg checks, cross-asset
// reference checks, and world-level graphics rules. Structural validation
// (name/type present, known type, unique names) happens earlier in
// crate::world::load_world.

pub(crate) mod cross_reference;
pub(crate) mod cubemap_texture;
pub(crate) mod environment_map;
pub(crate) mod instanced_prop;
pub(crate) mod mesh;
pub(crate) mod prop;
pub(crate) mod scene_reel;
pub(crate) mod sdf_volume;
pub(crate) mod shader;
pub(crate) mod texture;
pub(crate) mod voxel_chunk;
pub(crate) mod voxel_world;

use crate::world::WorldJsonlAsset;

// Print each validation error in CLI form and collapse them into a single
// io::Error. Shared by the `cn test` command and the build orchestrator so a
// failed world surfaces every problem in one pass.
pub fn report_validation_errors(errors: &[String]) -> std::io::Error {
    for e in errors {
        eprintln!("error:   {}", e);
    }
    eprintln!("\nvalidation failed ({} error(s))", errors.len());
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("validation failed with {} error(s)", errors.len()),
    )
}

pub fn check_asset(type_norm: &str, name: &str, args: &serde_json::Value) -> Result<(), String> {
    match type_norm {
        "texture" => texture::check(name, args),
        "cubemaptexture" | "cubemap" => cubemap_texture::check(name, args),
        "environmentmap" | "envmap" | "ibl" => environment_map::check(name, args),
        "mesh" | "proceduralmesh" => mesh::check(name, args),
        "shaderstage" => shader::check(name, args),
        "prop" => prop::check(name, args),
        "scenereel" | "scenreel" => scene_reel::check(name, args),
        "sdfvolume" | "sdf" => sdf_volume::check(name, args),
        "voxelchunk" | "chunk" => voxel_chunk::check(name, args),
        "voxelworld" => voxel_world::check(name, args),
        "instancedprop" | "instanced" => instanced_prop::check(name, args),
        _ => Ok(()),
    }
}

// Run all semantic validation on a fully expanded world. Collects every
// problem found (per-asset arg errors, unresolved cross-references, and
// graphics-rule violations) so the caller can report them in a single pass.
pub fn check_world(assets: &[WorldJsonlAsset]) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    for asset in assets {
        let type_norm = asset.asset_type.to_lowercase().replace('_', "");
        if let Err(e) = check_asset(&type_norm, &asset.name, &asset.args) {
            errors.push(e);
        }
    }

    if let Err(ref_errors) = cross_reference::validate_cross_references(assets) {
        errors.extend(ref_errors);
    }

    check_graphics_rules(assets, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// World-level graphics constraints. These depend on the set of assets as a
// whole rather than any single asset's args.
fn check_graphics_rules(assets: &[WorldJsonlAsset], errors: &mut Vec<String>) {
    let norm = |a: &WorldJsonlAsset| a.asset_type.to_lowercase().replace('_', "");

    // A rendering world (one with a GraphicsConfig) needs a vertex ShaderStage
    // to drive its geometry pipeline. Companion injection supplies a default
    // vertex + fragment pair for any world that declares no ShaderStage at all,
    // so this only fires when the world declares an incomplete shader set.
    let has_graphics = assets.iter().any(|a| norm(a) == "graphicsconfig");
    if has_graphics {
        let has_vertex_stage = assets.iter().any(|a| {
            norm(a) == "shaderstage"
                && a.args.get("kind").and_then(|v| v.as_str()) == Some("vertex")
        });
        if !has_vertex_stage {
            errors.push(
                "world renders (has a GraphicsConfig) but has no vertex ShaderStage, \
                 add a ShaderStage with kind \"vertex\" and a `source` path"
                    .to_string(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(name: &str, asset_type: &str, args: serde_json::Value) -> WorldJsonlAsset {
        WorldJsonlAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
        }
    }

    #[test]
    fn graphics_config_without_vertex_stage_is_an_error() {
        let assets = vec![asset("gfx", "GraphicsConfig", serde_json::json!({}))];
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("vertex ShaderStage")));
    }

    #[test]
    fn graphics_config_with_vertex_stage_passes_graphics_rules() {
        let assets = vec![
            asset("gfx", "GraphicsConfig", serde_json::json!({})),
            asset(
                "vert",
                "ShaderStage",
                serde_json::json!({
                    "kind": "vertex",
                    "sources": {"metal": "x.metal", "hlsl": "x.hlsl", "glsl": "x.glsl"}
                }),
            ),
        ];
        assert!(check_world(&assets).is_ok());
    }

    #[test]
    fn per_asset_and_cross_reference_errors_both_collected() {
        // Prop with no mesh/model/prefab (per-asset error) plus a Material
        // with a missing albedo texture (cross-reference error).
        let assets = vec![
            asset("bad_prop", "Prop", serde_json::json!({})),
            asset("bad_mat", "Material", serde_json::json!({"albedo":"ghost"})),
        ];
        let errs = check_world(&assets).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("bad_prop")));
        assert!(errs.iter().any(|e| e.contains("ghost")));
    }
}
