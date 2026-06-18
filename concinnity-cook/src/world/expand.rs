// src/world/expand.rs
// Entry point for all build-time JSON-level world expansion.
// Operates purely on serde_json::Value; no type registry or blob compilation.

use super::camera_shot::expand_camera_shots;
use super::companion::inject_companions;
use super::light_rig::expand_light_rigs;
use super::main_menu::expand_main_menus;
use super::material_palette::expand_material_palettes;
use super::option_select::expand_option_selects;
use super::prefab::expand_prefabs;
use super::room::expand_room_textures;
use super::scene_import::expand_scene_imports;
use super::shader::normalize_shader_types;
use super::slider::expand_sliders;

use crate::world::load_world;

// Shared helpers used across expansion submodules.

pub(crate) fn type_norm(v: &serde_json::Value) -> String {
    v.get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .to_lowercase()
        .replace('_', "")
}

pub(crate) fn asset_name(v: &serde_json::Value) -> String {
    v.get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string()
}

// Run all expansion passes in order. Mutates the asset list in place.
// Returns an error only when a hard failure occurs (e.g. prefab cycle or
// missing prefab reference).
pub fn expand_world(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    normalize_shader_types(assets);
    // Imports expand first so the assets they generate (materials, meshes,
    // props, a framed camera) flow through every later pass, including
    // companion injection.
    expand_scene_imports(assets)?;
    expand_camera_shots(assets);
    expand_light_rigs(assets);
    expand_material_palettes(assets);
    expand_prefabs(assets)?;
    expand_room_textures(assets);
    // Menus expand to External UI assets (View / Sprite / TextLabel /
    // HitRegion / KeyBinding) that need no further expansion, but whose
    // TextLabels must still pull in their GraphicsConfig + Font companions, so
    // this runs last, right before companion injection.
    expand_main_menus(assets)?;
    // Menus emit OptionSelect rows for their settings sub-view; expand those to
    // their primitives (TextLabels + HitRegion) before companion injection so
    // the generated TextLabels pull in their Font.
    expand_option_selects(assets)?;
    // Menus also emit Slider rows (continuous settings); expand those to their
    // primitives (TextLabels + Sprites + HitRegion) on the same footing, before
    // companion injection.
    expand_sliders(assets)?;
    inject_companions(assets);
    Ok(())
}

// Load and structurally validate a world.jsonl string, then run all
// expansion passes. Returns the fully expanded asset list. Does not run
// semantic validation; see `crate::world::prepare_world` for the full
// build-pipeline front half.
pub fn expand_world_from_str(content: &str) -> std::io::Result<Vec<serde_json::Value>> {
    let mut assets = load_world(content)
        .map_err(|errs| std::io::Error::new(std::io::ErrorKind::InvalidData, errs.join("\n")))?;

    expand_world(&mut assets)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    Ok(assets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_norm_lowercases_and_strips_underscores() {
        let v = serde_json::json!({"type": "MaterialPalette"});
        assert_eq!(type_norm(&v), "materialpalette");
    }

    #[test]
    fn type_norm_handles_underscored_type() {
        let v = serde_json::json!({"type": "Camera3D"});
        assert_eq!(type_norm(&v), "camera3d");
    }

    #[test]
    fn type_norm_missing_type_returns_empty() {
        let v = serde_json::json!({"name": "x"});
        assert_eq!(type_norm(&v), "");
    }

    #[test]
    fn asset_name_extracts_name() {
        let v = serde_json::json!({"name": "my_asset", "type": "Logger"});
        assert_eq!(asset_name(&v), "my_asset");
    }

    #[test]
    fn asset_name_missing_returns_empty() {
        let v = serde_json::json!({"type": "Logger"});
        assert_eq!(asset_name(&v), "");
    }

    #[test]
    fn expand_world_from_str_injects_companions() {
        let content = r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#;
        let assets = expand_world_from_str(content).unwrap();
        assert!(assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
        // GraphicsConfig pulls in a Window companion.
        assert!(assets.iter().any(|v| type_norm(v) == "window"));
    }

    #[test]
    fn bare_main_menu_world_expands_and_pulls_companions() {
        let content = r#"{"name":"main_menu","type":"MainMenu"}"#;
        let assets = expand_world_from_str(content).unwrap();
        // The MainMenu is gone, replaced by its UI assets.
        assert!(!assets.iter().any(|v| type_norm(v) == "mainmenu"));
        assert!(assets.iter().any(|v| type_norm(v) == "view"));
        assert!(assets.iter().any(|v| type_norm(v) == "hitregion"));
        // The generated TextLabels pull in GraphicsConfig + a Font companion.
        assert!(assets.iter().any(|v| type_norm(v) == "textlabel"));
        assert!(assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
        assert!(assets.iter().any(|v| type_norm(v) == "font"));
    }
}
