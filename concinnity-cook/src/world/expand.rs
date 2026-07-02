// src/world/expand.rs
// Entry point for all build-time JSON-level world expansion.
// Operates purely on serde_json::Value; no type registry or blob compilation.

use super::camera_shot::expand_camera_shots;
use super::companion::inject_companions;
use super::defaults::inject_engine_defaults;
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

// One asset added to the world by an injection pass rather than authored or
// macro-expanded. Recorded in world-lock.json so the user can see every
// default and copy its entry into world.jsonl as an override.
#[derive(Debug, Clone)]
pub struct InjectedAsset {
    pub name: String,
    pub asset_type: String,
    pub args: serde_json::Value,
    // The injection pass (an EngineDefaults flag name, "companion", or
    // "default_font"), so listings can say where a default came from.
    pub injected_by: &'static str,
}

// What the injection passes added during one expansion run.
#[derive(Debug, Default)]
pub struct ExpandReport {
    pub injected: Vec<InjectedAsset>,
}

impl ExpandReport {
    pub(crate) fn record(
        &mut self,
        name: &str,
        asset_type: &str,
        args: serde_json::Value,
        injected_by: &'static str,
    ) {
        self.injected.push(InjectedAsset {
            name: name.to_string(),
            asset_type: asset_type.to_string(),
            args,
            injected_by,
        });
    }
}

// Run all expansion passes in order. Mutates the asset list in place and
// reports what the injection passes added. Returns an error only when a hard
// failure occurs (e.g. prefab cycle or missing prefab reference).
pub fn expand_world(assets: &mut Vec<serde_json::Value>) -> Result<ExpandReport, String> {
    let mut report = ExpandReport::default();
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
    // First companion round: materialize the GraphicsConfig render marker (and
    // its Window / ShaderStage stack) implied by everything authored or
    // expanded above, so the defaults pass can key off "this world renders".
    inject_companions(assets, &mut report);
    // Engine defaults: complete a rendering world with the standard assets it
    // does not declare (MainMenu, HUDs + chips + font, sky mesh). Runs before
    // menu expansion so an injected MainMenu expands like an authored one.
    inject_engine_defaults(assets, &mut report)?;
    // Menus expand to External UI assets (View / Sprite / TextLabel /
    // HitRegion / KeyBinding) that need no further expansion, but whose
    // TextLabels must still pull in their GraphicsConfig + Font companions, so
    // this runs before the second companion round.
    expand_main_menus(assets)?;
    // Menus emit OptionSelect rows for their settings sub-view; expand those to
    // their primitives (TextLabels + HitRegion) before companion injection so
    // the generated TextLabels pull in their Font.
    expand_option_selects(assets)?;
    // Menus also emit Slider rows (continuous settings); expand those to their
    // primitives (TextLabels + Sprites + HitRegion) on the same footing, before
    // companion injection.
    expand_sliders(assets)?;
    // Second companion round: companions for the assets the defaults and menu
    // passes added. Idempotent for everything round one already covered.
    inject_companions(assets, &mut report);
    Ok(report)
}

// Load and structurally validate a world.jsonl string, then run all
// expansion passes. Returns the fully expanded asset list. Does not run
// semantic validation; see `crate::world::prepare_world` for the full
// build-pipeline front half.
pub fn expand_world_from_str(content: &str) -> std::io::Result<Vec<serde_json::Value>> {
    let mut assets = load_world(content)
        .map_err(|errs| std::io::Error::new(std::io::ErrorKind::InvalidData, errs.join("\n")))?;

    let _ = expand_world(&mut assets)
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
