// src/assets/material_palette.rs

use crate::ecs::{AssetOrigin, Component};

/// A named set of [Material](#material) entries with short aliases.
///
/// Expands into [Material](#material) assets named `<palette_name>_<alias>`.
/// [Prop](#prop)s reference the expanded names.
///
/// **Each entry:**
///
/// **Library presets** (JSON files in `assets/palettes/`):
///
/// ```jsonl
/// // Inline:
/// {"type":"MaterialPalette","name":"pal","args":{"entries":[
///   {"alias":"floor","albedo":"tex_stone","roughness":0.9},
///   {"alias":"wall","albedo":"tex_brick","roughness":0.85},
///   {"alias":"trim","albedo":"tex_metal","roughness":0.3,"metallic":0.8}
/// ]}}
/// // Props reference expanded names:
/// {"type":"Prop","name":"floor","args":{"mesh":"room_mesh","material":"pal_floor","position":[0,0,0]}}
///
/// // From library preset:
/// {"type":"MaterialPalette","name":"pal","args":{"preset":"pal_stone_dungeon"}}
/// {"type":"Prop","name":"south_wall","args":{"mesh":"wall_mesh","material":"pal_wall"}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MaterialPalette {
    /// Name of a built-in or file-backed preset (e.g. "pal_stone_dungeon").
    /// When set, `entries` is ignored.
    pub preset: String,
    /// Inline material entries. Each entry must have an `alias` plus Material
    /// fields (albedo, normal_map, roughness, metallic, tint, emissive_factor).
    /// Ignored when `preset` is set.
    pub entries: Vec<serde_json::Value>,
}

impl Component for MaterialPalette {
    const NAME: &'static str = "MaterialPalette";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
