// src/world/material_palette.rs
// Build-time expansion: MaterialPalette → Material assets.

use super::expand::{asset_name, type_norm};
use super::preset::load_preset_obj;

pub(crate) fn expand_material_palettes(asset_values: &mut Vec<serde_json::Value>) {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if type_norm(&value) != "materialpalette" {
            result.push(value);
            continue;
        }
        let palette_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));
        for mat in resolve_palette_materials(&palette_name, &args) {
            result.push(mat);
        }
    }
    *asset_values = result;
}

fn resolve_palette_materials(
    palette_name: &str,
    args: &serde_json::Value,
) -> Vec<serde_json::Value> {
    let preset = args.get("preset").and_then(|v| v.as_str()).unwrap_or("");
    let entries: Vec<serde_json::Value> = if !preset.is_empty() {
        let hardcoded = palette_preset_entries(preset);
        if hardcoded.is_empty() {
            load_preset_obj(preset, "palettes")
                .get("args")
                .and_then(|a| a.get("entries"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        } else {
            hardcoded
        }
    } else {
        args.get("entries")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default()
    };

    entries
        .iter()
        .map(|entry| {
            let alias = entry.get("alias").and_then(|v| v.as_str()).unwrap_or("surface");
            let expanded = format!("{}_{}", palette_name, alias);
            serde_json::json!({
                "name": expanded,
                "type": "Material",
                "args": {
                    "albedo":          entry.get("albedo").cloned().unwrap_or(serde_json::json!("")),
                    "normal_map":      entry.get("normal_map").cloned().unwrap_or(serde_json::json!("")),
                    "roughness":       entry.get("roughness").and_then(|v| v.as_f64()).unwrap_or(0.8),
                    "metallic":        entry.get("metallic").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    "tint":            entry.get("tint").cloned().unwrap_or(serde_json::json!([1.0, 1.0, 1.0])),
                    "emissive_factor": entry.get("emissive_factor").cloned().unwrap_or(serde_json::json!([0.0, 0.0, 0.0]))
                }
            })
        })
        .collect()
}

fn palette_preset_entries(preset: &str) -> Vec<serde_json::Value> {
    match preset {
        "pal_stone_dungeon" => vec![
            serde_json::json!({"alias":"floor",  "albedo":"tex_stone","roughness":0.9, "metallic":0.0}),
            serde_json::json!({"alias":"wall",   "albedo":"tex_stone","roughness":0.85,"metallic":0.0}),
            serde_json::json!({"alias":"ceiling","albedo":"tex_stone","roughness":0.9, "metallic":0.0}),
            serde_json::json!({"alias":"pillar", "albedo":"tex_stone","roughness":0.8, "metallic":0.0}),
        ],
        "pal_wood_cabin" => vec![
            serde_json::json!({"alias":"floor","albedo":"tex_wood",   "roughness":0.7, "metallic":0.0}),
            serde_json::json!({"alias":"wall", "albedo":"tex_plaster","roughness":0.85,"metallic":0.0}),
            serde_json::json!({"alias":"beam", "albedo":"tex_wood",   "roughness":0.65,"metallic":0.0}),
            serde_json::json!({"alias":"trim", "albedo":"tex_wood",   "roughness":0.6, "metallic":0.0}),
        ],
        "pal_metal_industrial" => vec![
            serde_json::json!({"alias":"floor","albedo":"tex_concrete","roughness":0.85,"metallic":0.0}),
            serde_json::json!({"alias":"wall", "albedo":"tex_concrete","roughness":0.8, "metallic":0.0}),
            serde_json::json!({"alias":"pipe", "albedo":"tex_metal",   "roughness":0.4, "metallic":1.0}),
            serde_json::json!({"alias":"grate","albedo":"tex_metal",   "roughness":0.5, "metallic":0.8}),
        ],
        "pal_plaster_cottage" => vec![
            serde_json::json!({"alias":"floor","albedo":"tex_wood",   "roughness":0.7, "metallic":0.0}),
            serde_json::json!({"alias":"wall", "albedo":"tex_plaster","roughness":0.9, "metallic":0.0}),
            serde_json::json!({"alias":"trim", "albedo":"tex_wood",   "roughness":0.6, "metallic":0.0}),
            serde_json::json!({"alias":"door", "albedo":"tex_wood",   "roughness":0.65,"metallic":0.0}),
        ],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_entries_expand_to_materials() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"entries": [
                {"alias":"floor","albedo":"tex_stone","roughness":0.9,"metallic":0.0},
                {"alias":"wall","albedo":"tex_brick","roughness":0.85,"metallic":0.0}
            ]}
        })];
        expand_material_palettes(&mut assets);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "pal_floor");
        assert_eq!(assets[0]["type"], "Material");
        assert_eq!(assets[1]["name"], "pal_wall");
        assert_eq!(assets[0]["args"]["albedo"], "tex_stone");
    }

    #[test]
    fn preset_stone_dungeon_expands_four_materials() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"preset": "pal_stone_dungeon"}
        })];
        expand_material_palettes(&mut assets);
        assert_eq!(assets.len(), 4);
        let names: Vec<&str> = assets.iter().filter_map(|v| v["name"].as_str()).collect();
        assert!(names.contains(&"pal_floor"));
        assert!(names.contains(&"pal_wall"));
        assert!(names.contains(&"pal_ceiling"));
        assert!(names.contains(&"pal_pillar"));
    }

    #[test]
    fn material_palette_consumed_from_list() {
        let mut assets = vec![
            serde_json::json!({"name":"pal","type":"MaterialPalette","args":{"entries":[
                {"alias":"x","roughness":0.5}
            ]}}),
            serde_json::json!({"name":"other","type":"Logger","args":{}}),
        ];
        expand_material_palettes(&mut assets);
        assert!(!assets.iter().any(|v| v["type"] == "MaterialPalette"));
        assert!(assets.iter().any(|v| v["type"] == "Logger"));
    }

    #[test]
    fn material_defaults_applied() {
        let mut assets = vec![serde_json::json!({
            "name": "pal",
            "type": "MaterialPalette",
            "args": {"entries": [{"alias":"base"}]}
        })];
        expand_material_palettes(&mut assets);
        assert_eq!(assets[0]["args"]["roughness"], 0.8);
        assert_eq!(assets[0]["args"]["metallic"], 0.0);
    }
}
