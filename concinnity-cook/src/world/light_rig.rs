// src/world/light_rig.rs
// Build-time expansion: LightRig → DirectionalLight / PointLight assets.

use super::expand::{asset_name, type_norm};
use super::preset::load_preset_obj;

pub(crate) fn expand_light_rigs(asset_values: &mut Vec<serde_json::Value>) {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if type_norm(&value) != "lightrig" {
            result.push(value);
            continue;
        }
        let rig_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));
        let preset = args.get("preset").and_then(|v| v.as_str()).unwrap_or("");
        if !preset.is_empty() {
            for light in expand_light_rig_preset(&rig_name, preset) {
                result.push(light);
            }
        }
        // lights: Vec<String>; referenced lights are already declared; the rig
        // entry is consumed and those lights pass through untouched.
    }
    *asset_values = result;
}

fn expand_light_rig_preset(rig_name: &str, preset: &str) -> Vec<serde_json::Value> {
    let defs = {
        let hardcoded = rig_preset_lights(preset);
        if hardcoded.is_empty() {
            load_preset_obj(preset, "light_rigs")
                .get("args")
                .and_then(|a| a.get("lights"))
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default()
        } else {
            hardcoded
        }
    };

    defs.iter()
        .map(|light| {
            let kind = light.get("kind").and_then(|v| v.as_str()).unwrap_or("directional");
            let lname = light.get("name").and_then(|v| v.as_str()).unwrap_or("light");
            let expanded = format!("{}_{}", rig_name, lname);
            match kind {
                "point" => serde_json::json!({
                    "name": expanded,
                    "type": "PointLight",
                    "args": {
                        "position": light.get("position").cloned().unwrap_or(serde_json::json!([0.0, 2.5, 0.0])),
                        "color":    light.get("color").cloned().unwrap_or(serde_json::json!([1.0, 1.0, 1.0])),
                        "intensity": light.get("intensity").and_then(|v| v.as_f64()).unwrap_or(8.0),
                        "range":     light.get("range").and_then(|v| v.as_f64()).unwrap_or(6.0)
                    }
                }),
                _ => serde_json::json!({
                    "name": expanded,
                    "type": "DirectionalLight",
                    "args": {
                        "direction": light.get("direction").cloned().unwrap_or(serde_json::json!([-0.3, 0.85, 0.4])),
                        "color":     light.get("color").cloned().unwrap_or(serde_json::json!([1.0, 1.0, 1.0])),
                        "intensity": light.get("intensity").and_then(|v| v.as_f64()).unwrap_or(1.0)
                    }
                }),
            }
        })
        .collect()
}

fn rig_preset_lights(preset: &str) -> Vec<serde_json::Value> {
    match preset {
        "rig_outdoor_sun" => vec![
            serde_json::json!({"kind":"directional","name":"sun","direction":[-0.4,0.7,0.3],"color":[1.0,0.95,0.8],"intensity":1.2}),
        ],
        "rig_outdoor_sun_fill" => vec![
            serde_json::json!({"kind":"directional","name":"sun","direction":[-0.4,0.7,0.3],"color":[1.0,0.95,0.8],"intensity":1.2}),
            serde_json::json!({"kind":"directional","name":"fill","direction":[0.3,0.5,-0.5],"color":[0.6,0.8,1.0],"intensity":0.3}),
        ],
        "rig_studio_three_point" => vec![
            serde_json::json!({"kind":"directional","name":"key","direction":[-0.6,0.7,0.4],"color":[1.0,0.95,0.9],"intensity":1.2}),
            serde_json::json!({"kind":"directional","name":"fill","direction":[0.8,0.4,0.3],"color":[0.8,0.9,1.0],"intensity":0.4}),
            serde_json::json!({"kind":"directional","name":"rim","direction":[0.2,0.6,-0.8],"color":[0.9,0.9,1.0],"intensity":0.6}),
        ],
        "rig_interior_candles" => vec![
            serde_json::json!({"kind":"directional","name":"ambient","direction":[0.0,1.0,0.0],"color":[0.8,0.6,0.4],"intensity":0.2}),
            serde_json::json!({"kind":"point","name":"candle_a","position":[3.0,1.5,-3.0],"color":[1.0,0.7,0.3],"intensity":8.0,"range":5.0}),
            serde_json::json!({"kind":"point","name":"candle_b","position":[-3.0,1.5,-3.0],"color":[1.0,0.7,0.3],"intensity":8.0,"range":5.0}),
            serde_json::json!({"kind":"point","name":"candle_c","position":[0.0,1.5,4.0],"color":[1.0,0.7,0.3],"intensity":8.0,"range":5.0}),
        ],
        "rig_night_moon" => vec![
            serde_json::json!({"kind":"directional","name":"moon","direction":[-0.2,0.8,0.3],"color":[0.7,0.8,1.0],"intensity":0.4}),
        ],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_lights_consumed_leaves_lights_intact() {
        let mut assets = vec![
            serde_json::json!({"name":"sun","type":"DirectionalLight","args":{"direction":[-0.4,0.7,0.3]}}),
            serde_json::json!({"name":"torch","type":"PointLight","args":{"position":[3.0,2.0,-5.0]}}),
            serde_json::json!({"name":"rig","type":"LightRig","args":{"lights":["sun","torch"]}}),
        ];
        expand_light_rigs(&mut assets);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "sun");
        assert_eq!(assets[1]["name"], "torch");
    }

    #[test]
    fn preset_sun_fill_expands_to_two_lights() {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": "rig_outdoor_sun_fill"}
        })];
        expand_light_rigs(&mut assets);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "rig_sun");
        assert_eq!(assets[1]["name"], "rig_fill");
        assert_eq!(assets[0]["type"], "DirectionalLight");
    }

    #[test]
    fn preset_interior_candles_includes_point_lights() {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": "rig_interior_candles"}
        })];
        expand_light_rigs(&mut assets);
        assert_eq!(assets.len(), 4);
        let point_count = assets.iter().filter(|v| v["type"] == "PointLight").count();
        assert_eq!(point_count, 3);
    }

    #[test]
    fn preset_studio_three_point_expands_to_three() {
        let mut assets = vec![serde_json::json!({
            "name": "rig",
            "type": "LightRig",
            "args": {"preset": "rig_studio_three_point"}
        })];
        expand_light_rigs(&mut assets);
        assert_eq!(assets.len(), 3);
    }

    #[test]
    fn non_rig_assets_pass_through() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_light_rigs(&mut assets);
        assert_eq!(assets[0]["type"], "Logger");
    }
}
