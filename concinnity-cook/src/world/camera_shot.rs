// src/world/camera_shot.rs
// Build-time expansion: CameraShot → Camera3D.

use super::expand::{asset_name, type_norm};
use super::preset::load_preset_obj;

pub(crate) fn expand_camera_shots(asset_values: &mut Vec<serde_json::Value>) {
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in asset_values.drain(..) {
        if type_norm(&value) != "camerashot" {
            result.push(value);
            continue;
        }
        let shot_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));

        let preset_str = args.get("preset").and_then(|v| v.as_str()).unwrap_or("");
        let preset_args: serde_json::Value = if !preset_str.is_empty() {
            let hardcoded = camera_shot_preset(preset_str);
            if hardcoded.is_null() {
                load_preset_obj(preset_str, "shots")
                    .get("args")
                    .cloned()
                    .unwrap_or(serde_json::json!({}))
            } else {
                hardcoded
            }
        } else {
            serde_json::json!({})
        };

        let fov = args
            .get("fov_y_degrees")
            .or_else(|| preset_args.get("fov_y_degrees"))
            .and_then(|v| v.as_f64())
            .unwrap_or(75.0);
        let near = args
            .get("near")
            .or_else(|| preset_args.get("near"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.05);
        let far = args
            .get("far")
            .or_else(|| preset_args.get("far"))
            .and_then(|v| v.as_f64())
            .unwrap_or(200.0);
        let pos_v = args.get("position").or_else(|| preset_args.get("position"));
        let position = pos_v
            .and_then(|a| a.as_array())
            .map(|a| {
                serde_json::json!([
                    a.first().and_then(|v| v.as_f64()).unwrap_or(0.0),
                    a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0),
                    a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0)
                ])
            })
            .unwrap_or(serde_json::json!([0.0, 0.0, 0.0]));
        let yaw = args
            .get("yaw")
            .or_else(|| preset_args.get("yaw"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let pitch = args
            .get("pitch")
            .or_else(|| preset_args.get("pitch"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        result.push(serde_json::json!({
            "name": shot_name,
            "type": "Camera3D",
            "args": {
                "fov_y_degrees": fov,
                "near": near,
                "far": far,
                "position": position,
                "yaw": yaw,
                "pitch": pitch
            }
        }));
    }
    *asset_values = result;
}

fn camera_shot_preset(preset: &str) -> serde_json::Value {
    match preset {
        "shot_eye_level" => {
            serde_json::json!({"fov_y_degrees":75.0,"position":[0.0,1.75,0.0],"yaw":std::f64::consts::PI,"near":0.05,"far":200.0})
        }
        "shot_overhead" => {
            serde_json::json!({"fov_y_degrees":60.0,"position":[0.0,8.0,0.0],"pitch":-1.3963,"near":0.05,"far":200.0})
        }
        "shot_dramatic_low" => {
            serde_json::json!({"fov_y_degrees":85.0,"position":[0.0,0.4,0.0],"pitch":0.2618,"near":0.05,"far":200.0})
        }
        "shot_outdoor_wide" => {
            serde_json::json!({"fov_y_degrees":80.0,"position":[0.0,1.75,0.0],"near":0.05,"far":500.0})
        }
        _ => serde_json::Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_to_camera3d() {
        let mut assets = vec![serde_json::json!({
            "name": "wide",
            "type": "CameraShot",
            "args": {"fov_y_degrees": 80.0, "position": [0.0, 1.75, 8.0], "yaw": std::f64::consts::PI}
        })];
        expand_camera_shots(&mut assets);
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["name"], "wide");
        assert_eq!(assets[0]["type"], "Camera3D");
        assert_eq!(assets[0]["args"]["fov_y_degrees"], 80.0);
    }

    #[test]
    fn preset_eye_level_expands() {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": {"preset": "shot_eye_level"}
        })];
        expand_camera_shots(&mut assets);
        assert_eq!(assets[0]["type"], "Camera3D");
        let fov = assets[0]["args"]["fov_y_degrees"].as_f64().unwrap();
        assert!((fov - 75.0).abs() < 0.01);
    }

    #[test]
    fn inline_args_override_preset() {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": {"preset": "shot_eye_level", "fov_y_degrees": 90.0}
        })];
        expand_camera_shots(&mut assets);
        let fov = assets[0]["args"]["fov_y_degrees"].as_f64().unwrap();
        assert!((fov - 90.0).abs() < 0.01);
    }

    #[test]
    fn non_camera_shot_assets_pass_through() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_camera_shots(&mut assets);
        assert_eq!(assets[0]["type"], "Logger");
    }

    #[test]
    fn defaults_applied_when_no_args() {
        let mut assets = vec![serde_json::json!({
            "name": "cam",
            "type": "CameraShot",
            "args": {}
        })];
        expand_camera_shots(&mut assets);
        assert_eq!(assets[0]["type"], "Camera3D");
        let fov = assets[0]["args"]["fov_y_degrees"].as_f64().unwrap();
        assert!((fov - 75.0).abs() < 0.01);
        let near = assets[0]["args"]["near"].as_f64().unwrap();
        assert!((near - 0.05).abs() < 0.001);
    }
}
