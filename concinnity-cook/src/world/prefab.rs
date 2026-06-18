// src/world/prefab.rs
// Build-time expansion: Prefab templates + Prop instances → concrete assets.

use super::expand::{asset_name, type_norm};
use super::preset::load_preset_obj;

pub(crate) fn expand_prefabs(asset_values: &mut Vec<serde_json::Value>) -> Result<(), String> {
    // Step 1: collect all Prefab definitions.
    let mut prefab_defs: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    let mut non_prefab: Vec<serde_json::Value> = Vec::new();

    for value in asset_values.drain(..) {
        if type_norm(&value) == "prefab" {
            let name = asset_name(&value);
            if !name.is_empty() {
                prefab_defs.insert(name, value);
            }
        } else {
            non_prefab.push(value);
        }
    }

    // Step 2: expand Prop entries that reference a prefab.
    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in non_prefab {
        if type_norm(&value) != "prop" {
            result.push(value);
            continue;
        }
        let prefab_ref = value
            .get("args")
            .and_then(|a| a.get("prefab"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if prefab_ref.is_empty() {
            result.push(value);
            continue;
        }

        let instance_name = asset_name(&value);
        let args = value.get("args").cloned().unwrap_or(serde_json::json!({}));
        let inst_pos = f32_arr3(&args, "position", [0.0, 0.0, 0.0]);
        let inst_rot = f32_arr3(&args, "rotation_deg", [0.0, 0.0, 0.0]);
        let inst_scale = f32_arr3(&args, "scale", [1.0, 1.0, 1.0]);

        let prefab_def = if let Some(def) = prefab_defs.get(prefab_ref) {
            def.clone()
        } else {
            let loaded = load_preset_obj(prefab_ref, "prefabs");
            if loaded.is_null() {
                return Err(format!(
                    "Prop '{}': prefab '{}' not found, declare a Prefab asset with that name",
                    instance_name, prefab_ref
                ));
            }
            loaded
        };

        let mut call_stack: Vec<String> = vec![prefab_ref.to_string()];
        let expanded = expand_prefab_entries(
            &instance_name,
            inst_pos,
            inst_rot,
            inst_scale,
            &prefab_def,
            &prefab_defs,
            &mut call_stack,
        )?;
        result.extend(expanded);
    }

    *asset_values = result;
    Ok(())
}

fn expand_prefab_entries(
    instance_name: &str,
    inst_pos: [f32; 3],
    inst_rot: [f32; 3],
    inst_scale: [f32; 3],
    prefab_def: &serde_json::Value,
    prefab_defs: &std::collections::HashMap<String, serde_json::Value>,
    call_stack: &mut Vec<String>,
) -> Result<Vec<serde_json::Value>, String> {
    let entries = prefab_def
        .get("args")
        .and_then(|a| a.get("props"))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let mut result: Vec<serde_json::Value> = Vec::new();

    for entry in &entries {
        let kind = entry.get("kind").and_then(|v| v.as_str()).unwrap_or("prop");
        let entry_name = entry.get("name").and_then(|v| v.as_str()).unwrap_or("obj");
        let expanded_name = format!("{}_{}", instance_name, entry_name);

        let local_pos = f32_arr3(entry, "position", [0.0, 0.0, 0.0]);
        let local_rot = f32_arr3(entry, "rotation_deg", [0.0, 0.0, 0.0]);
        let local_scale = f32_arr3(entry, "scale", [1.0, 1.0, 1.0]);

        let rotated = rotate_local(local_pos, inst_rot);
        let world_pos = [
            inst_pos[0] + inst_scale[0] * rotated[0],
            inst_pos[1] + inst_scale[1] * rotated[1],
            inst_pos[2] + inst_scale[2] * rotated[2],
        ];
        // Component-wise rotation composition (accurate for common yaw-only case).
        let world_rot = [
            inst_rot[0] + local_rot[0],
            inst_rot[1] + local_rot[1],
            inst_rot[2] + local_rot[2],
        ];
        let world_scale = [
            inst_scale[0] * local_scale[0],
            inst_scale[1] * local_scale[1],
            inst_scale[2] * local_scale[2],
        ];

        match kind {
            "point_light" => {
                let color = entry
                    .get("light_color")
                    .cloned()
                    .unwrap_or(serde_json::json!([1.0, 1.0, 1.0]));
                let intensity = entry
                    .get("light_intensity")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(8.0);
                let range = entry
                    .get("light_range")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(6.0);
                result.push(serde_json::json!({
                    "name": expanded_name,
                    "type": "PointLight",
                    "args": {
                        "position": [world_pos[0], world_pos[1], world_pos[2]],
                        "color": color,
                        "intensity": intensity,
                        "range": range
                    }
                }));
            }
            "prefab" => {
                let nested_ref = entry.get("prefab").and_then(|v| v.as_str()).unwrap_or("");
                if nested_ref.is_empty() {
                    return Err(format!(
                        "Prefab entry '{}': kind=prefab but 'prefab' field is empty",
                        expanded_name
                    ));
                }
                if call_stack.contains(&nested_ref.to_string()) {
                    return Err(format!(
                        "Prefab '{}': cycle detected (via '{}')",
                        call_stack[0], nested_ref
                    ));
                }
                let nested_def = if let Some(def) = prefab_defs.get(nested_ref) {
                    def.clone()
                } else {
                    let loaded = load_preset_obj(nested_ref, "prefabs");
                    if loaded.is_null() {
                        return Err(format!(
                            "Prefab entry '{}': nested prefab '{}' not found",
                            expanded_name, nested_ref
                        ));
                    }
                    loaded
                };
                call_stack.push(nested_ref.to_string());
                let nested = expand_prefab_entries(
                    &expanded_name,
                    world_pos,
                    world_rot,
                    world_scale,
                    &nested_def,
                    prefab_defs,
                    call_stack,
                )?;
                call_stack.pop();
                result.extend(nested);
            }
            _ => {
                // "prop"
                let collider = entry.get("collider").cloned();
                let mut prop_args = serde_json::json!({
                    "position":    [world_pos[0], world_pos[1], world_pos[2]],
                    "rotation_deg":[world_rot[0], world_rot[1], world_rot[2]],
                    "scale":       [world_scale[0], world_scale[1], world_scale[2]]
                });
                for field in &[
                    "model",
                    "mesh",
                    "material",
                    "texture",
                    "parent",
                    "interactable",
                    "pickup",
                ] {
                    if let Some(v) = entry.get(*field) {
                        prop_args[field] = v.clone();
                    }
                }
                if let Some(c) = collider {
                    prop_args["collider"] = c;
                }
                result.push(serde_json::json!({
                    "name": expanded_name,
                    "type": "Prop",
                    "args": prop_args
                }));
            }
        }
    }

    Ok(result)
}

// Rotate a 3-D local-space offset by a YXZ Euler rotation (degrees).
// Mirrors the rotation part of Prop::model_matrix().
fn rotate_local(pos: [f32; 3], rotation_deg: [f32; 3]) -> [f32; 3] {
    let [px, py, pz] = pos;
    let [pitch_deg, yaw_deg, roll_deg] = rotation_deg;
    let (sp, cp) = (pitch_deg.to_radians().sin(), pitch_deg.to_radians().cos());
    let (sy, cy) = (yaw_deg.to_radians().sin(), yaw_deg.to_radians().cos());
    let (sr, cr) = (roll_deg.to_radians().sin(), roll_deg.to_radians().cos());
    let rx = (cy * cr + sy * sp * sr) * px + (-cy * sr + sy * sp * cr) * py + (sy * cp) * pz;
    let ry = (cp * sr) * px + (cp * cr) * py + (-sp) * pz;
    let rz = (-sy * cr + cy * sp * sr) * px + (sy * sr + cy * sp * cr) * py + (cy * cp) * pz;
    [rx, ry, rz]
}

fn f32_arr3(v: &serde_json::Value, key: &str, default: [f32; 3]) -> [f32; 3] {
    v.get(key)
        .and_then(|a| a.as_array())
        .and_then(|a| {
            if a.len() == 3 {
                Some([
                    a[0].as_f64().unwrap_or(default[0] as f64) as f32,
                    a[1].as_f64().unwrap_or(default[1] as f64) as f32,
                    a[2].as_f64().unwrap_or(default[2] as f64) as f32,
                ])
            } else {
                None
            }
        })
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_norm_str(v: &serde_json::Value) -> String {
        v.get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_lowercase()
            .replace('_', "")
    }

    #[test]
    fn single_instance_expands() {
        let mut assets = vec![
            serde_json::json!({"name":"box_mesh","type":"ProceduralMesh","args":{}}),
            serde_json::json!({"name":"table_set","type":"Prefab","args":{"props":[
                {"name":"table","kind":"prop","mesh":"box_mesh","position":[0,0,0]},
                {"name":"chair","kind":"prop","mesh":"box_mesh","position":[0,0,1]}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"table_set","position":[3,0,-5]}}),
        ];
        expand_prefabs(&mut assets).unwrap();
        let names: Vec<&str> = assets
            .iter()
            .filter(|v| type_norm_str(v) == "prop")
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert!(names.contains(&"inst_table"));
        assert!(names.contains(&"inst_chair"));
        assert!(!assets.iter().any(|v| type_norm_str(v) == "prefab"));
    }

    #[test]
    fn two_instances_with_rotation() {
        let mut assets = vec![
            serde_json::json!({"name":"box","type":"ProceduralMesh","args":{}}),
            serde_json::json!({"name":"pair","type":"Prefab","args":{"props":[
                {"name":"a","kind":"prop","mesh":"box","position":[1,0,0]},
                {"name":"b","kind":"prop","mesh":"box","position":[-1,0,0]}
            ]}}),
            serde_json::json!({"name":"i1","type":"Prop","args":{"prefab":"pair","position":[0,0,0]}}),
            serde_json::json!({"name":"i2","type":"Prop","args":{"prefab":"pair","position":[10,0,0],"rotation_deg":[0,90,0]}}),
        ];
        expand_prefabs(&mut assets).unwrap();
        let props: Vec<_> = assets
            .iter()
            .filter(|v| type_norm_str(v) == "prop")
            .collect();
        assert_eq!(props.len(), 4);
    }

    #[test]
    fn point_light_entry_expands() {
        let mut assets = vec![
            serde_json::json!({"name":"alcove","type":"Prefab","args":{"props":[
                {"name":"lamp","kind":"point_light","position":[0,2,0],
                 "light_color":[1.0,0.9,0.7],"light_intensity":8.0,"light_range":5.0}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"alcove","position":[5,0,-3]}}),
        ];
        expand_prefabs(&mut assets).unwrap();
        let lights: Vec<_> = assets
            .iter()
            .filter(|v| type_norm_str(v) == "pointlight")
            .collect();
        assert_eq!(lights.len(), 1);
        assert_eq!(lights[0]["name"], "inst_lamp");
    }

    #[test]
    fn cycle_is_detected() {
        let mut assets = vec![
            serde_json::json!({"name":"pa","type":"Prefab","args":{"props":[
                {"name":"n","kind":"prefab","prefab":"pb"}
            ]}}),
            serde_json::json!({"name":"pb","type":"Prefab","args":{"props":[
                {"name":"n","kind":"prefab","prefab":"pa"}
            ]}}),
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"pa","position":[0,0,0]}}),
        ];
        let err = expand_prefabs(&mut assets).unwrap_err();
        assert!(err.contains("cycle"));
    }

    #[test]
    fn missing_prefab_returns_error() {
        let mut assets = vec![
            serde_json::json!({"name":"inst","type":"Prop","args":{"prefab":"ghost","position":[0,0,0]}}),
        ];
        let err = expand_prefabs(&mut assets).unwrap_err();
        assert!(err.contains("ghost"));
    }

    #[test]
    fn rotate_local_identity_at_zero_rotation() {
        let result = rotate_local([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!(result[1].abs() < 1e-5);
        assert!(result[2].abs() < 1e-5);
    }

    #[test]
    fn rotate_local_yaw_90_rotates_x_to_neg_z() {
        let result = rotate_local([1.0, 0.0, 0.0], [0.0, 90.0, 0.0]);
        assert!(result[0].abs() < 1e-5);
        assert!(result[1].abs() < 1e-5);
        assert!((result[2] - (-1.0)).abs() < 1e-5);
    }
}
