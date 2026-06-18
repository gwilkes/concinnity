// src/world/shader.rs
// Normalization of legacy shader type names (VertexStage / FragmentStage) to
// the unified ShaderStage type.

// Normalize legacy type names in-place across the whole asset list.
pub(crate) fn normalize_shader_types(asset_values: &mut [serde_json::Value]) {
    for v in asset_values.iter_mut() {
        let t = v
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_lowercase()
            .replace('_', "");
        let kind = match t.as_str() {
            "vertexstage" | "vert" => "vertex",
            "fragmentstage" | "frag" => "fragment",
            _ => continue,
        };
        v["type"] = serde_json::Value::String("ShaderStage".to_string());
        if let Some(args) = v.get_mut("args").and_then(|a| a.as_object_mut()) {
            args.entry("kind")
                .or_insert_with(|| serde_json::Value::String(kind.to_string()));
        }
    }
}

// Normalize a single asset's type/args for the validate_asset path.
// Returns the (possibly updated) type string and args value.
pub fn normalize_single_shader_type(
    asset_type: &str,
    args: &serde_json::Value,
) -> (String, serde_json::Value) {
    let t = asset_type.to_lowercase().replace('_', "");
    let kind = match t.as_str() {
        "vertexstage" | "vert" => "vertex",
        "fragmentstage" | "frag" => "fragment",
        _ => return (asset_type.to_string(), args.clone()),
    };
    let mut new_args = args.clone();
    if let Some(obj) = new_args.as_object_mut() {
        obj.entry("kind")
            .or_insert_with(|| serde_json::Value::String(kind.to_string()));
    }
    ("ShaderStage".to_string(), new_args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_vertex_stage_type() {
        let mut assets = vec![serde_json::json!({"name":"v","type":"VertexStage","args":{}})];
        normalize_shader_types(&mut assets);
        assert_eq!(assets[0]["type"], "ShaderStage");
        assert_eq!(assets[0]["args"]["kind"], "vertex");
    }

    #[test]
    fn normalize_fragment_stage_type() {
        let mut assets = vec![serde_json::json!({"name":"f","type":"FragmentStage","args":{}})];
        normalize_shader_types(&mut assets);
        assert_eq!(assets[0]["type"], "ShaderStage");
        assert_eq!(assets[0]["args"]["kind"], "fragment");
    }

    #[test]
    fn normalize_short_aliases() {
        let mut assets = vec![
            serde_json::json!({"name":"v","type":"vert","args":{}}),
            serde_json::json!({"name":"f","type":"frag","args":{}}),
        ];
        normalize_shader_types(&mut assets);
        assert_eq!(assets[0]["args"]["kind"], "vertex");
        assert_eq!(assets[1]["args"]["kind"], "fragment");
    }

    #[test]
    fn normalize_does_not_overwrite_existing_kind() {
        let mut assets =
            vec![serde_json::json!({"name":"v","type":"VertexStage","args":{"kind":"fragment"}})];
        normalize_shader_types(&mut assets);
        // existing kind preserved
        assert_eq!(assets[0]["args"]["kind"], "fragment");
    }

    #[test]
    fn normalize_non_shader_type_unchanged() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        normalize_shader_types(&mut assets);
        assert_eq!(assets[0]["type"], "Logger");
    }

    #[test]
    fn normalize_single_vertex_stage() {
        let (ty, args) = normalize_single_shader_type("VertexStage", &serde_json::json!({}));
        assert_eq!(ty, "ShaderStage");
        assert_eq!(args["kind"], "vertex");
    }

    #[test]
    fn normalize_single_non_shader_passthrough() {
        let (ty, args) = normalize_single_shader_type("Logger", &serde_json::json!({"x": 1}));
        assert_eq!(ty, "Logger");
        assert_eq!(args["x"], 1);
    }
}
