// src/world/room.rs
// Auto-expansion of Room texture references into implicit Texture assets.

const GENERATORS: &[&str] = &[
    "brick", "checker", "concrete", "grass", "sky", "wood", "tile", "metal",
];

// Auto-create Texture assets for Room texture fields that name a known
// procedural generator but don't already reference a declared Texture asset.
// This lets authors write e.g. `wall_texture: "brick"` without a separate
// Texture entry.
pub(crate) fn expand_room_textures(asset_values: &mut Vec<serde_json::Value>) {
    let existing: std::collections::HashSet<String> = asset_values
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .collect();

    let mut auto: Vec<serde_json::Value> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for value in asset_values.iter() {
        let type_norm = value
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_lowercase()
            .replace('_', "");
        if type_norm != "room" {
            continue;
        }
        if let Some(args) = value.get("args") {
            for field in &[
                "texture",
                "wall_texture",
                "floor_texture",
                "ceiling_texture",
            ] {
                if let Some(name) = args
                    .get(*field)
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    && !existing.contains(name)
                    && !seen.contains(name)
                    && GENERATORS.contains(&name)
                {
                    auto.push(serde_json::json!({
                        "name": name,
                        "type": "Texture",
                        "args": {"generator": name}
                    }));
                    seen.insert(name.to_string());
                    tracing::debug!("build: auto-created Texture '{}' for Room reference", name);
                }
            }
        }
    }

    if !auto.is_empty() {
        // Prepend so auto-created textures are compiled before Room components.
        auto.append(asset_values);
        *asset_values = auto;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_creates_known_generator() {
        let mut assets =
            vec![serde_json::json!({"name":"r","type":"Room","args":{"wall_texture":"brick"}})];
        expand_room_textures(&mut assets);
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "brick");
        assert_eq!(assets[0]["type"], "Texture");
        assert_eq!(assets[0]["args"]["generator"], "brick");
    }

    #[test]
    fn no_duplicate_if_already_declared() {
        let mut assets = vec![
            serde_json::json!({"name":"brick","type":"Texture","args":{"generator":"brick"}}),
            serde_json::json!({"name":"r","type":"Room","args":{"wall_texture":"brick"}}),
        ];
        expand_room_textures(&mut assets);
        assert_eq!(assets.iter().filter(|v| v["name"] == "brick").count(), 1);
    }

    #[test]
    fn unknown_generator_not_created() {
        let mut assets = vec![
            serde_json::json!({"name":"r","type":"Room","args":{"wall_texture":"unknown_gen"}}),
        ];
        expand_room_textures(&mut assets);
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn multiple_fields_all_auto_created() {
        let mut assets = vec![serde_json::json!({
            "name": "r",
            "type": "Room",
            "args": {
                "wall_texture": "brick",
                "floor_texture": "wood",
                "ceiling_texture": "concrete"
            }
        })];
        expand_room_textures(&mut assets);
        let tex_names: Vec<&str> = assets
            .iter()
            .filter(|v| v["type"] == "Texture")
            .filter_map(|v| v["name"].as_str())
            .collect();
        assert!(tex_names.contains(&"brick"));
        assert!(tex_names.contains(&"wood"));
        assert!(tex_names.contains(&"concrete"));
    }

    #[test]
    fn same_generator_used_twice_only_created_once() {
        let mut assets = vec![serde_json::json!({
            "name": "r",
            "type": "Room",
            "args": {"wall_texture": "brick", "floor_texture": "brick"}
        })];
        expand_room_textures(&mut assets);
        assert_eq!(assets.iter().filter(|v| v["name"] == "brick").count(), 1);
    }

    #[test]
    fn non_room_assets_unchanged() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Logger","args":{}})];
        expand_room_textures(&mut assets);
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["type"], "Logger");
    }
}
