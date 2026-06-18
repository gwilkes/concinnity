// src/world/companion.rs
// Inject companion assets implied by the presence of other assets.
//
// Each Component declares its companions in its own file via the
// `companions(args, world)` trait method; see `crate::ecs::CompanionSpec`.
// This module dispatches over the typed registry and applies the specs to a
// world JSONL value list. Two passes:
//
//   1. Injection runs to a fixed point. Each round snapshots the current
//      world, asks every declared asset for its companion specs, then filters
//      out specs whose `asset_type` already appears in the world or whose
//      `name` was already collected this round (by-name dedup within a round
//      so a single asset can request multiple companions of the same type,
//      e.g. GraphicsSystem's default vertex + fragment ShaderStages).
//
//   2. The default-font pass (`apply_default_font`) runs once with the final
//      world: when there are TextLabels but no Font, it injects one from the
//      engine's bundled font and points empty-`font` labels at it.

use crate::ecs::ComponentType;
use std::collections::HashSet;

// Same normalization the rest of the codebase uses for type-name dedup:
// lowercase + strip underscores. Keeps "Camera3DSystem" / "camera3d_system"
// from being treated as different types.
fn type_norm_str(s: &str) -> String {
    s.to_lowercase().replace('_', "")
}

fn asset_type_norm(v: &serde_json::Value) -> String {
    v.get("type")
        .and_then(|t| t.as_str())
        .map(type_norm_str)
        .unwrap_or_default()
}

// Dispatch a companion lookup for one asset to the component registry.
fn companions_for_type(
    asset_type: &str,
    args: &serde_json::Value,
    world: &[serde_json::Value],
) -> Vec<crate::ecs::CompanionSpec> {
    if let Ok(ct) = ComponentType::parse(asset_type) {
        ct.companions(args, world)
    } else {
        Vec::new()
    }
}

// Pixel size of the auto-injected default font. Deliberately larger than a
// hand-declared Font's default `size_px`: this is the big, HUD-style fallback
// face that font-less labels render with.
const DEFAULT_FONT_SIZE: u32 = 48;

// Ensure a font-less world still has a usable Font. When any TextLabel exists
// but no Font is declared, inject one compiled from the engine's bundled default
// font (a Font with no `path`), named after that file exactly as `cn add` would
// name it. Then point every empty-`font` label at it (and default such labels to
// centered, HUD-style). A world that declares its own Font pre-empts all of this.
fn apply_default_font(assets: &mut Vec<serde_json::Value>) {
    let has_text = assets.iter().any(|v| asset_type_norm(v) == "textlabel");
    if !has_text {
        return;
    }
    let font_name = super::asset_name_from_path(crate::font::BUILTIN_FONT_FILE);
    let has_font = assets.iter().any(|v| asset_type_norm(v) == "font");
    if !has_font {
        assets.push(serde_json::json!({
            "name": font_name,
            "type": "Font",
            "args": { "size_px": DEFAULT_FONT_SIZE },
        }));
    }
    // Patch only if the default actually landed (nothing pre-empted it).
    let default_present = assets.iter().any(|v| {
        asset_type_norm(v) == "font"
            && v.get("name").and_then(|n| n.as_str()) == Some(font_name.as_str())
    });
    if !default_present {
        return;
    }
    for v in assets.iter_mut() {
        if asset_type_norm(v) != "textlabel" {
            continue;
        }
        let Some(args) = v.get_mut("args").and_then(|a| a.as_object_mut()) else {
            continue;
        };
        let font_empty = args
            .get("font")
            .and_then(|x| x.as_str())
            .map(|s| s.is_empty())
            .unwrap_or(true);
        if font_empty {
            args.insert(
                "font".to_string(),
                serde_json::Value::String(font_name.clone()),
            );
            args.entry("centered")
                .or_insert_with(|| serde_json::json!(true));
        }
    }
}

pub fn inject_companions(assets: &mut Vec<serde_json::Value>) {
    loop {
        // Snapshot the world for this round. New companions added in this
        // round only enter the visible set on the next iteration; that keeps
        // multi-spec batches (e.g. vertex + fragment ShaderStages) from
        // shadowing each other through the per-spec type-dedup.
        let snapshot = assets.clone();
        let present_types: HashSet<String> = snapshot.iter().map(asset_type_norm).collect();

        // Collect every spec implied by every declared asset.
        let mut candidates: Vec<crate::ecs::CompanionSpec> = Vec::new();
        for value in &snapshot {
            let Some(t) = value.get("type").and_then(|s| s.as_str()) else {
                continue;
            };
            let args = value
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            candidates.extend(companions_for_type(t, &args, &snapshot));
        }

        // Apply: skip a spec whose asset_type already exists in the
        // pre-round world. Within the round, dedup by `name` so two specs
        // sharing a type (e.g. the default shader stages) both pass.
        let mut seen_names: HashSet<String> = HashSet::new();
        let mut to_inject = Vec::new();
        for spec in candidates {
            if present_types.contains(&type_norm_str(spec.asset_type)) {
                continue;
            }
            if !seen_names.insert(spec.name.to_string()) {
                continue;
            }
            to_inject.push(spec);
        }

        if to_inject.is_empty() {
            break;
        }
        for spec in to_inject {
            assets.push(serde_json::json!({
                "name": spec.name,
                "type": spec.asset_type,
                "args": spec.args,
            }));
        }
    }

    // Default-font pass: ensure font-less labels have a Font to reference.
    apply_default_font(assets);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn type_norm(v: &serde_json::Value) -> String {
        asset_type_norm(v)
    }

    #[test]
    fn no_injection_without_trigger() {
        let mut assets = vec![serde_json::json!({"name":"w","type":"Window","args":{}})];
        inject_companions(&mut assets);
        assert!(!assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
    }

    #[test]
    fn text_injects_graphics_config_and_font() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject_companions(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "graphicsconfig"));
        assert!(assets.iter().any(|v| type_norm(v) == "font"));
    }

    #[test]
    fn text_does_not_inject_duplicate_graphics_config() {
        let mut assets = vec![
            serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}}),
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
        ];
        inject_companions(&mut assets);
        let gfx_count = assets
            .iter()
            .filter(|v| type_norm(v) == "graphicsconfig")
            .count();
        assert_eq!(gfx_count, 1);
    }

    #[test]
    fn text_does_not_inject_font_when_font_present() {
        let mut assets = vec![
            serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}}),
            serde_json::json!({"name":"f","type":"Font","args":{"path":"my.ttf","size_px":20}}),
        ];
        inject_companions(&mut assets);
        let font_count = assets.iter().filter(|v| type_norm(v) == "font").count();
        assert_eq!(font_count, 1);
    }

    #[test]
    fn text_patches_empty_font_field_to_default() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject_companions(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        let font = label["args"]["font"].as_str().unwrap_or("");
        // The default font is named after its bundled file, exactly as `cn add`
        // would name it (`Questrial-Regular.ttf` -> `Questrial-Regular`); the
        // patched reference must match, and a Font by that name must exist.
        let expected = crate::world::asset_name_from_path(crate::font::BUILTIN_FONT_FILE);
        assert_eq!(font, expected);
        assert!(
            assets
                .iter()
                .any(|v| type_norm(v) == "font" && v["name"].as_str() == Some(expected.as_str()))
        );
    }

    #[test]
    fn text_sets_centered_when_patching() {
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject_companions(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        assert_eq!(label["args"]["centered"], true);
    }

    #[test]
    fn text_does_not_override_explicit_font_on_label() {
        let mut assets = vec![serde_json::json!({
            "name": "t",
            "type": "TextLabel",
            "args": {"content": "hi", "font": "myfont"}
        })];
        inject_companions(&mut assets);
        let label = assets.iter().find(|v| type_norm(v) == "textlabel").unwrap();
        assert_eq!(label["args"]["font"].as_str().unwrap(), "myfont");
    }

    #[test]
    fn graphics_config_injects_default_shader_stages() {
        // TextLabel injects a GraphicsConfig, which in turn injects the default
        // vertex + fragment shader stages.
        let mut assets =
            vec![serde_json::json!({"name":"t","type":"TextLabel","args":{"content":"hi"}})];
        inject_companions(&mut assets);
        let kinds: Vec<&str> = assets
            .iter()
            .filter(|v| type_norm(v) == "shaderstage")
            .filter_map(|v| v["args"]["kind"].as_str())
            .collect();
        assert!(kinds.contains(&"vertex"));
        assert!(kinds.contains(&"fragment"));
    }

    #[test]
    fn does_not_inject_shader_stages_when_one_declared() {
        let mut assets = vec![
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
            serde_json::json!({
                "name":"vert","type":"ShaderStage",
                "args":{"kind":"vertex","source":"custom.metal"}
            }),
        ];
        inject_companions(&mut assets);
        let shader_count = assets
            .iter()
            .filter(|v| type_norm(v) == "shaderstage")
            .count();
        assert_eq!(shader_count, 1);
    }

    #[test]
    fn graphics_config_injects_window() {
        let mut assets = vec![serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}})];
        inject_companions(&mut assets);
        assert!(assets.iter().any(|v| type_norm(v) == "window"));
    }

    #[test]
    fn camera3d_injects_no_companions() {
        // The camera controller is now a field on Camera3D, not an injected
        // system, so a bare Camera3D pulls in nothing.
        let mut assets = vec![serde_json::json!({"name":"c","type":"Camera3D","args":{}})];
        inject_companions(&mut assets);
        assert_eq!(assets.len(), 1);
        assert!(type_norm(&assets[0]) == "camera3d");
    }
}
