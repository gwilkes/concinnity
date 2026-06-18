// src/world/slider.rs
// Build-time expansion: Slider -> a name TextLabel + a value TextLabel + a
// track Sprite + a handle Sprite + a HitRegion that fires a
// "setting:<key>:drag" action (carrying the handle so the runtime can move it).
//
// The handle and value label show a placeholder position here; the runtime
// corrects them to the live value on the first frame and while dragging. Names
// are prefixed with the Slider's own name so generated elements stay scoped to
// its View via the build pipeline's `<view>_*` rule and never collide with
// hand-authored assets.

use std::collections::HashMap;

use super::expand::{asset_name, type_norm};
use crate::assets::{Font, Slider};

// Where the control group (track + value) starts, as a fraction of the row
// width. The name occupies the left part, the control the right. Matches
// `OptionSelect` so slider and cycle rows line up in a shared menu.
const CONTROL_FRAC: f32 = 0.42;
// The control group is capped to this fixed width, anchored to the right of the
// row, so on a wide row the control stays a compact column aligned with the
// cycle rows (a narrow row falls back to `CONTROL_FRAC`). Mirrors
// `world/option_select.rs` and the settings menu; keep in sync.
const MAX_CONTROL_WIDTH: f32 = 360.0;
// Average glyph advance as a fraction of the font pixel size (the built-in font
// is proportional, so this is approximate). Used to reserve room for the value.
const AVG_ADVANCE_RATIO: f32 = 0.5;
// Characters of value text to reserve at the right (e.g. "+1.0 EV").
const VALUE_CHARS: f32 = 7.0;
// Placeholder shown until the runtime sets the live value on the first frame.
const VALUE_PLACEHOLDER: &str = "--";

// Replace every Slider asset with the concrete UI assets it expands to.
pub(crate) fn expand_sliders(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    if !assets.iter().any(|v| type_norm(v) == "slider") {
        return Ok(());
    }

    let font_px_by_name = font_sizes(assets);

    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in assets.drain(..) {
        if type_norm(&value) != "slider" {
            result.push(value);
            continue;
        }

        let name = asset_name(&value);
        if name.is_empty() {
            return Err("Slider: missing `name`".to_string());
        }
        let args = value
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let slider: Slider = serde_json::from_value(args)
            .map_err(|e| format!("Slider '{}': invalid args: {}", name, e))?;

        let default_px = slider.font_px;
        let font_px = if slider.font.is_empty() {
            default_px
        } else {
            *font_px_by_name.get(&slider.font).unwrap_or(&default_px)
        };

        result.extend(expand_one(&name, &slider, font_px));
    }

    *assets = result;
    Ok(())
}

// The Sprite/TextLabel child names a Slider named `base` expands to (the
// elements a scroll panel reflows + clips with its row). The drag HitRegion is
// excluded: it has no asset id and is reflowed by position. Locked to the
// expansion output by `element_names_match_expansion`.
pub(crate) fn element_names(base: &str) -> Vec<String> {
    vec![
        format!("{base}_label"),
        format!("{base}_value"),
        format!("{base}_track"),
        format!("{base}_handle"),
    ]
}

fn expand_one(name: &str, s: &Slider, font_px: f32) -> Vec<serde_json::Value> {
    let glyph_w = font_px * AVG_ADVANCE_RATIO * s.text_scale;
    let line_h = font_px * s.text_scale;
    let text_y = s.y + (s.height - line_h) / 2.0;

    // Layout, left to right: the name fills the left part; then a track bar, and
    // the value text at the far right. The handle rides on the track.
    let ctrl_x = (s.x + s.width * CONTROL_FRAC).max(s.x + s.width - MAX_CONTROL_WIDTH);
    let right = s.x + s.width;
    let value_w = glyph_w * VALUE_CHARS;
    let value_x = right - value_w;
    let gap = glyph_w;
    let track_x = ctrl_x;
    let track_w = (value_x - gap - track_x).max(20.0);
    let track_h = (s.height * 0.12).max(4.0);
    let track_y = s.y + (s.height - track_h) / 2.0;
    let handle_w = (s.height * 0.45).max(10.0);
    let handle_h = (s.height * 0.7).max(16.0);
    let handle_y = s.y + (s.height - handle_h) / 2.0;

    let value_name = format!("{}_value", name);
    let handle_name = format!("{}_handle", name);

    vec![
        // Name (left).
        label_value(
            &format!("{}_label", name),
            &s.label,
            &s.font,
            s.x,
            text_y,
            s.text_color,
            s.text_scale,
        ),
        // Track bar (static background).
        sprite_value(
            &format!("{}_track", name),
            track_x,
            track_y,
            track_w,
            track_h,
            s.track_color,
        ),
        // Handle (placed at the left here; the runtime moves it to the live
        // value's fraction on the first frame and while dragging).
        sprite_value(
            &handle_name,
            track_x,
            handle_y,
            handle_w,
            handle_h,
            s.handle_color,
        ),
        // Value text (display only) at the far right.
        label_value(
            &value_name,
            VALUE_PLACEHOLDER,
            &s.font,
            value_x,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // Drag region spanning the track over the full row height. `label`
        // points at the value text (the runtime updates it); `drag_handle`
        // points at the handle sprite (the runtime moves it).
        serde_json::json!({
            "name": format!("{}_drag", name),
            "type": "HitRegion",
            "args": {
                "x": track_x,
                "y": s.y,
                "width": track_w,
                "height": s.height,
                "label": value_name,
                "drag_handle": handle_name,
                "action": format!("setting:{}:drag", s.setting),
            }
        }),
    ]
}

// Build a Sprite value (a solid-coloured rectangle).
fn sprite_value(
    name: &str,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    tint: [f32; 4],
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "Sprite",
        "args": { "x": x, "y": y, "width": width, "height": height, "tint": tint }
    })
}

// Build a TextLabel value with `centered` pinned to false, matching the other
// settings expansions (the post-companion patch would otherwise force centered
// labels to the viewport center).
fn label_value(
    name: &str,
    content: &str,
    font: &str,
    x: f32,
    y: f32,
    color: [f32; 3],
    scale: f32,
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "TextLabel",
        "args": {
            "content": content,
            "font": font,
            "x": x,
            "y": y,
            "color": color,
            "scale": scale,
            "centered": false,
        }
    })
}

// Map of declared Font name to its pixel size, for vertical centering.
fn font_sizes(assets: &[serde_json::Value]) -> HashMap<String, f32> {
    let mut out = HashMap::new();
    for v in assets {
        if type_norm(v) != "font" {
            continue;
        }
        let name = asset_name(v);
        if name.is_empty() {
            continue;
        }
        let px = v
            .get("args")
            .and_then(|a| a.get("size_px"))
            .and_then(|x| x.as_u64())
            .unwrap_or_else(|| Font::default().size_px as u64) as f32;
        out.insert(name, px);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_name<'a>(assets: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
        assets
            .iter()
            .find(|v| asset_name(v) == name)
            .unwrap_or_else(|| panic!("no asset named {name}"))
    }

    #[test]
    fn passes_through_without_sliders() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Window","args":{}})];
        expand_sliders(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn expands_to_label_value_track_handle_and_drag_region() {
        let mut assets = vec![serde_json::json!({
            "name": "sld_exposure",
            "type": "Slider",
            "args": {
                "setting": "exposure", "label": "Exposure",
                "x": 100.0, "y": 200.0, "width": 400.0, "height": 48.0
            }
        })];
        expand_sliders(&mut assets).unwrap();

        assert!(!assets.iter().any(|v| type_norm(v) == "slider"));

        let lbl = by_name(&assets, "sld_exposure_label");
        assert_eq!(lbl["type"], "TextLabel");
        assert_eq!(lbl["args"]["content"], "Exposure");
        assert_eq!(lbl["args"]["centered"], false);
        assert_eq!(lbl["args"]["x"], 100.0);

        let val = by_name(&assets, "sld_exposure_value");
        assert_eq!(val["type"], "TextLabel");
        assert_eq!(val["args"]["content"], VALUE_PLACEHOLDER);

        assert_eq!(by_name(&assets, "sld_exposure_track")["type"], "Sprite");
        assert_eq!(by_name(&assets, "sld_exposure_handle")["type"], "Sprite");

        // The drag region carries the value label, the handle, and the action.
        let drag = by_name(&assets, "sld_exposure_drag");
        assert_eq!(drag["type"], "HitRegion");
        assert_eq!(drag["args"]["action"], "setting:exposure:drag");
        assert_eq!(drag["args"]["label"], "sld_exposure_value");
        assert_eq!(drag["args"]["drag_handle"], "sld_exposure_handle");
        // The region spans the track, at the full row height.
        assert_eq!(drag["args"]["y"], 200.0);
        assert_eq!(drag["args"]["height"], 48.0);
    }

    #[test]
    fn missing_name_is_an_error() {
        let mut assets = vec![serde_json::json!({"type":"Slider","args":{}})];
        assert!(expand_sliders(&mut assets).is_err());
    }

    // `element_names` must list exactly the Sprite/TextLabel children the
    // expansion emits (a scroll panel relies on these to reflow + clip the row).
    #[test]
    fn element_names_match_expansion() {
        let mut assets = vec![serde_json::json!({
            "name": "sld", "type": "Slider",
            "args": { "setting": "exposure", "label": "Exposure" }
        })];
        expand_sliders(&mut assets).unwrap();
        let emitted: std::collections::HashSet<String> = assets
            .iter()
            .filter(|v| matches!(type_norm(v).as_str(), "textlabel" | "sprite"))
            .map(asset_name)
            .collect();
        let listed: std::collections::HashSet<String> = element_names("sld").into_iter().collect();
        assert_eq!(listed, emitted, "element_names drifted from the expansion");
    }
}
