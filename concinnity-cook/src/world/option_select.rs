// src/world/option_select.rs
// Build-time expansion: OptionSelect -> a name TextLabel + a value TextLabel +
// a HitRegion that fires a "setting:<key>:next" action.
//
// The value label shows a placeholder here; the runtime corrects it to the live
// value on the first frame. Names are prefixed with the OptionSelect's own name
// so generated elements stay scoped to its View via the build pipeline's
// `<view>_*` rule and never collide with hand-authored assets.

use std::collections::HashMap;

use super::expand::{asset_name, type_norm};
use crate::assets::{Font, OptionSelect};

// Where the control group (the `<` button + value + `>`) starts, as a fraction
// of the row width. The name occupies the left part, the control the right.
const CONTROL_FRAC: f32 = 0.42;
// The control group is capped to this fixed width, anchored to the right of the
// row, so on a wide row the control stays a compact column that lines up across
// rows (a narrow row falls back to `CONTROL_FRAC`). Mirrors `world/slider.rs`
// and the settings menu; keep in sync so all rows align.
const MAX_CONTROL_WIDTH: f32 = 360.0;
// Average glyph advance as a fraction of the font pixel size, for placing the
// single-character `<` / `>` glyphs (the built-in font is proportional, so this
// is approximate).
const AVG_ADVANCE_RATIO: f32 = 0.5;
// Padding around the `<` / `>` glyphs and the value, in pixels.
const GLYPH_PAD: f32 = 8.0;
// Placeholder shown until the runtime sets the live value on the first frame.
const VALUE_PLACEHOLDER: &str = "--";

// Replace every OptionSelect asset with the concrete UI assets it expands to.
pub(crate) fn expand_option_selects(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    if !assets.iter().any(|v| type_norm(v) == "optionselect") {
        return Ok(());
    }

    let font_px_by_name = font_sizes(assets);

    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in assets.drain(..) {
        if type_norm(&value) != "optionselect" {
            result.push(value);
            continue;
        }

        let name = asset_name(&value);
        if name.is_empty() {
            return Err("OptionSelect: missing `name`".to_string());
        }
        let args = value
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let select: OptionSelect = serde_json::from_value(args)
            .map_err(|e| format!("OptionSelect '{}': invalid args: {}", name, e))?;

        let default_px = select.font_px;
        let font_px = if select.font.is_empty() {
            default_px
        } else {
            *font_px_by_name.get(&select.font).unwrap_or(&default_px)
        };

        result.extend(expand_one(&name, &select, font_px));
    }

    *assets = result;
    Ok(())
}

// The Sprite/TextLabel child names an OptionSelect named `base` expands to (the
// elements a scroll panel reflows + clips with its row). The HitRegions are
// excluded: they have no asset id and are reflowed by position. Locked to the
// expansion output by `element_names_match_expansion`.
pub(crate) fn element_names(base: &str) -> Vec<String> {
    vec![
        format!("{base}_label"),
        format!("{base}_prev_glyph"),
        format!("{base}_value"),
        format!("{base}_next_glyph"),
    ]
}

fn expand_one(name: &str, s: &OptionSelect, font_px: f32) -> Vec<serde_json::Value> {
    let line_h = font_px * s.text_scale;
    let text_y = s.y + (s.height - line_h) / 2.0;
    let value_name = format!("{}_value", name);

    // Layout, left to right: the name fills the left part; then a `<` button,
    // the value (left-aligned and display-only), and a `>` at the far right.
    // Two non-overlapping click regions only -- `<` cycles to the previous
    // option, and everything to its right (value + `>`) cycles to the next.
    // Overlapping regions must be avoided: UiInputSystem keeps scanning after a
    // setting action fires (it returns no StepResult), so two regions hit by
    // one click would both fire and cancel out.
    let sw = s.stepper_width;
    let glyph_w = font_px * AVG_ADVANCE_RATIO * s.text_scale;
    let ctrl_x = (s.x + s.width * CONTROL_FRAC).max(s.x + s.width - MAX_CONTROL_WIDTH);
    let next_x = ctrl_x + sw;
    let right = s.x + s.width;

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
        // `<` glyph, centered in the prev button.
        label_value(
            &format!("{}_prev_glyph", name),
            "<",
            &s.font,
            ctrl_x + (sw - glyph_w) / 2.0,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // Value (display only), left-aligned just past the `<` button.
        label_value(
            &value_name,
            VALUE_PLACEHOLDER,
            &s.font,
            next_x + GLYPH_PAD,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // `>` glyph, flush to the right edge of the row, so it mirrors the
        // left-aligned name and the row's left/right padding stays symmetric.
        label_value(
            &format!("{}_next_glyph", name),
            ">",
            &s.font,
            right - glyph_w,
            text_y,
            s.value_color,
            s.text_scale,
        ),
        // Prev click region (the `<` button).
        region(
            &format!("{}_prev", name),
            ctrl_x,
            s.y,
            sw,
            s.height,
            &value_name,
            s,
            &format!("setting:{}:prev", s.setting),
        ),
        // Next click region (value + `>`).
        region(
            &format!("{}_next", name),
            next_x,
            s.y,
            right - next_x,
            s.height,
            &value_name,
            s,
            &format!("setting:{}:next", s.setting),
        ),
    ]
}

// Build a HitRegion value scoped to a settings row.
#[allow(clippy::too_many_arguments)]
fn region(
    name: &str,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    value_label: &str,
    s: &OptionSelect,
    action: &str,
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "HitRegion",
        "args": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "label": value_label,
            "hover_color": s.hover_color,
            "hover_scale": s.hover_scale,
            "action": action,
        }
    })
}

// Build a TextLabel value with `centered` pinned to false, matching the menu
// expansion (the post-companion patch would otherwise force centered labels to
// the viewport center).
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
    fn passes_through_without_selects() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Window","args":{}})];
        expand_option_selects(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
    }

    #[test]
    fn expands_to_name_value_glyphs_and_two_stepper_regions() {
        let mut assets = vec![serde_json::json!({
            "name": "opt_vsync",
            "type": "OptionSelect",
            "args": {
                "setting": "vsync", "label": "Vsync",
                "x": 100.0, "y": 200.0, "width": 300.0, "stepper_width": 40.0
            }
        })];
        expand_option_selects(&mut assets).unwrap();

        assert!(!assets.iter().any(|v| type_norm(v) == "optionselect"));

        let lbl = by_name(&assets, "opt_vsync_label");
        assert_eq!(lbl["type"], "TextLabel");
        assert_eq!(lbl["args"]["content"], "Vsync");
        assert_eq!(lbl["args"]["centered"], false);
        assert_eq!(lbl["args"]["x"], 100.0);

        let val = by_name(&assets, "opt_vsync_value");
        assert_eq!(val["args"]["content"], VALUE_PLACEHOLDER);

        // ASCII glyphs (the built-in font atlas is ASCII-only).
        assert_eq!(
            by_name(&assets, "opt_vsync_prev_glyph")["args"]["content"],
            "<"
        );
        assert_eq!(
            by_name(&assets, "opt_vsync_next_glyph")["args"]["content"],
            ">"
        );

        // Two non-overlapping click regions. The row is narrow enough that the
        // fraction wins over the right-anchored cap: ctrl_x =
        // max(100 + 300*0.42, 100 + 300 - 360) = max(226, 40) = 226.
        let prev = by_name(&assets, "opt_vsync_prev");
        assert_eq!(prev["type"], "HitRegion");
        assert_eq!(prev["args"]["action"], "setting:vsync:prev");
        assert_eq!(prev["args"]["label"], "opt_vsync_value");
        assert_eq!(prev["args"]["x"], 226.0);
        assert_eq!(prev["args"]["width"], 40.0);

        let next = by_name(&assets, "opt_vsync_next");
        assert_eq!(next["args"]["action"], "setting:vsync:next");
        assert_eq!(next["args"]["label"], "opt_vsync_value");
        // next starts where prev ends (ctrl_x + stepper_width = 266) -> no overlap.
        assert_eq!(next["args"]["x"], 266.0);
        assert_eq!(next["args"]["width"], 134.0);
    }

    #[test]
    fn missing_name_is_an_error() {
        let mut assets = vec![serde_json::json!({"type":"OptionSelect","args":{}})];
        assert!(expand_option_selects(&mut assets).is_err());
    }

    // `element_names` must list exactly the Sprite/TextLabel children the
    // expansion emits (a scroll panel relies on these to reflow + clip the row).
    #[test]
    fn element_names_match_expansion() {
        let mut assets = vec![serde_json::json!({
            "name": "opt", "type": "OptionSelect",
            "args": { "setting": "vsync", "label": "Vsync" }
        })];
        expand_option_selects(&mut assets).unwrap();
        let emitted: std::collections::HashSet<String> = assets
            .iter()
            .filter(|v| matches!(type_norm(v).as_str(), "textlabel" | "sprite"))
            .map(asset_name)
            .collect();
        let listed: std::collections::HashSet<String> = element_names("opt").into_iter().collect();
        assert_eq!(listed, emitted, "element_names drifted from the expansion");
    }
}
