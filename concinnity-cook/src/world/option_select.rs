// src/world/option_select.rs
// Build-time expansion of an OptionSelect settings row. A setting with more than
// two options expands to a dropdown (name + current value + a downward chevron
// under one click region firing "setting:<key>:open", which opens a floating
// option list at runtime); a setting with two options (an Off/On toggle) expands
// to a `<`/`>` stepper (name + `<` + value + `>` over two regions firing
// "setting:<key>:prev" / ":next"). The option count is read from the shared
// registry in `concinnity_core::gfx::settings`, so the row form always matches
// the setting the engine will apply.
//
// The value label shows a placeholder here; the runtime corrects it to the live
// value on the first frame. Names are prefixed with the OptionSelect's own name
// so generated elements stay scoped to its View via the build pipeline's
// `<view>_*` rule and never collide with hand-authored assets.

use std::collections::HashMap;

use super::expand::{asset_name, type_norm};
use crate::assets::{Font, OptionSelect};

// Whether a setting row expands to a dropdown (more than two options) rather
// than a `<`/`>` stepper. An unknown key (no registered options) falls back to
// the stepper form.
fn is_dropdown(setting: &str) -> bool {
    concinnity_core::gfx::settings::options(setting).is_some_and(|o| o.len() > 2)
}

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

// The Sprite/TextLabel child names an OptionSelect named `base` (for `setting`)
// expands to (the elements a scroll panel reflows + clips with its row). The
// HitRegions are excluded: they have no asset id and are reflowed by position.
// The child set depends on the row form (dropdown vs stepper), so it takes the
// setting key too. Locked to the expansion output by
// `element_names_match_expansion`.
pub(crate) fn element_names(base: &str, setting: &str) -> Vec<String> {
    if is_dropdown(setting) {
        vec![
            format!("{base}_label"),
            format!("{base}_value"),
            format!("{base}_chevron"),
        ]
    } else {
        vec![
            format!("{base}_label"),
            format!("{base}_prev_glyph"),
            format!("{base}_value"),
            format!("{base}_next_glyph"),
        ]
    }
}

fn expand_one(name: &str, s: &OptionSelect, font_px: f32) -> Vec<serde_json::Value> {
    let line_h = font_px * s.text_scale;
    let text_y = s.y + (s.height - line_h) / 2.0;
    let value_name = format!("{}_value", name);

    let glyph_w = font_px * AVG_ADVANCE_RATIO * s.text_scale;
    let ctrl_x = (s.x + s.width * CONTROL_FRAC).max(s.x + s.width - MAX_CONTROL_WIDTH);
    let right = s.x + s.width;

    // A setting with more than two options expands to a dropdown: the name, the
    // current value left-aligned in the control column, and a downward chevron
    // at the far right, all under one region that opens the floating list. The
    // chevron is an ASCII `v` (the built-in font atlas is ASCII-only).
    if is_dropdown(&s.setting) {
        return vec![
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
            // Current value, left-aligned at the start of the control column.
            label_value(
                &value_name,
                VALUE_PLACEHOLDER,
                &s.font,
                ctrl_x + GLYPH_PAD,
                text_y,
                s.value_color,
                s.text_scale,
            ),
            // Downward chevron flush to the right edge (mirrors the stepper `>`).
            label_value(
                &format!("{}_chevron", name),
                "v",
                &s.font,
                right - glyph_w,
                text_y,
                s.value_color,
                s.text_scale,
            ),
            // One click region over the whole control column opens the list.
            region(
                &format!("{}_open", name),
                ctrl_x,
                s.y,
                right - ctrl_x,
                s.height,
                &value_name,
                s,
                &format!("setting:{}:open", s.setting),
            ),
        ];
    }

    // Two options: a `<`/`>` stepper. Layout, left to right: the name fills the
    // left part; then a `<` button, the value (left-aligned and display-only),
    // and a `>` at the far right. Two non-overlapping click regions only -- `<`
    // cycles to the previous option, and everything to its right (value + `>`)
    // cycles to the next. Overlapping regions must be avoided: UiInputSystem
    // keeps scanning after a setting action fires (it returns no StepResult), so
    // two regions hit by one click would both fire and cancel out.
    let sw = s.stepper_width;
    let next_x = ctrl_x + sw;

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

    // A setting with more than two options (window_mode has three) expands to a
    // dropdown: name + value + chevron under a single open region, no `<`/`>`.
    #[test]
    fn expands_to_dropdown_with_open_region() {
        let mut assets = vec![serde_json::json!({
            "name": "opt_wm",
            "type": "OptionSelect",
            "args": {
                "setting": "window_mode", "label": "Window Mode",
                "x": 100.0, "y": 200.0, "width": 300.0
            }
        })];
        expand_option_selects(&mut assets).unwrap();

        assert_eq!(
            by_name(&assets, "opt_wm_label")["args"]["content"],
            "Window Mode"
        );
        assert_eq!(
            by_name(&assets, "opt_wm_value")["args"]["content"],
            VALUE_PLACEHOLDER
        );
        // ASCII chevron (the built-in atlas is ASCII-only), and no stepper glyphs.
        assert_eq!(by_name(&assets, "opt_wm_chevron")["args"]["content"], "v");
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_prev_glyph"));
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_next_glyph"));

        // A single click region opens the floating list; no prev/next regions.
        let open = by_name(&assets, "opt_wm_open");
        assert_eq!(open["type"], "HitRegion");
        assert_eq!(open["args"]["action"], "setting:window_mode:open");
        assert_eq!(open["args"]["label"], "opt_wm_value");
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_prev"));
        assert!(!assets.iter().any(|v| asset_name(v) == "opt_wm_next"));
    }

    // `element_names` must list exactly the Sprite/TextLabel children the
    // expansion emits (a scroll panel relies on these to reflow + clip the row),
    // for both the stepper (vsync, two options) and dropdown (window_mode, three)
    // forms.
    #[test]
    fn element_names_match_expansion() {
        for (setting, label) in [("vsync", "Vsync"), ("window_mode", "Window Mode")] {
            let mut assets = vec![serde_json::json!({
                "name": "opt", "type": "OptionSelect",
                "args": { "setting": setting, "label": label }
            })];
            expand_option_selects(&mut assets).unwrap();
            let emitted: std::collections::HashSet<String> = assets
                .iter()
                .filter(|v| matches!(type_norm(v).as_str(), "textlabel" | "sprite"))
                .map(asset_name)
                .collect();
            let listed: std::collections::HashSet<String> =
                element_names("opt", setting).into_iter().collect();
            assert_eq!(listed, emitted, "element_names drifted for '{setting}'");
        }
    }
}
