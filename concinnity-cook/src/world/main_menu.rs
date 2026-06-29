// Build-time expansion: MainMenu -> View + Sprite (backdrop) + TextLabel +
// HitRegion per item + an optional Escape KeyBinding + an optional in-engine
// cursor Sprite, plus a generated settings sub-view when an item asks for one.
//
// Everything is prefixed with the menu's own name so the build pipeline's
// `<view>_*` rule scopes each generated UI element to the menu's View. The menu
// shows/hides at runtime purely as a View visibility flip; this pass adds no
// runtime behaviour, only the assets the existing UI systems already drive.

use std::collections::{HashMap, HashSet};

use super::expand::{asset_name, type_norm};
use crate::assets::{Font, MainMenu};
use crate::gfx::overlay::UI_REFERENCE_SIZE;

// Average glyph advance as a fraction of the font pixel size, used to estimate
// a label's width so item text can be roughly centered at build time without
// font metrics. The built-in font is proportional, so this is an approximation
// (good enough for short menu labels); a custom font may center less precisely.
const AVG_ADVANCE_RATIO: f32 = 0.5;

// Settings tabs, left to right: (view-name suffix, tab label). Each tab is its
// own View; the active tab bakes its own highlight, so switching tabs needs no
// runtime state, only a view:show.
const SETTINGS_TABS: [(&str, &str); 3] = [
    ("video", "Video"),
    ("audio", "Audio"),
    ("controls", "Controls"),
];
// Setting rows per tab, top to bottom: (setting key, display label). The runtime
// (`concinnity_client::gfx::settings`) knows each key's options and how to apply
// it; this only chooses which rows appear.
const VIDEO_ROWS: [(&str, &str); 3] = [
    ("vsync", "Vsync"),
    ("window_mode", "Window Mode"),
    ("window_size", "Window Size"),
];
// Rows tucked under the Video "Advanced" collapsible group (collapsed by
// default), so the top of the Video tab stays uncrowded. More live
// post-process sliders join these later. Cycle rows then slider rows.
const VIDEO_ADVANCED_ROWS: [(&str, &str); 7] = [
    ("render_scale", "Render Scale"),
    // Display-output / upscaling preferences (Off/On + render-scale cycle).
    // Restart-required and independent of the quality preset.
    ("temporal_upscaling", "Temporal Upscaling"),
    ("hdr_display", "HDR Display"),
    ("hdr_pq", "HDR10 (PQ)"),
    // System / streaming restart preferences. Buffering depth, two-pass occlusion
    // culling, and texture-streaming quality (pool size + upload budget together).
    ("frames_in_flight", "Frame Buffering"),
    ("occlusion_two_pass", "Occlusion Culling"),
    ("texture_quality", "Texture Quality"),
];
// Live post-process sliders in the Advanced group. Each key's value range,
// display format, and apply path live in the client (`concinnity_client::gfx::settings` +
// `graphics_system`); a row here only chooses which sliders appear. All but
// `ambient_intensity` are pure `PostProcessParams` fields applied via
// `update_post_process`; `ambient_intensity` rides a dedicated backend setter
// (Metal live; see the client `graphics_system`).
const VIDEO_ADVANCED_SLIDERS: [(&str, &str); 8] = [
    ("exposure", "Exposure"),
    ("bloom_intensity", "Bloom"),
    ("bloom_threshold", "Bloom Threshold"),
    ("bloom_knee", "Bloom Knee"),
    ("vignette", "Vignette"),
    ("lut_strength", "Color Grade"),
    ("ambient_intensity", "Ambient"),
    // Camera vertical field of view (degrees). Live, independent of the preset.
    ("fov", "Field of View"),
];
// Quality toggles in the Video "Quality" collapsible group (collapsed by
// default): the heavier render features. Each is an Off/On cycle row. The
// client (`concinnity_client::gfx::settings` + `graphics_system`) knows each key's options and
// applies it live by rebuilding the affected render resources; on backends
// without a live path the choice persists and applies at the next launch.
const VIDEO_QUALITY_ROWS: [(&str, &str); 14] = [
    ("aa_mode", "Anti-Aliasing"),
    ("ssao", "Ambient Occlusion"),
    ("ssr", "Screen-Space Reflections"),
    ("ray_traced_reflections", "Ray-Traced Reflections"),
    // Reflection blur resolution dropdown, grouped under the reflection toggles
    // it governs (SSR + ray-traced).
    ("reflection_blur_resolution", "Reflection Blur"),
    ("ssgi", "Global Illumination"),
    // SSGI gather sub-quality (multi-option dropdowns), grouped under the GI
    // toggle. The runtime knows each key's options and applies them live.
    ("ssgi_resolution", "GI Resolution"),
    ("ssgi_rays", "GI Rays"),
    ("ssgi_steps", "GI Steps"),
    // Shadow quality: cascade map resolution (restart-required) + re-render
    // cadence (live) + distance (live). Preset-governed like the toggles above.
    ("shadow_map_size", "Shadow Resolution"),
    ("shadow_update", "Shadow Update"),
    ("shadow_distance", "Shadow Distance"),
    ("auto_exposure", "Auto Exposure"),
    // Anisotropic texture filtering (restart-required). Preset-governed like the
    // toggles above.
    ("anisotropy", "Anisotropic Filtering"),
];
// Per-feature sub-quality sliders in the Video "Quality" group, tuning the
// features the toggles / dropdowns above enable. Applied live on Metal by
// mutating the backend's stored *Settings (no pass rebuild); look-tuning knobs,
// independent of the master quality preset.
const VIDEO_QUALITY_SLIDERS: [(&str, &str); 9] = [
    ("ssao_radius", "AO Radius"),
    ("ssao_intensity", "AO Intensity"),
    ("ssr_intensity", "Reflection Intensity"),
    ("ssr_max_distance", "Reflection Distance"),
    ("ssgi_intensity", "GI Intensity"),
    ("ssgi_max_distance", "GI Distance"),
    ("auto_exposure_min_ev", "Auto Exposure Min"),
    ("auto_exposure_max_ev", "Auto Exposure Max"),
    ("auto_exposure_speed", "Auto Exposure Speed"),
];
const AUDIO_ROWS: [(&str, &str); 1] = [("master_volume", "Master Volume")];
// Controls-tab sliders, top to bottom: (setting key, display label). Mouse
// sensitivity is a continuous slider (the client maps the 1..100 track to a
// radians-per-pixel value) applied live by the camera controller.
const CONTROLS_SLIDERS: [(&str, &str); 1] = [("mouse_sensitivity", "Sensitivity")];
// Rebindable gameplay actions shown under the Controls tab: (display label,
// setting key). Each emits a clickable row that captures a new key; the client
// (`concinnity_client::gfx::keymap` + `graphics_system`) owns the live key map and applies a
// rebind without a restart. The setting keys match `Bindable::setting_key`.
const CONTROLS_REBINDS: [(&str, &str); 7] = [
    ("Move Forward", "key_forward"),
    ("Move Back", "key_backward"),
    ("Move Left", "key_left"),
    ("Move Right", "key_right"),
    ("Sprint", "key_sprint"),
    ("Jump", "key_jump"),
    ("Interact", "key_interact"),
];
// Read-only key reference shown under the Controls tab: (action, key). Pause
// (Escape) carries cursor-release / menu semantics that are fixed per-backend,
// so it is shown for reference rather than made rebindable.
const CONTROLS_KEYS: [(&str, &str); 1] = [("Pause", "Esc")];
// A non-centered settings tab sizes its rows from the menu button width; a
// centered tab spans most of the window instead (see `SETTINGS_SIDE_MARGIN`).
const SETTINGS_ROW_WIDTH_MULT: f32 = 1.85;
// Settings rows use a smaller text scale than the menu buttons so the longer
// option names fit beside their controls.
const SETTINGS_ROW_SCALE: f32 = 0.6;
// A centered settings tab spans the window minus this side margin on each edge
// (plus the scrollbar gutter), in reference pixels, so the rows use most of the
// screen width instead of a narrow central column.
const SETTINGS_SIDE_MARGIN: f32 = 90.0;
// The interactive control (the value + steppers, or the slider track) sits in a
// fixed-width column anchored to the right of each row, so cycle, slider, and
// key-reference rows line up like a table. Mirrors `CONTROL_FRAC` /
// `MAX_CONTROL_WIDTH` in `crate::world::option_select` + `crate::world::slider`; keep the
// values in sync so all three column kinds align.
const SETTINGS_CONTROL_FRAC: f32 = 0.42;
const SETTINGS_CONTROL_WIDTH: f32 = 360.0;
// Per-row card backgrounds, drawn behind each settings row so the rows read as a
// table: a semi-transparent dark blue for normal rows and a slightly stronger
// fill behind group headers.
const SETTINGS_ROW_BG: [f32; 4] = [0.10, 0.13, 0.24, 0.55];
const SETTINGS_HEADER_BG: [f32; 4] = [0.16, 0.20, 0.34, 0.70];
// Horizontal padding inside each row card: the row content (name on the left,
// control on the right) is inset by this much from both card edges, so the text
// does not touch the edges and the left/right gaps match.
const SETTINGS_ROW_PAD: f32 = 18.0;
// Top margin of a centered menu as a fraction of the reference height. The menu
// is top-aligned (not vertically centered) so the heading and tab bar hold a
// fixed position when switching between tabs with different row counts.
const TOP_MARGIN_FRAC: f32 = 0.07;
// How many setting rows the settings scroll band shows at once; a tab with more
// body rows than this scrolls (mouse wheel or scrollbar thumb). Sized so the
// band plus the heading, tab bar, and Back button all fit the reference canvas.
const VISIBLE_SETTINGS_ROWS: usize = 5;
// Scrollbar gutter width and its gap from the content band, in reference pixels.
const SCROLLBAR_GAP: f32 = 12.0;
const SCROLLBAR_WIDTH: f32 = 8.0;
// Gap between the scroll band's bottom and the Back button (fixed chrome below
// the band), in reference pixels.
const BACK_GAP: f32 = 24.0;

// Replace every MainMenu asset with the concrete UI assets it expands to.
// Generated names are prefixed with the menu's (unique) asset name, so they
// never collide with hand-authored assets; a collision is a hard error.
pub(crate) fn expand_main_menus(assets: &mut Vec<serde_json::Value>) -> Result<(), String> {
    if !assets.iter().any(|v| type_norm(v) == "mainmenu") {
        return Ok(());
    }

    // Menus lay out against a fixed reference canvas; the renderer uniformly
    // scales the overlay to the live window (see crate::gfx::overlay), so the
    // declared Window size does not affect the menu layout.
    let (win_w, win_h) = (UI_REFERENCE_SIZE[0], UI_REFERENCE_SIZE[1]);
    let font_px_by_name = font_sizes(assets);

    // Names already in use: authored assets plus entries generated by earlier
    // menus. A generated name landing on one of these is rejected.
    let mut taken: HashSet<String> = assets
        .iter()
        .filter(|v| type_norm(v) != "mainmenu")
        .map(asset_name)
        .filter(|n| !n.is_empty())
        .collect();

    let mut result: Vec<serde_json::Value> = Vec::new();
    for value in assets.drain(..) {
        if type_norm(&value) != "mainmenu" {
            result.push(value);
            continue;
        }

        let menu_name = asset_name(&value);
        if menu_name.is_empty() {
            return Err("MainMenu: missing `name`".to_string());
        }
        let args = value
            .get("args")
            .cloned()
            .unwrap_or_else(|| serde_json::json!({}));
        let menu: MainMenu = serde_json::from_value(args)
            .map_err(|e| format!("MainMenu '{}': invalid args: {}", menu_name, e))?;

        let font_px = if menu.font.is_empty() {
            menu.font_px
        } else {
            *font_px_by_name.get(&menu.font).unwrap_or(&menu.font_px)
        };

        for entry in expand_one(&menu_name, &menu, win_w, win_h, font_px) {
            let name = asset_name(&entry);
            if !name.is_empty() && !taken.insert(name.clone()) {
                return Err(format!(
                    "MainMenu '{}': generated asset name '{}' collides with an existing \
                     asset; rename the menu or the conflicting asset",
                    menu_name, name
                ));
            }
            result.push(entry);
        }
    }

    *assets = result;
    Ok(())
}

// Generate every asset for one menu: the main view and its buttons, an optional
// Escape key binding, and (when an item resolves to the settings convenience) a
// settings sub-view with a Back button.
fn expand_one(
    menu_name: &str,
    menu: &MainMenu,
    win_w: f32,
    win_h: f32,
    font_px: f32,
) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut wants_settings = false;

    // Resolve the font. Use the user's font when set; otherwise emit a built-in
    // font for this menu and reference it explicitly. We cannot rely on the
    // auto-injected default font, because that pass only injects one when the
    // world declares no Font at all (a HUD font would suppress it), which would
    // leave the menu labels with no font and no rendered text.
    let font_name = if menu.font.is_empty() {
        let name = format!("{}_font", menu_name);
        out.push(serde_json::json!({
            "name": name,
            "type": "Font",
            "args": { "size_px": font_px as u32 }
        }));
        name
    } else {
        menu.font.clone()
    };

    // Resolve the per-menu convenience actions against this menu's name.
    let items: Vec<(String, String)> = menu
        .items
        .iter()
        .map(|item| {
            let action = match item.action.trim().to_lowercase().as_str() {
                "return" | "close" => "view:hide".to_string(),
                "settings" => {
                    wants_settings = true;
                    format!("view:show:{}_settings_video", menu_name)
                }
                _ => item.action.clone(),
            };
            (item.label.clone(), action)
        })
        .collect();

    out.extend(emit_menu_view(
        menu_name,
        &menu.title,
        &items,
        menu,
        &font_name,
        win_w,
        win_h,
        font_px,
        menu.initial,
    ));

    if !menu.toggle_key.is_empty() {
        out.push(serde_json::json!({
            "name": format!("{}_toggle", menu_name),
            "type": "KeyBinding",
            "args": { "key": menu.toggle_key, "action": format!("view:toggle:{}", menu_name) }
        }));
    }

    if wants_settings {
        for (suffix, _) in SETTINGS_TABS {
            out.extend(emit_settings_tab(
                menu_name, suffix, menu, &font_name, win_w, win_h, font_px,
            ));
        }
    }

    out
}

// Emit one settings tab as its own View: a "Settings" heading, the tab bar
// (this tab highlighted, the others clickable), the tab's setting rows, an
// optional read-only key reference, and a Back button. Each tab is a separate
// View so the active-tab highlight is baked in; switching tabs is a view:show.
fn emit_settings_tab(
    menu_name: &str,
    active: &str,
    style: &MainMenu,
    font: &str,
    win_w: f32,
    win_h: f32,
    font_px: f32,
) -> Vec<serde_json::Value> {
    let view = format!("{}_settings_{}", menu_name, active);
    let mut out = Vec::new();

    out.push(serde_json::json!({
        "name": view,
        "type": "View",
        "args": { "initial": false }
    }));

    if style.dim[3] > 0.0 {
        out.push(serde_json::json!({
            "name": format!("{}_dim", view),
            "type": "Sprite",
            "args": { "x": 0.0, "y": 0.0, "width": win_w, "height": win_h, "tint": style.dim }
        }));
    }

    // Stacked from a fixed top margin: heading, tab bar, a scrollable body band,
    // then the Back button below the band. Top-aligned (not vertically centered)
    // so the heading and tab bar hold position across tabs.
    let pitch = style.button_height + style.row_gap;
    let center_x = if style.centered { win_w / 2.0 } else { style.x };
    let start_y = if style.centered {
        win_h * TOP_MARGIN_FRAC
    } else {
        style.y
    };
    let row_y = |i: usize| start_y + i as f32 * pitch;
    let text_y = |i: usize, scale: f32| row_y(i) + (style.button_height - font_px * scale) / 2.0;

    let row_scale = style.text_scale * SETTINGS_ROW_SCALE;
    // A centered tab spans most of the width (leaving a side margin and room for
    // the scrollbar gutter); a non-centered menu keeps the narrower column form.
    let (row_x, row_width) = if style.centered {
        let w = win_w - 2.0 * SETTINGS_SIDE_MARGIN - SCROLLBAR_GAP - SCROLLBAR_WIDTH;
        (SETTINGS_SIDE_MARGIN, w)
    } else {
        let w = (style.button_width * SETTINGS_ROW_WIDTH_MULT).min(win_w - 80.0);
        (center_x - w / 2.0, w)
    };
    // The row content sits inside the card, inset by a uniform padding on both
    // sides so the name does not touch the left edge and the left/right gaps
    // match. The card background still spans the full row width.
    let content_x = row_x + SETTINGS_ROW_PAD;
    let content_w = (row_width - 2.0 * SETTINGS_ROW_PAD).max(0.0);
    // Left edge of the interactive control column, shared by the cycle, slider,
    // and key-reference rows so their controls line up. A fixed-width column
    // anchored to the right of the content area, falling back to a fraction of
    // the width if the row is too narrow (matches `crate::world::option_select` +
    // `crate::world::slider`).
    let control_x = (content_x + content_w * SETTINGS_CONTROL_FRAC)
        .max(content_x + content_w - SETTINGS_CONTROL_WIDTH);

    // Row 0: heading.
    let title_scale = style.text_scale * 1.4;
    out.push(label_value(
        &format!("{}_title", view),
        "Settings",
        font,
        centered_x(center_x, "Settings", font_px, title_scale),
        text_y(0, title_scale),
        style.text_color,
        title_scale,
    ));

    // Row 1: tab bar, laid out as a centered horizontal row. The active tab is
    // accent-colored with an underline marker and has no button (you are
    // already here); every other tab is a button that switches to its view.
    let tab_scale = style.text_scale * 1.1;
    let tab_text_y = text_y(1, tab_scale);
    let tab_gap = font_px * AVG_ADVANCE_RATIO * tab_scale * 1.2;
    let tab_widths: Vec<f32> = SETTINGS_TABS
        .iter()
        .map(|&(_, label)| text_width(label, font_px, tab_scale))
        .collect();
    let tabs_total: f32 =
        tab_widths.iter().sum::<f32>() + tab_gap * (SETTINGS_TABS.len() as f32 - 1.0);
    let mut tab_x = center_x - tabs_total / 2.0;
    for (&(suffix, label), w) in SETTINGS_TABS.iter().zip(&tab_widths) {
        let is_active = suffix == active;
        let color = if is_active {
            style.hover_color
        } else {
            style.text_color
        };
        let label_name = format!("{}_tab_{}", view, suffix);
        out.push(label_value(
            &label_name,
            label,
            font,
            tab_x,
            tab_text_y,
            color,
            tab_scale,
        ));
        if is_active {
            // Underline marker just below the active tab's text.
            let mark_h = (font_px * tab_scale * 0.08).max(2.0);
            out.push(serde_json::json!({
                "name": format!("{}_tabmark", view),
                "type": "Sprite",
                "args": {
                    "x": tab_x,
                    "y": tab_text_y + font_px * tab_scale + mark_h,
                    "width": *w,
                    "height": mark_h,
                    "tint": [style.hover_color[0], style.hover_color[1], style.hover_color[2], 1.0],
                }
            }));
        } else {
            out.push(serde_json::json!({
                "name": format!("{}_tabbtn_{}", view, suffix),
                "type": "HitRegion",
                "args": {
                    "x": tab_x,
                    "y": row_y(1),
                    "width": *w,
                    "height": style.button_height,
                    "label": label_name,
                    "hover_color": style.hover_color,
                    "hover_scale": tab_scale * style.hover_scale,
                    "action": format!("view:show:{}_settings_{}", menu_name, suffix),
                }
            }));
        }
        tab_x += w + tab_gap;
    }

    // Body band: the rows live in a fixed window starting just below the tab
    // bar. Rows past `VISIBLE_SETTINGS_ROWS` (or revealed by expanding a group)
    // overflow the band and scroll. `text_y_at` centers row text on an absolute
    // row top (the body rows are placed by base_y, not by chrome row index).
    let band_top = row_y(2);
    let band_h = VISIBLE_SETTINGS_ROWS as f32 * pitch;
    let text_y_at = |y: f32, scale: f32| y + (style.button_height - font_px * scale) / 2.0;

    let (body, groups) = settings_body_rows(active);

    // Emit each body row at its band-relative position, collecting a ScrollRow
    // (the row's reflowed/clipped element ids, its height, and its group).
    let mut scroll_rows: Vec<serde_json::Value> = Vec::new();
    for (j, row) in body.iter().enumerate() {
        let base_y = band_top + j as f32 * pitch;
        // A card background behind the row. Pushed (and so drawn) before the
        // row's content and listed first in the row's elements, so it sits behind
        // the row and reflows / clips / hides with it when scrolled or collapsed.
        let is_header = matches!(*row, BodyRow::GroupHeader(..));
        let bg_name = format!("{}_bg_{}", view, j);
        out.push(row_background(
            &bg_name,
            row_x,
            base_y,
            row_width,
            style.button_height,
            if is_header {
                SETTINGS_HEADER_BG
            } else {
                SETTINGS_ROW_BG
            },
        ));
        let (mut elements, group): (Vec<String>, i32) = match *row {
            BodyRow::Option(setting, label, group) => {
                let name = format!("{}_opt_{}", view, setting);
                out.push(option_select_row(
                    &name, setting, label, font, content_x, base_y, content_w, row_scale, style,
                ));
                (super::option_select::element_names(&name), group)
            }
            BodyRow::Slider(setting, label, group) => {
                let name = format!("{}_sld_{}", view, setting);
                out.push(slider_row(
                    &name, setting, label, font, content_x, base_y, content_w, row_scale, style,
                ));
                (super::slider::element_names(&name), group)
            }
            BodyRow::Key(action_label, key, idx, group) => {
                let name = format!("{}_keyname_{}", view, idx);
                let val = format!("{}_keyval_{}", view, idx);
                out.push(label_value(
                    &name,
                    action_label,
                    font,
                    content_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                out.push(label_value(
                    &val,
                    key,
                    font,
                    control_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                (vec![name, val], group)
            }
            BodyRow::Rebind(action_label, setting, idx, group) => {
                let name = format!("{}_rebind_name_{}", view, idx);
                let val = format!("{}_rebind_val_{}", view, idx);
                out.push(label_value(
                    &name,
                    action_label,
                    font,
                    content_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                // The value (the bound key) is a placeholder until the client
                // syncs it to the live key map at init.
                out.push(label_value(
                    &val,
                    "--",
                    font,
                    control_x,
                    text_y_at(base_y, row_scale),
                    style.text_color,
                    row_scale,
                ));
                // A HitRegion over the control column captures a new key on
                // click. Its `label` points at the value label so the client can
                // refresh it; the `setting:<key>:rebind` action is a scroll
                // content region, so it reflows / clips / gates with its row.
                let ctrl_w = (content_x + content_w - control_x).max(0.0);
                out.push(serde_json::json!({
                    "name": format!("{}_rebind_btn_{}", view, idx),
                    "type": "HitRegion",
                    "args": {
                        "x": control_x,
                        "y": base_y,
                        "width": ctrl_w,
                        "height": style.button_height,
                        "label": val,
                        "hover_color": style.hover_color,
                        "hover_scale": row_scale * style.hover_scale,
                        "action": format!("setting:{}:rebind", setting),
                    }
                }));
                (vec![name, val], group)
            }
            BodyRow::GroupHeader(gid, title) => {
                let collapsed = groups.iter().any(|g| g.gid == gid && g.collapsed);
                let header = format!("{}_grphdr_{}", view, gid);
                let header_scale = row_scale * 1.05;
                out.push(label_value(
                    &header,
                    &format!("{} {}", if collapsed { "+" } else { "-" }, title),
                    font,
                    content_x,
                    text_y_at(base_y, header_scale),
                    style.hover_color,
                    header_scale,
                ));
                out.push(serde_json::json!({
                    "name": format!("{}_grpbtn_{}", view, gid),
                    "type": "HitRegion",
                    "args": {
                        "x": row_x,
                        "y": base_y,
                        "width": row_width,
                        "height": style.button_height,
                        "label": header,
                        "hover_color": style.hover_color,
                        "hover_scale": header_scale * style.hover_scale,
                        "action": format!("group:toggle:{}", gid),
                    }
                }));
                (vec![header], -1)
            }
        };
        // The card sits first so it reflows + clips + hides with the row.
        elements.insert(0, bg_name);
        scroll_rows.push(serde_json::json!({
            "elements": elements,
            "base_y": base_y,
            "height": pitch,
            "group": group,
        }));
    }

    // Scrollbar gutter (track + thumb) to the right of the band. The runtime
    // sizes + moves the thumb and hides both when the content fits.
    let track_x = row_x + row_width + SCROLLBAR_GAP;
    let track_tint = [
        style.text_color[0],
        style.text_color[1],
        style.text_color[2],
        0.25,
    ];
    out.push(serde_json::json!({
        "name": format!("{}_scrolltrack", view),
        "type": "Sprite",
        "args": { "x": track_x, "y": band_top, "width": SCROLLBAR_WIDTH, "height": band_h, "tint": track_tint }
    }));
    out.push(serde_json::json!({
        "name": format!("{}_scrollthumb", view),
        "type": "Sprite",
        "args": {
            "x": track_x, "y": band_top, "width": SCROLLBAR_WIDTH, "height": band_h * 0.4,
            "tint": [style.hover_color[0], style.hover_color[1], style.hover_color[2], 1.0],
        }
    }));

    let scroll_groups: Vec<serde_json::Value> = groups
        .iter()
        .map(|g| {
            serde_json::json!({
                "collapsed": g.collapsed,
                "header": format!("{}_grphdr_{}", view, g.gid),
                "title": g.title,
            })
        })
        .collect();
    out.push(serde_json::json!({
        "name": format!("{}_scroll", view),
        "type": "ScrollPanel",
        "args": {
            "x": row_x, "y": band_top, "width": row_width, "height": band_h,
            "rows": scroll_rows,
            "groups": scroll_groups,
            "thumb": format!("{}_scrollthumb", view),
            "track": format!("{}_scrolltrack", view),
            "track_x": track_x, "track_y": band_top,
            "track_w": SCROLLBAR_WIDTH, "track_h": band_h,
        }
    }));

    // Back button: fixed chrome below the band, returns to the menu view.
    let back_y = band_top + band_h + BACK_GAP;
    let back_label = format!("{}_label_back", view);
    out.push(label_value(
        &back_label,
        "Back",
        font,
        centered_x(center_x, "Back", font_px, style.text_scale),
        text_y_at(back_y, style.text_scale),
        style.text_color,
        style.text_scale,
    ));
    out.push(serde_json::json!({
        "name": format!("{}_btn_back", view),
        "type": "HitRegion",
        "args": {
            "x": center_x - style.button_width / 2.0,
            "y": back_y,
            "width": style.button_width,
            "height": style.button_height,
            "label": back_label,
            "hover_color": style.hover_color,
            "hover_scale": style.text_scale * style.hover_scale,
            "action": format!("view:show:{}", menu_name),
        }
    }));

    if style.cursor {
        out.push(serde_json::json!({
            "name": format!("{}_cursor", view),
            "type": "Sprite",
            "args": {
                "x": 0.0, "y": 0.0,
                "width": style.cursor_size, "height": style.cursor_size,
                "tint": style.cursor_color,
                "follow_cursor": true,
            }
        }));
    }

    out
}

// One row of a settings tab's scrollable body.
#[derive(Clone, Copy)]
enum BodyRow {
    // An OptionSelect cycle row: (setting key, label, group index or -1).
    Option(&'static str, &'static str, i32),
    // A Slider row: (setting key, label, group index or -1).
    Slider(&'static str, &'static str, i32),
    // A read-only key-reference row: (action label, key text, index, group).
    Key(&'static str, &'static str, usize, i32),
    // A key-rebind row: (action label, setting key, index, group). Like a Key
    // row but with a HitRegion that captures a new binding on click.
    Rebind(&'static str, &'static str, usize, i32),
    // A collapsible-group header: (group index, title). Always shown.
    GroupHeader(usize, &'static str),
}

// A collapsible group declared by a tab.
struct GroupSpec {
    gid: usize,
    title: &'static str,
    collapsed: bool,
}

// The body rows + collapsible groups for one settings tab, top to bottom.
fn settings_body_rows(active: &str) -> (Vec<BodyRow>, Vec<GroupSpec>) {
    match active {
        "audio" => (
            AUDIO_ROWS
                .iter()
                .map(|&(s, l)| BodyRow::Option(s, l, -1))
                .collect(),
            Vec::new(),
        ),
        "controls" => {
            let mut rows: Vec<BodyRow> = CONTROLS_SLIDERS
                .iter()
                .map(|&(s, l)| BodyRow::Slider(s, l, -1))
                .collect();
            // Rebindable gameplay keys, each a clickable capture row.
            for (i, &(label, setting)) in CONTROLS_REBINDS.iter().enumerate() {
                rows.push(BodyRow::Rebind(label, setting, i, -1));
            }
            // Read-only reference (Pause / Escape) below the rebindable rows.
            for (i, &(action, key)) in CONTROLS_KEYS.iter().enumerate() {
                rows.push(BodyRow::Key(action, key, i, -1));
            }
            (rows, Vec::new())
        }
        // Video: the three core rows, then a "Quality" group holding the
        // render-feature toggles, then an "Advanced" group holding the
        // render-scale row + the live sliders. Both groups collapsed by default
        // so the top of the tab stays uncrowded.
        //
        // A group's `gid` is used at runtime as an index into the panel's groups
        // list, so each group's gid MUST equal its position in the `GroupSpec`
        // vec below (and a row's group tag references that same gid). Quality is
        // declared first, so it is gid 0; Advanced second, so gid 1.
        _ => {
            // The master "Graphics Quality" preset leads the tab (ungrouped, so it
            // is always visible); the runtime cycles Auto/Low/Medium/High/Ultra/
            // Custom and re-derives the toggles + render scale under its ceiling.
            let mut rows: Vec<BodyRow> =
                vec![BodyRow::Option("graphics_quality", "Graphics Quality", -1)];
            rows.extend(VIDEO_ROWS.iter().map(|&(s, l)| BodyRow::Option(s, l, -1)));
            rows.push(BodyRow::GroupHeader(0, "Quality"));
            for &(s, l) in &VIDEO_QUALITY_ROWS {
                rows.push(BodyRow::Option(s, l, 0));
            }
            // The per-feature sub-quality sliders follow the toggles in the same
            // Quality group.
            for &(s, l) in &VIDEO_QUALITY_SLIDERS {
                rows.push(BodyRow::Slider(s, l, 0));
            }
            rows.push(BodyRow::GroupHeader(1, "Advanced"));
            for &(s, l) in &VIDEO_ADVANCED_ROWS {
                rows.push(BodyRow::Option(s, l, 1));
            }
            for &(s, l) in &VIDEO_ADVANCED_SLIDERS {
                rows.push(BodyRow::Slider(s, l, 1));
            }
            (
                rows,
                vec![
                    GroupSpec {
                        gid: 0,
                        title: "Quality",
                        collapsed: true,
                    },
                    GroupSpec {
                        gid: 1,
                        title: "Advanced",
                        collapsed: true,
                    },
                ],
            )
        }
    }
}

// Build an OptionSelect cycle-row asset for the settings body.
#[allow(clippy::too_many_arguments)]
fn option_select_row(
    name: &str,
    setting: &str,
    label: &str,
    font: &str,
    x: f32,
    y: f32,
    width: f32,
    scale: f32,
    style: &MainMenu,
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "OptionSelect",
        "args": {
            "setting": setting,
            "label": label,
            "x": x,
            "y": y,
            "width": width,
            "height": style.button_height,
            "font": font,
            "text_color": style.text_color,
            "value_color": style.text_color,
            "text_scale": scale,
            "hover_color": style.hover_color,
            // `style.hover_scale` is a multiplier on the row's text scale, so the
            // value label keeps its size on hover (only the color changes) unless
            // the menu opts into a grow. The OptionSelect forwards this absolute
            // scale to its value-label hover region.
            "hover_scale": scale * style.hover_scale,
        }
    })
}

// Build a Slider row asset for the settings body.
#[allow(clippy::too_many_arguments)]
fn slider_row(
    name: &str,
    setting: &str,
    label: &str,
    font: &str,
    x: f32,
    y: f32,
    width: f32,
    scale: f32,
    style: &MainMenu,
) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "type": "Slider",
        "args": {
            "setting": setting,
            "label": label,
            "x": x,
            "y": y,
            "width": width,
            "height": style.button_height,
            "font": font,
            "text_color": style.text_color,
            "value_color": style.text_color,
            "text_scale": scale,
            "handle_color": [
                style.hover_color[0], style.hover_color[1], style.hover_color[2], 1.0
            ],
        }
    })
}

// Build a card-background Sprite for one settings row.
fn row_background(
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

// Emit the assets for one menu layer: a View, an optional dim backdrop, an
// optional heading, a TextLabel + HitRegion per item, and an optional cursor.
#[allow(clippy::too_many_arguments)]
fn emit_menu_view(
    view: &str,
    title: &str,
    items: &[(String, String)],
    style: &MainMenu,
    font: &str,
    win_w: f32,
    win_h: f32,
    font_px: f32,
    initial: bool,
) -> Vec<serde_json::Value> {
    let mut out = Vec::new();

    out.push(serde_json::json!({
        "name": view,
        "type": "View",
        "args": { "initial": initial }
    }));

    if style.dim[3] > 0.0 {
        out.push(serde_json::json!({
            "name": format!("{}_dim", view),
            "type": "Sprite",
            "args": { "x": 0.0, "y": 0.0, "width": win_w, "height": win_h, "tint": style.dim }
        }));
    }

    let line_h = font_px * style.text_scale;
    let center_x = if style.centered { win_w / 2.0 } else { style.x };

    let has_title = !title.is_empty();
    let pitch = style.button_height + style.row_gap;
    // Top-aligned from a fixed margin (not vertically centered), per the
    // settings-tab layout, so menu text hugs the top of the overlay.
    let start_y = if style.centered {
        win_h * TOP_MARGIN_FRAC
    } else {
        style.y
    };

    let mut row = 0usize;
    if has_title {
        let title_scale = style.text_scale * 1.4;
        out.push(label_value(
            &format!("{}_title", view),
            title,
            font,
            centered_x(center_x, title, font_px, title_scale),
            start_y + (style.button_height - font_px * title_scale) / 2.0,
            style.text_color,
            title_scale,
        ));
        row += 1;
    }

    for (i, (label, action)) in items.iter().enumerate() {
        let row_y = start_y + row as f32 * pitch;
        let label_name = format!("{}_label_{}", view, i);

        out.push(label_value(
            &label_name,
            label,
            font,
            centered_x(center_x, label, font_px, style.text_scale),
            row_y + (style.button_height - line_h) / 2.0,
            style.text_color,
            style.text_scale,
        ));

        out.push(serde_json::json!({
            "name": format!("{}_btn_{}", view, i),
            "type": "HitRegion",
            "args": {
                "x": center_x - style.button_width / 2.0,
                "y": row_y,
                "width": style.button_width,
                "height": style.button_height,
                "label": label_name,
                "hover_color": style.hover_color,
                "hover_scale": style.text_scale * style.hover_scale,
                "action": action,
            }
        }));
        row += 1;
    }

    if style.cursor {
        out.push(serde_json::json!({
            "name": format!("{}_cursor", view),
            "type": "Sprite",
            "args": {
                "x": 0.0, "y": 0.0,
                "width": style.cursor_size, "height": style.cursor_size,
                "tint": style.cursor_color,
                "follow_cursor": true,
            }
        }));
    }

    out
}

// Build a TextLabel value with `centered` pinned to false: the post-companion
// patch sets `centered: true` for default-font labels otherwise, which would
// stack every menu label on the viewport center.
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

// Estimated rendered width of `text`, from the average glyph advance. The
// built-in font is proportional, so this is an approximation good enough for
// centering and tab layout.
fn text_width(text: &str, font_px: f32, scale: f32) -> f32 {
    text.chars().count() as f32 * font_px * AVG_ADVANCE_RATIO * scale
}

// Left edge that horizontally centers `text` on `center_x`.
fn centered_x(center_x: f32, text: &str, font_px: f32, scale: f32) -> f32 {
    center_x - text_width(text, font_px, scale) / 2.0
}

// Map of declared Font name to its pixel size, for the centering estimate.
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

    // Dump a probe world that opens one settings tab directly, for the real-GPU
    // screenshot smoke (the headless probe cannot click through the menu). Picks
    // the tab from `CN_PROBE_TAB` (video | audio | controls, default controls),
    // expands a MainMenu, then flips that tab's View to `initial` so it shows at
    // launch. `#[ignore]`d: run explicitly, e.g.
    //   CN_PROBE_TAB=controls cargo test -p concinnity-cook \
    //       dump_settings_tab_probe_world -- --ignored
    // then `concinnity debug -f world.jsonl` + `debug_probe.py screenshot`.
    #[test]
    #[ignore]
    fn dump_settings_tab_probe_world() {
        let tab = std::env::var("CN_PROBE_TAB").unwrap_or_else(|_| "controls".to_string());
        let out = std::env::var("CN_PROBE_OUT").unwrap_or_else(|_| "world.jsonl".to_string());
        let mut assets = vec![
            serde_json::json!({"name":"win","type":"Window","args":{"width":1280,"height":720}}),
            serde_json::json!({"name":"gfx","type":"GraphicsConfig","args":{}}),
            serde_json::json!({"name":"main_menu","type":"MainMenu","args":{"title":"Probe"}}),
        ];
        expand_main_menus(&mut assets).unwrap();
        // Show the chosen tab's View at launch (the menu + other tabs off).
        let target = format!("main_menu_settings_{tab}");
        for v in &mut assets {
            if type_norm(v) == "view" {
                v["args"]["initial"] = serde_json::json!(asset_name(v) == target);
            }
        }
        let mut body = String::new();
        for v in &assets {
            body.push_str(&serde_json::to_string(v).unwrap());
            body.push('\n');
        }
        std::fs::write(&out, body).unwrap();
    }

    fn names(assets: &[serde_json::Value]) -> Vec<String> {
        assets.iter().map(asset_name).collect()
    }

    fn by_name<'a>(assets: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
        assets
            .iter()
            .find(|v| asset_name(v) == name)
            .unwrap_or_else(|| panic!("no asset named {name}"))
    }

    #[test]
    fn passes_through_without_menus() {
        let mut assets = vec![serde_json::json!({"name":"x","type":"Window","args":{}})];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0]["type"], "Window");
    }

    #[test]
    fn bare_menu_expands_to_default_layout() {
        let mut assets = vec![serde_json::json!({"name":"main_menu","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();

        // No MainMenu survives.
        assert!(!assets.iter().any(|v| type_norm(v) == "mainmenu"));

        // The main view and a toggle binding exist.
        assert_eq!(by_name(&assets, "main_menu")["type"], "View");
        assert_eq!(by_name(&assets, "main_menu")["args"]["initial"], true);
        assert_eq!(by_name(&assets, "main_menu_toggle")["type"], "KeyBinding");
        assert_eq!(
            by_name(&assets, "main_menu_toggle")["args"]["action"],
            "view:toggle:main_menu"
        );

        // Three items -> three label/button pairs.
        let ns = names(&assets);
        for i in 0..3 {
            assert!(ns.contains(&format!("main_menu_label_{i}")));
            assert!(ns.contains(&format!("main_menu_btn_{i}")));
        }

        // Return resolves to view:hide, Quit passes through, Settings opens the
        // generated sub-view.
        assert_eq!(
            by_name(&assets, "main_menu_btn_0")["args"]["action"],
            "view:hide"
        );
        assert_eq!(
            by_name(&assets, "main_menu_btn_1")["args"]["action"],
            "view:show:main_menu_settings_video"
        );
        assert_eq!(
            by_name(&assets, "main_menu_btn_2")["args"]["action"],
            "quit"
        );

        // The default-tab (video) settings view and its Back button exist.
        assert_eq!(by_name(&assets, "main_menu_settings_video")["type"], "View");
        assert_eq!(
            by_name(&assets, "main_menu_settings_video")["args"]["initial"],
            false
        );
        // Back returns to the menu view (not view:hide, since tabs navigate
        // explicitly rather than as a restore-prev modal).
        assert_eq!(
            by_name(&assets, "main_menu_settings_video_btn_back")["args"]["action"],
            "view:show:main_menu"
        );
        // The video tab carries its own (accent) tab header and a vsync row.
        assert_eq!(
            by_name(&assets, "main_menu_settings_video_tab_video")["args"]["content"],
            "Video"
        );
        let opt = by_name(&assets, "main_menu_settings_video_opt_vsync");
        assert_eq!(opt["type"], "OptionSelect");
        assert_eq!(opt["args"]["setting"], "vsync");

        // A follow-cursor sprite and a backdrop exist for the main view.
        assert_eq!(
            by_name(&assets, "main_menu_cursor")["args"]["follow_cursor"],
            true
        );
        assert_eq!(by_name(&assets, "main_menu_dim")["type"], "Sprite");
    }

    #[test]
    fn video_tab_emits_a_row_per_setting() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        for (setting, label) in [
            ("vsync", "Vsync"),
            ("window_mode", "Window Mode"),
            ("window_size", "Window Size"),
            ("render_scale", "Render Scale"),
        ] {
            let opt = by_name(&assets, &format!("m_settings_video_opt_{setting}"));
            assert_eq!(opt["type"], "OptionSelect");
            assert_eq!(opt["args"]["setting"], setting);
            assert_eq!(opt["args"]["label"], label);
        }
    }

    #[test]
    fn video_tab_leads_with_the_master_quality_row() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        // The master preset row is an ungrouped OptionSelect bound to the
        // graphics_quality setting (the runtime knows its options + how to apply).
        let opt = by_name(&assets, "m_settings_video_opt_graphics_quality");
        assert_eq!(opt["type"], "OptionSelect");
        assert_eq!(opt["args"]["setting"], "graphics_quality");
        assert_eq!(opt["args"]["label"], "Graphics Quality");
        // It leads the tab: it is emitted before the first core row (vsync).
        let master_pos = assets
            .iter()
            .position(|v| asset_name(v) == "m_settings_video_opt_graphics_quality")
            .expect("master row");
        let vsync_pos = assets
            .iter()
            .position(|v| asset_name(v) == "m_settings_video_opt_vsync")
            .expect("vsync row");
        assert!(
            master_pos < vsync_pos,
            "master quality row should lead the tab"
        );
    }

    #[test]
    fn video_tab_emits_an_exposure_slider() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        let sld = by_name(&assets, "m_settings_video_sld_exposure");
        assert_eq!(sld["type"], "Slider");
        assert_eq!(sld["args"]["setting"], "exposure");
        assert_eq!(sld["args"]["label"], "Exposure");
        // The slider carries the menu font through to its own expansion.
        assert_eq!(sld["args"]["font"], "m_font");
        // Sliders are a Video-only row today: the other tabs emit none.
        assert!(
            !assets
                .iter()
                .any(|v| asset_name(v) == "m_settings_audio_sld_exposure")
        );
    }

    #[test]
    fn settings_emits_a_view_per_tab() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        for suffix in ["video", "audio", "controls"] {
            let view = by_name(&assets, &format!("m_settings_{suffix}"));
            assert_eq!(view["type"], "View", "tab view {suffix} missing");
            assert_eq!(view["args"]["initial"], false);
            // Every tab returns to the menu view via Back.
            assert_eq!(
                by_name(&assets, &format!("m_settings_{suffix}_btn_back"))["args"]["action"],
                "view:show:m"
            );
        }
    }

    #[test]
    fn audio_and_controls_tabs_carry_their_rows() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        // Audio: a master-volume row.
        let vol = by_name(&assets, "m_settings_audio_opt_master_volume");
        assert_eq!(vol["type"], "OptionSelect");
        assert_eq!(vol["args"]["setting"], "master_volume");
        // Controls: a mouse-sensitivity slider plus rebind rows and the
        // read-only Pause reference.
        let sens = by_name(&assets, "m_settings_controls_sld_mouse_sensitivity");
        assert_eq!(sens["type"], "Slider");
        assert_eq!(sens["args"]["setting"], "mouse_sensitivity");
        // The read-only Pause reference is display-only (no HitRegion).
        assert_eq!(
            by_name(&assets, "m_settings_controls_keyname_0")["args"]["content"],
            "Pause"
        );
        assert_eq!(
            by_name(&assets, "m_settings_controls_keyval_0")["args"]["content"],
            "Esc"
        );
        assert!(
            !assets
                .iter()
                .any(|v| asset_name(v) == "m_settings_controls_keyname_0_btn")
        );
    }

    // Each rebindable action emits a name + value label and a HitRegion firing
    // its `setting:<key>:rebind` capture action; Pause stays display-only.
    #[test]
    fn controls_tab_emits_rebind_rows() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        for (i, (label, setting)) in [
            ("Move Forward", "key_forward"),
            ("Move Back", "key_backward"),
            ("Move Left", "key_left"),
            ("Move Right", "key_right"),
            ("Sprint", "key_sprint"),
            ("Jump", "key_jump"),
            ("Interact", "key_interact"),
        ]
        .iter()
        .enumerate()
        {
            assert_eq!(
                by_name(&assets, &format!("m_settings_controls_rebind_name_{i}"))["args"]["content"],
                *label
            );
            // A placeholder value, synced to the live key map at runtime.
            assert_eq!(
                by_name(&assets, &format!("m_settings_controls_rebind_val_{i}"))["args"]["content"],
                "--"
            );
            let btn = by_name(&assets, &format!("m_settings_controls_rebind_btn_{i}"));
            assert_eq!(btn["type"], "HitRegion");
            assert_eq!(btn["args"]["action"], format!("setting:{setting}:rebind"));
            // The region's label points at the value so the client refreshes it.
            assert_eq!(
                btn["args"]["label"],
                format!("m_settings_controls_rebind_val_{i}")
            );
        }
        // Pause is read-only: no rebind HitRegion targets it.
        assert!(
            !assets
                .iter()
                .any(|v| asset_name(v).starts_with("m_settings_controls_rebind_btn_7"))
        );
    }

    #[test]
    fn tab_bar_switches_between_tabs() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        // The active tab gets an accent label + underline marker and NO button;
        // the other tabs are buttons that switch to their view.
        assert_eq!(
            by_name(&assets, "m_settings_video_tabmark")["type"],
            "Sprite"
        );
        assert!(
            !assets
                .iter()
                .any(|v| asset_name(v) == "m_settings_video_tabbtn_video")
        );
        assert_eq!(
            by_name(&assets, "m_settings_video_tabbtn_audio")["args"]["action"],
            "view:show:m_settings_audio"
        );
        assert_eq!(
            by_name(&assets, "m_settings_video_tabbtn_controls")["args"]["action"],
            "view:show:m_settings_controls"
        );
        // From the controls tab you can hop back to video.
        assert_eq!(
            by_name(&assets, "m_settings_controls_tabbtn_video")["args"]["action"],
            "view:show:m_settings_video"
        );
    }

    #[test]
    fn labels_are_not_centered_so_layout_wins() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(by_name(&assets, "m_label_0")["args"]["centered"], false);
    }

    #[test]
    fn custom_items_pass_actions_through_verbatim() {
        let mut assets = vec![serde_json::json!({
            "name": "title",
            "type": "MainMenu",
            "args": { "items": [
                {"label":"New Game","action":"scene:level_1"},
                {"label":"Quit","action":"quit"}
            ]}
        })];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(
            by_name(&assets, "title_btn_0")["args"]["action"],
            "scene:level_1"
        );
        assert_eq!(by_name(&assets, "title_btn_1")["args"]["action"], "quit");
        // No settings item -> no settings sub-view.
        assert!(!assets.iter().any(|v| asset_name(v) == "title_settings"));
    }

    #[test]
    fn toggle_key_empty_emits_no_binding() {
        let mut assets = vec![serde_json::json!({
            "name": "m", "type": "MainMenu", "args": { "toggle_key": "" }
        })];
        expand_main_menus(&mut assets).unwrap();
        assert!(!assets.iter().any(|v| type_norm(v) == "keybinding"));
    }

    #[test]
    fn cursor_disabled_emits_no_cursor_sprite() {
        let mut assets = vec![serde_json::json!({
            "name": "m", "type": "MainMenu", "args": { "cursor": false }
        })];
        expand_main_menus(&mut assets).unwrap();
        assert!(!assets.iter().any(|v| asset_name(v) == "m_cursor"));
    }

    #[test]
    fn dim_alpha_zero_emits_no_backdrop() {
        let mut assets = vec![serde_json::json!({
            "name": "m", "type": "MainMenu", "args": { "dim": [0.0, 0.0, 0.0, 0.0] }
        })];
        expand_main_menus(&mut assets).unwrap();
        assert!(!assets.iter().any(|v| asset_name(v) == "m_dim"));
    }

    #[test]
    fn generated_name_collision_is_an_error() {
        let mut assets = vec![
            serde_json::json!({"name":"m","type":"MainMenu","args":{"toggle_key":""}}),
            serde_json::json!({"name":"m_btn_0","type":"Sprite","args":{}}),
        ];
        let err = expand_main_menus(&mut assets).unwrap_err();
        assert!(err.contains("m_btn_0"));
        assert!(err.contains("collides"));
    }

    #[test]
    fn title_emits_a_heading_label() {
        let mut assets = vec![serde_json::json!({
            "name": "m", "type": "MainMenu", "args": { "title": "Paused" }
        })];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(by_name(&assets, "m_title")["args"]["content"], "Paused");
    }

    // The menu emits its own built-in font and references it explicitly. It
    // cannot rely on the auto-injected default font, which is only injected when
    // the world declares no Font at all (a HUD font would suppress it), leaving
    // the labels with no font and no rendered text.
    #[test]
    fn emits_own_font_and_labels_reference_it() {
        let mut assets =
            vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"toggle_key":""}})];
        expand_main_menus(&mut assets).unwrap();
        let font = by_name(&assets, "m_font");
        assert_eq!(font["type"], "Font");
        // No `path` means the menu font compiles from the bundled default font.
        assert!(font["args"].get("path").is_none());
        assert_eq!(by_name(&assets, "m_label_0")["args"]["font"], "m_font");
        // The generated settings sub-view shares the same font.
        assert_eq!(
            by_name(&assets, "m_settings_video_label_back")["args"]["font"],
            "m_font"
        );
        // The emitted OptionSelect carries the menu font through to its own
        // expansion.
        assert_eq!(
            by_name(&assets, "m_settings_video_opt_vsync")["args"]["font"],
            "m_font"
        );
    }

    #[test]
    fn emitted_font_size_follows_font_px() {
        // With no override the emitted font uses the MainMenu `font_px` default.
        let mut assets =
            vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"toggle_key":""}})];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(by_name(&assets, "m_font")["args"]["size_px"], 48);

        // An explicit `font_px` is the size the build leans on for the font it
        // emits when the menu declares none.
        let mut assets = vec![serde_json::json!({
            "name": "m",
            "type": "MainMenu",
            "args": { "toggle_key": "", "font_px": 32 }
        })];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(by_name(&assets, "m_font")["args"]["size_px"], 32);
    }

    #[test]
    fn custom_font_is_used_and_none_emitted() {
        let mut assets = vec![
            serde_json::json!({"name":"f","type":"Font","args":{"path":"my.ttf","size_px":32}}),
            serde_json::json!({"name":"m","type":"MainMenu","args":{"font":"f","toggle_key":""}}),
        ];
        expand_main_menus(&mut assets).unwrap();
        assert!(!assets.iter().any(|v| asset_name(v) == "m_font"));
        assert_eq!(by_name(&assets, "m_label_0")["args"]["font"], "f");
    }

    // The chrome of a settings tab (heading, tab bar, scroll band, scrollbar,
    // Back) stays within the reference canvas. Body rows may overflow the band
    // (that is what scrolling is for) and are clipped, so they are excluded;
    // only the band itself and the fixed chrome must fit.
    #[test]
    fn settings_chrome_and_band_fit_on_screen() {
        let [ref_w, ref_h] = UI_REFERENCE_SIZE;
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        // The Controls tab is the tallest chrome (it carries the most rows under
        // the band); its band + Back must clear the canvas.
        for suffix in ["video", "audio", "controls"] {
            let view = format!("m_settings_{suffix}");
            // The ScrollPanel's band fits.
            let panel = by_name(&assets, &format!("{view}_scroll"));
            let by = panel["args"]["y"].as_f64().unwrap();
            let bh = panel["args"]["height"].as_f64().unwrap();
            assert!(
                by >= 0.0 && by + bh <= ref_h as f64,
                "{view} band off-screen"
            );
            // Back sits below the band but on screen.
            let back = by_name(&assets, &format!("{view}_btn_back"));
            let back_y = back["args"]["y"].as_f64().unwrap();
            let back_h = back["args"]["height"].as_f64().unwrap();
            assert!(back_y >= by + bh, "{view} Back overlaps the band");
            assert!(back_y + back_h <= ref_h as f64, "{view} Back off-screen");
            // The scrollbar gutter stays inside the canvas width.
            let track = by_name(&assets, &format!("{view}_scrolltrack"));
            let tx = track["args"]["x"].as_f64().unwrap();
            let tw = track["args"]["width"].as_f64().unwrap();
            assert!(
                tx + tw <= ref_w as f64,
                "{view} scrollbar off the right edge"
            );
        }
    }

    // The settings body lives in a ScrollPanel: a band rect, a thumb + track,
    // and one row per body element pointing at that element's expanded children.
    #[test]
    fn settings_tab_emits_a_scroll_panel() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        let panel = by_name(&assets, "m_settings_video_scroll");
        assert_eq!(panel["type"], "ScrollPanel");
        assert_eq!(panel["args"]["thumb"], "m_settings_video_scrollthumb");
        assert_eq!(panel["args"]["track"], "m_settings_video_scrolltrack");
        // The thumb + track sprites exist.
        assert_eq!(
            by_name(&assets, "m_settings_video_scrollthumb")["type"],
            "Sprite"
        );
        // The vsync row references the OptionSelect's expanded value label.
        let rows = panel["args"]["rows"].as_array().unwrap();
        let vsync_row = rows
            .iter()
            .find(|r| {
                r["elements"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e == "m_settings_video_opt_vsync_value")
            })
            .expect("a row listing the vsync value label");
        assert_eq!(vsync_row["group"], -1);
    }

    // The Video "Advanced" group (gid 1): a header row that toggles group 1, the
    // render-scale row + exposure slider tagged into group 1, and a ScrollGroup
    // that starts collapsed.
    #[test]
    fn video_advanced_group_collapses_render_scale_and_exposure() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        // Header label + toggle region.
        assert_eq!(
            by_name(&assets, "m_settings_video_grphdr_1")["args"]["content"],
            "+ Advanced"
        );
        assert_eq!(
            by_name(&assets, "m_settings_video_grpbtn_1")["args"]["action"],
            "group:toggle:1"
        );
        // The panel declares the Quality + Advanced groups, both collapsed.
        let panel = by_name(&assets, "m_settings_video_scroll");
        let groups = panel["args"]["groups"].as_array().unwrap();
        assert_eq!(groups.len(), 2);
        let advanced = groups
            .iter()
            .find(|g| g["header"] == "m_settings_video_grphdr_1")
            .expect("Advanced group present");
        assert_eq!(advanced["collapsed"], true);
        // render_scale + exposure rows are tagged into group 1. The first element
        // of each row is its background card, so search every element.
        let rows = panel["args"]["rows"].as_array().unwrap();
        let in_advanced = |needle: &str| {
            rows.iter()
                .find(|r| {
                    r["elements"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|e| e.as_str().unwrap().contains(needle))
                })
                .map(|r| r["group"].as_i64().unwrap())
        };
        assert_eq!(in_advanced("opt_render_scale"), Some(1));
        assert_eq!(in_advanced("sld_exposure"), Some(1));
        // The live post-process sliders also live inside the Advanced group.
        for key in [
            "sld_bloom_intensity",
            "sld_bloom_threshold",
            "sld_bloom_knee",
            "sld_vignette",
            "sld_lut_strength",
            "sld_ambient_intensity",
            "sld_fov",
        ] {
            assert_eq!(
                in_advanced(key),
                Some(1),
                "{key} should be in the Advanced group"
            );
        }
        // The display-output / upscaling preference + system / streaming restart
        // rows also live in Advanced.
        for key in [
            "opt_temporal_upscaling",
            "opt_hdr_display",
            "opt_hdr_pq",
            "opt_frames_in_flight",
            "opt_occlusion_two_pass",
            "opt_texture_quality",
        ] {
            assert_eq!(
                in_advanced(key),
                Some(1),
                "{key} should be in the Advanced group"
            );
        }
    }

    // The Video "Quality" group (gid 0): a header row that toggles group 0 and
    // the render-feature toggles + SSGI sub-quality dropdowns tagged into group
    // 0, the panel declaring it collapsed.
    #[test]
    fn video_quality_group_holds_render_feature_toggles() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        assert_eq!(
            by_name(&assets, "m_settings_video_grphdr_0")["args"]["content"],
            "+ Quality"
        );
        assert_eq!(
            by_name(&assets, "m_settings_video_grpbtn_0")["args"]["action"],
            "group:toggle:0"
        );
        let panel = by_name(&assets, "m_settings_video_scroll");
        let groups = panel["args"]["groups"].as_array().unwrap();
        let quality = groups
            .iter()
            .find(|g| g["header"] == "m_settings_video_grphdr_0")
            .expect("Quality group present");
        assert_eq!(quality["collapsed"], true);
        assert_eq!(quality["title"], "Quality");
        // Every toggle row is tagged into group 0.
        let rows = panel["args"]["rows"].as_array().unwrap();
        let group_of = |needle: &str| {
            rows.iter()
                .find(|r| {
                    r["elements"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|e| e.as_str().unwrap().contains(needle))
                })
                .map(|r| r["group"].as_i64().unwrap())
        };
        for key in [
            "opt_aa_mode",
            "opt_ssao",
            "opt_ssr",
            "opt_ray_traced_reflections",
            "opt_reflection_blur_resolution",
            "opt_ssgi",
            "opt_ssgi_resolution",
            "opt_ssgi_rays",
            "opt_ssgi_steps",
            "opt_shadow_map_size",
            "opt_shadow_update",
            "opt_shadow_distance",
            "opt_auto_exposure",
            "opt_anisotropy",
            // The per-feature sub-quality sliders share the Quality group.
            "sld_ssao_radius",
            "sld_ssao_intensity",
            "sld_ssr_intensity",
            "sld_ssr_max_distance",
            "sld_ssgi_intensity",
            "sld_ssgi_max_distance",
            "sld_auto_exposure_min_ev",
            "sld_auto_exposure_max_ev",
            "sld_auto_exposure_speed",
        ] {
            assert_eq!(
                group_of(key),
                Some(0),
                "{key} should be in the Quality group"
            );
        }
    }

    // Regression: the gid in a group header's `group:toggle:<gid>` action is
    // used at runtime as an INDEX into `ScrollPanel.groups`, and a row's `group`
    // tag is the same index. So each group's position in the groups vec must
    // equal the gid baked into its header/row references. A mismatch toggled the
    // wrong group (clicking "Quality" flipped "Advanced" and vice versa).
    #[test]
    fn group_toggle_gid_indexes_its_own_group() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        let panel = by_name(&assets, "m_settings_video_scroll");
        let groups = panel["args"]["groups"].as_array().unwrap();
        for (gid, group) in groups.iter().enumerate() {
            // The group at index `gid` owns the `grphdr_<gid>` / `grpbtn_<gid>`
            // header, whose toggle action carries that same gid.
            let header_name = format!("m_settings_video_grphdr_{gid}");
            assert_eq!(group["header"], header_name, "group {gid} out of gid order");
            assert_eq!(
                by_name(&assets, &format!("m_settings_video_grpbtn_{gid}"))["args"]["action"],
                format!("group:toggle:{gid}")
            );
            // The header label's title matches the group at this index.
            let content = by_name(&assets, &header_name)["args"]["content"]
                .as_str()
                .unwrap()
                .to_string();
            let title = group["title"].as_str().unwrap();
            assert!(
                content.ends_with(title),
                "header {content:?} does not match group {title:?} at index {gid}"
            );
        }
    }

    // Every settings body row gets a semi-transparent card background, drawn
    // before (behind) the row's content and listed first in the row's elements
    // so it reflows / clips / hides with the row.
    #[test]
    fn settings_rows_have_card_backgrounds() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        let panel = by_name(&assets, "m_settings_video_scroll");
        let rows = panel["args"]["rows"].as_array().unwrap();
        for row in rows {
            let elems = row["elements"].as_array().unwrap();
            let bg = elems[0].as_str().unwrap();
            assert!(
                bg.contains("_bg_"),
                "row missing a leading bg element: {elems:?}"
            );
            let sprite = by_name(&assets, bg);
            assert_eq!(sprite["type"], "Sprite");
            let alpha = sprite["args"]["tint"][3].as_f64().unwrap();
            assert!(
                alpha > 0.0 && alpha < 1.0,
                "bg card should be semi-transparent"
            );
        }
        // The first row's card is declared before that row's content (the vsync
        // OptionSelect), so the shared sprite/text pass draws it behind the row.
        let names: Vec<String> = assets.iter().map(asset_name).collect();
        let bg_idx = names
            .iter()
            .position(|n| n == "m_settings_video_bg_0")
            .expect("first row card");
        let content_idx = names
            .iter()
            .position(|n| n == "m_settings_video_opt_vsync")
            .expect("vsync row");
        assert!(
            bg_idx < content_idx,
            "card must precede row content for z-order"
        );
    }

    // Hover is color-only by default: each generated HitRegion's hover_scale
    // equals the scale of the label it restyles (the MainMenu hover_scale
    // multiplier defaults to 1.0), so a hovered item changes color without
    // growing or shifting out of its card.
    #[test]
    fn default_menu_hover_is_color_only() {
        let mut assets =
            vec![serde_json::json!({"name":"m","type":"MainMenu","args":{"title":"Paused"}})];
        expand_main_menus(&mut assets).unwrap();

        let label_scale = |name: &str| by_name(&assets, name)["args"]["scale"].as_f64().unwrap();

        // Raw HitRegions (menu items, tabs, group headers, Back): the region's
        // hover_scale matches its label's scale, so hover does not resize it.
        let mut checked = 0;
        for v in &assets {
            if type_norm(v) != "hitregion" {
                continue;
            }
            let args = &v["args"];
            let (Some(label), Some(hs)) = (args["label"].as_str(), args["hover_scale"].as_f64())
            else {
                continue;
            };
            if label.is_empty() {
                continue;
            }
            let ls = label_scale(label);
            assert!(
                (hs - ls).abs() < 1e-6,
                "{}: hover_scale {hs} != label scale {ls}",
                asset_name(v)
            );
            checked += 1;
        }
        assert!(checked > 0, "no labeled hover regions were checked");

        // OptionSelect rows carry an absolute hover_scale equal to their text
        // scale, so the value label also keeps its size on hover.
        for v in &assets {
            if type_norm(v) != "optionselect" {
                continue;
            }
            let ts = v["args"]["text_scale"].as_f64().unwrap();
            let hs = v["args"]["hover_scale"].as_f64().unwrap();
            assert!(
                (hs - ts).abs() < 1e-6,
                "optionselect hover_scale {hs} != text_scale {ts}"
            );
        }
    }

    // A non-default hover_scale still grows the hovered text, as a multiplier on
    // each item's own scale, so the emphasis feature is preserved.
    #[test]
    fn hover_scale_multiplies_label_scale() {
        let mut assets = vec![serde_json::json!({
            "name":"m","type":"MainMenu","args":{"hover_scale":2.0,"toggle_key":""}
        })];
        expand_main_menus(&mut assets).unwrap();
        let label_scale = by_name(&assets, "m_label_0")["args"]["scale"]
            .as_f64()
            .unwrap();
        let region_hs = by_name(&assets, "m_btn_0")["args"]["hover_scale"]
            .as_f64()
            .unwrap();
        assert!(
            (region_hs - label_scale * 2.0).abs() < 1e-6,
            "expected {} got {region_hs}",
            label_scale * 2.0
        );
    }

    // The row content (the OptionSelect) is inset within its card by the same
    // padding on the left and right, so the name does not touch the card edge.
    #[test]
    fn settings_row_content_is_inset_evenly_within_card() {
        let mut assets = vec![serde_json::json!({"name":"m","type":"MainMenu"})];
        expand_main_menus(&mut assets).unwrap();
        let card = by_name(&assets, "m_settings_video_bg_0");
        let opt = by_name(&assets, "m_settings_video_opt_vsync");
        let card_x = card["args"]["x"].as_f64().unwrap();
        let card_w = card["args"]["width"].as_f64().unwrap();
        let opt_x = opt["args"]["x"].as_f64().unwrap();
        let opt_w = opt["args"]["width"].as_f64().unwrap();
        let left_pad = opt_x - card_x;
        let right_pad = (card_x + card_w) - (opt_x + opt_w);
        assert!(left_pad > 0.0, "content should be inset from the left edge");
        assert!(
            (left_pad - right_pad).abs() < 1e-3,
            "left pad {left_pad} != right pad {right_pad}"
        );
    }
}
