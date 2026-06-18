// src/gfx/text.rs
//
// Font atlas data and text draw-call assembly. No backend ownership; the
// renderer uploads the atlas textures; this module only builds the quad
// geometry from TextLabel components each frame.

use crate::assets::{LabelBox, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::gfx::render_types::{TextDrawCall, TextVertex};
use concinnity_core::gfx::overlay::OverlayTransform;

// Per-font data kept in memory after init() so step() can build text quads each frame.
pub(crate) struct LoadedFont {
    // Index into the backend's text atlas texture array.
    pub(crate) atlas_slot: usize,
    // Per-glyph metrics keyed by Unicode code point.
    pub(crate) metrics: std::collections::HashMap<u32, crate::build::font::GlyphMetrics>,
    pub(crate) atlas_w: u32,
    pub(crate) atlas_h: u32,
    // Rasterisation height (px) used to position glyphs vertically.
    pub(crate) size_px: f32,
    // Cap height (logical px): the bearing of an uppercase reference glyph, used
    // to vertically center the visible text within its line box. The full em
    // (`size_px`) is taller than the visible glyphs, so centering on the em alone
    // leaves a gap above the caps; centering the cap band fixes that.
    pub(crate) cap_px: f32,
    // Atlas supersample factor: glyph `atlas_w`/`atlas_h` are stored in atlas
    // texels, which are this many times larger than the glyph's size in logical
    // (layout) pixels. The on-screen quad divides by it so the text lays out at
    // its requested size while the extra texels supersample the glyph.
    pub(crate) supersample: f32,
}

// Cap height (logical px) for vertical centering: the bearing of an uppercase
// reference glyph ('H'), falling back to the tallest uppercase glyph, then to a
// fraction of the em when no metrics are available.
pub(crate) fn derive_cap_px(
    metrics: &std::collections::HashMap<u32, crate::build::font::GlyphMetrics>,
    size_px: f32,
) -> f32 {
    if let Some(h) = metrics.get(&('H' as u32))
        && h.bearing_y > 0.0
    {
        return h.bearing_y;
    }
    let max_upper = ('A'..='Z')
        .filter_map(|c| metrics.get(&(c as u32)))
        .map(|m| m.bearing_y)
        .fold(0.0_f32, f32::max);
    if max_upper > 0.0 {
        return max_upper;
    }
    0.7 * size_px
}

// Sum the advance widths for a string given a font and scale. Used to centre
// text. Newlines carry no advance and are skipped (a multi-line label is
// measured as if the lines were concatenated).
fn measure_text_width(content: &str, font: &LoadedFont, scale: f32) -> f32 {
    content
        .chars()
        .filter(|&ch| ch != '\n')
        .map(|ch| {
            font.metrics
                .get(&(ch as u32))
                .map(|m| m.advance_px * scale)
                .unwrap_or_else(|| {
                    font.metrics
                        .get(&(b' ' as u32))
                        .map(|m| m.advance_px * scale)
                        .unwrap_or(0.0)
                })
        })
        .sum()
}

// Baseline position relative to a label's top-left `y`, so the cap-height band
// is vertically centered within the line box `[y, y + line_height]`. Pinning the
// baseline to the box bottom (the old behaviour) left a large gap above the
// glyphs; centering the cap band makes short UI text sit centered in its box.
fn baseline_offset(font: &LoadedFont, scale: f32) -> f32 {
    let line_height = font.size_px * scale;
    (line_height + font.cap_px * scale) / 2.0
}

// The visible glyphs' vertical extent above and below the first line's baseline,
// in scaled pixels: how far the tallest glyph rises (ascent) and the lowest
// glyph drops (descent). A tight background box and the layout measurement both
// hug this, so the box wraps the ink with `padding` on every side instead of the
// full em line box.
fn content_v_extent(content: &str, font: &LoadedFont, scale: f32) -> (f32, f32) {
    let mut top_above = 0.0_f32;
    let mut bot_below = 0.0_f32;
    for ch in content.chars() {
        if ch == '\n' {
            continue;
        }
        if let Some(m) = font.metrics.get(&(ch as u32)) {
            if m.atlas_h == 0 {
                continue;
            }
            top_above = top_above.max(m.bearing_y * scale);
            bot_below = bot_below.max((m.atlas_h as f32 / font.supersample - m.bearing_y) * scale);
        }
    }
    (top_above, bot_below)
}

// Measure a label's background-box extent for layout: a box hugging the visible
// glyphs grown by the label's padding on every side, plus one line height per
// extra `\n`-split line. Mirrors the background-box math in `build_text_calls`.
// `top_inset` is the gap from the box top down to the text origin (the label's
// `y`), which `LayoutContainer` uses to place the box. Returns `None` for a
// hidden label or one whose font isn't loaded, so a `LayoutContainer` drops it
// and reserves no space.
pub(crate) fn measure_label_box(
    label: &TextLabel,
    loaded_fonts: &std::collections::HashMap<AssetId, LoadedFont>,
) -> Option<LabelBox> {
    if !label.visible {
        return None;
    }
    let font = label.font.and_then(|fid| loaded_fonts.get(&fid))?;
    let scale = label.scale;
    let line_height = font.size_px * scale;
    let lines = label.content.split('\n').count().max(1) as f32;
    let text_w = label
        .content
        .split('\n')
        .map(|line| measure_text_width(line, font, scale))
        .fold(0.0_f32, f32::max);
    let pad = label.padding;
    let (top_above, bot_below) = content_v_extent(&label.content, font, scale);
    let base_off = baseline_offset(font, scale);
    Some(LabelBox {
        w: text_w + 2.0 * pad,
        h: top_above + bot_below + (lines - 1.0) * line_height + 2.0 * pad,
        pad,
        // Box top is `base_off - top_above - pad` below the origin; the inset is
        // the origin's distance below the box top.
        top_inset: top_above + pad - base_off,
    })
}

// Build one TextDrawCall per TextLabel, laying out character quads using the
// loaded font metrics. When `win_w` and `win_h` are both > 0.0, labels with
// `centered = true` are repositioned to the centre of the viewport. `clips`
// maps an element id to a reference-space clip band; a label found there has
// its call scissored to that band (mapped to the window), so a scrollable
// panel's off-band rows do not bleed over its chrome.
pub(crate) fn build_text_calls(
    labels: &[&TextLabel],
    loaded_fonts: &std::collections::HashMap<AssetId, LoadedFont>,
    win_w: f32,
    win_h: f32,
    clips: &std::collections::HashMap<AssetId, [f32; 4]>,
) -> Vec<TextDrawCall> {
    // View-owned labels are overlay UI authored in the reference canvas; map
    // them to the live window so menus scale with the window. HUD labels
    // (view == None) keep literal window pixels.
    let overlay = OverlayTransform::from_viewport([win_w, win_h]);
    let mut calls = Vec::new();
    for label in labels {
        if !label.visible {
            continue;
        }
        let font = match label.font.and_then(|fid| loaded_fonts.get(&fid)) {
            Some(f) => f,
            None => continue,
        };
        let mut vertices: Vec<TextVertex> = Vec::new();
        let mut indices: Vec<u16> = Vec::new();

        // For centered labels, auto-scale to fill ~85% of the viewport while
        // preserving the text's aspect ratio. The label's scale field is used
        // for non-centered labels only.
        let (x0, y0, scale) = if label.centered && win_w > 0.0 && win_h > 0.0 {
            let w1 = measure_text_width(&label.content, font, 1.0);
            let h1 = font.size_px;
            let scale = if w1 > 0.0 && h1 > 0.0 {
                let sw = win_w * 0.85 / w1;
                let sh = win_h * 0.85 / h1;
                sw.min(sh)
            } else {
                label.scale
            };
            let tw = measure_text_width(&label.content, font, scale);
            let th = h1 * scale;
            ((win_w - tw) / 2.0, (win_h - th) / 2.0, scale)
        } else if label.view.is_some() {
            let (sx, sy) = overlay.forward(label.x, label.y);
            (sx, sy, label.scale * overlay.scale())
        } else {
            (label.x, label.y, label.scale)
        };

        let mut x_cursor = x0;
        // baseline: positioned so the cap-height band is centered within the line
        // box, so short UI text sits vertically centered rather than pinned to
        // the box bottom. Advanced by one line height on each newline so
        // multi-line labels lay out down the screen.
        let line_height = font.size_px * scale;
        let mut baseline = y0 + baseline_offset(font, scale);
        let aw = font.atlas_w as f32;
        let ah = font.atlas_h as f32;

        // Background box: a filled quad behind the glyphs, sized to hug the
        // visible glyphs grown by `padding` on every side (not the full em line
        // box, which left a large gap above the caps). Emitted first so the
        // glyphs composite on top. It carries a sentinel UV (a negative u) that
        // the text shader reads as "solid fill", with the box alpha passed
        // through in v. Empty content draws nothing at all (so a blanked label
        // fully disappears).
        if label.background[3] > 0.0 && !label.content.is_empty() {
            let lines = label.content.split('\n').count().max(1) as f32;
            let text_w = label
                .content
                .split('\n')
                .map(|line| measure_text_width(line, font, scale))
                .fold(0.0_f32, f32::max);
            let pad = label.padding;
            let (top_above, bot_below) = content_v_extent(&label.content, font, scale);
            let last_baseline = baseline + (lines - 1.0) * line_height;
            let (x0b, y0b) = (x0 - pad, baseline - top_above - pad);
            let (x1b, y1b) = (x0 + text_w + pad, last_baseline + bot_below + pad);
            let bg = [
                label.background[0],
                label.background[1],
                label.background[2],
            ];
            let ba = label.background[3];
            let box_vtx = |x: f32, y: f32| TextVertex {
                pos: [x, y],
                uv: [-1.0, ba],
                color: bg,
                _pad: 0.0,
            };
            vertices.extend_from_slice(&[
                box_vtx(x0b, y0b),
                box_vtx(x1b, y0b),
                box_vtx(x1b, y1b),
                box_vtx(x0b, y1b),
            ]);
            indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        }

        for ch in label.content.chars() {
            if ch == '\n' {
                x_cursor = x0;
                baseline += line_height;
                continue;
            }
            let m = match font.metrics.get(&(ch as u32)) {
                Some(m) => m,
                None => {
                    if let Some(sp) = font.metrics.get(&(b' ' as u32)) {
                        x_cursor += sp.advance_px * scale;
                    }
                    continue;
                }
            };
            if m.atlas_w == 0 || m.atlas_h == 0 {
                x_cursor += m.advance_px * scale;
                continue;
            }
            // atlas_w/atlas_h are in supersampled atlas texels; divide by the
            // supersample factor to get the glyph's logical size before scaling
            // to the screen. The UVs below still address the full texel extent.
            let gw = m.atlas_w as f32 / font.supersample * scale;
            let gh = m.atlas_h as f32 / font.supersample * scale;
            let gx = x_cursor + m.bearing_x * scale;
            let gy = baseline - m.bearing_y * scale;
            let u0 = m.atlas_x as f32 / aw;
            let v0 = m.atlas_y as f32 / ah;
            let u1 = (m.atlas_x as f32 + m.atlas_w as f32) / aw;
            let v1 = (m.atlas_y as f32 + m.atlas_h as f32) / ah;
            let base = vertices.len() as u16;
            vertices.extend_from_slice(&[
                TextVertex {
                    pos: [gx, gy],
                    uv: [u0, v0],
                    color: label.color,
                    _pad: 0.0,
                },
                TextVertex {
                    pos: [gx + gw, gy],
                    uv: [u1, v0],
                    color: label.color,
                    _pad: 0.0,
                },
                TextVertex {
                    pos: [gx + gw, gy + gh],
                    uv: [u1, v1],
                    color: label.color,
                    _pad: 0.0,
                },
                TextVertex {
                    pos: [gx, gy + gh],
                    uv: [u0, v1],
                    color: label.color,
                    _pad: 0.0,
                },
            ]);
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            x_cursor += m.advance_px * scale;
        }
        if !vertices.is_empty() {
            calls.push(TextDrawCall {
                vertices,
                indices,
                atlas_slot: font.atlas_slot,
                clip_rect: clips
                    .get(&label.asset_id)
                    .map(|b| band_to_window(&overlay, *b)),
            });
        }
    }
    calls
}

// Map a reference-space clip band `[x, y, width, height]` to a window-space
// rectangle through the overlay transform, so the backend can scissor to it.
pub(crate) fn band_to_window(overlay: &OverlayTransform, band: [f32; 4]) -> [f32; 4] {
    let (x0, y0) = overlay.forward(band[0], band[1]);
    let (x1, y1) = overlay.forward(band[0] + band[2], band[1] + band[3]);
    [x0, y0, x1 - x0, y1 - y0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::font::GlyphMetrics;

    // No clip bands: every label draws unclipped.
    fn no_clips() -> std::collections::HashMap<AssetId, [f32; 4]> {
        std::collections::HashMap::new()
    }

    fn make_glyph(atlas_w: u16, atlas_h: u16, advance_px: f32) -> GlyphMetrics {
        GlyphMetrics {
            char_code: 0,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w,
            atlas_h,
            advance_px,
            bearing_x: 0.0,
            bearing_y: atlas_h as f32,
        }
    }

    fn make_font(chars: &[(char, GlyphMetrics)]) -> LoadedFont {
        let metrics: std::collections::HashMap<u32, GlyphMetrics> =
            chars.iter().map(|(c, m)| (*c as u32, *m)).collect();
        let cap_px = derive_cap_px(&metrics, 16.0);
        LoadedFont {
            atlas_slot: 0,
            cap_px,
            metrics,
            atlas_w: 128,
            atlas_h: 128,
            size_px: 16.0,
            // 1x: the unit tests express glyph sizes directly in atlas texels.
            supersample: 1.0,
        }
    }

    fn make_label(font: AssetId, content: &str, x: f32) -> TextLabel {
        TextLabel {
            asset_id: AssetId::default(),
            font: Some(font),
            content: content.to_string(),
            x,
            y: 0.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            view: None,
        }
    }

    #[test]
    fn empty_labels_returns_empty_calls() {
        let fonts = std::collections::HashMap::new();
        assert!(build_text_calls(&[], &fonts, 0.0, 0.0, &no_clips()).is_empty());
    }

    #[test]
    fn unknown_font_produces_no_call() {
        let fonts = std::collections::HashMap::new();
        let label = make_label(AssetId(99), "hello", 0.0);
        assert!(build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips()).is_empty());
    }

    #[test]
    fn single_glyph_produces_quad() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        let label = make_label(AssetId(0), "A", 0.0);
        let calls = build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4);
        assert_eq!(calls[0].indices.len(), 6);
        assert_eq!(calls[0].atlas_slot, 0);
    }

    #[test]
    fn background_prepends_a_box_quad() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        let mut label = make_label(AssetId(0), "A", 0.0);
        label.background = [0.0, 0.3, 0.1, 0.85];
        label.padding = 4.0;
        let calls = build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips());
        assert_eq!(calls.len(), 1);
        // 4 box verts prepended + 4 glyph verts; 6 box indices + 6 glyph.
        assert_eq!(calls[0].vertices.len(), 8);
        assert_eq!(calls[0].indices.len(), 12);
        // The box quad comes first: sentinel u (< 0), box alpha carried in v.
        for v in &calls[0].vertices[..4] {
            assert!(v.uv[0] < 0.0, "box vert should carry the sentinel u");
            assert!((v.uv[1] - 0.85).abs() < 1e-4, "box alpha travels in v");
        }
        // Glyph verts keep real, non-negative atlas UVs.
        assert!(calls[0].vertices[4].uv[0] >= 0.0);
    }

    #[test]
    fn derive_cap_px_uses_uppercase_reference() {
        // 'H' is the cap-height reference, even when a lowercase glyph is taller.
        let mut m = std::collections::HashMap::new();
        m.insert('H' as u32, make_glyph(8, 10, 9.0)); // bearing_y = 10
        m.insert('g' as u32, make_glyph(8, 14, 9.0)); // taller, but lowercase
        assert!((derive_cap_px(&m, 16.0) - 10.0).abs() < 1e-4);
        // With no glyphs, fall back to a fraction of the em.
        let empty = std::collections::HashMap::new();
        assert!((derive_cap_px(&empty, 20.0) - 14.0).abs() < 1e-4);
    }

    #[test]
    fn background_box_hugs_glyph_with_symmetric_padding() {
        // The box wraps the visible glyph with `padding` above and below (instead
        // of the full em line box, which left a large gap above the caps).
        let g = make_glyph(10, 12, 11.0); // bearing_y = 12, no descent
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        let mut label = make_label(AssetId(0), "A", 0.0);
        label.background = [0.1, 0.1, 0.1, 1.0];
        label.padding = 4.0;
        let calls = build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips());
        let v = &calls[0].vertices;
        // Verts 0..4 are the box; 4..8 the glyph quad.
        let (box_top, box_bot) = (v[0].pos[1], v[2].pos[1]);
        let (glyph_top, glyph_bot) = (v[4].pos[1], v[6].pos[1]);
        assert!(
            (glyph_top - box_top - 4.0).abs() < 1e-4,
            "top pad = {}",
            glyph_top - box_top
        );
        assert!(
            (box_bot - glyph_bot - 4.0).abs() < 1e-4,
            "bottom pad = {}",
            box_bot - glyph_bot
        );
    }

    #[test]
    fn background_with_empty_content_draws_nothing() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        let mut label = make_label(AssetId(0), "", 0.0);
        label.background = [0.0, 0.3, 0.1, 0.85];
        // A blanked label (e.g. a toggled-off HUD chip) draws no box.
        assert!(build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips()).is_empty());
    }

    #[test]
    fn space_advances_cursor_without_quad() {
        let space = make_glyph(0, 0, 8.0);
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[(' ', space), ('A', g)]));
        // Two spaces then 'A': only 'A' produces geometry.
        let label = make_label(AssetId(0), "  A", 0.0);
        let calls = build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4);
        // 'A' quad starts after 2 × advance_px(space) = 16.0
        let gx = calls[0].vertices[0].pos[0];
        assert!((gx - 16.0).abs() < 1e-4, "expected gx=16.0, got {gx}");
    }

    #[test]
    fn zero_size_glyph_advances_cursor_without_quad() {
        // A glyph whose atlas dimensions are 0×0 is invisible but still advances x.
        let zero = GlyphMetrics {
            char_code: b'X' as u32,
            atlas_x: 0,
            atlas_y: 0,
            atlas_w: 0,
            atlas_h: 0,
            advance_px: 5.0,
            bearing_x: 0.0,
            bearing_y: 0.0,
        };
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('X', zero), ('A', g)]));
        let label = make_label(AssetId(0), "XA", 0.0);
        let calls = build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4); // only 'A'
        // 'A' starts at x = advance_px('X') = 5.0
        assert!((calls[0].vertices[0].pos[0] - 5.0).abs() < 1e-4);
    }

    #[test]
    fn newline_starts_a_new_line() {
        // "A\nA": the second glyph resets x to the label origin and drops
        // down by one line height (font size_px * scale = 16).
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        let label = make_label(AssetId(0), "A\nA", 0.0);
        let calls = build_text_calls(&[&label], &fonts, 0.0, 0.0, &no_clips());
        assert_eq!(calls.len(), 1);
        // Two glyphs -> two quads -> 8 vertices, 12 indices.
        assert_eq!(calls[0].vertices.len(), 8);
        assert_eq!(calls[0].indices.len(), 12);
        let first = &calls[0].vertices[0];
        let second = &calls[0].vertices[4];
        // x resets to the label origin on the new line.
        assert!((first.pos[0] - second.pos[0]).abs() < 1e-4);
        // y drops by exactly one line height.
        assert!(
            (second.pos[1] - first.pos[1] - 16.0).abs() < 1e-4,
            "expected +16 line height, got {}",
            second.pos[1] - first.pos[1]
        );
    }

    #[test]
    fn centered_label_is_repositioned() {
        let g = make_glyph(10, 12, 20.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        let mut label = make_label(AssetId(0), "A", 0.0);
        label.centered = true;
        // Viewport 200×100; glyph advance=20, size_px=16, cap_px=12 ('A' bearing).
        // Auto-scale: sw = 200*0.85/20 = 8.5, sh = 100*0.85/16 = 5.3125 -> scale = 5.3125
        // tw = 20*5.3125 = 106.25, th = 16*5.3125 = 85.0
        // x0 = (200 - 106.25) / 2 = 46.875, y0 = (100 - 85.0) / 2 = 7.5
        // line_height = 16*5.3125 = 85; baseline centers the cap band:
        // baseline = 7.5 + (85 + 12*5.3125)/2 = 7.5 + 74.375 = 81.875
        // gx = x0 + bearing_x*scale = 46.875, gy = baseline - bearing_y*scale = 81.875 - 63.75 = 18.125
        let calls = build_text_calls(&[&label], &fonts, 200.0, 100.0, &no_clips());
        assert_eq!(calls.len(), 1);
        let v = &calls[0].vertices[0];
        assert!((v.pos[0] - 46.875).abs() < 1e-3, "gx={}", v.pos[0]);
        assert!((v.pos[1] - 18.125).abs() < 1e-3, "gy={}", v.pos[1]);
    }

    #[test]
    fn view_owned_label_scales_and_repositions_with_overlay() {
        // A view-owned (overlay) label is authored in the reference canvas and
        // mapped onto the window. At a 2x viewport its origin moves to the
        // forward-mapped position and its scale doubles. A HUD label (view ==
        // None) at the same coordinates stays put.
        let g = make_glyph(10, 12, 20.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));

        let hud = make_label(AssetId(0), "A", 100.0); // view == None
        let mut overlay_label = make_label(AssetId(0), "A", 100.0);
        overlay_label.y = 100.0;
        overlay_label.view = Some(AssetId(5));

        // 2x reference viewport (1280x720 -> 2560x1440): scale 2, centered.
        let vp = (2560.0, 1440.0);
        let hud_calls = build_text_calls(&[&hud], &fonts, vp.0, vp.1, &no_clips());
        let ovl_calls = build_text_calls(&[&overlay_label], &fonts, vp.0, vp.1, &no_clips());
        // HUD label keeps its literal origin (x = 100).
        assert!((hud_calls[0].vertices[0].pos[0] - 100.0).abs() < 1e-3);
        // Overlay label: forward(100,100) at scale 2 -> x = 1280 + (100-640)*2 = 200.
        assert!(
            (ovl_calls[0].vertices[0].pos[0] - 200.0).abs() < 1e-3,
            "x={}",
            ovl_calls[0].vertices[0].pos[0]
        );
        // Glyph width doubles (atlas_w 10 -> 20 on screen).
        let w = ovl_calls[0].vertices[1].pos[0] - ovl_calls[0].vertices[0].pos[0];
        assert!((w - 20.0).abs() < 1e-3, "w={w}");
    }

    #[test]
    fn measure_label_box_grows_text_by_padding() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g), ('B', g)]));
        let mut label = make_label(AssetId(0), "AB", 0.0);
        label.padding = 4.0;
        let b = measure_label_box(&label, &fonts).unwrap();
        // text width = 2 * advance(11) = 22, grown by padding on both sides.
        assert!((b.w - 30.0).abs() < 1e-4, "w={}", b.w);
        // The box hugs the glyphs: ascent(bearing_y=12) + descent(0) + 2*pad(4) = 20.
        assert!((b.h - 20.0).abs() < 1e-4, "h={}", b.h);
        assert!((b.pad - 4.0).abs() < 1e-4);
        // top_inset = ascent(12) + pad(4) - baseline_offset((16+12)/2=14) = 2.
        assert!(
            (b.top_inset - 2.0).abs() < 1e-4,
            "top_inset={}",
            b.top_inset
        );
    }

    #[test]
    fn measure_label_box_skips_hidden_and_unloaded() {
        let g = make_glyph(10, 12, 11.0);
        let mut fonts = std::collections::HashMap::new();
        fonts.insert(AssetId(0), make_font(&[('A', g)]));
        // Hidden label → None even with a loaded font.
        let mut hidden = make_label(AssetId(0), "A", 0.0);
        hidden.visible = false;
        assert!(measure_label_box(&hidden, &fonts).is_none());
        // Visible label whose font isn't loaded → None.
        let orphan = make_label(AssetId(99), "A", 0.0);
        assert!(measure_label_box(&orphan, &fonts).is_none());
    }
}
