// src/gfx/cursor.rs
//
// In-engine mouse cursor geometry. A `follow_cursor` Sprite is drawn as a
// classic arrow pointer rather than a plain quad: a filled polygon with a
// contrasting outline so it stays legible over any scene. Like the rest of the
// UI overlay, it rides the text pass's sentinel-UV solid-fill path (u < 0), so
// it needs no new pipeline and renders on every backend. The arrow's diagonal
// edges are real geometry, not a stair-stepped stack of quads.

use crate::assets::Sprite;
use crate::gfx::render_types::{TextDrawCall, TextVertex};
use concinnity_core::gfx::overlay::OverlayTransform;

// Arrow silhouette in a normalised space: tip (the hotspot) at the origin,
// pointing down-right, height 1.0 and width ~0.62. Vertices run clockwise
// around the boundary in screen space (y grows downward):
//   V0 tip, V1 left-edge foot, V2 inner notch, V3 tail tip,
//   V4 tail heel, V5 barb root, V6 right barb.
const ARROW: [(f32, f32); 7] = [
    (0.00, 0.00),
    (0.00, 0.86),
    (0.21, 0.65),
    (0.35, 1.00),
    (0.50, 0.93),
    (0.35, 0.59),
    (0.62, 0.59),
];

// Triangulation of the arrow: a fan over the head (tip to the two barbs) plus
// the small tail quad. Indices reference ARROW.
const ARROW_TRIS: [[u16; 3]; 5] = [[0, 1, 2], [0, 2, 5], [0, 5, 6], [2, 3, 4], [2, 4, 5]];

// Eight unit directions used to stamp the outline around the fill, giving an
// even border ring of one outline-width radius.
const OUTLINE_OFFSETS: [(f32, f32); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (0.707, 0.707),
    (-0.707, 0.707),
    (0.707, -0.707),
    (-0.707, -0.707),
];

// Outline width as a fraction of the cursor height, floored at one pixel.
const OUTLINE_RATIO: f32 = 0.085;
// Arrow height in pixels when a cursor sprite leaves its size unset.
const DEFAULT_CURSOR_PX: f32 = 22.0;

// Build the cursor draw calls (one mesh per visible `follow_cursor` sprite) at
// the pointer. Each sprite's tint is the fill colour and its `height` the arrow
// height; `width` is ignored so the arrow keeps its aspect ratio. The arrow
// height is authored in the reference canvas, so it is scaled by the overlay
// factor for `viewport` to stay proportional with the menu it belongs to; the
// pointer stays at the live cursor position. Returns empty when no font atlas
// is loaded (the text pipeline is inactive then).
pub(crate) fn build_cursor_calls(
    cursors: &[&Sprite],
    pointer: (f32, f32),
    default_atlas_slot: Option<usize>,
    viewport: [f32; 2],
) -> Vec<TextDrawCall> {
    let atlas_slot = match default_atlas_slot {
        Some(s) => s,
        None => return Vec::new(),
    };
    let overlay_scale = OverlayTransform::from_viewport(viewport).scale();
    let mut calls = Vec::new();
    for s in cursors {
        if !s.visible {
            continue;
        }
        let alpha = s.tint[3];
        if alpha <= 0.0 {
            continue;
        }
        let size = if s.height > 0.0 {
            s.height
        } else {
            DEFAULT_CURSOR_PX
        } * overlay_scale;
        let fill = [s.tint[0], s.tint[1], s.tint[2]];
        let outline = outline_color(fill);
        let outline_w = (size * OUTLINE_RATIO).max(1.0);

        let mut call = TextDrawCall {
            vertices: Vec::new(),
            indices: Vec::new(),
            atlas_slot,
            // The cursor is never clipped: it draws on top of everything.
            clip_rect: None,
        };
        // Outline first so the fill, appended after, composites on top of it
        // (the overlay draws indexed triangles in order, with no depth test).
        for (dx, dy) in OUTLINE_OFFSETS {
            let o = (pointer.0 + dx * outline_w, pointer.1 + dy * outline_w);
            push_arrow(&mut call, o, size, outline, alpha);
        }
        push_arrow(&mut call, pointer, size, fill, alpha);
        calls.push(call);
    }
    calls
}

// Append one arrow (tip at `origin`, scaled by `size`) to a draw call.
fn push_arrow(call: &mut TextDrawCall, origin: (f32, f32), size: f32, color: [f32; 3], alpha: f32) {
    let base = call.vertices.len() as u16;
    for (nx, ny) in ARROW {
        call.vertices.push(TextVertex {
            pos: [origin.0 + nx * size, origin.1 + ny * size],
            // sentinel u < 0 -> solid-fill path; v carries alpha
            uv: [-1.0, alpha],
            color,
            _pad: 0.0,
        });
    }
    for tri in ARROW_TRIS {
        call.indices.push(base + tri[0]);
        call.indices.push(base + tri[1]);
        call.indices.push(base + tri[2]);
    }
}

// Pick an outline that contrasts the fill: a near-black border under a light
// cursor, a near-white border under a dark one. Keeps any tint legible.
fn outline_color(fill: [f32; 3]) -> [f32; 3] {
    let luma = 0.299 * fill[0] + 0.587 * fill[1] + 0.114 * fill[2];
    if luma > 0.5 {
        [0.05, 0.05, 0.06]
    } else {
        [0.95, 0.95, 0.96]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;

    fn cursor(tint: [f32; 4], height: f32) -> Sprite {
        Sprite {
            asset_id: AssetId::default(),
            x: 0.0,
            y: 0.0,
            width: height,
            height,
            texture: None,
            tint,
            follow_cursor: true,
            visible: true,
            view: None,
        }
    }

    #[test]
    fn no_fonts_means_no_calls() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        assert!(build_cursor_calls(&[&c], (10.0, 10.0), None, [0.0, 0.0]).is_empty());
    }

    #[test]
    fn builds_outline_then_fill_mesh() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        let calls = build_cursor_calls(&[&c], (100.0, 50.0), Some(0), [0.0, 0.0]);
        assert_eq!(calls.len(), 1);
        // Eight outline stamps plus one fill, seven vertices each.
        assert_eq!(calls[0].vertices.len(), 9 * ARROW.len());
        assert_eq!(calls[0].indices.len(), 9 * ARROW_TRIS.len() * 3);
        // The tip of the fill arrow (last stamp's first vertex) sits exactly on
        // the pointer; the outline stamps are offset off it.
        let tip = calls[0].vertices[8 * ARROW.len()];
        assert_eq!(tip.pos, [100.0, 50.0]);
        // Fill keeps the sprite tint; outline does not.
        assert_eq!(tip.color, [1.0, 1.0, 1.0]);
        assert_ne!(calls[0].vertices[0].color, [1.0, 1.0, 1.0]);
        // Every vertex uses the solid-fill sentinel and carries the alpha.
        for v in &calls[0].vertices {
            assert!(v.uv[0] < 0.0);
            assert!((v.uv[1] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn invisible_or_transparent_cursor_is_skipped() {
        let mut hidden = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        hidden.visible = false;
        assert!(build_cursor_calls(&[&hidden], (0.0, 0.0), Some(0), [0.0, 0.0]).is_empty());
        let clear = cursor([1.0, 1.0, 1.0, 0.0], 22.0);
        assert!(build_cursor_calls(&[&clear], (0.0, 0.0), Some(0), [0.0, 0.0]).is_empty());
    }

    #[test]
    fn outline_contrasts_the_fill() {
        // Light fill -> dark outline, dark fill -> light outline.
        assert!(outline_color([1.0, 1.0, 1.0])[0] < 0.5);
        assert!(outline_color([0.0, 0.0, 0.0])[0] > 0.5);
    }

    #[test]
    fn unset_height_falls_back_to_default_size() {
        let c = cursor([1.0, 1.0, 1.0, 1.0], 0.0);
        let calls = build_cursor_calls(&[&c], (0.0, 0.0), Some(0), [0.0, 0.0]);
        // The lowest vertex (tail tip, ny = 1.0) reaches the default height.
        let max_y = calls[0]
            .vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MIN, f32::max);
        assert!((max_y - DEFAULT_CURSOR_PX).abs() < OUTLINE_PX_TOLERANCE);
    }

    #[test]
    fn arrow_scales_with_the_overlay() {
        // At twice the reference size the overlay scale is 2.0, so the arrow
        // height doubles while the tip stays on the pointer. Measure the fill
        // arrow (the last stamp) so the outline ring's extra width is excluded.
        let c = cursor([1.0, 1.0, 1.0, 1.0], 22.0);
        let calls = build_cursor_calls(&[&c], (0.0, 0.0), Some(0), [2560.0, 1440.0]);
        let fill = &calls[0].vertices[8 * ARROW.len()..];
        let max_y = fill.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
        // Tail tip (ny = 1.0) at pointer y = 0 reaches the doubled height.
        assert!((max_y - 44.0).abs() < 1e-3, "max_y={max_y}");
    }

    // The outline stamp pushes the tail a fraction of a pixel past the fill
    // height, so allow a small tolerance in the size check.
    const OUTLINE_PX_TOLERANCE: f32 = 2.0;
}
