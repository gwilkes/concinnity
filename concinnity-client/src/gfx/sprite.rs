// src/gfx/sprite.rs
//
// Sprite quad assembly. Piggybacks on the text render pass: each Sprite is
// emitted as a TextDrawCall containing a single quad with the sentinel UV
// (u < 0) the text shader interprets as a solid-coloured fill, with the alpha
// carried in v. This lets us draw screen-space rectangles without adding a new
// Metal pipeline. The `texture` field on Sprite is reserved for a future
// extension and is ignored here.

use crate::assets::Sprite;
use crate::gfx::render_types::{TextDrawCall, TextVertex};
use concinnity_core::gfx::overlay::{OverlayTransform, UI_REFERENCE_SIZE};

// A view-owned sprite that spans the whole reference canvas is a full-screen
// backdrop (e.g. a menu dim): it is stretched to fill the live window rather
// than uniform-scaled, and an opaque one hides the scene behind it.
pub(crate) fn covers_canvas(s: &Sprite) -> bool {
    let [ref_w, ref_h] = UI_REFERENCE_SIZE;
    s.view.is_some()
        && s.x <= 0.0
        && s.y <= 0.0
        && s.x + s.width >= ref_w
        && s.y + s.height >= ref_h
}

// Build a TextDrawCall per visible Sprite. `default_atlas_slot` is the atlas
// the call binds (the shader does not sample for sentinel-UV verts, but the
// backend still expects a valid slot). Pass the slot of any loaded font;
// returns an empty list when there are no fonts (the text pipeline isn't
// initialised in that case). `viewport` is the live logical window size:
// view-owned sprites are overlay UI authored in the reference canvas and are
// mapped onto the window so menus scale with it; HUD / scene sprites
// (view == None) keep literal window pixels.
pub(crate) fn build_sprite_calls(
    sprites: &[&Sprite],
    default_atlas_slot: Option<usize>,
    viewport: [f32; 2],
    clips: &std::collections::HashMap<crate::ecs::asset_id::AssetId, [f32; 4]>,
) -> Vec<TextDrawCall> {
    let atlas_slot = match default_atlas_slot {
        Some(s) => s,
        None => return Vec::new(),
    };
    let overlay = OverlayTransform::from_viewport(viewport);
    let [vw, vh] = viewport;
    let mut calls = Vec::new();
    for s in sprites {
        if !s.visible {
            continue;
        }
        let [r, g, b, a] = s.tint;
        if a <= 0.0 {
            continue;
        }
        let (x0, y0, x1, y1) = if s.view.is_some() {
            // A view-owned sprite spanning the whole reference canvas is a
            // full-screen backdrop (e.g. a menu dim): always fill the live
            // window instead of uniform-scaling, which would letterbox it.
            if covers_canvas(s) && vw > 0.0 && vh > 0.0 {
                (0.0, 0.0, vw, vh)
            } else {
                let (ax, ay) = overlay.forward(s.x, s.y);
                let (bx, by) = overlay.forward(s.x + s.width, s.y + s.height);
                (ax, ay, bx, by)
            }
        } else {
            (s.x, s.y, s.x + s.width, s.y + s.height)
        };
        let v = |x: f32, y: f32| TextVertex {
            pos: [x, y],
            // sentinel u < 0 ⇒ solid-fill path; v carries alpha
            uv: [-1.0, a],
            color: [r, g, b],
            _pad: 0.0,
        };
        let vertices = vec![v(x0, y0), v(x1, y0), v(x1, y1), v(x0, y1)];
        let indices = vec![0, 1, 2, 0, 2, 3];
        calls.push(TextDrawCall {
            vertices,
            indices,
            atlas_slot,
            clip_rect: clips
                .get(&s.asset_id)
                .map(|b| crate::gfx::text::band_to_window(&overlay, *b)),
        });
    }
    calls
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::AssetId;

    fn no_clips() -> std::collections::HashMap<AssetId, [f32; 4]> {
        std::collections::HashMap::new()
    }

    fn sprite(x: f32, y: f32, w: f32, h: f32, tint: [f32; 4]) -> Sprite {
        Sprite {
            asset_id: AssetId::default(),
            x,
            y,
            width: w,
            height: h,
            texture: None,
            tint,
            follow_cursor: false,
            visible: true,
            view: None,
        }
    }

    #[test]
    fn no_fonts_means_no_calls() {
        let s = sprite(0.0, 0.0, 100.0, 100.0, [1.0, 0.0, 0.0, 1.0]);
        assert!(build_sprite_calls(&[&s], None, [0.0, 0.0], &no_clips()).is_empty());
    }

    #[test]
    fn visible_sprite_emits_quad_with_sentinel_uv() {
        let s = sprite(10.0, 20.0, 100.0, 50.0, [0.5, 0.5, 0.5, 0.75]);
        let calls = build_sprite_calls(&[&s], Some(0), [0.0, 0.0], &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices.len(), 4);
        assert_eq!(calls[0].indices, vec![0, 1, 2, 0, 2, 3]);
        for v in &calls[0].vertices {
            assert!(v.uv[0] < 0.0, "sentinel u should be negative");
            assert!((v.uv[1] - 0.75).abs() < 1e-5, "alpha carried in v");
            assert_eq!(v.color, [0.5, 0.5, 0.5]);
        }
        assert_eq!(calls[0].vertices[0].pos, [10.0, 20.0]);
        assert_eq!(calls[0].vertices[2].pos, [110.0, 70.0]);
    }

    #[test]
    fn invisible_sprite_is_skipped() {
        let mut s = sprite(0.0, 0.0, 100.0, 100.0, [1.0, 1.0, 1.0, 1.0]);
        s.visible = false;
        assert!(build_sprite_calls(&[&s], Some(0), [0.0, 0.0], &no_clips()).is_empty());
    }

    #[test]
    fn zero_alpha_sprite_is_skipped() {
        let s = sprite(0.0, 0.0, 100.0, 100.0, [1.0, 1.0, 1.0, 0.0]);
        assert!(build_sprite_calls(&[&s], Some(0), [0.0, 0.0], &no_clips()).is_empty());
    }

    #[test]
    fn view_owned_sprite_scales_to_window() {
        // A view-owned (overlay) sprite is authored in the reference canvas and
        // uniformly scaled onto the window. At twice the reference size the
        // rect doubles and stays centered.
        let mut s = sprite(100.0, 100.0, 200.0, 100.0, [1.0, 1.0, 1.0, 1.0]);
        s.view = Some(AssetId(7));
        let calls = build_sprite_calls(&[&s], Some(0), [2560.0, 1440.0], &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices[0].pos, [200.0, 200.0]);
        assert_eq!(calls[0].vertices[2].pos, [600.0, 400.0]);
    }

    #[test]
    fn view_owned_full_canvas_backdrop_fills_window() {
        // A view-owned sprite spanning the whole reference canvas is a
        // full-screen backdrop: it fills the live window rather than letterboxing.
        let mut s = sprite(0.0, 0.0, 1280.0, 720.0, [0.0, 0.0, 0.0, 0.5]);
        s.view = Some(AssetId(7));
        let calls = build_sprite_calls(&[&s], Some(0), [2560.0, 1440.0], &no_clips());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].vertices[0].pos, [0.0, 0.0]);
        assert_eq!(calls[0].vertices[2].pos, [2560.0, 1440.0]);
    }

    #[test]
    fn view_less_sprite_keeps_literal_pixels() {
        // A HUD / scene sprite (view == None) is never overlay-scaled.
        let s = sprite(10.0, 20.0, 100.0, 50.0, [0.5, 0.5, 0.5, 1.0]);
        let calls = build_sprite_calls(&[&s], Some(0), [2560.0, 1440.0], &no_clips());
        assert_eq!(calls[0].vertices[0].pos, [10.0, 20.0]);
        assert_eq!(calls[0].vertices[2].pos, [110.0, 70.0]);
    }

    #[test]
    fn clipped_element_carries_window_space_clip_rect() {
        // A view-owned sprite whose id is in the clips map gets a clip_rect
        // mapped from the reference-space band through the overlay; one not in
        // the map stays unclipped.
        let mut s = sprite(100.0, 100.0, 50.0, 50.0, [1.0, 1.0, 1.0, 1.0]);
        s.asset_id = AssetId(7);
        s.view = Some(AssetId(1));
        let mut clips = no_clips();
        // Reference band [200,200] size [200,60] at a 2x viewport (1280x720 ->
        // 2560x1440, scale 2 about the centre): forward(200,200)=(400,400),
        // forward(400,260)=(800,520) -> clip [400,400,400,120].
        clips.insert(AssetId(7), [200.0, 200.0, 200.0, 60.0]);
        let calls = build_sprite_calls(&[&s], Some(0), [2560.0, 1440.0], &clips);
        let clip = calls[0].clip_rect.expect("clipped sprite has a clip rect");
        assert!((clip[0] - 400.0).abs() < 1e-3, "x={}", clip[0]);
        assert!((clip[1] - 400.0).abs() < 1e-3, "y={}", clip[1]);
        assert!((clip[2] - 400.0).abs() < 1e-3, "w={}", clip[2]);
        assert!((clip[3] - 120.0).abs() < 1e-3, "h={}", clip[3]);

        // A sprite not in the clips map is unclipped.
        let mut other = sprite(0.0, 0.0, 10.0, 10.0, [1.0, 1.0, 1.0, 1.0]);
        other.asset_id = AssetId(9);
        other.view = Some(AssetId(1));
        let calls = build_sprite_calls(&[&other], Some(0), [2560.0, 1440.0], &clips);
        assert!(calls[0].clip_rect.is_none());
    }
}
