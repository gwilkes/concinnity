// src/gfx/overlay.rs
//
// Screen-overlay scaling math shared by the build pipeline (which lays menus
// out against a fixed reference canvas) and the client renderer (which scales
// that canvas to the live window). View-owned UI (menus, settings) is authored
// in a fixed reference resolution; at runtime the whole overlay is uniformly
// scaled to fit the window, preserving aspect and staying centered, so a menu
// looks the same proportion of the screen at any window size.

// Reference resolution menus are authored against. Window-pixel coordinates of
// view-owned UI are interpreted in this space and scaled to the live window.
pub const UI_REFERENCE_SIZE: [f32; 2] = [1280.0, 720.0];

// A uniform similarity transform mapping the reference canvas to the live
// window: a single scale plus a recentering. Built from the live viewport; an
// invalid (zero) viewport yields the identity (overlay drawn at reference
// pixels), which is what unit tests and the pre-backend init frames see.
#[derive(Debug, Clone, Copy)]
pub struct OverlayTransform {
    scale: f32,
    // Window-space center the reference center maps to.
    screen_cx: f32,
    screen_cy: f32,
    // Reference-space center (half the reference size).
    ref_cx: f32,
    ref_cy: f32,
}

impl OverlayTransform {
    // Build the transform for a live logical viewport `[width, height]`. A
    // degenerate viewport gives the identity transform.
    pub fn from_viewport(viewport: [f32; 2]) -> Self {
        let [rw, rh] = UI_REFERENCE_SIZE;
        let ref_cx = rw / 2.0;
        let ref_cy = rh / 2.0;
        let [vw, vh] = viewport;
        if vw <= 0.0 || vh <= 0.0 {
            return Self {
                scale: 1.0,
                screen_cx: ref_cx,
                screen_cy: ref_cy,
                ref_cx,
                ref_cy,
            };
        }
        // Uniform "fit": the smaller axis ratio, so the reference canvas always
        // fits inside the window without distorting text.
        let scale = (vw / rw).min(vh / rh);
        Self {
            scale,
            screen_cx: vw / 2.0,
            screen_cy: vh / 2.0,
            ref_cx,
            ref_cy,
        }
    }

    // The uniform scale factor applied to sizes (glyph scale, sprite extent).
    pub fn scale(&self) -> f32 {
        self.scale
    }

    // Map a reference-space point to window space.
    pub fn forward(&self, x: f32, y: f32) -> (f32, f32) {
        (
            self.screen_cx + (x - self.ref_cx) * self.scale,
            self.screen_cy + (y - self.ref_cy) * self.scale,
        )
    }

    // Map a window-space point back to reference space (the inverse of
    // `forward`). Used to hit-test the live cursor against reference-space UI
    // rects.
    pub fn inverse(&self, x: f32, y: f32) -> (f32, f32) {
        let s = if self.scale != 0.0 { self.scale } else { 1.0 };
        (
            self.ref_cx + (x - self.screen_cx) / s,
            self.ref_cy + (y - self.screen_cy) / s,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_viewport_is_identity() {
        let t = OverlayTransform::from_viewport([0.0, 0.0]);
        assert_eq!(t.scale(), 1.0);
        // A point maps to itself.
        let (x, y) = t.forward(100.0, 200.0);
        assert!((x - 100.0).abs() < 1e-4 && (y - 200.0).abs() < 1e-4);
    }

    #[test]
    fn exact_reference_size_is_unit_scale_and_centered() {
        let t = OverlayTransform::from_viewport(UI_REFERENCE_SIZE);
        assert!((t.scale() - 1.0).abs() < 1e-4);
        // The reference center maps to the window center.
        let (cx, cy) = t.forward(UI_REFERENCE_SIZE[0] / 2.0, UI_REFERENCE_SIZE[1] / 2.0);
        assert!((cx - UI_REFERENCE_SIZE[0] / 2.0).abs() < 1e-4);
        assert!((cy - UI_REFERENCE_SIZE[1] / 2.0).abs() < 1e-4);
    }

    #[test]
    fn doubling_both_axes_doubles_scale() {
        let [rw, rh] = UI_REFERENCE_SIZE;
        let t = OverlayTransform::from_viewport([rw * 2.0, rh * 2.0]);
        assert!((t.scale() - 2.0).abs() < 1e-4);
        // The reference origin maps such that the canvas stays centered: the
        // reference center sits at the window center.
        let (cx, cy) = t.forward(rw / 2.0, rh / 2.0);
        assert!((cx - rw).abs() < 1e-4 && (cy - rh).abs() < 1e-4);
    }

    #[test]
    fn wider_window_fits_to_height_and_letterboxes_width() {
        let [rw, rh] = UI_REFERENCE_SIZE;
        // Twice as wide, same height: the limiting axis is height (ratio 1.0).
        let t = OverlayTransform::from_viewport([rw * 2.0, rh]);
        assert!((t.scale() - 1.0).abs() < 1e-4);
        // The canvas stays centered horizontally: reference left edge (x=0)
        // lands at half a reference width in from the window's left.
        let (x0, _) = t.forward(0.0, 0.0);
        assert!((x0 - rw / 2.0).abs() < 1e-4, "x0={x0}");
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        let t = OverlayTransform::from_viewport([2560.0, 1440.0]);
        let (sx, sy) = t.forward(300.0, 410.0);
        let (rx, ry) = t.inverse(sx, sy);
        assert!((rx - 300.0).abs() < 1e-3, "rx={rx}");
        assert!((ry - 410.0).abs() < 1e-3, "ry={ry}");
    }
}
