// src/gfx/input.rs
//
// Backend-agnostic input snapshot returned by RenderBackend::take_input.
// Each backend was previously carrying its own structurally-identical
// InputState; this single type replaces those duplicates.

// Accumulated input state since the last poll. Drained and reset every
// frame by GraphicsSystem and converted into a FrameInput component for
// Camera3DSystem to consume.
#[derive(Default, Debug, Clone, Copy)]
pub struct RenderInput {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub sprint: bool,
    // True for exactly one frame per interact-key press.
    pub interact: bool,
    // True for exactly one frame per jump-key press.
    pub jump: bool,
    // Accumulated mouse delta since the last take_input() call.
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    // Accumulated vertical scroll-wheel delta since the last take_input().
    // Only delivered while the cursor is free.
    pub scroll_delta: f32,
    // Absolute cursor position in window pixels (origin top-left).
    // Only meaningful when the cursor is not captured.
    pub mouse_x: f32,
    pub mouse_y: f32,
    // True for exactly one frame when the left mouse button is pressed
    // while the cursor is not captured.
    pub left_click: bool,
    // True while the left mouse button is held (cursor not captured). Persists
    // across frames until release so a UI drag can track the cursor.
    pub left_button_down: bool,
    // True for exactly one frame when the HUD-toggle key is pressed (F1).
    pub hud_toggle: bool,
    // True for exactly one frame when Escape is pressed while the cursor is
    // not captured. (In captured-cursor worlds Escape continues to release
    // the cursor, as before, and this pulse stays false.)
    pub escape: bool,
    // The canonical key pressed this poll, for the settings-menu rebind
    // capture, or `None`. A one-frame pulse, surfaced regardless of menu /
    // capture state. Wired on Metal; DirectX / Vulkan set it from their key
    // callbacks.
    pub captured_key: Option<crate::assets::Key>,
}

// Scroll units emitted per physical wheel notch on backends whose wheel events
// arrive as discrete notches (DirectX WM_MOUSEWHEEL, GLFW Scroll). macOS reports
// precise (often large) scroll deltas directly, so Metal feeds scrollingDeltaY
// raw and does not use this. The shared UI multiplies scroll_delta by its own
// WHEEL_SCROLL_SPEED (see ui.rs), so this is scroll-delta units per notch.
// Consumed by DirectX (WM_MOUSEWHEEL) and Vulkan (GLFW Scroll); dead on a Metal
// build, which feeds scrollingDeltaY raw.
#[cfg_attr(backend_metal, allow(dead_code))]
pub const WHEEL_NOTCH_SCROLL_UNITS: f32 = 20.0;

// Convert a signed wheel rotation in notches (positive = rotated away from the
// user, i.e. scroll up) into an additive scroll_delta increment. Negated so a
// positive scroll_delta scrolls a panel's content up, matching
// FrameInput.scroll_delta's convention (see ui.rs and metal/input.rs).
#[cfg_attr(backend_metal, allow(dead_code))]
pub fn wheel_notches_to_scroll_delta(notches: f32) -> f32 {
    -notches * WHEEL_NOTCH_SCROLL_UNITS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wheel_notch_sign_and_scale() {
        // Rotating the wheel away from the user (positive notches, "scroll up")
        // yields a negative scroll_delta so a panel's content moves down,
        // revealing the top.
        assert!(wheel_notches_to_scroll_delta(1.0) < 0.0);
        // Rotating toward the user ("scroll down") yields a positive
        // scroll_delta so the content moves up, revealing lower rows.
        assert!(wheel_notches_to_scroll_delta(-1.0) > 0.0);
        // The increment scales linearly with the number of notches.
        assert_eq!(
            wheel_notches_to_scroll_delta(-2.0),
            2.0 * wheel_notches_to_scroll_delta(-1.0)
        );
    }
}
