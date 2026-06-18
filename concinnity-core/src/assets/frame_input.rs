// src/assets/frame_input.rs

use crate::ecs::Component;

/// Per-frame keyboard and mouse input state.
///
/// One `FrameInput` is updated each frame from the window's keyboard and mouse
/// state and read by camera and UI behavior. It is maintained automatically and
/// is never saved with the world.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrameInput {
    /// True while the move-forward key (W) is held.
    pub forward: bool,
    /// True while the move-backward key (S) is held.
    pub backward: bool,
    /// True while the strafe-left key (A) is held.
    pub left: bool,
    /// True while the strafe-right key (D) is held.
    pub right: bool,
    /// True while the sprint key (Shift) is held.
    pub sprint: bool,
    /// True for exactly one frame when the interact key (E) is pressed.
    pub interact: bool,
    /// True for exactly one frame when the jump key (Space) is pressed.
    pub jump: bool,
    /// Accumulated horizontal mouse movement since the last frame (pixels).
    pub mouse_dx: f32,
    /// Accumulated vertical mouse movement since the last frame (pixels).
    pub mouse_dy: f32,
    /// Accumulated vertical scroll-wheel movement since the last frame. Positive
    /// scrolls the content up (a scrollable UI panel moves its rows up). Cleared
    /// each frame like the mouse deltas.
    pub scroll_delta: f32,
    /// Absolute cursor X position in window pixels (origin top-left).
    /// Only meaningful when the cursor is not captured.
    pub mouse_x: f32,
    /// Absolute cursor Y position in window pixels (origin top-left).
    /// Only meaningful when the cursor is not captured.
    pub mouse_y: f32,
    /// True for exactly one frame when the left mouse button is pressed
    /// while the cursor is not captured.
    pub left_click: bool,
    /// True while the left mouse button is held down (cursor not captured).
    /// Unlike `left_click` this stays true across frames until release, so a
    /// UI drag (e.g. a slider) can track the cursor for its whole duration.
    pub left_button_down: bool,
    /// Live logical viewport size in pixels `[width, height]`. Used to map
    /// overlay (View-owned) UI between its fixed reference resolution and the
    /// window, so menus scale with the window and the cursor still hit-tests
    /// against the scaled controls. `[0.0, 0.0]` before the backend is ready.
    pub viewport: [f32; 2],
    /// True for exactly one frame when the HUD-toggle key (F1) is pressed.
    pub hud_toggle: bool,
    /// True for exactly one frame when Escape is pressed while the cursor is
    /// not captured (menu / UI worlds). Used to fire [KeyBinding](#keybinding)
    /// actions. In worlds that capture the cursor, Escape instead releases the
    /// cursor and this stays false.
    pub escape: bool,
    /// The canonical key pressed this frame, for one frame, or `None`. Surfaced
    /// regardless of menu state (unlike the gameplay keys, which freeze while a
    /// menu is open) so the settings menu can capture a key for rebinding.
    pub captured_key: Option<crate::assets::Key>,
}

impl Component for FrameInput {
    const NAME: &'static str = "FrameInput";

    type Args = FrameInput;

    fn to_args(&self) -> FrameInput {
        self.clone()
    }

    fn from_args(args: FrameInput) -> Self {
        args
    }
}
