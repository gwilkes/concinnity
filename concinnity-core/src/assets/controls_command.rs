// src/assets/controls_command.rs

// Runtime-only event sent by GraphicsSystem when a live control setting
// changes, read by Camera3DSystem from its Events<ControlsCommand> queue.
//
// Control settings (mouse sensitivity today) are owned by the camera
// controller, not the renderer, so a change made in the settings menu is
// handed across as this event rather than read from disk each frame: the
// camera updates its live value on the same tick. World authors never declare
// this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ControlsCommand {
    // New mouse-look sensitivity, in radians per pixel. Applied live by the
    // camera controller.
    pub mouse_sensitivity: f32,
}
