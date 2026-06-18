// src/assets/controls_command.rs

use crate::ecs::Component;

/// Runtime-only signal pushed by `GraphicsSystem` when a live control setting
/// changes, drained by `Camera3DSystem`.
///
/// Control settings (mouse sensitivity today) are owned by the camera
/// controller, not the renderer, so a change made in the settings menu is
/// handed across as this signal rather than read from disk each frame: the
/// camera updates its live value on the same tick. World authors never declare
/// this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct ControlsCommand {
    /// New mouse-look sensitivity, in radians per pixel. Applied live by the
    /// camera controller.
    pub mouse_sensitivity: f32,
}

impl Component for ControlsCommand {
    const NAME: &'static str = "ControlsCommand";
    type Args = Self;

    fn to_args(&self) -> Self {
        *self
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
