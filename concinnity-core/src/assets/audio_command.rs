// src/assets/audio_command.rs

use crate::ecs::Component;

/// Runtime-only signal pushed by `GraphicsSystem` when the master-volume setting
/// changes, drained by `AudioSystem`.
///
/// The master volume is owned by the audio engine, not the renderer, so a change
/// made in the settings menu is handed across as this signal rather than read
/// from disk each frame: the audio system scales its output on the same tick.
/// World authors never declare this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct AudioCommand {
    /// New master output volume as a linear gain (0.0 = silent, 1.0 = full).
    /// Applied live by the audio system.
    pub master_volume: f32,
}

impl Component for AudioCommand {
    const NAME: &'static str = "AudioCommand";
    type Args = Self;

    fn to_args(&self) -> Self {
        *self
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
