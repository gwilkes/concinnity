// src/assets/audio_command.rs

// Runtime-only event sent by GraphicsSystem when the master-volume setting
// changes, read by AudioSystem from its Events<AudioCommand> queue.
//
// The master volume is owned by the audio engine, not the renderer, so a change
// made in the settings menu is handed across as this event rather than read
// from disk each frame: the audio system scales its output on the same tick.
// World authors never declare this type directly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AudioCommand {
    // New master output volume as a linear gain (0.0 = silent, 1.0 = full).
    // Applied live by the audio system.
    pub master_volume: f32,
}
