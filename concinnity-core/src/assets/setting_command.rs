// src/assets/setting_command.rs

use crate::assets::Key;
use crate::ecs::asset_id::AssetId;

// What a "setting:*" action does to its value: cycle one step (for a stepper
// row), jump to an absolute option index (for a dropdown pick), set an absolute
// position in [0, 1] (for a Slider row's drag), or bind a key (for a key-rebind
// row).
//
// Eq is intentionally not derived: SetFraction carries an f32.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SettingOp {
    #[default]
    Next,
    Prev,
    // Jump straight to this option index (clamped to the option count). Sent
    // once when the user picks an entry from an open dropdown list.
    SetIndex(usize),
    // Set the value to this fraction of its range, 0.0..=1.0. Sent each frame
    // while a slider is dragged.
    SetFraction(f32),
    // Bind the named action (the command's setting) to this key. Sent once when
    // the user presses a key while a rebind row is capturing.
    Rebind(Key),
}

// Runtime-only event sent by UiInputSystem when a "setting:*" action fires.
// GraphicsSystem reads these each step: it applies the change to the named
// setting (cycling it or setting it from a fraction), updates the value_label
// text, and (when persist is set) writes the new value to the settings store.
// World authors never declare this type directly.
#[derive(Debug, Clone)]
pub struct SettingCommand {
    // Engine setting key (e.g. "vsync").
    pub setting: String,
    // How to change the value.
    pub op: SettingOp,
    // The value TextLabel to update with the new value, when known.
    pub value_label: Option<AssetId>,
    // Whether to write the new value to the settings store. A cycle is one
    // discrete change and always persists; a slider drag persists only on
    // release (the in-progress frames apply live but skip the disk write).
    pub persist: bool,
}

impl Default for SettingCommand {
    fn default() -> Self {
        Self {
            setting: String::new(),
            op: SettingOp::Next,
            value_label: None,
            persist: true,
        }
    }
}
