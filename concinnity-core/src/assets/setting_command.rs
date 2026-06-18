// src/assets/setting_command.rs

use crate::assets::Key;
use crate::ecs::Component;
use crate::ecs::asset_id::AssetId;

/// What a `"setting:*"` action does to its value: cycle one step (for an
/// [OptionSelect](#optionselect) row), set an absolute position in `[0, 1]`
/// (for a [Slider](#slider) row's drag), or bind a key (for a key-rebind row).
///
/// `Eq` is intentionally not derived: `SetFraction` carries an `f32`.
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum SettingOp {
    #[default]
    Next,
    Prev,
    /// Set the value to this fraction of its range, `0.0`..=`1.0`. Pushed each
    /// frame while a slider is dragged.
    SetFraction(f32),
    /// Bind the named action (the command's `setting`) to this key. Pushed once
    /// when the user presses a key while a rebind row is capturing.
    Rebind(Key),
}

/// Runtime-only signal pushed by `UiInputSystem` when a `"setting:*"` action
/// fires.
///
/// `GraphicsSystem` drains these each step: it applies the change to the
/// named setting (cycling it or setting it from a fraction), updates the
/// `value_label` text, and (when `persist` is set) writes the new value to the
/// settings store. World authors never declare this type directly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SettingCommand {
    /// Engine setting key (e.g. `"vsync"`).
    pub setting: String,
    /// How to change the value.
    pub op: SettingOp,
    /// The value `TextLabel` to update with the new value, when known.
    pub value_label: Option<AssetId>,
    /// Whether to write the new value to the settings store. A cycle is one
    /// discrete change and always persists; a slider drag persists only on
    /// release (the in-progress frames apply live but skip the disk write).
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

impl Component for SettingCommand {
    const NAME: &'static str = "SettingCommand";
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
