// src/assets/view_command.rs

use crate::ecs::Component;
use crate::ecs::asset_id::AssetId;

/// Runtime-only signal pushed by `UiInputSystem` when a `view:*` action fires.
///
/// Drained by `UiInputSystem` itself on the next step to apply the show/hide
/// transition. World authors never declare this type directly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub enum ViewCommand {
    Show(AssetId),
    #[default]
    Hide,
    Toggle(AssetId),
}

impl Component for ViewCommand {
    const NAME: &'static str = "ViewCommand";
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
