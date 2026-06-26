// src/assets/view_command.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event sent by UiInputSystem when a `view:*` action fires.
// UiInputSystem reads these from its Events<ViewCommand> queue on the next step
// and applies the show/hide transition. World authors never declare this type
// directly.
#[derive(Debug, Clone, Default)]
pub enum ViewCommand {
    Show(AssetId),
    #[default]
    Hide,
    Toggle(AssetId),
}
