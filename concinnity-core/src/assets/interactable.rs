// src/assets/interactable.rs

use crate::ecs::{AssetOrigin, Component};

/// Marks an entity the player can interact with (press the interact key while
/// close and facing it to trigger its behavior).
///
/// Runtime-only zero-size tag. Present on an entity whose `Prop` set
/// `interactable`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Interactable;

/// `Interactable` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct InteractableArgs {}

impl Component for Interactable {
    const NAME: &'static str = "Interactable";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = InteractableArgs;

    fn to_args(&self) -> InteractableArgs {
        InteractableArgs {}
    }
    fn from_args(_: InteractableArgs) -> Self {
        Interactable
    }
}
