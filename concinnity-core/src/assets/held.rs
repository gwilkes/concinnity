// src/assets/held.rs

use crate::ecs::{AssetOrigin, Component};

/// Marks an entity currently being carried by the player.
///
/// Runtime-only zero-size tag, added and removed by the physics system on
/// pickup and drop. While present, the entity is driven as a kinematic body
/// that follows the camera instead of simulating dynamically.
#[derive(Debug, Clone, Copy, Default)]
pub struct Held;

/// `Held` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeldArgs {}

impl Component for Held {
    const NAME: &'static str = "Held";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = HeldArgs;

    fn to_args(&self) -> HeldArgs {
        HeldArgs {}
    }
    fn from_args(_: HeldArgs) -> Self {
        Held
    }
}
