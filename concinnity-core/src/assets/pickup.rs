// src/assets/pickup.rs

use crate::ecs::{AssetOrigin, Component};

/// Marks an entity the player can pick up and carry with the interact key.
///
/// Runtime-only zero-size tag. Present on an entity whose `Prop` set `pickup`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pickup;

/// `Pickup` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct PickupArgs {}

impl Component for Pickup {
    const NAME: &'static str = "Pickup";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = PickupArgs;

    fn to_args(&self) -> PickupArgs {
        PickupArgs {}
    }
    fn from_args(_: PickupArgs) -> Self {
        Pickup
    }
}
