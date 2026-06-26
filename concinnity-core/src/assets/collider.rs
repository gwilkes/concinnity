// src/assets/collider.rs

use crate::assets::PropCollider;
use crate::ecs::{AssetOrigin, Component};

/// Collision volume attached to an entity, in local space scaled by the
/// entity's transform.
///
/// Runtime-only. Carries the same shape description a `Prop` declares through
/// its `collider` field.
#[derive(Debug, Clone, Default)]
pub struct Collider(pub PropCollider);

/// `Collider` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColliderArgs {}

impl Component for Collider {
    const NAME: &'static str = "Collider";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = ColliderArgs;

    fn to_args(&self) -> ColliderArgs {
        ColliderArgs {}
    }
    fn from_args(_: ColliderArgs) -> Self {
        Self::default()
    }
}
