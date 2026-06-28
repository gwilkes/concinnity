// src/assets/lifetime.rs

use crate::ecs::{AssetOrigin, Component};

// Runtime-only countdown on an entity: seconds remaining before it is
// automatically removed. A spawn carries it onto short-lived instances
// (projectiles, debris, timed effects); the graphics step decrements it each
// frame and despawns the entity (and its descendants) when it reaches zero,
// reclaiming the GPU draw slots for reuse. World authors never declare this
// type directly; it is set at spawn time.
#[derive(Debug, Default, Clone, Copy)]
pub struct Lifetime {
    pub remaining: f32,
}

// `Lifetime` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct LifetimeArgs {}

impl Component for Lifetime {
    const NAME: &'static str = "Lifetime";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = LifetimeArgs;

    fn to_args(&self) -> LifetimeArgs {
        LifetimeArgs {}
    }
    fn from_args(_: LifetimeArgs) -> Self {
        Self::default()
    }
}
