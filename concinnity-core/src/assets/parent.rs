// src/assets/parent.rs

use crate::ecs::{AssetOrigin, Component, Entity};

/// The entity whose world transform this entity inherits.
///
/// Runtime-only. When present, this entity's `Transform` is relative to the
/// parent's world transform. Carries the relationship a `Prop` declares with
/// its `parent` field, resolved from a name to a live `Entity`.
#[derive(Debug, Clone, Copy)]
pub struct Parent(pub Entity);

impl Default for Parent {
    fn default() -> Self {
        // Never observed: Parent is inserted at runtime with a real parent, not
        // built from serialized args.
        Parent(Entity::dangling())
    }
}

/// `Parent` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParentArgs {}

impl Component for Parent {
    const NAME: &'static str = "Parent";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = ParentArgs;

    fn to_args(&self) -> ParentArgs {
        ParentArgs {}
    }
    fn from_args(_: ParentArgs) -> Self {
        Self::default()
    }
}
