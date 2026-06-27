// src/assets/children.rs

use crate::ecs::{AssetOrigin, Component, Entity};

/// The entities that name this entity as their `Parent`.
///
/// Runtime-only, maintained automatically alongside `Parent` so a transform-
/// propagation pass can walk the hierarchy top-down and so despawning a parent
/// can cascade to its children.
#[derive(Debug, Clone, Default)]
pub struct Children(pub Vec<Entity>);

/// `Children` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChildrenArgs {}

impl Component for Children {
    const NAME: &'static str = "Children";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = ChildrenArgs;

    fn to_args(&self) -> ChildrenArgs {
        ChildrenArgs {}
    }
    fn from_args(_: ChildrenArgs) -> Self {
        Self::default()
    }
}
