// src/assets/scene_member.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// The `Scene` an entity belongs to, for per-scene show/hide.
///
/// Runtime-only. An entity without this component is visible in every scene.
/// Carries the scene identity a `Prop` resolves into its `scene` field.
#[derive(Debug, Clone, Copy, Default)]
pub struct SceneMember(pub AssetId);

/// `SceneMember` is never authored, so its args are empty.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct SceneMemberArgs {}

impl Component for SceneMember {
    const NAME: &'static str = "SceneMember";
    const ORIGIN: AssetOrigin = AssetOrigin::RuntimeOnly;
    type Args = SceneMemberArgs;

    fn to_args(&self) -> SceneMemberArgs {
        SceneMemberArgs {}
    }
    fn from_args(_: SceneMemberArgs) -> Self {
        Self::default()
    }
}
