// src/assets/scene_command.rs

use crate::ecs::Component;
use crate::ecs::asset_id::AssetId;

/// Runtime-only signal pushed by `UiInputSystem` when a `HitRegion` fires.
///
/// `GraphicsSystem` drains these each step and applies the scene jump.
/// World authors never declare this type directly.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SceneCommand {
    pub scene: AssetId,
    pub transition: String,
}

impl Default for SceneCommand {
    fn default() -> Self {
        Self {
            scene: AssetId::default(),
            transition: "FadeBlack".to_string(),
        }
    }
}

impl Component for SceneCommand {
    const NAME: &'static str = "SceneCommand";
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
