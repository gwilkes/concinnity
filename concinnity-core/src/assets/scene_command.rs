// src/assets/scene_command.rs

use crate::ecs::asset_id::AssetId;

// Runtime-only event sent by UiInputSystem when a scene-jump HitRegion fires.
// GraphicsSystem reads these from its Events<SceneCommand> queue each step and
// applies the scene jump. World authors never declare this type directly.
#[derive(Debug, Clone)]
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
