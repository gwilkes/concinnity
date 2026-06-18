// GraphicsSystem scene-reel wiring and per-frame scene visibility application.

use crate::assets::{Prop, Scene, SceneReel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::scene_reel;

use super::*;

impl GraphicsSystem {
    pub(super) fn setup_scene_reel(&mut self, ctx: &mut PipelineContext) {
        let scenes: Vec<Scene> = ctx.drain::<Scene>();
        let scene_map: std::collections::HashMap<AssetId, Scene> =
            scenes.into_iter().map(|s| (s.asset_id, s)).collect();
        {
            // Each prop's scene is resolved at build time (build/pipeline.rs::
            // resolve_scene_refs), so we just read the baked-in id here.
            let props: Vec<_> = ctx.query::<Prop>().collect();
            self.prop_scene = props.iter().map(|p| p.scene).collect();
        }
        let reel_opt = ctx.drain::<SceneReel>().into_iter().next();
        if let Some(reel) = reel_opt {
            if reel.scenes.is_empty() {
                tracing::warn!("SceneReel {} has no scenes, ignored", reel.asset_id);
            } else {
                let entries: Vec<scene_reel::ReelEntry> = reel
                    .scenes
                    .iter()
                    .map(|&scene_id| {
                        let scene = scene_map.get(&scene_id);
                        scene_reel::ReelEntry {
                            scene: scene_id,
                            duration_secs: scene.and_then(|s| s.duration_secs),
                            transition: scene
                                .map(|s| s.transition.clone())
                                .unwrap_or_else(|| "Cut".to_string()),
                        }
                    })
                    .collect();
                let start_idx = (reel.start_index as usize).min(entries.len() - 1);
                let active_scene = entries[start_idx].scene;
                self.apply_scene_visibility(active_scene);
                self.reel = Some(scene_reel::ReelState {
                    entries,
                    current_idx: start_idx,
                    looping: reel.looping,
                    scene_started_at: 0.0,
                    fade: scene_reel::FadePhase::None,
                    base_clear_color: self.clear_color,
                });
            }
        }
    }

    pub(super) fn apply_scene_visibility(&mut self, active_scene: AssetId) {
        if let Some(backend) = self.backend.as_deref_mut() {
            scene_reel::set_scene_visibility(
                &self.prop_draw_indices,
                &self.prop_scene,
                active_scene,
                backend,
            );
        }
    }
}
