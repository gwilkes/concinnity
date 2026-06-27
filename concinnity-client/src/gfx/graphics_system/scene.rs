// GraphicsSystem scene-reel wiring and per-frame scene visibility application.

use crate::assets::{RenderHandle, Scene, SceneMember, SceneReel};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::scene_reel;

use super::*;

// Build the (draw-slots, scene) visibility pairs from the per-entity
// components: every entity with a RenderHandle contributes its GPU draw slots,
// tagged with the SceneMember scene it belongs to (None = always visible),
// consumed by the scene_reel visibility functions. The two returned vectors are
// index-aligned: pair i is one entity's draws and its scene.
pub(super) fn decomposed_visibility_snapshot(
    ctx: &PipelineContext,
) -> (Vec<Vec<usize>>, Vec<Option<AssetId>>) {
    let scene_of: std::collections::HashMap<crate::ecs::Entity, AssetId> = ctx
        .join2::<SceneMember, RenderHandle>()
        .map(|(entity, member, _)| (entity, member.0))
        .collect();
    let mut draws = Vec::new();
    let mut scenes = Vec::new();
    for (entity, handle) in ctx.query_with_entity::<RenderHandle>() {
        draws.push(handle.draws.iter().map(|&slot| slot as usize).collect());
        scenes.push(scene_of.get(&entity).copied());
    }
    (draws, scenes)
}

impl GraphicsSystem {
    pub(super) fn setup_scene_reel(&mut self, ctx: &mut PipelineContext) {
        let scenes: Vec<Scene> = ctx.drain::<Scene>();
        let scene_map: std::collections::HashMap<AssetId, Scene> =
            scenes.into_iter().map(|s| (s.asset_id, s)).collect();
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
                self.apply_scene_visibility(ctx, active_scene);
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

    pub(super) fn apply_scene_visibility(&mut self, ctx: &PipelineContext, active_scene: AssetId) {
        // Snapshot visibility from the per-entity components before borrowing the
        // backend, so the ctx borrow is released by the time set_scene_visibility
        // runs.
        let (draws, scenes) = decomposed_visibility_snapshot(ctx);
        if let Some(backend) = self.backend.as_deref_mut() {
            scene_reel::set_scene_visibility(&draws, &scenes, active_scene, backend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::ecs::{ComponentStorage, Resources};
    use crate::gfx::profile::FrameProfile;

    // The snapshot pairs each entity's draw slots with its scene; scene-less
    // entities are always visible.
    #[test]
    fn snapshot_pairs_each_entity_draws_with_its_scene() {
        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        // Entity in scene 7 with two draw slots.
        let a = ctx.components.spawn();
        ctx.insert(
            a,
            RenderHandle {
                draws: vec![10, 11],
            },
        );
        ctx.insert(a, SceneMember(AssetId(7)));
        // Entity with no scene (always visible), one slot.
        let b = ctx.components.spawn();
        ctx.insert(b, RenderHandle { draws: vec![20] });
        // Entity in scene 8, one slot.
        let c = ctx.components.spawn();
        ctx.insert(c, RenderHandle { draws: vec![30] });
        ctx.insert(c, SceneMember(AssetId(8)));

        let (draws, scenes) = decomposed_visibility_snapshot(&ctx);

        // Pairs follow RenderHandle column order (a, b, c).
        assert_eq!(draws, vec![vec![10usize, 11], vec![20], vec![30]]);
        assert_eq!(scenes, vec![Some(AssetId(7)), None, Some(AssetId(8))]);
    }

    // An entity carrying SceneMember but no RenderHandle contributes no draws
    // (it is not in the render set), so it never appears in the snapshot.
    #[test]
    fn snapshot_skips_scene_members_without_a_render_handle() {
        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        let only_scene = ctx.components.spawn();
        ctx.insert(only_scene, SceneMember(AssetId(7)));
        let rendered = ctx.components.spawn();
        ctx.insert(rendered, RenderHandle { draws: vec![5] });

        let (draws, scenes) = decomposed_visibility_snapshot(&ctx);
        assert_eq!(draws, vec![vec![5usize]]);
        assert_eq!(scenes, vec![None]);
    }
}
