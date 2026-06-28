// src/gfx/graphics_system/despawn.rs
//
// Runtime entity despawn: remove an authored placement (and its descendants)
// from the live world and reclaim the GPU draw slots it occupied, so nothing it
// contributed lingers in any pass. Driven by DespawnRequest events the
// GraphicsSystem drains each step (see frame.rs).

use crate::assets::{Children, RenderHandle, SkeletonPose};
use crate::ecs::{Entity, PipelineContext};
use crate::gfx::backend::RenderBackend;

// Collect an entity together with every descendant reachable through Children
// edges, pre-order, de-duplicated. A read-only walk: the caller hides each
// entity's draw slots and despawns it. A cycle in the edges (which the parent
// builder never produces) terminates on the de-dup check.
pub(super) fn collect_subtree(ctx: &PipelineContext, root: Entity) -> Vec<Entity> {
    let mut out: Vec<Entity> = Vec::new();
    let mut stack = vec![root];
    while let Some(entity) = stack.pop() {
        if out.contains(&entity) {
            continue;
        }
        out.push(entity);
        if let Some(children) = ctx.get::<Children>(entity) {
            stack.extend(children.0.iter().copied());
        }
    }
    out
}

// A GPU slot retired by the despawn cascade: a static draw slot (from a
// RenderHandle) or a skinned instance slot (from a SkeletonPose). One sink
// covers both so the cascade keeps a single `&mut backend` borrow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RetiredSlot {
    Draw(usize),
    Skinned(usize),
}

// Despawn an entity and its descendants: retire every GPU slot each one owns,
// then remove it from every component column and recycle its id. A static
// renderable's slots come from its RenderHandle (RetiredSlot::Draw); a skinned
// entity has no RenderHandle but a SkeletonPose whose `skinned_index` is a
// RetiredSlot::Skinned (its pooled instance slot returns to the reserve, or an
// authored template is just hidden). The slots are hidden, not compacted.
// `retire` is the slot sink so the cascade is testable without a full backend;
// `despawn_subtree` dispatches it to the backend's retire methods. Returns the
// number of entities removed.
fn despawn_collected(
    ctx: &mut PipelineContext,
    root: Entity,
    mut retire: impl FnMut(RetiredSlot),
) -> usize {
    let entities = collect_subtree(ctx, root);
    for &entity in &entities {
        // Clone the slot list out so the immutable borrow ends before despawn.
        let slots: Vec<u32> = ctx
            .get::<RenderHandle>(entity)
            .map(|h| h.draws.clone())
            .unwrap_or_default();
        for slot in slots {
            retire(RetiredSlot::Draw(slot as usize));
        }
        if let Some(skinned_index) = ctx.get::<SkeletonPose>(entity).map(|p| p.skinned_index) {
            retire(RetiredSlot::Skinned(skinned_index));
        }
        ctx.despawn(entity);
    }
    entities.len()
}

// Despawn `root` and its descendants, hiding each entity's GPU slots (static
// draw slots and skinned instance slots) through the backend. Returns the
// number of entities removed.
pub(super) fn despawn_subtree(
    ctx: &mut PipelineContext,
    backend: &mut dyn RenderBackend,
    root: Entity,
) -> usize {
    despawn_collected(ctx, root, |slot| match slot {
        RetiredSlot::Draw(i) => backend.retire_draw_object(i),
        RetiredSlot::Skinned(i) => backend.retire_skinned_draw_object(i),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Parent, Transform};
    use crate::blob::BlobData;
    use crate::ecs::{ComponentStorage, Resources};
    use crate::gfx::profile::FrameProfile;

    // Build an isolated PipelineContext over fresh storage, like the draw_list
    // tests, so a despawn cascade can be exercised without a backend.
    fn run<R>(body: impl FnOnce(&mut PipelineContext) -> R) -> R {
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
        body(&mut ctx)
    }

    #[test]
    fn collect_subtree_gathers_root_and_descendants() {
        run(|ctx| {
            // root -> {a, b}; a -> {c}. A four-entity tree.
            let root = ctx.components.spawn();
            let a = ctx.components.spawn();
            let b = ctx.components.spawn();
            let c = ctx.components.spawn();
            ctx.insert(root, Children(vec![a, b]));
            ctx.insert(a, Children(vec![c]));

            let mut got = collect_subtree(ctx, root);
            got.sort_by_key(|e| e.index());
            let mut want = vec![root, a, b, c];
            want.sort_by_key(|e| e.index());
            assert_eq!(got, want);
        });
    }

    #[test]
    fn collect_subtree_of_a_leaf_is_just_itself() {
        run(|ctx| {
            let lone = ctx.components.spawn();
            ctx.insert(lone, Transform::default());
            assert_eq!(collect_subtree(ctx, lone), vec![lone]);
        });
    }

    #[test]
    fn despawn_cascade_removes_subtree_and_retires_its_slots() {
        run(|ctx| {
            // A parent with two render slots and one child with one slot.
            let parent = ctx.components.spawn();
            let child = ctx.components.spawn();
            // An unrelated entity that must survive the cascade.
            let other = ctx.components.spawn();

            ctx.insert(parent, Transform::default());
            ctx.insert(
                parent,
                RenderHandle {
                    draws: vec![10, 11],
                },
            );
            ctx.insert(parent, Children(vec![child]));
            ctx.insert(child, Transform::default());
            ctx.insert(child, RenderHandle { draws: vec![12] });
            ctx.insert(child, Parent(parent));
            ctx.insert(other, Transform::default());
            ctx.insert(other, RenderHandle { draws: vec![99] });

            let mut retired: Vec<usize> = Vec::new();
            let removed = despawn_collected(ctx, parent, |slot| {
                if let RetiredSlot::Draw(i) = slot {
                    retired.push(i);
                }
            });

            assert_eq!(removed, 2, "parent + child removed");
            retired.sort_unstable();
            assert_eq!(
                retired,
                vec![10, 11, 12],
                "every slot in the subtree retired"
            );

            // The subtree is gone; the unrelated entity and its slot survive.
            assert!(ctx.get::<Transform>(parent).is_none());
            assert!(ctx.get::<Transform>(child).is_none());
            assert!(ctx.get::<RenderHandle>(child).is_none());
            assert_eq!(ctx.query::<Transform>().count(), 1, "only `other` remains");
            let survivor = ctx.get::<RenderHandle>(other).expect("other survives");
            assert_eq!(survivor.draws, vec![99]);
        });
    }

    #[test]
    fn despawn_an_entity_without_a_render_handle_retires_nothing() {
        run(|ctx| {
            let e = ctx.components.spawn();
            ctx.insert(e, Transform::default());

            let mut retired: Vec<RetiredSlot> = Vec::new();
            let removed = despawn_collected(ctx, e, |slot| retired.push(slot));

            assert_eq!(removed, 1);
            assert!(retired.is_empty(), "no slots to retire");
            assert!(ctx.get::<Transform>(e).is_none(), "entity despawned");
        });
    }

    #[test]
    fn despawn_retires_a_skinned_instance_slot() {
        use crate::assets::SkeletonPose;
        use crate::ecs::asset_id::AssetId;
        use crate::gfx::skinning::Skeleton;
        run(|ctx| {
            // A skinned entity carries a SkeletonPose (no RenderHandle); its
            // skinned_index is the slot to retire.
            let skinned = ctx.components.spawn();
            ctx.insert(
                skinned,
                SkeletonPose::new(AssetId(1), 4, Skeleton::new(Vec::new())),
            );

            let mut retired: Vec<RetiredSlot> = Vec::new();
            let removed = despawn_collected(ctx, skinned, |slot| retired.push(slot));

            assert_eq!(removed, 1);
            assert_eq!(
                retired,
                vec![RetiredSlot::Skinned(4)],
                "the skinned_index was retired, and there were no draw slots"
            );
            assert!(
                ctx.get::<SkeletonPose>(skinned).is_none(),
                "entity despawned"
            );
        });
    }
}
