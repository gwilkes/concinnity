// src/gfx/graphics_system/spawn.rs
//
// Runtime entity spawn: instantiate a copy of an existing placement at a new
// transform and give it the GPU draw slots, components, and (optionally) a
// Lifetime the copy needs to live in the world. The symmetric counterpart to
// despawn.rs. Driven by SpawnRequest events the GraphicsSystem drains each step
// (see frame.rs), and paired with a Lifetime tick that auto-despawns expired
// instances so their freed draw slots can be recycled by the next spawn.

use crate::assets::{
    GlobalTransform, Lifetime, MeshRenderer, ModelRenderer, RenderHandle, SkeletonPose, Spawner,
    Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::decompose::EntityByName;
use crate::ecs::{Entity, PipelineContext};

// Instantiate a runtime copy of `template`'s renderable: clone each of its
// backend draw slots at `transform` through `clone_slot`, then build a new
// entity carrying the cloned slots, a copy of the template's renderer, the
// placement, and an optional Lifetime. `clone_slot(src_draw_idx, model)`
// returns the new backend slot index (a vacated slot reused, or a freshly
// appended one); it is the seam the test drives with a DrawSlotAllocator
// instead of a live backend. When `name` is Some the new entity is registered
// under it so it can later be addressed by name like an authored placement;
// transient spawns (a Spawner's churn) pass None to avoid interning a name per
// spawn. Returns the new entity, or None when the template has no draw slots to
// copy or a clone fails.
pub(super) fn spawn_from_template(
    ctx: &mut PipelineContext,
    template: Entity,
    name: Option<AssetId>,
    transform: Transform,
    lifetime: Option<f32>,
    mut clone_slot: impl FnMut(usize, [[f32; 4]; 4]) -> Option<usize>,
) -> Option<Entity> {
    let src_slots: Vec<u32> = ctx.get::<RenderHandle>(template).map(|h| h.draws.clone())?;
    if src_slots.is_empty() {
        return None;
    }
    let model = transform.model_matrix();
    let mut draws: Vec<u32> = Vec::with_capacity(src_slots.len());
    for src in src_slots {
        let new_slot = clone_slot(src as usize, model)?;
        draws.push(new_slot as u32);
    }

    // Copy whichever renderer the template carries so the new entity is a
    // first-class renderable for every system that joins on it.
    let mesh_renderer = ctx.get::<MeshRenderer>(template).cloned();
    let model_renderer = ctx.get::<ModelRenderer>(template).cloned();

    let entity = ctx.components.spawn();
    ctx.insert(entity, transform);
    ctx.insert(entity, GlobalTransform(model));
    ctx.insert(entity, RenderHandle { draws });
    if let Some(renderer) = mesh_renderer {
        ctx.insert(entity, renderer);
    } else if let Some(renderer) = model_renderer {
        ctx.insert(entity, renderer);
    }
    if let Some(secs) = lifetime {
        ctx.insert(entity, Lifetime { remaining: secs });
    }
    if let Some(name) = name
        && let Some(by_name) = ctx.resource_mut::<EntityByName>()
    {
        by_name.0.insert(name, entity);
    }
    Some(entity)
}

// Instantiate a runtime copy of a skinned `template` (a SkinnedMesh's
// SkeletonPose entity) at `transform`. Unlike the static path, a skinned
// instance is not a cloned draw slot: it claims one of the template's
// pre-reserved hidden bind-pose copies through `acquire_slot`, which reveals it
// and returns its skinned index. The new entity carries its own SkeletonPose
// (so AnimationSystem drives it, keyed on the shared mesh id, in lockstep with
// the template), a Transform (so the per-frame model push can move it), and an
// optional Lifetime. `acquire_slot(template_skinned_index, model)` is the seam
// the test drives with a pool instead of a live backend. When `name` is Some
// the instance is registered so it can be addressed (e.g. despawned) by name.
// Returns the new entity, or None when the template is not skinned or its
// instance pool is exhausted.
pub(super) fn spawn_skinned_from_template(
    ctx: &mut PipelineContext,
    template: Entity,
    name: Option<AssetId>,
    transform: Transform,
    lifetime: Option<f32>,
    mut acquire_slot: impl FnMut(usize, [[f32; 4]; 4]) -> Option<usize>,
) -> Option<Entity> {
    let (mesh_id, template_index, skeleton) = ctx
        .get::<SkeletonPose>(template)
        .map(|p| (p.mesh_id, p.skinned_index, p.skeleton.clone()))?;
    let model = transform.model_matrix();
    let skinned_index = acquire_slot(template_index, model)?;

    let entity = ctx.components.spawn();
    ctx.insert(entity, transform);
    ctx.insert(entity, SkeletonPose::new(mesh_id, skinned_index, skeleton));
    if let Some(secs) = lifetime {
        ctx.insert(entity, Lifetime { remaining: secs });
    }
    if let Some(name) = name
        && let Some(by_name) = ctx.resource_mut::<EntityByName>()
    {
        by_name.0.insert(name, entity);
    }
    Some(entity)
}

// One spawn a Spawner is due to emit this step: the template to copy, where to
// place it, and how long the copy should live. Returned by `tick_spawners` for
// the caller to route through `spawn_from_template` with the live backend, the
// same way `tick_lifetimes` returns expiries for the caller to despawn.
pub(super) struct DueSpawn {
    pub template: AssetId,
    pub transform: Transform,
    pub lifetime: Option<f32>,
}

// Advance every Spawner's clock by `dt` and return the spawns now due. A
// spawner emits one copy per whole `interval` elapsed (so a long frame that
// crosses several intervals catches up), at the spawner entity's own Transform.
// A non-positive interval is inert (never spawns). A zero `lifetime` means the
// copy is not auto-removed; otherwise it carries that countdown.
pub(super) fn tick_spawners(ctx: &mut PipelineContext, dt: f32) -> Vec<DueSpawn> {
    let spawners: Vec<(Entity, AssetId, f32, f32)> = ctx
        .query_with_entity::<Spawner>()
        .map(|(entity, s)| (entity, s.template, s.interval, s.lifetime))
        .collect();
    let mut due = Vec::new();
    for (entity, template, interval, lifetime) in spawners {
        if interval <= 0.0 {
            continue;
        }
        let transform = ctx.get::<Transform>(entity).copied().unwrap_or_default();
        if let Some(spawner) = ctx.get_mut::<Spawner>(entity) {
            spawner.elapsed += dt;
            while spawner.elapsed >= interval {
                spawner.elapsed -= interval;
                spawner.count += 1;
                due.push(DueSpawn {
                    template,
                    transform,
                    lifetime: (lifetime > 0.0).then_some(lifetime),
                });
            }
        }
    }
    due
}

// Decrement every Lifetime by `dt` and return the entities whose countdown
// reached zero this step, for the caller to despawn. Entities still alive keep
// their decremented remaining. Pulling the expired list out (rather than
// despawning inline) keeps this borrow read-only over the join and lets the
// caller route each expiry through the same despawn cascade a DespawnRequest
// uses.
pub(super) fn tick_lifetimes(ctx: &mut PipelineContext, dt: f32) -> Vec<Entity> {
    let entities: Vec<Entity> = ctx
        .query_with_entity::<Lifetime>()
        .map(|(entity, _)| entity)
        .collect();
    let mut expired = Vec::new();
    for entity in entities {
        if let Some(life) = ctx.get_mut::<Lifetime>(entity) {
            life.remaining -= dt;
            if life.remaining <= 0.0 {
                expired.push(entity);
            }
        }
    }
    expired
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::BlobData;
    use crate::ecs::{ComponentStorage, Resources};
    use crate::gfx::draw_slot::{DrawSlotAllocator, SlotAlloc};
    use crate::gfx::profile::FrameProfile;

    // Build an isolated PipelineContext over fresh storage, mirroring the
    // despawn tests, so the spawn/despawn loop can run without a backend.
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

    // A clone_slot seam backed by a DrawSlotAllocator: it pops a vacated slot
    // before growing, exactly like the backend's clone_static_draw_object does
    // against its real draw_objects vec.
    fn alloc_slot(alloc: &mut DrawSlotAllocator) -> usize {
        match alloc.allocate() {
            SlotAlloc::Reuse(slot) => slot,
            SlotAlloc::Append(idx) => idx,
        }
    }

    #[test]
    fn freed_draw_slot_is_reused_by_the_next_spawn() {
        run(|ctx| {
            ctx.insert_resource(EntityByName::default());

            // A template placement occupying draw slot 0.
            let template = ctx.components.spawn();
            ctx.insert(template, Transform::default());
            ctx.insert(
                template,
                MeshRenderer {
                    mesh: None,
                    material: None,
                    texture: None,
                    cull_distance: 0.0,
                },
            );
            ctx.insert(template, RenderHandle { draws: vec![0] });

            // The backend starts with one live slot (the template's).
            let mut alloc = DrawSlotAllocator::with_len(1);

            // First spawn appends a fresh slot past the template's.
            let first = spawn_from_template(
                ctx,
                template,
                Some(AssetId(1)),
                Transform::default(),
                Some(0.5),
                |_src, _model| Some(alloc_slot(&mut alloc)),
            )
            .expect("first spawn");
            let first_slot = ctx.get::<RenderHandle>(first).unwrap().draws.clone();
            assert_eq!(first_slot, vec![1], "first spawn appended slot 1");

            // Its Lifetime expires; the expiry frees the slot like a despawn's
            // retire -> free does, then despawns the entity.
            let expired = tick_lifetimes(ctx, 1.0);
            assert_eq!(expired, vec![first], "the short-lived spawn expired");
            let freed: Vec<u32> = ctx.get::<RenderHandle>(first).unwrap().draws.clone();
            for slot in &freed {
                alloc.free(*slot as usize);
            }
            ctx.despawn(first);
            assert!(ctx.get::<RenderHandle>(first).is_none(), "first despawned");

            // The next spawn reuses the freed slot instead of growing the vec.
            let second = spawn_from_template(
                ctx,
                template,
                Some(AssetId(2)),
                Transform::default(),
                None,
                |_src, _model| Some(alloc_slot(&mut alloc)),
            )
            .expect("second spawn");
            let second_slot = ctx.get::<RenderHandle>(second).unwrap().draws.clone();
            assert_eq!(
                second_slot, freed,
                "the freed draw slot must be recycled by the next spawn"
            );
        });
    }

    #[test]
    fn skinned_spawn_claims_and_recycles_a_pooled_slot() {
        use crate::ecs::asset_id::AssetId;
        use crate::gfx::skinned_pool::SkinnedInstancePool;
        use crate::gfx::skinning::Skeleton;
        run(|ctx| {
            ctx.insert_resource(EntityByName::default());

            // A skinned template at draw slot 0 with two pre-reserved hidden
            // copies (slots 1 and 2) in the pool.
            let template = ctx.components.spawn();
            ctx.insert(
                template,
                SkeletonPose::new(AssetId(10), 0, Skeleton::new(Vec::new())),
            );
            let mut pool = SkinnedInstancePool::new();
            pool.reserve(0, 1);
            pool.reserve(0, 2);

            // The spawn claims a pooled copy and the new entity points at it.
            let first = spawn_skinned_from_template(
                ctx,
                template,
                Some(AssetId(11)),
                Transform::default(),
                Some(0.5),
                |template_idx, _model| pool.acquire(template_idx),
            )
            .expect("first skinned spawn");
            let first_slot = ctx.get::<SkeletonPose>(first).unwrap().skinned_index;
            assert_eq!(
                ctx.get::<SkeletonPose>(first).unwrap().mesh_id,
                AssetId(10),
                "the instance shares the template's mesh id so it animates with it"
            );

            // Its Lifetime expires; the expiry releases the slot to the pool like
            // a despawn's retire does, then despawns the entity.
            let expired = tick_lifetimes(ctx, 1.0);
            assert_eq!(expired, vec![first]);
            pool.release(first_slot);
            ctx.despawn(first);

            // The next spawn recycles the freed slot instead of a fresh one.
            let second = spawn_skinned_from_template(
                ctx,
                template,
                None,
                Transform::default(),
                None,
                |template_idx, _model| pool.acquire(template_idx),
            )
            .expect("second skinned spawn");
            assert_eq!(
                ctx.get::<SkeletonPose>(second).unwrap().skinned_index,
                first_slot,
                "the freed skinned slot must be recycled by the next spawn"
            );
        });
    }

    #[test]
    fn skinned_spawn_with_exhausted_pool_returns_none() {
        use crate::ecs::asset_id::AssetId;
        use crate::gfx::skinned_pool::SkinnedInstancePool;
        use crate::gfx::skinning::Skeleton;
        run(|ctx| {
            let template = ctx.components.spawn();
            ctx.insert(
                template,
                SkeletonPose::new(AssetId(10), 0, Skeleton::new(Vec::new())),
            );
            // A template that reserved no instances has nothing to claim.
            let mut pool = SkinnedInstancePool::new();
            let spawned = spawn_skinned_from_template(
                ctx,
                template,
                None,
                Transform::default(),
                None,
                |template_idx, _model| pool.acquire(template_idx),
            );
            assert!(spawned.is_none(), "an exhausted pool drops the spawn");
        });
    }

    #[test]
    fn spawn_registers_the_instance_by_name() {
        run(|ctx| {
            ctx.insert_resource(EntityByName::default());
            let template = ctx.components.spawn();
            ctx.insert(template, Transform::default());
            ctx.insert(template, RenderHandle { draws: vec![0] });
            let mut alloc = DrawSlotAllocator::with_len(1);

            let spawned = spawn_from_template(
                ctx,
                template,
                Some(AssetId(42)),
                Transform::default(),
                None,
                |_src, _model| Some(alloc_slot(&mut alloc)),
            )
            .expect("spawn");

            let by_name = ctx.resource::<EntityByName>().unwrap();
            assert_eq!(by_name.get(AssetId(42)), Some(spawned));
        });
    }

    #[test]
    fn spawner_emits_one_copy_per_interval_elapsed() {
        run(|ctx| {
            let spawner = ctx.components.spawn();
            ctx.insert(
                spawner,
                Transform {
                    position: [1.0, 2.0, 3.0],
                    ..Transform::default()
                },
            );
            ctx.insert(
                spawner,
                Spawner {
                    template: AssetId(7),
                    interval: 1.0,
                    lifetime: 2.0,
                    elapsed: 0.0,
                    count: 0,
                },
            );

            // Below the interval: nothing due, but the clock advances.
            assert!(tick_spawners(ctx, 0.5).is_empty());
            // Crossing the interval emits one, carrying the lifetime + template
            // and the spawner's own position.
            let due = tick_spawners(ctx, 0.6);
            assert_eq!(due.len(), 1);
            assert_eq!(due[0].template, AssetId(7));
            assert_eq!(due[0].lifetime, Some(2.0));
            assert_eq!(due[0].transform.position, [1.0, 2.0, 3.0]);
            // A long frame crossing several intervals catches up.
            assert_eq!(tick_spawners(ctx, 2.5).len(), 2);
            assert_eq!(ctx.get::<Spawner>(spawner).unwrap().count, 3);
        });
    }

    #[test]
    fn spawner_with_nonpositive_interval_is_inert() {
        run(|ctx| {
            let spawner = ctx.components.spawn();
            ctx.insert(spawner, Transform::default());
            ctx.insert(
                spawner,
                Spawner {
                    template: AssetId(7),
                    interval: 0.0,
                    lifetime: 0.0,
                    elapsed: 0.0,
                    count: 0,
                },
            );
            assert!(tick_spawners(ctx, 100.0).is_empty());
        });
    }

    #[test]
    fn tick_only_expires_elapsed_lifetimes() {
        run(|ctx| {
            let short = ctx.components.spawn();
            ctx.insert(short, Lifetime { remaining: 0.1 });
            let long = ctx.components.spawn();
            ctx.insert(long, Lifetime { remaining: 5.0 });

            let expired = tick_lifetimes(ctx, 0.2);
            assert_eq!(expired, vec![short], "only the short lifetime expired");
            // The survivor's clock advanced but it is still alive.
            assert_eq!(ctx.get::<Lifetime>(long).unwrap().remaining, 4.8);
        });
    }
}
