// src/gfx/graphics_system/spawn.rs
//
// Runtime entity spawn: instantiate a copy of an existing placement at a new
// transform and give it the GPU draw slots, components, and (optionally) a
// Lifetime the copy needs to live in the world. The symmetric counterpart to
// despawn.rs. Driven by SpawnRequest events the GraphicsSystem drains each step
// (see frame.rs), and paired with a Lifetime tick that auto-despawns expired
// instances so their freed draw slots can be recycled by the next spawn.

use crate::assets::{
    GlobalTransform, Lifetime, MeshRenderer, ModelRenderer, RenderHandle, Transform,
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
// instead of a live backend. The new entity is registered under `name` so it
// can later be addressed by name like an authored placement. Returns the new
// entity, or None when the template has no draw slots to copy or a clone fails.
pub(super) fn spawn_from_template(
    ctx: &mut PipelineContext,
    template: Entity,
    name: AssetId,
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
    if let Some(by_name) = ctx.resource_mut::<EntityByName>() {
        by_name.0.insert(name, entity);
    }
    Some(entity)
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
                AssetId(1),
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
                AssetId(2),
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
                AssetId(42),
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
