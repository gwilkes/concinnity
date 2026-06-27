// src/ecs/decompose.rs
//
// Load-time pass that gives each Prop's entity the per-instance components it
// is composed of: a Transform, a mesh- or model-renderer, an optional collider,
// gameplay tags, scene membership, and parent/child links. Runs once at world
// start, after every Prop has been loaded (so each already owns an Entity) and
// before systems init.
//
// It then drains the Prop column: every renderer and gameplay system reads the
// per-instance components, so the source Props are no longer needed. drain<Prop>
// clears only the Prop component, so each entity survives on its Transform /
// renderer / tag components. Cross-references between placements (a Prop's
// parent) resolve through a name -> Entity index this pass also publishes as a
// resource.

use std::collections::HashMap;

use crate::assets::{
    Children, Collider, Held, Interactable, MeshRenderer, ModelRenderer, Parent, Pickup, Prop,
    SceneMember, Transform,
};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{Entity, PipelineContext};

// Maps a placement's asset identity (its declared name) to the live Entity it
// was loaded into. Built by the decomposition pass so later passes can resolve
// a name reference (a Prop parent, a PropBody owner, an audio emitter target)
// to an Entity without scanning.
#[derive(Debug, Default)]
pub struct EntityByName(pub HashMap<AssetId, Entity>);

impl EntityByName {
    pub fn get(&self, name: AssetId) -> Option<Entity> {
        self.0.get(&name).copied()
    }
}

// Decompose every loaded Prop into per-instance components on its own entity,
// then drain the Prop column.
pub(crate) fn run(ctx: &mut PipelineContext) {
    // Snapshot each Prop with its entity before mutating storage: inserting
    // components while iterating the Prop column would alias the borrow.
    let props: Vec<(Entity, Prop)> = ctx
        .query_with_entity::<Prop>()
        .map(|(entity, prop)| (entity, prop.clone()))
        .collect();
    if props.is_empty() {
        return;
    }

    // Name -> entity, over the full set, so a parent declared after its child
    // still resolves.
    let mut by_name: HashMap<AssetId, Entity> = HashMap::with_capacity(props.len());
    for (entity, prop) in &props {
        by_name.insert(prop.asset_id, *entity);
    }

    // Per-entity components. A Prop's `model` takes precedence over `mesh`,
    // encoded structurally as ModelRenderer-xor-MeshRenderer on the entity.
    for (entity, prop) in &props {
        ctx.insert(
            *entity,
            Transform {
                position: prop.position,
                rotation_deg: prop.rotation_deg,
                scale: prop.scale,
            },
        );
        if let Some(model) = prop.model {
            ctx.insert(
                *entity,
                ModelRenderer {
                    model,
                    cull_distance: prop.cull_distance,
                },
            );
        } else {
            ctx.insert(
                *entity,
                MeshRenderer {
                    mesh: prop.mesh,
                    material: prop.material,
                    texture: prop.texture,
                    cull_distance: prop.cull_distance,
                },
            );
        }
        if let Some(collider) = &prop.collider {
            ctx.insert(*entity, Collider(collider.clone()));
        }
        if prop.interactable {
            ctx.insert(*entity, Interactable);
        }
        if prop.pickup {
            ctx.insert(*entity, Pickup);
        }
        if prop.is_held {
            ctx.insert(*entity, Held);
        }
        if let Some(scene) = prop.scene {
            ctx.insert(*entity, SceneMember(scene));
        }
    }

    // Parent edges resolve once every entity exists; children accumulate so each
    // parent gets a single Children component.
    let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (entity, prop) in &props {
        if let Some(parent_id) = prop.parent
            && let Some(&parent) = by_name.get(&parent_id)
        {
            ctx.insert(*entity, Parent(parent));
            children.entry(parent).or_default().push(*entity);
        }
    }
    for (parent, kids) in children {
        ctx.insert(parent, Children(kids));
    }

    ctx.insert_resource(EntityByName(by_name));

    // Drop the Prop column now that every consumer reads the decomposed
    // components. drain<Prop> clears only the Prop component, so each entity
    // survives on its Transform / renderer / tag components.
    ctx.drain::<Prop>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::{Prop, PropCollider};
    use crate::ecs::World;

    fn prop(id: u32) -> Prop {
        Prop {
            asset_id: AssetId(id),
            ..Default::default()
        }
    }

    #[test]
    fn decomposes_props_onto_their_entities() {
        let mut world = World::new_empty();

        // A model-backed parent placement.
        let mut frame = prop(1);
        frame.model = Some(AssetId(100));
        frame.position = [1.0, 2.0, 3.0];
        world.add_component(frame);

        // A mesh-backed child: material, collider, interactable, scene, parent.
        let mut panel = prop(2);
        panel.mesh = Some(AssetId(101));
        panel.material = Some(AssetId(102));
        panel.collider = Some(PropCollider::default());
        panel.interactable = true;
        panel.scene = Some(AssetId(200));
        panel.parent = Some(AssetId(1));
        panel.position = [4.0, 5.0, 6.0];
        panel.rotation_deg = [0.0, 90.0, 0.0];
        world.add_component(panel);

        world.start().expect("start");

        // The Prop column is drained; the entities survive on their components.
        assert_eq!(world.query::<Prop>().count(), 0);

        // Model placement: ModelRenderer + Transform, no MeshRenderer.
        let models: Vec<_> = world
            .join2::<ModelRenderer, Transform>()
            .map(|(e, m, t)| (e, m.model, t.position))
            .collect();
        assert_eq!(models.len(), 1);
        let (frame_e, model_id, frame_pos) = models[0];
        assert_eq!(model_id, AssetId(100));
        assert_eq!(frame_pos, [1.0, 2.0, 3.0]);

        // Mesh placement: MeshRenderer + Transform, no ModelRenderer.
        let meshes: Vec<_> = world
            .join2::<MeshRenderer, Transform>()
            .map(|(e, m, t)| (e, m.mesh, m.material, t.position, t.rotation_deg))
            .collect();
        assert_eq!(meshes.len(), 1);
        let (panel_e, mesh_id, material_id, panel_pos, panel_rot) = meshes[0];
        assert_eq!(mesh_id, Some(AssetId(101)));
        assert_eq!(material_id, Some(AssetId(102)));
        assert_eq!(panel_pos, [4.0, 5.0, 6.0]);
        assert_eq!(panel_rot, [0.0, 90.0, 0.0]);

        // The child carries its tags and a Parent resolved to the frame entity.
        assert_eq!(world.query::<Collider>().count(), 1);
        assert_eq!(world.query::<Interactable>().count(), 1);
        assert_eq!(world.query::<Pickup>().count(), 0);
        let scene_members: Vec<_> = world
            .join2::<SceneMember, MeshRenderer>()
            .map(|(_, s, _)| s.0)
            .collect();
        assert_eq!(scene_members, vec![AssetId(200)]);
        let parents: Vec<_> = world
            .join2::<Parent, Transform>()
            .map(|(e, p, _)| (e, p.0))
            .collect();
        assert_eq!(parents, vec![(panel_e, frame_e)]);

        // The parent gained a Children list naming the child.
        let kids: Vec<_> = world
            .join2::<Children, ModelRenderer>()
            .map(|(e, c, _)| (e, c.0.clone()))
            .collect();
        assert_eq!(kids, vec![(frame_e, vec![panel_e])]);
    }

    #[test]
    fn forward_parent_reference_resolves() {
        // Child declared BEFORE its parent: the two-pass resolution still links.
        let mut world = World::new_empty();
        let mut child = prop(1);
        child.mesh = Some(AssetId(10));
        child.parent = Some(AssetId(2));
        world.add_component(child);
        let mut parent = prop(2);
        parent.mesh = Some(AssetId(11));
        world.add_component(parent);

        world.start().expect("start");

        let by_parent: Vec<_> = world
            .join2::<Parent, MeshRenderer>()
            .map(|(_, p, m)| (p.0, m.mesh))
            .collect();
        // The child (mesh 10) points at the parent entity.
        assert_eq!(by_parent.len(), 1);
        let (parent_e, child_mesh) = by_parent[0];
        assert_eq!(child_mesh, Some(AssetId(10)));
        // That parent entity is the one holding mesh 11.
        let parent_mesh = world
            .join2::<MeshRenderer, Transform>()
            .find(|(e, _, _)| *e == parent_e)
            .map(|(_, m, _)| m.mesh);
        assert_eq!(parent_mesh, Some(Some(AssetId(11))));
    }

    // The pass drains the Prop column but keeps each entity on its per-instance
    // components.
    #[test]
    fn decomposed_default_drains_prop_keeping_components() {
        let mut world = World::new_empty();
        let mut a = prop(1);
        a.mesh = Some(AssetId(10));
        world.add_component(a);
        let mut b = prop(2);
        b.model = Some(AssetId(20));
        world.add_component(b);

        world.start().expect("start");

        // The Prop column is gone, but both entities survive on their renderers
        // and Transforms.
        assert_eq!(world.query::<Prop>().count(), 0, "Prop column drained");
        assert_eq!(world.query::<Transform>().count(), 2, "Transforms survive");
        assert_eq!(world.query::<MeshRenderer>().count(), 1);
        assert_eq!(world.query::<ModelRenderer>().count(), 1);
    }
}
