// concinnity-eas/src/join.rs
//
// Index from an entity to the components it has and to its row in each
// component's column. A multi-component query uses it to find the entities that
// have a required set of components (and lack an excluded set), then reads each
// component by row without scanning. It is keyed by entity index and kept in
// sync as components are added to or removed from entities.
//
// The index records the full Entity occupying each slot, not just its index, so
// a stale handle (one whose slot has since been despawned and recycled to a new
// generation) resolves to an empty mask and no row rather than aliasing the new
// occupant's components. A new occupant of a recycled index also self-heals:
// the first `set` for it drops any leftover state the previous occupant left
// behind, so a missed `clear` can never leak the previous occupant's components
// into the new one.

use crate::entity::Entity;
use crate::mask::{ComponentId, ComponentMask};

// Sentinel for "this entity has no row in this component's column".
const NO_ROW: u32 = u32::MAX;

#[derive(Default, Debug)]
pub struct JoinIndex {
    // entity index -> the entity currently occupying that index (None = unused).
    // Generation-checked so a recycled index rejects the old occupant's handle.
    occupants: Vec<Option<Entity>>,
    // entity index -> the set of components that entity has.
    masks: Vec<ComponentMask>,
    // component id -> (entity index -> row in that component's column). The
    // outer vec grows per used component id, the inner per entity index.
    rows: Vec<Vec<u32>>,
}

impl JoinIndex {
    pub fn new() -> JoinIndex {
        JoinIndex::default()
    }

    // Record that `entity` has component `id`, stored at `row` in that
    // component's column. If `entity` is a fresh occupant of its index (first
    // use, or a recycled index), the previous occupant's state is dropped first.
    pub fn set(&mut self, entity: Entity, id: ComponentId, row: u32) {
        let index = entity.index() as usize;
        self.grow_to(index);
        if self.occupants[index] != Some(entity) {
            self.reset_index(index);
            self.occupants[index] = Some(entity);
        }
        self.masks[index].insert(id);
        let column = self.row_column(id.get());
        if index >= column.len() {
            column.resize(index + 1, NO_ROW);
        }
        column[index] = row;
    }

    // Forget that `entity` has component `id`. Frees the index slot once the
    // entity has no components left, so the index reads as unused again.
    pub fn clear(&mut self, entity: Entity, id: ComponentId) {
        let index = entity.index() as usize;
        if self.occupants.get(index).copied().flatten() != Some(entity) {
            return;
        }
        self.masks[index].remove(id);
        if let Some(slot) = self
            .rows
            .get_mut(id.get() as usize)
            .and_then(|column| column.get_mut(index))
        {
            *slot = NO_ROW;
        }
        if self.masks[index].is_empty() {
            self.occupants[index] = None;
        }
    }

    // Forget every component of `entity`. Called when the entity is despawned.
    pub fn clear_entity(&mut self, entity: Entity) {
        let index = entity.index() as usize;
        if self.occupants.get(index).copied().flatten() != Some(entity) {
            return;
        }
        self.reset_index(index);
    }

    // The set of components `entity` has. An empty mask for a stale handle.
    pub fn mask(&self, entity: Entity) -> ComponentMask {
        let index = entity.index() as usize;
        if self.occupants.get(index).copied().flatten() != Some(entity) {
            return ComponentMask::EMPTY;
        }
        self.masks
            .get(index)
            .copied()
            .unwrap_or(ComponentMask::EMPTY)
    }

    // The row of `entity` in component `id`'s column, or `None` if it lacks that
    // component (or the handle is stale).
    pub fn row(&self, entity: Entity, id: ComponentId) -> Option<u32> {
        let index = entity.index() as usize;
        if self.occupants.get(index).copied().flatten() != Some(entity) {
            return None;
        }
        let row = self.rows.get(id.get() as usize)?.get(index).copied()?;
        (row != NO_ROW).then_some(row)
    }

    // Whether `entity` has all of `required` and none of `excluded`.
    pub fn matches(
        &self,
        entity: Entity,
        required: ComponentMask,
        excluded: ComponentMask,
    ) -> bool {
        let mask = self.mask(entity);
        mask.contains_all(required) && mask.is_disjoint(excluded)
    }

    // Drop all state recorded for an index, without touching the occupant slot.
    fn reset_index(&mut self, index: usize) {
        if let Some(mask) = self.masks.get_mut(index) {
            *mask = ComponentMask::EMPTY;
        }
        for column in &mut self.rows {
            if let Some(slot) = column.get_mut(index) {
                *slot = NO_ROW;
            }
        }
        if index < self.occupants.len() {
            self.occupants[index] = None;
        }
    }

    // Grow the per-index vectors so `index` is addressable.
    fn grow_to(&mut self, index: usize) {
        if index >= self.occupants.len() {
            self.occupants.resize(index + 1, None);
        }
        if index >= self.masks.len() {
            self.masks.resize(index + 1, ComponentMask::EMPTY);
        }
    }

    fn row_column(&mut self, id: u8) -> &mut Vec<u32> {
        let id = id as usize;
        if id >= self.rows.len() {
            self.rows.resize_with(id + 1, Vec::new);
        }
        &mut self.rows[id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entities;

    fn ids() -> (ComponentId, ComponentId, ComponentId) {
        (
            ComponentId::new(1),
            ComponentId::new(2),
            ComponentId::new(3),
        )
    }

    #[test]
    fn set_records_mask_and_row() {
        let mut entities = Entities::new();
        let e = entities.alloc();
        let (transform, mesh, _) = ids();
        let mut join = JoinIndex::new();
        join.set(e, transform, 0);
        join.set(e, mesh, 7);

        assert!(join.mask(e).contains(transform));
        assert!(join.mask(e).contains(mesh));
        assert_eq!(join.row(e, transform), Some(0));
        assert_eq!(join.row(e, mesh), Some(7));
        // A component the entity does not have.
        assert_eq!(join.row(e, ComponentId::new(9)), None);
    }

    #[test]
    fn clear_removes_one_component() {
        let mut entities = Entities::new();
        let e = entities.alloc();
        let (transform, mesh, _) = ids();
        let mut join = JoinIndex::new();
        join.set(e, transform, 0);
        join.set(e, mesh, 1);

        join.clear(e, mesh);
        assert!(join.mask(e).contains(transform));
        assert!(!join.mask(e).contains(mesh));
        assert_eq!(join.row(e, mesh), None);
        assert_eq!(join.row(e, transform), Some(0));
    }

    #[test]
    fn clear_last_component_frees_the_slot() {
        let mut entities = Entities::new();
        let e = entities.alloc();
        let (transform, _, _) = ids();
        let mut join = JoinIndex::new();
        join.set(e, transform, 0);
        join.clear(e, transform);
        // With no components left the index reads as unused.
        assert!(join.mask(e).is_empty());
        assert_eq!(join.row(e, transform), None);
    }

    #[test]
    fn clear_entity_removes_everything() {
        let mut entities = Entities::new();
        let e = entities.alloc();
        let (transform, mesh, collider) = ids();
        let mut join = JoinIndex::new();
        join.set(e, transform, 0);
        join.set(e, mesh, 1);
        join.set(e, collider, 2);

        join.clear_entity(e);
        assert!(join.mask(e).is_empty());
        assert_eq!(join.row(e, transform), None);
        assert_eq!(join.row(e, mesh), None);
        assert_eq!(join.row(e, collider), None);
    }

    #[test]
    fn matches_required_and_excluded() {
        let mut entities = Entities::new();
        let e = entities.alloc();
        let (transform, mesh, collider) = ids();
        let mut join = JoinIndex::new();
        join.set(e, transform, 0);
        join.set(e, mesh, 1);

        let required = ComponentMask::with(transform);
        let want_mesh = {
            let mut m = ComponentMask::with(transform);
            m.insert(mesh);
            m
        };
        assert!(join.matches(e, required, ComponentMask::with(collider)));
        assert!(join.matches(e, want_mesh, ComponentMask::EMPTY));
        // Excluding a component it has fails the filter.
        assert!(!join.matches(e, required, ComponentMask::with(mesh)));
        // Requiring a component it lacks fails the filter.
        assert!(!join.matches(e, ComponentMask::with(collider), ComponentMask::EMPTY));
    }

    #[test]
    fn distinct_entities_are_independent() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let b = entities.alloc();
        let (transform, mesh, _) = ids();
        let mut join = JoinIndex::new();
        join.set(a, transform, 0);
        join.set(b, mesh, 0);

        assert!(join.mask(a).contains(transform));
        assert!(!join.mask(a).contains(mesh));
        assert!(join.mask(b).contains(mesh));
        assert!(!join.mask(b).contains(transform));
    }

    #[test]
    fn stale_handle_resolves_to_empty_not_the_recycled_occupant() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let (transform, mesh, _) = ids();
        let mut join = JoinIndex::new();
        join.set(a, transform, 0);
        join.set(a, mesh, 5);

        // Recycle a's index without clearing the join (simulating a missed
        // clear): the same index comes back at a new generation.
        entities.despawn(a);
        let b = entities.alloc();
        assert_eq!(a.index(), b.index());
        assert_ne!(a, b);

        // The new occupant's first set self-heals the stale state.
        join.set(b, transform, 0);
        // The new occupant only has what it was given.
        assert!(join.mask(b).contains(transform));
        assert!(!join.mask(b).contains(mesh));
        assert_eq!(join.row(b, mesh), None);
        // The stale handle reads as empty, never aliasing b's components.
        assert!(join.mask(a).is_empty());
        assert_eq!(join.row(a, transform), None);
    }

    #[test]
    fn recycled_index_rejects_old_generation_before_overwrite() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let (transform, _, _) = ids();
        let mut join = JoinIndex::new();
        join.set(a, transform, 0);
        // Recycle the index. Until the new occupant records anything, the join
        // still names `a` as the occupant, but the wrong-generation handle `b`
        // is rejected, so it can never read `a`'s components.
        entities.despawn(a);
        let b = entities.alloc();
        assert_eq!(a.index(), b.index());
        assert!(join.mask(b).is_empty());
        assert_eq!(join.row(b, transform), None);
    }
}
