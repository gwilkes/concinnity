// concinnity-eas/src/join.rs
//
// Index from an entity to the components it has and to its row in each
// component's column. A multi-component query uses it to find the entities that
// have a required set of components (and lack an excluded set), then reads each
// component by row without scanning. It is keyed by entity index and kept in
// sync as components are added to or removed from entities; clearing an entity
// on despawn is what keeps a reused index from reporting the previous
// occupant's components.

use crate::entity::Entity;
use crate::mask::{ComponentId, ComponentMask};

// Sentinel for "this entity has no row in this component's column".
const NO_ROW: u32 = u32::MAX;

#[derive(Default)]
pub struct JoinIndex {
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
    // component's column.
    pub fn set(&mut self, entity: Entity, id: ComponentId, row: u32) {
        let index = entity.index() as usize;
        if index >= self.masks.len() {
            self.masks.resize(index + 1, ComponentMask::EMPTY);
        }
        self.masks[index].insert(id);
        let column = self.row_column(id.get());
        if index >= column.len() {
            column.resize(index + 1, NO_ROW);
        }
        column[index] = row;
    }

    // Forget that `entity` has component `id`.
    pub fn clear(&mut self, entity: Entity, id: ComponentId) {
        let index = entity.index() as usize;
        if let Some(mask) = self.masks.get_mut(index) {
            mask.remove(id);
        }
        if let Some(slot) = self
            .rows
            .get_mut(id.get() as usize)
            .and_then(|column| column.get_mut(index))
        {
            *slot = NO_ROW;
        }
    }

    // Forget every component of `entity`. Called when the entity is despawned.
    pub fn clear_entity(&mut self, entity: Entity) {
        let index = entity.index() as usize;
        if let Some(mask) = self.masks.get_mut(index) {
            *mask = ComponentMask::EMPTY;
        }
        for column in &mut self.rows {
            if let Some(slot) = column.get_mut(index) {
                *slot = NO_ROW;
            }
        }
    }

    // The set of components `entity` has.
    pub fn mask(&self, entity: Entity) -> ComponentMask {
        self.masks
            .get(entity.index() as usize)
            .copied()
            .unwrap_or(ComponentMask::EMPTY)
    }

    // The row of `entity` in component `id`'s column, or `None` if it lacks that
    // component.
    pub fn row(&self, entity: Entity, id: ComponentId) -> Option<u32> {
        let row = self
            .rows
            .get(id.get() as usize)?
            .get(entity.index() as usize)
            .copied()?;
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
}
