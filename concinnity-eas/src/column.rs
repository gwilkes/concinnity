// concinnity-eas/src/column.rs
//
// A typed component column: the per-type storage primitive the closed-world
// ComponentStorage is built from. It bundles the component data with a
// row-aligned Entity id per row and column-level change/added tick stamps. All
// structural edits go through helpers that keep the data and id vectors the
// same length (checked with a debug assertion).
//
// Column derefs to its data slice, so read paths (iteration, indexing) behave
// like a plain Vec. Mutable access goes through
// `values_mut`, which stamps the change tick because any element may be written.

use std::ops::Deref;

use crate::entity::Entity;
use crate::tick::Tick;

// How a component type is stored. Table is the default dense column; SparseSet
// is opt-in for high-churn types (see SparseColumn). The engine selects the
// kind per component type.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum StorageKind {
    #[default]
    Table,
    SparseSet,
}

#[derive(Debug)]
pub struct Column<T> {
    data: Vec<T>,
    entities: Vec<Entity>,
    changed: Tick,
    added: Tick,
}

impl<T> Default for Column<T> {
    fn default() -> Column<T> {
        Column {
            data: Vec::new(),
            entities: Vec::new(),
            changed: Tick::ZERO,
            added: Tick::ZERO,
        }
    }
}

impl<T> Column<T> {
    pub fn new() -> Column<T> {
        Column::default()
    }

    // The Entity owning each row, aligned with the data slice.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    pub fn changed_tick(&self) -> Tick {
        self.changed
    }

    pub fn added_tick(&self) -> Tick {
        self.added
    }

    // Append a row. Stamps both ticks: the row is newly added and (trivially)
    // changed this tick.
    pub fn push(&mut self, entity: Entity, value: T, tick: Tick) {
        self.data.push(value);
        self.entities.push(entity);
        self.added = tick;
        self.changed = tick;
        debug_assert_eq!(self.data.len(), self.entities.len());
    }

    // Remove row `index`, moving the last row into its place. Returns the
    // removed value. O(1), but reorders the column: a caller that keys on a row
    // position must treat that position as invalidated.
    pub fn swap_remove(&mut self, index: usize, tick: Tick) -> T {
        let value = self.data.swap_remove(index);
        self.entities.swap_remove(index);
        self.changed = tick;
        debug_assert_eq!(self.data.len(), self.entities.len());
        value
    }

    // Take all values, leaving the column empty. Stamps the change tick.
    pub fn drain(&mut self, tick: Tick) -> Vec<T> {
        self.entities.clear();
        self.changed = tick;
        std::mem::take(&mut self.data)
    }

    // Empty the column without returning the values.
    pub fn clear(&mut self, tick: Tick) {
        self.data.clear();
        self.entities.clear();
        self.changed = tick;
    }

    // Mutable access to the values. Stamps the change tick because the caller
    // may write any element.
    pub fn values_mut(&mut self, tick: Tick) -> &mut [T] {
        self.changed = tick;
        &mut self.data
    }

    // Iterate rows paired with their owning entity.
    pub fn iter_with_entities(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.entities.iter().copied().zip(self.data.iter())
    }

    // Whether the column changed since a system's last run, wrap-safe.
    pub fn changed_since(&self, last_run: Tick) -> bool {
        self.changed.is_newer_than(last_run)
    }

    // Whether a row was added since a system's last run, wrap-safe.
    pub fn added_since(&self, last_run: Tick) -> bool {
        self.added.is_newer_than(last_run)
    }
}

impl<T> Deref for Column<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entities;

    fn three() -> (Entities, [Entity; 3]) {
        let mut entities = Entities::new();
        let ids = [entities.alloc(), entities.alloc(), entities.alloc()];
        (entities, ids)
    }

    #[test]
    fn push_keeps_rows_aligned_and_stamps_ticks() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(2));
        assert_eq!(col.len(), 2);
        assert_eq!(&col[..], &[10, 20]);
        assert_eq!(col.entities(), &[ids[0], ids[1]]);
        assert_eq!(col.added_tick(), Tick(2));
        assert_eq!(col.changed_tick(), Tick(2));
    }

    #[test]
    fn swap_remove_reorders_and_returns_value() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        col.push(ids[2], 30, Tick(1));
        let removed = col.swap_remove(0, Tick(5));
        assert_eq!(removed, 10);
        // Last row moved into slot 0; data and entity stay aligned.
        assert_eq!(&col[..], &[30, 20]);
        assert_eq!(col.entities(), &[ids[2], ids[1]]);
        assert_eq!(col.changed_tick(), Tick(5));
    }

    #[test]
    fn drain_empties_and_returns_data() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        col.push(ids[1], 20, Tick(1));
        let drained = col.drain(Tick(9));
        assert_eq!(drained, vec![10, 20]);
        assert!(col.is_empty());
        assert!(col.entities().is_empty());
        assert_eq!(col.changed_tick(), Tick(9));
    }

    #[test]
    fn values_mut_stamps_change() {
        let (_e, ids) = three();
        let mut col: Column<u32> = Column::new();
        col.push(ids[0], 10, Tick(1));
        for v in col.values_mut(Tick(7)) {
            *v += 1;
        }
        assert_eq!(&col[..], &[11]);
        assert!(col.changed_since(Tick(6)));
        assert!(!col.changed_since(Tick(7)));
    }

    #[test]
    fn iter_with_entities_pairs_rows() {
        let (_e, ids) = three();
        let mut col: Column<&str> = Column::new();
        col.push(ids[0], "a", Tick(1));
        col.push(ids[1], "b", Tick(1));
        let pairs: Vec<(Entity, &str)> = col.iter_with_entities().map(|(e, v)| (e, *v)).collect();
        assert_eq!(pairs, vec![(ids[0], "a"), (ids[1], "b")]);
    }
}
