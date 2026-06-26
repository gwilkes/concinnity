// concinnity-eas/src/sparse.rs
//
// Sparse-set column: O(1) insert, remove, and lookup keyed by an Entity's
// index, with a dense value array for cache-friendly iteration. This is the
// opt-in storage for high-churn component types (particles, transient gameplay
// tags) where a table column's swap-remove-across-every-column cost hurts. The
// engine selects the storage kind per component type (see StorageKind).
//
// The sparse side is paged so a single large entity index does not allocate one
// giant Vec; only the touched pages exist.

use crate::entity::Entity;

const PAGE: usize = 1024;
// Sentinel for an empty sparse slot. Entity indices never reach u32::MAX in
// practice (it is the reserve high-water ceiling), so it is safe as "no row".
const EMPTY: u32 = u32::MAX;

pub struct SparseColumn<T> {
    dense: Vec<T>,
    dense_entities: Vec<Entity>,
    // entity.index() -> dense row, paged. A missing page or EMPTY slot means the
    // entity is absent.
    pages: Vec<Option<Box<[u32; PAGE]>>>,
}

impl<T> Default for SparseColumn<T> {
    fn default() -> SparseColumn<T> {
        SparseColumn {
            dense: Vec::new(),
            dense_entities: Vec::new(),
            pages: Vec::new(),
        }
    }
}

impl<T> SparseColumn<T> {
    pub fn new() -> SparseColumn<T> {
        SparseColumn::default()
    }

    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    // Insert or overwrite the value for `entity`. Returns the previous value if
    // the entity already had one. Overwriting refreshes the stored generation.
    pub fn insert(&mut self, entity: Entity, value: T) -> Option<T> {
        let row = self.sparse_get(entity.index());
        if row != EMPTY {
            let row = row as usize;
            self.dense_entities[row] = entity;
            return Some(std::mem::replace(&mut self.dense[row], value));
        }
        let row = self.dense.len() as u32;
        self.dense.push(value);
        self.dense_entities.push(entity);
        self.sparse_set(entity.index(), row);
        None
    }

    // Remove and return the value for `entity`. The stored generation must
    // match, so a stale handle removes nothing. Moves the last dense row into
    // the hole and fixes its sparse slot.
    pub fn remove(&mut self, entity: Entity) -> Option<T> {
        let row = self.sparse_get(entity.index());
        if row == EMPTY {
            return None;
        }
        let row = row as usize;
        if self.dense_entities[row] != entity {
            return None;
        }
        let last = self.dense.len() - 1;
        self.dense.swap(row, last);
        self.dense_entities.swap(row, last);
        let value = self.dense.pop();
        self.dense_entities.pop();
        self.sparse_set(entity.index(), EMPTY);
        if row < self.dense.len() {
            // The former last row now lives at `row`; point its sparse slot here.
            let moved = self.dense_entities[row];
            self.sparse_set(moved.index(), row as u32);
        }
        value
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.get(entity).is_some()
    }

    pub fn get(&self, entity: Entity) -> Option<&T> {
        let row = self.sparse_get(entity.index());
        if row == EMPTY {
            return None;
        }
        let row = row as usize;
        (self.dense_entities[row] == entity).then(|| &self.dense[row])
    }

    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let row = self.sparse_get(entity.index());
        if row == EMPTY {
            return None;
        }
        let row = row as usize;
        if self.dense_entities[row] == entity {
            Some(&mut self.dense[row])
        } else {
            None
        }
    }

    pub fn values(&self) -> &[T] {
        &self.dense
    }

    pub fn iter_with_entities(&self) -> impl Iterator<Item = (Entity, &T)> {
        self.dense_entities.iter().copied().zip(self.dense.iter())
    }

    fn sparse_get(&self, index: u32) -> u32 {
        let page = index as usize / PAGE;
        let offset = index as usize % PAGE;
        match self.pages.get(page) {
            Some(Some(page)) => page[offset],
            _ => EMPTY,
        }
    }

    fn sparse_set(&mut self, index: u32, row: u32) {
        let page = index as usize / PAGE;
        let offset = index as usize % PAGE;
        if page >= self.pages.len() {
            self.pages.resize_with(page + 1, || None);
        }
        let page = self.pages[page].get_or_insert_with(|| Box::new([EMPTY; PAGE]));
        page[offset] = row;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::Entities;

    #[test]
    fn insert_get_remove_round_trip() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let b = entities.alloc();
        let mut col: SparseColumn<u32> = SparseColumn::new();

        assert_eq!(col.insert(a, 10), None);
        assert_eq!(col.insert(b, 20), None);
        assert_eq!(col.len(), 2);
        assert_eq!(col.get(a), Some(&10));
        assert_eq!(col.get(b), Some(&20));

        // Overwrite returns the old value.
        assert_eq!(col.insert(a, 11), Some(10));
        assert_eq!(col.get(a), Some(&11));

        assert_eq!(col.remove(a), Some(11));
        assert_eq!(col.get(a), None);
        assert_eq!(col.get(b), Some(&20));
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn remove_rejects_stale_generation() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let mut col: SparseColumn<u32> = SparseColumn::new();
        col.insert(a, 10);
        entities.despawn(a);
        let recycled = entities.alloc();
        assert_eq!(a.index(), recycled.index());
        // The stale handle must not remove the (unrelated) recycled slot's data.
        assert_eq!(col.remove(recycled), None);
        assert_eq!(col.get(a), Some(&10));
    }

    #[test]
    fn get_mut_edits_in_place() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let mut col: SparseColumn<u32> = SparseColumn::new();
        col.insert(a, 10);
        *col.get_mut(a).unwrap() += 5;
        assert_eq!(col.get(a), Some(&15));
    }

    #[test]
    fn spans_multiple_pages() {
        // Two entity indices on different sparse pages (index 0 and PAGE+2),
        // both with the first generation.
        let low = Entity::from_bits(1).unwrap();
        let high = Entity::from_bits(((PAGE as u64 + 2) << 32) | 1).unwrap();
        let mut col: SparseColumn<&str> = SparseColumn::new();
        col.insert(low, "low");
        col.insert(high, "high");
        assert_eq!(col.get(low), Some(&"low"));
        assert_eq!(col.get(high), Some(&"high"));
        assert!(col.contains(high));
        assert_eq!(col.remove(low), Some("low"));
        assert_eq!(col.get(high), Some(&"high"));
    }
}
