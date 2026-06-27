// concinnity-eas/src/mask.rs
//
// A component identity and a bitset over component identities. Each component
// type has a small integer id (0..128); a ComponentMask records a set of them
// in a single u128. Masks describe which components an entity has and which
// components a query requires or excludes, so a query filter reduces to a few
// bitwise ops.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ComponentId(u8);

impl ComponentId {
    // The largest id the 128-bit mask can hold.
    pub const MAX: u8 = 127;

    pub fn new(id: u8) -> ComponentId {
        debug_assert!(id <= Self::MAX, "component id {id} exceeds {}", Self::MAX);
        ComponentId(id)
    }

    pub fn get(self) -> u8 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ComponentMask(u128);

impl ComponentMask {
    pub const EMPTY: ComponentMask = ComponentMask(0);

    pub fn with(id: ComponentId) -> ComponentMask {
        let mut mask = ComponentMask::EMPTY;
        mask.insert(id);
        mask
    }

    pub fn insert(&mut self, id: ComponentId) {
        self.0 |= 1u128 << id.0;
    }

    pub fn remove(&mut self, id: ComponentId) {
        self.0 &= !(1u128 << id.0);
    }

    pub fn contains(self, id: ComponentId) -> bool {
        self.0 & (1u128 << id.0) != 0
    }

    // Whether this mask contains every id in `other` (i.e. is a superset).
    pub fn contains_all(self, other: ComponentMask) -> bool {
        self.0 & other.0 == other.0
    }

    // Whether this mask shares no id with `other`.
    pub fn is_disjoint(self, other: ComponentMask) -> bool {
        self.0 & other.0 == 0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_remove_contains() {
        let a = ComponentId::new(3);
        let b = ComponentId::new(70);
        let mut mask = ComponentMask::EMPTY;
        assert!(mask.is_empty());
        mask.insert(a);
        mask.insert(b);
        assert!(mask.contains(a));
        assert!(mask.contains(b));
        assert!(!mask.contains(ComponentId::new(4)));
        mask.remove(a);
        assert!(!mask.contains(a));
        assert!(mask.contains(b));
    }

    #[test]
    fn with_builds_single_bit_mask() {
        let id = ComponentId::new(127);
        let mask = ComponentMask::with(id);
        assert!(mask.contains(id));
        assert!(!mask.contains(ComponentId::new(0)));
    }

    #[test]
    fn contains_all_is_superset() {
        let mut have = ComponentMask::EMPTY;
        have.insert(ComponentId::new(1));
        have.insert(ComponentId::new(2));
        have.insert(ComponentId::new(3));

        let mut need = ComponentMask::EMPTY;
        need.insert(ComponentId::new(1));
        need.insert(ComponentId::new(3));
        assert!(have.contains_all(need));

        need.insert(ComponentId::new(9));
        assert!(!have.contains_all(need));
        // Every mask contains the empty set.
        assert!(have.contains_all(ComponentMask::EMPTY));
    }

    #[test]
    fn is_disjoint_detects_overlap() {
        let mut a = ComponentMask::EMPTY;
        a.insert(ComponentId::new(1));
        a.insert(ComponentId::new(2));
        let mut b = ComponentMask::EMPTY;
        b.insert(ComponentId::new(3));
        assert!(a.is_disjoint(b));
        b.insert(ComponentId::new(2));
        assert!(!a.is_disjoint(b));
        assert!(a.is_disjoint(ComponentMask::EMPTY));
    }
}
