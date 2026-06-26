// concinnity-eas/src/access.rs
//
// A system's declared data access: which components and resources it reads and
// writes, and whether it must run alone. The scheduler runs two systems
// concurrently only when their accesses do not conflict. A conflict is a
// read-write or write-write overlap on the same component or resource (two
// readers never conflict), or either system being exclusive.
//
// Components and resources use independent id spaces, each a ComponentMask, so
// a component id and a resource id never collide even though both are small
// integers.

use crate::mask::ComponentMask;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Access {
    component_reads: ComponentMask,
    component_writes: ComponentMask,
    resource_reads: ComponentMask,
    resource_writes: ComponentMask,
    exclusive: bool,
}

impl Access {
    pub fn new() -> Access {
        Access::default()
    }

    pub fn reads_components(mut self, components: ComponentMask) -> Access {
        self.component_reads = components;
        self
    }

    pub fn writes_components(mut self, components: ComponentMask) -> Access {
        self.component_writes = components;
        self
    }

    pub fn reads_resources(mut self, resources: ComponentMask) -> Access {
        self.resource_reads = resources;
        self
    }

    pub fn writes_resources(mut self, resources: ComponentMask) -> Access {
        self.resource_writes = resources;
        self
    }

    // Mark the system as exclusive: it conflicts with every other system and so
    // never runs concurrently. Used for a system that touches non-shareable
    // state the access masks do not model (e.g. a main-thread-only backend).
    pub fn exclusive(mut self) -> Access {
        self.exclusive = true;
        self
    }

    pub fn is_exclusive(self) -> bool {
        self.exclusive
    }

    // Whether this system can run concurrently with `other`.
    pub fn conflicts_with(self, other: Access) -> bool {
        self.exclusive
            || other.exclusive
            || overlaps(
                self.component_reads,
                self.component_writes,
                other.component_reads,
                other.component_writes,
            )
            || overlaps(
                self.resource_reads,
                self.resource_writes,
                other.resource_reads,
                other.resource_writes,
            )
    }
}

// A read-write or write-write overlap within one id space. Read-read overlap is
// not a conflict.
fn overlaps(
    a_reads: ComponentMask,
    a_writes: ComponentMask,
    b_reads: ComponentMask,
    b_writes: ComponentMask,
) -> bool {
    !a_writes.is_disjoint(b_reads)
        || !a_writes.is_disjoint(b_writes)
        || !a_reads.is_disjoint(b_writes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mask::{ComponentId, ComponentMask};

    fn mask(ids: &[u8]) -> ComponentMask {
        let mut m = ComponentMask::EMPTY;
        for &id in ids {
            m.insert(ComponentId::new(id));
        }
        m
    }

    #[test]
    fn two_readers_do_not_conflict() {
        let a = Access::new().reads_components(mask(&[1, 2]));
        let b = Access::new().reads_components(mask(&[2, 3]));
        assert!(!a.conflicts_with(b));
        assert!(!b.conflicts_with(a));
    }

    #[test]
    fn read_write_overlap_conflicts() {
        let reader = Access::new().reads_components(mask(&[2]));
        let writer = Access::new().writes_components(mask(&[2]));
        assert!(reader.conflicts_with(writer));
        // Conflict is symmetric.
        assert!(writer.conflicts_with(reader));
    }

    #[test]
    fn write_write_overlap_conflicts() {
        let a = Access::new().writes_components(mask(&[5]));
        let b = Access::new().writes_components(mask(&[5, 6]));
        assert!(a.conflicts_with(b));
    }

    #[test]
    fn disjoint_access_does_not_conflict() {
        let a = Access::new()
            .reads_components(mask(&[1]))
            .writes_components(mask(&[2]));
        let b = Access::new()
            .reads_components(mask(&[3]))
            .writes_components(mask(&[4]));
        assert!(!a.conflicts_with(b));
    }

    #[test]
    fn exclusive_conflicts_with_everything() {
        let solo = Access::new().exclusive();
        let empty = Access::new();
        assert!(solo.conflicts_with(empty));
        assert!(empty.conflicts_with(solo));
        // Even another exclusive.
        assert!(solo.conflicts_with(Access::new().exclusive()));
    }

    #[test]
    fn resource_and_component_spaces_are_independent() {
        // Same numeric id in different spaces must not collide.
        let writes_component_7 = Access::new().writes_components(mask(&[7]));
        let reads_resource_7 = Access::new().reads_resources(mask(&[7]));
        assert!(!writes_component_7.conflicts_with(reads_resource_7));

        // A genuine resource read-write overlap does conflict.
        let writes_resource_7 = Access::new().writes_resources(mask(&[7]));
        assert!(reads_resource_7.conflicts_with(writes_resource_7));
    }

    #[test]
    fn component_conflict_holds_despite_disjoint_resources() {
        let a = Access::new()
            .writes_components(mask(&[1]))
            .reads_resources(mask(&[10]));
        let b = Access::new()
            .reads_components(mask(&[1]))
            .reads_resources(mask(&[11]));
        assert!(a.conflicts_with(b));
    }

    #[test]
    fn empty_access_never_conflicts() {
        let a = Access::new();
        let b = Access::new()
            .reads_components(mask(&[1]))
            .writes_components(mask(&[2]));
        assert!(!a.conflicts_with(b));
        assert!(!b.conflicts_with(a));
    }
}
