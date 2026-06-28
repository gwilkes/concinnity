// src/gfx/skinned_pool.rs
//
// Free pool for pre-reserved skinned instance slots. A skinned mesh that opts
// into runtime spawning (SkinnedMesh.max_instances > 0) has that many hidden
// bind-pose copies appended to the skinned geometry at load. Each copy is its
// own skinned draw object with its own vertex region in the shared skinned
// buffer, which is required because the GPU skin fold writes the deformed
// buffer keyed by global vertex index: two live instances sharing a region
// would clobber each other's pose. This pool tracks, per template, which of
// those copies are currently free so a spawn can claim one and a despawn can
// return it. Slot indices are stable skinned-draw-object indices; nothing is
// compacted, so the per-frame skinned arrays that parallel them stay valid.
//
// Built and consumed by every graphics backend's runtime skinned-spawn path
// (Metal, DirectX, Vulkan). Allow dead code only on a hypothetical build with no
// backend compiled in, where nothing claims a pool slot.
#![cfg_attr(not(any(backend_metal, backend_dx, backend_vk)), allow(dead_code))]

use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct SkinnedInstancePool {
    // template skinned-draw-object index -> its currently free instance slots.
    free: HashMap<usize, Vec<usize>>,
    // instance slot -> the template it belongs to, so `release` returns it to
    // the right pool. Set once at `reserve` and never changed (a copy always
    // belongs to the template it was expanded from).
    owner: HashMap<usize, usize>,
}

impl SkinnedInstancePool {
    pub fn new() -> Self {
        Self::default()
    }

    // Record a pre-reserved instance slot as free and owned by `template`.
    // Called once per expanded copy at load.
    pub fn reserve(&mut self, template: usize, instance: usize) {
        self.owner.insert(instance, template);
        self.free.entry(template).or_default().push(instance);
    }

    // Claim a free instance slot for `template`, or `None` when the reserve is
    // exhausted (more live copies than were pre-reserved).
    pub fn acquire(&mut self, template: usize) -> Option<usize> {
        self.free.get_mut(&template).and_then(|slots| slots.pop())
    }

    // Return a live instance slot to its template's free list. Returns false if
    // the slot was never a pre-reserved instance (e.g. an authored template
    // slot), so the caller can tell a recyclable slot from a fixed one.
    pub fn release(&mut self, instance: usize) -> bool {
        let Some(&template) = self.owner.get(&instance) else {
            return false;
        };
        self.free.entry(template).or_default().push(instance);
        true
    }

    // Total free slots across every template. Surfaced through the debug
    // profile so a probe can watch the pool drain on spawn and refill on
    // despawn, a direct check on the free-list recycle.
    pub fn total_free(&self) -> usize {
        self.free.values().map(Vec::len).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_then_release_recycles_the_same_slot() {
        let mut pool = SkinnedInstancePool::new();
        // Template 0 owns two pre-reserved copies: slots 1 and 2.
        pool.reserve(0, 1);
        pool.reserve(0, 2);
        assert_eq!(pool.total_free(), 2);

        let a = pool.acquire(0).expect("first claim");
        let b = pool.acquire(0).expect("second claim");
        assert!(pool.acquire(0).is_none(), "reserve exhausted");
        assert_eq!(pool.total_free(), 0);

        // Releasing a claimed slot makes it available again, and the next claim
        // hands it back out instead of growing.
        assert!(pool.release(a));
        assert_eq!(pool.total_free(), 1);
        let reused = pool.acquire(0).expect("reuse after release");
        assert_eq!(reused, a, "a freed instance slot is recycled");
        let _ = b;
    }

    #[test]
    fn slots_return_only_to_their_own_template() {
        let mut pool = SkinnedInstancePool::new();
        pool.reserve(0, 10); // template 0
        pool.reserve(5, 20); // template 5
        let s0 = pool.acquire(0).unwrap();
        let s5 = pool.acquire(5).unwrap();
        assert_eq!((s0, s5), (10, 20));
        pool.release(s0);
        pool.release(s5);
        // Each slot went back to its own template's pool, not the other's.
        assert_eq!(pool.acquire(0), Some(10));
        assert_eq!(pool.acquire(5), Some(20));
    }

    #[test]
    fn releasing_an_unknown_slot_is_a_clean_false() {
        let mut pool = SkinnedInstancePool::new();
        pool.reserve(0, 1);
        // Slot 99 was never reserved (e.g. an authored template slot): release
        // reports it is not a pool slot and changes nothing.
        assert!(!pool.release(99));
        assert_eq!(pool.total_free(), 1);
    }

    #[test]
    fn total_free_sums_across_templates() {
        let mut pool = SkinnedInstancePool::new();
        pool.reserve(0, 1);
        pool.reserve(0, 2);
        pool.reserve(3, 4);
        assert_eq!(pool.total_free(), 3);
        pool.acquire(0);
        assert_eq!(pool.total_free(), 2);
    }
}
