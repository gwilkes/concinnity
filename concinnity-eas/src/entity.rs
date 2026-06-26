// concinnity-eas/src/entity.rs
//
// Generational entity handle and the allocator that hands them out. An Entity
// is a runtime-only identity for one live instance; it is never serialized. The
// generation makes a handle to a despawned-and-recycled slot detectable, so a
// stale handle resolves to a safe `None` rather than aliasing a new entity.
//
// The allocator supports two ways to mint ids: `alloc` under `&mut self`
// (recycles freed slots), and `reserve` under `&self` (lock-free, fresh ids
// only) for command recording on worker threads. `flush` materializes reserved
// ids into the metadata table before they are looked up.

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};

// Generation 1 is the first valid generation; the NonZeroU32 niche keeps a
// zeroed handle invalid and Option<Entity> at 8 bytes.
const FIRST_GEN: NonZeroU32 = NonZeroU32::MIN;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Entity {
    index: u32,
    generation: NonZeroU32,
}

impl Entity {
    pub(crate) fn new(index: u32, generation: NonZeroU32) -> Entity {
        Entity { index, generation }
    }

    pub fn index(self) -> u32 {
        self.index
    }

    pub fn generation(self) -> u32 {
        self.generation.get()
    }

    // Pack into a single u64 (index in the high half, generation in the low
    // half). Stable representation for an FFI / scripting boundary.
    pub fn to_bits(self) -> u64 {
        ((self.index as u64) << 32) | self.generation.get() as u64
    }

    // Inverse of `to_bits`. Returns `None` when the generation half is zero,
    // which no live handle ever has, so a zeroed or truncated value is rejected
    // instead of forged into a valid-looking entity.
    pub fn from_bits(bits: u64) -> Option<Entity> {
        let generation = NonZeroU32::new(bits as u32)?;
        Some(Entity {
            index: (bits >> 32) as u32,
            generation,
        })
    }
}

#[derive(Debug)]
struct EntityMeta {
    generation: NonZeroU32,
    alive: bool,
}

#[derive(Debug, Default)]
pub struct Entities {
    meta: Vec<EntityMeta>,
    // Recycled indices, available to `alloc`. Reservation never recycles.
    free: Vec<u32>,
    // Next fresh index. Always >= meta.len(); the gap is reserved-but-not-yet
    // materialized fresh ids, realized by `flush`. An atomic so `reserve` can
    // run under `&self` from worker threads.
    next_fresh: AtomicU32,
}

impl Entities {
    pub fn new() -> Entities {
        Entities::default()
    }

    // Allocate an entity, recycling a freed slot when one is available. The
    // recycled slot keeps the generation it was bumped to at despawn, so old
    // handles to it stay invalid.
    pub fn alloc(&mut self) -> Entity {
        if let Some(index) = self.free.pop() {
            let meta = &mut self.meta[index as usize];
            meta.alive = true;
            return Entity::new(index, meta.generation);
        }
        let index = {
            let next = self.next_fresh.get_mut();
            let i = *next;
            *next += 1;
            i
        };
        self.grow_to(index);
        let meta = &mut self.meta[index as usize];
        meta.alive = true;
        Entity::new(index, meta.generation)
    }

    // Reserve a fresh entity id without taking `&mut self`. Lock-free, so it is
    // safe to call from worker threads while recording commands. Reserved ids
    // are always fresh (never recycled) and carry the first generation; call
    // `flush` under `&mut self` before looking them up.
    pub fn reserve(&self) -> Entity {
        let index = self.next_fresh.fetch_add(1, Ordering::Relaxed);
        Entity::new(index, FIRST_GEN)
    }

    // Materialize any reserved-but-unmaterialized ids into the metadata table,
    // marking them alive. Idempotent.
    pub fn flush(&mut self) {
        let high = *self.next_fresh.get_mut();
        if high == 0 {
            return;
        }
        self.grow_to(high - 1);
    }

    // Despawn an entity. Validates the generation, so a stale or already-dead
    // handle is a no-op returning `false`. Bumps the slot's generation and
    // frees the index for reuse.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        self.flush();
        let Some(meta) = self.meta.get_mut(entity.index as usize) else {
            return false;
        };
        if !meta.alive || meta.generation != entity.generation {
            return false;
        }
        meta.alive = false;
        meta.generation = next_generation(meta.generation);
        self.free.push(entity.index);
        true
    }

    // Whether the handle refers to a currently-live entity. Reflects only
    // materialized state; reserved ids appear after `flush`.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.meta
            .get(entity.index as usize)
            .is_some_and(|meta| meta.alive && meta.generation == entity.generation)
    }

    // Number of metadata slots ever allocated (live plus recycled). Not the
    // live count.
    pub fn total_slots(&self) -> usize {
        self.meta.len()
    }

    // Grow the metadata table so `index` is in range. Every index below
    // `next_fresh` was handed out by `alloc` or `reserve` and is live until
    // despawned (the free-list tracks the despawned ones), so a newly
    // materialized slot starts alive at the first generation.
    fn grow_to(&mut self, index: u32) {
        while self.meta.len() as u32 <= index {
            self.meta.push(EntityMeta {
                generation: FIRST_GEN,
                alive: true,
            });
        }
    }
}

// Bump a generation, wrapping past u32::MAX back to the first valid generation
// (skipping 0, which the NonZeroU32 niche forbids). After 2^32 reuses of one
// slot a stale handle can alias a live entity again (generational ABA).
fn next_generation(generation: NonZeroU32) -> NonZeroU32 {
    NonZeroU32::new(generation.get().wrapping_add(1)).unwrap_or(FIRST_GEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_hands_out_distinct_fresh_ids() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let b = entities.alloc();
        assert_ne!(a.index(), b.index());
        assert_eq!(a.generation(), 1);
        assert_eq!(b.generation(), 1);
        assert!(entities.is_alive(a));
        assert!(entities.is_alive(b));
    }

    #[test]
    fn despawn_recycles_index_and_bumps_generation() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        assert!(entities.despawn(a));
        // Index reused, generation advanced.
        let b = entities.alloc();
        assert_eq!(a.index(), b.index());
        assert_eq!(b.generation(), 2);
        // The stale handle no longer resolves.
        assert!(!entities.is_alive(a));
        assert!(entities.is_alive(b));
    }

    #[test]
    fn double_despawn_and_stale_despawn_are_noops() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        assert!(entities.despawn(a));
        assert!(!entities.despawn(a));
        let b = entities.alloc();
        // Despawning with the old generation must not touch the live entity.
        assert!(!entities.despawn(a));
        assert!(entities.is_alive(b));
    }

    #[test]
    fn reserve_then_flush_materializes_live_entities() {
        let mut entities = Entities::new();
        let a = entities.reserve();
        let b = entities.reserve();
        assert_ne!(a.index(), b.index());
        entities.flush();
        assert!(entities.is_alive(a));
        assert!(entities.is_alive(b));
        assert_eq!(entities.total_slots(), 2);
    }

    #[test]
    fn reserve_is_distinct_across_threads() {
        use std::sync::Arc;
        let entities = Arc::new(Entities::new());
        let mut handles = Vec::new();
        for _ in 0..8 {
            let shared = Arc::clone(&entities);
            handles.push(std::thread::spawn(move || {
                (0..1000)
                    .map(|_| shared.reserve().index())
                    .collect::<Vec<_>>()
            }));
        }
        let mut all: Vec<u32> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_unstable();
        let count = all.len();
        all.dedup();
        assert_eq!(all.len(), count, "reserved indices must be unique");
    }

    #[test]
    fn alloc_and_reserve_never_collide() {
        let mut entities = Entities::new();
        let reserved = entities.reserve();
        let allocated = entities.alloc();
        assert_ne!(reserved.index(), allocated.index());
    }

    #[test]
    fn to_bits_round_trips_and_rejects_zero_generation() {
        let mut entities = Entities::new();
        let a = entities.alloc();
        let bits = a.to_bits();
        assert_eq!(Entity::from_bits(bits), Some(a));
        // A zero generation half is never a live handle.
        assert_eq!(Entity::from_bits(0), None);
        assert_eq!(Entity::from_bits(7u64 << 32), None);
    }

    #[test]
    fn generation_wraps_past_max_to_one() {
        let max = NonZeroU32::new(u32::MAX).unwrap();
        assert_eq!(next_generation(max), FIRST_GEN);
        assert_eq!(next_generation(FIRST_GEN).get(), 2);
    }
}
