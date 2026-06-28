// src/gfx/draw_slot.rs
//
// Free-list allocator for backend draw-object slots. A backend appends draw
// objects into a single `Vec` and stores raw indices into it on each entity's
// RenderHandle, so a despawned object's slot cannot be compacted away without
// invalidating every later index. Instead the allocator hands out a vacated
// slot before growing the vec: `retire` pushes a freed index, the next runtime
// spawn pops it. Streamed chunks were the first consumer (one freed chunk's
// slot reused by the next); runtime entity spawn/despawn is the second. All
// three backends (Metal, DirectX, Vulkan) route their draw-slot allocation
// through this.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotAlloc {
    // Reuse this vacated slot: overwrite the existing draw_objects entry.
    Reuse(usize),
    // No free slot was available: append at this index (== the prior length).
    Append(usize),
}

#[derive(Debug, Default)]
pub struct DrawSlotAllocator {
    free: Vec<usize>,
    len: usize,
}

impl DrawSlotAllocator {
    // Start with `len` slots already in use (the draw objects built at init).
    pub fn with_len(len: usize) -> Self {
        Self {
            free: Vec::new(),
            len,
        }
    }

    // Hand out a slot: a vacated one if any is free, else the next new index.
    // The caller writes its draw object at the returned slot and, on Append,
    // grows whatever side tables run parallel to draw_objects.
    pub fn allocate(&mut self) -> SlotAlloc {
        if let Some(slot) = self.free.pop() {
            SlotAlloc::Reuse(slot)
        } else {
            let idx = self.len;
            self.len += 1;
            SlotAlloc::Append(idx)
        }
    }

    // Return a slot to the free list for a later allocate to reuse.
    pub fn free(&mut self, slot: usize) {
        self.free.push(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_past_initial_len_then_reuses_freed_slots() {
        let mut alloc = DrawSlotAllocator::with_len(3);
        // No free slots yet: allocation appends past the initial three.
        assert_eq!(alloc.allocate(), SlotAlloc::Append(3));
        assert_eq!(alloc.allocate(), SlotAlloc::Append(4));

        // Freeing a slot makes the next allocation reuse it instead of growing.
        alloc.free(3);
        assert_eq!(alloc.allocate(), SlotAlloc::Reuse(3));

        // The reuse did not advance the high-water mark: with the free list
        // empty again, allocation resumes appending at 5 (not 6).
        assert_eq!(alloc.allocate(), SlotAlloc::Append(5));
    }

    #[test]
    fn freed_slots_pop_in_lifo_order() {
        let mut alloc = DrawSlotAllocator::with_len(10);
        alloc.free(4);
        alloc.free(7);
        assert_eq!(alloc.allocate(), SlotAlloc::Reuse(7));
        assert_eq!(alloc.allocate(), SlotAlloc::Reuse(4));
        assert_eq!(alloc.allocate(), SlotAlloc::Append(10));
    }
}
