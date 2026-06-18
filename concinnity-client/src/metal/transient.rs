// src/metal/transient.rs
//
// Ring-buffered per-frame upload buffers. The bindless object / draw-args /
// texture-argument buffers (and skinned joint palettes) used to be freshly
// `newBufferWith*`'d every frame: one driver allocation per buffer per frame,
// retained by the committed command buffer until the GPU retired it. With the
// frames-in-flight fence (`metal/frame_pacing.rs`) bounding the CPU to at most
// `frames_in_flight` frames ahead of the GPU, those allocations collapse into a
// small ring of persistent `StorageModeShared` buffers: frame `R` writes ring
// slot `R % depth` and binds it, and because the fence guarantees frame `R −
// depth` has already retired before frame `R` can acquire a slot, the slot the
// CPU is about to overwrite is provably no longer being read by the GPU.
//
// Each slot grows power-of-two on demand (like `ensure_icb_capacity`) and is
// never shrunk, so steady state does zero allocation.

#![allow(clippy::incompatible_msrv)]

use std::collections::VecDeque;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::context::{bytes_of_slice, write_buffer_region};

// A frame-tagged deferred-free pool. A GPU resource an in-flight frame may
// still read cannot be freed or overwritten the instant the CPU replaces it;
// instead the old handle is parked here, tagged with the frame that retired it,
// and dropped only once the frames-in-flight fence guarantees every frame that
// could still reference it has retired on the GPU.
//
// This deliberately does NOT key storage by `frame_index % depth` the way the
// rings above do. A modulo ring is safe only when every slot is rewritten every
// `depth` frames, so the fence's "slot `R − depth` has retired" guarantee
// covers the slot about to be reused. A resource the reflection trace reads
// breaks that assumption: a static / sparsely-moving scene keeps tracing the
// same last-built acceleration structure for many frames without rebuilding, so
// that resource is read by frames the fence does not pair with its writer.
// Deferring the free by retirement frame is the correct general mechanism.
//
// Generic over the payload so the retirement timing is unit-testable without a
// GPU; the real pool stores `Retained` Metal handles.
pub(super) struct RetirePool<T> {
    // (retiring_frame, payload), pushed in nondecreasing frame order.
    pending: VecDeque<(u64, T)>,
}

impl<T> RetirePool<T> {
    pub(super) fn new() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }

    // Park `payload`, keeping it alive until `collect` is called for a frame at
    // least `depth` ahead of `retired_at`. Call when the CPU replaces a GPU
    // resource an in-flight frame may still be reading.
    pub(super) fn push(&mut self, retired_at: u64, payload: T) {
        self.pending.push_back((retired_at, payload));
    }

    // Drop every payload whose retiring frame is at least `depth` frames behind
    // `frame_id` (`retired_at + depth <= frame_id`). The frames-in-flight fence
    // guarantees frame `frame_id − depth`, and every earlier frame, has retired
    // on the GPU, so any frame that could still reference such a payload is
    // done. Entries are pushed in nondecreasing frame order, so draining
    // front-to-back can stop at the first still-live entry.
    pub(super) fn collect(&mut self, frame_id: u64, depth: u64) {
        while let Some(&(retired_at, _)) = self.pending.front() {
            if retired_at.saturating_add(depth) <= frame_id {
                self.pending.pop_front();
            } else {
                break;
            }
        }
    }

    // Number of payloads still held alive. Diagnostics / tests only.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.pending.len()
    }
}

// A small ring of persistent shared-storage buffers, one usable slot per
// frame-in-flight. Hand out a slot's buffer for the current frame via
// [`Self::slot`] (capacity only) or [`Self::write`] (capacity + memcpy).
pub(super) struct TransientRing {
    slots: Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>,
}

impl TransientRing {
    // `depth` is the frames-in-flight count; clamped to ≥1. Buffers are
    // allocated lazily on first use of each slot.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1)).map(|_| None).collect(),
        }
    }

    // Return a cloned handle to `slot`'s buffer, (re)allocating it shared and
    // power-of-two-grown to hold at least `min_len` bytes. Contents are left
    // as-is: use this for buffers an argument encoder fills in place.
    pub(super) fn slot(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        min_len: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let idx = slot % self.slots.len();
        let needs_alloc = match &self.slots[idx] {
            Some(buf) => buf.length() < min_len,
            None => true,
        };
        if needs_alloc {
            // Round up so a slowly-growing draw list does not reallocate every
            // frame; `max(256)` keeps tiny buffers off the page-size floor.
            let cap = min_len.next_power_of_two().max(256);
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to allocate transient ring buffer")?;
            self.slots[idx] = Some(buf);
        }
        Ok(self.slots[idx]
            .as_ref()
            .expect("ring slot was just ensured")
            .clone())
    }

    // Copy `bytes` into `slot`'s buffer (growing it first) and return a cloned
    // handle to bind. The handle is a cheap refcount bump on a buffer the ring
    // owns; the committed command buffer keeps it resident until the GPU is
    // done, and the fence prevents the next writer from racing that read.
    pub(super) fn write(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        bytes: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let buf = self.slot(device, slot, bytes.len().max(1))?;
        write_buffer_region(&buf, 0, bytes)?;
        Ok(buf)
    }
}

// Ring of per-skinned-object joint-palette buffers for one pose stream (the
// current pose, or the previous pose the velocity pre-pass reprojects from).
// Each ring slot holds one buffer per skinned object; `write_all` fills this
// frame's slot and returns cloned handles in object order, matching the shape
// the per-pass encoders bind so they need no change. Current and previous
// poses use separate `JointRing`s because the velocity pass reads both in the
// same frame and they must not alias the same slot.
pub(super) struct JointRing {
    // slots[ring_slot][object] -> palette buffer
    #[allow(clippy::type_complexity)]
    slots: Vec<Vec<Option<Retained<ProtocolObject<dyn MTLBuffer>>>>>,
}

impl JointRing {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1)).map(|_| Vec::new()).collect(),
        }
    }

    // Ensure this frame's ring slot has a buffer per `palettes` entry (each
    // grown to fit its matrices), copy each palette in, and return cloned
    // handles in order. Empty when `palettes` is empty (no skinned meshes).
    pub(super) fn write_all(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        palettes: &[Vec<[[f32; 4]; 4]>],
    ) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, String> {
        let idx = slot % self.slots.len();
        let bufs = &mut self.slots[idx];
        if bufs.len() < palettes.len() {
            bufs.resize_with(palettes.len(), || None);
        }
        let mut out = Vec::with_capacity(palettes.len());
        for (i, mats) in palettes.iter().enumerate() {
            let bytes = bytes_of_slice(mats.as_slice());
            let needs_alloc = match &bufs[i] {
                Some(buf) => buf.length() < bytes.len().max(1),
                None => true,
            };
            if needs_alloc {
                let cap = bytes.len().max(1).next_power_of_two().max(256);
                let buf = device
                    .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                    .ok_or("failed to allocate joint ring buffer")?;
                bufs[i] = Some(buf);
            }
            let buf = bufs[i].as_ref().expect("joint ring slot was just ensured");
            write_buffer_region(buf, 0, bytes)?;
            out.push(buf.clone());
        }
        Ok(out)
    }
}

// Ring of instance-matrix buffers for GPU-instanced clusters. Unlike the
// fixed-purpose rings above, a frame allocates a *variable* number of these
// (one per visible cluster × LOD bucket), so each ring slot keeps a growable
// pool handed out sequentially: `begin_frame` resets the cursor, then each
// `write` returns the next buffer in the slot's pool (growing it as needed).
// The pool persists across frames; the frames-in-flight fence guarantees a
// slot's buffers are no longer GPU-read before that slot is reused.
pub(super) struct InstanceRing {
    slots: Vec<InstanceSlot>,
}

struct InstanceSlot {
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    cursor: usize,
}

impl InstanceRing {
    pub(super) fn new(depth: usize) -> Self {
        Self {
            slots: (0..depth.max(1))
                .map(|_| InstanceSlot {
                    buffers: Vec::new(),
                    cursor: 0,
                })
                .collect(),
        }
    }

    // Reset `slot`'s allocation cursor. Call once at the start of the frame,
    // before the frame's `write` calls.
    pub(super) fn begin_frame(&mut self, slot: usize) {
        let idx = slot % self.slots.len();
        self.slots[idx].cursor = 0;
    }

    // Hand out the next buffer in `slot`'s pool (grown to fit `bytes`), with
    // `bytes` copied in, and return a cloned handle. Sequential within a frame.
    pub(super) fn write(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        slot: usize,
        bytes: &[u8],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
        let idx = slot % self.slots.len();
        let pool = &mut self.slots[idx];
        let i = pool.cursor;
        pool.cursor += 1;
        let needs_alloc = match pool.buffers.get(i) {
            Some(buf) => buf.length() < bytes.len().max(1),
            None => true,
        };
        if needs_alloc {
            let cap = bytes.len().max(1).next_power_of_two().max(256);
            let buf = device
                .newBufferWithLength_options(cap, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to allocate instance ring buffer")?;
            if i < pool.buffers.len() {
                pool.buffers[i] = buf;
            } else {
                pool.buffers.push(buf);
            }
        }
        let buf = &pool.buffers[i];
        write_buffer_region(buf, 0, bytes)?;
        Ok(buf.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::RetirePool;

    #[test]
    fn retire_pool_holds_payloads_for_depth_frames() {
        // depth = 2: a payload retired at frame N must survive until the frame
        // that retires N (N + depth) so any frame that still reads it (≤ N − 1,
        // all retired by N + depth − 1) is provably done.
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.push(0, 100);
        pool.push(1, 101);
        pool.push(2, 102);

        // Frame 1 with depth 2: 0 + 2 = 2 > 1, nothing freed yet.
        pool.collect(1, 2);
        assert_eq!(pool.len(), 3);

        // Frame 2: 0 + 2 = 2 ≤ 2 frees the frame-0 payload; 1 + 2 = 3 > 2 stays.
        pool.collect(2, 2);
        assert_eq!(pool.len(), 2);

        // Frame 3 frees the frame-1 payload.
        pool.collect(3, 2);
        assert_eq!(pool.len(), 1);

        // A jump well past the last tag drains the rest.
        pool.collect(100, 2);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn retire_pool_depth_one_frees_one_frame_later() {
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.push(5, 7);
        // Same frame: 5 + 1 = 6 > 5, still live.
        pool.collect(5, 1);
        assert_eq!(pool.len(), 1);
        // Next frame: 5 + 1 = 6 ≤ 6, freed.
        pool.collect(6, 1);
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn retire_pool_drains_multiple_same_frame_pushes() {
        // More than one payload can retire in the same frame (e.g. a TLAS plus
        // its geometry table). They share a tag and free together.
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.push(4, 1);
        pool.push(4, 2);
        pool.push(4, 3);
        pool.collect(5, 2); // 4 + 2 = 6 > 5, all live
        assert_eq!(pool.len(), 3);
        pool.collect(6, 2); // 4 + 2 = 6 ≤ 6, all freed
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn retire_pool_collect_is_idempotent_and_empty_safe() {
        let mut pool: RetirePool<u32> = RetirePool::new();
        pool.collect(10, 2); // empty pool, no panic
        assert_eq!(pool.len(), 0);
        pool.push(0, 1);
        pool.collect(10, 2);
        pool.collect(10, 2); // second call is a no-op
        assert_eq!(pool.len(), 0);
    }
}
