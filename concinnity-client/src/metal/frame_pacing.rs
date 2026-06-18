// src/metal/frame_pacing.rs
//
// Frames-in-flight CPU↔GPU pacing. Without it the render loop's only
// backpressure is `currentDrawable()` blocking, so the CPU can queue frames
// arbitrarily far ahead of the GPU: the per-frame transient buffers (object /
// draw-args / joint / bindless-texture / instance) pile up, and the per-frame
// autorelease pool is the only thing keeping VRAM from running away. A
// counting semaphore seeded to the frames-in-flight depth bounds that queue:
// `draw_frame` acquires a slot before encoding, and the frame command buffer's
// completion handler releases it once the GPU has retired the frame, so at most
// `depth` frames are ever in flight. This is the foundation that lets the
// per-frame buffers move from fresh-allocation to ring-buffered reuse.

#![allow(clippy::incompatible_msrv)]

use dispatch2::{DispatchRetained, DispatchSemaphore, DispatchTime};

// Counting semaphore bounding how many frames the CPU may queue ahead of the
// GPU. Seeded to the frames-in-flight depth at construction.
pub(super) struct FrameInFlight {
    semaphore: DispatchRetained<DispatchSemaphore>,
}

impl FrameInFlight {
    // `depth` is the maximum number of frames allowed in flight at once,
    // clamped to ≥1 so a `0` can never deadlock the very first acquire.
    pub(super) fn new(depth: usize) -> Self {
        Self {
            semaphore: DispatchSemaphore::new(depth.max(1) as isize),
        }
    }

    // Block until a frame slot is free, then return an RAII [`FrameSlot`]
    // holding it. Dropping the slot releases it synchronously: the balanced
    // path for a frame abandoned before commit (an early return / error). The
    // normal path calls [`FrameSlot::into_gpu_release`] to hand the single
    // release to the frame's GPU completion handler instead.
    pub(super) fn acquire(&self) -> FrameSlot {
        let semaphore = self.semaphore.clone();
        // DISPATCH_TIME_FOREVER: the GPU always eventually retires an in-flight
        // frame and signals, so steady state cannot block here permanently.
        let _ = semaphore.wait(DispatchTime::FOREVER);
        FrameSlot {
            semaphore: Some(semaphore),
        }
    }

    // Non-destructive single-threaded probe: is at least one slot free right
    // now? Decrements then immediately re-signals so the count is unchanged,
    // leaving the semaphore balanced for disposal (libdispatch traps if a
    // semaphore is deallocated below its seed value). Test-only.
    #[cfg(test)]
    fn has_free_slot(&self) -> bool {
        // Low-level `wait` with a NOW timeout returns 0 if it decremented (a
        // slot was free) or non-zero on timeout (none free). Re-signal on
        // success so the probe leaves no net change.
        if self.semaphore.wait(DispatchTime::NOW) == 0 {
            self.semaphore.signal();
            true
        } else {
            false
        }
    }
}

// RAII holder for one acquired frame-in-flight slot. Releases the slot exactly
// once: on `Drop` for a frame abandoned before commit, or (when
// [`Self::into_gpu_release`] is called) from the GPU completion handler that
// owns the returned handle. The release is GPU-driven on the normal path so
// the semaphore paces the CPU against GPU *retirement* rather than against CPU
// encode completion.
pub(super) struct FrameSlot {
    semaphore: Option<DispatchRetained<DispatchSemaphore>>,
}

impl FrameSlot {
    // Transfer the slot's single release to the caller. The returned handle
    // must be signalled exactly once (from the frame command buffer's
    // completion handler). After this the guard's `Drop` is a no-op, so the
    // slot is never double-released.
    pub(super) fn into_gpu_release(mut self) -> DispatchRetained<DispatchSemaphore> {
        self.semaphore
            .take()
            .expect("FrameSlot::into_gpu_release called exactly once")
    }
}

impl Drop for FrameSlot {
    fn drop(&mut self) {
        if let Some(semaphore) = self.semaphore.take() {
            semaphore.signal();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_consumes_seeded_slots() {
        let fif = FrameInFlight::new(2);
        let _a = fif.acquire();
        let _b = fif.acquire();
        // Both seeded slots are now held; a third acquire would block.
        assert!(!fif.has_free_slot());
        // `_a` / `_b` drop here, releasing both slots back to the seed count so
        // the semaphore disposes balanced.
    }

    #[test]
    fn drop_releases_on_abandon() {
        let fif = FrameInFlight::new(1);
        {
            let _slot = fif.acquire();
            assert!(!fif.has_free_slot(), "slot should be held while alive");
        }
        // The abandoned slot's Drop must have released it back.
        assert!(fif.has_free_slot(), "Drop did not release the slot");
    }

    #[test]
    fn gpu_handoff_releases_exactly_once() {
        let fif = FrameInFlight::new(1);
        let slot = fif.acquire();
        let sem = slot.into_gpu_release();
        // Handing the release to the GPU path must suppress the guard's Drop,
        // so no slot is free until the handler signals.
        assert!(
            !fif.has_free_slot(),
            "into_gpu_release double-released via Drop"
        );
        sem.signal();
        // Exactly one slot came back: take it, then confirm none remain (a
        // double-release would leave a second free slot here).
        let taken = fif.acquire();
        assert!(!fif.has_free_slot(), "slot was released more than once");
        drop(taken);
    }

    #[test]
    fn zero_depth_clamps_to_one() {
        let fif = FrameInFlight::new(0);
        assert!(fif.has_free_slot(), "depth 0 should clamp to 1 usable slot");
        let taken = fif.acquire();
        assert!(!fif.has_free_slot());
        drop(taken);
    }
}
