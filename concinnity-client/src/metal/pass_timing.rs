// src/metal/pass_timing.rs
//
// Per-pass GPU timing on Metal via `MTLCounterSampleBuffer`. The whole-frame
// timer in `MtlContext.gpu_time_us` only captures `GPUStartTime` /
// `GPUEndTime`; this module supplements it with one start + end timestamp
// per pass so the profiler overlay can attribute milliseconds to shadow /
// main / SSAO / SSR / etc.
//
// Wiring. Each pass calls
// [`PassTimingResources::attach_render`] (or `attach_compute`) on its
// `MTLRenderPassDescriptor` / `MTLComputePassDescriptor` before creating
// the encoder. The helper writes start- and end-of-encoder sample indices
// into the descriptor's `sampleBufferAttachments[0]` and reserves a unique
// pair of slots for that pass. Multi-encoder passes (the four shadow
// cascades, the bloom mip chain) call `attach_render_first` on the first
// encoder and `attach_render_last` on the last; intermediate encoders
// don't write any timestamps and so don't contribute.
//
// Race avoidance. The sample buffer is per-frame: a ring of
// `FRAMES_IN_FLIGHT` buffers is rotated each frame so the CPU-side resolve
// of frame N-1's buffer never overlaps frame N's GPU writes. The completion
// handler for frame N reads frame-N's buffer and publishes the results into
// `MtlContext.pass_times_us_atomic[..]`; `render_stats()` then copies the
// atomics into the `RenderStats.pass_times_us` array.
//
// Calibration. Apple Silicon's `MTLCommonCounterSetTimestamp` reports
// the GPU clock in mach absolute time units, which on Apple Silicon
// matches nanoseconds 1:1. We treat the raw u64 as nanoseconds and divide
// by 1000 for microseconds. A proper `sampleTimestamps:gpuTimestamp:`
// calibration would be needed for a future Intel Mac path.

#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSRange, NSString};
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLComputePassDescriptor, MTLCounterResultTimestamp,
    MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor, MTLCounterSet, MTLDevice,
    MTLRenderPassDescriptor, MTLStorageMode,
};

// `PassId` and `PASS_COUNT` live in the shared render-graph module so the
// per-pass GPU timer and the future graph executor key off the same enum.
// Adding a new pass = adding a new variant there + a new entry in
// [`PASS_NAMES`] + a bumped `PASS_COUNT`; nothing else needs to change for
// timing to flow. The passes.rs `every_pass_id_round_trips_to_its_name` test
// forces those edits at compile time, and `slot_pair` debug_asserts the index
// at runtime, so a missed registration cannot silently report zero GPU time.
pub use crate::gfx::render_graph::{PASS_COUNT, PASS_NAMES, PassId};

// Frames in flight on Apple Silicon. The sample buffer ring is sized to
// this so frame N's resolve never overlaps frame N+1's GPU writes.
pub const FRAMES_IN_FLIGHT: usize = 3;

// `NSUInteger::MAX` sentinel for an unused sample slot inside a render-pass
// or compute-pass `sampleBufferAttachments[0]`. Metal treats this as
// "don't sample at this stage".
const NO_SAMPLE: usize = usize::MAX;

// One frame's worth of pass timestamps. The sample buffer lives on the GPU
// (private storage); `resolve` reads it back into CPU-visible bytes.
pub struct PassTimingResources {
    buffers: [Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>; FRAMES_IN_FLIGHT],
    // Rotates 0..FRAMES_IN_FLIGHT each frame. The active index picks which
    // buffer the next `attach_*` call binds to.
    frame_slot: usize,
    // Bitmask (1 << PassId index) of the passes attached this frame, cleared by
    // `begin_frame` and set by each `attach_*`. The sample buffer is reused
    // across frames and never cleared, so a pass that ran a few frames ago but
    // is absent this frame leaves stale timestamps in its slot; the resolve
    // zeroes any pass whose bit is clear so the profiler reports 0 for a pass
    // that did not actually run (e.g. every world pass behind an opaque menu).
    attached: std::sync::atomic::AtomicU64,
}

impl PassTimingResources {
    // Build per-frame timestamp sample buffers. Returns `None` if the
    // device does not expose the timestamp counter set (older Apple GPUs
    // or an Intel Mac without the right driver path).
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        // Look up the timestamp counter set among the device's reported sets.
        let sets: Retained<NSArray<ProtocolObject<dyn MTLCounterSet>>> = device.counterSets()?;
        let target_name = unsafe { MTLCommonCounterSetTimestamp };
        let mut timestamp_set: Option<Retained<ProtocolObject<dyn MTLCounterSet>>> = None;
        for set in sets.iter() {
            let name: Retained<NSString> = set.name();
            if &*name == target_name {
                timestamp_set = Some(set);
                break;
            }
        }
        let timestamp_set = timestamp_set?;

        let make_buf = || -> Option<Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>> {
            let desc = MTLCounterSampleBufferDescriptor::new();
            desc.setCounterSet(Some(&timestamp_set));
            // Shared storage so `resolveCounterRange` can copy the
            // CPU-visible result back without an explicit blit. Private
            // would be cheaper on the GPU side but is read-only from the
            // host on Apple Silicon and crashes the resolve.
            desc.setStorageMode(MTLStorageMode::Shared);
            // SAFETY: PASS_COUNT * 2 is a small constant; the descriptor
            // accepts any non-zero sample count up to the device's max.
            unsafe { desc.setSampleCount(PASS_COUNT * 2) };
            device
                .newCounterSampleBufferWithDescriptor_error(&desc)
                .ok()
        };
        let b0 = make_buf()?;
        let b1 = make_buf()?;
        let b2 = make_buf()?;
        Some(Self {
            buffers: [b0, b1, b2],
            frame_slot: 0,
            attached: std::sync::atomic::AtomicU64::new(0),
        })
    }

    // Active sample buffer for the in-progress frame. The render/compute
    // pass descriptors attach to this; the completion handler resolves it
    // after the matching frame retires.
    fn active(&self) -> &ProtocolObject<dyn MTLCounterSampleBuffer> {
        &self.buffers[self.frame_slot]
    }

    // Pick which sample buffer the next set of `attach_*` calls binds to.
    // Call at the top of each frame; the completion handler for the same
    // frame resolves the same buffer.
    pub fn begin_frame(&mut self) -> usize {
        let slot = self.frame_slot;
        self.frame_slot = (self.frame_slot + 1) % FRAMES_IN_FLIGHT;
        // Reset the per-frame attached mask; the passes that run this frame set
        // their bits via `attach_*`, and the resolve zeroes the rest.
        self.attached.store(0, std::sync::atomic::Ordering::Relaxed);
        slot
    }

    // Bitmask of the passes attached since the last `begin_frame`. Captured
    // after the frame's passes are encoded and handed to the completion handler
    // so it can zero the stale slots of passes that did not run.
    pub fn attached_mask(&self) -> u64 {
        self.attached.load(std::sync::atomic::Ordering::Relaxed)
    }

    // Record that `pass` was attached this frame (its slot holds fresh
    // timestamps). `pass as usize < PASS_COUNT <= 64`, so the shift is in range.
    fn mark_attached(&self, pass: PassId) {
        self.attached.fetch_or(
            1u64 << (pass as usize),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    // Buffer handle for an already-issued frame's completion handler.
    // `slot` is what [`begin_frame`] returned for that frame. Returns a
    // fresh `Retained` clone so the closure can outlive the borrow we
    // took on `self`.
    pub fn buffer_for(&self, slot: usize) -> Retained<ProtocolObject<dyn MTLCounterSampleBuffer>> {
        self.buffers[slot].clone()
    }

    // Attach a single-encoder render pass to its start + end slot pair.
    pub fn attach_render(&self, desc: &MTLRenderPassDescriptor, pass: PassId) {
        self.mark_attached(pass);
        let (start, end) = slot_pair(pass);
        // SAFETY: All Metal sample-buffer accessors are marked unsafe by
        // objc2-metal because they take `*mut self`-equivalent ObjC arrays.
        // We hold a unique borrow of `desc` through the call, so no
        // aliasing exists.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            // Sample at start-of-vertex (= encoder start) and
            // end-of-fragment (= encoder end). Intermediate stages stay
            // at NO_SAMPLE.
            entry.setStartOfVertexSampleIndex(start);
            entry.setEndOfVertexSampleIndex(NO_SAMPLE);
            entry.setStartOfFragmentSampleIndex(NO_SAMPLE);
            entry.setEndOfFragmentSampleIndex(end);
        }
    }

    // Attach the FIRST encoder of a multi-encoder pass (e.g. shadow
    // cascade 0, bloom prefilter). Writes only the start sample; the end
    // sample is written by [`attach_render_last`].
    pub fn attach_render_first(&self, desc: &MTLRenderPassDescriptor, pass: PassId) {
        self.mark_attached(pass);
        let (start, _) = slot_pair(pass);
        // SAFETY: see `attach_render`.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            entry.setStartOfVertexSampleIndex(start);
            entry.setEndOfVertexSampleIndex(NO_SAMPLE);
            entry.setStartOfFragmentSampleIndex(NO_SAMPLE);
            entry.setEndOfFragmentSampleIndex(NO_SAMPLE);
        }
    }

    // Attach the LAST encoder of a multi-encoder pass. Writes only the
    // end sample; the start was written by [`attach_render_first`].
    pub fn attach_render_last(&self, desc: &MTLRenderPassDescriptor, pass: PassId) {
        let (_, end) = slot_pair(pass);
        // SAFETY: see `attach_render`.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            entry.setStartOfVertexSampleIndex(NO_SAMPLE);
            entry.setEndOfVertexSampleIndex(NO_SAMPLE);
            entry.setStartOfFragmentSampleIndex(NO_SAMPLE);
            entry.setEndOfFragmentSampleIndex(end);
        }
    }

    // Attach a single-encoder compute pass to its start + end slot pair.
    // Mirrors [`attach_render`] for `MTLComputePassDescriptor`.
    #[allow(dead_code)] // Wiring lands incrementally.
    pub fn attach_compute(&self, desc: &MTLComputePassDescriptor, pass: PassId) {
        self.mark_attached(pass);
        let (start, end) = slot_pair(pass);
        // SAFETY: see `attach_render`.
        unsafe {
            let arr = desc.sampleBufferAttachments();
            let entry = arr.objectAtIndexedSubscript(0);
            entry.setSampleBuffer(Some(self.active()));
            entry.setStartOfEncoderSampleIndex(start);
            entry.setEndOfEncoderSampleIndex(end);
        }
    }
}

// (start_slot, end_slot) in the sample buffer for the given pass. The buffer
// is sized to `PASS_COUNT * 2` slots, so a `PassId` whose index reaches
// PASS_COUNT would address past the end and silently report zero GPU time.
// That only happens if a new pass variant was added without bumping
// PASS_COUNT + registering its PASS_NAMES entry; the debug_assert names the
// miss in dev builds (the passes.rs `every_pass_id_round_trips_to_its_name`
// test is the compile-time guard).
fn slot_pair(pass: PassId) -> (usize, usize) {
    debug_assert!(
        (pass as usize) < PASS_COUNT,
        "PassId {pass:?} (index {}) >= PASS_COUNT {PASS_COUNT}: register it in PASS_NAMES \
         and bump PASS_COUNT",
        pass as usize,
    );
    let base = pass as usize * 2;
    (base, base + 1)
}

// Resolve a frame's sample buffer into per-pass microsecond deltas. Reads
// every pass's start + end slot; an unwritten pair (both zero) reports
// zero microseconds. Returns one entry per pass in `PassKind` order.
//
// Assumes the timestamp counter set reports nanoseconds, which is the
// case on Apple Silicon. A future calibration pass using
// `MTLDevice::sampleTimestamps` could lift this assumption if it ever
// proves wrong.
pub fn resolve(buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>) -> [u32; PASS_COUNT] {
    let range = NSRange::new(0, PASS_COUNT * 2);
    let Some(data) = (unsafe { buffer.resolveCounterRange(range) }) else {
        return [0; PASS_COUNT];
    };
    let bytes = data.len();
    let needed = std::mem::size_of::<MTLCounterResultTimestamp>() * PASS_COUNT * 2;
    if bytes < needed {
        return [0; PASS_COUNT];
    }
    // SAFETY: `data` is `bytes` bytes long and `needed` <= `bytes`. The
    // underlying buffer is `MTLCounterResultTimestamp` (one u64 per slot).
    let timestamps: &[MTLCounterResultTimestamp] = unsafe {
        std::slice::from_raw_parts(
            data.as_bytes_unchecked().as_ptr() as *const MTLCounterResultTimestamp,
            PASS_COUNT * 2,
        )
    };
    let mut out = [0u32; PASS_COUNT];
    for i in 0..PASS_COUNT {
        let start = timestamps[i * 2].timestamp;
        let end = timestamps[i * 2 + 1].timestamp;
        // A pass that was not actually wired this frame leaves both
        // slots at the GPU-default (frequently zero or repeating); guard
        // by requiring monotonic non-zero values.
        if end == 0 || start == 0 || end <= start {
            out[i] = 0;
            continue;
        }
        let ns = end - start;
        out[i] = (ns / 1000).min(u32::MAX as u64) as u32;
    }
    out
}

// Whole-frame GPU span in microseconds: the earliest sampled pass start to the
// latest sampled pass end across this frame's buffer. A single command buffer's
// `GPUStartTime` / `GPUEndTime` covers only its own slice of the frame (the
// frame is split across many command buffers), so summing or reading one buffer
// under-reports the true frame time. The pass timestamps all share one GPU
// clock, so min-start to max-end across them is the real GPU-busy span.
// Returns `None` when no pass wrote a valid timestamp pair.
pub fn frame_span_us(buffer: &ProtocolObject<dyn MTLCounterSampleBuffer>) -> Option<u32> {
    let range = NSRange::new(0, PASS_COUNT * 2);
    let data = unsafe { buffer.resolveCounterRange(range) }?;
    let needed = std::mem::size_of::<MTLCounterResultTimestamp>() * PASS_COUNT * 2;
    if data.len() < needed {
        return None;
    }
    // SAFETY: as in `resolve`, `data` holds `PASS_COUNT * 2` u64 timestamps.
    let timestamps: &[MTLCounterResultTimestamp] = unsafe {
        std::slice::from_raw_parts(
            data.as_bytes_unchecked().as_ptr() as *const MTLCounterResultTimestamp,
            PASS_COUNT * 2,
        )
    };
    let mut min_start = u64::MAX;
    let mut max_end = 0u64;
    for i in 0..PASS_COUNT {
        let start = timestamps[i * 2].timestamp;
        let end = timestamps[i * 2 + 1].timestamp;
        if start == 0 || end == 0 || end <= start {
            continue;
        }
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    if max_end <= min_start {
        return None;
    }
    Some(((max_end - min_start) / 1000).min(u32::MAX as u64) as u32)
}
