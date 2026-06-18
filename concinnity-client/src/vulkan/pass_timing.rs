// src/vulkan/pass_timing.rs
//
// Per-pass GPU timing on Vulkan via TIMESTAMP queries. The query pool holds one
// per-frame block of `SLOTS_PER_FRAME` slots; the whole-frame timer lives in
// slots [0, 1] of each block and one (start, end) pair per `PassId` follows
// (start at slot 2 + 2*i, end at slot 3 + 2*i). Mirrors directx/pass_timing.rs.
//
// The start buffer resets the whole block and writes the whole-frame start; each
// per-pass command buffer writes its own (start, end) pair around its encode;
// the end buffer writes the whole-frame end. The CPU reads the previous trip's
// block at the top of `draw_frame` (after the matching fence wait gates the GPU
// writes) and publishes the per-pass microseconds into `RenderStats.pass_times_us`.
//
// Vulkan note. Unlike D3D12 (which can pre-write every slot so a pass that did
// not run still reads a value), Vulkan forbids writing a timestamp to a query
// that is already written without an intervening reset. So a pass absent from
// this frame's graph leaves its (reset-but-unwritten) slots `unavailable`; the
// readback uses `WITH_AVAILABILITY` and reports 0 for any slot whose pair is not
// both available. The shared `StatHud.passes_text` then filters the zero slots.
//
// Layout reasoning. Keeping the whole-frame pair at the front of each block lets
// the existing `gpu_frame_us` readback stay the first pair of the frame's block;
// only the per-frame stride changes (from 2 to `SLOTS_PER_FRAME`).

use crate::gfx::render_graph::{PASS_COUNT, PassId};

// Per-frame block: [whole_frame_start, whole_frame_end, pass0_start, pass0_end,
// ..., pass(PASS_COUNT-1)_start, pass(PASS_COUNT-1)_end]. 2 * (PASS_COUNT + 1)
// u64 query slots.
pub(in crate::vulkan) const SLOTS_PER_FRAME: usize = 2 * (PASS_COUNT + 1);

// First query slot of frame `frame`'s block.
pub(in crate::vulkan) const fn frame_block_base(frame: usize) -> u32 {
    (frame * SLOTS_PER_FRAME) as u32
}

// (start, end) query slots for the whole-frame pair of `frame`. Matches the
// legacy layout (whole-frame at the first pair of each block).
pub(in crate::vulkan) const fn whole_frame_pair(frame: usize) -> (u32, u32) {
    let base = frame_block_base(frame);
    (base, base + 1)
}

// (start, end) query slots for `pass` within `frame`'s block.
pub(in crate::vulkan) const fn pass_pair(frame: usize, pass: PassId) -> (u32, u32) {
    let base = frame_block_base(frame) + 2 + 2 * (pass as u32);
    (base, base + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_preserves_legacy_whole_frame_indexing() {
        // Frame 0's whole-frame pair sits at slots 0,1 (the legacy layout); the
        // per-frame stride is now SLOTS_PER_FRAME, not 2.
        assert_eq!(whole_frame_pair(0), (0, 1));
        assert_eq!(
            whole_frame_pair(1),
            (SLOTS_PER_FRAME as u32, SLOTS_PER_FRAME as u32 + 1)
        );
        assert_eq!(frame_block_base(0), 0);
        assert_eq!(frame_block_base(2), 2 * SLOTS_PER_FRAME as u32);
    }

    #[test]
    fn pass_pair_skips_the_whole_frame_pair() {
        // The first pass starts at slot 2 (offset past the whole-frame pair).
        assert_eq!(pass_pair(0, PassId::Cull), (2, 3));
    }

    #[test]
    fn pass_pairs_are_unique_within_a_frame() {
        use std::collections::HashSet;
        let mut seen: HashSet<u32> = HashSet::new();
        // The whole-frame pair owns slots 0, 1.
        seen.insert(0);
        seen.insert(1);
        for variant in [
            PassId::Cull,
            PassId::Shadow,
            PassId::SsrPrepass,
            PassId::SsaoPrepass,
            PassId::SsaoKernel,
            PassId::SsaoBlur,
            PassId::Main,
            PassId::AutoExposure,
            PassId::Decals,
            PassId::Fog,
            PassId::ParticlesSim,
            PassId::ParticlesDraw,
            PassId::SsrResolve,
            PassId::Velocity,
            PassId::TaaResolve,
            PassId::Bloom,
            PassId::Composite,
            PassId::FogFroxel,
            PassId::Upscale,
            PassId::Transparent,
            PassId::Raymarch,
            PassId::HizBuild,
            PassId::Cull2,
            PassId::Main2,
            PassId::Ssgi,
            PassId::RtReflections,
            PassId::GBufferPrepass,
        ] {
            let (s, e) = pass_pair(0, variant);
            assert!(seen.insert(s), "duplicate start slot for {variant:?}");
            assert!(seen.insert(e), "duplicate end slot for {variant:?}");
            assert!((e as usize) < SLOTS_PER_FRAME);
        }
        // Every slot of the block is accounted for (whole-frame pair + one pair
        // per pass).
        assert_eq!(seen.len(), SLOTS_PER_FRAME);
    }
}
