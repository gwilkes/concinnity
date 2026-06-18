// src/directx/pass_timing.rs
//
// Per-pass GPU timing on D3D12 via TIMESTAMP queries. The whole-frame timer
// lives in slots [0, 1] of each frame's block; one pair per `PassId` follows
// (start at slot 2 + 2*i, end at slot 3 + 2*i). The shared query heap is
// sized for `FRAMES` such blocks; `execute_graph` issues an `EndQuery`
// before and after each pass's `encode_*`, and the resolve at the end of
// the command list copies the whole block into the persistently-mapped
// readback buffer. The CPU reads the previous frame's block at the top of
// `draw_frame` (after the matching fence wait gates the GPU writes) and
// publishes the per-pass microseconds into `RenderStats.pass_times_us`.
//
// Layout reasoning. Keeping the whole-frame pair at the front of each
// block preserves the legacy `gpu_frame_us` indexing: the existing
// readback reads the first u64 pair of the frame's block, exactly as
// before; only the per-frame stride changes. SsaoPrepass / SsaoKernel /
// ParticlesSim are bundled inside their parent encoders (see
// graph_exec.rs) so the sub-pass slots stay zero; the FogFroxel /
// Upscale / Transparent / Raymarch arms are no-ops on DirectX so their
// slots also stay zero. The shared `StatHud.passes_text` helper picks
// the top six non-zero entries by descending microseconds, so zero
// slots are naturally filtered out of the on-screen chip.
//
// Cost. PASS_COUNT = 21, so per frame the heap holds 2 * (21 + 1) = 44
// slots; with FRAMES = 3 in flight that's 132 u64s on the GPU side and
// the same 1056 bytes in the READBACK buffer. The per-pass EndQuery
// pairs add 2 * 21 = 42 extra command-list ops per frame on top of the
// existing 2 whole-frame ops, all of which are coalesced by the driver
// into a single `ResolveQueryData` blit at the end of the list.

use crate::gfx::render_graph::{PASS_COUNT, PassId};

// Per-frame block: [whole_frame_start, whole_frame_end, pass0_start,
// pass0_end, ..., pass(PASS_COUNT-1)_start, pass(PASS_COUNT-1)_end].
// 2 * (PASS_COUNT + 1) u64 slots.
pub const SLOTS_PER_FRAME: usize = 2 * (PASS_COUNT + 1);

// Bytes consumed by one frame's block in the readback buffer. Each slot
// is a u64 timestamp.
pub const FRAME_BLOCK_BYTES: u64 = (SLOTS_PER_FRAME * 8) as u64;

// Slot indices for the whole-frame timestamp pair within the heap. Matches
// the legacy layout (whole-frame still at the first pair of each frame's
// block) so the existing `gpu_frame_us` readback only needs a stride
// adjustment.
pub const fn whole_frame_pair(frame: usize) -> (u32, u32) {
    let base = (frame * SLOTS_PER_FRAME) as u32;
    (base, base + 1)
}

// Slot indices for `pass`'s start + end timestamps within the heap.
pub const fn pass_pair(frame: usize, pass: PassId) -> (u32, u32) {
    let base = (frame * SLOTS_PER_FRAME + 2 + 2 * (pass as usize)) as u32;
    (base, base + 1)
}

// First heap slot the per-frame ResolveQueryData should walk. Pair this
// with `SLOTS_PER_FRAME` as the count.
pub const fn frame_resolve_start(frame: usize) -> u32 {
    (frame * SLOTS_PER_FRAME) as u32
}

// Byte offset into the readback buffer where this frame's block begins.
pub const fn frame_readback_byte_offset(frame: usize) -> u64 {
    (frame * SLOTS_PER_FRAME * 8) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_preserves_legacy_whole_frame_indexing() {
        // Frame 0's whole-frame pair sits at slots 0,1: the legacy
        // layout. The per-frame stride is now SLOTS_PER_FRAME, not 2.
        assert_eq!(whole_frame_pair(0), (0, 1));
        assert_eq!(
            whole_frame_pair(1),
            (SLOTS_PER_FRAME as u32, SLOTS_PER_FRAME as u32 + 1)
        );
        assert_eq!(frame_resolve_start(0), 0);
        assert_eq!(frame_resolve_start(1), SLOTS_PER_FRAME as u32);
        assert_eq!(frame_readback_byte_offset(0), 0);
        assert_eq!(frame_readback_byte_offset(1), FRAME_BLOCK_BYTES);
    }

    #[test]
    fn pass_pair_skips_the_whole_frame_pair() {
        // First pass starts at slot 2 (offset by the whole-frame pair).
        let (a, b) = pass_pair(0, PassId::Cull);
        assert_eq!((a, b), (2, 3));
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
        ] {
            let (s, e) = pass_pair(0, variant);
            assert!(seen.insert(s), "duplicate start slot for {variant:?}");
            assert!(seen.insert(e), "duplicate end slot for {variant:?}");
            assert!((s as usize) < SLOTS_PER_FRAME);
            assert!((e as usize) < SLOTS_PER_FRAME);
        }
    }
}
