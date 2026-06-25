// src/directx/probe_uniforms.rs
//
// repr(C) reflection-probe uniform structs shared between the DirectX frame
// encoder and the HLSL forward / SSR / RT shaders. Mirrors
// `crate::metal::uniforms::{ProbeUniforms, ProbeSet}` byte-for-byte so the
// backend-agnostic probe placements + the shared cube math (`gfx::reflection_probe`)
// drive an identical reflection on every backend.
//
// CRITICAL -- the 400-byte `ProbeSet` layout (see reflection_probes.md §7). The
// HLSL declaration MUST pad `count` with three SCALAR `uint`s, not a `uint3`: an
// HLSL `uint3` is 16-byte aligned and would push `probes` to offset 32 (struct
// 416 B), so every probe would read shifted by one `float4` -- `box_min.w` (the
// parallax-enable flag) would read the previous probe's `box_max.w` (always 0) and
// box parallax would silently never run. Keeping `_pad: [u32; 3]` here matches the
// three-scalar-uint HLSL layout, leaving `probes` at offset 16 (struct 400 B). The
// layout tests below lock the Rust side; the HLSL side is verified by the D3D12
// debug layer on the first direct-draw bind that length-checks the cbuffer.

// Reflection-probe parallax box. The specular IBL term box-projects the
// reflection vector against [box_min, box_max] (the probe's influence volume) and
// re-anchors the cube sample at the box hit relative to `probe_pos` (the capture
// point), so a static captured cube tracks a moving camera. Three float4s keep
// every field 16-byte aligned, matching the HLSL `float4` layout. `box_min.w` is
// the enabled flag: 0 disables parallax (and signals no baked probe), so the
// shader samples the raw reflection vector.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ProbeUniforms {
    // xyz = influence-box min; w = enabled (1.0 = parallax on, 0.0 = off).
    pub(super) box_min: [f32; 4],
    // xyz = influence-box max; w unused.
    pub(super) box_max: [f32; 4],
    // xyz = probe capture position; w unused.
    pub(super) probe_pos: [f32; 4],
}

impl ProbeUniforms {
    // The "no probe" value: parallax disabled, so the shader samples the raw
    // reflection vector (which, with the probe cube array slot aliasing the sky
    // until a bake, reproduces the pre-probe reflection exactly).
    pub(super) const DISABLED: ProbeUniforms = ProbeUniforms {
        box_min: [0.0; 4],
        box_max: [0.0; 4],
        probe_pos: [0.0; 4],
    };
}

// Maximum reflection probes a frame can bind. The HLSL `MAX_PROBES` constant
// (probe_common.hlsl) and the probe cube array must match this.
pub(super) const MAX_PROBES: usize = 8;

// Auto-seed must never request more probes than a frame can bind, or
// `set_reflection_probes` would truncate and silently drop placements. Checked at
// compile time (mirrors the Metal assertion).
const _: () = assert!(crate::gfx::reflection_probe::AUTO_SEED_BUDGET <= MAX_PROBES);

// The full set of reflection probes bound to the forward / SSR / RT shaders.
// `count` is how many of `probes` are live; the shader blends every probe whose
// influence box covers the surface (a partition-of-unity weight by signed box
// distance), falling back to the nearest when the surface is outside all boxes,
// and samples those slices of the probe cube array. Slices beyond `count` hold the
// sky fallback cube + a `DISABLED` box, so a sample at any index is always valid.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ProbeSet {
    pub(super) count: u32,
    pub(super) _pad: [u32; 3],
    pub(super) probes: [ProbeUniforms; MAX_PROBES],
}

impl ProbeSet {
    pub(super) const EMPTY: ProbeSet = ProbeSet {
        count: 0,
        _pad: [0; 3],
        probes: [ProbeUniforms::DISABLED; MAX_PROBES],
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // ProbeUniforms is three tightly-packed float4s (48 bytes). The HLSL struct
    // is `float4 box_min; float4 box_max; float4 probe_pos;`.
    #[test]
    fn probe_uniforms_layout_matches_hlsl() {
        assert_eq!(size_of::<ProbeUniforms>(), 48);
        assert_eq!(offset_of!(ProbeUniforms, box_min), 0);
        assert_eq!(offset_of!(ProbeUniforms, box_max), 16);
        assert_eq!(offset_of!(ProbeUniforms, probe_pos), 32);
    }

    // ProbeSet is `uint count; uint _pad[3]; ProbeUniforms probes[8];` = 400 bytes,
    // with `probes` at offset 16 (NOT 32 -- the uint3 trap from reflection_probes.md
    // §7 that silently disabled box parallax on Metal).
    #[test]
    fn probe_set_layout_matches_hlsl() {
        assert_eq!(size_of::<ProbeSet>(), 400);
        assert_eq!(offset_of!(ProbeSet, count), 0);
        assert_eq!(
            offset_of!(ProbeSet, probes),
            16,
            "probes must land at offset 16; a HLSL uint3 pad would push it to 32 (struct 416) \
             and silently disable box parallax"
        );
        assert_eq!(MAX_PROBES, 8);
        assert_eq!(ProbeSet::EMPTY.count, 0);
        assert_eq!(ProbeSet::EMPTY.probes.len(), MAX_PROBES);
        // The disabled box has the parallax-enable flag clear.
        assert_eq!(ProbeUniforms::DISABLED.box_min[3], 0.0);
    }
}
