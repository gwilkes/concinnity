// src/directx/init/heap_layout.rs
//
// CBV/SRV/UAV heap slot layout for the DirectX backend. The shader-visible
// descriptor heap is a flat array of fixed slots assigned in a positional
// cascade: each block's base is the previous block's base plus its
// reservation. Keeping the cascade in one function (instead of two dozen
// inline `let x_slot = prev_slot + prev_extra` bindings) lets a unit test
// assert it stays gap-free and that the total matches the created heap size,
// so a stray offset edit fails a test instead of silently misbinding a
// descriptor at shader time (a visual glitch or an out-of-bounds heap write
// with no compile-time signal).
//
// Heap order:
//   [0]                            shadow map array SRV (Texture2DArray)
//   [1]                            IBL irradiance cube SRV
//   [2]                            IBL prefilter cube SRV
//   [object_base_slot..]           per-object (albedo, normal) pairs (2N)
//   [.. +2C]                       per-cluster (albedo, normal) pairs
//   [.. +A]                        text atlas SRVs
//   [hdr_srv_slot]                 HDR scene target SRV (composite pass)
//   [bloom_srv_base_slot..]        bloom mip SRVs
//   [lut_srv_slot]                 3D colour-grading LUT SRV
//   [taa_srv_base_slot..]          (TAA) 2 ping-pong history SRVs
//   [ssao_srv_base_slot..]         (SSAO) ao_raw + ao_blurred
//   [ssao_white_srv_slot]          1x1 white occlusion fallback (always)
//   [ssr_srv_base_slot..]          (SSR) resolve output
//   [decal_depth_srv_slot]         main-depth SRV (decal + glass passes)
//   [decal_srv_base_slot..]        MAX_DECALS per-decal albedo SRVs
//   [chunk_srv_base_slot..+2]      VoxelWorld chunk material (albedo, normal)
//   [skinned_srv_base_slot..]      MAX_SKINNED_OBJECTS (albedo, normal) pairs
//   [particle_srv_base_slot..]     MAX_EMITTERS emitter albedo SRVs
//   [clone_srv_base_slot..]        MAX_CLONE_DRAWS (albedo, normal) pairs
//   [fog_froxel_uav_slot]          froxel-volume UAV
//   [fog_froxel_srv_slot]          froxel-volume SRV
//   [upscale_uav_slot]             temporal-upscale output UAV
//   [upscale_srv_slot]             temporal-upscale output SRV
//   [raymarch_srv_base_slot..+4]   raymarch t0..t3 (shadow, irr, prefilter, scene)
//   [hiz_srv_slot]                 Hi-Z pyramid SRV (covers every mip)
//   [hiz_uav_base_slot..]          HIZ_MAX_MIPS per-mip UAVs
//   [transparent_scene_copy_srv_slot] pre-transparent scene snapshot SRV
//   [ssgi_gi_srv_slot..]           (SSGI) gather-target SRV
//   srv_slots                      total descriptor count (heap size)

use crate::directx::context::{MAX_CLONE_DRAWS, MAX_SKINNED_OBJECTS};
use crate::directx::decal::MAX_DECALS;
use crate::directx::particle::MAX_EMITTERS;

use super::HIZ_MAX_MIPS;

// Per-world feature counts that size the variable-length blocks of the SRV
// heap. The fixed-size blocks (decals, skinned, clones, particles, raymarch,
// Hi-Z, fog, upscale) use module constants and are not parameters.
pub(in crate::directx) struct SrvHeapParams {
    pub n_objects: usize,
    pub n_clusters: usize,
    pub n_atlases: usize,
    pub bloom_count: usize,
    // Per-effect SRV reservations when enabled, else 0: TAA = 2 (history
    // ping-pong), SSAO = 2 (raw + blurred occlusion), SSR = 1 (resolve output).
    // The view normal / depth / roughness / velocity all come from the unified
    // G-buffer pre-pass (`gbuffer_srv_extra`).
    pub taa_srv_extra: usize,
    pub ssao_srv_extra: usize,
    pub ssr_srv_extra: usize,
    // 1 when SSGI is enabled, else 0.
    pub ssgi_srv_extra: usize,
    // 3 (normal+depth, roughness, velocity) when the unified G-buffer pre-pass
    // is active, else 0.
    pub gbuffer_srv_extra: usize,
    // 1 when hardware ray-traced reflections are enabled (the RT output target's
    // SRV), else 0.
    pub rt_output_srv_extra: usize,
    // Flat deduplicated bindless pool sizes: one SRV per distinct albedo-pool
    // texture (incl. emissive / ORM maps) followed by one per distinct normal
    // map. The bindless main pass and the RT hit shader address this region by a
    // flat index (`albedo = texture_slot`, `normal = albedo_count + normal_slot`),
    // mirroring Vulkan/Metal. `albedo_count` is the albedo resource count
    // (>= 1: a 1x1 white fallback stands in when no albedo textures exist);
    // `normal_count` includes the slot-0 flat-normal fallback.
    pub albedo_count: usize,
    pub normal_count: usize,
}

// Resolved slot indices into the CBV/SRV/UAV heap. Field order matches the
// heap order documented above; `srv_slots` is the total descriptor count the
// heap is created with.
pub(in crate::directx) struct SrvHeapLayout {
    pub object_base_slot: usize,
    pub hdr_srv_slot: usize,
    pub bloom_srv_base_slot: usize,
    pub lut_srv_slot: usize,
    pub taa_srv_base_slot: usize,
    pub ssao_srv_base_slot: usize,
    pub ssao_white_srv_slot: usize,
    pub ssr_srv_base_slot: usize,
    pub decal_depth_srv_slot: usize,
    pub decal_srv_base_slot: usize,
    pub chunk_srv_base_slot: usize,
    pub skinned_srv_base_slot: usize,
    pub particle_srv_base_slot: usize,
    pub clone_srv_base_slot: usize,
    pub fog_froxel_uav_slot: usize,
    pub fog_froxel_srv_slot: usize,
    pub upscale_uav_slot: usize,
    pub upscale_srv_slot: usize,
    pub raymarch_srv_base_slot: usize,
    pub hiz_srv_slot: usize,
    pub hiz_uav_base_slot: usize,
    pub transparent_scene_copy_srv_slot: usize,
    pub ssgi_gi_srv_slot: usize,
    pub gbuffer_srv_base_slot: usize,
    pub rt_output_srv_slot: usize,
    pub flat_pool_base_slot: usize,
    pub srv_slots: usize,
}

// The three global SRVs (shadow array, IBL irradiance, IBL prefilter) occupy
// slots [0, 3); the first per-world block starts here.
const GLOBAL_SRV_COUNT: usize = 3;

impl SrvHeapLayout {
    pub(in crate::directx) fn compute(p: &SrvHeapParams) -> Self {
        let object_base_slot = GLOBAL_SRV_COUNT;
        // Per-object + per-cluster (albedo, normal) pairs, then text atlases.
        // `n_atlases.max(1)` reserves one slot even with no atlas so the HDR
        // SRV that follows always lands at a stable offset.
        let hdr_srv_slot =
            object_base_slot + p.n_objects * 2 + p.n_clusters * 2 + p.n_atlases.max(1);
        // The composite pass binds {HDR, bloom mip 0} as one contiguous
        // 2-descriptor table, so bloom mip 0 sits right after the HDR SRV.
        let bloom_srv_base_slot = hdr_srv_slot + 1;
        let lut_srv_slot = bloom_srv_base_slot + p.bloom_count;
        let taa_srv_base_slot = lut_srv_slot + 1;
        let ssao_srv_base_slot = taa_srv_base_slot + p.taa_srv_extra;
        // The white fallback always sits one slot past the SSAO block (present
        // whether SSAO is on or off) so the main pass can bind a pass-through
        // occlusion when SSAO is disabled.
        let ssao_white_srv_slot = ssao_srv_base_slot + p.ssao_srv_extra;
        let ssr_srv_base_slot = ssao_white_srv_slot + 1;
        let decal_depth_srv_slot = ssr_srv_base_slot + p.ssr_srv_extra;
        let decal_srv_base_slot = decal_depth_srv_slot + 1;
        let chunk_srv_base_slot = decal_srv_base_slot + MAX_DECALS;
        let skinned_srv_base_slot = chunk_srv_base_slot + 2;
        let particle_srv_base_slot = skinned_srv_base_slot + MAX_SKINNED_OBJECTS * 2;
        let clone_srv_base_slot = particle_srv_base_slot + MAX_EMITTERS;
        let fog_froxel_uav_slot = clone_srv_base_slot + MAX_CLONE_DRAWS * 2;
        let fog_froxel_srv_slot = fog_froxel_uav_slot + 1;
        let upscale_uav_slot = fog_froxel_srv_slot + 1;
        let upscale_srv_slot = upscale_uav_slot + 1;
        let raymarch_srv_base_slot = upscale_srv_slot + 1;
        let hiz_srv_slot = raymarch_srv_base_slot + 4;
        let hiz_uav_base_slot = hiz_srv_slot + 1;
        let transparent_scene_copy_srv_slot = hiz_uav_base_slot + HIZ_MAX_MIPS;
        let ssgi_gi_srv_slot = transparent_scene_copy_srv_slot + 1;
        // Unified G-buffer SRVs (normal+depth, roughness, velocity). 3 slots
        // when any screen-space consumer drives the pre-pass, else 0.
        let gbuffer_srv_base_slot = ssgi_gi_srv_slot + p.ssgi_srv_extra;
        // RT-reflection output SRV: one slot at the heap tail when RT is on.
        let rt_output_srv_slot = gbuffer_srv_base_slot + p.gbuffer_srv_extra;
        // Flat deduplicated bindless pool: [albedo SRVs..] ++ [normal SRVs..].
        // The bindless main pass and the RT hit shader bind their unbounded pool
        // table base here and index it by a flat slot.
        let flat_pool_base_slot = rt_output_srv_slot + p.rt_output_srv_extra;
        let srv_slots = flat_pool_base_slot + p.albedo_count + p.normal_count;
        Self {
            object_base_slot,
            hdr_srv_slot,
            bloom_srv_base_slot,
            lut_srv_slot,
            taa_srv_base_slot,
            ssao_srv_base_slot,
            ssao_white_srv_slot,
            ssr_srv_base_slot,
            decal_depth_srv_slot,
            decal_srv_base_slot,
            chunk_srv_base_slot,
            skinned_srv_base_slot,
            particle_srv_base_slot,
            clone_srv_base_slot,
            fog_froxel_uav_slot,
            fog_froxel_srv_slot,
            upscale_uav_slot,
            upscale_srv_slot,
            raymarch_srv_base_slot,
            hiz_srv_slot,
            hiz_uav_base_slot,
            transparent_scene_copy_srv_slot,
            ssgi_gi_srv_slot,
            gbuffer_srv_base_slot,
            rt_output_srv_slot,
            flat_pool_base_slot,
            srv_slots,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Walk the cascade for a given feature set: pair each block's reported
    // base with the size that block is independently known to occupy, then
    // assert every base equals the running total of all earlier reservations
    // (gap-free, no overlap) and that `srv_slots` covers the whole chain.
    //
    // The reservations here are derived independently of `compute`'s
    // arithmetic, so an offset slip in `compute` (a `+ 2` where `+ 1` was
    // meant, or a block sized off the wrong constant) makes a base disagree
    // with the running total and fails the assert.
    fn assert_gap_free(p: &SrvHeapParams) {
        let l = SrvHeapLayout::compute(p);
        let blocks: [(usize, usize); 26] = [
            (
                l.object_base_slot,
                p.n_objects * 2 + p.n_clusters * 2 + p.n_atlases.max(1),
            ),
            (l.hdr_srv_slot, 1),
            (l.bloom_srv_base_slot, p.bloom_count),
            (l.lut_srv_slot, 1),
            (l.taa_srv_base_slot, p.taa_srv_extra),
            (l.ssao_srv_base_slot, p.ssao_srv_extra),
            (l.ssao_white_srv_slot, 1),
            (l.ssr_srv_base_slot, p.ssr_srv_extra),
            (l.decal_depth_srv_slot, 1),
            (l.decal_srv_base_slot, MAX_DECALS),
            (l.chunk_srv_base_slot, 2),
            (l.skinned_srv_base_slot, MAX_SKINNED_OBJECTS * 2),
            (l.particle_srv_base_slot, MAX_EMITTERS),
            (l.clone_srv_base_slot, MAX_CLONE_DRAWS * 2),
            (l.fog_froxel_uav_slot, 1),
            (l.fog_froxel_srv_slot, 1),
            (l.upscale_uav_slot, 1),
            (l.upscale_srv_slot, 1),
            (l.raymarch_srv_base_slot, 4),
            (l.hiz_srv_slot, 1),
            (l.hiz_uav_base_slot, HIZ_MAX_MIPS),
            (l.transparent_scene_copy_srv_slot, 1),
            (l.ssgi_gi_srv_slot, p.ssgi_srv_extra),
            (l.gbuffer_srv_base_slot, p.gbuffer_srv_extra),
            (l.rt_output_srv_slot, p.rt_output_srv_extra),
            (l.flat_pool_base_slot, p.albedo_count + p.normal_count),
        ];
        let mut expected_base = GLOBAL_SRV_COUNT;
        for (i, (base, count)) in blocks.iter().enumerate() {
            assert_eq!(
                *base, expected_base,
                "block {i} base {base} should sit at running total {expected_base}",
            );
            expected_base += count;
        }
        assert_eq!(
            l.srv_slots, expected_base,
            "srv_slots must cover every block exactly",
        );
        // The heap always reserves at least the three global SRVs.
        assert!(l.srv_slots >= GLOBAL_SRV_COUNT);
    }

    #[test]
    fn layout_gap_free_all_features_on() {
        assert_gap_free(&SrvHeapParams {
            n_objects: 7,
            n_clusters: 3,
            n_atlases: 2,
            bloom_count: 6,
            taa_srv_extra: 2,
            ssao_srv_extra: 2,
            ssr_srv_extra: 1,
            ssgi_srv_extra: 1,
            gbuffer_srv_extra: 3,
            rt_output_srv_extra: 1,
            albedo_count: 9,
            normal_count: 4,
        });
    }

    #[test]
    fn layout_gap_free_all_features_off() {
        assert_gap_free(&SrvHeapParams {
            n_objects: 0,
            n_clusters: 0,
            n_atlases: 0,
            bloom_count: 0,
            taa_srv_extra: 0,
            ssao_srv_extra: 0,
            ssr_srv_extra: 0,
            ssgi_srv_extra: 0,
            gbuffer_srv_extra: 0,
            rt_output_srv_extra: 0,
            albedo_count: 1,
            normal_count: 1,
        });
    }

    #[test]
    fn layout_gap_free_mixed_features() {
        assert_gap_free(&SrvHeapParams {
            n_objects: 100,
            n_clusters: 0,
            n_atlases: 1,
            bloom_count: 5,
            taa_srv_extra: 2,
            ssao_srv_extra: 0,
            ssr_srv_extra: 1,
            ssgi_srv_extra: 0,
            gbuffer_srv_extra: 3,
            rt_output_srv_extra: 1,
            albedo_count: 50,
            normal_count: 12,
        });
    }

    // The per-world blocks must start past the three fixed global SRVs
    // regardless of feature set, so slot 0/1/2 are never reused.
    #[test]
    fn first_block_clears_the_global_srvs() {
        let l = SrvHeapLayout::compute(&SrvHeapParams {
            n_objects: 0,
            n_clusters: 0,
            n_atlases: 0,
            bloom_count: 0,
            taa_srv_extra: 0,
            ssao_srv_extra: 0,
            ssr_srv_extra: 0,
            ssgi_srv_extra: 0,
            gbuffer_srv_extra: 0,
            rt_output_srv_extra: 0,
            albedo_count: 1,
            normal_count: 1,
        });
        assert_eq!(l.object_base_slot, GLOBAL_SRV_COUNT);
        assert!(l.hdr_srv_slot >= GLOBAL_SRV_COUNT);
    }
}
