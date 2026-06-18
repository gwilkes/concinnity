// src/metal/uniforms.rs
//
// repr(C) uniform structs shared by the Metal frame encoder and its passes.
// Each layout must match the corresponding struct in the MSL shader sources.
#![allow(clippy::incompatible_msrv)]

// Per-frame view-projection uniforms pushed at buffer(0) once per frame.
// Shared across all draw calls in a frame. `view` is the standalone view
// matrix used by the vertex shader to compute view-space depth for cascade
// selection in the fragment shader.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ViewUniforms {
    // Combined view-projection matrix (column-major).
    pub(super) vp: [[f32; 4]; 4],
    // Camera view matrix (column-major). Used to compute view-space depth
    // in the vertex shader for shadow cascade selection.
    pub(super) view: [[f32; 4]; 4],
    // Elapsed seconds, available to shaders for animation.
    pub(super) elapsed: f32,
    pub(super) _pad: f32,
    // World-space camera position (packed_float3 in shader, alignment 4).
    pub(super) cam_pos: [f32; 3],
    // Number of mip levels in the bound IBL prefilter cubemap. 0 means
    // "no EnvironmentMap bound": the fragment shader uses this as the IBL
    // enable flag and falls back to a flat ambient placeholder.
    pub(super) prefilter_mip_count: f32,
    // End-padding: MSL rounds struct size up to a multiple of float4x4's 16-byte
    // alignment, so we round explicitly to satisfy Metal validation.
    pub(super) _end_pad: [f32; 2],
}

// Per-draw-call model matrix pushed at buffer(2) before each draw.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ModelUniforms {
    // Model-to-world matrix (column-major).
    pub(super) model: [[f32; 4]; 4],
}

// Per-draw material roughness pushed to the SSR pre-pass fragment at
// buffer(0). Layout matches the `PpMat` struct in the SSR pre-pass MSL.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct SsrPrepassMat {
    // Perceptual roughness `[0, 1]` of this draw's material.
    pub(super) roughness: f32,
    pub(super) _pad: [f32; 3],
}

// Per-frame inputs to the GPU-driven cull kernel, pushed inline at
// the compute encoder's buffer(2). Layout (208 bytes, a multiple of 16) must
// match the `CullUniforms` struct in the cull kernel MSL (`build_cull_pipeline`).
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct CullUniforms {
    // The six frustum planes (left/right/bottom/top/near/far), each
    // `[normal.x, normal.y, normal.z, d]`, extracted CPU-side and already
    // normalised so the kernel's plane test matches `gfx::frustum` exactly.
    pub(super) planes: [[f32; 4]; 6],
    // World-space camera position (packed_float3 in MSL, alignment 4).
    pub(super) cam_pos: [f32; 3],
    // Number of valid `DrawObject` records; kernel threads past it return.
    pub(super) object_count: u32,
    // Previous frame's un-jittered view-projection. The kernel projects each
    // AABB through this so the NDC depths line up with the Hi-Z values the
    // previous frame's main pass produced. `float4x4` lands at offset 112,
    // already 16-aligned, so the layout matches MSL with no padding.
    pub(super) prev_view_proj: [[f32; 4]; 4],
    // Hi-Z mip-0 dimensions in texels. `[1.0, 1.0]` when no Hi-Z is bound.
    pub(super) hiz_size: [f32; 2],
    // Mip levels in the bound Hi-Z texture.
    pub(super) hiz_mip_count: u32,
    // `0` skips the Hi-Z occlusion test (first frame / after a resize, before
    // a valid pyramid exists); `1` runs it.
    pub(super) hiz_enabled: u32,
    // Unified-cull index where the folded skinned records begin (= static +
    // instances). The kernel draws records at or past this through the u16
    // skinned index buffer instead of the static u32 one. Equals `object_count`
    // when no skinned mesh is folded.
    pub(super) skinned_base: u32,
    // Command-slot base offset for the GPU-driven shadow cull: the
    // shadow ICB holds NUM_SHADOW_CASCADES * object_count slots and cascade `c`
    // writes its survivors at `cascade_base + tid` (= c * object_count). The
    // main cull leaves it 0 (writes at `tid`). Trailing `_pad_skin` rounds the
    // struct to 208 bytes so it matches the 16-aligned MSL `CullUniforms`.
    pub(super) cascade_base: u32,
    pub(super) _pad_skin: [u32; 2],
}

// Uniforms pushed to the TAA resolve fragment shader at buffer(0). Layout
// must match the `TaaUniforms` struct in `build_taa_pipeline`'s MSL. 16 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct TaaUniforms {
    // 0 on the first frame / after a resize, 1.0 otherwise.
    pub(super) history_valid: f32,
    pub(super) _pad0: f32,
    pub(super) _pad1: [f32; 2],
}

// Per-frame uniforms for the TAA velocity pre-pass at buffer(0). Layout must
// match `VelUniforms` in `pipeline.rs`'s velocity MSL.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct VelocityUniforms {
    // Jittered current view-projection: drives the rasterised position so
    // the pre-pass covers exactly the same pixels as the main pass.
    pub(super) jittered_vp: [[f32; 4]; 4],
    // Un-jittered current view-projection: keeps the stored motion vector
    // free of the sub-pixel projection jitter.
    pub(super) cur_vp: [[f32; 4]; 4],
    // Un-jittered previous-frame view-projection.
    pub(super) prev_vp: [[f32; 4]; 4],
}

// Per-object model matrices for the velocity / G-buffer pre-pass at buffer(2).
// Layout must match `VelModel` (velocity MSL) and `GbModel`
// (`shaders/gbuffer_prepass.metal`). For a static or skinned object with no
// motion the caller sets `prev == cur`.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct VelocityModelUniforms {
    pub(super) cur_model: [[f32; 4]; 4],
    pub(super) prev_model: [[f32; 4]; 4],
}

// Per-frame view inputs to the unified G-buffer pre-pass at buffer(0). The
// jittered current VP drives the rasterised position (matching the main pass);
// `view` takes the normal + position into view space (where SSR/SSAO/SSGI/RT
// work); the un-jittered cur/prev VPs derive a jitter-free motion vector.
// Layout must match `GBufferView` in `shaders/gbuffer_prepass.metal`. 256 bytes
// (four float4x4, all naturally 16-aligned, no padding).
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct GBufferView {
    pub(super) jittered_vp: [[f32; 4]; 4],
    pub(super) cur_vp: [[f32; 4]; 4],
    pub(super) prev_vp: [[f32; 4]; 4],
    pub(super) view: [[f32; 4]; 4],
}

// Inputs to the auto-exposure compute kernels at buffer(1) (build) and
// buffer(2) (average). Layout must match the `AutoExposureParams` struct in
// `shaders/auto_exposure.metal`. 16 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct AutoExposureParams {
    // Lowest log2(luminance) the histogram covers.
    pub(super) lum_log2_min: f32,
    // Width of the log2(luminance) span the histogram covers (max - min).
    pub(super) lum_log2_range: f32,
    // `HISTOGRAM_BINS / lum_log2_range`. The build kernel multiplies the
    // centred log-luminance by this to derive a bin index.
    pub(super) lum_to_bin_scale: f32,
    pub(super) _pad: f32,
}

// Per-frame view inputs to the projected-decal pass. Layout must match the
// `DecalView` MSL struct in `shaders/decal.metal`. 144 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct DecalView {
    // View-projection matrix used by the main pass (jittered when TAA is on).
    pub(super) vp: [[f32; 4]; 4],
    // Inverse of `vp`. The fragment shader uses it to reconstruct world space
    // from the MSAA depth attachment at each pixel.
    pub(super) inv_vp: [[f32; 4]; 4],
    // HDR target dimensions in pixels: drives the screen→NDC conversion.
    pub(super) viewport: [f32; 2],
    pub(super) _pad: [f32; 2],
}

// Per-decal uniforms pushed before each draw. Layout must match the
// `DecalParams` MSL struct in `shaders/decal.metal`. 160 bytes (two
// float4x4s + a float4 tint + four scalars).
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct DecalParams {
    pub(super) model: [[f32; 4]; 4],
    pub(super) inv_model: [[f32; 4]; 4],
    pub(super) tint: [f32; 4],
    pub(super) fade_pow: f32,
    pub(super) _pad0: f32,
    pub(super) _pad1: f32,
    pub(super) _pad2: f32,
}

// Per-frame view inputs to the particle render pass. Layout must match the
// `ParticleView` MSL struct in `shaders/particle.metal`. 96 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ParticleView {
    // View-projection matrix used by the main pass.
    pub(super) vp: [[f32; 4]; 4],
    // World-space camera right vector: drives the first billboard axis.
    // Packed as `packed_float3` in MSL, so the trailing float of the float4
    // is unused padding.
    pub(super) cam_right: [f32; 3],
    pub(super) _pad0: f32,
    // World-space camera up vector: drives the second billboard axis.
    pub(super) cam_up: [f32; 3],
    pub(super) _pad1: f32,
}

// Per-frame view inputs shared by every draw in the transparent pass (water,
// glass, ...). Bound once at vertex + fragment buffer(5). Layout matches the
// `TransparentView` MSL struct in the transparent shaders. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct TransparentView {
    pub(super) vp: [[f32; 4]; 4],
    pub(super) inv_vp: [[f32; 4]; 4],
    // World-space camera position (xyz). `.w` is ignored by the shader.
    pub(super) camera_pos: [f32; 4],
    // Render-target width / height in pixels: the shader uses this to
    // turn its fragment position into a normalised screen UV.
    pub(super) viewport: [f32; 2],
    // Wall-clock seconds since startup, fed to the Gerstner sum.
    pub(super) time: f32,
    pub(super) _pad: f32,
}

// One Gerstner wave coefficient set, packed for MSL float4 alignment.
// Matches `WaterWave` in `shaders/water.metal`. 32 bytes.
#[derive(Copy, Clone, Default)]
#[repr(C)]
pub(super) struct WaterWaveGpu {
    // `[direction.x, direction.y, amplitude, wavelength]`.
    pub(super) dir_amp_wave: [f32; 4],
    // `[speed, steepness, pad, pad]`.
    pub(super) speed_steep_pad: [f32; 4],
}

// Maximum waves per `WaterParams`. Mirrors `MAX_WATER_WAVES` in the MSL.
pub(super) const WATER_MAX_WAVES: usize = 4;

// Per-surface tunables uploaded once per WaterSurface per frame. Layout
// matches `WaterParams` in `shaders/water.metal`. Vec3-ish fields are
// stored as `[f32; 4]` (with the trailing element unused) so the layout
// is byte-identical to MSL's `float4` regardless of how the MSL
// compiler packs `float3` and adjacent scalars: that packing rule has
// already bitten this struct once.
// 48 + 32 + 32 × WATER_MAX_WAVES = 208 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct WaterParams {
    // `[x, y, z, _]`: world-space surface centre.
    pub(super) centre: [f32; 4],
    // `[r, g, b, _]`: water tint at full depth.
    pub(super) deep_colour: [f32; 4],
    // `[r, g, b, _]`: water tint just above the seabed.
    pub(super) shallow_colour: [f32; 4],
    pub(super) depth_falloff: f32,
    pub(super) foam_width: f32,
    pub(super) foam_intensity: f32,
    pub(super) fresnel_power: f32,
    pub(super) roughness: f32,
    pub(super) refraction_strength: f32,
    pub(super) wave_count: u32,
    // Mip count of the bound IBL prefilter cube; 0 disables the
    // cube-sample path and the shader falls back to a hand-tuned sky tint.
    pub(super) prefilter_mip_count: f32,
    pub(super) waves: [WaterWaveGpu; WATER_MAX_WAVES],
}

// Per-panel tunables for a `GlassPanel`, uploaded once per panel per frame at
// vertex + fragment buffer(6). Vec3-ish fields are `[f32; 4]` so the layout is
// byte-identical to MSL `float4`. Matches `GlassParams` in
// `shaders/glass.metal`. 64 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct GlassParams {
    // `[x, y, z, _]`: world-space panel centre.
    pub(super) centre: [f32; 4],
    // `[nx, ny, nz, _]`: unit panel normal (facing direction).
    pub(super) normal: [f32; 4],
    // `[r, g, b, _]`: colour multiplied into the refracted scene.
    pub(super) tint: [f32; 4],
    // Base alpha at normal incidence.
    pub(super) opacity: f32,
    // Screen-space refraction offset strength.
    pub(super) refraction_strength: f32,
    // Schlick-Fresnel exponent for the grazing-angle rim.
    pub(super) fresnel_power: f32,
    pub(super) _pad: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn view_uniforms_layout_matches_msl() {
        // MSL `ViewUniforms` in default.metal: two float4x4,
        // two scalars, packed_float3 cam_pos + prefilter_mip_count. MSL rounds
        // the struct up to a float4x4 multiple (160): `_end_pad` matches that.
        assert_eq!(size_of::<ViewUniforms>(), 160);
        assert_eq!(offset_of!(ViewUniforms, vp), 0);
        assert_eq!(offset_of!(ViewUniforms, view), 64);
        assert_eq!(offset_of!(ViewUniforms, elapsed), 128);
        assert_eq!(offset_of!(ViewUniforms, _pad), 132);
        assert_eq!(offset_of!(ViewUniforms, cam_pos), 136);
        assert_eq!(offset_of!(ViewUniforms, prefilter_mip_count), 148);
        assert_eq!(offset_of!(ViewUniforms, _end_pad), 152);
        assert_eq!(size_of::<ViewUniforms>() % 16, 0);
    }

    #[test]
    fn model_uniforms_layout_matches_msl() {
        // MSL `ModelUniforms` in default.metal / shadow_map.metal: one float4x4.
        assert_eq!(size_of::<ModelUniforms>(), 64);
        assert_eq!(offset_of!(ModelUniforms, model), 0);
    }

    #[test]
    fn ssr_prepass_mat_layout_matches_msl() {
        // MSL `PpMat` in ssr_prepass.metal: a roughness float padded to 16
        // bytes with plain floats (a float3 would bloat it to 32).
        assert_eq!(size_of::<SsrPrepassMat>(), 16);
        assert_eq!(offset_of!(SsrPrepassMat, roughness), 0);
        assert_eq!(offset_of!(SsrPrepassMat, _pad), 4);
    }

    #[test]
    fn cull_uniforms_layout_matches_msl() {
        // MSL `CullUniforms` in cull.metal: float4 planes[6], packed_float3
        // cam_pos + object_count, then a float4x4 at the 16-aligned offset 112,
        // a float2 + two uints, then skinned_base + cascade_base + 8B pad
        // rounding to 208.
        assert_eq!(size_of::<CullUniforms>(), 208);
        assert_eq!(offset_of!(CullUniforms, planes), 0);
        assert_eq!(offset_of!(CullUniforms, cam_pos), 96);
        assert_eq!(offset_of!(CullUniforms, object_count), 108);
        assert_eq!(offset_of!(CullUniforms, prev_view_proj), 112);
        assert_eq!(offset_of!(CullUniforms, hiz_size), 176);
        assert_eq!(offset_of!(CullUniforms, hiz_mip_count), 184);
        assert_eq!(offset_of!(CullUniforms, hiz_enabled), 188);
        assert_eq!(offset_of!(CullUniforms, skinned_base), 192);
        assert_eq!(offset_of!(CullUniforms, cascade_base), 196);
        assert_eq!(size_of::<CullUniforms>() % 16, 0);
    }

    #[test]
    fn taa_uniforms_layout_matches_msl() {
        // MSL `TaaUniforms` in taa.metal: history_valid + pad to 16 bytes.
        assert_eq!(size_of::<TaaUniforms>(), 16);
        assert_eq!(offset_of!(TaaUniforms, history_valid), 0);
        assert_eq!(offset_of!(TaaUniforms, _pad0), 4);
        assert_eq!(offset_of!(TaaUniforms, _pad1), 8);
    }

    #[test]
    fn velocity_uniforms_layout_matches_msl() {
        // MSL `VelUniforms` in velocity.metal: three float4x4.
        assert_eq!(size_of::<VelocityUniforms>(), 192);
        assert_eq!(offset_of!(VelocityUniforms, jittered_vp), 0);
        assert_eq!(offset_of!(VelocityUniforms, cur_vp), 64);
        assert_eq!(offset_of!(VelocityUniforms, prev_vp), 128);
    }

    #[test]
    fn velocity_model_uniforms_layout_matches_msl() {
        // MSL `VelModel` in velocity.metal / `GbModel` in gbuffer_prepass.metal:
        // two float4x4.
        assert_eq!(size_of::<VelocityModelUniforms>(), 128);
        assert_eq!(offset_of!(VelocityModelUniforms, cur_model), 0);
        assert_eq!(offset_of!(VelocityModelUniforms, prev_model), 64);
    }

    #[test]
    fn gbuffer_view_layout_matches_msl() {
        // MSL `GBufferView` in gbuffer_prepass.metal: four float4x4, all
        // naturally 16-aligned, so the 256-byte layout matches with no padding.
        assert_eq!(size_of::<GBufferView>(), 256);
        assert_eq!(offset_of!(GBufferView, jittered_vp), 0);
        assert_eq!(offset_of!(GBufferView, cur_vp), 64);
        assert_eq!(offset_of!(GBufferView, prev_vp), 128);
        assert_eq!(offset_of!(GBufferView, view), 192);
        assert_eq!(size_of::<GBufferView>() % 16, 0);
    }

    #[test]
    fn auto_exposure_params_layout_matches_msl() {
        // MSL `AutoExposureParams` in auto_exposure.metal: four floats.
        assert_eq!(size_of::<AutoExposureParams>(), 16);
        assert_eq!(offset_of!(AutoExposureParams, lum_log2_min), 0);
        assert_eq!(offset_of!(AutoExposureParams, lum_log2_range), 4);
        assert_eq!(offset_of!(AutoExposureParams, lum_to_bin_scale), 8);
        assert_eq!(offset_of!(AutoExposureParams, _pad), 12);
    }

    #[test]
    fn decal_view_layout_matches_msl() {
        // MSL `DecalView` in decal.metal: two float4x4, a float2 + pad.
        assert_eq!(size_of::<DecalView>(), 144);
        assert_eq!(offset_of!(DecalView, vp), 0);
        assert_eq!(offset_of!(DecalView, inv_vp), 64);
        assert_eq!(offset_of!(DecalView, viewport), 128);
        assert_eq!(offset_of!(DecalView, _pad), 136);
    }

    #[test]
    fn decal_params_layout_matches_msl() {
        // MSL `DecalParams` in decal.metal: two float4x4, a float4 tint, then
        // four scalars.
        assert_eq!(size_of::<DecalParams>(), 160);
        assert_eq!(offset_of!(DecalParams, model), 0);
        assert_eq!(offset_of!(DecalParams, inv_model), 64);
        assert_eq!(offset_of!(DecalParams, tint), 128);
        assert_eq!(offset_of!(DecalParams, fade_pow), 144);
        assert_eq!(offset_of!(DecalParams, _pad0), 148);
        assert_eq!(offset_of!(DecalParams, _pad1), 152);
        assert_eq!(offset_of!(DecalParams, _pad2), 156);
    }

    #[test]
    fn particle_view_layout_matches_msl() {
        // MSL `ParticleView` in particle.metal: float4x4 vp, two
        // packed_float3 + pad billboard axes.
        assert_eq!(size_of::<ParticleView>(), 96);
        assert_eq!(offset_of!(ParticleView, vp), 0);
        assert_eq!(offset_of!(ParticleView, cam_right), 64);
        assert_eq!(offset_of!(ParticleView, _pad0), 76);
        assert_eq!(offset_of!(ParticleView, cam_up), 80);
        assert_eq!(offset_of!(ParticleView, _pad1), 92);
    }

    #[test]
    fn transparent_view_layout_matches_msl() {
        // MSL `TransparentView` in glass.metal (and `WaterView` in water.metal,
        // an identical layout): two float4x4, a float4 camera_pos, float2
        // viewport, time + pad.
        assert_eq!(size_of::<TransparentView>(), 160);
        assert_eq!(offset_of!(TransparentView, vp), 0);
        assert_eq!(offset_of!(TransparentView, inv_vp), 64);
        assert_eq!(offset_of!(TransparentView, camera_pos), 128);
        assert_eq!(offset_of!(TransparentView, viewport), 144);
        assert_eq!(offset_of!(TransparentView, time), 152);
        assert_eq!(offset_of!(TransparentView, _pad), 156);
    }

    #[test]
    fn water_wave_gpu_layout_matches_msl() {
        // MSL `WaterWave` in water.metal: two float4.
        assert_eq!(size_of::<WaterWaveGpu>(), 32);
        assert_eq!(offset_of!(WaterWaveGpu, dir_amp_wave), 0);
        assert_eq!(offset_of!(WaterWaveGpu, speed_steep_pad), 16);
    }

    #[test]
    fn water_params_layout_matches_msl() {
        // MSL `WaterParams` in water.metal: three float4, eight scalars, then
        // the WaterWave array at the 16-aligned offset 80.
        assert_eq!(size_of::<WaterParams>(), 208);
        assert_eq!(offset_of!(WaterParams, centre), 0);
        assert_eq!(offset_of!(WaterParams, deep_colour), 16);
        assert_eq!(offset_of!(WaterParams, shallow_colour), 32);
        assert_eq!(offset_of!(WaterParams, depth_falloff), 48);
        assert_eq!(offset_of!(WaterParams, foam_width), 52);
        assert_eq!(offset_of!(WaterParams, foam_intensity), 56);
        assert_eq!(offset_of!(WaterParams, fresnel_power), 60);
        assert_eq!(offset_of!(WaterParams, roughness), 64);
        assert_eq!(offset_of!(WaterParams, refraction_strength), 68);
        assert_eq!(offset_of!(WaterParams, wave_count), 72);
        assert_eq!(offset_of!(WaterParams, prefilter_mip_count), 76);
        assert_eq!(offset_of!(WaterParams, waves), 80);
        assert_eq!(size_of::<WaterParams>() % 16, 0);
    }

    #[test]
    fn glass_params_layout_matches_msl() {
        // MSL `GlassParams` in glass.metal: three float4, then four scalars.
        assert_eq!(size_of::<GlassParams>(), 64);
        assert_eq!(offset_of!(GlassParams, centre), 0);
        assert_eq!(offset_of!(GlassParams, normal), 16);
        assert_eq!(offset_of!(GlassParams, tint), 32);
        assert_eq!(offset_of!(GlassParams, opacity), 48);
        assert_eq!(offset_of!(GlassParams, refraction_strength), 52);
        assert_eq!(offset_of!(GlassParams, fresnel_power), 56);
        assert_eq!(offset_of!(GlassParams, _pad), 60);
    }
}
