// src/metal/hiz.rs
//
// Hi-Z (depth-mip pyramid) build pass used by the GPU-cull compute kernel for
// occlusion culling. Each frame, after the main depth buffer has been written
// by the graph, we reduce it into a `R32Float` mip chain (MAX reduction:
// standard depth, so larger = farther). The *next* frame's `Cull` kernel
// projects each `DrawObject` AABB through the previous frame's un-jittered
// view-projection, picks the Hi-Z mip whose texels are ~the size of the
// projected rect, 4-tap-samples the max occluder depth, and culls the AABB when
// its nearest projected NDC depth is strictly behind. Mirrors the DirectX
// implementation in `directx/hiz.rs` + `directx/shaders/hiz_build.hlsl`.
//
// Two compute kernels share `shaders/hiz_build.metal`:
//
//   * `hiz_init_msaa`: reduce the MSAA main-depth resource into mip 0, taking
//                      the MAX over every sample so the result is conservative.
//   * `hiz_downsample`: MAX-reduce 2x2 source texels into the next mip.
//
// The pyramid is *not* a graph node: it runs inline on the outer command
// buffer at the end of `draw_frame`, after `execute_graph` returns (which
// already committed the Main pass's cmd buf, so the depth attachment is
// written). Treating it as an end-of-frame action keeps it off the per-pass
// worker fan-out and off the graph's RMW chain on the main depth attachment
// (decals, fog, water, and the SSAO/SSR pre-passes already share that target).
//
// The source mip is read through the whole-texture read binding (lod =
// `src_mip`); the destination mip is a single-level texture view bound with
// write access. Reading mip M while writing mip M+1 never aliases the same
// texels, and Metal auto-barriers successive dispatches in the serial compute
// encoder, so the downsample chain stays correct.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _,
    MTLLibrary as _, MTLPixelFormat, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};

use super::context::{HDR_SAMPLE_COUNT, MtlContext};
use super::pipeline::{ns_str, shader_source};
use super::scoped_encoder::ScopedEncoder;

// Compute threadgroup tile size for the Hi-Z build kernels (8x8, matching the
// DirectX `[numthreads(8, 8, 1)]`).
const HIZ_TILE: usize = 8;

// Per-dispatch params pushed inline at the kernels' buffer(0). Must match the
// `HizParams` struct in `shaders/hiz_build.metal`.
#[derive(Copy, Clone)]
#[repr(C)]
struct HizParams {
    dst_width: u32,
    dst_height: u32,
    src_mip: u32,
    sample_count: u32,
}

// Mip count for a Hi-Z of size (w, h): `floor(log2(max(w, h))) + 1`. Power-
// of-two sources end exactly at 1x1; non-power-of-two sources stop one mip
// short of true 1x1 in the smaller dimension, which is fine: the cull kernel
// clamps to the actual mip dims. Mirrors `directx::hiz::hiz_mip_count`.
pub(super) fn hiz_mip_count(width: u32, height: u32) -> u32 {
    let m = width.max(height).max(1);
    32 - m.leading_zeros()
}

// Compute pipelines + texture + per-mip write views for the Hi-Z build. Built
// alongside the GPU-cull pipeline (same gating condition: bindless static
// pass active). `Some` on the context exactly when `cull_pipeline` is `Some`.
pub(super) struct HiZResources {
    pub(super) init_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(super) downsample_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // `R32Float` 2D texture with a full mip chain. The cull kernel reads it
    // via `read(coord, mip)`; the build kernels write each mip through the
    // matching single-level view in `mip_views`.
    pub(super) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    // One single-level 2D view per mip, bound as the write target of the
    // init / downsample dispatch that produces that mip. Length = `mip_count`.
    pub(super) mip_views: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_count: u32,
}

// Compile both Hi-Z compute kernels from `hiz_build.metal`.
#[allow(clippy::type_complexity)]
pub(super) fn build_hiz_pipelines(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<
    (
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ),
    String,
> {
    let msl = shader_source(hot_reload, "hiz_build.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("hiz shader compile error: {:?}", e))?;
    let init_fn = library
        .newFunctionWithName(&ns_str("hiz_init_msaa"))
        .ok_or("hiz_init_msaa not found in hiz library")?;
    let downsample_fn = library
        .newFunctionWithName(&ns_str("hiz_downsample"))
        .ok_or("hiz_downsample not found in hiz library")?;
    let init_pipeline = device
        .newComputePipelineStateWithFunction_error(&init_fn)
        .map_err(|e| format!("failed to create hiz_init_msaa pipeline: {:?}", e))?;
    let downsample_pipeline = device
        .newComputePipelineStateWithFunction_error(&downsample_fn)
        .map_err(|e| format!("failed to create hiz_downsample pipeline: {:?}", e))?;
    Ok((init_pipeline, downsample_pipeline))
}

// Create the `R32Float` mip-chain texture plus one single-level view per mip.
// Same pixel format for parent + views, so no `PixelFormatView` usage is
// needed. Private storage keeps it GPU-only (kernel-written, cull-read).
#[allow(clippy::type_complexity)]
fn create_hiz_texture_and_views(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<
    (
        Retained<ProtocolObject<dyn MTLTexture>>,
        Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    ),
    String,
> {
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setPixelFormat(MTLPixelFormat::R32Float);
        desc.setWidth(width.max(1) as usize);
        desc.setHeight(height.max(1) as usize);
        desc.setMipmapLevelCount(mip_count.max(1) as usize);
        desc.setUsage(MTLTextureUsage(
            MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::ShaderWrite.0,
        ));
        desc.setStorageMode(MTLStorageMode::Private);
    }
    let texture = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create hiz texture")?;

    let mut mip_views = Vec::with_capacity(mip_count as usize);
    for mip in 0..mip_count {
        // SAFETY: `mip` is in `0..mip_count`, the texture has `mip_count` mips
        // and a single slice; the view shares the parent's R32Float format /
        // Type2D so no reinterpretation occurs.
        let view = unsafe {
            texture.newTextureViewWithPixelFormat_textureType_levels_slices(
                MTLPixelFormat::R32Float,
                MTLTextureType::Type2D,
                NSRange::new(mip as usize, 1),
                NSRange::new(0, 1),
            )
        }
        .ok_or_else(|| format!("failed to create hiz mip {} view", mip))?;
        mip_views.push(view);
    }
    Ok((texture, mip_views))
}

impl HiZResources {
    // Build the Hi-Z pipelines + texture sized to the render (depth)
    // resolution. Called from the init path when the bindless static pass +
    // cull pipeline are active.
    pub(super) fn new(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        width: u32,
        height: u32,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let mip_count = hiz_mip_count(width, height);
        let (init_pipeline, downsample_pipeline) = build_hiz_pipelines(device, hot_reload)?;
        let (texture, mip_views) = create_hiz_texture_and_views(device, width, height, mip_count)?;
        Ok(Self {
            init_pipeline,
            downsample_pipeline,
            texture,
            mip_views,
            width,
            height,
            mip_count,
        })
    }

    // Recreate the texture + per-mip views at new render-target dimensions.
    // The pipelines are unaffected. The caller flips `hiz_valid` to false so
    // the next cull dispatch ignores the now-stale (about-to-be-rebuilt)
    // pyramid.
    pub(super) fn resize_to(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let mip_count = hiz_mip_count(width, height);
        let (texture, mip_views) = create_hiz_texture_and_views(device, width, height, mip_count)?;
        self.texture = texture;
        self.mip_views = mip_views;
        self.width = width;
        self.height = height;
        self.mip_count = mip_count;
        Ok(())
    }

    // Swap freshly-rebuilt pipelines into the live resource. Used by the
    // shader hot-reload pass; the texture + views are kept.
    pub(super) fn swap_pipelines(
        &mut self,
        init_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        downsample_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) {
        self.init_pipeline = init_pipeline;
        self.downsample_pipeline = downsample_pipeline;
    }
}

impl MtlContext {
    // Encode the Hi-Z build into `cmd_buf`. Reads this frame's MSAA main
    // depth and writes the mip chain that *next* frame's cull dispatch
    // consults. A no-op when no Hi-Z resource was built (bindless cull
    // pipeline not active). Caller runs this on the outer command buffer
    // after `execute_graph` so the Main pass has already written depth.
    pub(in crate::metal) fn encode_hiz_build(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) {
        let Some(hiz) = self.cull.hiz.as_ref() else {
            return;
        };
        if hiz.mip_count == 0 || hiz.mip_views.is_empty() {
            return;
        }
        let depth: &ProtocolObject<dyn MTLTexture> = self.hdr_targets.depth.as_ref();

        let Some(enc) = cmd_buf.computeCommandEncoder() else {
            tracing::error!("hiz: failed to get compute encoder");
            return;
        };
        let enc = ScopedEncoder::new(enc, "hiz-build");

        // Init: mip 0 from the MSAA main depth, MAX over samples.
        let init_params = HizParams {
            dst_width: hiz.width,
            dst_height: hiz.height,
            src_mip: 0,
            sample_count: HDR_SAMPLE_COUNT,
        };
        enc.setComputePipelineState(&hiz.init_pipeline);
        unsafe {
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&init_params).cast(),
                std::mem::size_of::<HizParams>(),
                0,
            );
            enc.setTexture_atIndex(Some(depth), 0);
            enc.setTexture_atIndex(Some(hiz.mip_views[0].as_ref()), 1);
        }
        dispatch_2d(&enc, hiz.width, hiz.height);

        // Downsample chain: each dispatch reads the prior mip (via the
        // whole-texture read binding at `src_mip`) and writes the next mip
        // through its single-level view.
        let mut cur_w = hiz.width;
        let mut cur_h = hiz.height;
        for mip in 1..hiz.mip_count {
            let next_w = (cur_w / 2).max(1);
            let next_h = (cur_h / 2).max(1);
            let params = HizParams {
                dst_width: next_w,
                dst_height: next_h,
                src_mip: mip - 1,
                sample_count: 0,
            };
            enc.setComputePipelineState(&hiz.downsample_pipeline);
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&params).cast(),
                    std::mem::size_of::<HizParams>(),
                    0,
                );
                enc.setTexture_atIndex(Some(hiz.texture.as_ref()), 0);
                enc.setTexture_atIndex(Some(hiz.mip_views[mip as usize].as_ref()), 1);
            }
            dispatch_2d(&enc, next_w, next_h);
            cur_w = next_w;
            cur_h = next_h;
        }
    }
}

// Dispatch a non-uniform 2D grid covering `(w, h)` with `HIZ_TILE`-square
// threadgroups. The kernels bounds-guard against the dst dimensions, so the
// non-uniform remainder threads return early.
fn dispatch_2d(enc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>, w: u32, h: u32) {
    let grid = MTLSize {
        width: w.max(1) as usize,
        height: h.max(1) as usize,
        depth: 1,
    };
    let tg = MTLSize {
        width: HIZ_TILE,
        height: HIZ_TILE,
        depth: 1,
    };
    enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::{HizParams, hiz_mip_count};
    use std::mem::{offset_of, size_of};

    #[test]
    fn hiz_params_layout_matches_msl() {
        // MSL `HizParams` in hiz_build.metal: four tightly packed uints.
        assert_eq!(size_of::<HizParams>(), 16);
        assert_eq!(offset_of!(HizParams, dst_width), 0);
        assert_eq!(offset_of!(HizParams, dst_height), 4);
        assert_eq!(offset_of!(HizParams, src_mip), 8);
        assert_eq!(offset_of!(HizParams, sample_count), 12);
    }

    #[test]
    fn mip_count_power_of_two() {
        // Power-of-two square: log2(N) + 1 full chain down to 1x1.
        assert_eq!(hiz_mip_count(1, 1), 1);
        assert_eq!(hiz_mip_count(2, 2), 2);
        assert_eq!(hiz_mip_count(256, 256), 9);
        assert_eq!(hiz_mip_count(1024, 1024), 11);
    }

    #[test]
    fn mip_count_uses_larger_dimension() {
        // Driven by max(w, h): a 1920x1080 target keys off 1920.
        assert_eq!(hiz_mip_count(1920, 1080), hiz_mip_count(1920, 1920));
        assert_eq!(hiz_mip_count(1920, 1080), 11);
        assert_eq!(hiz_mip_count(1280, 720), 11);
    }

    #[test]
    fn mip_count_clamps_zero() {
        // A zero dimension (minimised window) must not underflow.
        assert_eq!(hiz_mip_count(0, 0), 1);
        assert_eq!(hiz_mip_count(0, 8), 4);
    }
}
