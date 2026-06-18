// src/metal/fog.rs
//
// Per-frame encoder for the volumetric-fog pass. Runs after the main HDR
// pass (and after the decal pass, so fog sits on top of decals just like it
// does for any other resolved scene colour) and before SSR / TAA, so the
// reflections and history reproject through the integrated fog colour and
// transmittance.
//
// The pass is a single fullscreen triangle: the fragment shader samples the
// main pass's MSAA depth attachment, reconstructs each pixel's world-space
// surface point via the inverse VP, ray-marches a sun-lit homogeneous
// medium with exponential height falloff, and writes `(scattered_rgb, 1 -
// transmittance)` so the pipeline's `over` blend yields
// `scene * T + scattered` automatically.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _,
    MTLLibrary as _, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineState, MTLSize, MTLStoreAction, MTLTexture,
    MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::gfx::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use crate::gfx::render_types::{FogFroxelParams, FogParams};
use crate::gfx::volumetric_fog::FogSettings;

use super::context::MtlContext;
use super::pipeline::{ns_str, shader_source};
use super::post::fullscreen::{FullscreenBlend, build_fullscreen_pipeline, compile_library};
use super::scoped_encoder::ScopedEncoder;

// All volumetric-fog state grouped into one feature unit: the resolved
// tunables, the fullscreen ray-march pipeline, and the froxel-volume compute
// pipeline + its 3D output volume. All `Some` only when the world declares a
// `VolumetricFog` with `enabled = true` (or one is set at runtime via
// `update_fog_settings`); `None` skips the pass entirely.
pub(crate) struct FogState {
    pub settings: Option<FogSettings>,
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub froxel_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub froxel_volume: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
}

impl MtlContext {
    // Hot-reload entry point for the volumetric-fog tunables. Writes the new
    // `Option<FogSettings>` into `self.fog.settings`; the next `draw_frame`
    // then re-builds `FogParams` from it. `None` disables the pass even if
    // the pipeline is live (the guard in [`MtlContext::draw_frame`] needs
    // both fog_pipeline and fog_settings).
    //
    // If the world started with no `VolumetricFog` (so `fog_pipeline` is
    // `None`), a `Some` update logs once and is dropped: re-enabling fog
    // mid-run requires a relaunch.
    pub fn update_fog_settings(&mut self, settings: Option<FogSettings>) {
        if settings.is_some() && self.fog.pipeline.is_none() {
            tracing::warn!(
                "VolumetricFog hot-reload: world started without fog, so the fog \
                 pipeline was never built: re-enabling fog mid-run is not \
                 supported (relaunch required). Ignoring update."
            );
            return;
        }
        self.fog.settings = settings;
    }

    // Encode the volumetric-fog pass. Caller has already ended the main HDR
    // pass (and the decal pass, if any), so `hdr_targets.depth` (MSAA) holds
    // the scene depth and `hdr_targets.hdr_resolve` holds the resolved
    // scene + decals colour. The pass alpha-blends a single lit ray-march
    // over `hdr_resolve`.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_fog(
        &self,
        cmd_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &FogParams,
        froxel_params: &FogFroxelParams,
    ) -> Result<u32, String> {
        let pipeline = match &self.fog.pipeline {
            Some(p) => p,
            None => return Ok(0),
        };
        let volume = match &self.fog.froxel_volume {
            Some(v) => v,
            None => return Ok(0),
        };

        let pass_desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Fog);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get fog render encoder")?,
            "volumetric fog",
        );
        enc.setRenderPipelineState(pipeline);

        unsafe {
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(params).cast(),
                std::mem::size_of::<FogParams>(),
                0,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(froxel_params).cast(),
                std::mem::size_of::<FogFroxelParams>(),
                1,
            );
            // Sample the single-sample `depth_resolve` (post-
            // Main depth + any raymarched surface depth) so fog
            // attenuates raymarched surfaces by their true distance.
            enc.setFragmentTexture_atIndex(Some(self.hdr_targets.depth_resolve.as_ref()), 0);
            enc.setFragmentTexture_atIndex(Some(volume.as_ref()), 1);
            // Fullscreen triangle: 3 vertices, no vertex buffer.
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }

        Ok(1)
    }

    // Encode the volumetric-fog froxel-volume compute pass. One thread per
    // (x, y) tile of the 3D volume; the kernel walks the Z slices from front
    // to back, accumulating per-slab scatter + transmittance with a CSM
    // shadow tap per slice. Writes `(scattered_rgb, 1 - transmittance)` into
    // the slice of `fog_froxel_volume`. Caller must dispatch this before
    // `encode_fog`, which samples the same volume.
    pub(in crate::metal) fn encode_fog_froxel(
        &self,
        cmd_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        params: &FogParams,
        froxel_params: &FogFroxelParams,
    ) -> Result<u32, String> {
        let pipeline = match &self.fog.froxel_pipeline {
            Some(p) => p,
            None => return Ok(0),
        };
        let volume = match &self.fog.froxel_volume {
            Some(v) => v,
            None => return Ok(0),
        };

        let cmd_buf_dyn: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer> = cmd_buf;
        let desc = objc2_metal::MTLComputePassDescriptor::computePassDescriptor();
        if let Some(t) = &self.pass_timing {
            t.attach_compute(&desc, super::pass_timing::PassId::FogFroxel);
        }
        let enc = ScopedEncoder::new(
            cmd_buf_dyn
                .computeCommandEncoderWithDescriptor(&desc)
                .ok_or("failed to get fog froxel compute encoder")?,
            "fog froxel volume",
        );
        enc.setComputePipelineState(pipeline);

        unsafe {
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(params).cast(),
                std::mem::size_of::<FogParams>(),
                0,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(froxel_params).cast(),
                std::mem::size_of::<FogFroxelParams>(),
                1,
            );
            // ShadowUniforms at buffer(2) so the kernel can pick a CSM
            // cascade per froxel.
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&self.shadow_uniforms).cast(),
                std::mem::size_of_val(&self.shadow_uniforms),
                2,
            );
            enc.setTexture_atIndex(Some(self.shadow_map.as_ref()), 0);
            enc.setTexture_atIndex(Some(volume.as_ref()), 1);

            // One thread per (x, y) tile; the kernel walks Z internally.
            // Threadgroup of 8x8x1 keeps occupancy high without thrashing
            // registers (the inner Z loop has decent working set).
            let tg = MTLSize {
                width: 8,
                height: 8,
                depth: 1,
            };
            let grid = MTLSize {
                width: FOG_FROXEL_X as usize,
                height: FOG_FROXEL_Y as usize,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        }
        Ok(0)
    }
}

// Build the volumetric-fog pipeline: a fullscreen triangle that samples the
// main pass's MSAA depth attachment, ray-marches the view ray through a lit
// homogeneous medium with exponential height falloff and a Henyey-Greenstein
// phase function, and composites the result over the resolved HDR target
// with a standard `over` alpha blend.
pub(super) fn build_fog_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "fog.metal");
    let library = compile_library(device, msl.as_ref(), "fog")?;
    // `(scattered, 1 - T)` over `scene` -> `scene * T + scattered`.
    build_fullscreen_pipeline(
        device,
        &library,
        "fog_vertex",
        "fog_fragment",
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::PremultipliedOver,
    )
}

// Build the volumetric-fog froxel-volume compute pipeline. Mirrors
// `build_fog_pipeline`'s shader-source pickup (hot-reload-aware) but
// produces an `MTLComputePipelineState` from the `fog_froxel_kernel`
// function in the same source file.
pub(super) fn build_fog_froxel_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, String> {
    let msl = shader_source(hot_reload, "fog.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("fog froxel shader compile error: {:?}", e))?;
    let func = library
        .newFunctionWithName(&ns_str("fog_froxel_kernel"))
        .ok_or("fog_froxel_kernel not found")?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("failed to create fog froxel pipeline: {:?}", e))
}

// Allocate the 3D `RGBA16Float` volume the froxel kernel writes and the
// fog fragment shader samples. Dimensions live in
// [`crate::gfx::render_graph::FOG_FROXEL_X`] / `Y` / `Z`.
pub(super) fn build_fog_froxel_volume(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let desc = MTLTextureDescriptor::new();
    desc.setTextureType(MTLTextureType::Type3D);
    desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
    unsafe {
        desc.setWidth(FOG_FROXEL_X as usize);
        desc.setHeight(FOG_FROXEL_Y as usize);
        desc.setDepth(FOG_FROXEL_Z as usize);
        desc.setMipmapLevelCount(1);
    }
    desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
    desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| "failed to allocate fog froxel volume texture".to_string())
}
