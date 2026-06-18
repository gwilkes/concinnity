// src/metal/post/taa.rs
//
// Temporal anti-aliasing: the velocity (motion-vector) pre-pass and the
// resolve pass that blends the current frame with reprojected history.
// Pipelines, ping-pong targets, velocity target allocation, and both
// per-frame encoders live together so the effect is a single unit Vulkan /
// DirectX can mirror.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLRenderCommandEncoder as _,
    MTLRenderPipelineState, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::metal::context::MtlContext;
use crate::metal::pipeline::shader_source;
use crate::metal::post::fullscreen::{
    FullscreenBlend, PassTimer, build_fullscreen_pipeline, compile_library,
};
use crate::metal::uniforms::TaaUniforms;

// All temporal-anti-aliasing state grouped into one feature unit: the on/off
// toggle, the resolve pipeline, the two ping-pong history buffers, and the
// per-frame bookkeeping (write index, history-valid flag, Halton jitter
// frame counter). The pipeline / targets are `Some` / non-empty only when TAA
// is enabled (and not bypassed by the upscaler).
pub(crate) struct TaaState {
    // Toggle resolved from `PostProcessConfig.taa`; false skips the TAA pass
    // + projection jitter entirely.
    pub enabled: bool,
    // Resolve pipeline (fullscreen triangle). `Some` only when TAA is on.
    pub pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // The two `RGBA16Float` history buffers the resolve ping-pongs between;
    // empty when TAA is disabled, re-created with `hdr_targets` on resize.
    pub targets: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    // Index into `targets` this frame writes into; the other slot is history.
    pub dst: usize,
    // False on the first frame and after a resize: the resolve then passes
    // the current frame through untouched.
    pub history_valid: bool,
    // Frame counter driving the Halton projection-jitter sequence.
    pub frame: u32,
}

// Pipelines

// Build the temporal anti-aliasing (TAA) resolve pipeline: a fullscreen
// triangle that blends the current HDR frame with a reprojected history
// buffer. Renders into a single-sample `RGBA16Float` target (the new
// history). Per-pixel motion comes from the velocity pre-pass
// (`build_velocity_pipeline`), so both camera motion and per-object / skinned
// motion reproject correctly -- moving props no longer ghost.
pub(crate) fn build_taa_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "taa.metal");
    let library = compile_library(device, msl.as_ref(), "TAA")?;
    build_fullscreen_pipeline(
        device,
        &library,
        "taa_vertex_main",
        "taa_fragment_main",
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
    )
}

// Targets

// Create the two single-sample `RGBA16Float` targets the TAA resolve pass
// ping-pongs between: one frame's output is the next frame's history. Both
// are full drawable resolution, `ShaderRead | RenderTarget`, GPU-private.
pub(crate) fn create_taa_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<[Retained<ProtocolObject<dyn MTLTexture>>; 2], String> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let make = || -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setTextureType(MTLTextureType::Type2D);
            desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
            desc.setWidth(w);
            desc.setHeight(h);
            desc.setUsage(MTLTextureUsage(
                MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0,
            ));
            desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
        }
        device
            .newTextureWithDescriptor(&desc)
            .ok_or("failed to create TAA target texture".to_string())
    };
    Ok([make()?, make()?])
}

// Encoders

impl MtlContext {
    // Encode the TAA resolve pass: one fullscreen-triangle draw that blends
    // `scene_input` (the SSR output, or `hdr_resolve` when SSR is off) with
    // the reprojected history buffer. Runs between SSR and bloom; the output
    // is both the scene colour the later passes consume and next frame's
    // history.
    pub(in crate::metal) fn encode_taa(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        taa_uniforms: &TaaUniforms,
        scene_input: &ProtocolObject<dyn objc2_metal::MTLTexture>,
    ) -> Result<u32, String> {
        let pipeline = self
            .taa
            .pipeline_state
            .as_ref()
            .ok_or("TAA enabled but pipeline missing")?;
        let gbuf = self
            .gbuffer
            .targets
            .as_ref()
            .ok_or("TAA enabled but G-buffer targets missing")?;
        let history = &self.taa.targets[1 - self.taa.dst];
        let dst = &self.taa.targets[self.taa.dst];

        self.fullscreen_pass(
            cmd_buf,
            dst.as_ref(),
            MTLLoadAction::DontCare,
            PassTimer::Whole(crate::metal::pass_timing::PassId::TaaResolve),
            pipeline,
            "TAA resolve",
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(scene_input), 0);
                enc.setFragmentTexture_atIndex(Some(gbuf.velocity.as_ref()), 1);
                enc.setFragmentTexture_atIndex(Some(history.as_ref()), 2);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(taa_uniforms).cast(),
                    std::mem::size_of::<TaaUniforms>(),
                    0,
                );
            },
        )?;
        Ok(0)
    }
}
