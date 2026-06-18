// src/metal/post/ssr.rs
//
// Screen-space reflections: a depth + normal + roughness pre-pass that runs
// before the main pass, and a fullscreen ray-march resolve that runs after.
// Pipelines, targets, and both encoders live together so the effect is a
// single unit Vulkan / DirectX can mirror.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLRenderCommandEncoder as _,
    MTLRenderPipelineState, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::gfx::ssr::SsrSettings;
use crate::metal::context::MtlContext;
use crate::metal::pipeline::shader_source;
use crate::metal::post::fullscreen::{
    FullscreenBlend, PassTimer, build_fullscreen_pipeline, compile_library,
};

// All screen-space-reflection feature state grouped into one unit: the
// resolved tunables, the resolve-output target, and the resolve pipeline.
// `targets` is `Some` when SSR, SSGI, *or* RT reflections are on (they share
// the G-buffer pre-pass output and RT reuses `targets.output`); `settings`
// and `resolve_pipeline` are `Some` only when SSR itself is on.
pub(crate) struct SsrState {
    pub settings: Option<SsrSettings>,
    pub targets: Option<SsrTargets>,
    pub resolve_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

// Pipelines

// Build the SSR resolve pipeline: a fullscreen-triangle pass that ray-marches
// the reflection and composites it over the scene, writing a single-sample
// `RGBA16Float` target.
pub(crate) fn build_ssr_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "ssr.metal");
    let library = compile_library(device, msl.as_ref(), "SSR")?;
    build_fullscreen_pipeline(
        device,
        &library,
        "ssr_fullscreen_vertex",
        "ssr_resolve_fragment",
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
    )
}

// Targets

// Off-screen target for the screen-space reflection (SSR) resolve pass: the HDR
// scene with reflections composited in. The view-space normal / linear depth /
// roughness the resolve reads now come from the unified G-buffer pre-pass
// (`metal/post/gbuffer.rs`); only this resolve output lives here. Single-sample,
// full drawable resolution; created when SSR is enabled and rebuilt with the HDR
// targets on resize.
pub(crate) struct SsrTargets {
    // SSR resolve output (`RGBA16Float`): the HDR scene with reflections
    // composited in. Becomes the scene colour the TAA / bloom / composite
    // passes consume when SSR is on.
    pub output: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Create or recreate the SSR resolve-output target at `width`x`height`.
pub(crate) fn create_ssr_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<SsrTargets, String> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;

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
    let output = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create SSR output texture")?;
    Ok(SsrTargets { output })
}

// Encoders

impl MtlContext {
    // Encode the SSR resolve: a fullscreen ray-march over the pre-pass
    // G-buffer that reflects `hdr_resolve` and composites the result into
    // `ssr_targets.output`. Rays that miss -- or fade out near a screen
    // border -- fall back to the IBL prefilter cubemap so the reflection
    // hands off to the environment rather than snapping to the base shading;
    // with no EnvironmentMap bound the cube is skipped and a miss keeps the
    // base shading. Runs after the main pass; only called when SSR is on.
    pub(in crate::metal) fn encode_ssr_resolve(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssr_params: &crate::gfx::render_types::SsrParams,
    ) -> Result<u32, String> {
        let (targets, resolve_ps, gbuf) = match (
            &self.ssr.targets,
            &self.ssr.resolve_pipeline,
            &self.gbuffer.targets,
        ) {
            (Some(t), Some(b), Some(g)) => (t, b, g),
            _ => return Ok(0),
        };

        // Resolve: ray-march the reflection over `hdr_resolve` -> output.
        self.fullscreen_pass(
            cmd_buf,
            targets.output.as_ref(),
            MTLLoadAction::DontCare,
            PassTimer::Whole(crate::metal::pass_timing::PassId::SsrResolve),
            resolve_ps,
            "SSR resolve",
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(self.hdr_targets.hdr_resolve.as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(gbuf.normal_depth.as_ref()), 1);
                enc.setFragmentTexture_atIndex(Some(gbuf.roughness.as_ref()), 2);
                // The IBL prefilter cubemap is the miss / screen-edge fallback.
                // It is always valid (a grey fallback when no EnvironmentMap is
                // bound); `SsrParams.prefilter_mip_count == 0` tells the shader to
                // ignore it in that case.
                enc.setFragmentTexture_atIndex(Some(self.env_map.prefilter.as_ref()), 3);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 1);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(ssr_params).cast(),
                    std::mem::size_of::<crate::gfx::render_types::SsrParams>(),
                    0,
                );
            },
        )?;
        Ok(0)
    }
}
