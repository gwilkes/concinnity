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
    // Roughness-aware blur + composite of the reflection target over the scene.
    // Shared by the SSR and RT-reflection resolves (both write the reflection
    // target, then run this). Built whenever the reflection targets exist.
    pub composite_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // First half of the composite: the roughness blur, run at reduced resolution
    // into `SsrTargets::blur`. Built alongside `composite_pipeline`.
    pub blur_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
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

// Build the reflection composite pipeline: the full-resolution second pass that
// lerps the sharp reflection against the upsampled half-res blur by roughness
// and composites it over the scene, writing the `RGBA16Float` scene output the
// SSR / RT resolve used to write directly. Shared by both reflection paths.
pub(crate) fn build_reflection_composite_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "reflection_composite.metal");
    let library = compile_library(device, msl.as_ref(), "reflection composite")?;
    build_fullscreen_pipeline(
        device,
        &library,
        "reflection_composite_vertex",
        "reflection_composite_fragment",
        MTLPixelFormat::RGBA16Float,
        FullscreenBlend::Replace,
    )
}

// Build the reflection blur pipeline: the reduced-resolution first pass that
// weight-averages the reflection target over the roughness cone into the blur
// target the composite then upsamples. The expensive multi-tap blur runs here.
pub(crate) fn build_reflection_blur_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "reflection_composite.metal");
    let library = compile_library(device, msl.as_ref(), "reflection blur")?;
    build_fullscreen_pipeline(
        device,
        &library,
        "reflection_composite_vertex",
        "reflection_blur_fragment",
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
    // Reflection target (`RGBA16Float`): the SSR / RT resolve writes reflected
    // radiance in `.rgb` and the Fresnel/gloss composite weight in `.a` here,
    // and the reflection composite blurs + composites it into `output`.
    pub reflection: Retained<ProtocolObject<dyn MTLTexture>>,
    // Scene with reflections composited in. Becomes the scene colour the TAA /
    // bloom / composite passes consume when SSR or RT reflections are on.
    pub output: Retained<ProtocolObject<dyn MTLTexture>>,
    // Reduced-resolution roughness blur of `reflection` (the blur pass writes it,
    // the composite pass upsamples it). Sized at render / REFLECTION_BLUR_SCALE.
    pub blur: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Per-axis render-resolution divisor for the roughness blur pass. The blur is
// low-frequency (a widening glossy cone), so running it at half resolution and
// bilinear-upsampling in the composite is visually free while quartering the
// blur's pixel count. Mirrors stay sharp: the composite lerps in the FULL-RES
// reflection for low roughness (see reflection_composite.metal). Matches the
// `SsgiResolution::Half` default the SSGI gather uses.
const REFLECTION_BLUR_SCALE: u32 = 2;

// Create or recreate the reflection + resolve-output targets at `width`x`height`,
// plus the reduced-resolution blur target.
pub(crate) fn create_ssr_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<SsrTargets, String> {
    let make_at = |w: usize, h: usize| -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
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
        device.newTextureWithDescriptor(&desc)
    };
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let bw = (width / REFLECTION_BLUR_SCALE).max(1) as usize;
    let bh = (height / REFLECTION_BLUR_SCALE).max(1) as usize;
    let reflection = make_at(w, h).ok_or("failed to create reflection texture")?;
    let output = make_at(w, h).ok_or("failed to create SSR output texture")?;
    let blur = make_at(bw, bh).ok_or("failed to create reflection blur texture")?;
    Ok(SsrTargets {
        reflection,
        output,
        blur,
    })
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

        // Resolve: ray-march the reflection over `hdr_resolve` -> reflection
        // target (reflected radiance + composite weight, not yet blended).
        self.fullscreen_pass(
            cmd_buf,
            targets.reflection.as_ref(),
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
                // Local reflection-probe cubes at texture(4..4+MAX_PROBES): when a
                // probe is baked a missed/edge ray reflects its box-projected scene
                // capture instead of the foreign sky HDR (the source the forward IBL
                // specular term uses). `probe_cube_or_sky` returns the sky for
                // unbaked slots, so binding all MAX_PROBES is always valid; the
                // ProbeSet's `count` gates whether the shader samples them.
                for i in 0..crate::metal::uniforms::MAX_PROBES {
                    enc.setFragmentTexture_atIndex(Some(self.probe_cube_or_sky(i)), 4 + i);
                }
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 1);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(ssr_params).cast(),
                    std::mem::size_of::<crate::gfx::render_types::SsrParams>(),
                    0,
                );
                // Reflection-probe set (count + per-probe parallax boxes) at
                // buffer(1); count == 0 keeps the sky fallback above.
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&self.probe_set).cast(),
                    std::mem::size_of::<crate::metal::uniforms::ProbeSet>(),
                    1,
                );
            },
        )?;
        // Blur by roughness + composite the reflection over the scene -> output.
        self.encode_reflection_composite(cmd_buf)?;
        Ok(0)
    }

    // Blur the reflection target by surface roughness and composite it over
    // `hdr_resolve` into `ssr_targets.output`. Shared by the SSR and
    // RT-reflection resolves: both write the reflection target first, then call
    // this. A no-op (leaves `output` untouched) when the composite pipeline or
    // G-buffer is absent, which only happens when no reflection path is active.
    pub(in crate::metal) fn encode_reflection_composite(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> Result<(), String> {
        let (targets, composite_ps, blur_ps, gbuf) = match (
            &self.ssr.targets,
            &self.ssr.composite_pipeline,
            &self.ssr.blur_pipeline,
            &self.gbuffer.targets,
        ) {
            (Some(t), Some(cp), Some(bp), Some(g)) => (t, cp, bp, g),
            _ => return Ok(()),
        };
        // Pass 1: the roughness blur, at reduced resolution into `blur`. Times the
        // span start; the composite below times its end (both under one slot).
        self.fullscreen_pass(
            cmd_buf,
            targets.blur.as_ref(),
            MTLLoadAction::DontCare,
            PassTimer::First(crate::metal::pass_timing::PassId::ReflectionComposite),
            blur_ps,
            "reflection blur",
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(targets.reflection.as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(gbuf.roughness.as_ref()), 1);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
            },
        )?;
        // Pass 2: lerp the sharp full-res reflection against the upsampled blur by
        // roughness, then composite over the scene into `output`.
        self.fullscreen_pass(
            cmd_buf,
            targets.output.as_ref(),
            MTLLoadAction::DontCare,
            PassTimer::Last(crate::metal::pass_timing::PassId::ReflectionComposite),
            composite_ps,
            "reflection composite",
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(targets.reflection.as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(self.hdr_targets.hdr_resolve.as_ref()), 1);
                enc.setFragmentTexture_atIndex(Some(gbuf.normal_depth.as_ref()), 2);
                enc.setFragmentTexture_atIndex(Some(gbuf.roughness.as_ref()), 3);
                enc.setFragmentTexture_atIndex(Some(targets.blur.as_ref()), 4);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
            },
        )?;
        Ok(())
    }
}
