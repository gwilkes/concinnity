// src/metal/post/ssgi.rs
//
// Screen-space global illumination: a refinement of SSR. It reuses the SSR
// depth + normal pre-pass G-buffer (so turning SSGI on forces that pre-pass to
// run even when SSR resolve is off) and runs two fullscreen passes after the
// main pass:
//
//   * gather:    per pixel, cone of cosine-weighted hemisphere rays marched
//                against the G-buffer, accumulating the lit scene colour at
//                each on-screen hit into an off-screen `gi` target.
//   * composite: a depth-aware blur of that noisy `gi` target, additively
//                blended (ONE / ONE) into `hdr_resolve` so the near-field
//                indirect bounce layers on top of the IBL ambient.
//
// Pipelines, target, and both encoders live together so the effect is a single
// unit Vulkan / DirectX can mirror.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLLoadAction, MTLPixelFormat, MTLRenderCommandEncoder as _,
    MTLRenderPipelineState, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::gfx::ssgi::SsgiSettings;
use crate::metal::context::MtlContext;
use crate::metal::pipeline::shader_source;
use crate::metal::post::fullscreen::{
    FullscreenBlend, PassTimer, build_fullscreen_pipeline, compile_library,
};

// All screen-space-GI feature state grouped into one unit: the resolved
// tunables, the `gi` gather target, and the gather + composite pipelines.
// All `Some` only when SSGI is on (and the SSR pre-pass G-buffer it gathers
// against therefore exists).
pub(crate) struct SsgiState {
    pub settings: Option<SsgiSettings>,
    pub targets: Option<SsgiTargets>,
    pub gather_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub composite_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
}

// Pipelines

// Build one SSGI fullscreen pipeline for the given fragment entry point.
// `additive` configures an `ONE / ONE` blend (the composite pass blends the
// indirect term into `hdr_resolve`); the gather pass leaves it off so it
// writes its `gi` target straight. Both write a single-sample `RGBA16Float`
// target.
fn build_ssgi_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    fragment_entry: &str,
    additive: bool,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "ssgi.metal");
    let library = compile_library(device, msl.as_ref(), "SSGI")?;
    let blend = if additive {
        FullscreenBlend::Additive
    } else {
        FullscreenBlend::Replace
    };
    build_fullscreen_pipeline(
        device,
        &library,
        "ssgi_fullscreen_vertex",
        fragment_entry,
        MTLPixelFormat::RGBA16Float,
        blend,
    )
}

// Build the SSGI gather pipeline (hemisphere ray-march → `gi` target, no blend).
pub(crate) fn build_ssgi_gather_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_ssgi_pipeline(device, "ssgi_gather_fragment", false, hot_reload)
}

// Build the SSGI composite pipeline (depth-aware blur, additively blended into
// `hdr_resolve`).
pub(crate) fn build_ssgi_composite_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_ssgi_pipeline(device, "ssgi_composite_fragment", true, hot_reload)
}

// Targets

// Off-screen target for the SSGI gather pass. The composite blends straight
// into `hdr_resolve`, so only the intermediate `gi` texture lives here.
// Single-sample, full render resolution; created when SSGI is enabled and
// rebuilt with the HDR targets on resize.
pub(crate) struct SsgiTargets {
    // Gathered indirect radiance (`RGBA16Float`), before the depth-aware blur
    // the composite pass applies.
    pub gi: Retained<ProtocolObject<dyn MTLTexture>>,
}

// Create or recreate the SSGI targets at `width`x`height`.
pub(crate) fn create_ssgi_targets(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    width: u32,
    height: u32,
) -> Result<SsgiTargets, String> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let usage = MTLTextureUsage(MTLTextureUsage::ShaderRead.0 | MTLTextureUsage::RenderTarget.0);
    let desc = MTLTextureDescriptor::new();
    unsafe {
        desc.setTextureType(MTLTextureType::Type2D);
        desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
        desc.setWidth(w);
        desc.setHeight(h);
        desc.setUsage(usage);
        desc.setStorageMode(objc2_metal::MTLStorageMode::Private);
    }
    let gi = device
        .newTextureWithDescriptor(&desc)
        .ok_or("failed to create SSGI gi texture")?;
    Ok(SsgiTargets { gi })
}

// Encoder

impl MtlContext {
    // Encode the SSGI gather + composite. The gather marches hemisphere rays
    // over the SSR pre-pass G-buffer and writes the noisy indirect radiance
    // into `ssgi_targets.gi`; the composite depth-aware-blurs it and additively
    // blends it into `hdr_resolve`. Runs on the hdr_resolve RMW chain after the
    // main pass; only called when SSGI is on (and the SSR pre-pass G-buffer
    // therefore exists).
    pub(in crate::metal) fn encode_ssgi(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        ssgi_params: &crate::gfx::render_types::SsgiParams,
    ) -> Result<u32, String> {
        let (targets, gather_ps, composite_ps, gbuffer) = match (
            &self.ssgi.targets,
            &self.ssgi.gather_pipeline,
            &self.ssgi.composite_pipeline,
            self.gbuffer.targets.as_ref().map(|t| &t.normal_depth),
        ) {
            (Some(t), Some(g), Some(c), Some(gb)) => (t, g, c, gb),
            // SSGI requires the SSR pre-pass G-buffer for normals + depth; if
            // it is missing (no pre-pass built this session) there is nothing
            // to gather against, so skip the pass.
            _ => return Ok(0),
        };

        // Gather: hemisphere ray-march over the G-buffer -> gi target.
        self.fullscreen_pass(
            cmd_buf,
            targets.gi.as_ref(),
            MTLLoadAction::DontCare,
            PassTimer::First(crate::metal::pass_timing::PassId::Ssgi),
            gather_ps,
            "SSGI gather",
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(self.hdr_targets.hdr_resolve.as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(gbuffer.as_ref()), 1);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(ssgi_params).cast(),
                    std::mem::size_of::<crate::gfx::render_types::SsgiParams>(),
                    0,
                );
            },
        )?;

        // Composite: depth-aware blur of gi, additively blended into the
        // scene. Loads the existing hdr_resolve so the ONE/ONE blend adds the
        // indirect term on top.
        self.fullscreen_pass(
            cmd_buf,
            self.hdr_targets.hdr_resolve.as_ref(),
            MTLLoadAction::Load,
            PassTimer::Last(crate::metal::pass_timing::PassId::Ssgi),
            composite_ps,
            "SSGI composite",
            |enc| unsafe {
                enc.setFragmentTexture_atIndex(Some(targets.gi.as_ref()), 0);
                enc.setFragmentTexture_atIndex(Some(gbuffer.as_ref()), 1);
                enc.setFragmentSamplerState_atIndex(Some(&self.post_sampler), 0);
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(ssgi_params).cast(),
                    std::mem::size_of::<crate::gfx::render_types::SsgiParams>(),
                    0,
                );
            },
        )?;

        Ok(0)
    }
}
