// src/metal/post/fullscreen.rs
//
// Shared builders for fullscreen-triangle post-process passes. Every
// screen-space effect (SSAO, SSR, SSGI, TAA, bloom, fog, RT reflections, the
// final composite) draws one `[[vertex_id]]`-generated triangle into a single
// colour attachment with no vertex descriptor and no depth, differing only in
// shader source, attachment format, and blend. These helpers fold that shared
// pipeline-descriptor boilerplate into one place so each effect file keeps only
// what is unique to it.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBlendFactor, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLDevice as _, MTLLibrary as _,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
    MTLTexture,
};

use crate::metal::context::MtlContext;
use crate::metal::pass_timing::PassId;
use crate::metal::pipeline::ns_str;

// Blend configuration for a fullscreen pass's single colour attachment.
#[derive(Clone, Copy)]
pub(crate) enum FullscreenBlend {
    // No blending; the fragment output replaces the destination. Used by every
    // pass that writes a fresh target (SSAO kernel/blur, SSR resolve, TAA
    // resolve, the SSGI gather, the bloom prefilter/downsample, the composite).
    Replace,
    // Additive accumulation (`src·1 + dst·1`). Used where a pass layers an
    // extra term onto content it loaded: the bloom upsample chain and the
    // SSGI composite.
    Additive,
    // Premultiplied "over" (`src·1 + dst·(1 − srcA)`): the fragment already
    // folded coverage into its colour, so the source factor is `One`. Used by
    // the volumetric-fog composite.
    PremultipliedOver,
}

// Compile a post-pass shader library from inline MSL `source`. `label` names
// the shader in the returned compile error (e.g. "SSR", "bloom"). Pulled out
// so a builder that produces several pipelines from one source (bloom) still
// compiles it once and the per-effect files don't each repeat the compile
// dance.
pub(crate) fn compile_library(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    source: &str,
    label: &str,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>, String> {
    let options = objc2_metal::MTLCompileOptions::new();
    device
        .newLibraryWithSource_options_error(&ns_str(source), Some(&options))
        .map_err(|e| format!("{} shader compile error: {:?}", label, e))
}

// Build a render pipeline state for a fullscreen-triangle post pass: the two
// named functions from `library`, a single colour attachment at `format` with
// the requested `blend`, single-sample, no vertex descriptor, no depth. The
// pipeline-create error is tagged with `fragment_name` so a failure points at
// the exact entry point.
pub(crate) fn build_fullscreen_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    vertex_name: &str,
    fragment_name: &str,
    format: MTLPixelFormat,
    blend: FullscreenBlend,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let vert_fn = library
        .newFunctionWithName(&ns_str(vertex_name))
        .ok_or_else(|| format!("{} not found", vertex_name))?;
    let frag_fn = library
        .newFunctionWithName(&ns_str(fragment_name))
        .ok_or_else(|| format!("{} not found", fragment_name))?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(format);
        match blend {
            FullscreenBlend::Replace => ca.setBlendingEnabled(false),
            FullscreenBlend::Additive => {
                ca.setBlendingEnabled(true);
                ca.setSourceRGBBlendFactor(MTLBlendFactor::One);
                ca.setDestinationRGBBlendFactor(MTLBlendFactor::One);
                ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                ca.setDestinationAlphaBlendFactor(MTLBlendFactor::One);
            }
            FullscreenBlend::PremultipliedOver => {
                ca.setBlendingEnabled(true);
                ca.setSourceRGBBlendFactor(MTLBlendFactor::One);
                ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
                ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
                ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            }
        }
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create {} pipeline: {:?}", fragment_name, e))
}

// Where a fullscreen pass sits within an effect's GPU-timing span. Most
// effects are a single encoder (`Whole`); bloom and SSGI span several, so they
// mark the start sample on the first encoder and the end sample on the last.
#[derive(Clone, Copy)]
pub(crate) enum PassTimer {
    // Record no timing sample on this pass.
    None,
    // The effect's only encoder: record both its start and end samples here.
    Whole(PassId),
    // The first encoder of a multi-encoder effect: record the start sample.
    First(PassId),
    // The last encoder of a multi-encoder effect: record the end sample.
    Last(PassId),
}

impl MtlContext {
    // Run one fullscreen-triangle pass: open a single-attachment render encoder
    // on `target` (with the given `load` action and an always-`Store`), attach
    // GPU timing per `timer`, set `pipeline`, let `bind` set the pass's
    // fragment inputs, draw the `[[vertex_id]]` triangle, and end encoding.
    // Centralises the encoder open / draw / close skeleton every screen-space
    // effect repeats so each `encode_*` supplies only its unique bindings.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::metal) fn fullscreen_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        target: &ProtocolObject<dyn MTLTexture>,
        load: MTLLoadAction,
        timer: PassTimer,
        pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
        label: &str,
        bind: impl FnOnce(&ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>),
    ) -> Result<(), String> {
        let desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(target));
            ca.setLoadAction(load);
            ca.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            match timer {
                PassTimer::None => {}
                PassTimer::Whole(id) => t.attach_render(&desc, id),
                PassTimer::First(id) => t.attach_render_first(&desc, id),
                PassTimer::Last(id) => t.attach_render_last(&desc, id),
            }
        }
        let enc = cmd_buf
            .renderCommandEncoderWithDescriptor(&desc)
            .ok_or_else(|| format!("failed to get {} encoder", label))?;
        enc.setRenderPipelineState(pipeline);
        bind(&enc);
        unsafe {
            enc.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
        enc.endEncoding();
        Ok(())
    }
}
