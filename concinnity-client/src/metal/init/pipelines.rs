// src/metal/init/pipelines.rs
//
// Core render-pipeline construction extracted from MtlContext::new:
//   * The shared vertex descriptor (interleaved [pos, normal, tangent, color, uv]).
//   * The main static pipeline (with optional bindless fragment + GPU-driven
//     cull pipeline + bindless texture argument encoder).
//   * The optional instanced pipeline.
//   * The shared depth-stencil state used by main + shadow passes.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLArgumentEncoder, MTLCompareFunction, MTLComputePipelineState, MTLDepthStencilDescriptor,
    MTLDepthStencilState, MTLDevice, MTLFunction as _, MTLLibrary as _, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

use crate::gfx::mesh_payload::Vertex;
use crate::metal::context::{BINDLESS_TEXTURE_ARG_BUFFER_INDEX, HDR_SAMPLE_COUNT};
use crate::metal::cull::build_cull_pipeline;
use crate::metal::pipeline::{load_library, ns_str, shader_source};
use crate::metal::post::fullscreen::compile_library;

pub(crate) struct MainPipelineBundle {
    pub pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub bindless: bool,
    pub cull_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub cull_icb_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    // Phase-2 cull pipeline + its ICB argument encoder for two-pass
    // occlusion. Built alongside the phase-1 cull pipeline whenever the
    // bindless path is active (cheap: one extra compute pipeline from the
    // same library); only used when `occlusion_two_pass` is on at runtime.
    pub cull_pipeline_phase2: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub cull_icb2_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub bindless_tex_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
}

// Describes the per-vertex buffer layout so Metal can map [[stage_in]]:
//   buffer(1): interleaved [float3 pos, float3 normal, float3 tangent, float3 color, float2 uv]
//   stride = sizeof(Vertex) = 56 bytes
pub(crate) fn make_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        // attribute 0: pos, float3, offset 0
        let attr0 = vert_desc.attributes().objectAtIndexedSubscript(0);
        attr0.setFormat(MTLVertexFormat::Float3);
        attr0.setOffset(0);
        attr0.setBufferIndex(1);

        // attribute 1: normal, float3, offset 12
        let attr1 = vert_desc.attributes().objectAtIndexedSubscript(1);
        attr1.setFormat(MTLVertexFormat::Float3);
        attr1.setOffset(12);
        attr1.setBufferIndex(1);

        // attribute 2: tangent, float3, offset 24
        let attr2 = vert_desc.attributes().objectAtIndexedSubscript(2);
        attr2.setFormat(MTLVertexFormat::Float3);
        attr2.setOffset(24);
        attr2.setBufferIndex(1);

        // attribute 3: color, float3, offset 36
        let attr3 = vert_desc.attributes().objectAtIndexedSubscript(3);
        attr3.setFormat(MTLVertexFormat::Float3);
        attr3.setOffset(36);
        attr3.setBufferIndex(1);

        // attribute 4: uv, float2, offset 48
        let attr4 = vert_desc.attributes().objectAtIndexedSubscript(4);
        attr4.setFormat(MTLVertexFormat::Float2);
        attr4.setOffset(48);
        attr4.setBufferIndex(1);

        // layout for buffer(1): per-vertex stride
        let layout1 = vert_desc.layouts().objectAtIndexedSubscript(1);
        layout1.setStride(std::mem::size_of::<Vertex>());
        layout1.setStepFunction(MTLVertexStepFunction::PerVertex);
    }
    vert_desc
}

// Build the main static pipeline together with everything it implies:
//
//   * If the fragment library exposes `fragment_main_bindless`, the static main
//     pass is GPU-driven. That requires:
//       - the pipeline to opt into indirect command buffers
//       - a compute cull pipeline + ICB argument encoder
//       - an argument encoder for the BindlessTextures argument buffer
//   * Otherwise the pipeline pairs with the per-draw `fragment_main` and the
//     three Optional fields are `None`.
pub(crate) fn build_main_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    vert_lib_bytes: &[u8],
    frag_lib_bytes: &[u8],
    hot_reload: bool,
) -> Result<MainPipelineBundle, String> {
    let vert_library = load_library(device, vert_lib_bytes)
        .map_err(|e| format!("failed to load vertex metallib: {}", e))?;
    let frag_library = load_library(device, frag_lib_bytes)
        .map_err(|e| format!("failed to load fragment metallib: {}", e))?;

    let vert_fn = vert_library
        .newFunctionWithName(&ns_str("vertex_main"))
        .ok_or("vertex_main not found in metallib")?;

    let frag_fn = frag_library
        .newFunctionWithName(&ns_str("fragment_main"))
        .ok_or("fragment_main not found in metallib")?;

    // GPU-driven static pass: when the world's fragment shader provides a
    // `fragment_main_bindless` entry point (default.metal does), the static
    // main pass reads each object's model matrix, material, and texture
    // indices from one GpuObjectData buffer and a bindless texture pool --
    // every static draw call then carries no per-draw state. Shaders
    // without it (custom shaders) fall back to the legacy
    // per-draw binding path in `draw_frame`.
    let bindless_frag_fn = frag_library.newFunctionWithName(&ns_str("fragment_main_bindless"));
    let bindless = bindless_frag_fn.is_some();
    let main_frag_fn = bindless_frag_fn.as_deref().unwrap_or(&*frag_fn);

    let pipeline_desc = MTLRenderPipelineDescriptor::new();
    pipeline_desc.setVertexDescriptor(Some(vert_desc));
    pipeline_desc.setVertexFunction(Some(&vert_fn));
    pipeline_desc.setFragmentFunction(Some(main_frag_fn));
    // Off-screen HDR pass: RGBA16Float colour + 4x MSAA. Output is linear
    // light; ACES tonemap + gamma + FXAA run in the composite pass.
    pipeline_desc.setRasterSampleCount(HDR_SAMPLE_COUNT as usize);
    unsafe {
        pipeline_desc
            .colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    if bindless {
        pipeline_desc.setSupportIndirectCommandBuffers(true);
    }

    let pipeline_state = device
        .newRenderPipelineStateWithDescriptor_error(&pipeline_desc)
        .map_err(|e| format!("failed to create pipeline state: {:?}", e))?;

    let (cull_pipeline, cull_icb_arg_encoder, cull_pipeline_phase2, cull_icb2_arg_encoder) =
        if bindless {
            let cull = build_cull_pipeline(device, hot_reload)?;
            (
                Some(cull.state),
                Some(cull.icb_arg_encoder),
                Some(cull.state_phase2),
                Some(cull.icb2_arg_encoder),
            )
        } else {
            (None, None, None, None)
        };

    // Argument encoder for the bindless pass's `BindlessTextures` buffer.
    // Derived from `fragment_main_bindless`'s buffer(7) parameter.
    let bindless_tex_arg_encoder = if bindless {
        // SAFETY: BINDLESS_TEXTURE_ARG_BUFFER_INDEX is the static buffer
        // index `fragment_main_bindless` declares its argument buffer at.
        Some(unsafe {
            main_frag_fn.newArgumentEncoderWithBufferIndex(BINDLESS_TEXTURE_ARG_BUFFER_INDEX)
        })
    } else {
        None
    };

    Ok(MainPipelineBundle {
        pipeline_state,
        bindless,
        cull_pipeline,
        cull_icb_arg_encoder,
        cull_pipeline_phase2,
        cull_icb2_arg_encoder,
        bindless_tex_arg_encoder,
    })
}

// Optional instanced pipeline: pairs vertex_main_instanced with the existing
// fragment_main. Built only when both an instanced vertex shader payload is
// supplied AND at least one cluster needs to render.
pub(crate) fn build_instanced_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    vert_instanced_lib_bytes: &[u8],
    frag_lib_bytes: &[u8],
    has_clusters: bool,
) -> Result<Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>, String> {
    if vert_instanced_lib_bytes.is_empty() || !has_clusters {
        return Ok(None);
    }

    let inst_library = load_library(device, vert_instanced_lib_bytes)
        .map_err(|e| format!("failed to load instanced vertex metallib: {}", e))?;
    let inst_vert_fn = inst_library
        .newFunctionWithName(&ns_str("vertex_main_instanced"))
        .ok_or("vertex_main_instanced not found in instanced metallib")?;

    // The instanced pipeline always pairs with the per-draw fragment_main
    // (bindless is static-only).
    let frag_library = load_library(device, frag_lib_bytes)
        .map_err(|e| format!("failed to load fragment metallib: {}", e))?;
    let frag_fn = frag_library
        .newFunctionWithName(&ns_str("fragment_main"))
        .ok_or("fragment_main not found in metallib")?;

    let inst_pipeline_desc = MTLRenderPipelineDescriptor::new();
    inst_pipeline_desc.setVertexDescriptor(Some(vert_desc));
    inst_pipeline_desc.setVertexFunction(Some(&inst_vert_fn));
    inst_pipeline_desc.setFragmentFunction(Some(&frag_fn));
    inst_pipeline_desc.setRasterSampleCount(HDR_SAMPLE_COUNT as usize);
    unsafe {
        inst_pipeline_desc
            .colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    inst_pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    let ps = device
        .newRenderPipelineStateWithDescriptor_error(&inst_pipeline_desc)
        .map_err(|e| format!("failed to create instanced pipeline state: {:?}", e))?;
    Ok(Some(ps))
}

// Shadow pipeline: depth-only, no fragment function, no MSAA. Compiled from the
// engine-internal `shadow_map.metal` source (entry `shadow_vertex_main`). Shared
// by init (one-shot at startup) and the internal-shader hot-reload path
// (`reload_shaders`, rebuild on `.metal` save) so the two stay consistent.
pub(crate) fn build_shadow_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "shadow_map.metal");
    let shadow_lib = compile_library(device, msl.as_ref(), "shadow_map")?;
    let shadow_fn = shadow_lib
        .newFunctionWithName(&ns_str("shadow_vertex_main"))
        .ok_or("shadow_vertex_main not found in shadow library")?;
    let shadow_pipeline_desc = MTLRenderPipelineDescriptor::new();
    shadow_pipeline_desc.setVertexDescriptor(Some(vert_desc));
    shadow_pipeline_desc.setVertexFunction(Some(&shadow_fn));
    shadow_pipeline_desc.setRasterSampleCount(1);
    shadow_pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&shadow_pipeline_desc)
        .map_err(|e| format!("failed to create shadow pipeline state: {:?}", e))
}

// GPU-driven cascaded-shadow render pipeline: depth-only, no
// fragment, no MSAA, but `supportIndirectCommandBuffers` so each cascade's
// casters can draw through the shadow ICB the `cull_encode_shadow` kernel
// fills. Entry `shadow_vertex_bindless` reads the per-object model matrix from
// the GpuObjectData buffer at buffer(9) by `[[base_instance]]` (the record id
// the cull baked), exactly like the main bindless `vertex_main`. Reuses the
// full static vertex descriptor (the VS consumes only attribute(0) = position;
// the deformed skinned tail shares the same 56-byte layout).
pub(crate) fn build_shadow_bindless_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vert_desc: &MTLVertexDescriptor,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "shadow_map.metal");
    let shadow_lib = compile_library(device, msl.as_ref(), "shadow_map")?;
    let shadow_fn = shadow_lib
        .newFunctionWithName(&ns_str("shadow_vertex_bindless"))
        .ok_or("shadow_vertex_bindless not found in shadow library")?;
    let shadow_pipeline_desc = MTLRenderPipelineDescriptor::new();
    shadow_pipeline_desc.setVertexDescriptor(Some(vert_desc));
    shadow_pipeline_desc.setVertexFunction(Some(&shadow_fn));
    shadow_pipeline_desc.setRasterSampleCount(1);
    shadow_pipeline_desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    shadow_pipeline_desc.setSupportIndirectCommandBuffers(true);
    device
        .newRenderPipelineStateWithDescriptor_error(&shadow_pipeline_desc)
        .map_err(|e| format!("failed to create shadow bindless pipeline state: {:?}", e))
}

// Depth-stencil state: less-than test, writes enabled (shared for main and
// shadow pass).
pub(crate) fn make_depth_state(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, String> {
    let depth_desc = MTLDepthStencilDescriptor::new();
    depth_desc.setDepthCompareFunction(MTLCompareFunction::Less);
    depth_desc.setDepthWriteEnabled(true);
    device
        .newDepthStencilStateWithDescriptor(&depth_desc)
        .ok_or_else(|| "failed to create depth stencil state".to_string())
}

// Read-only depth-stencil state: less-or-equal test, no write. Translucent
// passes (volumetric raymarch) bind this so they early-z against nearer
// opaque geometry without touching the depth buffer. A non-nil state is
// required: Metal's validation layer asserts on `setDepthStencilState(nil)`.
pub(crate) fn make_depth_state_read_only(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLDepthStencilState>>, String> {
    let depth_desc = MTLDepthStencilDescriptor::new();
    depth_desc.setDepthCompareFunction(MTLCompareFunction::LessEqual);
    depth_desc.setDepthWriteEnabled(false);
    device
        .newDepthStencilStateWithDescriptor(&depth_desc)
        .ok_or_else(|| "failed to create read-only depth stencil state".to_string())
}
