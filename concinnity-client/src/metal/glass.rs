// src/metal/glass.rs
//
// GlassPanel: the simplest producer for the engine's transparent pass. Each
// panel is a flat world-space quad (built once at init) that contributes one
// [`TransparentDraw`] per frame. The shared `encode_transparent` encoder sorts
// it back-to-front against water + other panels and draws it; the fragment
// shader refracts the pre-transparent scene snapshot, tints it, and adds a
// Fresnel rim (see shaders/glass.metal).

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBuffer, MTLDevice, MTLLibrary as _, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLVertexDescriptor,
    MTLVertexFormat, MTLVertexStepFunction,
};

use crate::assets::GlassPanel;
use crate::geometry::glass_quad::build_glass_quad;
use crate::gfx::mesh_payload::Vertex;

use super::context::MtlContext;
use super::pipeline::{ns_str, shader_source};
use super::transparent::{TransparentDraw, bytes_of};
use super::uniforms::{GlassParams, TransparentView};

// Per-panel GPU state: the static world-space quad VB + IB plus the per-panel
// uniform block. The quad is pre-transformed at build time, so there is no
// per-frame vertex work beyond projection.
pub(in crate::metal) struct GlassPanelRecord {
    pub(in crate::metal) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_count: u32,
    pub(in crate::metal) params: GlassParams,
    pub(in crate::metal) visible: bool,
    // World-space centre, used for the back-to-front camera-distance sort.
    pub(in crate::metal) centre: [f32; 3],
}

fn glass_params_from(panel: &GlassPanel) -> GlassParams {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParams {
        centre: [panel.centre[0], panel.centre[1], panel.centre[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        _pad: 0.0,
    }
}

// Build the GPU record for one `GlassPanel`: generate the quad, upload it, and
// snapshot the per-panel uniforms.
pub(in crate::metal) fn build_glass_panel_record(
    device: &ProtocolObject<dyn MTLDevice>,
    panel: &GlassPanel,
) -> Result<GlassPanelRecord, String> {
    let (verts, idxs) = build_glass_quad(panel.centre, panel.normal, panel.half_size);

    // Flatten into the standard Vertex layout. Tangent is a placeholder (the
    // glass shader rebuilds its frame from the panel normal) and per-vertex
    // colour is unused.
    let mut packed: Vec<Vertex> = Vec::with_capacity(verts.len());
    for (pos, normal, color, uv) in verts {
        packed.push(Vertex {
            pos,
            normal,
            tangent: [1.0, 0.0, 0.0],
            color,
            uv,
        });
    }
    let vb_bytes = packed.len() * std::mem::size_of::<Vertex>();
    let ib_bytes = idxs.len() * std::mem::size_of::<u16>();

    let vb = unsafe {
        let ptr = std::ptr::NonNull::new(packed.as_ptr() as *mut _)
            .ok_or("glass vertex buffer: source pointer is null")?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate glass vertex buffer")?
    };
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(idxs.as_ptr() as *mut _)
            .ok_or("glass index buffer: source pointer is null")?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate glass index buffer")?
    };

    Ok(GlassPanelRecord {
        vertex_buffer: vb,
        index_buffer: ib,
        index_count: idxs.len() as u32,
        params: glass_params_from(panel),
        visible: panel.visible,
        centre: panel.centre,
    })
}

// Build the shared glass render pipeline. Standard 5-attribute vertex layout
// at buffer(1) (same as water + the main pass); SRC_ALPHA blend into the
// RGBA16Float scene-pre-taa target, no depth attachment.
pub(super) fn build_glass_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "glass.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("glass shader compile error: {:?}", e))?;

    let vert_fn = library
        .newFunctionWithName(&ns_str("glass_vertex"))
        .ok_or("glass_vertex not found")?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("glass_fragment"))
        .ok_or("glass_fragment not found")?;

    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        let attr0 = vert_desc.attributes().objectAtIndexedSubscript(0);
        attr0.setFormat(MTLVertexFormat::Float3);
        attr0.setOffset(0);
        attr0.setBufferIndex(1);

        let attr1 = vert_desc.attributes().objectAtIndexedSubscript(1);
        attr1.setFormat(MTLVertexFormat::Float3);
        attr1.setOffset(12);
        attr1.setBufferIndex(1);

        let attr2 = vert_desc.attributes().objectAtIndexedSubscript(2);
        attr2.setFormat(MTLVertexFormat::Float3);
        attr2.setOffset(24);
        attr2.setBufferIndex(1);

        let attr3 = vert_desc.attributes().objectAtIndexedSubscript(3);
        attr3.setFormat(MTLVertexFormat::Float3);
        attr3.setOffset(36);
        attr3.setBufferIndex(1);

        let attr4 = vert_desc.attributes().objectAtIndexedSubscript(4);
        attr4.setFormat(MTLVertexFormat::Float2);
        attr4.setOffset(48);
        attr4.setBufferIndex(1);

        let layout1 = vert_desc.layouts().objectAtIndexedSubscript(1);
        layout1.setStride(std::mem::size_of::<Vertex>());
        layout1.setStepFunction(MTLVertexStepFunction::PerVertex);
    }

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create glass pipeline state: {:?}", e))
}

impl MtlContext {
    // Contribute one [`TransparentDraw`] per visible glass panel. The shared
    // transparent encoder owns sorting + the scene-copy snapshot; each draw
    // binds the snapshot (refraction source) at texture(0) and the resolved
    // depth at texture(1).
    pub(in crate::metal) fn collect_glass_transparent_draws(
        &self,
        view: &TransparentView,
        out: &mut Vec<TransparentDraw>,
    ) {
        let pipeline = match &self.glass_pipeline {
            Some(p) => p,
            None => return,
        };
        let cam = view.camera_pos;
        for panel in &self.glass_panels {
            if !panel.visible {
                continue;
            }
            let c = panel.centre;
            let sort_distance =
                ((c[0] - cam[0]).powi(2) + (c[1] - cam[1]).powi(2) + (c[2] - cam[2]).powi(2))
                    .sqrt();
            out.push(TransparentDraw {
                pipeline: pipeline.clone(),
                vertex_buffer: panel.vertex_buffer.clone(),
                index_buffer: panel.index_buffer.clone(),
                index_count: panel.index_count,
                params: bytes_of(&panel.params),
                fragment_textures: vec![
                    (0, self.hdr_targets.transparent_scene_copy.clone()),
                    (1, self.hdr_targets.depth_resolve.clone()),
                ],
                fragment_samplers: vec![(0, self.post_sampler.clone())],
                sort_distance,
            });
        }
    }
}
