// src/metal/decal.rs
//
// Per-frame encoder for the projected (deferred) decal pass. Runs after the
// main HDR pass has resolved into `hdr_targets.hdr_resolve` and before SSR /
// TAA pick the resolved scene up: so a decal is reflected by SSR and tracked
// by TAA's history just like the rest of the scene.
//
// Each decal is drawn as a unit cube (positions in `[-0.5, 0.5]^3`) transformed
// by its world model matrix and the camera VP; the fragment shader samples the
// main pass's MSAA depth attachment to reconstruct the world-space sample
// point at each pixel and tests it against the decal's local bounding box,
// stamping the texture onto whatever surface fills the box.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLCommandBuffer as _, MTLDevice as _, MTLIndexType, MTLLibrary as _,
    MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLStoreAction,
    MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction,
};

use super::context::MtlContext;
use super::pipeline::{ns_str, shader_source};
use super::scoped_encoder::ScopedEncoder;
use super::uniforms::{DecalParams, DecalView};
use crate::gfx::decal::DecalRecord;

// All projected-decal state grouped into one feature unit: the decal records
// (with their tombstone free-list), the pipeline, the shared unit-cube
// geometry, and the sampler. The pipeline / cube buffers / sampler are built
// lazily either at init (≥1 declared decal) or on the first runtime
// [`MtlContext::add_decal`]; they stay `None` only when the world has never
// had a decal, in which case the pass is skipped before iteration.
pub(crate) struct DecalState {
    // One slot per decal; `None` slots are tombstones from
    // [`MtlContext::remove_decal`], reused by the next add via `free_slots`.
    pub records: Vec<Option<DecalRecord>>,
    // Tombstoned slot indices, reused by the next add so a spawn/despawn
    // cycle does not grow `records` without bound.
    pub free_slots: Vec<usize>,
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub cube_vertex_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    pub cube_index_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    pub sampler: Option<Retained<ProtocolObject<dyn objc2_metal::MTLSamplerState>>>,
}

impl MtlContext {
    // Encode the projected-decal pass. Caller has ended the main pass, so
    // `hdr_targets.depth` (MSAA) holds the scene depth and
    // `hdr_targets.hdr_resolve` holds the resolved scene colour. The pass
    // alpha-blends one textured stamp per decal into `hdr_resolve`.
    //
    // `vp` is the same view-projection the main pass rasterised with:
    // jittered when TAA is on, so the reconstructed world position lands on
    // the same pixel the main pass shaded.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_decals(
        &self,
        cmd_buf: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        vp: [[f32; 4]; 4],
        // Inverse of `vp`, computed once in `draw_frame` and shared across the
        // depth-reconstruction passes; see `GraphFrameParams::inv_vp`.
        inv_vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) -> Result<u32, String> {
        let pipeline = match &self.decal.pipeline {
            Some(p) => p,
            None => return Ok(0),
        };
        if self.decal.records.is_empty() {
            return Ok(0);
        }
        // Visibility-cull first so a world where every decal lands off-screen
        // skips the whole pass, including opening the render encoder. We
        // pre-compute the visible mask and bail when nothing is left.
        // Tombstoned (None) slots are always invisible.
        let visible: Vec<bool> = self
            .decal
            .records
            .iter()
            .map(|slot| match slot {
                Some(d) => {
                    let (mn, mx) = d.aabb();
                    frustum.intersects_aabb(mn, mx)
                }
                None => false,
            })
            .collect();
        if !visible.iter().any(|v| *v) {
            return Ok(0);
        }
        let vbuf = self
            .decal
            .cube_vertex_buffer
            .as_ref()
            .ok_or("decal cube vertex buffer missing")?;
        let ibuf = self
            .decal
            .cube_index_buffer
            .as_ref()
            .ok_or("decal cube index buffer missing")?;
        let sampler = self.decal.sampler.as_ref().ok_or("decal sampler missing")?;

        let viewport = [
            self.hdr_targets.width as f32,
            self.hdr_targets.height as f32,
        ];
        let view = DecalView {
            vp,
            inv_vp,
            viewport,
            _pad: [0.0; 2],
        };

        let pass_desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
        }

        if let Some(t) = &self.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Decals);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get decal render encoder")?,
            "decals",
        );
        enc.setRenderPipelineState(pipeline);

        unsafe {
            // Per-frame view inputs at buffer(0); rebound once.
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(&view).cast(),
                std::mem::size_of::<DecalView>(),
                0,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&view).cast(),
                std::mem::size_of::<DecalView>(),
                0,
            );
            // Unit-cube vertices at vertex buffer(2); the vertex shader declares
            // a single `[[attribute(0)]] float3` mapped to buffer(2) by the
            // pipeline's vertex descriptor.
            enc.setVertexBuffer_offset_atIndex(Some(vbuf), 0, 2);
            // Decal sampler at fragment sampler(0); texture(0) is the MSAA
            // scene depth.
            // Sample the single-sample `depth_resolve` (post-
            // Main depth, plus any raymarched surface depth) instead
            // of the MSAA original. Lets decals project correctly onto
            // raymarched surfaces.
            enc.setFragmentTexture_atIndex(Some(self.hdr_targets.depth_resolve.as_ref()), 0);
            enc.setFragmentSamplerState_atIndex(Some(sampler), 0);
        }

        let last_tex = self.textures.len().saturating_sub(1);
        let mut draw_calls: u32 = 0;
        for (i, slot) in self.decal.records.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            let d = match slot {
                Some(d) => d,
                None => continue,
            };
            let params = DecalParams {
                model: d.model,
                inv_model: d.inv_model,
                tint: d.tint,
                fade_pow: 2.0,
                _pad0: 0.0,
                _pad1: 0.0,
                _pad2: 0.0,
            };
            let slot = d.texture_slot.min(last_tex);
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&params).cast(),
                    std::mem::size_of::<DecalParams>(),
                    1,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&params).cast(),
                    std::mem::size_of::<DecalParams>(),
                    1,
                );
                enc.setFragmentTexture_atIndex(Some(self.textures[slot].as_ref()), 1);
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    36,
                    MTLIndexType::UInt16,
                    ibuf,
                    0,
                );
            }
            draw_calls += 1;
        }

        Ok(draw_calls)
    }
}

// Build the projected-decal pipeline. The pass runs after the main HDR pass:
// a per-decal unit cube is rasterised, and the fragment shader reconstructs
// the world-space sample point at each pixel from the main pass's MSAA depth
// attachment, transforms it into decal-local space, and stamps the decal
// texture onto whatever sits inside the unit box. The output is alpha-blended
// into the resolved HDR target (`hdr_resolve`).
//
// Depth state is `Always` / no write -- every rasterised pixel inside the box
// is a candidate; the shader's own bounds test does the volumetric culling.
pub(super) fn build_decal_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "decal.metal");

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("decal shader compile error: {:?}", e))?;

    let vert_fn = library
        .newFunctionWithName(&ns_str("decal_vertex"))
        .ok_or("decal_vertex not found")?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("decal_fragment"))
        .ok_or("decal_fragment not found")?;

    // Vertex layout: a single float3 position at buffer(2). The cube buffer
    // holds 8 unit-cube corners in [-0.5, 0.5]^3.
    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        let attr0 = vert_desc.attributes().objectAtIndexedSubscript(0);
        attr0.setFormat(MTLVertexFormat::Float3);
        attr0.setOffset(0);
        attr0.setBufferIndex(2);
        let layout = vert_desc.layouts().objectAtIndexedSubscript(2);
        layout.setStride(12);
        layout.setStepFunction(MTLVertexStepFunction::PerVertex);
    }

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(1);
    unsafe {
        let ca = desc.colorAttachments().objectAtIndexedSubscript(0);
        ca.setPixelFormat(MTLPixelFormat::RGBA16Float);
        // Standard premultiplied-style over blend; the fragment writes the
        // sampled texture x tint with its own alpha as the blend weight.
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }
    // No depth attachment on this pass; depth testing happens analytically in
    // the fragment via the unit-box bounds check against reconstructed world.

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create decal pipeline state: {:?}", e))
}
