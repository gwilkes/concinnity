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
use super::uniforms::{GlassMeshParams, GlassParams, TransparentView};

// Refraction offset + Fresnel falloff for a transparent glass MESH. A `Material`
// carries no glass-specific tunables (unlike a `GlassPanel`), so these match the
// GlassPanel defaults: a gentle screen-space refraction and a fresnel power of 1
// (subtle reflection head-on, full mirror at grazing).
const GLASS_MESH_REFRACTION: f32 = 0.02;
const GLASS_MESH_FRESNEL_POWER: f32 = 1.0;

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
    // Planar reflection slot this pane samples (index into the
    // `PlanarReflectionSet`). `None` when the world has no planar set or this
    // pane's plane overflowed the budget; the shader then keeps the probe/sky
    // path. Assigned at init by `assign_planar_slots`.
    pub(in crate::metal) planar_slot: Option<usize>,
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
        // Patched per-frame in `collect_glass_transparent_draws`.
        prefilter_mip_count: 0.0,
        // Off by default; `collect_glass_transparent_draws` sets `planar.x = 1`
        // when the planar pass ran this frame and the pane has a slot.
        planar: [0.0, 0.0, 0.0, 0.0],
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
        // Patched after `assign_planar_slots` runs over all reflectors in init.
        planar_slot: None,
    })
}

// Build the shared glass render pipeline. Standard 5-attribute vertex layout
// at buffer(1) (same as water + the main pass); SRC_ALPHA blend into the
// RGBA16Float scene-pre-taa target, no depth attachment.
pub(super) fn build_glass_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_glass_pipeline_from(
        device,
        hot_reload,
        "glass.metal",
        "glass_vertex",
        "glass_fragment",
    )
}

// Build the ray-traced glass pipeline: the same vertex layout + blend, but the
// `glass_fragment_rt` fragment (glass_rt.metal) traces a sharp reflection ray
// against the scene acceleration structure instead of sampling a probe cube.
// Compiled only on RT-capable devices (the shader uses `metal_raytracing`);
// selected per-frame only while `self.rt.accel` is live, the probe pipeline
// otherwise.
pub(super) fn build_glass_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_glass_pipeline_from(
        device,
        hot_reload,
        "glass_rt.metal",
        "glass_vertex",
        "glass_fragment_rt",
    )
}

// Build the ray-traced transparent glass MESH pipeline (`glass_mesh_rt.metal`):
// the same 5-attribute vertex layout + blend as the pane pipelines, but the
// `glass_mesh_vertex` stage applies a per-draw model matrix and the fragment uses
// the interpolated mesh normal. Compiled only on RT-capable devices. Drives the
// FLAT trace (reflected-hit material tint as albedo).
pub(super) fn build_glass_mesh_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_glass_pipeline_from(
        device,
        hot_reload,
        "glass_mesh_rt.metal",
        "glass_mesh_vertex",
        "glass_mesh_fragment_rt",
    )
}

// The textured transparent glass MESH variant: reflected hits sample the bindless
// pool (buffer 10). Selected over the flat variant only in a bindless world.
pub(super) fn build_glass_mesh_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_glass_pipeline_from(
        device,
        hot_reload,
        "glass_mesh_rt.metal",
        "glass_mesh_vertex",
        "glass_mesh_fragment_rt_textured",
    )
}

// Build the textured ray-traced glass pipeline: the same trace as
// `glass_fragment_rt`, but the reflected hit's albedo / normal / emissive are
// sampled from the bindless texture pool (buffer 10) instead of a flat
// per-object tint. Selected over the flat variant only in a bindless world.
pub(super) fn build_glass_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_glass_pipeline_from(
        device,
        hot_reload,
        "glass_rt.metal",
        "glass_vertex",
        "glass_fragment_rt_textured",
    )
}

// Shared glass pipeline builder: both variants use the identical `glass_vertex`
// + vertex descriptor + blend state and differ only in their shader source file
// and fragment entry point.
fn build_glass_pipeline_from(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    shader_name: &str,
    vertex_entry: &str,
    fragment_entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, shader_name);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("{} compile error: {:?}", shader_name, e))?;

    let vert_fn = library
        .newFunctionWithName(&ns_str(vertex_entry))
        .ok_or_else(|| format!("{} not found", vertex_entry))?;
    let frag_fn = library
        .newFunctionWithName(&ns_str(fragment_entry))
        .ok_or_else(|| format!("{} not found", fragment_entry))?;

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
    // depth at texture(1). When `planar_live` and the pane has a planar slot, the
    // draw also binds its slot's resolve at texture(11) and flips `params.planar`
    // so the shader samples the sharp planar reflection instead of the probe cube;
    // slotless panes (budget overflow, logged at init) keep the probe path.
    pub(in crate::metal) fn collect_glass_transparent_draws(
        &self,
        view: &TransparentView,
        bindless: bool,
        planar_live: bool,
        out: &mut Vec<TransparentDraw>,
    ) {
        // Pipeline selection (matched by `encode_transparent`'s binding):
        //   RT on + bindless world  -> textured RT trace (bindless albedo)
        //   RT on                   -> flat RT trace (per-object tint)
        //   RT off                  -> box-projected probe cube
        // `rt.accel` live means RT is on; `bindless` means the texture pool
        // exists. Falls back through to the probe pipeline.
        let rt_on = self.rt.accel.is_some();
        let pipeline = match (
            rt_on && bindless,
            &self.glass_pipeline_rt_textured,
            rt_on,
            &self.glass_pipeline_rt,
        ) {
            (true, Some(p), _, _) => p,
            (_, _, true, Some(p)) => p,
            _ => match &self.glass_pipeline {
                Some(p) => p,
                None => return,
            },
        };
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        let cam = view.camera_pos;
        let planar_set = self.planar_reflection.as_ref();
        for panel in &self.glass_panels {
            if !panel.visible {
                continue;
            }
            // Patch the live prefilter mip count (0 = no env map -> white rim);
            // the reflection-probe cubes + set are bound globally by
            // `encode_transparent`.
            let mut params = panel.params;
            params.prefilter_mip_count = prefilter_mip_count;
            let mut fragment_textures = vec![
                (0, self.hdr_targets.transparent_scene_copy.clone()),
                (1, self.hdr_targets.depth_resolve.clone()),
            ];
            // Select the sharp planar reflection when the planar pass ran this
            // frame and this pane was assigned a slot; bind that slot's resolve at
            // texture(11) (overriding the global default). Otherwise the shader
            // keeps the probe / sky path.
            if planar_live
                && let Some(targets) = panel
                    .planar_slot
                    .and_then(|s| planar_set.and_then(|set| set.targets.get(s)))
            {
                params.planar = [1.0, 0.0, 0.0, 0.0];
                fragment_textures.push((11, targets.resolve.clone()));
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
                index_type: objc2_metal::MTLIndexType::UInt16,
                index_offset_bytes: 0,
                base_vertex: 0,
                params: bytes_of(&params),
                fragment_textures,
                fragment_samplers: vec![(0, self.post_sampler.clone())],
                sort_distance,
            });
        }
    }

    // Whether any material opted into Layer 2 see-through glass AND the device can
    // drive it (the mesh pipeline built). Independent of `rt.accel`, so it answers
    // "would the see-through path run if RT is on" -- used at the RT-BLAS build,
    // which must exclude the meshes it will reroute before `rt.accel` itself is
    // assigned. Data-driven: see-through is opt-in per `Material::see_through`, so
    // a scene with no see-through material (e.g. Bistro) never engages Layer 2 and
    // its transparent glass stays Layer 1 (opaque, low roughness, reflective).
    pub(in crate::metal) fn seethrough_meshes_enabled(&self) -> bool {
        !self.seethrough_mesh_indices.is_empty() && self.glass_mesh_pipeline_rt.is_some()
    }

    // Whether the transparent-mesh (Layer 2) path is live this frame: a material
    // opted into see-through, the mesh pipeline was built, AND RT is on
    // (`rt.accel`, the per-pixel trace needs the BVH). When false, those meshes
    // render opaque + reflective in the main pass (Layer 1) and the producer /
    // opaque-skip / BLAS-exclude all stay inert.
    pub(in crate::metal) fn mesh_glass_active(&self) -> bool {
        self.seethrough_meshes_enabled() && self.rt.accel.is_some()
    }

    // Contribute one [`TransparentDraw`] per visible see-through glass MESH (Layer
    // 2): a `Material` flagged `see_through` (which implies `transparent`) on an
    // RT-capable device. Each mesh draws from the SHARED scene vertex/index buffers
    // via its `DrawObject` offsets + model matrix; the `glass_mesh_rt` shader traces
    // a per-pixel reflection off the interpolated mesh normal. A no-op unless RT is
    // live (`mesh_glass_active`); when inactive the meshes render opaque (Layer 1)
    // in the main pass. The same gate skips them in the opaque pass + the RT BLAS,
    // so a mesh neither double-draws nor self-reflects.
    pub(in crate::metal) fn collect_mesh_transparent_draws(
        &self,
        view: &TransparentView,
        bindless: bool,
        out: &mut Vec<TransparentDraw>,
    ) {
        if !self.mesh_glass_active() {
            return;
        }
        // Textured trace in a bindless world (reflected hits carry their textures),
        // else the flat trace (reflected-hit material tint). `mesh_glass_active`
        // guarantees the flat pipeline exists.
        let pipeline = match (bindless, &self.glass_mesh_pipeline_rt_textured) {
            (true, Some(p)) => p,
            _ => match &self.glass_mesh_pipeline_rt {
                Some(p) => p,
                None => return,
            },
        };
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        let cam = view.camera_pos;
        for &idx in &self.seethrough_mesh_indices {
            let Some(obj) = self.draw_objects.get(idx) else {
                continue;
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            let centre = [
                0.5 * (obj.bb_min[0] + obj.bb_max[0]),
                0.5 * (obj.bb_min[1] + obj.bb_max[1]),
                0.5 * (obj.bb_min[2] + obj.bb_max[2]),
            ];
            let d = ((centre[0] - cam[0]).powi(2)
                + (centre[1] - cam[1]).powi(2)
                + (centre[2] - cam[2]).powi(2))
            .sqrt();
            let (index_offset, index_count) = obj.active_lod(d);
            let t = obj.material.tint;
            let params = GlassMeshParams {
                model: obj.model,
                tint: [t[0], t[1], t[2], 0.0],
                opacity: obj.material.opacity,
                refraction_strength: GLASS_MESH_REFRACTION,
                fresnel_power: GLASS_MESH_FRESNEL_POWER,
                prefilter_mip_count,
            };
            out.push(TransparentDraw {
                pipeline: pipeline.clone(),
                vertex_buffer: self.vertex_buffer.clone(),
                index_buffer: self.index_buffer.clone(),
                index_count: index_count as u32,
                index_type: objc2_metal::MTLIndexType::UInt32,
                index_offset_bytes: index_offset * std::mem::size_of::<u32>(),
                base_vertex: obj.base_vertex,
                params: bytes_of(&params),
                fragment_textures: vec![
                    (0, self.hdr_targets.transparent_scene_copy.clone()),
                    (1, self.hdr_targets.depth_resolve.clone()),
                ],
                fragment_samplers: vec![(0, self.post_sampler.clone())],
                sort_distance: d,
            });
        }
    }
}
