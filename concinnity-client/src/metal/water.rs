// src/metal/water.rs
//
// Water is a producer for the engine's transparent pass (`PassId::Transparent`:
// after SsrResolve, before TaaResolve / Upscale). It contributes one
// `TransparentDraw` per `WaterSurface`; the shared `encode_transparent`
// encoder owns the render pass, the scene snapshot, and back-to-front sorting.
//
// For each surface the vertex shader displaces a flat tessellated quad by a sum
// of Gerstner waves; the fragment shader composites:
//   * Refraction: sample the pre-transparent scene snapshot at a
//     normal-perturbed screen UV.
//   * Tint: shallow→deep colour mix by water-column thickness derived from
//     the difference between the main depth and the water surface depth.
//   * Foam: a soft mask where the seabed is just below the surface.
//   * Reflection: IBL prefilter cubemap at the reflected view direction
//     (with a hand-tuned sky fallback when no EnvironmentMap is bound).
//   * Fresnel: Schlick-power mix of refraction-tinted vs. reflected colour.
// Output blends with SRC_ALPHA / ONE_MINUS_SRC_ALPHA into `scene_pre_taa`.
//
// Refraction samples `hdr_targets.transparent_scene_copy` (the snapshot the
// transparent encoder blits from the current scene-pre-taa before drawing) so
// water renders correctly whether or not SSR produced a distinct scene texture
// (with SSR off, scene-pre-taa aliases `hdr_resolve`, and sampling it directly
// would be reading the attachment being written).

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

use crate::assets::{MAX_WATER_WAVES, WaterSurface, WaterWave};
use crate::geometry::water_grid::build_water_grid;
use crate::gfx::mesh_payload::Vertex;

use super::context::MtlContext;
use super::pipeline::{ns_str, shader_source};
use super::transparent::{TransparentDraw, bytes_of};
use super::uniforms::{TransparentView, WATER_MAX_WAVES, WaterParams, WaterWaveGpu};

// Per-surface GPU state: a static tessellated grid VB + IB.
pub(in crate::metal) struct WaterSurfaceRecord {
    pub(in crate::metal) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(in crate::metal) index_count: u32,
    pub(in crate::metal) params: WaterParams,
    // Planar reflection slot this surface samples (index into the
    // `PlanarReflectionSet`). `None` when the world has no planar set or this
    // surface's plane overflowed the budget; the shader then keeps the probe/sky
    // path. Assigned at init by `assign_planar_slots`.
    pub(in crate::metal) planar_slot: Option<usize>,
}

// Build a [`WaterSurfaceRecord`] for one `WaterSurface` asset. Calls the
// shared `geometry::water_grid` to produce the tessellated mesh and uploads
// it once; per-frame uniforms (time, view) come in through the encoder.
pub(in crate::metal) fn build_water_surface_record(
    device: &ProtocolObject<dyn MTLDevice>,
    surface: &WaterSurface,
) -> Result<WaterSurfaceRecord, String> {
    // Synthesise the args the shared generator consumes from the asset.
    let args = serde_json::json!({
        "half_width": surface.extent[0],
        "half_depth": surface.extent[1],
        "subdivisions": surface.subdivisions,
    });
    let (verts, idxs) = build_water_grid(&args)?;

    // Flatten into the standard Vertex layout. Tangent + colour are filled
    // with placeholders since the water shader rebuilds the normal frame
    // analytically and the fragment ignores per-vertex colour.
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
            .ok_or("water vertex buffer: source pointer is null")?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate water vertex buffer")?
    };
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(idxs.as_ptr() as *mut _)
            .ok_or("water index buffer: source pointer is null")?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate water index buffer")?
    };

    let params = water_params_from(surface, 0.0);

    Ok(WaterSurfaceRecord {
        vertex_buffer: vb,
        index_buffer: ib,
        index_count: idxs.len() as u32,
        params,
        // Patched after `assign_planar_slots` runs over all reflectors in init.
        planar_slot: None,
    })
}

fn water_params_from(surface: &WaterSurface, prefilter_mip_count: f32) -> WaterParams {
    let mut waves = [WaterWaveGpu::default(); WATER_MAX_WAVES];
    for (slot, src) in waves.iter_mut().zip(surface.waves.iter()) {
        *slot = wave_to_gpu(src);
    }
    let wave_count = surface.waves.len().min(MAX_WATER_WAVES) as u32;
    WaterParams {
        centre: [surface.centre[0], surface.centre[1], surface.centre[2], 0.0],
        deep_colour: [
            surface.deep_colour[0],
            surface.deep_colour[1],
            surface.deep_colour[2],
            0.0,
        ],
        shallow_colour: [
            surface.shallow_colour[0],
            surface.shallow_colour[1],
            surface.shallow_colour[2],
            0.0,
        ],
        depth_falloff: surface.depth_falloff_metres,
        foam_width: surface.foam_width_metres,
        foam_intensity: surface.foam_intensity,
        fresnel_power: surface.fresnel_power,
        roughness: surface.roughness,
        refraction_strength: surface.refraction_strength,
        wave_count,
        prefilter_mip_count,
        waves,
        // Planar reflection off by default; `collect_water_transparent_draws`
        // patches `planar.x` to 1 (and the distortion scale) when the planar
        // pass ran this frame.
        planar: [0.0, 0.0, 0.0, 0.0],
    }
}

// Wave-normal screen-space distortion scale for the planar reflection sample.
// Small: the planar reflection is a flat-plane render, so the wave normal only
// perturbs the lookup a little to fake ripple displacement.
const PLANAR_DISTORTION: f32 = 0.03;

fn wave_to_gpu(w: &WaterWave) -> WaterWaveGpu {
    WaterWaveGpu {
        dir_amp_wave: [w.direction[0], w.direction[1], w.amplitude, w.wavelength],
        speed_steep_pad: [w.speed, w.steepness, 0.0, 0.0],
    }
}

impl MtlContext {
    // Contribute one [`TransparentDraw`] per water surface to the transparent
    // pass. The shared `encode_transparent` encoder owns the render pass, the
    // scene snapshot, back-to-front sorting, and the shared reflection bindings
    // (prefilter cube + probe cubes + probe set + cube sampler). Each draw binds
    // the snapshot (refraction source) at texture(0) and the resolved main depth
    // at texture(1). Sampling the snapshot rather than `hdr_resolve` is what lets
    // water render with SSR off.
    pub(in crate::metal) fn collect_water_transparent_draws(
        &self,
        view: &TransparentView,
        bindless: bool,
        planar_live: bool,
        out: &mut Vec<TransparentDraw>,
    ) {
        // Pipeline selection (matched by `encode_transparent`'s binding):
        //   RT on + bindless world  -> textured RT trace (bindless albedo)
        //   RT on                   -> flat RT trace (per-object tint)
        //   RT off                  -> box-projected probe cube / sky prefilter
        // `rt.accel` live means RT is on; `bindless` means the texture pool
        // exists. Falls back through to the probe pipeline.
        let rt_on = self.rt.accel.is_some();
        let pipeline = match (
            rt_on && bindless,
            &self.water_pipeline_rt_textured,
            rt_on,
            &self.water_pipeline_rt,
        ) {
            (true, Some(p), _, _) => p,
            (_, _, true, Some(p)) => p,
            _ => match &self.water_pipeline {
                Some(p) => p,
                None => return,
            },
        };
        let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
        let cam = view.camera_pos;
        let planar_set = self.planar_reflection.as_ref();
        for surface in &self.water_surfaces {
            // Rebuild params with the current prefilter mip count; everything
            // else is asset-side-static.
            let mut params = surface.params;
            params.prefilter_mip_count = prefilter_mip_count;
            let mut fragment_textures = vec![
                // The refraction snapshot (texture 0) + resolved main depth
                // (texture 1). The IBL prefilter cube (texture 2), probe cubes
                // (texture 3..), cube sampler (sampler 1), and probe set are bound
                // globally by `encode_transparent` (shared with glass).
                (0, self.hdr_targets.transparent_scene_copy.clone()),
                (1, self.hdr_targets.depth_resolve.clone()),
            ];
            // Select the sharp planar reflection when the planar pass ran this
            // frame and this surface was assigned a slot; bind that slot's resolve
            // at texture(11). Otherwise the shader keeps the probe / sky path.
            if planar_live
                && let Some(targets) = surface
                    .planar_slot
                    .and_then(|s| planar_set.and_then(|set| set.targets.get(s)))
            {
                params.planar = [1.0, PLANAR_DISTORTION, 0.0, 0.0];
                fragment_textures.push((11, targets.resolve.clone()));
            }
            let c = params.centre;
            let sort_distance =
                ((c[0] - cam[0]).powi(2) + (c[1] - cam[1]).powi(2) + (c[2] - cam[2]).powi(2))
                    .sqrt();
            out.push(TransparentDraw {
                pipeline: pipeline.clone(),
                vertex_buffer: surface.vertex_buffer.clone(),
                index_buffer: surface.index_buffer.clone(),
                index_count: surface.index_count,
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
}

// Build the water render pipeline. Standard 5-attribute vertex layout at
// buffer(1); the same descriptor the main pass uses, so any
// `ProceduralMesh::water_grid` mesh can bind directly. Output target is
// `scene_pre_taa` (RGBA16Float single-sample); SRC_ALPHA blend writes the
// transparent water on top of whatever the SsrResolve pass produced.
pub(super) fn build_water_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_water_pipeline_from(device, hot_reload, "water.metal", "water_fragment")
}

// Build the ray-traced water pipeline: the same vertex layout + blend, but the
// `water_fragment_rt` fragment (water_rt.metal) traces a sharp reflection ray
// against the scene acceleration structure instead of sampling a probe cube.
// Compiled only on RT-capable devices (the shader uses `metal_raytracing`);
// selected per-frame only while `self.rt.accel` is live, the probe pipeline
// otherwise. This is the FLAT variant (per-object material tint as albedo).
pub(super) fn build_water_pipeline_rt(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_water_pipeline_from(device, hot_reload, "water_rt.metal", "water_fragment_rt")
}

// Build the textured ray-traced water pipeline: the same trace as
// `water_fragment_rt`, but the reflected hit's albedo / normal / emissive are
// sampled from the bindless texture pool (buffer 10) instead of a flat
// per-object tint. Selected over the flat variant only in a bindless world.
pub(super) fn build_water_pipeline_rt_textured(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    build_water_pipeline_from(
        device,
        hot_reload,
        "water_rt.metal",
        "water_fragment_rt_textured",
    )
}

// Shared water pipeline builder: every variant uses the identical `water_vertex`
// + vertex descriptor + blend state and differs only in its shader source file
// and fragment entry point.
fn build_water_pipeline_from(
    device: &ProtocolObject<dyn MTLDevice>,
    hot_reload: bool,
    shader_name: &str,
    fragment_entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, shader_name);
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("{} compile error: {:?}", shader_name, e))?;

    let vert_fn = library
        .newFunctionWithName(&ns_str("water_vertex"))
        .ok_or("water_vertex not found")?;
    let frag_fn = library
        .newFunctionWithName(&ns_str(fragment_entry))
        .ok_or_else(|| format!("{} not found", fragment_entry))?;

    // Standard mesh vertex layout (pos / normal / tangent / colour / uv at
    // buffer(1)). Stride = sizeof(Vertex) = 56 bytes.
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
        .map_err(|e| format!("failed to create water pipeline state: {:?}", e))
}
