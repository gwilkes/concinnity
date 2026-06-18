// src/metal/raymarch.rs
//
// Per-frame encoder for the raymarched SDF volume pass. Runs at
// `PassId::Raymarch`, between `AutoExposure` and `Decals` on the
// hdr_resolve RMW chain. Each `SdfVolume` rasterises the back faces of
// its world-space bounding box and runs a user-authored fragment
// shader that sphere-traces the SDF inside the box.
//
// Architecture:
//   * One MTLRenderPipelineState per `SdfVolume` (built lazily at init
//     from the engine-shipped helpers + the user's source bytes + the
//     engine-shipped template). The wrap order is helpers → user →
//     template so the template's `fragment_main` can call the user's
//     `map` and `shade` functions through the forward declarations the
//     helpers expose.
//   * One shared unit-cube VB+IB for the proxy geometry; 8 corners /
//     36 indices, allocated once at init. The encoder draws back faces
//     only (cull mode = Front) so we get exactly one fragment per pixel
//     inside the box regardless of whether the camera is outside or
//     inside it.
//   * Color attachment = `hdr_resolve` (LoadAction::Load, opaque write).
//     No depth attachment, matching the projected-decal pass. Depth
//     compositing is shader-side via the early-out against
//     `main_depth` (texture(0)).

#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder as _, MTLBuffer,
    MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCullMode, MTLDevice, MTLIndexType,
    MTLLibrary as _, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType, MTLRenderCommandEncoder as _,
    MTLRenderPassDescriptor, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLResourceOptions, MTLStoreAction, MTLVertexDescriptor, MTLVertexFormat,
    MTLVertexStepFunction,
};

use crate::assets::sdf_volume::{SDF_PARAMS_LEN, SdfVolume};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::LightUniforms;

use super::context::MtlContext;
use super::pipeline::ns_str;
use super::scoped_encoder::ScopedEncoder;

// Engine-shipped header / template prepended + appended around each
// user shader. The helpers carry the IQ primitive library + the cone-
// marcher + the PBR shading helpers; the template owns the vertex +
// fragment_main that drive the pass. Forward declarations of `map` /
// `shade` live in the helpers; the user shader sandwiched in the
// middle provides their definitions; the template appended at the end
// calls them.
const RAYMARCH_HELPERS_MSL: &str = include_str!("shaders/raymarch_helpers.metal");
const RAYMARCH_TEMPLATE_MSL: &str = include_str!("shaders/raymarch_template.metal");
// Depth-only shadow-caster template. Wrapped around the user source the same
// way as the main template (helpers → user → this), but compiled into a
// separate per-volume pipeline that rasterises the proxy cube through a CSM
// cascade's light VP and writes the cone-marched SDF hit depth. Mirrors
// `directx/shaders/raymarch_shadow.hlsl`.
const RAYMARCH_SHADOW_MSL: &str = include_str!("shaders/raymarch_shadow.metal");
// Volumetric raymarching template. Wrapped the same way as the main template
// (helpers → user → this) but the user provides `sampleVolume` instead of
// `map`/`shade`. The volume marches in fixed steps through the box,
// accumulating Beer-Lambert transmittance and in-scattered light. Output is
// alpha-blended; no depth write. Mirrors `directx/shaders/raymarch_volumetric_template.hlsl`.
const RAYMARCH_VOLUMETRIC_MSL: &str = include_str!("shaders/raymarch_volumetric_template.metal");

// Per-frame view inputs the raymarch pass binds at buffer(0). Layout
// matches `RaymarchView` in `shaders/raymarch_helpers.metal`. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::metal) struct RaymarchView {
    pub(in crate::metal) vp: [[f32; 4]; 4],
    pub(in crate::metal) inv_vp: [[f32; 4]; 4],
    // World-space camera position (xyz). `.w` is ignored.
    pub(in crate::metal) cam_pos: [f32; 4],
    // HDR target width / height in pixels: the shader divides
    // `position.xy` by this to read the depth attachment with
    // integer pixel coordinates.
    pub(in crate::metal) viewport: [f32; 2],
    // Wall-clock seconds since startup, available to the user SDF.
    pub(in crate::metal) time: f32,
    // Mip count of the bound IBL prefilter cube; 0 disables the cube-
    // sample IBL path and the helper falls back to the hand-tuned
    // hemispheric ambient. Mirrors `ViewUniforms.prefilter_mip_count`
    // from the Main pass: same semantics, same gate.
    pub(in crate::metal) prefilter_mip_count: f32,
}

// Per-volume uniforms uploaded at buffer(1). Layout matches
// `SdfVolumeUniforms` in `shaders/raymarch_helpers.metal`. 176 bytes
// (two packed_float3 + pad = 32, four scalars = 16, 32 float params = 128).
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::metal) struct RaymarchVolumeUniforms {
    // World-space centre (`packed_float3` + pad).
    centre: [f32; 3],
    _pad0: f32,
    // XYZ half-widths of the bounding box (`packed_float3` + pad).
    extent: [f32; 3],
    _pad1: f32,
    // `1 / max_gradient`; the cone-step scale factor in `coneRaymarch`.
    cone_ratio: f32,
    // Per-volume march far-clip in metres.
    max_distance: f32,
    // Per-volume step cap (clamped 8..256 at asset load).
    max_steps: i32,
    // Currently unused; reserved in the layout so user shaders that
    // probe it find a stable slot.
    receive_shadows: i32,
    // Generic parameter block; the user shader casts it to whatever
    // struct it interprets.
    params: [f32; SDF_PARAMS_LEN],
}

// Cascade selector pushed at buffer(4) for the shadow-caster pipeline. Picks
// `shadow.light_vps[cascade_idx]` in both stages. Matches
// `RaymarchShadowCascade` in `shaders/raymarch_shadow.metal`. 16 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
struct RaymarchShadowCascade {
    cascade_idx: u32,
    _pad: [u32; 3],
}

// `RaymarchLights` mirror of `crate::gfx::render_types::LightUniforms`.
// The Rust struct already has the right layout; we just hand the
// buffer over to the shader at buffer(2). Kept as a type alias so the
// raymarch encoder can reference it without re-defining the layout.
type RaymarchLightsGpu = LightUniforms;

// Per-`SdfVolume` GPU state: the compiled render pipeline (one PSO per
// volume) plus the static per-volume uniforms.
pub(in crate::metal) struct RaymarchVolumeRecord {
    // The volume's draw pipeline. Compiled as the opaque surface variant
    // (cone-marched SDF, depth write) for a normal volume, or the
    // alpha-blended volumetric variant (Beer-Lambert march, no depth write)
    // when the asset's `volumetric` flag is set. A volume is one or the
    // other, never both: a volumetric shader provides `sampleVolume`
    // instead of `map`/`shade`, so the surface template would not link
    // against it. Mirrors DirectX's single per-volume `pso`.
    pub(in crate::metal) pipeline: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // Depth-only shadow-caster pipeline. `Some` exactly when the asset's
    // `cast_shadows` is set; the shadow pass draws this volume into each CSM
    // cascade when it is `Some` AND `visible` AND `cast_shadows`.
    pub(in crate::metal) shadow_pipeline:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(in crate::metal) uniforms: RaymarchVolumeUniforms,
    pub(in crate::metal) visible: bool,
    // Whether this volume is volumetric (participating medium). Mirrors the
    // asset flag; the draw loop reads it to bind the read-only depth state
    // (no write) instead of the write-on state. The PSO variant is already
    // baked into `pipeline` at build time.
    pub(in crate::metal) volumetric: bool,
    // Whether this volume casts SDF shadows into the CSM cascades. Mirrors the
    // asset flag; paired with `shadow_pipeline` so the shadow encoder can skip
    // non-casters without inspecting the pipeline option.
    pub(in crate::metal) cast_shadows: bool,
    // Asset-side AABB centre / half-widths. `encode_raymarch` derives the
    // world-space AABB from these to frustum-cull the volume each frame.
    pub(in crate::metal) world_centre: [f32; 3],
    pub(in crate::metal) world_extent: [f32; 3],
}

// True when a volume at `centre` with half-widths `extent` is not entirely
// outside the camera frustum. Factored out of the draw loop so the cull
// predicate can be unit-tested without a GPU-backed `RaymarchVolumeRecord`.
pub(in crate::metal) fn volume_in_frustum(
    centre: [f32; 3],
    extent: [f32; 3],
    frustum: &crate::gfx::frustum::Frustum,
) -> bool {
    let min = [
        centre[0] - extent[0],
        centre[1] - extent[1],
        centre[2] - extent[2],
    ];
    let max = [
        centre[0] + extent[0],
        centre[1] + extent[1],
        centre[2] + extent[2],
    ];
    frustum.intersects_aabb(min, max)
}

// Compile + link a per-volume raymarch pipeline. Wraps the user
// fragment source bytes between the engine-shipped helpers and the
// engine-shipped fragment_main template, then compiles with
// `newLibraryWithSource_options_error` (same path the water / fog /
// decal / particle passes use for their built-in MSL).
//
// `asset_label` is included in error messages so a malformed user
// shader points at the right SdfVolume in the world.jsonl.
pub(in crate::metal) fn build_raymarch_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    user_source: &str,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let wrapped = format!(
        "{}\n// === user SdfVolume::fragment_shader: {} ===\n{}\n// === engine raymarch template ===\n{}\n",
        RAYMARCH_HELPERS_MSL, asset_label, user_source, RAYMARCH_TEMPLATE_MSL
    );

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&wrapped), Some(&options))
        .map_err(|e| {
            format!(
                "raymarch shader compile error for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })?;

    let vert_fn = library
        .newFunctionWithName(&ns_str("raymarch_vertex"))
        .ok_or_else(|| {
            format!(
                "raymarch_vertex entry not found in compiled library for SdfVolume '{}'",
                asset_label
            )
        })?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("raymarch_fragment"))
        .ok_or_else(|| {
            format!(
                "raymarch_fragment entry not found in compiled library for SdfVolume '{}'",
                asset_label
            )
        })?;

    // Standard 5-attribute mesh layout at buffer(2). The proxy cube
    // ships only positions, but the vertex shader declares the full
    // `Vertex` struct so the pipeline matches the engine's standard
    // vertex descriptor: keeps the pass compatible with the same
    // mesh format the rest of the engine uses.
    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        let attr0 = vert_desc.attributes().objectAtIndexedSubscript(0);
        attr0.setFormat(MTLVertexFormat::Float3);
        attr0.setOffset(0);
        attr0.setBufferIndex(2);

        let attr1 = vert_desc.attributes().objectAtIndexedSubscript(1);
        attr1.setFormat(MTLVertexFormat::Float3);
        attr1.setOffset(12);
        attr1.setBufferIndex(2);

        let attr2 = vert_desc.attributes().objectAtIndexedSubscript(2);
        attr2.setFormat(MTLVertexFormat::Float3);
        attr2.setOffset(24);
        attr2.setBufferIndex(2);

        let attr3 = vert_desc.attributes().objectAtIndexedSubscript(3);
        attr3.setFormat(MTLVertexFormat::Float3);
        attr3.setOffset(36);
        attr3.setBufferIndex(2);

        let attr4 = vert_desc.attributes().objectAtIndexedSubscript(4);
        attr4.setFormat(MTLVertexFormat::Float2);
        attr4.setOffset(48);
        attr4.setBufferIndex(2);

        let layout = vert_desc.layouts().objectAtIndexedSubscript(2);
        layout.setStride(std::mem::size_of::<Vertex>());
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
        // This variant writes opaque colour (the surface is opaque);
        // blending off keeps the per-pixel cost low. Volumetric
        // raymarching is the case that needs blend.
        ca.setBlendingEnabled(false);
    }
    // The fragment writes `[[depth(less)]]` into the bound
    // `depth_resolve` attachment so downstream passes that sample it
    // see raymarched-surface depth. Pipeline must declare the same
    // depth format the attachment uses.
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            format!(
                "failed to create raymarch pipeline state for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })
}

// Compile a per-volume depth-only shadow-caster pipeline. Wraps the user
// source between the helpers and the shadow template (the main template is
// *not* included: this library defines only `raymarch_shadow_vertex` /
// `raymarch_shadow_fragment`), then builds a render pipeline with no colour
// attachment and a `Depth32Float` depth attachment matching the CSM
// `shadow_map`. Mirrors `directx/raymarch.rs::compile_volume_shadow_pso`.
pub(in crate::metal) fn build_raymarch_shadow_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    user_source: &str,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let wrapped = format!(
        "{}\n// === user SdfVolume::fragment_shader (shadow): {} ===\n{}\n// === engine raymarch shadow template ===\n{}\n",
        RAYMARCH_HELPERS_MSL, asset_label, user_source, RAYMARCH_SHADOW_MSL
    );

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&wrapped), Some(&options))
        .map_err(|e| {
            format!(
                "raymarch shadow shader compile error for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })?;

    let vert_fn = library
        .newFunctionWithName(&ns_str("raymarch_shadow_vertex"))
        .ok_or_else(|| {
            format!(
                "raymarch_shadow_vertex entry not found in compiled library for SdfVolume '{}'",
                asset_label
            )
        })?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("raymarch_shadow_fragment"))
        .ok_or_else(|| {
            format!(
                "raymarch_shadow_fragment entry not found in compiled library for SdfVolume '{}'",
                asset_label
            )
        })?;

    // Same proxy-cube vertex layout as the main pass (Vertex at buffer(2)).
    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        let attr0 = vert_desc.attributes().objectAtIndexedSubscript(0);
        attr0.setFormat(MTLVertexFormat::Float3);
        attr0.setOffset(0);
        attr0.setBufferIndex(2);

        let attr1 = vert_desc.attributes().objectAtIndexedSubscript(1);
        attr1.setFormat(MTLVertexFormat::Float3);
        attr1.setOffset(12);
        attr1.setBufferIndex(2);

        let attr2 = vert_desc.attributes().objectAtIndexedSubscript(2);
        attr2.setFormat(MTLVertexFormat::Float3);
        attr2.setOffset(24);
        attr2.setBufferIndex(2);

        let attr3 = vert_desc.attributes().objectAtIndexedSubscript(3);
        attr3.setFormat(MTLVertexFormat::Float3);
        attr3.setOffset(36);
        attr3.setBufferIndex(2);

        let attr4 = vert_desc.attributes().objectAtIndexedSubscript(4);
        attr4.setFormat(MTLVertexFormat::Float2);
        attr4.setOffset(48);
        attr4.setBufferIndex(2);

        let layout = vert_desc.layouts().objectAtIndexedSubscript(2);
        layout.setStride(std::mem::size_of::<Vertex>());
        layout.setStepFunction(MTLVertexStepFunction::PerVertex);
    }

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(&vert_desc));
    desc.setVertexFunction(Some(&vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    // Shadow map is single-sample; no colour attachment is bound in the
    // shadow pass (depth-only). Only the depth format is declared.
    desc.setRasterSampleCount(1);
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            format!(
                "failed to create raymarch shadow pipeline state for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })
}

// Compile a per-volume volumetric raymarching pipeline. Wraps the user
// source between the helpers and the volumetric template (the main template is
// *not* included: this library defines only `raymarch_volumetric_vertex` /
// `raymarch_volumetric_fragment`), then builds a render pipeline with alpha
// blending and no depth write. The user shader provides `sampleVolume(p, params, time)`
// returning density + scattering + emission instead of `map`/`shade`.
pub(in crate::metal) fn build_raymarch_volumetric_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    user_source: &str,
    asset_label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let wrapped = format!(
        "{}\n// === user SdfVolume::fragment_shader (volumetric): {} ===\n{}\n// === engine raymarch volumetric template ===\n{}\n",
        RAYMARCH_HELPERS_MSL, asset_label, user_source, RAYMARCH_VOLUMETRIC_MSL
    );

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(&wrapped), Some(&options))
        .map_err(|e| {
            format!(
                "raymarch volumetric shader compile error for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })?;

    let vert_fn = library
        .newFunctionWithName(&ns_str("raymarch_volumetric_vertex"))
        .ok_or_else(|| {
            format!(
                "raymarch_volumetric_vertex entry not found in compiled library for SdfVolume '{}'",
                asset_label
            )
        })?;
    let frag_fn = library
        .newFunctionWithName(&ns_str("raymarch_volumetric_fragment"))
        .ok_or_else(|| {
            format!(
                "raymarch_volumetric_fragment entry not found in compiled library for SdfVolume '{}'",
                asset_label
            )
        })?;

    // Same proxy-cube vertex layout as the main pass (Vertex at buffer(2)).
    let vert_desc = MTLVertexDescriptor::new();
    unsafe {
        let attr0 = vert_desc.attributes().objectAtIndexedSubscript(0);
        attr0.setFormat(MTLVertexFormat::Float3);
        attr0.setOffset(0);
        attr0.setBufferIndex(2);

        let attr1 = vert_desc.attributes().objectAtIndexedSubscript(1);
        attr1.setFormat(MTLVertexFormat::Float3);
        attr1.setOffset(12);
        attr1.setBufferIndex(2);

        let attr2 = vert_desc.attributes().objectAtIndexedSubscript(2);
        attr2.setFormat(MTLVertexFormat::Float3);
        attr2.setOffset(24);
        attr2.setBufferIndex(2);

        let attr3 = vert_desc.attributes().objectAtIndexedSubscript(3);
        attr3.setFormat(MTLVertexFormat::Float3);
        attr3.setOffset(36);
        attr3.setBufferIndex(2);

        let attr4 = vert_desc.attributes().objectAtIndexedSubscript(4);
        attr4.setFormat(MTLVertexFormat::Float2);
        attr4.setOffset(48);
        attr4.setBufferIndex(2);

        let layout = vert_desc.layouts().objectAtIndexedSubscript(2);
        layout.setStride(std::mem::size_of::<Vertex>());
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
        // Volumetrics alpha-blend over the scene: output is translucent
        // so we use SRC_ALPHA / ONE_MINUS_SRC_ALPHA at the blend level.
        ca.setBlendingEnabled(true);
        ca.setSourceRGBBlendFactor(MTLBlendFactor::SourceAlpha);
        ca.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setRgbBlendOperation(MTLBlendOperation::Add);
        ca.setSourceAlphaBlendFactor(MTLBlendFactor::One);
        ca.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        ca.setAlphaBlendOperation(MTLBlendOperation::Add);
    }
    // No depth write for volumetrics: they're translucent and don't update depth.
    // Depth test is still enabled to early-out against rasterised geometry.
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);

    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| {
            format!(
                "failed to create raymarch volumetric pipeline state for SdfVolume '{}': {:?}",
                asset_label, e
            )
        })
}

// Build the per-volume record (PSO + per-volume uniforms) from one
// declared `SdfVolume` + its compiled-payload source bytes (which the
// build pipeline packed via `SdfVolume::compile_payload`).
pub(in crate::metal) fn build_raymarch_volume_record(
    device: &ProtocolObject<dyn MTLDevice>,
    volume: &SdfVolume,
    user_source_bytes: &[u8],
    asset_label: &str,
) -> Result<RaymarchVolumeRecord, String> {
    let user_source = std::str::from_utf8(user_source_bytes).map_err(|e| {
        format!(
            "SdfVolume '{}': fragment shader payload is not valid UTF-8: {}",
            asset_label, e
        )
    })?;
    // A volumetric volume builds the alpha-blended Beer-Lambert pipeline;
    // every other volume builds the opaque cone-marched surface pipeline.
    // Only one is built: a volumetric user shader provides `sampleVolume`
    // and omits `map`/`shade`, so wrapping it with the surface template
    // would fail to link (`Undefined symbol(s) ... map / shade`).
    let pipeline = if volume.volumetric {
        build_raymarch_volumetric_pipeline(device, user_source, asset_label)?
    } else {
        build_raymarch_pipeline(device, user_source, asset_label)?
    };
    // Build the depth-only shadow-caster pipeline only when the asset opts in.
    // Reuses the same user source: the wrap just swaps the main template for
    // the shadow template. Volumetrics never cast shadows (cast_shadows is
    // force-disabled in SdfVolume::from_args when volumetric is true).
    let shadow_pipeline = if volume.cast_shadows {
        Some(build_raymarch_shadow_pipeline(
            device,
            user_source,
            asset_label,
        )?)
    } else {
        None
    };
    Ok(RaymarchVolumeRecord {
        pipeline,
        shadow_pipeline,
        uniforms: volume_uniforms_from(volume),
        visible: volume.visible,
        volumetric: volume.volumetric,
        cast_shadows: volume.cast_shadows,
        world_centre: volume.centre,
        world_extent: volume.extent,
    })
}

fn volume_uniforms_from(volume: &SdfVolume) -> RaymarchVolumeUniforms {
    RaymarchVolumeUniforms {
        centre: volume.centre,
        _pad0: 0.0,
        extent: volume.extent,
        _pad1: 0.0,
        cone_ratio: volume.cone_ratio(),
        max_distance: volume.max_distance,
        max_steps: volume.max_steps as i32,
        receive_shadows: if volume.receive_shadows { 1 } else { 0 },
        params: volume.params,
    }
}

// Build the shared unit-cube proxy geometry buffers. The vertices ride
// in `Vertex` shape so the proxy works through the engine's standard
// vertex descriptor (the same five-attribute layout the main pass and
// every custom mesh shader expect). 8 corners in `[-0.5, 0.5]^3`; the
// vertex shader scales by `vol.extent` and translates by `vol.centre`.
// Indices wind 36 CCW triangles (the encoder culls front faces so the
// rasteriser only fires for back faces).
#[allow(clippy::type_complexity)]
pub(in crate::metal) fn build_raymarch_cube_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<
    (
        Retained<ProtocolObject<dyn MTLBuffer>>,
        Retained<ProtocolObject<dyn MTLBuffer>>,
    ),
    String,
> {
    // `extent` in SdfVolume is the AABB half-widths: the box spans
    // `centre ± extent`. The vertex shader computes `pos * extent +
    // centre`, so the proxy corners must be at `±1.0` for the scaled
    // corners to land at `centre ± extent`.
    #[rustfmt::skip]
    let corners: [Vertex; 8] = [
        v([-1.0, -1.0, -1.0]),
        v([ 1.0, -1.0, -1.0]),
        v([ 1.0,  1.0, -1.0]),
        v([-1.0,  1.0, -1.0]),
        v([-1.0, -1.0,  1.0]),
        v([ 1.0, -1.0,  1.0]),
        v([ 1.0,  1.0,  1.0]),
        v([-1.0,  1.0,  1.0]),
    ];
    // 36 CCW indices (outward winding when viewed from +x / +y / +z
    // halfspaces). Front-face cull will render back faces only at
    // encode time.
    #[rustfmt::skip]
    let indices: [u16; 36] = [
        // -Z
        0, 2, 1,  0, 3, 2,
        // +Z
        4, 5, 6,  4, 6, 7,
        // -X
        0, 4, 7,  0, 7, 3,
        // +X
        1, 2, 6,  1, 6, 5,
        // -Y
        0, 1, 5,  0, 5, 4,
        // +Y
        3, 7, 6,  3, 6, 2,
    ];

    let vb_bytes = std::mem::size_of_val(&corners);
    let ib_bytes = std::mem::size_of_val(&indices);

    let vb = unsafe {
        let ptr = std::ptr::NonNull::new(corners.as_ptr() as *mut _)
            .ok_or("raymarch cube vertex pointer null")?;
        device
            .newBufferWithBytes_length_options(ptr, vb_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate raymarch cube vertex buffer")?
    };
    let ib = unsafe {
        let ptr = std::ptr::NonNull::new(indices.as_ptr() as *mut _)
            .ok_or("raymarch cube index pointer null")?;
        device
            .newBufferWithBytes_length_options(ptr, ib_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate raymarch cube index buffer")?
    };
    Ok((vb, ib))
}

fn v(pos: [f32; 3]) -> Vertex {
    Vertex {
        pos,
        normal: [0.0, 0.0, 0.0],
        tangent: [0.0, 0.0, 0.0],
        color: [0.0, 0.0, 0.0],
        uv: [0.0, 0.0],
    }
}

impl MtlContext {
    // Encode the raymarched SDF volume pass. Caller has ended the main
    // pass (so `hdr_targets.depth` carries scene depth) and the
    // post-Main hdr_resolve writes from Decals / Fog / ParticlesDraw
    // have not yet fired. Each visible `SdfVolume` issues one indexed
    // draw of the proxy cube; the user's `map` + `shade` run per
    // fragment.
    pub(in crate::metal) fn encode_raymarch(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &RaymarchView,
        frustum: &crate::gfx::frustum::Frustum,
    ) -> Result<u32, String> {
        if self.raymarch_volumes.is_empty() {
            return Ok(0);
        }
        // Frustum-cull each volume's world-space AABB up front. A volume that
        // is hidden or entirely off-screen costs nothing this frame -- not even
        // the scene-copy blit + render encoder below. Mirrors the decal /
        // particle passes' pre-pass visibility mask.
        let visible: Vec<bool> = self
            .raymarch_volumes
            .iter()
            .map(|v| v.visible && volume_in_frustum(v.world_centre, v.world_extent, frustum))
            .collect();
        if !visible.iter().any(|&v| v) {
            return Ok(0);
        }
        let vbuf = self
            .raymarch_cube_vertex_buffer
            .as_ref()
            .ok_or("raymarch cube vertex buffer missing")?;
        let ibuf = self
            .raymarch_cube_index_buffer
            .as_ref()
            .ok_or("raymarch cube index buffer missing")?;
        let depth_sampler = self.post_sampler.as_ref();

        let lights_gpu: RaymarchLightsGpu = self.light_uniforms;
        let shadow_uniforms = self.shadow_uniforms;

        // Refraction support: snapshot the pre-raymarch
        // `hdr_resolve` into `hdr_resolve_copy` so user SDF shaders can
        // sample the scene below the surface without violating Metal's
        // attachment-aliasing rule (the same `hdr_resolve` we'd want to
        // read is also the colour attachment we're about to write).
        // Single full-screen blit per frame; AutoExposure has already
        // sampled `hdr_resolve_v1`, so this captures the same un-
        // decorated scene the next post-Main pass starts with.
        let blit = cmd_buf
            .blitCommandEncoder()
            .ok_or("failed to get raymarch scene-copy blit encoder")?;
        blit.pushDebugGroup(&NSString::from_str("raymarch_scene_copy"));
        unsafe {
            blit.copyFromTexture_toTexture(
                self.hdr_targets.hdr_resolve.as_ref(),
                self.hdr_targets.hdr_resolve_copy.as_ref(),
            );
        }
        blit.popDebugGroup();
        blit.endEncoding();

        let pass_desc = MTLRenderPassDescriptor::new();
        unsafe {
            let ca = pass_desc.colorAttachments().objectAtIndexedSubscript(0);
            ca.setTexture(Some(self.hdr_targets.hdr_resolve.as_ref()));
            ca.setLoadAction(MTLLoadAction::Load);
            ca.setStoreAction(MTLStoreAction::Store);
            // Bind the single-sample depth resolve as the
            // writable depth attachment. `Load` keeps the rasterised
            // depth that the Main pass resolved into it, so the
            // hardware depth test rejects raymarched fragments behind
            // existing geometry (and behind earlier raymarch volumes
            // in this pass). `Store` keeps the new depth (the min of
            // rasterised and raymarched per pixel) alive for
            // water / decal / fog to consume.
            let da = pass_desc.depthAttachment();
            da.setTexture(Some(self.hdr_targets.depth_resolve.as_ref()));
            da.setLoadAction(MTLLoadAction::Load);
            da.setStoreAction(MTLStoreAction::Store);
        }
        if let Some(t) = &self.pass_timing {
            t.attach_render(&pass_desc, super::pass_timing::PassId::Raymarch);
        }
        // Blit above is ended explicitly; this render encoder spans to the end
        // of the function, so the guard ends it on drop.
        let enc = ScopedEncoder::new(
            cmd_buf
                .renderCommandEncoderWithDescriptor(&pass_desc)
                .ok_or("failed to get raymarch render encoder")?,
            "raymarch",
        );
        // Front-face cull so each pixel inside the box receives exactly
        // one fragment shader invocation regardless of whether the
        // camera is outside or inside the bounding box. (Outside →
        // back faces visible; inside → front faces behind camera, only
        // back faces in view.)
        enc.setCullMode(MTLCullMode::Front);
        // Standard forward-render depth state: compare = less, write
        // = on. The fragment shader's `[[depth(less)]]` output further
        // gates: even if the rasterised proxy fragment passes the
        // depth test, the actual raymarch hit depth has to be < the
        // existing value to commit.
        enc.setDepthStencilState(Some(self.depth_state.as_ref()));

        unsafe {
            // Per-frame view at buffer(0); same value for vertex + fragment.
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(view).cast(),
                std::mem::size_of::<RaymarchView>(),
                0,
            );
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(view).cast(),
                std::mem::size_of::<RaymarchView>(),
                0,
            );
            // Lights at buffer(2); rebound once.
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&lights_gpu).cast(),
                std::mem::size_of::<RaymarchLightsGpu>(),
                2,
            );
            // Cascade-shadow uniforms at buffer(3).
            // Always bound; the helper falls back to `shadow = 1.0`
            // when `vol.receive_shadows == 0` or when the world has no
            // shadow stage (in which case `shadow_map` is the 1×1
            // fallback texture and the cascade compare returns full
            // light).
            enc.setFragmentBytes_length_atIndex(
                std::ptr::NonNull::from(&shadow_uniforms).cast(),
                std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
                3,
            );
            // Proxy-cube vertices at vertex buffer(2); index buffer
            // bound per-draw. The vertex descriptor declares the full
            // 56-byte Vertex layout at this binding.
            enc.setVertexBuffer_offset_atIndex(Some(vbuf), 0, 2);
            // Main pass MSAA depth at fragment texture(0); sampled by
            // `main_depth.read(px, 0)` in the template fragment for
            // the shader-side cone-march early-out (separate texture
            // from the writable `depth_resolve` attachment so no
            // aliasing).
            enc.setFragmentTexture_atIndex(Some(self.hdr_targets.depth.as_ref()), 0);
            // CSM shadow map array + IBL cubes.
            // Always bound (1×1 fallback when the world has no shadow
            // stage / no EnvironmentMap), matching the Main pass.
            enc.setFragmentTexture_atIndex(Some(self.shadow_map.as_ref()), 1);
            enc.setFragmentTexture_atIndex(Some(self.env_map.irradiance.as_ref()), 2);
            enc.setFragmentTexture_atIndex(Some(self.env_map.prefilter.as_ref()), 3);
            // Pre-raymarch scene snapshot for refraction
            // sampling. The blit at the top of this function populated
            // `hdr_resolve_copy` from `hdr_resolve`; user shaders that
            // care call `sampleSceneRefracted` against this binding.
            // Always bound (even when no shader uses it) so the per-
            // volume PSO doesn't need a "refraction enabled" variant.
            enc.setFragmentTexture_atIndex(Some(self.hdr_targets.hdr_resolve_copy.as_ref()), 4);
            enc.setFragmentSamplerState_atIndex(Some(depth_sampler), 0);
            enc.setFragmentSamplerState_atIndex(Some(self.shadow_sampler.as_ref()), 1);
            enc.setFragmentSamplerState_atIndex(Some(self.cube_sampler.as_ref()), 2);
            // Reuse the linear-clamp post sampler for the scene-copy
            // tap: same filter the existing water + bloom passes use.
            enc.setFragmentSamplerState_atIndex(Some(depth_sampler), 3);
        }

        let mut draws: u32 = 0;
        for (i, vol) in self.raymarch_volumes.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            // `pipeline` is already the right variant for this volume: the
            // volumetric (alpha-blended) PSO when `volumetric`, the opaque
            // surface PSO otherwise (selected at build time).
            enc.setRenderPipelineState(&vol.pipeline);
            // Volumetric media are translucent and must not write depth, but
            // they should still be occluded by nearer opaque geometry. Bind the
            // read-only `LessEqual` state (no write): matching the DirectX
            // volumetric PSO's `DepthFunc=LESS_EQUAL, WriteMask=ZERO`. Passing
            // `None` here would trip Metal's validation layer
            // (`setDepthStencilState(nil)` is illegal). Opaque SDF surfaces keep
            // the write-on state so they composite into the depth buffer
            // downstream passes sample.
            if vol.volumetric {
                enc.setDepthStencilState(Some(self.depth_state_read_only.as_ref()));
            } else {
                enc.setDepthStencilState(Some(self.depth_state.as_ref()));
            }
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&vol.uniforms).cast(),
                    std::mem::size_of::<RaymarchVolumeUniforms>(),
                    1,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&vol.uniforms).cast(),
                    std::mem::size_of::<RaymarchVolumeUniforms>(),
                    1,
                );
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    36,
                    MTLIndexType::UInt16,
                    ibuf,
                    0,
                );
            }
            draws += 1;
        }

        Ok(draws)
    }

    // `true` when at least one visible `SdfVolume` opted into `cast_shadows`
    // and built a shadow pipeline. The shadow pass builds the per-frame
    // `RaymarchView` and dispatches `encode_sdf_shadow_casters` only when this
    // holds. Mirrors `directx::raymarch`'s `any_shadow_casters`.
    pub(in crate::metal) fn any_raymarch_shadow_casters(&self) -> bool {
        self.raymarch_volumes
            .iter()
            .any(|v| v.visible && v.cast_shadows && v.shadow_pipeline.is_some())
    }

    // Encode raymarched SDF shadow casters into the CSM cascades. Called from
    // `encode_shadow_pass` after the rasterised + skinned casters, on the same
    // command buffer so the writes land before the Main pass samples the
    // shadow map. For each cascade this opens a depth-only render pass on that
    // `shadow_map` slice with `Load` / `Store` (keeping the rasterised depth
    // already written into the slice), then draws each caster's proxy cube
    // with front faces culled. The depth-only fragment cone-marches the SDF
    // from the light side and writes the hit's NDC.z via `[[depth(less)]]`;
    // the slice's LESS depth test keeps the nearest caster (rasterised or
    // raymarched) per texel. A no-op (returns 0) when no volume casts.
    pub(in crate::metal) fn encode_sdf_shadow_casters(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        view: &RaymarchView,
    ) -> Result<u32, String> {
        use crate::gfx::render_types::NUM_SHADOW_CASCADES;
        if !self.any_raymarch_shadow_casters() {
            return Ok(0);
        }
        let vbuf = self
            .raymarch_cube_vertex_buffer
            .as_ref()
            .ok_or("raymarch shadow: cube vertex buffer missing")?;
        let ibuf = self
            .raymarch_cube_index_buffer
            .as_ref()
            .ok_or("raymarch shadow: cube index buffer missing")?;
        let lights_gpu: RaymarchLightsGpu = self.light_uniforms;
        let shadow_uniforms = self.shadow_uniforms;

        let mut draws: u32 = 0;
        // Only cast into cascades the rasterised shadow pass re-rendered this
        // frame: a skipped cascade's slice must stay exactly as it was last
        // fully rendered (raster + SDF), so we neither clear nor add to it.
        let render_mask = if self.shadow_render_mask == 0 {
            (1u32 << NUM_SHADOW_CASCADES) - 1
        } else {
            self.shadow_render_mask
        };
        for cascade_idx in 0..NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let pass_desc = MTLRenderPassDescriptor::new();
            let da = pass_desc.depthAttachment();
            da.setTexture(Some(self.shadow_map.as_ref()));
            da.setSlice(cascade_idx);
            // Load the rasterised depth this cascade already holds, draw the
            // SDF casters on top, and keep the merged depth for the Main pass.
            da.setLoadAction(MTLLoadAction::Load);
            da.setStoreAction(MTLStoreAction::Store);

            // Loop-local guard: each cascade's encoder ends on drop at the end
            // of this iteration, before the next cascade opens one.
            let enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&pass_desc)
                    .ok_or("failed to get raymarch shadow render encoder")?,
                "raymarch shadow",
            );
            // Front-face cull → exactly one fragment per texel inside the box's
            // light-space projection. Same depth state (compare = less, write
            // on) as the rasterised casters so the two layers composite.
            enc.setCullMode(MTLCullMode::Front);
            enc.setDepthStencilState(Some(self.depth_state.as_ref()));

            let cascade = RaymarchShadowCascade {
                cascade_idx: cascade_idx as u32,
                _pad: [0; 3],
            };
            unsafe {
                // Per-cascade shared bindings: view@0 (fragment reads
                // view.time), lights@2 (fragment), shadow uniforms@3 (vertex
                // projection + fragment reprojection), cascade selector@4
                // (both stages), proxy-cube vertices@2 (vertex).
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(view).cast(),
                    std::mem::size_of::<RaymarchView>(),
                    0,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&lights_gpu).cast(),
                    std::mem::size_of::<RaymarchLightsGpu>(),
                    2,
                );
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&shadow_uniforms).cast(),
                    std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
                    3,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&shadow_uniforms).cast(),
                    std::mem::size_of::<crate::gfx::render_types::ShadowUniforms>(),
                    3,
                );
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&cascade).cast(),
                    std::mem::size_of::<RaymarchShadowCascade>(),
                    4,
                );
                enc.setFragmentBytes_length_atIndex(
                    std::ptr::NonNull::from(&cascade).cast(),
                    std::mem::size_of::<RaymarchShadowCascade>(),
                    4,
                );
                enc.setVertexBuffer_offset_atIndex(Some(vbuf), 0, 2);
            }

            for vol in &self.raymarch_volumes {
                if !vol.visible || !vol.cast_shadows {
                    continue;
                }
                let Some(pso) = vol.shadow_pipeline.as_ref() else {
                    continue;
                };
                enc.setRenderPipelineState(pso);
                unsafe {
                    enc.setVertexBytes_length_atIndex(
                        std::ptr::NonNull::from(&vol.uniforms).cast(),
                        std::mem::size_of::<RaymarchVolumeUniforms>(),
                        1,
                    );
                    enc.setFragmentBytes_length_atIndex(
                        std::ptr::NonNull::from(&vol.uniforms).cast(),
                        std::mem::size_of::<RaymarchVolumeUniforms>(),
                        1,
                    );
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                        MTLPrimitiveType::Triangle,
                        36,
                        MTLIndexType::UInt16,
                        ibuf,
                        0,
                    );
                }
                draws += 1;
            }
        }
        Ok(draws)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn raymarch_view_layout_matches_msl() {
        // MSL `RaymarchView` in raymarch_helpers.metal: two float4x4, a
        // packed_float3 cam_pos (+ pad), float2 viewport, then two scalars.
        // The Rust `cam_pos: [f32; 4]` covers the same 16 bytes (xyz + pad)
        // as the MSL `packed_float3 cam_pos; float _pad0;`.
        assert_eq!(size_of::<RaymarchView>(), 160);
        assert_eq!(offset_of!(RaymarchView, vp), 0);
        assert_eq!(offset_of!(RaymarchView, inv_vp), 64);
        assert_eq!(offset_of!(RaymarchView, cam_pos), 128);
        assert_eq!(offset_of!(RaymarchView, viewport), 144);
        assert_eq!(offset_of!(RaymarchView, time), 152);
        assert_eq!(offset_of!(RaymarchView, prefilter_mip_count), 156);
        assert_eq!(size_of::<RaymarchView>() % 16, 0);
    }

    #[test]
    fn raymarch_volume_uniforms_layout_matches_msl() {
        // MSL `SdfVolumeUniforms` in raymarch_helpers.metal: two packed_float3
        // (+ pad), four scalars, then `SdfParams { float vals[32]; }` at offset
        // 48. The 176-byte size pins SDF_PARAMS_LEN == 32 (48 + 32*4).
        assert_eq!(size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, centre), 0);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, _pad0), 12);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, extent), 16);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, _pad1), 28);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, cone_ratio), 32);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_distance), 36);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, max_steps), 40);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, receive_shadows), 44);
        assert_eq!(offset_of!(RaymarchVolumeUniforms, params), 48);
    }

    #[test]
    fn volume_in_frustum_culls_offscreen_boxes() {
        use crate::gfx::frustum::Frustum;
        // Identity view-projection -> the visible region is the [-1, 1]^3 clip
        // cube. A unit box at the origin overlaps it; a box far to the right is
        // entirely past the right clip plane and is culled.
        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let f = Frustum::from_view_projection(identity);
        assert!(volume_in_frustum([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], &f));
        assert!(!volume_in_frustum([10.0, 0.0, 0.0], [0.5, 0.5, 0.5], &f));
        // A box the camera sits inside (origin within its extent) still
        // overlaps the frustum.
        assert!(volume_in_frustum(
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
            &f
        ));
    }

    #[test]
    fn raymarch_shadow_cascade_layout_matches_msl() {
        // MSL `RaymarchShadowCascade` in raymarch_shadow.metal: a uint + pad.
        assert_eq!(size_of::<RaymarchShadowCascade>(), 16);
        assert_eq!(offset_of!(RaymarchShadowCascade, cascade_idx), 0);
        assert_eq!(offset_of!(RaymarchShadowCascade, _pad), 4);
    }
}
