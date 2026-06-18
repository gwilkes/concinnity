// src/directx/raymarch.rs
//
// Per-frame encoder for the raymarched SDF volume pass on D3D12. Runs at
// `PassId::Raymarch`, between `AutoExposure` and `Decals` on the
// hdr_resolve RMW chain. Each `SdfVolume` rasterises the back faces of
// its world-space bounding box and runs a user-authored HLSL fragment
// shader that sphere-traces the SDF inside the box. HLSL port of
// `src/metal/raymarch.rs`: same shader interface, same
// depth-compositing rules.
//
// DX architecture:
//   * One `ID3D12PipelineState` per `SdfVolume` (built at init from the
//     engine-shipped helpers + the user's HLSL bytes + the engine-shipped
//     template). The wrap order is helpers → user → template so the
//     template's `raymarch_fragment` can call the user's `map` / `shade`
//     through the forward declarations in the helpers.
//   * One shared unit-cube VB + IB for the proxy geometry; 8 corners /
//     36 indices, allocated once at init. The encoder draws back faces
//     only (cull mode = Front) so we get exactly one fragment per pixel
//     inside the box regardless of camera position.
//   * Per-volume `SdfVolumeUniforms` cbuffer (static: `centre`, `extent`,
//     `params`, ... don't change frame-to-frame) allocated once at init.
//   * Per-frame `RaymarchView` cbuffer ring (triple-buffered).
//   * Colour attachment = `hdr_resolve` (LOAD, opaque write). Depth
//     attachment = the main depth buffer in `DEPTH_WRITE`: the fragment
//     writes hit depth via `SV_DepthLessEqual` so downstream passes
//     (decals, fog, SSR, TAA, ...) see the raymarched surface.
//
// Backend filter. The asset's `fragment_shader` field holds a path
// to the user shader; the build pipeline packs the file bytes verbatim
// into the payload. On D3D12 we can only consume `.hlsl` payloads:
// `.metal` SDFs (the Metal-first authoring path) are skipped at init
// with a logged warning, and the rest of the world renders unchanged.
// Authors who want cross-backend SDFs ship parallel `.metal` + `.hlsl`
// files and declare one `SdfVolume` per backend.
//
// Currently unimplemented on DirectX:
//   * No `depth_copy` snapshot, so no in-shader rasterised-depth early-
//     out (hardware depth test still composites correctly: see the
//     template's caveat). Volumes whose bbox sits fully behind
//     rasterised geometry pay the full march cost.
//   * No `hdr_resolve_copy` snapshot, so `scene_color` is a 1×1 black
//     fallback: refractive user shaders (water, glass) get zero from
//     their `sampleSceneRefracted` call.

use std::ffi::c_void;

use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::assets::sdf_volume::{SDF_PARAMS_LEN, SdfVolume};
use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{
    compile_hlsl, main_input_layout, serialize_desc_and_create, shader_source,
};
use crate::directx::texture::{
    HDR_FORMAT, create_buffer, create_fallback_white_resource, create_hdr_resolve_target,
    transition_barrier,
};
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::LightUniforms;

const RAYMARCH_HELPERS_HLSL: &str = include_str!("shaders/raymarch_helpers.hlsl");
const RAYMARCH_TEMPLATE_HLSL: &str = include_str!("shaders/raymarch_template.hlsl");
const RAYMARCH_SHADOW_HLSL: &str = include_str!("shaders/raymarch_shadow.hlsl");
const RAYMARCH_VOLUMETRIC_TEMPLATE_HLSL: &str =
    include_str!("shaders/raymarch_volumetric_template.hlsl");

// Per-frame view cbuffer the raymarch pass binds at b0. Layout matches
// `RaymarchView` in `shaders/raymarch_helpers.hlsl`. 160 bytes; aligned
// to 256 for D3D12 cbuffer requirements.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::directx) struct RaymarchView {
    pub(in crate::directx) vp: [[f32; 4]; 4],
    pub(in crate::directx) inv_vp: [[f32; 4]; 4],
    pub(in crate::directx) cam_pos: [f32; 3],
    pub(in crate::directx) _pad0: f32,
    pub(in crate::directx) viewport: [f32; 2],
    pub(in crate::directx) time: f32,
    pub(in crate::directx) prefilter_mip_count: f32,
}

// Per-volume uniforms at b1. Layout matches `SdfVolumeUniforms` in the
// HLSL helpers. 176 bytes; aligned to 256 in the cbuffer allocation.
#[derive(Copy, Clone)]
#[repr(C)]
struct RaymarchVolumeUniforms {
    centre: [f32; 3],
    _pad0: f32,
    extent: [f32; 3],
    _pad1: f32,
    cone_ratio: f32,
    max_distance: f32,
    max_steps: i32,
    receive_shadows: i32,
    params: [f32; SDF_PARAMS_LEN],
}

fn volume_uniforms_from(v: &SdfVolume) -> RaymarchVolumeUniforms {
    RaymarchVolumeUniforms {
        centre: v.centre,
        _pad0: 0.0,
        extent: v.extent,
        _pad1: 0.0,
        cone_ratio: v.cone_ratio(),
        max_distance: v.max_distance,
        max_steps: v.max_steps as i32,
        receive_shadows: if v.receive_shadows { 1 } else { 0 },
        params: v.params,
    }
}

// Per-`SdfVolume` GPU state: the compiled render pipeline, the static
// per-volume cbuffer (uploaded once at init), the optional shadow-cast
// PSO, and a couple of asset-side scalars kept around for a future
// CPU frustum cull.
pub(in crate::directx) struct RaymarchVolumeRecord {
    pub(in crate::directx) pso: ID3D12PipelineState,
    // Depth-only shadow PSO. `Some` when the asset's `cast_shadows`
    // is true at init; the shadow encoder iterates only the records
    // where this is `Some` AND `visible` AND `cast_shadows` (the
    // runtime flag, currently can't toggle, but the field is
    // preserved for future runtime mutation).
    pub(in crate::directx) shadow_pso: Option<ID3D12PipelineState>,
    // Per-volume cbuffer (CPU-visible upload heap, mapped once at
    // build time, never modified: the asset's centre / extent / params
    // are static).
    #[allow(dead_code)]
    volume_cbuffer: ID3D12Resource,
    pub(in crate::directx) volume_cbuffer_gva: u64,
    pub(in crate::directx) visible: bool,
    pub(in crate::directx) cast_shadows: bool,
    #[allow(dead_code)]
    pub(in crate::directx) world_centre: [f32; 3],
    #[allow(dead_code)]
    pub(in crate::directx) world_extent: [f32; 3],
}

// Engine-side raymarch resources: shared cube buffers, per-frame view
// cbuffer ring, scene-color fallback texture, root signature, per-
// volume records. Built only when at least one `.hlsl` `SdfVolume`
// landed at init; the encoder is a no-op otherwise.
pub(in crate::directx) struct RaymarchResources {
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    // Depth-only root signature for raymarched shadow casters. Only
    // binds the cbuffers the shadow template needs (no SRV / sampler
    // tables) plus a 1-DWORD cascade index root constant.
    pub(in crate::directx) shadow_root_sig: ID3D12RootSignature,
    // Shared unit-cube proxy geometry (vertices in ±1; the vertex
    // shader scales by `vol_extent`). Held only to keep the resources
    // resident; the encoder binds them through the `_view` siblings.
    #[allow(dead_code)]
    cube_vb: ID3D12Resource,
    #[allow(dead_code)]
    cube_ib: ID3D12Resource,
    cube_vbv: D3D12_VERTEX_BUFFER_VIEW,
    cube_ibv: D3D12_INDEX_BUFFER_VIEW,
    // Per-frame `RaymarchView` cbuffer ring. Persistently mapped; the
    // encoder memcpys this frame's view into `view_ptrs[frame_idx]`
    // before binding `view_cbuffers[frame_idx]` at b0.
    view_cbuffers: Vec<ID3D12Resource>,
    view_ptrs: Vec<*mut u8>,
    // 1×1 white fallback for the `scene_color` SRV slot, kept around
    // only to hold a resource open while init runs; the live SRV at
    // `scene_color_srv_cpu` is rewritten to point at `hdr_resolve_copy`
    // (below) before the first frame. Lives in case a future per-volume
    // "no refraction needed" opt-out wants to re-point the slot at a
    // constant tap.
    #[allow(dead_code)]
    scene_color_fallback: ID3D12Resource,
    // Pre-raymarch HDR scene snapshot. At the top of `encode_raymarch`
    // we `CopyResource` from `hdr_resolve` (or `hdr_color` when MSAA
    // is off) into this resource, so refractive user shaders that
    // sample `scene_color` through `sampleSceneRefracted` see the
    // scene as it stood after Main + AutoExposure but before raymarch
    // itself writes / decals / fog / particles. Sized to render dims;
    // recreated by `resize_to` on window resize. Mirrors the Metal
    // `hdr_resolve_copy` snapshot.
    hdr_resolve_copy: ID3D12Resource,
    // CPU descriptor handle of the `scene_color` SRV slot, captured
    // at init so `resize_to` can rewrite the descriptor in place after
    // recreating `hdr_resolve_copy` (the GPU handle stays valid since
    // the heap slot itself doesn't move).
    scene_color_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    // GPU descriptor handle for the start of the 5-SRV descriptor
    // table the pixel shader binds at t0..t4 (main_depth, shadow_map,
    // irradiance, prefilter, scene_color). Written at init by
    // `write_raymarch_srvs`; live for the renderer's lifetime.
    pub(in crate::directx) srv_table_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // GPU descriptor handle for the start of the 3-sampler descriptor
    // table at s0..s2 (shadow_samp, cube_samp, scene_samp).
    pub(in crate::directx) sampler_table_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Per-volume records. Drained from the world's `SdfVolume`s at init.
    pub(in crate::directx) volumes: Vec<RaymarchVolumeRecord>,
}

// Compile the per-volume HLSL source by wrapping the user's bytes
// between the engine-shipped helpers and the template. The wrap order
// is helpers → user → template so the template's `raymarch_fragment`
// can call the user's `map` / `shade` through the helpers' forward
// decls.
fn wrap_user_source(user_source: &str, hot_reload: bool) -> String {
    let helpers = shader_source(hot_reload, "raymarch_helpers.hlsl", RAYMARCH_HELPERS_HLSL);
    let template = shader_source(hot_reload, "raymarch_template.hlsl", RAYMARCH_TEMPLATE_HLSL);
    format!(
        "{}\n// === user SdfVolume::fragment_shader ===\n{}\n// === engine raymarch template ===\n{}\n",
        helpers, user_source, template
    )
}

// Root signature shared by every per-volume raymarch PSO.
//
//   [0] CBV b0 (RaymarchView)            : root descriptor
//   [1] CBV b1 (SdfVolumeUniforms)       : root descriptor
//   [2] CBV b2 (RaymarchLights)          : root descriptor
//   [3] CBV b3 (RaymarchShadowUniforms)  : root descriptor
//   [4] Descriptor table SRV  t0..t3     : shadow / IBL / scene fallback
//   [5] Descriptor table Sampler s0..s2  : shadow_samp / cube_samp / scene_samp
fn create_raymarch_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 4,
        BaseShaderRegister: 0, // t0..t3 (shadow_map, irradiance, prefilter, scene_color)
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let samp_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 3,
        BaseShaderRegister: 0, // s0..s2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let cbv = |reg: u32, vis: D3D12_SHADER_VISIBILITY| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: vis,
    };
    let params = [
        cbv(0, D3D12_SHADER_VISIBILITY_ALL),
        cbv(1, D3D12_SHADER_VISIBILITY_ALL),
        cbv(2, D3D12_SHADER_VISIBILITY_PIXEL),
        cbv(3, D3D12_SHADER_VISIBILITY_PIXEL),
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &samp_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "raymarch root sig")
}

// Build the per-volume PSO. Front-face culled so back faces of the
// proxy cube rasterise (which works regardless of whether the camera
// is inside or outside the bbox). Depth attachment is the main scene
// depth (D32_FLOAT); the shader writes hit depth via
// `SV_DepthLessEqual` so downstream passes see raymarched-surface
// depth.
fn create_raymarch_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    msaa_samples: u32,
) -> Result<ID3D12PipelineState, String> {
    let input_layout = main_input_layout();
    let mut rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_FRONT,
        FrontCounterClockwise: windows::core::BOOL(0),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: windows::core::BOOL(1),
        MultisampleEnable: windows::core::BOOL(if msaa_samples > 1 { 1 } else { 0 }),
        AntialiasedLineEnable: windows::core::BOOL(0),
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    };
    // No depth bias.
    rasterizer.DepthBias = 0;

    let mut blend = D3D12_BLEND_DESC {
        AlphaToCoverageEnable: windows::core::BOOL(0),
        IndependentBlendEnable: windows::core::BOOL(0),
        RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8],
    };
    blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: windows::core::BOOL(0),
        LogicOpEnable: windows::core::BOOL(0),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };

    // Hardware z-test against the existing MSAA main depth, and write
    // hit depth back via `SV_DepthLessEqual` so downstream
    // depth-sampling passes (decals, fog, SSR) see the raymarched
    // surface. Renders into the MSAA `hdr_color` target so the depth
    // sample-count matches; the encoder re-resolves `hdr_color →
    // hdr_resolve` after the pass.
    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: windows::core::BOOL(1),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        StencilEnable: windows::core::BOOL(0),
        ..Default::default()
    };

    let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
    rtv_formats[0] = HDR_FORMAT;
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as _,
            BytecodeLength: ps.len(),
        },
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        DepthStencilState: depth_stencil,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_layout.as_ptr(),
            NumElements: input_layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: rtv_formats,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: msaa_samples.max(1),
            Quality: 0,
        },
        ..Default::default()
    };
    unsafe { device.CreateGraphicsPipelineState(&desc) }
        .map_err(|e| format!("create raymarch PSO: {e}"))
}

// Returns the wrapped HLSL for a single volume + the asset label used
// in error messages. Bytes are compiled at the caller; the wrap
// itself is allocation-only.
fn compile_volume_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    user_source_bytes: &[u8],
    asset_label: &str,
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<ID3D12PipelineState, String> {
    let user_source = std::str::from_utf8(user_source_bytes).map_err(|e| {
        format!(
            "SdfVolume '{}': fragment shader payload is not valid UTF-8: {}",
            asset_label, e
        )
    })?;
    let wrapped = wrap_user_source(user_source, hot_reload);
    let vs = compile_hlsl(&wrapped, "raymarch_vertex", "vs_5_1")
        .map_err(|e| format!("SdfVolume '{}': vertex compile: {}", asset_label, e))?;
    let ps = compile_hlsl(&wrapped, "raymarch_fragment", "ps_5_1")
        .map_err(|e| format!("SdfVolume '{}': fragment compile: {}", asset_label, e))?;
    create_raymarch_pso(device, root_sig, &vs, &ps, msaa_samples)
}

// Wrap a volumetric user shader: helpers → user → volumetric template.
// FXC DCEs the unused surface forward decls (`map`, `shade`) along
// with engine helpers that reference them (`sdfNormal`, `coneRaymarch`),
// so the volumetric author doesn't need to provide stub definitions.
fn wrap_user_source_volumetric(user_source: &str, hot_reload: bool) -> String {
    let helpers = shader_source(hot_reload, "raymarch_helpers.hlsl", RAYMARCH_HELPERS_HLSL);
    let template = shader_source(
        hot_reload,
        "raymarch_volumetric_template.hlsl",
        RAYMARCH_VOLUMETRIC_TEMPLATE_HLSL,
    );
    format!(
        "{}\n// === user SdfVolume::fragment_shader (volumetric) ===\n{}\n// === engine raymarch volumetric template ===\n{}\n",
        helpers, user_source, template
    )
}

// Volumetric variant of the raymarch PSO: same root signature + same
// vertex layout (cube proxy back faces), but the colour output
// alpha-blends over the existing scene and the depth stencil keeps
// early-z (DepthFunc LESS_EQUAL) without writing: volumetrics are
// translucent and never update the depth buffer.
fn create_raymarch_volumetric_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    msaa_samples: u32,
) -> Result<ID3D12PipelineState, String> {
    let input_layout = main_input_layout();
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_FRONT,
        FrontCounterClockwise: windows::core::BOOL(0),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: windows::core::BOOL(1),
        MultisampleEnable: windows::core::BOOL(if msaa_samples > 1 { 1 } else { 0 }),
        AntialiasedLineEnable: windows::core::BOOL(0),
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    };

    let mut blend = D3D12_BLEND_DESC {
        AlphaToCoverageEnable: windows::core::BOOL(0),
        IndependentBlendEnable: windows::core::BOOL(0),
        RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8],
    };
    blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: windows::core::BOOL(1),
        LogicOpEnable: windows::core::BOOL(0),
        SrcBlend: D3D12_BLEND_SRC_ALPHA,
        DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };

    // Early-z against the bbox far face, but no depth write: the
    // medium doesn't occlude itself or update SSR/decal depth.
    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: windows::core::BOOL(1),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        StencilEnable: windows::core::BOOL(0),
        ..Default::default()
    };

    let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
    rtv_formats[0] = HDR_FORMAT;
    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as _,
            BytecodeLength: ps.len(),
        },
        BlendState: blend,
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        DepthStencilState: depth_stencil,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_layout.as_ptr(),
            NumElements: input_layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: rtv_formats,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: msaa_samples.max(1),
            Quality: 0,
        },
        ..Default::default()
    };
    unsafe { device.CreateGraphicsPipelineState(&desc) }
        .map_err(|e| format!("create raymarch volumetric PSO: {e}"))
}

// Volumetric counterpart of `compile_volume_pso`. Wraps the user
// source with the volumetric template, compiles the vol-specific
// entry points, and builds the alpha-blended PSO.
fn compile_volume_volumetric_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    user_source_bytes: &[u8],
    asset_label: &str,
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<ID3D12PipelineState, String> {
    let user_source = std::str::from_utf8(user_source_bytes).map_err(|e| {
        format!(
            "SdfVolume '{}' (volumetric): fragment shader payload is not valid UTF-8: {}",
            asset_label, e
        )
    })?;
    let wrapped = wrap_user_source_volumetric(user_source, hot_reload);
    let vs = compile_hlsl(&wrapped, "raymarch_volumetric_vertex", "vs_5_1").map_err(|e| {
        format!(
            "SdfVolume '{}' (volumetric): vertex compile: {}",
            asset_label, e
        )
    })?;
    let ps = compile_hlsl(&wrapped, "raymarch_volumetric_fragment", "ps_5_1").map_err(|e| {
        format!(
            "SdfVolume '{}' (volumetric): fragment compile: {}",
            asset_label, e
        )
    })?;
    create_raymarch_volumetric_pso(device, root_sig, &vs, &ps, msaa_samples)
}

// Root signature for the depth-only shadow PSO. Smaller surface than
// the main pass: no SRV / sampler tables since the shadow template
// only marches the SDF and writes depth.
//
//   [0] CBV b0 (RaymarchView)            : root descriptor
//   [1] CBV b1 (SdfVolumeUniforms)       : root descriptor
//   [2] CBV b2 (RaymarchLights)          : root descriptor
//   [3] CBV b3 (RaymarchShadowUniforms)  : root descriptor
//   [4] Root constants b4 (cascade_idx)  : 1 DWORD
fn create_raymarch_shadow_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let cbv = |reg: u32, vis: D3D12_SHADER_VISIBILITY| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: vis,
    };
    let params = [
        cbv(0, D3D12_SHADER_VISIBILITY_ALL),
        cbv(1, D3D12_SHADER_VISIBILITY_ALL),
        // Lights cbuffer is read by the pixel stage only (ray dir).
        cbv(2, D3D12_SHADER_VISIBILITY_PIXEL),
        // Shadow VPs read by both stages (VS projects through the
        // cascade VP, PS reprojects the hit through the same matrix).
        cbv(3, D3D12_SHADER_VISIBILITY_ALL),
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 4,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "raymarch shadow root sig")
}

// Wrap the user's HLSL for the shadow PSO. Helpers → user → shadow
// template. The user's `shade` is dead code (FXC DCE strips it) so
// only `map` ends up sampled by the shadow march.
fn wrap_user_source_shadow(user_source: &str, hot_reload: bool) -> String {
    let helpers = shader_source(hot_reload, "raymarch_helpers.hlsl", RAYMARCH_HELPERS_HLSL);
    let template = shader_source(hot_reload, "raymarch_shadow.hlsl", RAYMARCH_SHADOW_HLSL);
    format!(
        "{}\n// === user SdfVolume::fragment_shader ===\n{}\n// === engine raymarch shadow template ===\n{}\n",
        helpers, user_source, template
    )
}

// Build the depth-only shadow PSO for one volume. No RTV, no MSAA
// (shadow map is single-sample), front-face cull so back-face
// fragments produce rays through the bbox. Writes hit depth via
// `SV_DepthLessEqual`; the writable shadow DSV at draw time supplies
// the comparison. Format matches the existing `create_shadow_pso`:
// D32_FLOAT, sample count 1.
fn create_raymarch_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let input_layout = main_input_layout();
    let rasterizer = D3D12_RASTERIZER_DESC {
        FillMode: D3D12_FILL_MODE_SOLID,
        CullMode: D3D12_CULL_MODE_FRONT,
        FrontCounterClockwise: windows::core::BOOL(0),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: windows::core::BOOL(1),
        MultisampleEnable: windows::core::BOOL(0),
        AntialiasedLineEnable: windows::core::BOOL(0),
        ForcedSampleCount: 0,
        ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
    };

    let depth_stencil = D3D12_DEPTH_STENCIL_DESC {
        DepthEnable: windows::core::BOOL(1),
        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
        DepthFunc: D3D12_COMPARISON_FUNC_LESS,
        StencilEnable: windows::core::BOOL(0),
        ..Default::default()
    };

    let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as _,
            BytecodeLength: ps.len(),
        },
        BlendState: D3D12_BLEND_DESC::default(),
        SampleMask: u32::MAX,
        RasterizerState: rasterizer,
        DepthStencilState: depth_stencil,
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: input_layout.as_ptr(),
            NumElements: input_layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 0,
        RTVFormats: [DXGI_FORMAT_UNKNOWN; 8],
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    unsafe { device.CreateGraphicsPipelineState(&desc) }
        .map_err(|e| format!("create raymarch shadow PSO: {e}"))
}

// Compile and link the per-volume shadow PSO. Mirrors `compile_volume_pso`
// for the main pass; uses the shadow template + shadow root sig.
fn compile_volume_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    user_source_bytes: &[u8],
    asset_label: &str,
    hot_reload: bool,
) -> Result<ID3D12PipelineState, String> {
    let user_source = std::str::from_utf8(user_source_bytes).map_err(|e| {
        format!(
            "SdfVolume '{}': fragment shader payload is not valid UTF-8: {}",
            asset_label, e
        )
    })?;
    let wrapped = wrap_user_source_shadow(user_source, hot_reload);
    let vs = compile_hlsl(&wrapped, "raymarch_shadow_vertex", "vs_5_1")
        .map_err(|e| format!("SdfVolume '{}': shadow vertex compile: {}", asset_label, e))?;
    let ps = compile_hlsl(&wrapped, "raymarch_shadow_fragment", "ps_5_1").map_err(|e| {
        format!(
            "SdfVolume '{}': shadow fragment compile: {}",
            asset_label, e
        )
    })?;
    create_raymarch_shadow_pso(device, root_sig, &vs, &ps)
}

// Build the shared unit-cube proxy geometry. 8 corners at ±1; 36 CCW
// indices (the encoder culls front faces so only back faces fire).
// The vertex shader scales positions by `vol_extent` to land at the
// AABB corners, matching the asset semantic where `extent` is the
// half-widths.
fn build_cube_buffers(
    device: &ID3D12Device,
) -> Result<
    (
        ID3D12Resource,
        ID3D12Resource,
        D3D12_VERTEX_BUFFER_VIEW,
        D3D12_INDEX_BUFFER_VIEW,
    ),
    String,
> {
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

    let vb_bytes = std::mem::size_of_val(&corners) as u64;
    let ib_bytes = std::mem::size_of_val(&indices) as u64;

    let vb = create_buffer(
        device,
        vb_bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let ib = create_buffer(
        device,
        ib_bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    unsafe {
        let mut p = std::ptr::null_mut::<c_void>();
        vb.Map(0, None, Some(&mut p))
            .map_err(|e| format!("raymarch cube vb map: {e}"))?;
        std::ptr::copy_nonoverlapping(
            corners.as_ptr() as *const u8,
            p as *mut u8,
            vb_bytes as usize,
        );
        vb.Unmap(0, None);

        let mut p = std::ptr::null_mut::<c_void>();
        ib.Map(0, None, Some(&mut p))
            .map_err(|e| format!("raymarch cube ib map: {e}"))?;
        std::ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            p as *mut u8,
            ib_bytes as usize,
        );
        ib.Unmap(0, None);
    }

    let vbv = D3D12_VERTEX_BUFFER_VIEW {
        BufferLocation: unsafe { vb.GetGPUVirtualAddress() },
        SizeInBytes: vb_bytes as u32,
        StrideInBytes: std::mem::size_of::<Vertex>() as u32,
    };
    let ibv = D3D12_INDEX_BUFFER_VIEW {
        BufferLocation: unsafe { ib.GetGPUVirtualAddress() },
        SizeInBytes: ib_bytes as u32,
        Format: DXGI_FORMAT_R16_UINT,
    };
    Ok((vb, ib, vbv, ibv))
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

// Write the 4-SRV descriptor table for the raymarch pass into the SRV
// heap, starting at `base_slot`. The slots are:
//   [+0] shadow_map  (Texture2DArray<float>)
//   [+1] irradiance  (TextureCube<float4>)
//   [+2] prefilter   (TextureCube<float4>)
//   [+3] scene_color (Texture2D<float4>, written separately by
//                     `write_scene_color_srv` so resize can re-point
//                     just this slot)
//
// The shadow / IBL descriptors duplicate views the engine already binds
// elsewhere (slots 0 / 1 / 2 of the SRV heap); we re-write them here so
// the raymarch root sig can bind a single contiguous descriptor table
// without multi-range offset trickery.
fn write_raymarch_srvs(
    device: &ID3D12Device,
    shadow_resource: Option<&ID3D12Resource>,
    irradiance_resource: &ID3D12Resource,
    prefilter_resource: &ID3D12Resource,
    base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    descriptor_size: usize,
    shadow_layers: u32,
) {
    let slot_cpu = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: base_cpu.ptr + i * descriptor_size,
    };

    // shadow_map at +0. Texture2DArray<float> matching the main pass's
    // shadow SRV view dimension.
    if let Some(shadow) = shadow_resource {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: 0,
                    MipLevels: 1,
                    FirstArraySlice: 0,
                    ArraySize: shadow_layers,
                    PlaneSlice: 0,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe { device.CreateShaderResourceView(shadow, Some(&desc), slot_cpu(0)) };
    }

    // irradiance + prefilter cubes at +1 / +2. Resource format is
    // R32G32B32A32_FLOAT (see directx/texture.rs::upload_environment_map);
    // the SRV view format must match its family.
    for (i, res) in [irradiance_resource, prefilter_resource].iter().enumerate() {
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                TextureCube: D3D12_TEXCUBE_SRV {
                    MostDetailedMip: 0,
                    MipLevels: u32::MAX,
                    ResourceMinLODClamp: 0.0,
                },
            },
        };
        unsafe { device.CreateShaderResourceView(*res, Some(&desc), slot_cpu(1 + i)) };
    }
    // scene_color at +3, written by `write_scene_color_srv` from the
    // caller so resize can re-point just this slot.
}

// Write the `scene_color` SRV (slot +4 of the raymarch SRV block) at
// the supplied CPU descriptor. Split out so `resize_to` can re-point
// the slot after recreating `hdr_resolve_copy` without rewriting any
// of the other four descriptors.
fn write_scene_color_srv(
    device: &ID3D12Device,
    scene_resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(scene_resource, Some(&desc), srv_cpu) };
}

// Write the 3-sampler descriptor table for the raymarch pass into the
// sampler heap, starting at `base_slot`. The slots are:
//   [+0] shadow_samp (LESS_EQUAL comparison sampler)
//   [+1] cube_samp   (linear-clamp + mip linear, for IBL cubes)
//   [+2] scene_samp  (linear-clamp, for the scene_color tap)
fn write_raymarch_samplers(
    device: &ID3D12Device,
    base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    descriptor_size: usize,
) {
    let slot_cpu = |i: usize| D3D12_CPU_DESCRIPTOR_HANDLE {
        ptr: base_cpu.ptr + i * descriptor_size,
    };
    let shadow = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe { device.CreateSampler(&shadow, slot_cpu(0)) };

    let cube = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe { device.CreateSampler(&cube, slot_cpu(1)) };

    let scene = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ..Default::default()
    };
    unsafe { device.CreateSampler(&scene, slot_cpu(2)) };
}

impl RaymarchResources {
    // Build every raymarch resource and the per-volume records. `sdf_volumes`
    // is the drained-and-payload-paired list from `graphics_system::init`;
    // each volume's `fragment_shader` path is checked here: `.hlsl`
    // payloads compile, anything else (today: `.metal` for Metal-first
    // authors) is skipped with a logged warning. Returns `Ok(None)`
    // when no volume survived the filter so the engine simply omits
    // the pass.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn try_new(
        device: &ID3D12Device,
        info_queue: Option<&ID3D12InfoQueue>,
        command_queue: &ID3D12CommandQueue,
        sdf_volumes: &[(SdfVolume, Vec<u8>, String)],
        width: u32,
        height: u32,
        msaa_samples: u32,
        shadow_resource: Option<&ID3D12Resource>,
        shadow_layers: u32,
        irradiance_resource: &ID3D12Resource,
        prefilter_resource: &ID3D12Resource,
        srv_base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_base_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        srv_descriptor_size: usize,
        sampler_base_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        sampler_base_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        sampler_descriptor_size: usize,
        hot_reload: bool,
    ) -> Result<Option<Self>, String> {
        // Filter `.hlsl` volumes; Metal-first SDFs get dropped with a
        // warning so the rest of the world keeps rendering.
        let active: Vec<&(SdfVolume, Vec<u8>, String)> = sdf_volumes
            .iter()
            .filter(|(v, _, label)| {
                let p = v.fragment_shader.to_ascii_lowercase();
                if p.ends_with(".hlsl") {
                    true
                } else {
                    tracing::warn!(
                        "SdfVolume '{}': fragment shader '{}' is not .hlsl; \
                         skipping on DirectX (Metal-first SDF, the rest of \
                         the world still renders)",
                        label,
                        v.fragment_shader
                    );
                    false
                }
            })
            .collect();
        if active.is_empty() {
            return Ok(None);
        }

        let root_sig = dump_on_err(info_queue, create_raymarch_root_signature(device))?;
        let shadow_root_sig =
            dump_on_err(info_queue, create_raymarch_shadow_root_signature(device))?;

        let (cube_vb, cube_ib, cube_vbv, cube_ibv) = build_cube_buffers(device)?;

        // Per-frame view cbuffer ring.
        let view_size = align256(std::mem::size_of::<RaymarchView>() as u64);
        let mut view_cbuffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut view_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut p = std::ptr::null_mut::<c_void>();
            unsafe { buf.Map(0, None, Some(&mut p)) }
                .map_err(|e| format!("raymarch view ubo map: {e}"))?;
            view_ptrs.push(p as *mut u8);
            view_cbuffers.push(buf);
        }

        // 1×1 white fallback retained for the resource lifetime (would
        // be needed again if a per-volume opt-out re-points the SRV
        // slot away from the snapshot).
        let scene_color_fallback = create_fallback_white_resource(device, command_queue)?;

        // Pre-raymarch HDR scene snapshot; `encode_raymarch` copies
        // `hdr_resolve` / `hdr_color` into this resource each frame
        // before binding the SRV. Sized to render dims; created in
        // COPY_DEST so the first frame's `CopyResource` doesn't need
        // a leading transition.
        let hdr_resolve_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;

        // Build per-volume records. Any failure here aborts init:
        // unlike the .hlsl filter above, a compile error in an active
        // volume is a developer-time bug, not a graceful fallback.
        let mut volumes: Vec<RaymarchVolumeRecord> = Vec::with_capacity(active.len());
        for (vol, bytes, label) in &active {
            let pso = dump_on_err(
                info_queue,
                if vol.volumetric {
                    compile_volume_volumetric_pso(
                        device,
                        &root_sig,
                        bytes,
                        label,
                        msaa_samples,
                        hot_reload,
                    )
                } else {
                    compile_volume_pso(device, &root_sig, bytes, label, msaa_samples, hot_reload)
                },
            )?;
            // Shadow PSO only when the asset opts in. Compile failures
            // here abort init alongside the main PSO; the shadow
            // template is engine-shipped, so the only realistic failure
            // is a user `map` that doesn't compile in HLSL, which would
            // already have failed for the main PSO above.
            let shadow_pso = if vol.cast_shadows {
                Some(dump_on_err(
                    info_queue,
                    compile_volume_shadow_pso(device, &shadow_root_sig, bytes, label, hot_reload),
                )?)
            } else {
                None
            };
            // Per-volume cbuffer (static: `centre`, `extent`,
            // `params` don't change frame-to-frame).
            let uniforms = volume_uniforms_from(vol);
            let cb_size = align256(std::mem::size_of::<RaymarchVolumeUniforms>() as u64);
            let cb = create_buffer(
                device,
                cb_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut p = std::ptr::null_mut::<c_void>();
            unsafe { cb.Map(0, None, Some(&mut p)) }
                .map_err(|e| format!("raymarch volume cb map: {e}"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &uniforms as *const RaymarchVolumeUniforms as *const u8,
                    p as *mut u8,
                    std::mem::size_of::<RaymarchVolumeUniforms>(),
                );
                // Persistently mapped, never unmap.
            }
            let gva = unsafe { cb.GetGPUVirtualAddress() };
            volumes.push(RaymarchVolumeRecord {
                pso,
                shadow_pso,
                volume_cbuffer: cb,
                volume_cbuffer_gva: gva,
                visible: vol.visible,
                cast_shadows: vol.cast_shadows,
                world_centre: vol.centre,
                world_extent: vol.extent,
            });
        }

        // Write the descriptor tables.
        write_raymarch_srvs(
            device,
            shadow_resource,
            irradiance_resource,
            prefilter_resource,
            srv_base_cpu,
            srv_descriptor_size,
            shadow_layers,
        );
        // scene_color slot points at the snapshot. Captured CPU handle
        // so `resize_to` can re-point the descriptor after the
        // snapshot is recreated at the new resolution.
        let scene_color_srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_base_cpu.ptr + 3 * srv_descriptor_size,
        };
        write_scene_color_srv(device, &hdr_resolve_copy, scene_color_srv_cpu);
        write_raymarch_samplers(device, sampler_base_cpu, sampler_descriptor_size);

        Ok(Some(Self {
            root_sig,
            shadow_root_sig,
            cube_vb,
            cube_ib,
            cube_vbv,
            cube_ibv,
            view_cbuffers,
            view_ptrs,
            scene_color_fallback,
            hdr_resolve_copy,
            scene_color_srv_cpu,
            srv_table_gpu: srv_base_gpu,
            sampler_table_gpu: sampler_base_gpu,
            volumes,
        }))
    }

    // Recreate the HDR scene snapshot at new render-target dimensions
    // and rewrite the `scene_color` SRV descriptor in place. Called
    // from the swapchain-resize handler. The descriptor *slot* itself
    // doesn't move; the live raymarch root-table binding (which holds
    // the GPU handle) stays valid without a re-bind.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.hdr_resolve_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_color_srv(device, &self.hdr_resolve_copy, self.scene_color_srv_cpu);
        Ok(())
    }

    // True when any volume in the world is currently visible. Used by
    // `record_frame` to flip `FrameGraphInputs::raymarch_enabled`.
    pub(in crate::directx) fn any_visible(&self) -> bool {
        self.volumes.iter().any(|v| v.visible)
    }
}

// Pointer drops in the resources struct are POD-style raw pointers; the
// underlying mapped upload buffers stay alive through the `Vec<ID3D12Resource>`
// fields, and the pointers are read on the render thread only.
unsafe impl Send for RaymarchResources {}
unsafe impl Sync for RaymarchResources {}

impl DxContext {
    // Encode the raymarched SDF volume pass onto `cmd`. Called from the
    // render graph executor for `PassId::Raymarch`. Assumes the main
    // HDR target was resolved into `hdr_resolve` by the Main pass and
    // that resolve sits in `PIXEL_SHADER_RESOURCE` state (the post-
    // Main pipeline contract). Restores the same state at the end so
    // Decals + Fog + SSR / TAA / Bloom continue to see the resolve as
    // a sampler input.
    pub(in crate::directx) fn encode_raymarch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view: &RaymarchView,
    ) -> Result<(), String> {
        let Some(rm) = self.raymarch.as_ref() else {
            return Ok(());
        };
        if !rm.any_visible() {
            return Ok(());
        }

        // Upload this frame's view into the cbuffer ring.
        let view_ptr = rm
            .view_ptrs
            .get(frame_idx)
            .copied()
            .ok_or("raymarch: view_ptrs index OOB")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const RaymarchView as *const u8,
                view_ptr,
                std::mem::size_of::<RaymarchView>(),
            );
        }
        let view_gva = unsafe { rm.view_cbuffers[frame_idx].GetGPUVirtualAddress() };
        let light_gva = unsafe { self.uniforms.light_ubo.GetGPUVirtualAddress() };
        let shadow_gva =
            unsafe { self.uniforms.shadow_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // State entering this pass (post-Main, post-AutoExposure):
        //   * hdr_color (MSAA):  RENDER_TARGET  (the Main pass resolved
        //                        into hdr_resolve and restored hdr_color
        //                        back to RENDER_TARGET; see the resolve
        //                        block in `directx/draw/main.rs`).
        //   * hdr_color (no MSAA): PIXEL_SHADER_RESOURCE.
        //   * hdr_resolve:       PIXEL_SHADER_RESOURCE  (AutoExposure
        //                        sampled it; only present when MSAA on).
        //   * depth_resource:    DEPTH_WRITE.
        //
        // We snapshot the single-sample scene for the refraction SRV,
        // then render the raymarch into the MSAA `hdr_color` target
        // with the writable MSAA DSV bound. After the draws we
        // re-resolve hdr_color → hdr_resolve so all downstream
        // single-sample post-stack passes (Decals, Fog, SsrResolve,
        // TaaResolve, Bloom, Composite) pick up the raymarched
        // colour AND the raymarched-surface depth (which flowed into
        // `depth_resource` via SV_DepthLessEqual). The MSAA-off path
        // skips the resolve and renders into hdr_color directly.
        let msaa = self.hdr.resolve.is_some();

        // Pre-pass: snapshot the resolved scene into `hdr_resolve_copy`
        // for refractive user shaders. Source is hdr_resolve on the
        // MSAA path, hdr_color on the MSAA-off path; both rest in
        // PIXEL_SHADER_RESOURCE at this point.
        let snapshot_src = if msaa {
            self.hdr
                .resolve
                .as_ref()
                .expect("hdr_resolve checked above")
        } else {
            &self.hdr.color
        };
        let snapshot_src_to_copy = transition_barrier(
            snapshot_src,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let snapshot_dst_to_copy = transition_barrier(
            &rm.hdr_resolve_copy,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        unsafe { cmd.ResourceBarrier(&[snapshot_src_to_copy, snapshot_dst_to_copy]) };
        unsafe { cmd.CopyResource(&rm.hdr_resolve_copy, snapshot_src) };
        let snapshot_src_back = transition_barrier(
            snapshot_src,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let snapshot_dst_to_psr = transition_barrier(
            &rm.hdr_resolve_copy,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[snapshot_src_back, snapshot_dst_to_psr]) };

        // On the MSAA path hdr_color is already in RENDER_TARGET; no
        // transition needed. On the MSAA-off path we flip
        // PIXEL_SHADER_RESOURCE → RENDER_TARGET for the draw. Depth
        // stays in DEPTH_WRITE; the DSV is writable + the LESS_EQUAL
        // test composites against existing rasterised depth.
        if !msaa {
            let hdr_color_to_rt = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { cmd.ResourceBarrier(&[hdr_color_to_rt]) };
        }

        let w = self.render_width;
        let h = self.render_height;
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&self.hdr.color_rtv), false, Some(&self.depth_dsv));
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[rm.cube_vbv]));
            cmd.IASetIndexBuffer(Some(&rm.cube_ibv));

            cmd.SetGraphicsRootSignature(&rm.root_sig);
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootConstantBufferView(2, light_gva);
            cmd.SetGraphicsRootConstantBufferView(3, shadow_gva);
            cmd.SetGraphicsRootDescriptorTable(4, rm.srv_table_gpu);
            cmd.SetGraphicsRootDescriptorTable(5, rm.sampler_table_gpu);
        }

        for vol in &rm.volumes {
            if !vol.visible {
                continue;
            }
            unsafe {
                cmd.SetPipelineState(&vol.pso);
                cmd.SetGraphicsRootConstantBufferView(1, vol.volume_cbuffer_gva);
                cmd.DrawIndexedInstanced(36, 1, 0, 0, 0);
            }
            self.inc_draw_calls(1);
        }

        // Post-pass.
        //
        // MSAA path: re-resolve hdr_color → hdr_resolve so downstream
        // single-sample readers see the composited scene + raymarched
        // pixels. Restore hdr_color to RENDER_TARGET (matching Main's
        // post-resolve baseline) and hdr_resolve to
        // PIXEL_SHADER_RESOURCE for Decals / Fog / SsrResolve / etc.
        //
        // MSAA-off path: just flip hdr_color back to
        // PIXEL_SHADER_RESOURCE so the downstream chain reads it as a
        // texture. No resolve needed.
        if msaa {
            let hdr_resolve = self
                .hdr
                .resolve
                .as_ref()
                .expect("hdr_resolve checked above");
            let hdr_color_to_resolve_src = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            let resolve_to_dst = transition_barrier(
                hdr_resolve,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
            );
            unsafe {
                cmd.ResourceBarrier(&[hdr_color_to_resolve_src, resolve_to_dst]);
                cmd.ResolveSubresource(hdr_resolve, 0, &self.hdr.color, 0, HDR_FORMAT);
            }
            let resolve_back = transition_barrier(
                hdr_resolve,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            let hdr_color_back_to_rt = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { cmd.ResourceBarrier(&[resolve_back, hdr_color_back_to_rt]) };
        } else {
            let hdr_color_to_psr = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            unsafe { cmd.ResourceBarrier(&[hdr_color_to_psr]) };
        }
        Ok(())
    }
}

impl RaymarchResources {
    // True when at least one volume both opted in to shadow casting at
    // init (`cast_shadows`, gated to `Some(shadow_pso)`) AND is visible
    // this frame. Drives the early-out for the shadow caster encoder
    // so non-cast-shadow worlds pay zero cost.
    pub(in crate::directx) fn any_shadow_casters(&self) -> bool {
        self.volumes
            .iter()
            .any(|v| v.visible && v.cast_shadows && v.shadow_pso.is_some())
    }
}

impl DxContext {
    // Encode the raymarched SDF shadow casters into the existing CSM
    // shadow DSVs, right before `encode_shadow_pass` transitions the
    // shadow map array to `PIXEL_SHADER_RESOURCE`. One draw per visible
    // caster per cascade; the proxy unit cube rasterises through the
    // cascade's light VP (front-face cull means back faces produce one
    // fragment per texel inside the box), the depth-only fragment
    // marches the SDF, and writes the hit's NDC.z via
    // `SV_DepthLessEqual` so the cascade DSV's existing LESS depth test
    // keeps only the nearest caster between rasterised and raymarched.
    //
    // Uploads `view` into the same per-frame cbuffer ring
    // `encode_raymarch` uses. Bytes for the same frame_idx are written
    // twice (once here, once by `encode_raymarch` later in the frame);
    // both writes are byte-identical when the caller passes the same
    // view, and the shadow march only reads `view_time` so even if the
    // later write differs in fields the shadow path ignores, behaviour
    // is unchanged.
    pub(in crate::directx) fn encode_sdf_shadow_casters(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        shadow_ubo_gva: u64,
        view: &RaymarchView,
    ) -> Result<(), String> {
        let Some(rm) = self.raymarch.as_ref() else {
            return Ok(());
        };
        if !rm.any_shadow_casters() {
            return Ok(());
        }
        if self.shadow.dsvs.is_empty() {
            return Ok(());
        }

        // Share the cbuffer ring with the main raymarch pass. See the
        // docstring above for why the double-write is safe.
        let view_ptr = rm
            .view_ptrs
            .get(frame_idx)
            .copied()
            .ok_or("raymarch shadow: view_ptrs index OOB")?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const RaymarchView as *const u8,
                view_ptr,
                std::mem::size_of::<RaymarchView>(),
            );
        }
        let view_gva = unsafe { rm.view_cbuffers[frame_idx].GetGPUVirtualAddress() };
        let light_gva = unsafe { self.uniforms.light_ubo.GetGPUVirtualAddress() };

        let sm = self.shadow.map_size;
        unsafe {
            cmd.SetGraphicsRootSignature(&rm.shadow_root_sig);
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd.IASetVertexBuffers(0, Some(&[rm.cube_vbv]));
            cmd.IASetIndexBuffer(Some(&rm.cube_ibv));

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: sm as f32,
                Height: sm as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: sm as i32,
                bottom: sm as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);

            // Shared cbuffers (view (b0), lights (b2), shadow (b3)) stay
            // bound across all cascades and all volumes. Volume cbuffer
            // (b1) + cascade root constant (b4) + PSO get updated inside
            // the loop.
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootConstantBufferView(2, light_gva);
            cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
        }

        // Only cast into cascades the rasterised shadow pass re-rendered this
        // frame: a skipped cascade's slice must stay exactly as it was last fully
        // rendered (raster + SDF), so we neither clear nor add to it. The 0
        // sentinel falls back to all cascades. Mirrors Metal.
        let all_cascades = (1u32 << crate::gfx::render_types::NUM_SHADOW_CASCADES) - 1;
        let render_mask = if self.shadow.render_mask == 0 {
            all_cascades
        } else {
            self.shadow.render_mask
        };
        for cascade_idx in 0..crate::gfx::render_types::NUM_SHADOW_CASCADES {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let dsv = self.shadow.dsvs[cascade_idx];
            unsafe {
                cmd.OMSetRenderTargets(0, None, false, Some(&dsv));
                let constants = [cascade_idx as u32, 0u32, 0u32, 0u32];
                cmd.SetGraphicsRoot32BitConstants(
                    4,
                    4,
                    constants.as_ptr() as *const std::ffi::c_void,
                    0,
                );
            }
            for vol in &rm.volumes {
                if !vol.visible || !vol.cast_shadows {
                    continue;
                }
                let Some(pso) = vol.shadow_pso.as_ref() else {
                    continue;
                };
                unsafe {
                    cmd.SetPipelineState(pso);
                    cmd.SetGraphicsRootConstantBufferView(1, vol.volume_cbuffer_gva);
                    cmd.DrawIndexedInstanced(36, 1, 0, 0, 0);
                }
                self.inc_draw_calls(1);
            }
        }
        Ok(())
    }
}

// Silence the unused-import warning when no `SdfVolume` ships in a build.
// `LightUniforms` is referenced only through the cbuffer GVA we read
// from `self.uniforms.light_ubo`, but the type is part of the encode_raymarch
// contract surface for future readers.
const _LIGHT_LAYOUT_REF: usize = std::mem::size_of::<LightUniforms>();

#[cfg(test)]
mod tests {
    use super::*;

    // RaymarchView must match the `RaymarchView` cbuffer (b0) in
    // shaders/raymarch_helpers.hlsl: two column-major float4x4 then the
    // packed cam_pos/pad/viewport/time/prefilter scalars (160 B total).
    #[test]
    fn raymarch_view_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<RaymarchView>(), 160);
        assert_eq!(std::mem::offset_of!(RaymarchView, vp), 0);
        assert_eq!(std::mem::offset_of!(RaymarchView, inv_vp), 64);
        assert_eq!(std::mem::offset_of!(RaymarchView, cam_pos), 128);
        assert_eq!(std::mem::offset_of!(RaymarchView, _pad0), 140);
        assert_eq!(std::mem::offset_of!(RaymarchView, viewport), 144);
        assert_eq!(std::mem::offset_of!(RaymarchView, time), 152);
        assert_eq!(std::mem::offset_of!(RaymarchView, prefilter_mip_count), 156);
    }

    // RaymarchVolumeUniforms must match the `SdfVolumeUniforms` cbuffer (b1):
    // centre/pad, extent/pad, the four scalars, then 32 floats of params
    // packed as 8 float4 rows (176 B total).
    #[test]
    fn raymarch_volume_uniforms_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<RaymarchVolumeUniforms>(), 176);
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, centre), 0);
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, _pad0), 12);
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, extent), 16);
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, _pad1), 28);
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, cone_ratio), 32);
        assert_eq!(
            std::mem::offset_of!(RaymarchVolumeUniforms, max_distance),
            36
        );
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, max_steps), 40);
        assert_eq!(
            std::mem::offset_of!(RaymarchVolumeUniforms, receive_shadows),
            44
        );
        assert_eq!(std::mem::offset_of!(RaymarchVolumeUniforms, params), 48);
    }
}
