// src/directx/post/gbuffer.rs
//
// Unified geometry G-buffer pre-pass for the D3D12 backend. One jittered
// traversal of the visible set (static + instanced + skinned, via the shared
// draw_iter helpers) rasterises into a single MRT:
//
//   target 0  RGBA16F  view-space normal (rgb) + positive linear view depth (a)
//   target 1  R8       perceptual roughness
//   target 2  RG16F    screen-space motion (prev_uv - cur_uv)
//
// plus a private single-sample depth buffer. Every screen-space consumer (SSR
// resolve, SSAO kernel/blur, SSGI gather/composite, TAA resolve, FSR upscaler)
// reads this one output instead of re-rasterising, replacing the separate
// SsrPrepass + SSAO pre-pass + Velocity passes. Rasterisation uses the jittered
// VP (matching the main pass coverage); the motion vector derives from the
// un-jittered current / previous VPs in-shader so projection jitter never
// contaminates motion. Mirrors src/metal/post/gbuffer.rs.

use std::cell::RefCell;

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::math::IDENTITY4;
use crate::directx::pipeline::{
    compile_hlsl, main_input_layout, serialize_and_create_root_sig, shader_source,
    skinned_input_layout,
};
use crate::directx::texture::{
    create_buffer, create_main_depth_texture, create_rt_target, create_rt_target_with_clear,
    transition_barrier, write_format_rtv, write_format_srv,
};

// HLSL sources

pub const GBUFFER_PREPASS_VERT_HLSL: &str = include_str!("../shaders/gbuffer_prepass_vert.hlsl");
pub const GBUFFER_PREPASS_VERT_INSTANCED_HLSL: &str =
    include_str!("../shaders/gbuffer_prepass_vert_instanced.hlsl");
pub const GBUFFER_PREPASS_VERT_SKINNED_HLSL: &str =
    include_str!("../shaders/gbuffer_prepass_vert_skinned.hlsl");
pub const GBUFFER_PREPASS_FRAG_HLSL: &str = include_str!("../shaders/gbuffer_prepass_frag.hlsl");

// GPU-driven (bindless) G-buffer pre-pass shaders. The VS reads model +
// roughness from the per-frame `GpuObjectData` buffer by the per-command b0
// object-id root constant and the previous-frame model from a parallel buffer;
// the FS mirrors gbuffer_prepass_frag but sources roughness from a flat VS
// varying. Drive the same MRT the legacy pre-pass writes, but reuse the main
// pass's GPU-culled indirect command buffer (one `ExecuteIndirect` per region).
pub const GBUFFER_BINDLESS_VERT_HLSL: &str = include_str!("../shaders/gbuffer_bindless_vert.hlsl");
pub const GBUFFER_BINDLESS_FRAG_HLSL: &str = include_str!("../shaders/gbuffer_bindless_frag.hlsl");

// Normal+depth target: rgb = unit view-space normal, a = positive linear view
// depth (-view_z). Alpha 0 (cleared background) marks "no geometry". Matches
// the SSR / SSAO G-buffer so the resolve / kernel maths is byte-identical.
pub const GBUFFER_NORMAL_DEPTH_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Single-channel perceptual roughness. 1.0 = fully rough (cleared background),
// 0.0 = mirror.
pub const GBUFFER_ROUGHNESS_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8_UNORM;

// Screen-space motion (prev_uv - cur_uv). Cleared to 0 (no motion).
pub const GBUFFER_VELOCITY_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16_FLOAT;

// Background roughness the prepass clears the roughness target to: fully rough,
// so untouched pixels emit no reflection. Used for both the optimized clear
// value and the per-frame clear so they match.
const GBUFFER_ROUGHNESS_CLEAR: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

// Size of the per-frame view UBO: jittered_vp + cur_vp + prev_vp + view_mat
// (four float4x4 = 256 B). Matches the `GbView` cbuffer in every pre-pass VS.
const GBUFFER_VIEW_UBO_SIZE: u64 = 256;

// View block uploaded to the G-buffer pre-pass vertex shader. Matches the
// `GbView` cbuffer (b0) in gbuffer_prepass_vert*.hlsl: the jittered VP
// rasterises, the un-jittered cur/prev VPs drive the motion vector, and the
// view matrix transforms the normal + depth.
#[derive(Copy, Clone)]
#[repr(C)]
struct GbViewUniforms {
    jittered_vp: [[f32; 4]; 4],
    cur_vp: [[f32; 4]; 4],
    prev_vp: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
}

// Per-draw model push (32 root constants at b1). Matches the `GbModel` cbuffer
// in the static + skinned pre-pass VS: the current and previous model matrices
// for the motion vector.
#[derive(Copy, Clone)]
#[repr(C)]
struct GbModelPush {
    cur_model: [[f32; 4]; 4],
    prev_model: [[f32; 4]; 4],
}

// Shader compilation

struct GbufferShaders {
    vs_static: Vec<u8>,
    vs_instanced: Vec<u8>,
    vs_skinned: Vec<u8>,
    ps: Vec<u8>,
}

// Compile every G-buffer pre-pass shader stage. `need_instanced` /
// `need_skinned` gate the geometry-kind-specific vertex shaders.
fn compile_gbuffer_shaders(
    need_instanced: bool,
    need_skinned: bool,
    hot_reload: bool,
) -> Result<GbufferShaders, String> {
    Ok(GbufferShaders {
        vs_static: compile_hlsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass_vert.hlsl",
                GBUFFER_PREPASS_VERT_HLSL,
            ),
            "main",
            "vs_5_1",
        )?,
        vs_instanced: if need_instanced {
            compile_hlsl(
                &shader_source(
                    hot_reload,
                    "gbuffer_prepass_vert_instanced.hlsl",
                    GBUFFER_PREPASS_VERT_INSTANCED_HLSL,
                ),
                "main",
                "vs_5_1",
            )?
        } else {
            Vec::new()
        },
        vs_skinned: if need_skinned {
            compile_hlsl(
                &shader_source(
                    hot_reload,
                    "gbuffer_prepass_vert_skinned.hlsl",
                    GBUFFER_PREPASS_VERT_SKINNED_HLSL,
                ),
                "main",
                "vs_5_1",
            )?
        } else {
            Vec::new()
        },
        ps: compile_hlsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass_frag.hlsl",
                GBUFFER_PREPASS_FRAG_HLSL,
            ),
            "main",
            "ps_5_1",
        )?,
    })
}

// Root signatures

// Static: root CBV at b0 (GbView), 32 root constants at b1 (cur+prev model),
// 4 root constants at b0 PS-visibility (roughness).
fn create_gbuffer_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Num32BitValues: 32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    serialize_and_create_root_sig(device, &params, "gbuffer prepass root sig")
}

// Instanced: root CBV at b0 (GbView), root SRV at t0 (per-instance models),
// 4 root constants at b0 PS-visibility (roughness).
fn create_gbuffer_instanced_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    serialize_and_create_root_sig(device, &params, "gbuffer prepass instanced root sig")
}

// Skinned: root CBV at b0 (GbView), 32 root constants at b1 (cur+prev model),
// root SRV at t0 (current joints), root SRV at t1 (previous joints), 4 root
// constants at b0 PS-visibility (roughness).
fn create_gbuffer_skinned_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Num32BitValues: 32,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    serialize_and_create_root_sig(device, &params, "gbuffer prepass skinned root sig")
}

// PSO for the static / instanced / skinned G-buffer pre-pass. Writes the three
// MRT targets over a private single-sample depth buffer. Mirrors the main
// pass's no-cull rasteriser + LESS depth test so the G-buffer matches the main
// pass's visible surfaces.
fn create_gbuffer_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    layout: &[D3D12_INPUT_ELEMENT_DESC],
) -> Result<ID3D12PipelineState, String> {
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
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
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 3,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = GBUFFER_NORMAL_DEPTH_FORMAT;
            a[1] = GBUFFER_ROUGHNESS_FORMAT;
            a[2] = GBUFFER_VELOCITY_FORMAT;
            a
        },
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: {
                let mut arr = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
                let mt = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                arr[0] = mt;
                arr[1] = mt;
                arr[2] = mt;
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };
    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create gbuffer prepass PSO: {e}"))
}

// Vertex input layout for the GPU-driven (bindless) G-buffer pre-pass: the
// current-frame attributes the VS reads (position / normal / colour for the
// skybox sentinel) on slot 0, plus the previous-frame position on slot 1. Both
// slots carry the 56-byte `Vertex`; the static prefix binds the static VB to
// both slots (prev_pos == cur_pos), the skinned tail binds the current deformed
// buffer to slot 0 and the previous-frame deformed buffer to slot 1. Tangent +
// UV are unused (the pre-pass samples no textures), so they are omitted.
fn gbuffer_bindless_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    vec![
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("NORMAL"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 12,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 36,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 1,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 1,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

// Root signature for the GPU-driven G-buffer pre-pass. Mirrors the shadow
// bindless root signature's object-id delivery so the shared cull command
// signature works against it: [0] is the per-command b0 object-id root constant
// (set by the `ExecuteIndirect` command signature, so it MUST stay at root
// parameter 0), [1] the GbView CBV (jittered/cur/prev VP + view matrix), [2] the
// per-frame `StructuredBuffer<GpuObjectData>` (model + roughness), and [3] the
// parallel previous-frame model buffer. All vertex-stage only (roughness reaches
// the pixel shader through a flat varying; the FS reads no resources).
fn create_gbuffer_bindless_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] Root constant b0: object id (set per command by the command sig).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Root CBV b1: GbView (jittered_vp + cur_vp + prev_vp + view).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [2] Root SRV t0: per-frame StructuredBuffer<GpuObjectData>.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [3] Root SRV t1: per-frame previous-frame model buffer.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
    ];
    serialize_and_create_root_sig(device, &params, "gbuffer bindless root sig")
}

// Build the GPU-driven G-buffer pre-pass pipeline: the bindless VS/FS, its root
// signature, and the shared cull command signature rebuilt against that root sig
// (object id at root param 0). Returns the trio the cull state stores; the
// per-frame `prev_model` buffers it reads are allocated alongside the other cull
// buffers. Reuses `create_gbuffer_pso` (3 MRT, private D32, single-sample, LESS
// depth) with the two-stream bindless input layout.
#[allow(clippy::type_complexity)]
pub(in crate::directx) fn build_gbuffer_bindless(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    hot_reload: bool,
) -> Result<
    (
        ID3D12RootSignature,
        ID3D12PipelineState,
        ID3D12CommandSignature,
    ),
    String,
> {
    let vs = compile_hlsl(
        &shader_source(
            hot_reload,
            "gbuffer_bindless_vert.hlsl",
            GBUFFER_BINDLESS_VERT_HLSL,
        ),
        "main",
        "vs_5_1",
    )?;
    let ps = compile_hlsl(
        &shader_source(
            hot_reload,
            "gbuffer_bindless_frag.hlsl",
            GBUFFER_BINDLESS_FRAG_HLSL,
        ),
        "main",
        "ps_5_1",
    )?;
    let root_sig = dump_on_err(info_queue, create_gbuffer_bindless_root_signature(device))?;
    let layout = gbuffer_bindless_input_layout();
    let pso = dump_on_err(
        info_queue,
        create_gbuffer_pso(device, &root_sig, &vs, &ps, &layout),
    )?;
    let cmd_sig = dump_on_err(
        info_queue,
        crate::directx::cull::create_cull_command_signature(device, &root_sig),
    )?;
    Ok((root_sig, pso, cmd_sig))
}

// Descriptor-slot handles for the three G-buffer SRVs, minted by the caller
// (which owns the heap layout). `Copy` so the caller can both pass it to `new`
// and stash a copy for the live `apply_quality_settings` rebuild.
#[derive(Clone, Copy)]
pub(in crate::directx) struct GbufferSlots {
    pub normal_depth_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub normal_depth_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub roughness_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub roughness_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub velocity_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub velocity_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
}

// Unified G-buffer resources held by `DxContext` when any screen-space consumer
// (SSR, SSGI, SSAO, TAA, or temporal upscaling) is enabled. Drops cleanly with
// the context: every D3D12 object is COM-refcounted.
pub(in crate::directx) struct GbufferResources {
    // MRT targets + their private single-sample depth.
    pub(in crate::directx) normal_depth: ID3D12Resource,
    pub(in crate::directx) normal_depth_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) normal_depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) roughness: ID3D12Resource,
    pub(in crate::directx) roughness_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) roughness_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) velocity: ID3D12Resource,
    pub(in crate::directx) velocity_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) velocity_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    #[allow(dead_code)]
    pub(in crate::directx) depth: ID3D12Resource,
    pub(in crate::directx) depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,

    // Per-frame view UBO (jittered_vp + cur_vp + prev_vp + view), mapped.
    pub(in crate::directx) view_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) view_ubo_ptrs: Vec<*mut u8>,

    // Pipelines. Instanced / skinned are `Some` only when the world declares
    // that geometry kind (the skinned one builds lazily via `ensure_skinned_pso`
    // once the joint-bound vertex layout exists).
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    pub(in crate::directx) pso: ID3D12PipelineState,
    pub(in crate::directx) instanced_root_sig: Option<ID3D12RootSignature>,
    pub(in crate::directx) instanced_pso: Option<ID3D12PipelineState>,
    pub(in crate::directx) skinned_root_sig: Option<ID3D12RootSignature>,
    pub(in crate::directx) skinned_pso: Option<ID3D12PipelineState>,

    // Previous-frame motion state, owned here so the velocity channel works for
    // any consumer (TAA or FSR) independent of whether engine-TAA is on.
    // `prev_view_proj` is last frame's un-jittered VP; `prev_models` is each
    // draw's previous transform. Both advance once per frame in `record_frame`.
    pub(in crate::directx) prev_view_proj: RefCell<[[f32; 4]; 4]>,
    pub(in crate::directx) prev_models: RefCell<Vec<[[f32; 4]; 4]>>,
}

impl GbufferResources {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        need_instanced: bool,
        need_skinned: bool,
        slots: GbufferSlots,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let normal_depth = create_rt_target(device, width, height, GBUFFER_NORMAL_DEPTH_FORMAT)?;
        write_format_rtv(
            device,
            &normal_depth,
            slots.normal_depth_rtv,
            GBUFFER_NORMAL_DEPTH_FORMAT,
        );
        write_format_srv(
            device,
            &normal_depth,
            slots.normal_depth_srv.0,
            GBUFFER_NORMAL_DEPTH_FORMAT,
        );

        let roughness = create_rt_target_with_clear(
            device,
            width,
            height,
            GBUFFER_ROUGHNESS_FORMAT,
            GBUFFER_ROUGHNESS_CLEAR,
        )?;
        write_format_rtv(
            device,
            &roughness,
            slots.roughness_rtv,
            GBUFFER_ROUGHNESS_FORMAT,
        );
        write_format_srv(
            device,
            &roughness,
            slots.roughness_srv.0,
            GBUFFER_ROUGHNESS_FORMAT,
        );

        let velocity = create_rt_target(device, width, height, GBUFFER_VELOCITY_FORMAT)?;
        write_format_rtv(
            device,
            &velocity,
            slots.velocity_rtv,
            GBUFFER_VELOCITY_FORMAT,
        );
        write_format_srv(
            device,
            &velocity,
            slots.velocity_srv.0,
            GBUFFER_VELOCITY_FORMAT,
        );

        let depth = create_main_depth_texture(device, width, height, slots.depth_dsv, 1, true)?;

        // Per-frame view UBO.
        let view_size = align256(GBUFFER_VIEW_UBO_SIZE);
        let mut view_ubo_resources: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut view_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                view_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map gbuffer view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        // Pipelines.
        let shaders = compile_gbuffer_shaders(need_instanced, need_skinned, hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_gbuffer_root_signature(device))?;
        let static_layout = main_input_layout();
        let pso = dump_on_err(
            info_queue,
            create_gbuffer_pso(
                device,
                &root_sig,
                &shaders.vs_static,
                &shaders.ps,
                &static_layout,
            ),
        )?;

        let (instanced_root_sig, instanced_pso) = if need_instanced {
            let rs = dump_on_err(info_queue, create_gbuffer_instanced_root_signature(device))?;
            let pso = dump_on_err(
                info_queue,
                create_gbuffer_pso(
                    device,
                    &rs,
                    &shaders.vs_instanced,
                    &shaders.ps,
                    &static_layout,
                ),
            )?;
            (Some(rs), Some(pso))
        } else {
            (None, None)
        };

        let (skinned_root_sig, skinned_pso) = if need_skinned {
            let rs = dump_on_err(info_queue, create_gbuffer_skinned_root_signature(device))?;
            let layout = skinned_input_layout();
            let pso = dump_on_err(
                info_queue,
                create_gbuffer_pso(device, &rs, &shaders.vs_skinned, &shaders.ps, &layout),
            )?;
            (Some(rs), Some(pso))
        } else {
            (None, None)
        };

        Ok(Self {
            normal_depth,
            normal_depth_rtv: slots.normal_depth_rtv,
            normal_depth_srv_gpu: slots.normal_depth_srv.1,
            roughness,
            roughness_rtv: slots.roughness_rtv,
            roughness_srv_gpu: slots.roughness_srv.1,
            velocity,
            velocity_rtv: slots.velocity_rtv,
            velocity_srv_gpu: slots.velocity_srv.1,
            depth,
            depth_dsv: slots.depth_dsv,
            view_ubo_resources,
            view_ubo_ptrs,
            root_sig,
            pso,
            instanced_root_sig,
            instanced_pso,
            skinned_root_sig,
            skinned_pso,
            prev_view_proj: RefCell::new(IDENTITY4),
            prev_models: RefCell::new(Vec::new()),
        })
    }

    // Build the skinned G-buffer pre-pass root signature + PSO. Called by
    // `upload_skinned` once the skinned vertex layout exists. Idempotent: a
    // second call replaces the existing PSO.
    pub(in crate::directx) fn ensure_skinned_pso(
        &mut self,
        device: &ID3D12Device,
        hot_reload: bool,
        info_queue: Option<&ID3D12InfoQueue>,
    ) -> Result<(), String> {
        let vs = compile_hlsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass_vert_skinned.hlsl",
                GBUFFER_PREPASS_VERT_SKINNED_HLSL,
            ),
            "main",
            "vs_5_1",
        )?;
        let ps = compile_hlsl(
            &shader_source(
                hot_reload,
                "gbuffer_prepass_frag.hlsl",
                GBUFFER_PREPASS_FRAG_HLSL,
            ),
            "main",
            "ps_5_1",
        )?;
        let root_sig = match self.skinned_root_sig.as_ref() {
            Some(rs) => rs.clone(),
            None => dump_on_err(info_queue, create_gbuffer_skinned_root_signature(device))?,
        };
        let layout = skinned_input_layout();
        let pso = dump_on_err(
            info_queue,
            create_gbuffer_pso(device, &root_sig, &vs, &ps, &layout),
        )?;
        self.skinned_root_sig = Some(root_sig);
        self.skinned_pso = Some(pso);
        Ok(())
    }

    // Rebuild the MRT targets + private depth at a new resolution. The
    // descriptor *slots* stay put; only the resources backing them change.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<(), String> {
        let srv_cpu = |gpu: D3D12_GPU_DESCRIPTOR_HANDLE| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_cpu_base.ptr + (gpu.ptr - srv_gpu_base.ptr) as usize,
        };

        self.normal_depth = create_rt_target(device, width, height, GBUFFER_NORMAL_DEPTH_FORMAT)?;
        write_format_rtv(
            device,
            &self.normal_depth,
            self.normal_depth_rtv,
            GBUFFER_NORMAL_DEPTH_FORMAT,
        );
        write_format_srv(
            device,
            &self.normal_depth,
            srv_cpu(self.normal_depth_srv_gpu),
            GBUFFER_NORMAL_DEPTH_FORMAT,
        );

        self.roughness = create_rt_target_with_clear(
            device,
            width,
            height,
            GBUFFER_ROUGHNESS_FORMAT,
            GBUFFER_ROUGHNESS_CLEAR,
        )?;
        write_format_rtv(
            device,
            &self.roughness,
            self.roughness_rtv,
            GBUFFER_ROUGHNESS_FORMAT,
        );
        write_format_srv(
            device,
            &self.roughness,
            srv_cpu(self.roughness_srv_gpu),
            GBUFFER_ROUGHNESS_FORMAT,
        );

        self.velocity = create_rt_target(device, width, height, GBUFFER_VELOCITY_FORMAT)?;
        write_format_rtv(
            device,
            &self.velocity,
            self.velocity_rtv,
            GBUFFER_VELOCITY_FORMAT,
        );
        write_format_srv(
            device,
            &self.velocity,
            srv_cpu(self.velocity_srv_gpu),
            GBUFFER_VELOCITY_FORMAT,
        );

        self.depth = create_main_depth_texture(device, width, height, self.depth_dsv, 1, true)?;
        Ok(())
    }
}

// Replacement G-buffer PSOs returned by a hot-reload rebuild. Each field is
// `Some` when the matching live PSO exists; the caller swaps them in atomically
// only if every required build succeeded.
pub(in crate::directx) struct RebuiltGbufferPipelines {
    pub pso: ID3D12PipelineState,
    pub instanced_pso: Option<ID3D12PipelineState>,
    pub skinned_pso: Option<ID3D12PipelineState>,
}

// Rebuild the G-buffer pre-pass PSOs from disk-resident HLSL for shader
// hot-reload. Returns `None` for a variant whose live PSO does not exist, so
// the caller leaves it untouched.
pub(in crate::directx) fn rebuild_gbuffer_pipelines(
    device: &ID3D12Device,
    gbuffer: &GbufferResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltGbufferPipelines, String> {
    let shaders = compile_gbuffer_shaders(
        gbuffer.instanced_pso.is_some(),
        gbuffer.skinned_pso.is_some(),
        hot_reload,
    )?;
    let static_layout = main_input_layout();
    let pso = dump_on_err(
        info_queue,
        create_gbuffer_pso(
            device,
            &gbuffer.root_sig,
            &shaders.vs_static,
            &shaders.ps,
            &static_layout,
        ),
    )?;
    let instanced_pso = match gbuffer.instanced_root_sig.as_ref() {
        Some(rs) => Some(dump_on_err(
            info_queue,
            create_gbuffer_pso(
                device,
                rs,
                &shaders.vs_instanced,
                &shaders.ps,
                &static_layout,
            ),
        )?),
        None => None,
    };
    let skinned_pso = match gbuffer.skinned_root_sig.as_ref() {
        Some(rs) => {
            let layout = skinned_input_layout();
            Some(dump_on_err(
                info_queue,
                create_gbuffer_pso(device, rs, &shaders.vs_skinned, &shaders.ps, &layout),
            )?)
        }
        None => None,
    };
    Ok(RebuiltGbufferPipelines {
        pso,
        instanced_pso,
        skinned_pso,
    })
}

impl DxContext {
    // Encode the unified G-buffer pre-pass: one jittered traversal of the
    // visible set (static + instanced + skinned) into the normal+depth /
    // roughness / velocity MRT. `velocity_active` is true when a consumer (TAA
    // or FSR) reads motion; when false, cur == prev so the motion channel is a
    // harmless zero.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn encode_gbuffer_prepass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        jittered_vp: [[f32; 4]; 4],
        cur_vp: [[f32; 4]; 4],
        visible: &[u32],
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        let gb = match &self.gbuffer {
            Some(g) => g,
            None => return,
        };

        // Upload this frame's view UBO. When velocity is inactive the previous
        // VP equals the current one, so instanced + sky motion is zero.
        let prev_vp = if velocity_active {
            *gb.prev_view_proj.borrow()
        } else {
            cur_vp
        };
        let view_uni = GbViewUniforms {
            jittered_vp,
            cur_vp,
            prev_vp,
            view: self.view_matrix,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const GbViewUniforms as *const u8,
                gb.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<GbViewUniforms>(),
            );
        }
        let view_gva = unsafe { gb.view_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        let w = self.render_width;
        let h = self.render_height;

        // Targets → RENDER_TARGET, clear, draw.
        let nd_to_rt = transition_barrier(
            &gb.normal_depth,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let r_to_rt = transition_barrier(
            &gb.roughness,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let v_to_rt = transition_barrier(
            &gb.velocity,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { cmd.ResourceBarrier(&[nd_to_rt, r_to_rt, v_to_rt]) };

        let rtvs = [gb.normal_depth_rtv, gb.roughness_rtv, gb.velocity_rtv];
        unsafe {
            cmd.OMSetRenderTargets(3, Some(rtvs.as_ptr()), false, Some(&gb.depth_dsv));
            // Cleared alpha 0 marks "no geometry"; roughness 1.0 = non-reflective
            // background; velocity 0 = no motion.
            cmd.ClearRenderTargetView(gb.normal_depth_rtv, &[0.0_f32; 4], None);
            cmd.ClearRenderTargetView(gb.roughness_rtv, &GBUFFER_ROUGHNESS_CLEAR, None);
            cmd.ClearRenderTargetView(gb.velocity_rtv, &[0.0_f32; 4], None);
            cmd.ClearDepthStencilView(gb.depth_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: w as i32,
                bottom: h as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        }

        // When the bindless GPU-cull path is active, the G-buffer pre-pass is
        // GPU-driven: it reuses the main pass's per-frame indirect command buffer
        // (same camera frustum + active LOD) with two `ExecuteIndirect` draws
        // (static + instance prefix, then the skinned tail over the deformed VB),
        // instead of the CPU per-object loops -- plus a legacy extra loop for
        // streamed chunks / runtime clones not in the cull records. A non-bindless
        // world (custom shader) keeps the legacy path. Both write the same MRT, so
        // the targets -> pixel-shader-resource transition below is shared.
        if self.cull.gbuffer_bindless_pso.is_some() && self.cull_count() > 0 {
            self.encode_gbuffer_prepass_gpu_driven(
                cmd,
                frame_idx,
                view_gva,
                visible,
                cam_pos,
                velocity_active,
            );
        } else {
            self.encode_gbuffer_prepass_legacy(
                cmd,
                frame_idx,
                view_gva,
                visible,
                frustum,
                cam_pos,
                velocity_active,
            );
        }

        // Targets → PIXEL_SHADER_RESOURCE for the consumers.
        let nd_to_psr = transition_barrier(
            &gb.normal_depth,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let r_to_psr = transition_barrier(
            &gb.roughness,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let v_to_psr = transition_barrier(
            &gb.velocity,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[nd_to_psr, r_to_psr, v_to_psr]) };
    }

    // Legacy CPU-driven G-buffer pre-pass: per-object `DrawIndexedInstanced` for
    // static + instanced + skinned geometry. Used for non-bindless worlds (custom
    // shader) or worlds with no build-time geometry. The caller has already
    // transitioned the targets to RENDER_TARGET, cleared them, and set the
    // viewport / scissor / topology.
    #[allow(clippy::too_many_arguments)]
    fn encode_gbuffer_prepass_legacy(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view_gva: u64,
        visible: &[u32],
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        let gb = match &self.gbuffer {
            Some(g) => g,
            None => return,
        };
        unsafe {
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));

            cmd.SetPipelineState(&gb.pso);
            cmd.SetGraphicsRootSignature(&gb.root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
        }

        // Static geometry: same visible set + LOD pick as the main pass so the
        // G-buffer covers exactly what main rasterised.
        {
            let prev_models = gb.prev_models.borrow();
            self.draw_static_objects(visible, cam_pos, |obj, i, index_offset, index_count| {
                let prev_model = if velocity_active {
                    prev_models.get(i).copied().unwrap_or(obj.model)
                } else {
                    obj.model
                };
                let push = GbModelPush {
                    cur_model: obj.model,
                    prev_model,
                };
                let mat = [obj.material.roughness, 0.0_f32, 0.0, 0.0];
                unsafe {
                    cmd.SetGraphicsRoot32BitConstants(
                        1,
                        32,
                        &push as *const GbModelPush as *const std::ffi::c_void,
                        0,
                    );
                    cmd.SetGraphicsRoot32BitConstants(
                        2,
                        4,
                        mat.as_ptr() as *const std::ffi::c_void,
                        0,
                    );
                    cmd.DrawIndexedInstanced(
                        index_count as u32,
                        1,
                        index_offset as u32,
                        obj.base_vertex,
                        0,
                    );
                }
            });
        }

        // GPU-instanced clusters: instance transforms never change, so the
        // motion is camera-only (the instanced VS feeds the same matrix to cur
        // and prev clip). Reuses the per-cluster matrix buffer the main
        // instanced pass already filled this frame; roughness rides b0(PS), the
        // per-bucket instance SRV bumps t0.
        if let (Some(inst_pso), Some(inst_root_sig)) =
            (gb.instanced_pso.as_ref(), gb.instanced_root_sig.as_ref())
            && !self.instanced.clusters.is_empty()
        {
            unsafe {
                cmd.SetPipelineState(inst_pso);
                cmd.SetGraphicsRootSignature(inst_root_sig);
                cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            }
            self.draw_instanced_clusters(
                frame_idx,
                frustum,
                cam_pos,
                |_cluster_idx, cluster| {
                    let mat = [cluster.material.roughness, 0.0_f32, 0.0, 0.0];
                    unsafe {
                        cmd.SetGraphicsRoot32BitConstants(
                            2,
                            4,
                            mat.as_ptr() as *const std::ffi::c_void,
                            0,
                        );
                    }
                },
                |bucket, inst_gva_base| unsafe {
                    cmd.SetGraphicsRootShaderResourceView(
                        1,
                        inst_gva_base + bucket.instance_byte_offset,
                    );
                    cmd.DrawIndexedInstanced(
                        bucket.index_count as u32,
                        bucket.instance_count,
                        bucket.index_offset as u32,
                        0,
                        0,
                    );
                },
            );
        }

        // Skinned meshes: redraw with the current + previous pose so per-vertex
        // deformation produces a correct motion vector. The model matrix is
        // static (skinned meshes are self-placing), so cur and prev model are
        // identical; the deformation motion comes from the current +
        // previous-frame joint palettes at t0 / t1. Previous joints live in the
        // per-frame joint ring at slot (frame_idx - 1) mod FRAMES.
        if let (Some(sk_pso), Some(sk_root_sig)) =
            (gb.skinned_pso.as_ref(), gb.skinned_root_sig.as_ref())
            && !self.skinned.draw_objects.is_empty()
        {
            let prev_frame_idx = (frame_idx + FRAMES - 1) % FRAMES;
            unsafe {
                cmd.SetPipelineState(sk_pso);
                cmd.SetGraphicsRootSignature(sk_root_sig);
                cmd.IASetVertexBuffers(0, Some(&[self.skinned.vertex_buffer_view]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            }
            self.draw_skinned_objects(cam_pos, |obj, i, index_offset, index_count| {
                let push = GbModelPush {
                    cur_model: obj.model,
                    prev_model: obj.model,
                };
                let mat = [obj.material.roughness, 0.0_f32, 0.0, 0.0];
                // When velocity is inactive, point the previous palette at
                // the current one so the motion channel stays zero.
                let prev_slot = if velocity_active {
                    prev_frame_idx
                } else {
                    frame_idx
                };
                unsafe {
                    cmd.SetGraphicsRoot32BitConstants(
                        1,
                        32,
                        &push as *const GbModelPush as *const std::ffi::c_void,
                        0,
                    );
                    cmd.SetGraphicsRootShaderResourceView(2, self.skinned_joint_gva(frame_idx, i));
                    cmd.SetGraphicsRootShaderResourceView(3, self.skinned_joint_gva(prev_slot, i));
                    cmd.SetGraphicsRoot32BitConstants(
                        4,
                        4,
                        mat.as_ptr() as *const std::ffi::c_void,
                        0,
                    );
                    cmd.DrawIndexedInstanced(index_count as u32, 1, index_offset as u32, 0, 0);
                }
            });
            // Restore the static vertex/index buffers for later passes.
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
                cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            }
        }
    }

    // GPU-driven G-buffer pre-pass raster. Reuses the main pass's per-frame
    // indirect command buffer (the camera-frustum cull already produced it, so no
    // extra cull dispatch) with two `ExecuteIndirect` draws: the static + instance
    // prefix `[0, skinned_record_base())` over the static VB (bound to BOTH vertex
    // streams, so prev_pos == cur_pos and the motion is the per-object model delta
    // plus camera), then the skinned tail `[skinned_record_base(), cull_count())`
    // over the current deformed VB (slot 0) + the previous-frame deformed VB
    // (slot 1), so per-vertex skin deformation produces a correct motion vector.
    // model + roughness ride the per-frame GpuObjectData buffer; the previous-frame
    // model rides a parallel buffer. Streamed chunks / runtime clones (records past
    // `n_objects`) keep a legacy per-object loop. The CPU never walks the static /
    // skinned draw lists.
    #[allow(clippy::too_many_arguments)]
    fn encode_gbuffer_prepass_gpu_driven(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view_gva: u64,
        visible: &[u32],
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        let (Some(pso), Some(root_sig), Some(cmd_sig), Some(prev_model_res)) = (
            self.cull.gbuffer_bindless_pso.as_ref(),
            self.cull.gbuffer_bindless_root_sig.as_ref(),
            self.cull.gbuffer_bindless_cmd_sig.as_ref(),
            self.cull.prev_model_buffers.get(frame_idx),
        ) else {
            return;
        };
        let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
        let stride = crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize;
        let prefix = self.skinned_record_base();
        let object_gva =
            unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };

        // Build this frame's previous-frame model buffer (static + skinned regions;
        // the instance region is init-written + immutable). Honours velocity_active.
        self.build_gbuffer_prev_models(frame_idx, velocity_active);
        let prev_model_gva = unsafe { prev_model_res.GetGPUVirtualAddress() };

        // Static + instance prefix: bind the static VB to BOTH vertex streams
        // (prev_pos == cur_pos) + the static u32 IB, then one `ExecuteIndirect`
        // over `[0, skinned_record_base())`.
        unsafe {
            cmd.SetPipelineState(pso);
            cmd.SetGraphicsRootSignature(root_sig);
            cmd.IASetVertexBuffers(
                0,
                Some(&[
                    self.geometry.vertex_buffer_view,
                    self.geometry.vertex_buffer_view,
                ]),
            );
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            // [1] GbView, [2] GpuObjectData, [3] previous-frame models.
            cmd.SetGraphicsRootConstantBufferView(1, view_gva);
            cmd.SetGraphicsRootShaderResourceView(2, object_gva);
            cmd.SetGraphicsRootShaderResourceView(3, prev_model_gva);
            cmd.ExecuteIndirect(
                cmd_sig,
                prefix as u32,
                indirect,
                0,
                None::<&ID3D12Resource>,
                0,
            );
        }
        self.inc_draw_calls(1);

        // Skinned tail: bind the current deformed VB (slot 0) + the previous-frame
        // deformed VB (slot 1) + the skinned u16 IB, then one `ExecuteIndirect`
        // over `[skinned_record_base(), cull_count())`. The records carry
        // base_vertex = 0 (global skinned indexing). When velocity is inactive the
        // previous deformed VB is the current one, so prev_pos == cur_pos and the
        // motion channel stays zero (GbView prev_vp also equals cur_vp).
        if self.n_skinned > 0
            && let Some(cur_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
        {
            // Read the previous frame's deformed pose only once the ring has been
            // primed (a prior frame's `encode_skin` filled that slot). On the
            // first frame (or after a runtime ring rebuild) the prev slot is
            // unposed, so bind the current deformed buffer as the previous one --
            // prev_pos == cur_pos gives a harmless zero skinned motion vector
            // instead of garbage. Same collapse `velocity_active == false` uses.
            let use_prev_pose = velocity_active
                && self
                    .skinned
                    .deformed_primed
                    .load(std::sync::atomic::Ordering::Relaxed);
            let prev_frame_idx = if use_prev_pose {
                (frame_idx + FRAMES - 1) % FRAMES
            } else {
                frame_idx
            };
            let prev_vbv = self
                .skinned
                .deformed_vbvs
                .get(prev_frame_idx)
                .copied()
                .unwrap_or(*cur_vbv);
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[*cur_vbv, prev_vbv]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                cmd.ExecuteIndirect(
                    cmd_sig,
                    self.n_skinned as u32,
                    indirect,
                    (prefix * stride) as u64,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);
            // The current deformed buffer is posed this frame, so next frame's
            // history slot (this slot) is valid -- prime the ring.
            self.skinned
                .deformed_primed
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Legacy extra: streamed chunks + runtime clones (records past `n_objects`)
        // are not in the GpuObjectData buffer, so draw them with the legacy
        // per-object pipeline into the same MRT. Converged by the chunk phase.
        self.encode_gbuffer_legacy_extra(cmd, view_gva, visible, cam_pos, velocity_active);
    }

    // Legacy per-object G-buffer draws for runtime clones past the bindless range
    // (`i >= n_objects` AND in `clone.slot_by_draw_idx`). Streamed VoxelWorld chunks
    // now fold into the GPU-driven cull records (drawn by the prefix indirect draw),
    // so they are skipped here. Mirrors the legacy static loop, appending into the
    // same MRT after the indirect draws (no re-clear). A no-op for worlds with no
    // clones (the common case, incl. pure-voxel worlds).
    fn encode_gbuffer_legacy_extra(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        view_gva: u64,
        visible: &[u32],
        cam_pos: [f32; 3],
        velocity_active: bool,
    ) {
        if self.clone.slot_by_draw_idx.is_empty() {
            return;
        }
        let gb = match &self.gbuffer {
            Some(g) => g,
            None => return,
        };
        unsafe {
            cmd.SetPipelineState(&gb.pso);
            cmd.SetGraphicsRootSignature(&gb.root_sig);
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
        }
        let prev_models = gb.prev_models.borrow();
        self.draw_static_objects(visible, cam_pos, |obj, i, index_offset, index_count| {
            if i < self.n_objects {
                return; // build-time object, already drawn via ExecuteIndirect
            }
            if !self.clone.slot_by_draw_idx.contains_key(&i) {
                return; // streamed chunk -> folded into the cull records
            }
            let prev_model = if velocity_active {
                prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            let push = GbModelPush {
                cur_model: obj.model,
                prev_model,
            };
            let mat = [obj.material.roughness, 0.0_f32, 0.0, 0.0];
            unsafe {
                cmd.SetGraphicsRoot32BitConstants(
                    1,
                    32,
                    &push as *const GbModelPush as *const std::ffi::c_void,
                    0,
                );
                cmd.SetGraphicsRoot32BitConstants(2, 4, mat.as_ptr() as *const std::ffi::c_void, 0);
                cmd.DrawIndexedInstanced(
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
            }
            self.inc_draw_calls(1);
        });
    }

    // Fill this frame's previous-frame model buffer for the GPU-driven G-buffer
    // velocity. Indexed by cull record id, parallel to the GpuObjectData buffer:
    // the static prefix `[0, n_objects)` gets last frame's model (so a moving
    // static object reprojects correctly), the chunk region
    // `[chunk_record_base(), +n_chunk)` gets the chunk's current model (camera-only
    // velocity -- chunk terrain is static-in-world; the camera-relative origin
    // rebase nets to zero screen motion, matching the legacy chunk path), the
    // skinned tail `[skinned_record_base(), cull_count())` gets the current model
    // (skinned deformation motion comes from the previous-frame deformed buffer).
    // The instance region `[n_objects, chunk_record_base())` is init-written +
    // immutable. When velocity is inactive every written record gets its current
    // model, so the motion channel stays zero (GbView prev_vp also equals cur_vp).
    // Mirrors build_object_buffer's record indexing.
    fn build_gbuffer_prev_models(&self, frame_idx: usize, velocity_active: bool) {
        let Some(&ptr) = self.cull.prev_model_buffer_ptrs.get(frame_idx) else {
            return;
        };
        let Some(gb) = self.gbuffer.as_ref() else {
            return;
        };
        let stride = std::mem::size_of::<[[f32; 4]; 4]>();
        let prev_models = gb.prev_models.borrow();
        for (i, obj) in self.draw_objects.iter().take(self.n_objects).enumerate() {
            let prev = if velocity_active {
                prev_models.get(i).copied().unwrap_or(obj.model)
            } else {
                obj.model
            };
            // SAFETY: the buffer was sized for `cull_count()` records and the loop
            // is bounded by `take(n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &prev as *const [[f32; 4]; 4] as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }
        // Streamed chunks: current model -> camera-only velocity. (Unused reserve
        // slots keep stale prev_models, but their draw-args are disabled, so the
        // gbuffer never rasterises them.)
        let chunk_base = self.chunk_record_base();
        self.for_each_chunk_record(|k, obj| {
            let prev = obj.model;
            // SAFETY: `for_each_chunk_record` caps `k < n_chunk`, so
            // `chunk_base + k < skinned_record_base()`, in range for `cull_count()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &prev as *const [[f32; 4]; 4] as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        });
        let base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.n_skinned)
            .enumerate()
        {
            // Skinned motion is per-vertex (previous deformed buffer), so the model
            // matrix is the current one (cur == prev model, like the legacy path).
            let prev = obj.model;
            // SAFETY: the buffer reserved `n_skinned` records past
            // `skinned_record_base()` at init; the loop is bounded by
            // `self.skinned.draw_objects.len() == self.n_skinned`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &prev as *const [[f32; 4]; 4] as *const u8,
                    ptr.add((base + k) * stride),
                    stride,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GbViewUniforms must match the `GbView` cbuffer (b0) in every pre-pass VS:
    // four column-major float4x4 at offsets 0, 64, 128, 192 (256 B total).
    #[test]
    fn gb_view_uniforms_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<GbViewUniforms>(), 256);
        assert_eq!(std::mem::offset_of!(GbViewUniforms, jittered_vp), 0);
        assert_eq!(std::mem::offset_of!(GbViewUniforms, cur_vp), 64);
        assert_eq!(std::mem::offset_of!(GbViewUniforms, prev_vp), 128);
        assert_eq!(std::mem::offset_of!(GbViewUniforms, view), 192);
        // Upload size must not exceed the 256-aligned UBO allocation.
        assert!(std::mem::size_of::<GbViewUniforms>() as u64 <= align256(GBUFFER_VIEW_UBO_SIZE));
    }

    // GbModelPush is pushed as 32 root constants at b1, matching the `GbModel`
    // cbuffer: cur_model then prev_model (two column-major float4x4).
    #[test]
    fn gb_model_push_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<GbModelPush>(), 128);
        assert_eq!(std::mem::size_of::<GbModelPush>() / 4, 32);
        assert_eq!(std::mem::offset_of!(GbModelPush, cur_model), 0);
        assert_eq!(std::mem::offset_of!(GbModelPush, prev_model), 64);
    }
}
