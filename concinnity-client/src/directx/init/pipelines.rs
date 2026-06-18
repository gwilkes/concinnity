// src/directx/init/pipelines.rs
//
// Core render-pipeline construction extracted from DxContext::new:
//   * Built-in HLSL shader sources for main + shadow passes (the equivalents
//     of Metal's vertex/fragment_main metallibs).
//   * Shader compilation (`compile_shaders`, `compile_main_bindless_shaders`).
//   * Root-signature + PSO builders for the main pass, the GPU-cull bindless
//     variant, the GPU-instanced main pass, and the depth-only shadow pass.
//   * High-level `build_main_pipelines`/`build_shadow_pipeline`/etc.
//     orchestration helpers consumed by init/mod.rs.
//
// Mirrors src/metal/init/pipelines.rs (the same set of pipelines built at
// init time). Text + composite pipelines live in `directx/pipeline.rs`;
// bloom/TAA/SSAO live in `directx/post/`; the GPU-cull compute pipeline lives
// in `directx/cull.rs`; the skinned-mesh pipelines (built lazily once a
// `SkinnedMesh` is uploaded) live in `directx/resources.rs`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::{FRAMES, align256, dump_on_err};
use crate::directx::cull::{
    INDIRECT_COMMAND_STRIDE, compile_cull_shader, compile_cull_shader_phase2,
    compile_cull_shader_shadow, create_cull_command_signature, create_cull_pso,
    create_cull_root_signature,
};
use crate::directx::pipeline::{
    compile_composite_shaders, compile_hlsl, compile_text_shaders, create_composite_pso,
    create_composite_root_signature, create_text_pso, create_text_root_signature,
    main_input_layout, serialize_and_create_root_sig, shader_source,
};
use crate::directx::texture::{HDR_FORMAT, create_buffer, create_uav_buffer};

// Built-in HLSL sources for the main + shadow passes
//
// The build emits HLSL bytecode into `build/shaders/`; the SHADOW_VERT_HLSL
// path matches the per-world shadow VS. Built-ins are used when the world
// supplied no override; pre-compiled DXBC overrides skip these.

const MAIN_VERT_HLSL: &str = concinnity_core::build::shader::BUILTIN_DEFAULT_VERT_HLSL;
const MAIN_FRAG_HLSL: &str = concinnity_core::build::shader::BUILTIN_DEFAULT_FRAG_HLSL;

// Bindless siblings of MAIN_VERT_HLSL / MAIN_FRAG_HLSL for the bindless
// static main pass. Instead of rebinding the model matrix + material
// per draw, each object's record lives in a per-frame
// `StructuredBuffer<GpuObjectData>` (root SRV at t3); the draw call passes the
// object id through the b0 root constant. Albedo + normal maps are fetched
// from an unbounded bindless `Texture2D` pool (register space1) by the
// per-object pool indices. Only build-time static objects render through
// these; streamed VoxelWorld chunks keep the legacy per-draw pipeline
// (MAIN_VERT_HLSL / MAIN_FRAG_HLSL), which the instanced + skinned passes also
// still use. The fragment BRDF mirrors MAIN_FRAG_HLSL; only the
// model/material/texture binding model differs.
const MAIN_VERT_BINDLESS_HLSL: &str = include_str!("../shaders/main_bindless_vert.hlsl");
const MAIN_FRAG_BINDLESS_HLSL: &str = include_str!("../shaders/main_bindless_frag.hlsl");

// GPU-instanced sibling of MAIN_VERT_HLSL. Reads per-instance world matrices
// from a root SRV at t3 instead of the PushConstants `model` field (which is
// ignored here). Paired with the regular MAIN_FRAG_HLSL.
const MAIN_VERT_INSTANCED_HLSL: &str =
    concinnity_core::build::shader::BUILTIN_DEFAULT_VERT_INSTANCED_HLSL;

const SHADOW_VERT_HLSL: &str = concinnity_core::build::shader::BUILTIN_SHADOW_MAP_VERT_HLSL;

// Depth-only bindless sibling of SHADOW_VERT_HLSL for the GPU-driven shadow pass.
// Reads `model` from the per-frame `StructuredBuffer<GpuObjectData>` (root SRV at
// t0) by the per-command b0 object-id root constant and projects through
// `light_vps[cascade_idx]` (cascade index = a per-ExecuteIndirect b2 root
// constant). Consumes the same cull-written indirect buffers the bindless main
// pass uses, so the shadow pass issues each cascade with one `ExecuteIndirect`
// instead of a CPU per-object loop.
const SHADOW_VERT_BINDLESS_HLSL: &str = include_str!("../shaders/shadow_bindless_vert.hlsl");

// Shader compilation

pub(super) struct CompiledShaders {
    pub main_vs: Vec<u8>,
    pub main_ps: Vec<u8>,
    pub shadow_vs: Option<Vec<u8>>,
    pub main_vs_instanced: Option<Vec<u8>>,
    pub text_vs: Vec<u8>,
    pub text_ps: Vec<u8>,
}

// Compile every shader stage the init path needs. `vert_bytes`, `frag_bytes`,
// `vert_instanced_bytes`, and `shadow_bytes` are pre-compiled DXBC overrides
// when non-empty (matching the metallib override model on Metal). `shadow_vs`
// is `None` when no shadow shader is configured.
pub(super) fn compile_all_shaders(
    vert_bytes: &[u8],
    frag_bytes: &[u8],
    shadow_bytes: &[u8],
    vert_instanced_bytes: &[u8],
    need_instanced: bool,
    hot_reload: bool,
) -> Result<CompiledShaders, String> {
    let main_vs = if !vert_bytes.is_empty() {
        vert_bytes.to_vec()
    } else {
        compile_hlsl(MAIN_VERT_HLSL, "main", "vs_5_1")?
    };
    let main_ps = if !frag_bytes.is_empty() {
        frag_bytes.to_vec()
    } else {
        compile_hlsl(MAIN_FRAG_HLSL, "main", "ps_5_1")?
    };
    // The shadow vertex shader is engine-internal: a real DXBC override (>4
    // bytes) is used verbatim, otherwise (empty / stub) the baked
    // SHADOW_VERT_HLSL is compiled. Whether the shadow pass runs is gated by
    // `effective_shadow_size` at the call site, not by an empty override here.
    let shadow_vs = if shadow_bytes.len() > 4 {
        Some(shadow_bytes.to_vec())
    } else {
        Some(compile_hlsl(SHADOW_VERT_HLSL, "main", "vs_5_1")?)
    };
    let main_vs_instanced = if !vert_instanced_bytes.is_empty() {
        Some(vert_instanced_bytes.to_vec())
    } else if need_instanced {
        Some(compile_hlsl(MAIN_VERT_INSTANCED_HLSL, "main", "vs_5_1")?)
    } else {
        None
    };
    let (text_vs, text_ps) = compile_text_shaders(hot_reload)?;
    Ok(CompiledShaders {
        main_vs,
        main_ps,
        shadow_vs,
        main_vs_instanced,
        text_vs,
        text_ps,
    })
}

// Compile the bindless static-pass shaders (bindless static pass). Always built
// from the inline HLSL; the bindless path only ever drives the built-in
// shader; worlds that supply a custom main shader keep the legacy pipeline.
pub(in crate::directx) fn compile_main_bindless_shaders(
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vs = compile_hlsl(
        &shader_source(
            hot_reload,
            "main_bindless_vert.hlsl",
            MAIN_VERT_BINDLESS_HLSL,
        ),
        "main",
        "vs_5_1",
    )?;
    let ps = compile_hlsl(
        &shader_source(
            hot_reload,
            "main_bindless_frag.hlsl",
            MAIN_FRAG_BINDLESS_HLSL,
        ),
        "main",
        "ps_5_1",
    )?;
    Ok((vs, ps))
}

// Compile the GPU-driven shadow pass's depth-only bindless vertex shader. Built
// alongside the bindless main pass (same built-in-shader gate); a depth-only
// PSO with no pixel shader consumes it.
pub(in crate::directx) fn compile_shadow_bindless_vs(hot_reload: bool) -> Result<Vec<u8>, String> {
    compile_hlsl(
        &shader_source(
            hot_reload,
            "shadow_bindless_vert.hlsl",
            SHADOW_VERT_BINDLESS_HLSL,
        ),
        "main",
        "vs_5_1",
    )
}

// Root signature builders

fn create_main_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    // Descriptor ranges for tables.
    // [4] table layout (SRVs at heap slots 0..3 inclusive):
    //   range 1: shadow_map_array at t0    (heap slot 0)
    //   range 2: irradiance + prefilter cubes at t5..t6 (heap slots 1..2)
    // Both ranges use APPEND so the runtime places them back-to-back from the
    // table base; matches the heap layout in context.rs.
    let shadow_srv_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0, // t0
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 2,
            BaseShaderRegister: 5, // t5..t6
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
    ];
    let object_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // t1..t2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let shadow_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // s0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [7] table covers linear repeat sampler (s1) + cube sampler (s2)
    // contiguous in the sampler heap.
    let linear_cube_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // s1..s2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [8] table: SSAO occlusion SRV at t4 (or a 1x1 white fallback so the
    // shader's ambient *= ssao_tex.r is a pass-through when SSAO is off).
    let ssao_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 4, // t4
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constants: model mat4 + material = 28 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 28,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Root CBV: view UBO at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root CBV: light UBO at b2
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] Root CBV: shadow UBO at b3
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [4] Descriptor table: shadow map array (t0) + IBL cubes (t5..t6)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: shadow_srv_ranges.len() as u32,
                    pDescriptorRanges: shadow_srv_ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [5] Descriptor table: albedo + normal SRVs (t1..t2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &object_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [6] Descriptor table: shadow comparison sampler (s0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &shadow_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [7] Descriptor table: linear repeat (s1) + cube sampler (s2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &linear_cube_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [8] Descriptor table: SSAO occlusion SRV (t4)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ssao_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    serialize_and_create_root_sig(device, &params, "main root sig")
}

// Root signature for the bindless static main pass (bindless static pass).
//
// Differs from `create_main_root_signature`: slot [0] is a single-DWORD root
// constant carrying just the per-draw object id (D3D12 `SV_InstanceID` does
// not include `StartInstanceLocation`, so the id rides a root constant); slot
// [5] is the unbounded bindless `Texture2D` pool (`t0, space1`) instead of the
// per-object albedo/normal table; slot [8] is a root SRV at `t3` carrying the
// per-frame `StructuredBuffer<GpuObjectData>`. The per-object descriptor table
// is gone; that was the per-draw binding the compute-driven cull
// needed removed.
fn create_main_bindless_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let shadow_srv_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0, // t0
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 2,
            BaseShaderRegister: 5, // t5..t6
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
    ];
    // Unbounded bindless pool: `Texture2D tex_pool[] : register(t0, space1)`.
    // The table base GPU handle points at the per-object SRV region (heap slot
    // `object_base_slot`), so pool index `2*i` / `2*i+1` resolves to object
    // `i`'s albedo / normal SRV.
    let pool_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: u32::MAX, // unbounded
        BaseShaderRegister: 0,    // t0
        RegisterSpace: 1,         // space1
        OffsetInDescriptorsFromTableStart: 0,
    };
    let shadow_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // s0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let linear_cube_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // s1..s2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [9] table: SSAO occlusion SRV at t4 (same convention as the legacy
    // main root sig; the same bindless fragment shader samples it).
    let ssao_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 4, // t4
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constant: per-draw object id at b0 (1 DWORD).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Root CBV: view UBO at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root CBV: light UBO at b2
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] Root CBV: shadow UBO at b3
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [4] Descriptor table: shadow map array (t0) + IBL cubes (t5..t6)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: shadow_srv_ranges.len() as u32,
                    pDescriptorRanges: shadow_srv_ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [5] Descriptor table: unbounded bindless texture pool (t0, space1)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &pool_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [6] Descriptor table: shadow comparison sampler (s0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &shadow_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [7] Descriptor table: linear repeat (s1) + cube sampler (s2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &linear_cube_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [8] Root SRV: per-frame StructuredBuffer<GpuObjectData> at t3
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [9] Descriptor table: SSAO occlusion SRV (t4)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ssao_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    serialize_and_create_root_sig(device, &params, "main bindless root sig")
}

// Same as the main root signature but with one extra root SRV at slot [8]
// (t3) carrying per-instance world matrices. Used by the GPU-instanced PSO
// and also the skinned PSO (whose root SRV at the same slot carries joint
// matrices instead).
pub(in crate::directx) fn create_main_instanced_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let shadow_srv_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0, // t0 shadow_map_array
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 2,
            BaseShaderRegister: 5, // t5..t6 IBL cubes
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
        },
    ];
    let object_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 2,
        BaseShaderRegister: 1,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let shadow_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let linear_cube_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 2,
        BaseShaderRegister: 1, // s1..s2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // [9] table: SSAO occlusion SRV at t4 (matches main + bindless layout).
    let ssao_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 4, // t4
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constants at b0 (same as main; model field is ignored by VS)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 28,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Root CBV: view UBO at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root CBV: light UBO at b2
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] Root CBV: shadow UBO at b3
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [4] Descriptor table: shadow array (t0) + IBL cubes (t5..t6)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: shadow_srv_ranges.len() as u32,
                    pDescriptorRanges: shadow_srv_ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [5] Descriptor table: albedo + normal SRVs (t1..t2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &object_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [6] Descriptor table: shadow comparison sampler (s0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &shadow_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [7] Descriptor table: linear repeat (s1) + cube sampler (s2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &linear_cube_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [8] Root SRV: per-instance world matrices (t3, VS-only)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [9] Descriptor table: SSAO occlusion SRV (t4)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ssao_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    serialize_and_create_root_sig(device, &params, "main instanced root sig")
}

pub(in crate::directx) fn create_shadow_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] Root constants: model mat4 (16) + cascade_idx + 3 pad = 20 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 20,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Root CBV: shadow UBO (light_vps[4] + cascade_splits) at b1
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
    ];

    serialize_and_create_root_sig(device, &params, "shadow root sig")
}

// Root signature for the GPU-driven shadow pass's depth-only bindless pipeline.
// Mirrors the bindless main root signature's object-id delivery so the shared
// cull command signature works against it: [0] is the per-command b0 object-id
// root constant (set by the `ExecuteIndirect` command signature, so it MUST stay
// at root parameter 0), [1] the shadow UBO CBV (light_vps), [2] a per-cascade b2
// cascade-index root constant (set once per cascade's `ExecuteIndirect`), and [3]
// the per-frame `StructuredBuffer<GpuObjectData>` root SRV the VS reads `model`
// from. All vertex-stage only (depth-only pass, no pixel shader).
pub(in crate::directx) fn create_shadow_bindless_root_signature(
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
        // [1] Root CBV b1: shadow UBO (light_vps[4] + cascade_splits).
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
        // [2] Root constant b2: cascade index (set per cascade's ExecuteIndirect).
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [3] Root SRV t0: per-frame StructuredBuffer<GpuObjectData>.
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
    ];

    serialize_and_create_root_sig(device, &params, "shadow bindless root sig")
}

// PSO builders

// PSO for the main (static + instanced + bindless) pass. The instanced
// pipeline reuses this with the appropriate VS + root sig.
pub(in crate::directx) fn create_main_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
) -> Result<ID3D12PipelineState, String> {
    let layout = main_input_layout();
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
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = rtv_format;
            a
        },
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            // Match Metal's default (no culling) so meshes with mixed winding
            // (e.g. procedural floor/ceiling planes) render from both sides.
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
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
                arr[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };

    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create main PSO: {e}"))
}

pub(in crate::directx) fn create_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let layout = main_input_layout();
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
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 0,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            // Match Metal: shadow pass also uses no culling so double-sided
            // procedural meshes cast shadows correctly.
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: 1,
            DepthBiasClamp: 0.01,
            SlopeScaledDepthBias: 1.0,
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
            ..Default::default()
        },
        ..Default::default()
    };

    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create shadow PSO: {e}"))
}

// Init-time orchestration

pub(super) struct MainPipelines {
    pub main_root_sig: ID3D12RootSignature,
    pub main_pso: ID3D12PipelineState,
    pub main_bindless_root_sig: Option<ID3D12RootSignature>,
    pub main_bindless_pso: Option<ID3D12PipelineState>,
    pub object_buffer_resources: Vec<ID3D12Resource>,
    pub object_buffer_ptrs: Vec<*mut u8>,
    pub cull_root_sig: Option<ID3D12RootSignature>,
    pub cull_pso: Option<ID3D12PipelineState>,
    // Phase-2 cull PSO for two-pass occlusion (`main_phase2` entry, same root
    // signature as `cull_pso`). `Some` only when the world requested
    // `occlusion_two_pass` AND the bindless cull path is active.
    pub cull_pso_phase2: Option<ID3D12PipelineState>,
    pub cull_command_signature: Option<ID3D12CommandSignature>,
    pub draw_args_buffer_resources: Vec<ID3D12Resource>,
    pub draw_args_buffer_ptrs: Vec<*mut u8>,
    pub indirect_cmd_buffers: Vec<ID3D12Resource>,
    // Per-frame per-object cull-status buffers (one u32 each). Phase-1 cull
    // writes drawn / hi-z-candidate / culled; phase-2 cull reads it. Always
    // allocated when the bindless cull path is active (mirrors Metal, where the
    // status buffer is always present and ignored under single-pass).
    pub cull_status_buffers: Vec<ID3D12Resource>,
    // Per-frame second indirect-command buffers the phase-2 cull writes and
    // `Main2` consumes. `Some`/non-empty only under two-pass occlusion.
    pub indirect_cmd_buffers_2: Vec<ID3D12Resource>,
    // GPU-driven shadow pass. Depth-only bindless pipeline + the
    // shared cull command signature rebuilt against its root sig + per-frame
    // indirect buffers (one region per cascade) + a scratch cull-status buffer.
    // All `Some`/non-empty only when the bindless cull path is active AND shadows
    // are enabled.
    pub shadow_bindless_root_sig: Option<ID3D12RootSignature>,
    pub shadow_bindless_pso: Option<ID3D12PipelineState>,
    pub shadow_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    // Frustum-only shadow cull PSO (`main_shadow` entry, shares the cull root sig).
    pub cull_pso_shadow: Option<ID3D12PipelineState>,
    pub shadow_indirect_buffers: Vec<ID3D12Resource>,
    pub shadow_cull_status_buffers: Vec<ID3D12Resource>,
    // GPU-driven G-buffer pre-pass. A 3-MRT bindless pipeline + the
    // shared cull command signature rebuilt against its root sig + per-frame
    // previous-frame model upload buffers. All `Some`/non-empty only when the
    // bindless cull path is active AND the G-buffer is enabled.
    pub gbuffer_bindless_root_sig: Option<ID3D12RootSignature>,
    pub gbuffer_bindless_pso: Option<ID3D12PipelineState>,
    pub gbuffer_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    pub prev_model_buffer_resources: Vec<ID3D12Resource>,
    pub prev_model_buffer_ptrs: Vec<*mut u8>,
}

// Build the main static pass + (when the world ships no custom main shader)
// the bindless variant and GPU-cull compute pipeline. Allocates the per-frame
// `StructuredBuffer<GpuObjectData>` / `StructuredBuffer<GpuDrawArgs>` upload
// buffers and the per-frame indirect-command UAV buffers that the cull kernel
// writes into.
//
// The bindless + cull infrastructure is only built when
// `vert_bytes`+`frag_bytes` are empty (built-in shader path) AND
// `n_objects > 0`. Otherwise the corresponding fields are `None` / empty.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_main_pipelines(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    shaders: &CompiledShaders,
    vert_bytes: &[u8],
    frag_bytes: &[u8],
    msaa_samples: u32,
    n_objects: usize,
    // Total instanced-cluster instances folded into the GPU-driven bindless pass.
    // The cull / object / draw-args / indirect buffers are sized for
    // `n_objects + n_instances` so each instance gets a record at `n_objects + k`;
    // the cull dispatch + `ExecuteIndirect` then count the merged total.
    n_instances: usize,
    // Skinned draw objects folded in after the instances (records at
    // `n_objects + n_instances + k`, drawn by the main pass's 2nd `ExecuteIndirect`
    // against the per-frame deformed-vertex buffer). Sizes the shared buffers'
    // reserved skinned tail; the records are written per frame in
    // `build_object_buffer` / `build_draw_args_buffer` once skinned geometry is
    // resident.
    n_skinned: usize,
    // Worst-case resident chunk count for a streaming VoxelWorld (0 otherwise).
    // Reserves a chunk record region BETWEEN the instances and the skinned tail
    // (`[n_objects + n_instances, +n_chunk_max)`). Chunk geometry already lives in
    // the shared VB/IB, so resident chunks fold into this region each frame and are
    // drawn by the static+instance prefix `ExecuteIndirect`; the skinned tail base
    // shifts past the reserve. Sizes the cull buffers' merged total.
    n_chunk_max: usize,
    // `PostProcessConfig.occlusion_two_pass`. When set (and the bindless cull
    // path is active), build the phase-2 cull PSO + the second indirect buffers
    // that drive two-pass Hi-Z occlusion.
    occlusion_two_pass: bool,
    // When the world has shadows enabled (shadow map size > 0) AND the bindless
    // cull path is active, build the GPU-driven shadow pass's depth-only
    // pipeline + per-cascade indirect buffers. A shadow-disabled world skips
    // them (they would never be issued).
    shadow_enabled: bool,
    // When the G-buffer pre-pass is enabled (any screen-space consumer drives it)
    // AND the bindless cull path is active, build the GPU-driven G-buffer pre-pass
    // pipeline + the per-frame previous-frame model buffers it reads for velocity.
    // A world with no G-buffer skips them (the pre-pass never runs).
    gbuffer_enabled: bool,
    hot_reload: bool,
) -> Result<MainPipelines, String> {
    // Merged record count: static build-time objects, the instanced-cluster
    // instances, the streamed-chunk reserve, then the skinned objects. The
    // per-frame static fills write only the first `n_objects`; the instance records
    // are written once at init; chunk records + skinned records are written each
    // frame into their reserved regions.
    let n_cull = n_objects + n_instances + n_chunk_max + n_skinned;
    let main_root_sig = dump_on_err(info_queue, create_main_root_signature(device))?;
    let main_pso = dump_on_err(
        info_queue,
        create_main_pso(
            device,
            &main_root_sig,
            &shaders.main_vs,
            &shaders.main_ps,
            HDR_FORMAT,
            msaa_samples,
        ),
    )?;

    // Bindless static main pass (bindless static pass). Built only when no custom
    // main shader was supplied; a world with its own shader keeps the legacy
    // per-draw pipeline.
    let main_is_builtin = vert_bytes.is_empty() && frag_bytes.is_empty();
    let (main_bindless_root_sig, main_bindless_pso) = if main_is_builtin {
        let (bvs, bps) = compile_main_bindless_shaders(hot_reload)?;
        let brs = dump_on_err(info_queue, create_main_bindless_root_signature(device))?;
        let bpso = dump_on_err(
            info_queue,
            create_main_pso(device, &brs, &bvs, &bps, HDR_FORMAT, msaa_samples),
        )?;
        (Some(brs), Some(bpso))
    } else {
        (None, None)
    };

    // Per-frame StructuredBuffer<GpuObjectData> upload buffers. Allocated only
    // when the bindless pass is active and the world has build-time static
    // geometry; rebuilt each frame in `build_object_buffer`.
    let mut object_buffer_resources: Vec<ID3D12Resource> = Vec::new();
    let mut object_buffer_ptrs: Vec<*mut u8> = Vec::new();
    if main_bindless_pso.is_some() && n_cull > 0 {
        let object_buffer_size = align256(
            (n_cull * std::mem::size_of::<crate::gfx::render_types::GpuObjectData>()) as u64,
        );
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                object_buffer_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map object buffer: {e}"))?;
            object_buffer_ptrs.push(ptr as *mut u8);
            object_buffer_resources.push(buf);
        }
    }

    // Compute cull: cull compute pipeline + per-frame draw-args /
    // indirect-command buffers. Built under the same condition as the object
    // buffer.
    let mut cull_root_sig: Option<ID3D12RootSignature> = None;
    let mut cull_pso: Option<ID3D12PipelineState> = None;
    let mut cull_pso_phase2: Option<ID3D12PipelineState> = None;
    let mut cull_command_signature: Option<ID3D12CommandSignature> = None;
    let mut draw_args_buffer_resources: Vec<ID3D12Resource> = Vec::new();
    let mut draw_args_buffer_ptrs: Vec<*mut u8> = Vec::new();
    let mut indirect_cmd_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut cull_status_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut indirect_cmd_buffers_2: Vec<ID3D12Resource> = Vec::new();
    let mut shadow_bindless_root_sig: Option<ID3D12RootSignature> = None;
    let mut shadow_bindless_pso: Option<ID3D12PipelineState> = None;
    let mut shadow_bindless_cmd_sig: Option<ID3D12CommandSignature> = None;
    let mut cull_pso_shadow: Option<ID3D12PipelineState> = None;
    let mut shadow_indirect_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut shadow_cull_status_buffers: Vec<ID3D12Resource> = Vec::new();
    let mut gbuffer_bindless_root_sig: Option<ID3D12RootSignature> = None;
    let mut gbuffer_bindless_pso: Option<ID3D12PipelineState> = None;
    let mut gbuffer_bindless_cmd_sig: Option<ID3D12CommandSignature> = None;
    let mut prev_model_buffer_resources: Vec<ID3D12Resource> = Vec::new();
    let mut prev_model_buffer_ptrs: Vec<*mut u8> = Vec::new();
    if let (Some(bindless_root), true) = (
        main_bindless_root_sig.as_ref(),
        main_bindless_pso.is_some() && n_cull > 0,
    ) {
        let cs = compile_cull_shader(hot_reload)?;
        let crs = dump_on_err(info_queue, create_cull_root_signature(device))?;
        let cps = dump_on_err(info_queue, create_cull_pso(device, &crs, &cs))?;
        let csig = dump_on_err(
            info_queue,
            create_cull_command_signature(device, bindless_root),
        )?;
        // Phase-2 cull PSO for two-pass occlusion (same root sig, `main_phase2`
        // entry). Built only when the world opted in.
        if occlusion_two_pass {
            let cs2 = compile_cull_shader_phase2(hot_reload)?;
            cull_pso_phase2 = Some(dump_on_err(
                info_queue,
                create_cull_pso(device, &crs, &cs2),
            )?);
        }

        let draw_args_size = align256(
            (n_cull * std::mem::size_of::<crate::gfx::render_types::GpuDrawArgs>()) as u64,
        );
        // Default-heap indirect-command buffers (UAV target for the cull
        // kernel; ExecuteIndirect source for the bindless static pass).
        let indirect_size = align256((n_cull as u64) * INDIRECT_COMMAND_STRIDE as u64);
        // Per-object cull-status buffer (one u32 each). Always allocated when
        // the cull path is active (matches Metal); resting state `UAV` so it
        // binds as a root UAV with no transition.
        let status_size = align256((n_cull as u64) * std::mem::size_of::<u32>() as u64);
        for _ in 0..FRAMES {
            let da = create_buffer(
                device,
                draw_args_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { da.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map draw args buffer: {e}"))?;
            draw_args_buffer_ptrs.push(ptr as *mut u8);
            draw_args_buffer_resources.push(da);

            // Created in COMMON (D3D12 always makes committed buffers in COMMON
            // regardless of the requested state); the cull pass transitions them
            // to UNORDERED_ACCESS / INDIRECT_ARGUMENT as it writes + executes them.
            indirect_cmd_buffers.push(create_uav_buffer(
                device,
                indirect_size,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            cull_status_buffers.push(create_uav_buffer(
                device,
                status_size,
                D3D12_RESOURCE_STATE_COMMON,
            )?);
            // Second indirect buffer for the phase-2 (disocclusion) draws.
            // Only allocated under two-pass occlusion.
            if occlusion_two_pass {
                indirect_cmd_buffers_2.push(create_uav_buffer(
                    device,
                    indirect_size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
            }
        }
        // GPU-driven shadow pass: a depth-only bindless pipeline + the shared
        // cull command signature rebuilt against its root sig (object id still
        // at root param 0) + per-frame indirect buffers carrying one cull region
        // per cascade (`NUM_SHADOW_CASCADES * n_cull` commands) + a scratch
        // cull-status buffer the shadow cull dispatches write but never read.
        if shadow_enabled {
            let svs = compile_shadow_bindless_vs(hot_reload)?;
            let sbrs = dump_on_err(info_queue, create_shadow_bindless_root_signature(device))?;
            // Reuse the depth-only shadow PSO builder (no pixel shader, 0 RTVs,
            // D32 DSV, slope-scaled depth bias, main vertex layout).
            let sbpso = dump_on_err(info_queue, create_shadow_pso(device, &sbrs, &svs))?;
            let sbsig = dump_on_err(info_queue, create_cull_command_signature(device, &sbrs))?;
            // Frustum-only shadow cull kernel (`main_shadow`), shares the cull root sig.
            let scs = compile_cull_shader_shadow(hot_reload)?;
            cull_pso_shadow = Some(dump_on_err(
                info_queue,
                create_cull_pso(device, &crs, &scs),
            )?);
            let cascades = crate::gfx::render_types::NUM_SHADOW_CASCADES as u64;
            let shadow_indirect_size =
                align256(cascades * (n_cull as u64) * INDIRECT_COMMAND_STRIDE as u64);
            for _ in 0..FRAMES {
                shadow_indirect_buffers.push(create_uav_buffer(
                    device,
                    shadow_indirect_size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
                shadow_cull_status_buffers.push(create_uav_buffer(
                    device,
                    status_size,
                    D3D12_RESOURCE_STATE_COMMON,
                )?);
            }
            shadow_bindless_root_sig = Some(sbrs);
            shadow_bindless_pso = Some(sbpso);
            shadow_bindless_cmd_sig = Some(sbsig);
        }

        // GPU-driven G-buffer pre-pass: a 3-MRT bindless pipeline whose VS reads
        // model + roughness from `GpuObjectData[object_id]` + the previous-frame
        // model from a parallel buffer, drawn by reusing the main pass's per-frame
        // indirect command buffer (NO new cull -- the camera-frustum cull already
        // ran). Plus the per-frame `prev_model` upload buffers (one column-major
        // `float4x4` per cull record): the instance region is init-written, the
        // static + skinned regions rewritten each frame.
        if gbuffer_enabled {
            let (grs, gpso, gsig) = crate::directx::post::gbuffer::build_gbuffer_bindless(
                device, info_queue, hot_reload,
            )?;
            let prev_model_size = align256((n_cull * std::mem::size_of::<[[f32; 4]; 4]>()) as u64);
            for _ in 0..FRAMES {
                let buf = create_buffer(
                    device,
                    prev_model_size,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                unsafe { buf.Map(0, None, Some(&mut ptr)) }
                    .map_err(|e| format!("map prev_model buffer: {e}"))?;
                prev_model_buffer_ptrs.push(ptr as *mut u8);
                prev_model_buffer_resources.push(buf);
            }
            gbuffer_bindless_root_sig = Some(grs);
            gbuffer_bindless_pso = Some(gpso);
            gbuffer_bindless_cmd_sig = Some(gsig);
        }

        cull_root_sig = Some(crs);
        cull_pso = Some(cps);
        cull_command_signature = Some(csig);
    }

    Ok(MainPipelines {
        main_root_sig,
        main_pso,
        main_bindless_root_sig,
        main_bindless_pso,
        object_buffer_resources,
        object_buffer_ptrs,
        cull_root_sig,
        cull_pso,
        cull_pso_phase2,
        cull_command_signature,
        draw_args_buffer_resources,
        draw_args_buffer_ptrs,
        indirect_cmd_buffers,
        cull_status_buffers,
        indirect_cmd_buffers_2,
        shadow_bindless_root_sig,
        shadow_bindless_pso,
        shadow_bindless_cmd_sig,
        cull_pso_shadow,
        shadow_indirect_buffers,
        shadow_cull_status_buffers,
        gbuffer_bindless_root_sig,
        gbuffer_bindless_pso,
        gbuffer_bindless_cmd_sig,
        prev_model_buffer_resources,
        prev_model_buffer_ptrs,
    })
}

pub(super) fn build_shadow_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    shadow_vs: Option<&[u8]>,
) -> Result<(Option<ID3D12RootSignature>, Option<ID3D12PipelineState>), String> {
    if let Some(svs) = shadow_vs {
        let sr = dump_on_err(info_queue, create_shadow_root_signature(device))?;
        let sp = dump_on_err(info_queue, create_shadow_pso(device, &sr, svs))?;
        Ok((Some(sr), Some(sp)))
    } else {
        Ok((None, None))
    }
}

pub(super) fn build_main_instanced_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    instanced_vs: Option<&[u8]>,
    main_ps: &[u8],
    msaa_samples: u32,
) -> Result<(Option<ID3D12RootSignature>, Option<ID3D12PipelineState>), String> {
    if let Some(ivs) = instanced_vs {
        let irs = dump_on_err(info_queue, create_main_instanced_root_signature(device))?;
        let ips = dump_on_err(
            info_queue,
            create_main_pso(device, &irs, ivs, main_ps, HDR_FORMAT, msaa_samples),
        )?;
        Ok((Some(irs), Some(ips)))
    } else {
        Ok((None, None))
    }
}

pub(super) fn build_text_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    text_vs: &[u8],
    text_ps: &[u8],
    swap_format: DXGI_FORMAT,
    has_atlases: bool,
) -> Result<(ID3D12RootSignature, Option<ID3D12PipelineState>), String> {
    let text_root_sig = dump_on_err(info_queue, create_text_root_signature(device))?;
    // Text renders in the composite pass into the single-sample swapchain
    // backbuffer (post-tonemap), so its PSO targets the swapchain format at
    // sample count 1.
    let text_pso = if has_atlases {
        Some(dump_on_err(
            info_queue,
            create_text_pso(device, &text_root_sig, text_vs, text_ps, swap_format, 1),
        )?)
    } else {
        None
    };
    Ok((text_root_sig, text_pso))
}

pub(super) fn build_composite_pipeline(
    device: &ID3D12Device,
    info_queue: Option<&ID3D12InfoQueue>,
    swap_format: DXGI_FORMAT,
    hot_reload: bool,
) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
    let composite_root_sig = dump_on_err(info_queue, create_composite_root_signature(device))?;
    let (composite_vs, composite_ps) = compile_composite_shaders(hot_reload)?;
    let composite_pso = dump_on_err(
        info_queue,
        create_composite_pso(
            device,
            &composite_root_sig,
            &composite_vs,
            &composite_ps,
            swap_format,
        ),
    )?;
    Ok((composite_root_sig, composite_pso))
}
