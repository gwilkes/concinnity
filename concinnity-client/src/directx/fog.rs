// src/directx/fog.rs
//
// Volumetric fog for the D3D12 backend. Frostbite-style froxel volume:
//
//   * The `fog_froxel_kernel` compute pass (`encode_fog_froxel`) populates a
//     screen-aligned `(80 × 45 × 64)` 3D `RGBA16Float` volume of
//     `(scattered_rgb, 1 - T)` across the view frustum. One thread per
//     (x, y) tile; each thread walks Z front-to-back, accumulating the
//     per-slab scatter + transmittance with a CSM shadow tap per slice.
//
//   * The fullscreen `Fog` render pass (`encode_fog`) samples the volume by
//     `(screen_uv, view_z)` instead of marching per-pixel and composites
//     `(scattered, 1 - T)` over the resolved HDR target with the standard
//     `over` blend (`final = scene * T + scattered`).
//
// Runs between the projected-decal pass and the SSR resolve so the fog wraps
// the decal-stamped scene and SSR reflects through it; TAA history then
// reprojects the integrated fog colour and transmittance.
//
// Mirrors src/metal/fog.rs.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::{HDR_FORMAT, create_buffer, transition_barrier};
use crate::gfx::render_graph::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z};
use crate::gfx::render_types::{FogFroxelParams, FogParams};

// HLSL sources

pub const FOG_VERT_HLSL: &str = include_str!("shaders/fog_vert.hlsl");
pub const FOG_FRAG_HLSL: &str = include_str!("shaders/fog_frag.hlsl");
pub const FOG_FROXEL_HLSL: &str = include_str!("shaders/fog_froxel.hlsl");

// Compile the fog vertex + fragment shaders, prepending the MSAA define so
// the depth SRV declaration in the fragment shader matches the resource's
// sample count. Used by [`FogResources::new`] at init and by shader hot-
// reload to rebuild the fog PSO.
pub(in crate::directx) fn compile_fog_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let define_line = if msaa_samples > 1 {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    let vs_body = shader_source(hot_reload, "fog_vert.hlsl", FOG_VERT_HLSL);
    let ps_body = shader_source(hot_reload, "fog_frag.hlsl", FOG_FRAG_HLSL);
    let vs_src = format!("{define_line}{vs_body}");
    let ps_src = format!("{define_line}{ps_body}");
    let vs = compile_hlsl(&vs_src, "main", "vs_5_1")?;
    let ps = compile_hlsl(&ps_src, "main", "ps_5_1")?;
    Ok((vs, ps))
}

// Compile the froxel-volume compute kernel.
pub(in crate::directx) fn compile_fog_froxel_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    let src = shader_source(hot_reload, "fog_froxel.hlsl", FOG_FROXEL_HLSL);
    compile_hlsl(&src, "main", "cs_5_1")
}

// Rebuild the fog PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; reuses the existing root signature.
pub(in crate::directx) fn rebuild_fog_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = compile_fog_shaders(msaa_samples, hot_reload)?;
    dump_on_err(info_queue, create_fog_pso(device, root_sig, &vs, &ps))
}

// Rebuild the froxel compute PSO against fresh shader source.
pub(in crate::directx) fn rebuild_fog_froxel_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let cs = compile_fog_froxel_shader(hot_reload)?;
    dump_on_err(info_queue, create_fog_froxel_pso(device, root_sig, &cs))
}

// Fog render-pass root signature:
//   [0] root CBV b0   FogParams         (per-frame)
//   [1] root CBV b1   FogFroxelParams   (per-frame)
//   [2] table  t0     scene depth SRV (Texture2D[MS]<float>)
//   [3] table  t1     froxel volume SRV (Texture3D<float4>)
// Static linear-clamp sampler s0 for the trilinear volume sample.
fn create_fog_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let depth_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let volume_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1, // t1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &depth_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &volume_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let volume_sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &volume_sampler,
        // The fullscreen pass uses SV_VertexID; no input assembler is needed.
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "fog root sig")
}

// Froxel compute root signature:
//   [0] root CBV b0   FogParams         (per-frame)
//   [1] root CBV b1   FogFroxelParams   (per-frame)
//   [2] root CBV b2   ShadowUniforms    (per-frame, shared with Main / Shadow)
//   [3] table  t0     shadow map SRV (Texture2DArray<float>)
//   [4] table  u0     froxel volume UAV (RWTexture3D<float4>)
// Static comparison sampler s0 for the shadow tap.
fn create_fog_froxel_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let shadow_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let volume_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // u0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
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
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &shadow_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &volume_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    // Comparison sampler matching the existing `shadow_sampler_gpu` static
    // sampler. Clamp on every axis so cascades that fail their NDC bounds
    // check fall back to 1.0 via the explicit `if (any(uv < 0.0)...)` guard
    // in the kernel anyway.
    let shadow_sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
        MaxLOD: f32::MAX,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &shadow_sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "fog froxel root sig")
}

// PSO for the fog pass. Writes the resolved HDR target with `(scattered,
// 1 - T)` over scene blending: the fragment emits the in-scattered colour
// at `1 - transmittance` alpha and the blend resolves to
// `scene * T + scattered`. No depth attachment; the shader handles the
// depth-based ray-length termination itself.
fn create_fog_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
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
        // No input layout; the fullscreen triangle is emitted by SV_VertexID.
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = HDR_FORMAT;
            a
        },
        DSVFormat: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthClipEnable: false.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: {
                let mut arr = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
                arr[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: true.into(),
                    SrcBlend: D3D12_BLEND_ONE,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ONE,
                    DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
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
        .map_err(|e| format!("create fog PSO: {e}"))
}

// Compute PSO for the froxel kernel.
fn create_fog_froxel_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs.as_ptr() as _,
            BytecodeLength: cs.len(),
        },
        ..Default::default()
    };
    unsafe { device.CreateComputePipelineState(&desc) }
        .map_err(|e| format!("create fog froxel PSO: {e}"))
}

// Create the 3D `RGBA16Float` froxel volume. Rests in `PIXEL_SHADER_RESOURCE`
// between frames: the graph's FogFroxel producer barrier transitions it to
// `UNORDERED_ACCESS` for the compute write, and the Fog consumer barrier returns
// it to `PIXEL_SHADER_RESOURCE` for the trilinear sample. Both transitions are
// graph-driven (no inline cross-frame reset); creating it sampled makes frame 0's
// producer barrier (sampled -> UAV) start from the resource's real state.
fn create_fog_froxel_volume(
    device: &ID3D12Device,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: FOG_FROXEL_X as u64,
        Height: FOG_FROXEL_Y,
        DepthOrArraySize: FOG_FROXEL_Z as u16,
        MipLevels: 1,
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            // Rest sampled; the graph drives both transitions (see the fn doc).
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create fog froxel volume: {e}"))?;
    let resource = tex_opt.ok_or_else(|| "create fog froxel volume returned None".to_string())?;

    let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_UAV {
                MipSlice: 0,
                FirstWSlice: 0,
                WSize: FOG_FROXEL_Z,
            },
        },
    };
    unsafe {
        device.CreateUnorderedAccessView(&resource, None, Some(&uav_desc), uav_cpu);
    }

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(&resource, Some(&srv_desc), srv_cpu) };

    Ok(resource)
}

// Owned by `DxContext` exactly when the world declared a `VolumetricFog`:
// the fog pipeline + per-frame uniform rings + the froxel volume + the
// kernel that populates it. The depth SRV the fog pass reads through t0
// is shared with the projected-decal pass.
pub(in crate::directx) struct FogResources {
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    pub(in crate::directx) pso: ID3D12PipelineState,

    pub(in crate::directx) froxel_root_sig: ID3D12RootSignature,
    pub(in crate::directx) froxel_pso: ID3D12PipelineState,

    // Per-frame FogParams ring (176-byte block, persistently mapped).
    pub(in crate::directx) params_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) params_ubo_ptrs: Vec<*mut u8>,

    // Per-frame FogFroxelParams ring (96-byte block, persistently mapped).
    pub(in crate::directx) froxel_params_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) froxel_params_ubo_ptrs: Vec<*mut u8>,

    // 3D `RGBA16Float` volume the kernel writes and the fragment shader
    // samples. The handle backs the graph-driven UAV ↔ PIXEL_SHADER_RESOURCE
    // producer + consumer barriers (resolved by the executor's barrier
    // registry); the shader reads/writes go through the heap-stored UAV + SRV
    // descriptors.
    pub(in crate::directx) volume_resource: ID3D12Resource,
    pub(in crate::directx) volume_uav_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) volume_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Heap GPU handle of the main-depth SRV. Bound at fog pass t0; the
    // resource is transitioned to PIXEL_SHADER_RESOURCE around the pass and
    // restored to DEPTH_WRITE afterward.
    pub(in crate::directx) depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Heap GPU handle of the shadow map array SRV. Bound at the froxel
    // kernel's t0 so each slab can do a CSM tap. Shared with the rest of
    // the engine; the resource is transitioned to PIXEL_SHADER_RESOURCE
    // by `encode_shadow_pass` ahead of every later pass, including this
    // one, and restored to DEPTH_WRITE at the end of `record_frame`.
    pub(in crate::directx) shadow_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

impl FogResources {
    // Build the fog pipeline + per-frame uniform rings + the froxel volume
    // + the compute kernel. Called from `DxContext::new` only when the
    // world declared a `VolumetricFog`. The depth SRV is already written
    // into the heap by the decal-init path (the projected-decal pass
    // writes the main-depth SRV unconditionally so runtime `add_decal`
    // works from a world that started empty); the fog pass reuses the
    // same descriptor. `volume_uav_cpu` / `volume_srv_cpu` are dedicated
    // SRV-heap slots reserved by init for the froxel volume.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        msaa_samples: u32,
        depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        shadow_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        volume_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        volume_uav_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        volume_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        volume_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let (vs, ps) = compile_fog_shaders(msaa_samples, hot_reload)?;
        let cs = compile_fog_froxel_shader(hot_reload)?;

        let root_sig = dump_on_err(info_queue, create_fog_root_signature(device))?;
        let pso = dump_on_err(info_queue, create_fog_pso(device, &root_sig, &vs, &ps))?;

        let froxel_root_sig = dump_on_err(info_queue, create_fog_froxel_root_signature(device))?;
        let froxel_pso = dump_on_err(
            info_queue,
            create_fog_froxel_pso(device, &froxel_root_sig, &cs),
        )?;

        let volume_resource = create_fog_froxel_volume(device, volume_uav_cpu, volume_srv_cpu)?;

        // Per-frame FogParams ring.
        let params_ubo_size = align256(std::mem::size_of::<FogParams>() as u64);
        let mut params_ubo_resources: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                params_ubo_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map fog params ubo: {e}"))?;
            params_ubo_ptrs.push(ptr as *mut u8);
            params_ubo_resources.push(buf);
        }

        // Per-frame FogFroxelParams ring.
        let froxel_ubo_size = align256(std::mem::size_of::<FogFroxelParams>() as u64);
        let mut froxel_params_ubo_resources: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut froxel_params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                froxel_ubo_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map fog froxel params ubo: {e}"))?;
            froxel_params_ubo_ptrs.push(ptr as *mut u8);
            froxel_params_ubo_resources.push(buf);
        }

        Ok(Self {
            root_sig,
            pso,
            froxel_root_sig,
            froxel_pso,
            params_ubo_resources,
            params_ubo_ptrs,
            froxel_params_ubo_resources,
            froxel_params_ubo_ptrs,
            volume_resource,
            volume_uav_gpu,
            volume_srv_gpu,
            depth_srv_gpu,
            shadow_srv_gpu,
        })
    }
}

impl DxContext {
    // Hot-reload entry point for the volumetric-fog tunables. Writes the new
    // `Option<FogSettings>` into `self.fog.settings`; the next frame's
    // `encode_fog_froxel` + `encode_fog` re-derive `FogParams` /
    // `FogFroxelParams` from it (both bail to a no-op when it is `None`, so a
    // `None` here disables the pass). Mirrors `MtlContext::update_fog_settings`.
    //
    // If the world started with no `VolumetricFog` (so `self.fog.resources` is
    // `None` and the froxel + fog PSOs were never built), a `Some` update logs
    // once and is dropped: re-enabling fog mid-run requires a relaunch.
    //
    // `#[allow(dead_code)]` because the only caller is the bin-only `cn debug`
    // world hot-reload (`debug::hot_reload::passes`), reached through the
    // `RenderBackend` vtable; the FFI lib build sees no caller. Mirrors the
    // other bin-only runtime-mutation seams on this backend.
    #[allow(dead_code)]
    pub fn update_fog_settings(
        &mut self,
        settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    ) {
        if settings.is_some() && self.fog.resources.is_none() {
            tracing::warn!(
                "VolumetricFog hot-reload: world started without fog, so the fog \
                 pipeline was never built: re-enabling fog mid-run is not \
                 supported (relaunch required). Ignoring update."
            );
            return;
        }
        self.fog.settings = settings;
    }

    // Compute the per-frame `FogFroxelParams` block. Mirrors the Metal
    // `draw::mod::record_frame` block: view matrix + volume dimensions +
    // near/far. `near` is the camera near-plane (clamped to ≥ 1e-3 so the
    // linear-Z mapping stays finite), and `z_far` is the fog's authored
    // `max_distance` (the volume covers `[z_near, max_distance]`).
    fn fog_froxel_params(&self, near: f32) -> Option<FogFroxelParams> {
        let fog = self.fog.settings?;
        Some(FogFroxelParams {
            view: self.view_matrix,
            froxel_dims: [FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z],
            _pad_align: 0,
            z_near: near.max(1e-3),
            z_far: fog.max_distance,
            _pad: [0.0; 2],
        })
    }

    // Encode the volumetric-fog froxel-volume compute pass. Populates the
    // 3D `(scattered, 1 - T)` volume the fog fragment shader samples. The
    // caller (`execute_graph`) seeds this PassId before `Fog` so the RAW
    // edge in the shared graph orders the dispatch correctly.
    pub(in crate::directx) fn encode_fog_froxel(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        near: f32,
        vp: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        shadow_ubo_gva: u64,
    ) {
        let fog_settings = match &self.fog.settings {
            Some(s) => *s,
            None => return,
        };
        let fog = match &self.fog.resources {
            Some(f) => f,
            None => return,
        };
        let froxel_params = match self.fog_froxel_params(near) {
            Some(p) => p,
            None => return,
        };

        // Write per-frame `FogParams` + `FogFroxelParams` into their ring
        // slots. The `Fog` render pass below reads from the same slot, so
        // both passes see the same params this frame.
        let inv_vp = super::math::mat4_inverse(vp);
        let viewport = [self.render_width as f32, self.render_height as f32];
        let params = fog_settings.params(
            inv_vp,
            cam_pos,
            self.fog.sun_dir,
            self.fog.sun_color,
            viewport,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const FogParams as *const u8,
                fog.params_ubo_ptrs[frame_idx],
                std::mem::size_of::<FogParams>(),
            );
            std::ptr::copy_nonoverlapping(
                &froxel_params as *const FogFroxelParams as *const u8,
                fog.froxel_params_ubo_ptrs[frame_idx],
                std::mem::size_of::<FogFroxelParams>(),
            );
        }
        let params_gva = unsafe { fog.params_ubo_resources[frame_idx].GetGPUVirtualAddress() };
        let froxel_params_gva =
            unsafe { fog.froxel_params_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Shadow map needs to flip from PIXEL_SHADER_RESOURCE (where every
        // earlier pass left it (either `encode_shadow_pass` for the real
        // path or the init upload for the 1×1 fallback) to
        // NON_PIXEL_SHADER_RESOURCE so the compute kernel can sample it.
        // The volume stays in `UNORDERED_ACCESS`; `encode_fog` is what
        // transitions it to PIXEL_SHADER_RESOURCE for the render pass.
        let shadow_to_compute = self.shadow.resource.as_ref().map(|s| {
            transition_barrier(
                &s.resource,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )
        });
        if let Some(b) = shadow_to_compute {
            unsafe { cmd.ResourceBarrier(&[b]) };
        }

        unsafe {
            cmd.SetComputeRootSignature(&fog.froxel_root_sig);
            cmd.SetPipelineState(&fog.froxel_pso);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRootConstantBufferView(0, params_gva);
            cmd.SetComputeRootConstantBufferView(1, froxel_params_gva);
            cmd.SetComputeRootConstantBufferView(2, shadow_ubo_gva);
            cmd.SetComputeRootDescriptorTable(3, fog.shadow_srv_gpu);
            cmd.SetComputeRootDescriptorTable(4, fog.volume_uav_gpu);

            // 8×8 threadgroups, one thread per (x, y) froxel.
            let groups_x = FOG_FROXEL_X.div_ceil(8);
            let groups_y = FOG_FROXEL_Y.div_ceil(8);
            cmd.Dispatch(groups_x, groups_y, 1);
        }

        // Restore the shadow map to PIXEL_SHADER_RESOURCE so the end-of-
        // frame transition back to DEPTH_WRITE (or the fallback's permanent
        // PIXEL state) finds it where it expects.
        let shadow_back = self.shadow.resource.as_ref().map(|s| {
            transition_barrier(
                &s.resource,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            )
        });
        if let Some(b) = shadow_back {
            unsafe { cmd.ResourceBarrier(&[b]) };
        }
    }

    // Encode the volumetric-fog pass. Samples the 3D froxel volume the
    // `FogFroxel` compute pass populated this frame. Caller has already
    // ended the main HDR pass + the projected-decal pass (if any), so
    // `depth_resource` (MSAA when MSAA is on) holds the scene depth and
    // the resolved scene target holds the resolved scene + decal colour.
    // The pass alpha-blends `(scattered, 1 - T)` over the resolved HDR
    // target.
    pub(in crate::directx) fn encode_fog(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        _vp: [[f32; 4]; 4],
        _cam_pos: [f32; 3],
    ) {
        let _ = match &self.fog.settings {
            Some(s) => *s,
            None => return,
        };
        let fog = match &self.fog.resources {
            Some(f) => f,
            None => return,
        };

        // `FogParams` / `FogFroxelParams` were uploaded by `encode_fog_froxel`
        // for this frame's slot, so we only read their GVAs here.
        let params_gva = unsafe { fog.params_ubo_resources[frame_idx].GetGPUVirtualAddress() };
        let froxel_params_gva =
            unsafe { fog.froxel_params_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Transition main depth → PIXEL_SHADER_RESOURCE so the fragment can
        // sample it; restored to DEPTH_WRITE after the pass.
        let depth_to_psr = transition_barrier(
            &self.depth_resource,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        // The froxel volume's transitions are fully graph-driven:
        // fog_froxel_volume is the graph's resource, so the FogFroxel producer
        // barrier (PIXEL_SHADER_RESOURCE → UNORDERED_ACCESS) runs before that
        // pass and this Fog pass's consumer barrier (UNORDERED_ACCESS →
        // PIXEL_SHADER_RESOURCE) before this pass. There is no inline reset.
        unsafe { cmd.ResourceBarrier(&[depth_to_psr]) };

        // hdr_resolve / hdr_color is in PIXEL_SHADER_RESOURCE after the main
        // pass (decals, if they ran, restored it to PIXEL_SHADER_RESOURCE).
        // Flip it back to RENDER_TARGET so we can blend into it.
        let scene_rtv: D3D12_CPU_DESCRIPTOR_HANDLE = if let Some(hdr_resolve) = &self.hdr.resolve {
            let resolve_to_rt = transition_barrier(
                hdr_resolve,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { cmd.ResourceBarrier(&[resolve_to_rt]) };
            self.hdr
                .resolve_rtv
                .expect("hdr_resolve_rtv set when hdr_resolve is Some")
        } else {
            let to_rt = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { cmd.ResourceBarrier(&[to_rt]) };
            self.hdr.color_rtv
        };

        let w = self.render_width;
        let h = self.render_height;
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&scene_rtv), false, None);
            let vp_state = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: w as f32,
                Height: h as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp_state]);
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

            cmd.SetPipelineState(&fog.pso);
            cmd.SetGraphicsRootSignature(&fog.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, params_gva);
            cmd.SetGraphicsRootConstantBufferView(1, froxel_params_gva);
            cmd.SetGraphicsRootDescriptorTable(2, fog.depth_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(3, fog.volume_srv_gpu);
            cmd.DrawInstanced(3, 1, 0, 0);
        }

        // Restore: scene target back to PIXEL_SHADER_RESOURCE so the SSR
        // resolve / TAA / bloom / composite can sample it; main depth back to
        // DEPTH_WRITE for next frame's main pass; volume back to UNORDERED_ACCESS
        // for next frame's compute pass. This volume reset is the graph seam's
        // kept inline restore: it returns fog_froxel_volume to its resting state
        // so the executor's FogFroxel producer barrier stays a no-op.
        if let Some(hdr_resolve) = &self.hdr.resolve {
            let rt_to_psr = transition_barrier(
                hdr_resolve,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            unsafe { cmd.ResourceBarrier(&[rt_to_psr]) };
        } else {
            let to_psr = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            unsafe { cmd.ResourceBarrier(&[to_psr]) };
        }
        // The main depth is restored to DEPTH_WRITE for the next frame. The
        // froxel volume rests sampled and is reset to UNORDERED_ACCESS by next
        // frame's graph-driven FogFroxel producer barrier, so it needs no inline
        // reset here.
        let depth_back = transition_barrier(
            &self.depth_resource,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        unsafe { cmd.ResourceBarrier(&[depth_back]) };
    }
}
