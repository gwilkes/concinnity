// src/directx/particle.rs
//
// GPU-compute particle system for the D3D12 backend. Each `ParticleEmitter`
// declared in the world produces one persistent `ParticleEmitterGpuState`
// carrying a default-heap pool buffer (UAV in the compute pass, SRV in the
// vertex pass) and a default-heap atomic spawn-counter buffer. Each frame the
// renderer:
//
//   1. Computes the per-emitter spawn budget CPU-side (a fractional
//      accumulator drives integer particle spawns per dispatch).
//   2. Copies the integer budgets into the per-emitter counter buffers from a
//      single per-frame upload ring.
//   3. Dispatches the `particle_simulate` compute kernel to age + integrate +
//      respawn each pool.
//   4. Transitions visible pools to NON_PIXEL_SHADER_RESOURCE and rasterises
//      one alpha-blended billboard quad per live particle into `hdr_resolve`.
//
// The render pass alpha-blends into the resolved HDR target after the
// volumetric-fog pass and before SSR / TAA so particles appear in screen-
// space reflections and are temporally stabilised by TAA history. Mirrors
// src/metal/particle.rs.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::{
    HDR_FORMAT, create_buffer, create_uav_buffer, transition_barrier, write_rgba8_srv,
};
use crate::gfx::particles::{ParticleEmitterRecord, ParticleSpawnState};
use crate::gfx::render_types::ParticleParams;

// HLSL sources (resolved by `shader_source`).

pub const PARTICLE_SIMULATE_HLSL: &str = include_str!("shaders/particle_simulate.hlsl");
pub const PARTICLE_VERT_HLSL: &str = include_str!("shaders/particle_vert.hlsl");
pub const PARTICLE_FRAG_HLSL: &str = include_str!("shaders/particle_frag.hlsl");

// Cap on the number of simultaneously-live particle emitters. The SRV heap
// reserves a fixed block of `MAX_EMITTERS` per-emitter albedo SRV slots at
// init, so runtime `add_emitter` past this many returns an error. Matches
// the storage shape of `MAX_DECALS`.
pub(in crate::directx) const MAX_EMITTERS: usize = 256;

// One particle slot on the GPU. Layout must match the `Particle` HLSL struct
// in `shaders/particle_simulate.hlsl` (32 bytes: `float3 + float` twice).
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct GpuParticle {
    position: [f32; 3],
    age: f32,
    velocity: [f32; 3],
    lifetime: f32,
}

// Per-frame view inputs to the particle render pass. Mirrors the
// `ParticleView` cbuffer in `particle_vert.hlsl` (96 bytes: float4x4 + two
// (float3, pad) slots).
#[derive(Copy, Clone)]
#[repr(C)]
struct ParticleView {
    vp: [[f32; 4]; 4],
    cam_right: [f32; 3],
    _pad0: f32,
    cam_up: [f32; 3],
    _pad1: f32,
}

// Compile the particle compute + vertex + fragment shaders. Used by
// [`ParticleResources::new`] at init and (in the future) by shader hot-reload.
// Compiled particle kernels: simulate cs, vertex vs, fragment ps bytecode.
type ParticleShaders = (Vec<u8>, Vec<u8>, Vec<u8>);

pub(in crate::directx) fn compile_particle_shaders(
    hot_reload: bool,
) -> Result<ParticleShaders, String> {
    let cs_src = shader_source(hot_reload, "particle_simulate.hlsl", PARTICLE_SIMULATE_HLSL);
    let vs_src = shader_source(hot_reload, "particle_vert.hlsl", PARTICLE_VERT_HLSL);
    let ps_src = shader_source(hot_reload, "particle_frag.hlsl", PARTICLE_FRAG_HLSL);
    let cs = compile_hlsl(&cs_src, "main", "cs_5_1")?;
    let vs = compile_hlsl(&vs_src, "main", "vs_5_1")?;
    let ps = compile_hlsl(&ps_src, "main", "ps_5_1")?;
    Ok((cs, vs, ps))
}

// Per-emitter persistent GPU state: the particle pool, the atomic spawn
// counter, and the CPU-side spawn accumulator. Dropping releases the
// underlying COM resources; the D3D12 driver keeps them alive until any
// in-flight command list referencing them completes.
pub(in crate::directx) struct ParticleEmitterGpuState {
    // Particle pool: `record.max_particles` slots of `GpuParticle`, used as a
    // UAV by the compute kernel and as a structured-buffer SRV by the vertex
    // stage. Resting state: UNORDERED_ACCESS.
    pub pool: ID3D12Resource,
    // One u32 atomic counter (4 bytes). Reset to the integer spawn budget
    // each frame via CopyBufferRegion from the per-frame upload ring;
    // decremented by the compute kernel as threads claim spawn slots.
    // Resting state: UNORDERED_ACCESS.
    pub spawn_counter: ID3D12Resource,
    // Carry-over fractional spawn count. Combined with `dt` and the
    // emitter's `spawn_rate` to produce the integer spawn budget for each
    // dispatch. Interior-mutable because `record_frame` (which calls into
    // `encode_particles`) holds `&self`; the field is only touched on the
    // render thread.
    pub spawn_state: std::cell::Cell<ParticleSpawnState>,
}

// Compute root signature for `particle_simulate`:
//   [0] root CBV b0 : ParticleParams (per-emitter, per-frame)
//   [1] root UAV u0 : pool (RWStructuredBuffer<Particle>)
//   [2] root UAV u1 : spawn_counter (RWByteAddressBuffer)
fn create_simulate_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
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
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "particle simulate root sig")
}

// Graphics root signature for `particle_vert` + `particle_frag`:
//   [0] root CBV b0   : ParticleView   (per-frame)
//   [1] root CBV b1   : ParticleParams (per-emitter)
//   [2] root SRV t0   : pool           (structured-buffer SRV)
//   [3] descriptor table SRV t1 : emitter albedo texture
//   static sampler s0 : linear clamp
fn create_render_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let albedo_range = D3D12_DESCRIPTOR_RANGE {
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
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
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
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0, // t0
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &albedo_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let samp = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
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
        pStaticSamplers: &samp,
        // The vertex shader emits the quad from SV_VertexID; no input layout.
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "particle render root sig")
}

fn create_simulate_pso(
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
        .map_err(|e| format!("create particle simulate PSO: {e}"))
}

fn create_render_pso(
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
        // No input layout; the vertex shader reads the pool by SV_InstanceID
        // and synthesises the quad corner from SV_VertexID.
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
                    SrcBlend: D3D12_BLEND_SRC_ALPHA,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_SRC_ALPHA,
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
        .map_err(|e| format!("create particle render PSO: {e}"))
}

// Pipelines + per-frame uniform rings shared across emitters. Owned by
// `DxContext` exactly once; built lazily either at init (when the world
// declares ≥1 emitter) or on the first runtime `add_emitter`.
pub(in crate::directx) struct ParticleResources {
    pub(in crate::directx) simulate_root_sig: ID3D12RootSignature,
    pub(in crate::directx) simulate_pso: ID3D12PipelineState,
    pub(in crate::directx) render_root_sig: ID3D12RootSignature,
    pub(in crate::directx) render_pso: ID3D12PipelineState,

    // Per-frame view UBO (single 96-byte block), persistently mapped.
    pub(in crate::directx) view_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) view_ubo_ptrs: Vec<*mut u8>,

    // Per-frame, per-emitter `ParticleParams` ring. Each slot is
    // `align256(sizeof(ParticleParams))` so the per-emitter CBV GPU address is
    // naturally 256-aligned.
    pub(in crate::directx) params_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) params_ubo_ptrs: Vec<*mut u8>,
    pub(in crate::directx) params_stride: u64,

    // Per-frame upload ring for the integer spawn budgets. One u32 per slot;
    // copied into each emitter's atomic counter at the top of `encode_particles`.
    pub(in crate::directx) budget_upload_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) budget_upload_ptrs: Vec<*mut u8>,
    pub(in crate::directx) budget_stride: u64,

    // Heap slot of the first per-emitter albedo SRV; slot `i` is the SRV for
    // emitter id `i`. Written by `add_emitter`.
    pub(in crate::directx) emitter_srv_base_slot: usize,
}

impl ParticleResources {
    // Build the particle compute + render pipelines and the per-frame
    // uniform rings. Called from `DxContext::new` (when the world declared
    // any emitter) or from the first runtime `add_emitter`.
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        emitter_srv_base_slot: usize,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let (cs, vs, ps) = compile_particle_shaders(hot_reload)?;

        let simulate_root_sig = dump_on_err(info_queue, create_simulate_root_signature(device))?;
        let simulate_pso = dump_on_err(
            info_queue,
            create_simulate_pso(device, &simulate_root_sig, &cs),
        )?;
        let render_root_sig = dump_on_err(info_queue, create_render_root_signature(device))?;
        let render_pso = dump_on_err(
            info_queue,
            create_render_pso(device, &render_root_sig, &vs, &ps),
        )?;

        // Per-frame view UBO.
        let view_size = align256(std::mem::size_of::<ParticleView>() as u64);
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
                .map_err(|e| format!("map particle view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        // Per-frame, per-emitter params ring. One CBV is 256-aligned, so size
        // each slot to align256(sizeof(ParticleParams)).
        let params_stride = align256(std::mem::size_of::<ParticleParams>() as u64);
        let params_total = params_stride * MAX_EMITTERS as u64;
        let mut params_ubo_resources: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                params_total,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map particle params ubo: {e}"))?;
            params_ubo_ptrs.push(ptr as *mut u8);
            params_ubo_resources.push(buf);
        }

        // Per-frame spawn-budget upload ring. One u32 per slot; D3D12 requires
        // CopyBufferRegion source offsets to be 4-byte aligned, which a u32 stride
        // satisfies. No align256 needed here; this buffer is never bound as a CBV.
        let budget_stride: u64 = std::mem::size_of::<u32>() as u64;
        let budget_total = budget_stride * MAX_EMITTERS as u64;
        let mut budget_upload_resources: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut budget_upload_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                budget_total,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map particle budget upload: {e}"))?;
            budget_upload_ptrs.push(ptr as *mut u8);
            budget_upload_resources.push(buf);
        }

        Ok(Self {
            simulate_root_sig,
            simulate_pso,
            render_root_sig,
            render_pso,
            view_ubo_resources,
            view_ubo_ptrs,
            params_ubo_resources,
            params_ubo_ptrs,
            params_stride,
            budget_upload_resources,
            budget_upload_ptrs,
            budget_stride,
            emitter_srv_base_slot,
        })
    }
}

// Allocate the per-emitter GPU state: a zero-initialised pool buffer (UAV)
// and a 4-byte atomic spawn counter (UAV). Both resting in UNORDERED_ACCESS.
pub(in crate::directx) fn build_emitter_gpu_state(
    device: &ID3D12Device,
    command_queue: &ID3D12CommandQueue,
    record: &ParticleEmitterRecord,
) -> Result<ParticleEmitterGpuState, String> {
    let slots = record.max_particles as u64;
    let pool_bytes = slots * std::mem::size_of::<GpuParticle>() as u64;

    // Default-heap UAV buffer for the pool. Created in COMMON (D3D12 always makes
    // committed buffers in COMMON regardless of the requested state); zero_default_
    // buffer leaves it in its UNORDERED_ACCESS resting state, then the encoder flips
    // it to NON_PIXEL_SHADER_RESOURCE around the render pass and back.
    let pool = create_uav_buffer(device, pool_bytes, D3D12_RESOURCE_STATE_COMMON)?;
    zero_default_buffer(device, command_queue, &pool, pool_bytes)?;

    // 4-byte default-heap UAV counter, initial value 0. The encoder copies the
    // per-frame integer budget into it before each compute dispatch.
    let spawn_counter = create_uav_buffer(
        device,
        std::mem::size_of::<u32>() as u64,
        D3D12_RESOURCE_STATE_COMMON,
    )?;
    zero_default_buffer(
        device,
        command_queue,
        &spawn_counter,
        std::mem::size_of::<u32>() as u64,
    )?;

    Ok(ParticleEmitterGpuState {
        pool,
        spawn_counter,
        spawn_state: std::cell::Cell::new(ParticleSpawnState::default()),
    })
}

// Zero-initialise a freshly-created (COMMON) default-heap buffer by uploading
// from a temporary upload-heap buffer through a one-shot command list. The target
// is transitioned COMMON → COPY_DEST for the copy and then to UNORDERED_ACCESS,
// its resting state for the per-frame compute passes.
fn zero_default_buffer(
    device: &ID3D12Device,
    command_queue: &ID3D12CommandQueue,
    target: &ID3D12Resource,
    bytes: u64,
) -> Result<(), String> {
    let upload = create_buffer(
        device,
        bytes,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    // Zero the upload buffer via its persistent mapping.
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { upload.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| format!("zero_default_buffer: map upload: {e}"))?;
    unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, bytes as usize) };
    unsafe { upload.Unmap(0, None) };

    // One-shot copy command list. Pattern matches `upload_buffer` in texture.rs.
    let alloc: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| format!("zero_default_buffer: alloc: {e}"))?;
    let list: ID3D12GraphicsCommandList =
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None) }
            .map_err(|e| format!("zero_default_buffer: list: {e}"))?;

    let to_copy_dest = transition_barrier(
        target,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    unsafe { list.ResourceBarrier(&[to_copy_dest]) };
    unsafe { list.CopyBufferRegion(target, 0, &upload, 0, bytes) };
    let back_to_uav = transition_barrier(
        target,
        D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    unsafe { list.ResourceBarrier(&[back_to_uav]) };
    unsafe { list.Close() }.map_err(|e| format!("zero_default_buffer: close: {e}"))?;
    let cmd: ID3D12CommandList = windows::core::Interface::cast(&list)
        .map_err(|e| format!("zero_default_buffer: cast: {e}"))?;
    unsafe { command_queue.ExecuteCommandLists(&[Some(cmd)]) };

    // Wait for completion before returning so the upload buffer (about to go
    // out of scope) is not freed while still referenced. One-shot init only,
    // not on the hot path.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| format!("zero_default_buffer: fence: {e}"))?;
    unsafe { command_queue.Signal(&fence, 1) }
        .map_err(|e| format!("zero_default_buffer: signal: {e}"))?;
    if unsafe { fence.GetCompletedValue() } < 1 {
        let event =
            unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
                .map_err(|e| format!("zero_default_buffer: event: {e}"))?;
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| format!("zero_default_buffer: set event: {e}"))?;
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
        unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    }

    Ok(())
}

impl DxContext {
    // GPU descriptor handle for emitter `i`'s albedo SRV.
    pub(in crate::directx) fn emitter_albedo_srv_gpu(
        &self,
        i: usize,
    ) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let base = self
            .particle
            .resources
            .as_ref()
            .map(|s| s.emitter_srv_base_slot)
            .unwrap_or(0);
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + ((base + i) * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // Encode the per-emitter compute + render passes. A no-op when no
    // pipeline has been built (no emitter has ever existed in this session)
    // or when every slot is tombstoned. `elapsed` is the same value the rest
    // of the frame computed; the diff against `particle_last_elapsed` is the
    // frame `dt` driving spawn rates + integration. Takes `&self` because
    // `record_frame` is `&self`; per-frame mutable state (last-elapsed,
    // frame index, per-emitter spawn accumulators) lives in `Cell`s.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn encode_particles(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        elapsed: f32,
        vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) {
        let Some(resources) = self.particle.resources.as_ref() else {
            return;
        };
        if self.particle.records.is_empty() || self.particle.emitter_state.is_empty() {
            return;
        }

        let dt = (elapsed - self.particle.last_elapsed.get()).max(0.0);
        self.particle.last_elapsed.set(elapsed);
        let frame_index = self.particle.frame_index.get().wrapping_add(1);
        self.particle.frame_index.set(frame_index);

        // Visibility-cull per emitter for the *render* pass only. The compute
        // simulation still ticks every live pool so off-screen emitters stay
        // in a realistic mid-life state for when the camera turns back.
        // Tombstoned (None) slots are always invisible.
        let visible: Vec<bool> = self
            .particle
            .records
            .iter()
            .map(|slot| match slot {
                Some(r) => {
                    let (mn, mx) = r.aabb();
                    frustum.intersects_aabb(mn, mx)
                }
                None => false,
            })
            .collect();

        // Camera basis for camera-facing billboards: rows 0 and 1 of the view
        // matrix's 3×3 are the world-space right and up vectors (the view
        // matrix is column-major, so we read those rows out element-wise).
        let v = self.view_matrix;
        let cam_right = [v[0][0], v[1][0], v[2][0]];
        let cam_up = [v[0][1], v[1][1], v[2][1]];
        let view_uni = ParticleView {
            vp,
            cam_right,
            _pad0: 0.0,
            cam_up,
            _pad1: 0.0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const ParticleView as *const u8,
                resources.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<ParticleView>(),
            );
        }
        let view_gva = unsafe { resources.view_ubo_resources[frame_idx].GetGPUVirtualAddress() };
        let params_base_gva =
            unsafe { resources.params_ubo_resources[frame_idx].GetGPUVirtualAddress() };
        let budget_upload = &resources.budget_upload_resources[frame_idx];

        // Pass 1: take per-emitter spawn budgets, write the integer budget
        // into this frame's upload buffer + the matching `ParticleParams` slot
        // in the per-frame params ring. Collect per-emitter draw params so the
        // compute + render loops below can read them without re-borrowing.
        struct EmitterFrameData {
            params_gva: u64,
            pool_gva: u64,
            counter_gva: u64,
        }
        let mut frame_data: Vec<Option<EmitterFrameData>> =
            Vec::with_capacity(self.particle.records.len());

        for (i, (rec_slot, gpu_slot)) in self
            .particle
            .records
            .iter()
            .zip(self.particle.emitter_state.iter())
            .enumerate()
        {
            let (rec, gpu) = match (rec_slot.as_ref(), gpu_slot.as_ref()) {
                (Some(r), Some(g)) => (r, g),
                _ => {
                    frame_data.push(None);
                    continue;
                }
            };
            // Pull the persistent fractional accumulator out of the Cell,
            // harvest this frame's integer budget, and write the updated
            // accumulator back. `ParticleSpawnState` is `Copy`, so the
            // get/set pair around the in-place mutation is cheap.
            let mut spawn_state = gpu.spawn_state.get();
            let spawn_budget = spawn_state.take_budget(dt, rec.spawn_rate, rec.max_particles);
            gpu.spawn_state.set(spawn_state);

            // Write this frame's spawn budget into the per-emitter upload slot.
            unsafe {
                let dst = resources.budget_upload_ptrs[frame_idx]
                    .add((i as u64 * resources.budget_stride) as usize);
                std::ptr::copy_nonoverlapping(
                    &spawn_budget as *const u32 as *const u8,
                    dst,
                    std::mem::size_of::<u32>(),
                );
            }

            // Write this frame's ParticleParams into the per-emitter params slot.
            let params = rec.params(dt, spawn_budget, frame_index);
            unsafe {
                let dst = resources.params_ubo_ptrs[frame_idx]
                    .add((i as u64 * resources.params_stride) as usize);
                std::ptr::copy_nonoverlapping(
                    &params as *const ParticleParams as *const u8,
                    dst,
                    std::mem::size_of::<ParticleParams>(),
                );
            }

            frame_data.push(Some(EmitterFrameData {
                params_gva: params_base_gva + i as u64 * resources.params_stride,
                pool_gva: unsafe { gpu.pool.GetGPUVirtualAddress() },
                counter_gva: unsafe { gpu.spawn_counter.GetGPUVirtualAddress() },
            }));
        }

        // Pass 2: copy each live emitter's spawn budget into its counter
        // buffer. Counter is in UNORDERED_ACCESS (resting state); transition
        // to COPY_DEST, copy, transition back to UAV. Batched into a single
        // barrier per direction so the validation noise stays low.
        let mut to_copy: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for (i, slot) in self.particle.emitter_state.iter().enumerate() {
            if frame_data.get(i).and_then(|d| d.as_ref()).is_none() {
                continue;
            }
            if let Some(gpu) = slot.as_ref() {
                to_copy.push(transition_barrier(
                    &gpu.spawn_counter,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
        }
        if !to_copy.is_empty() {
            unsafe { cmd.ResourceBarrier(&to_copy) };
        }
        for (i, slot) in self.particle.emitter_state.iter().enumerate() {
            if frame_data.get(i).and_then(|d| d.as_ref()).is_none() {
                continue;
            }
            if let Some(gpu) = slot.as_ref() {
                unsafe {
                    cmd.CopyBufferRegion(
                        &gpu.spawn_counter,
                        0,
                        budget_upload,
                        i as u64 * resources.budget_stride,
                        std::mem::size_of::<u32>() as u64,
                    );
                }
            }
        }
        let mut to_uav: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for (i, slot) in self.particle.emitter_state.iter().enumerate() {
            if frame_data.get(i).and_then(|d| d.as_ref()).is_none() {
                continue;
            }
            if let Some(gpu) = slot.as_ref() {
                to_uav.push(transition_barrier(
                    &gpu.spawn_counter,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ));
            }
        }
        if !to_uav.is_empty() {
            unsafe { cmd.ResourceBarrier(&to_uav) };
        }

        // Pass 3: compute dispatches. Pool + counter are both in
        // UNORDERED_ACCESS state already; the kernel reads + writes through
        // its root UAVs. Each emitter is independent so no UAV barrier is
        // needed between dispatches (resources are disjoint).
        unsafe {
            cmd.SetComputeRootSignature(&resources.simulate_root_sig);
            cmd.SetPipelineState(&resources.simulate_pso);
        }
        for (i, data) in frame_data.iter().enumerate() {
            let Some(data) = data else {
                continue;
            };
            let Some(rec) = self.particle.records[i].as_ref() else {
                continue;
            };
            unsafe {
                cmd.SetComputeRootConstantBufferView(0, data.params_gva);
                cmd.SetComputeRootUnorderedAccessView(1, data.pool_gva);
                cmd.SetComputeRootUnorderedAccessView(2, data.counter_gva);
                let groups = rec.max_particles.div_ceil(64);
                cmd.Dispatch(groups, 1, 1);
            }
        }

        // Pass 4: transition visible pools UAV → NON_PIXEL_SHADER_RESOURCE
        // so the vertex shader's structured-buffer SRV reads them. Invisible
        // pools stay in UAV (no render draw, no transition needed).
        let mut to_srv: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        for (i, slot) in self.particle.emitter_state.iter().enumerate() {
            if !visible[i] {
                continue;
            }
            if let Some(gpu) = slot.as_ref() {
                to_srv.push(transition_barrier(
                    &gpu.pool,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ));
            }
        }
        let any_visible = !to_srv.is_empty();
        if any_visible {
            unsafe { cmd.ResourceBarrier(&to_srv) };
        }

        // Pass 5: render the visible emitters. `hdr_resolve` (or
        // `hdr_color` MSAA-off) is in PIXEL_SHADER_RESOURCE after the fog
        // pass; flip it to RENDER_TARGET, render, flip back.
        if any_visible {
            let scene_rtv: D3D12_CPU_DESCRIPTOR_HANDLE =
                if let Some(hdr_resolve) = &self.hdr.resolve {
                    let to_rt = transition_barrier(
                        hdr_resolve,
                        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                    );
                    unsafe { cmd.ResourceBarrier(&[to_rt]) };
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
                let viewport = D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: w as f32,
                    Height: h as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                };
                cmd.RSSetViewports(&[viewport]);
                let scissor = RECT {
                    left: 0,
                    top: 0,
                    right: w as i32,
                    bottom: h as i32,
                };
                cmd.RSSetScissorRects(&[scissor]);
                cmd.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
                );
                cmd.SetPipelineState(&resources.render_pso);
                cmd.SetGraphicsRootSignature(&resources.render_root_sig);
                cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
                cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            }

            for (i, data) in frame_data.iter().enumerate() {
                if !visible[i] {
                    continue;
                }
                let Some(data) = data else {
                    continue;
                };
                let Some(rec) = self.particle.records[i].as_ref() else {
                    continue;
                };
                let albedo_srv_gpu = self.emitter_albedo_srv_gpu(i);
                unsafe {
                    cmd.SetGraphicsRootConstantBufferView(1, data.params_gva);
                    cmd.SetGraphicsRootShaderResourceView(2, data.pool_gva);
                    cmd.SetGraphicsRootDescriptorTable(3, albedo_srv_gpu);
                    cmd.DrawInstanced(4, rec.max_particles, 0, 0);
                }
                self.inc_draw_calls(1);
            }

            // Restore the scene target to PIXEL_SHADER_RESOURCE for the SSR
            // resolve / TAA / bloom / composite chain.
            if let Some(hdr_resolve) = &self.hdr.resolve {
                let to_psr = transition_barrier(
                    hdr_resolve,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                unsafe { cmd.ResourceBarrier(&[to_psr]) };
            } else {
                let to_psr = transition_barrier(
                    &self.hdr.color,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                unsafe { cmd.ResourceBarrier(&[to_psr]) };
            }

            // Restore visible pools back to UAV (their resting state for the
            // next frame's compute dispatch).
            let mut to_uav: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
            for (i, slot) in self.particle.emitter_state.iter().enumerate() {
                if !visible[i] {
                    continue;
                }
                if let Some(gpu) = slot.as_ref() {
                    to_uav.push(transition_barrier(
                        &gpu.pool,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ));
                }
            }
            if !to_uav.is_empty() {
                unsafe { cmd.ResourceBarrier(&to_uav) };
            }
        }
    }
}

// Runtime mutation (RenderBackend::add_emitter / remove_emitter)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/decal.rs].
#[allow(
    dead_code,
    reason = "cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots, live in the concinnity binary"
)]
impl DxContext {
    // Append a runtime emitter. Builds the particle pipelines + per-frame
    // uniform rings on first use (matching the init-time path) so a world
    // that never declared an emitter pays zero pipeline cost until the
    // first add. Reuses tombstoned slots from a prior `remove_emitter`
    // before growing the vec.
    pub fn add_emitter(&mut self, record: ParticleEmitterRecord) -> Result<usize, String> {
        if self.particle.resources.is_none() {
            let resources = ParticleResources::new(
                &self.device,
                self.particle.srv_base_slot,
                self.info_queue.as_ref(),
                self.hot_reload.enabled,
            )?;
            self.particle.resources = Some(resources);
        }
        let base_slot = self
            .particle
            .resources
            .as_ref()
            .map(|r| r.emitter_srv_base_slot)
            .ok_or_else(|| "add_emitter: particle pipeline unavailable".to_string())?;

        let gpu_state = build_emitter_gpu_state(&self.device, &self.command_queue, &record)?;
        let last_tex = self.descriptors.textures.len().saturating_sub(1);
        let tex_idx = record.texture_slot.min(last_tex);

        // Reuse a tombstoned slot if available; otherwise grow the vec.
        let id = if let Some(slot) = self.particle.free_slots.pop() {
            self.particle.records[slot] = Some(record);
            self.particle.emitter_state[slot] = Some(gpu_state);
            slot
        } else {
            if self.particle.records.len() >= MAX_EMITTERS {
                return Err(format!(
                    "add_emitter: MAX_EMITTERS ({MAX_EMITTERS}) exceeded"
                ));
            }
            self.particle.records.push(Some(record));
            self.particle.emitter_state.push(Some(gpu_state));
            self.particle.records.len() - 1
        };

        let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: unsafe {
                self.descriptors
                    .srv_heap
                    .GetCPUDescriptorHandleForHeapStart()
            }
            .ptr + (base_slot + id) * self.descriptors.srv_descriptor_size,
        };
        write_rgba8_srv(&self.device, &self.descriptors.textures[tex_idx], srv_cpu);
        Ok(id)
    }

    // Tombstone a runtime emitter slot. The id becomes invalid; the next
    // `add_emitter` may reuse it. The pool + counter resources are dropped;
    // the D3D12 driver keeps them alive until any in-flight command list
    // that referenced them completes, so this is safe to call mid-frame
    // between encode passes.
    pub fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String> {
        let rec_slot = self
            .particle
            .records
            .get_mut(emitter_id)
            .ok_or_else(|| format!("remove_emitter: id {emitter_id} out of range"))?;
        if rec_slot.is_none() {
            return Err(format!("remove_emitter: id {emitter_id} already removed"));
        }
        *rec_slot = None;
        if let Some(gpu_slot) = self.particle.emitter_state.get_mut(emitter_id) {
            *gpu_slot = None;
        }
        self.particle.free_slots.push(emitter_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_particle_layout_matches_hlsl() {
        // Mirrors the `Particle` struct in `shaders/particle_simulate.hlsl`:
        // float3 + float, twice = 32 bytes, layout 0/12/16/28.
        assert_eq!(std::mem::size_of::<GpuParticle>(), 32);
        assert_eq!(std::mem::offset_of!(GpuParticle, position), 0);
        assert_eq!(std::mem::offset_of!(GpuParticle, age), 12);
        assert_eq!(std::mem::offset_of!(GpuParticle, velocity), 16);
        assert_eq!(std::mem::offset_of!(GpuParticle, lifetime), 28);
    }

    #[test]
    fn particle_view_layout_matches_hlsl() {
        // Mirrors the `ParticleView` cbuffer in `particle_vert.hlsl`:
        // float4x4 (64) + (float3 + pad) + (float3 + pad) = 96.
        assert_eq!(std::mem::size_of::<ParticleView>(), 96);
        assert_eq!(std::mem::offset_of!(ParticleView, vp), 0);
        assert_eq!(std::mem::offset_of!(ParticleView, cam_right), 64);
        assert_eq!(std::mem::offset_of!(ParticleView, cam_up), 80);
    }
}
