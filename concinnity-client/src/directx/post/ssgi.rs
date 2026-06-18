// src/directx/post/ssgi.rs
//
// Screen-space global illumination for the D3D12 backend: a refinement of SSR.
// It reuses the SSR depth + normal pre-pass G-buffer (so turning SSGI on forces
// that pre-pass to run even when the SSR resolve is off) and runs two
// fullscreen passes on the hdr_resolve RMW chain after the main pass:
//
//   * gather:    per pixel, a cone of cosine-weighted hemisphere rays marched
//                against the G-buffer, accumulating the lit scene colour at
//                each on-screen hit into an off-screen `gi` target.
//   * composite: a depth-aware blur of that noisy `gi` target, additively
//                blended (ONE / ONE) into `hdr_resolve` so the near-field
//                indirect bounce layers on top of the IBL ambient.
//
// Pipelines, the `gi` target, and the encoder live together so the effect is a
// single unit. Mirrors src/metal/post/ssgi.rs.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::fullscreen::{FullscreenPass, encode_fullscreen};
use crate::gfx::render_types::SsgiParams;
use crate::gfx::ssgi::SsgiSettings;

use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::{
    HDR_FORMAT, create_buffer, create_rt_target, write_format_rtv, write_format_srv,
};

// HLSL source

pub const SSGI_HLSL: &str = include_str!("../shaders/ssgi.hlsl");

// Size of the SSGI gather + composite fragment-shader uniform block. 32 bytes;
// see `gfx::render_types::SsgiParams`.
const SSGI_PARAMS_UBO_SIZE: u64 = 32;

// Shader compilation

struct SsgiShaders {
    vs: Vec<u8>,
    gather_ps: Vec<u8>,
    composite_ps: Vec<u8>,
}

// Compile the SSGI fullscreen vertex shader and both fragment entry points.
fn compile_ssgi_shaders(hot_reload: bool) -> Result<SsgiShaders, String> {
    let src = shader_source(hot_reload, "ssgi.hlsl", SSGI_HLSL);
    Ok(SsgiShaders {
        vs: compile_hlsl(&src, "ssgi_fullscreen_vert", "vs_5_1")?,
        gather_ps: compile_hlsl(&src, "ssgi_gather_frag", "ps_5_1")?,
        composite_ps: compile_hlsl(&src, "ssgi_composite_frag", "ps_5_1")?,
    })
}

// Root signature

// Root signature shared by both SSGI passes: a root CBV at b0 (the 32-byte
// `SsgiParams` block) and two 1-SRV descriptor tables: t0 (scene radiance for
// the gather / the noisy gather output for the composite) and t1 (the SSR
// pre-pass G-buffer). One static linear-clamp sampler at s0.
fn create_ssgi_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let t0_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let t1_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1, // t1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // [0] Root CBV: SsgiParams at b0
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
        // [1] scene / gi SRV (t0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &t0_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [2] G-buffer SRV (t1)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &t1_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    // s0: linear-clamp for the scene / gi / G-buffer samples.
    let sampler = D3D12_STATIC_SAMPLER_DESC {
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
    let samplers = [sampler];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "ssgi root sig")
}

// PSO builder

// PSO for one SSGI fullscreen pass. Writes `HDR_FORMAT`; no depth, no vertex
// input (fullscreen triangle from `SV_VertexID`). `additive` configures an
// `ONE / ONE` add blend (the composite blends the indirect term into the scene)
// vs. a plain write (the gather fills its own `gi` target).
fn create_ssgi_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    additive: bool,
) -> Result<ID3D12PipelineState, String> {
    let blend = if additive {
        D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: true.into(),
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ONE,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ONE,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        }
    } else {
        D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: false.into(),
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        }
    };
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
                arr[0] = blend;
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };
    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create ssgi PSO: {e}"))
}

// Resources

// SSGI resources held by `DxContext` when `PostProcessConfig.indirect_lighting`
// is `ssgi`. Drops cleanly with the context: all D3D12 objects are
// COM-refcounted.
pub(in crate::directx) struct SsgiResources {
    // Resolved authored tunables; turned into a per-frame `SsgiParams` push.
    pub(in crate::directx) settings: SsgiSettings,

    // Gathered indirect radiance (`HDR_FORMAT`), before the depth-aware blur the
    // composite pass applies. The composite blends straight into `hdr_resolve`,
    // so only this intermediate `gi` texture lives here.
    gi: ID3D12Resource,
    gi_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    gi_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Per-frame params UBO (32-byte SsgiParams), persistently mapped.
    params_ubo_resources: Vec<ID3D12Resource>,
    params_ubo_ptrs: Vec<*mut u8>,

    // One root signature shared by both PSOs; gather (plain write) + composite
    // (additive blend).
    root_sig: ID3D12RootSignature,
    gather_pso: ID3D12PipelineState,
    composite_pso: ID3D12PipelineState,
}

#[allow(clippy::too_many_arguments)]
impl SsgiResources {
    // Build all SSGI resources. Called from `DxContext::new` only when the
    // world's `PostProcessConfig` selects `indirect_lighting: ssgi`.
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        settings: SsgiSettings,
        gi_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        gi_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        // The gather runs at `gi_scale`-reduced resolution; the composite
        // bilateral-upsamples it back to full resolution (it reads the gi
        // texture's own dimensions for the tap stride). Mirrors metal/post/ssgi.
        let (gw, gh) = settings.gi_dimensions(width, height);
        let gi = create_rt_target(device, gw, gh, HDR_FORMAT)?;
        write_format_rtv(device, &gi, gi_rtv, HDR_FORMAT);
        write_format_srv(device, &gi, gi_srv.0, HDR_FORMAT);

        let params_size = align256(SSGI_PARAMS_UBO_SIZE);
        let mut params_ubo_resources: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut params_ubo_ptrs: Vec<*mut u8> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                params_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("map ssgi params ubo: {e}"))?;
            params_ubo_ptrs.push(ptr as *mut u8);
            params_ubo_resources.push(buf);
        }

        let shaders = compile_ssgi_shaders(hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_ssgi_root_signature(device))?;
        let gather_pso = dump_on_err(
            info_queue,
            create_ssgi_pso(device, &root_sig, &shaders.vs, &shaders.gather_ps, false),
        )?;
        let composite_pso = dump_on_err(
            info_queue,
            create_ssgi_pso(device, &root_sig, &shaders.vs, &shaders.composite_ps, true),
        )?;

        Ok(Self {
            settings,
            gi,
            gi_rtv,
            gi_srv_gpu: gi_srv.1,
            params_ubo_resources,
            params_ubo_ptrs,
            root_sig,
            gather_pso,
            composite_pso,
        })
    }

    // Rebuild the `gi` target at a new resolution. The descriptor *slot* stays
    // where it was; only the backing resource changes, so the live pass
    // bindings (which point at the SRV slot's GPU handle) stay valid.
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
        // Reduced-res gather target (see `new`); the composite upsamples it.
        let (gw, gh) = self.settings.gi_dimensions(width, height);
        self.gi = create_rt_target(device, gw, gh, HDR_FORMAT)?;
        write_format_rtv(device, &self.gi, self.gi_rtv, HDR_FORMAT);
        write_format_srv(device, &self.gi, srv_cpu(self.gi_srv_gpu), HDR_FORMAT);
        Ok(())
    }
}

// Replacement SSGI PSOs returned by [`rebuild_ssgi_pipelines`]. The caller
// swaps them into the live `SsgiResources` only if both builds succeeded.
pub(in crate::directx) struct RebuiltSsgiPipelines {
    pub gather_pso: ID3D12PipelineState,
    pub composite_pso: ID3D12PipelineState,
}

// Rebuild both SSGI PSOs against fresh shader source, reusing the existing root
// signature. Returns the new PSOs for the caller to swap into the live
// `SsgiResources`.
pub(in crate::directx) fn rebuild_ssgi_pipelines(
    device: &ID3D12Device,
    ssgi: &SsgiResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltSsgiPipelines, String> {
    let shaders = compile_ssgi_shaders(hot_reload)?;
    let gather_pso = dump_on_err(
        info_queue,
        create_ssgi_pso(
            device,
            &ssgi.root_sig,
            &shaders.vs,
            &shaders.gather_ps,
            false,
        ),
    )?;
    let composite_pso = dump_on_err(
        info_queue,
        create_ssgi_pso(
            device,
            &ssgi.root_sig,
            &shaders.vs,
            &shaders.composite_ps,
            true,
        ),
    )?;
    Ok(RebuiltSsgiPipelines {
        gather_pso,
        composite_pso,
    })
}

// Swap freshly compiled SSGI PSOs into the live resources after a hot-reload.
pub(in crate::directx) fn swap_ssgi_pipelines(
    ssgi: &mut SsgiResources,
    rebuilt: RebuiltSsgiPipelines,
) {
    ssgi.gather_pso = rebuilt.gather_pso;
    ssgi.composite_pso = rebuilt.composite_pso;
}

// Encoder

impl DxContext {
    // Encode the SSGI gather + composite. The gather marches hemisphere rays
    // over the SSR pre-pass G-buffer and writes the noisy indirect radiance into
    // the `gi` target; the composite depth-aware-blurs it and additively blends
    // it into `hdr_resolve` (or `hdr_color` with MSAA off). Runs on the
    // hdr_resolve RMW chain after the main pass; only dispatched when SSGI is on
    // (and the SSR pre-pass G-buffer therefore exists).
    //
    // Both sub-passes are single-draw fullscreen passes, so each runs through the
    // shared `gfx::fullscreen` driver; the PSR<->RENDER_TARGET barrier bracket +
    // render-target bind live once in `DxContext::begin/end_fullscreen_rt`.
    pub(in crate::directx) fn encode_ssgi(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        // Resolve the pass's resources up front, before constructing either
        // encoder, so the driver never leaves a barrier bracket half-open. SSGI
        // gathers against the unified G-buffer pre-pass (view normal + linear
        // depth); if it is absent there is nothing to gather against, so skip
        // rather than read a stale descriptor.
        let Some(ssgi) = &self.ssgi else { return };
        let gbuffer_srv = match &self.gbuffer {
            Some(g) => g.normal_depth_srv_gpu,
            None => return,
        };

        // Build + upload this frame's SsgiParams; both sub-passes read the same
        // block via its GPU virtual address.
        let params = ssgi.settings.params(fov_y_radians, aspect);
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const SsgiParams as *const u8,
                ssgi.params_ubo_ptrs[frame_idx],
                std::mem::size_of::<SsgiParams>(),
            );
        }
        let params_gva = unsafe { ssgi.params_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Gather: hemisphere ray-march over the G-buffer -> gi target. t0 = lit
        // scene (the bounce-radiance source); t1 = pre-pass G-buffer.
        encode_fullscreen(
            &SsgiPass {
                ctx: self,
                ssgi,
                output: &ssgi.gi,
                output_rtv: ssgi.gi_rtv,
                pso: &ssgi.gather_pso,
                source_srv: self.hdr.srv_gpu,
                gbuffer_srv,
                params_gva,
            },
            cmd,
        );

        // Composite: depth-aware blur of gi, additively blended into the scene.
        // The scene target is `hdr_resolve` with MSAA on, `hdr_color` with MSAA
        // off (the same selection the decal / fog RMW passes make). t0 = noisy
        // gather output; t1 = G-buffer for depth weighting.
        let (scene_res, scene_rtv): (&ID3D12Resource, D3D12_CPU_DESCRIPTOR_HANDLE) =
            if let Some(hdr_resolve) = &self.hdr.resolve {
                (
                    hdr_resolve,
                    self.hdr
                        .resolve_rtv
                        .expect("hdr_resolve_rtv set when hdr_resolve is Some"),
                )
            } else {
                (&self.hdr.color, self.hdr.color_rtv)
            };
        encode_fullscreen(
            &SsgiPass {
                ctx: self,
                ssgi,
                output: scene_res,
                output_rtv: scene_rtv,
                pso: &ssgi.composite_pso,
                source_srv: ssgi.gi_srv_gpu,
                gbuffer_srv,
                params_gva,
            },
            cmd,
        );
    }
}

// Encoder for one SSGI fullscreen sub-pass (gather or composite): the target +
// pipeline + source SRV that distinguish the two, plus the shared G-buffer SRV
// and per-frame SsgiParams address. The PSR<->RENDER_TARGET barrier bracket +
// render-target bind live in `DxContext::begin/end_fullscreen_rt`
// (post/fullscreen.rs); only the SSGI-specific bind + draw is here. Constructed
// + driven by `encode_ssgi` through `gfx::fullscreen::encode_fullscreen`.
struct SsgiPass<'a> {
    ctx: &'a DxContext,
    ssgi: &'a SsgiResources,
    // Target this sub-pass writes (gi for the gather, the scene for the composite).
    output: &'a ID3D12Resource,
    output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pso: &'a ID3D12PipelineState,
    // t0: lit scene for the gather, the noisy gi target for the composite.
    source_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    // t1: the unified pre-pass G-buffer, shared by both sub-passes.
    gbuffer_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    params_gva: u64,
}

impl FullscreenPass for SsgiPass<'_> {
    type Rec = ID3D12GraphicsCommandList;

    fn begin(&self, cmd: &Self::Rec) {
        self.ctx
            .begin_fullscreen_rt(cmd, self.output, self.output_rtv);
    }

    fn draw(&self, cmd: &Self::Rec) {
        unsafe {
            cmd.SetPipelineState(self.pso);
            cmd.SetGraphicsRootSignature(&self.ssgi.root_sig);
            cmd.SetGraphicsRootConstantBufferView(0, self.params_gva);
            cmd.SetGraphicsRootDescriptorTable(1, self.source_srv);
            cmd.SetGraphicsRootDescriptorTable(2, self.gbuffer_srv);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }

    fn end(&self, cmd: &Self::Rec) {
        self.ctx.end_fullscreen_rt(cmd, self.output);
    }
}
