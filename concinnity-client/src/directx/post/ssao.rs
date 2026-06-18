// src/directx/post/ssao.rs
//
// SSAO (GTAO) for the D3D12 backend. Owns the GTAO horizon-search kernel
// pipeline, the depth-aware blur pipeline, and the `encode_ssao` per-frame
// encoder. The view normal + linear depth it samples come from the unified
// G-buffer pre-pass (post/gbuffer.rs).
//
// The main pass samples `ssao.ao_srv_gpu` (the blurred occlusion) to modulate
// its ambient term; when SSAO is disabled the renderer binds the 1×1 white
// fallback (built once in init/effects.rs) so the multiplier is a pass-through
// 1.0. Mirrors src/metal/post/ssao.rs.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::render_types::SsaoParams;

use crate::directx::context::{DxContext, dump_on_err};
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::{
    create_rt_target, transition_barrier, write_format_rtv, write_format_srv,
};

// HLSL sources

pub const SSAO_FULLSCREEN_VERT_HLSL: &str = include_str!("../shaders/ssao_fullscreen_vert.hlsl");
pub const SSAO_KERNEL_FRAG_HLSL: &str = include_str!("../shaders/ssao_kernel_frag.hlsl");
pub const SSAO_BLUR_FRAG_HLSL: &str = include_str!("../shaders/ssao_blur_frag.hlsl");

// Single-channel occlusion target format. 1.0 = unoccluded; the main pass
// multiplies the ambient term by this value. Both the GTAO kernel and the
// depth-aware blur target this format.
pub const SSAO_OCCLUSION_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8_UNORM;

// Shader compilation

// Compiled bytecode for every SSAO shader stage.
struct SsaoShaders {
    fullscreen_vs: Vec<u8>,
    kernel_ps: Vec<u8>,
    blur_ps: Vec<u8>,
}

// Compile every SSAO shader stage. Both the kernel + blur are fullscreen
// passes that read the unified G-buffer; neither has a geometry input.
fn compile_ssao_shaders(hot_reload: bool) -> Result<SsaoShaders, String> {
    Ok(SsaoShaders {
        fullscreen_vs: compile_hlsl(
            &shader_source(
                hot_reload,
                "ssao_fullscreen_vert.hlsl",
                SSAO_FULLSCREEN_VERT_HLSL,
            ),
            "main",
            "vs_5_1",
        )?,
        kernel_ps: compile_hlsl(
            &shader_source(hot_reload, "ssao_kernel_frag.hlsl", SSAO_KERNEL_FRAG_HLSL),
            "main",
            "ps_5_1",
        )?,
        blur_ps: compile_hlsl(
            &shader_source(hot_reload, "ssao_blur_frag.hlsl", SSAO_BLUR_FRAG_HLSL),
            "main",
            "ps_5_1",
        )?,
    })
}

// Root signatures + PSOs

// Root signature for the GTAO kernel fullscreen pass: four 32-bit root
// constants at b0 (SsaoParams: radius, intensity, tan_half_fov_y, aspect),
// a 1-SRV descriptor table at t0 (the pre-pass G-buffer), and a static
// linear-clamp sampler at s0.
fn create_ssao_kernel_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let gbuffer_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
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
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &gbuffer_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let static_sampler = D3D12_STATIC_SAMPLER_DESC {
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
        pStaticSamplers: &static_sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "ssao kernel root sig")
}

// Root signature for the depth-aware blur pass: two 1-SRV descriptor tables
// (raw occlusion at t0, G-buffer at t1) and a static linear-clamp sampler at
// s0.
fn create_ssao_blur_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let ao_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let gbuffer_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1, // t1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &ao_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &gbuffer_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];
    let static_sampler = D3D12_STATIC_SAMPLER_DESC {
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
        pStaticSamplers: &static_sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "ssao blur root sig")
}

// PSO for the fullscreen GTAO kernel and the depth-aware blur. Both write
// `SSAO_OCCLUSION_FORMAT`, share the fullscreen-triangle VS, and disable
// depth + blending.
fn create_ssao_fullscreen_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    label: &str,
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
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = SSAO_OCCLUSION_FORMAT;
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
        .map_err(|e| format!("create ssao {label} PSO: {e}"))
}

// Resources

// SSAO resources held by `DxContext` when `PostProcessConfig.ssao` is on.
// Drops cleanly with the context: all D3D12 objects are COM-refcounted.
pub(in crate::directx) struct SsaoResources {
    // Resolved authored tunables; turned into a per-frame `SsaoParams` push.
    pub(in crate::directx) settings: crate::gfx::ssao::SsaoSettings,

    // Raw GTAO kernel output (R8) and the blurred final occlusion (R8) the
    // main pass samples.
    pub(in crate::directx) ao_raw: ID3D12Resource,
    pub(in crate::directx) ao_raw_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) ao_raw_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // The blurred `ao_output` the main pass samples is the graph's transient and
    // is owned by `DxContext::transient_pool` (a placed resource); SSAO holds
    // only its RTV (blur writes it) + SRV (main samples it), written from the
    // pooled resource at build / resize time.
    pub(in crate::directx) ao_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) ao_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // CPU handle of the AO SRV; the same slot main-pass paths copy to the
    // per-draw "object" SRV table when rebinding the AO descriptor (not
    // used yet but handy for future descriptor-table reshuffles).
    #[allow(dead_code)]
    pub(in crate::directx) ao_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,

    // GTAO horizon-search kernel + depth-aware blur (fullscreen triangle).
    pub(in crate::directx) kernel_root_sig: ID3D12RootSignature,
    pub(in crate::directx) kernel_pso: ID3D12PipelineState,
    pub(in crate::directx) blur_root_sig: ID3D12RootSignature,
    pub(in crate::directx) blur_pso: ID3D12PipelineState,
}

#[allow(clippy::too_many_arguments)]
impl SsaoResources {
    // Build all SSAO resources. Called from `DxContext::new` only when the
    // world's `PostProcessConfig` enables SSAO. `ao_raw_*` / `ao_*` reserve
    // heap slots in the SRV + RTV heaps the caller laid out at init; the view
    // normal + depth the kernel samples come from the unified G-buffer pre-pass.
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        settings: crate::gfx::ssao::SsaoSettings,
        ao_raw_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        ao_raw_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
        ao_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        ao_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
        // The pooled `ao_output` resource (placed in `DxContext::transient_pool`);
        // SSAO writes its RTV + SRV but does not own it.
        ao_resource: &ID3D12Resource,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        // Raw occlusion is SSAO-internal (committed); the blurred `ao` is the
        // pooled `ao_output`, so SSAO only writes its RTV + SRV.
        let ao_raw = create_rt_target(device, width, height, SSAO_OCCLUSION_FORMAT)?;
        write_format_rtv(device, &ao_raw, ao_raw_rtv, SSAO_OCCLUSION_FORMAT);
        write_format_srv(device, &ao_raw, ao_raw_srv.0, SSAO_OCCLUSION_FORMAT);
        write_format_rtv(device, ao_resource, ao_rtv, SSAO_OCCLUSION_FORMAT);
        write_format_srv(device, ao_resource, ao_srv.0, SSAO_OCCLUSION_FORMAT);

        // Pipelines.
        let shaders = compile_ssao_shaders(hot_reload)?;
        let kernel_root_sig = dump_on_err(info_queue, create_ssao_kernel_root_signature(device))?;
        let kernel_pso = dump_on_err(
            info_queue,
            create_ssao_fullscreen_pso(
                device,
                &kernel_root_sig,
                &shaders.fullscreen_vs,
                &shaders.kernel_ps,
                "kernel",
            ),
        )?;
        let blur_root_sig = dump_on_err(info_queue, create_ssao_blur_root_signature(device))?;
        let blur_pso = dump_on_err(
            info_queue,
            create_ssao_fullscreen_pso(
                device,
                &blur_root_sig,
                &shaders.fullscreen_vs,
                &shaders.blur_ps,
                "blur",
            ),
        )?;

        Ok(Self {
            settings,
            ao_raw,
            ao_raw_rtv,
            ao_raw_srv_gpu: ao_raw_srv.1,
            ao_rtv,
            ao_srv_gpu: ao_srv.1,
            ao_srv_cpu: ao_srv.0,
            kernel_root_sig,
            kernel_pso,
            blur_root_sig,
            blur_pso,
        })
    }
}

// Replacement SSAO PSOs returned by [`rebuild_ssao_pipelines`]. The caller
// swaps them in atomically only if every build succeeded. Mirrors the safety
// pattern used by Metal's shader hot-reload.
pub(in crate::directx) struct RebuiltSsaoPipelines {
    pub kernel_pso: ID3D12PipelineState,
    pub blur_pso: ID3D12PipelineState,
}

impl SsaoResources {
    // Rebuild the raw + blurred occlusion targets at a new resolution. The
    // descriptor *slots* stay where they were; only the resources backing them
    // change.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
        // The rebuilt pooled `ao_output` resource; SSAO rewrites its RTV + SRV.
        ao_resource: &ID3D12Resource,
    ) -> Result<(), String> {
        let srv_cpu = |gpu: D3D12_GPU_DESCRIPTOR_HANDLE| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_cpu_base.ptr + (gpu.ptr - srv_gpu_base.ptr) as usize,
        };

        self.ao_raw = create_rt_target(device, width, height, SSAO_OCCLUSION_FORMAT)?;
        write_format_rtv(device, &self.ao_raw, self.ao_raw_rtv, SSAO_OCCLUSION_FORMAT);
        write_format_srv(
            device,
            &self.ao_raw,
            srv_cpu(self.ao_raw_srv_gpu),
            SSAO_OCCLUSION_FORMAT,
        );

        write_format_rtv(device, ao_resource, self.ao_rtv, SSAO_OCCLUSION_FORMAT);
        write_format_srv(
            device,
            ao_resource,
            srv_cpu(self.ao_srv_gpu),
            SSAO_OCCLUSION_FORMAT,
        );

        Ok(())
    }
}

// Rebuild every SSAO PSO against fresh shader source. Reuses each PSO's
// existing root signature, so descriptor-table layouts stay stable. Returns
// the new PSOs for the caller to swap into the live `SsaoResources`.
pub(in crate::directx) fn rebuild_ssao_pipelines(
    device: &ID3D12Device,
    ssao: &SsaoResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltSsaoPipelines, String> {
    let shaders = compile_ssao_shaders(hot_reload)?;
    let kernel_pso = dump_on_err(
        info_queue,
        create_ssao_fullscreen_pso(
            device,
            &ssao.kernel_root_sig,
            &shaders.fullscreen_vs,
            &shaders.kernel_ps,
            "kernel",
        ),
    )?;
    let blur_pso = dump_on_err(
        info_queue,
        create_ssao_fullscreen_pso(
            device,
            &ssao.blur_root_sig,
            &shaders.fullscreen_vs,
            &shaders.blur_ps,
            "blur",
        ),
    )?;
    Ok(RebuiltSsaoPipelines {
        kernel_pso,
        blur_pso,
    })
}

// Encoder

impl DxContext {
    // GPU descriptor handle of the AO SRV the main pass should sample.
    // Returns the blurred SSAO output when SSAO is on, otherwise the 1x1
    // white fallback so the ambient multiplier is a constant 1.0.
    pub(in crate::directx) fn ssao_ao_srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        match &self.ssao.resources {
            Some(s) => s.ao_srv_gpu,
            None => self.ssao.white_srv_gpu,
        }
    }

    // Encode the SSAO depth + normal pre-pass, the GTAO horizon-search
    // kernel, and the depth-aware blur. Called from `encode_main_pass` after
    // the main-pass RT/viewport setup so the main fragment shader can sample
    // the blurred occlusion. No-op when SSAO is disabled.
    pub(in crate::directx) fn encode_ssao(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        fov_y_radians: f32,
        aspect: f32,
    ) {
        let ssao = match &self.ssao.resources {
            Some(s) => s,
            None => return,
        };

        // SSAO reads the unified G-buffer pre-pass (view normal + linear
        // depth) and runs no geometry redraw of its own; skip if the G-buffer
        // is absent.
        let gbuffer_srv = match &self.gbuffer {
            Some(g) => g.normal_depth_srv_gpu,
            None => return,
        };
        let params = ssao.settings.params(fov_y_radians, aspect);
        let w = self.render_width;
        let h = self.render_height;

        // The kernel + blur are fullscreen passes; restore the viewport /
        // scissor / primitive topology the (now removed) geometry pre-pass used
        // to leave bound. This pass records into its own command list, where the
        // topology starts UNDEFINED, so it must be set here for the draws below.
        unsafe {
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

        // Kernel: GTAO horizon search over the G-buffer → raw occlusion.
        let to_rt = transition_barrier(
            &ssao.ao_raw,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { cmd.ResourceBarrier(&[to_rt]) };
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&ssao.ao_raw_rtv), false, None);
            cmd.SetPipelineState(&ssao.kernel_pso);
            cmd.SetGraphicsRootSignature(&ssao.kernel_root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRoot32BitConstants(
                0,
                4,
                &params as *const SsaoParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetGraphicsRootDescriptorTable(1, gbuffer_srv);
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        let raw_to_psr = transition_barrier(
            &ssao.ao_raw,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[raw_to_psr]) };

        // Blur: depth-aware smoothing of raw occlusion → final AO. The blurred
        // `ao` is the graph's `ao_output` resource: its transition into
        // RENDER_TARGET before this draw and back to PIXEL_SHADER_RESOURCE for
        // the main pass is graph-driven (the executor emits ao_output's
        // `barriers_before` around the SsaoBlur and Main passes), so no inline
        // barrier on `ao` is issued here. `ao_raw` stays inline above.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&ssao.ao_rtv), false, None);
            cmd.SetPipelineState(&ssao.blur_pso);
            cmd.SetGraphicsRootSignature(&ssao.blur_root_sig);
            cmd.SetGraphicsRootDescriptorTable(0, ssao.ao_raw_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(1, gbuffer_srv);
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }

        // Restore the SRV + sampler heaps the main pass expects.
        unsafe {
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
        }
    }
}
