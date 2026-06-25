// src/directx/post/rt_reflections.rs
//
// Hardware ray-traced reflection pass for the D3D12 backend. A fullscreen pixel
// pass that, per glossy pixel, rebuilds a world-space surface point + normal
// from the SSR pre-pass G-buffer, traces a reflection ray against the scene's
// DXR top-level acceleration structure ([`crate::directx::raytrace`]) with
// `RayQuery`, shades the hit (sun + IBL split-sum, optionally textured) or the
// IBL prefilter cube on a miss, and composites the result over the scene with
// the same Fresnel/gloss weighting SSR uses.
//
// It occupies the `SsrResolve` slot in the frame graph (reads `hdr_resolve`,
// writes its own output target) and is mutually exclusive with SSR resolve.
// Like SSGI it relies on the SSR pre-pass G-buffer, so the pre-pass is forced on
// whenever RT reflections are enabled. The `RayQuery` shader needs shader model
// 6.5, so it compiles through DXC (`crate::directx::dxc`), not FXC. Mirrors
// src/metal/post/rt_reflections.rs.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::render_types::RtParams;
use crate::gfx::rt_reflections::RtReflectionSettings;

use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::dxc::compile_hlsl_dxc;
use crate::directx::pipeline::{reflection_cut_prelude, serialize_desc_and_create, shader_source};
use crate::directx::texture::{
    HDR_FORMAT, create_buffer, create_rt_target, transition_barrier, write_format_rtv,
    write_format_srv,
};

// HLSL source (compiled via DXC to SM 6.5 for inline `RayQuery`).
pub const RT_REFLECTIONS_HLSL: &str = include_str!("../shaders/rt_reflections.hlsl");
// Shared reflection-probe sampling, concatenated ahead of the RT shader (no #include
// handler on DX) so a missed ray box-projects the local probe. The RT shader already
// uses t7 (prefilter) and s2 (repeat sampler), so the probe cube array + its sampler
// remap to t10 / s3 via the prepended #defines; b4 (ProbeSet) is free.
const PROBE_COMMON_HLSL: &str = include_str!("../shaders/probe_common.hlsl");
const PROBE_REGISTER_DEFINES: &str =
    "#define PROBE_CUBES_REGISTER t10\n#define PROBE_SAMPLER_REGISTER s3\n";

// Size of the RT-reflection fragment-shader uniform block. 144 bytes; see
// `gfx::render_types::RtParams`.
const RT_PARAMS_UBO_SIZE: u64 = 144;

// Shader compilation

struct RtShaders {
    vs: Vec<u8>,
    flat_ps: Vec<u8>,
    textured_ps: Vec<u8>,
}

// Compile the RT-reflection vertex shader + both fragment entry points through
// DXC (SM 6.5). Returns an `Err` (which the caller turns into an SSR fallback)
// when DXC is unavailable or the shader fails to compile.
fn compile_rt_shaders(hot_reload: bool) -> Result<RtShaders, String> {
    let body = shader_source(hot_reload, "rt_reflections.hlsl", RT_REFLECTIONS_HLSL);
    // The shared REFLECTION_ROUGHNESS_CUT prelude, then the register remap #defines
    // (before probe_common's #ifndef-guarded defaults), then the shared probe
    // helpers, then the RT shader body.
    let cut = reflection_cut_prelude();
    let probe_common = shader_source(hot_reload, "probe_common.hlsl", PROBE_COMMON_HLSL);
    let src = format!("{cut}{PROBE_REGISTER_DEFINES}{probe_common}\n{body}");
    Ok(RtShaders {
        vs: compile_hlsl_dxc(&src, "rt_fullscreen_vert", "vs_6_5")?,
        flat_ps: compile_hlsl_dxc(&src, "rt_reflections_frag", "ps_6_5")?,
        textured_ps: compile_hlsl_dxc(&src, "rt_reflections_frag_textured", "ps_6_5")?,
    })
}

// Root signature
//
// Root CBV b0 (RtParams); four root SRVs t0..t3 (TLAS / vertex / index / geometry
// table - all raw or structured buffers bound by GPU virtual address, which inline
// ray tracing supports for the acceleration structure too); five descriptor tables
// for the textures: scene t4, gbuffer normal+depth t5, roughness t6, prefilter cube
// t7, and the unbounded bindless pool at (t0, space1); and two more root SRVs t8/t9
// (deformed skinned verts / u16 skinned indices, for skinned hits). Three static
// samplers: linear-clamp s0, cube linear-clamp s1, linear-repeat s2.
fn create_rt_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let table_range = |reg: u32| D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: reg,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let scene_range = table_range(4); // t4
    let gbuffer_range = table_range(5); // t5
    let rough_range = table_range(6); // t6
    let cube_range = table_range(7); // t7
    let pool_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: u32::MAX, // unbounded bindless pool
        BaseShaderRegister: 0,    // t0
        RegisterSpace: 1,         // space1
        OffsetInDescriptorsFromTableStart: 0,
    };
    // The reflection-probe cube array at t10..t10+MAX_PROBES, space0 (t7 is the
    // prefilter cube; remapped via PROBE_CUBES_REGISTER). Unbaked slots hold the sky
    // prefilter cube, so a sample at any index is valid; the miss fallback box-projects
    // these when ProbeSet.count > 0.
    let probe_cube_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: crate::directx::probe_uniforms::MAX_PROBES as u32,
        BaseShaderRegister: 10, // t10..
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let root_cbv = |reg: u32| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let root_srv = |reg: u32| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: reg,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let table = |range: &D3D12_DESCRIPTOR_RANGE| D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: 1,
                pDescriptorRanges: range,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };

    let params = [
        root_cbv(0),              // [0] b0 RtParams
        root_srv(0),              // [1] t0 TLAS
        root_srv(1),              // [2] t1 vertex buffer (raw)
        root_srv(2),              // [3] t2 index buffer (raw)
        root_srv(3),              // [4] t3 geometry table (structured)
        table(&scene_range),      // [5] t4 scene
        table(&gbuffer_range),    // [6] t5 gbuffer normal+depth
        table(&rough_range),      // [7] t6 roughness
        table(&cube_range),       // [8] t7 prefilter cube
        table(&pool_range),       // [9] t0,space1 bindless pool
        root_srv(8),              // [10] t8 deformed skinned verts (raw)
        root_srv(9),              // [11] t9 skinned u16 indices (raw)
        table(&probe_cube_range), // [12] t10.. reflection-probe cube array
        root_cbv(4),              // [13] b4 ProbeSet
    ];

    let linear_clamp = |reg: u32| D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        ComparisonFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: reg,
        RegisterSpace: 0,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        ..Default::default()
    };
    let repeat = D3D12_STATIC_SAMPLER_DESC {
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        ShaderRegister: 2, // s2
        ..linear_clamp(2)
    };
    // s3: cube mip-linear clamp for the reflection-probe cube array (probe_common.hlsl
    // `cube_sampler`, remapped via PROBE_SAMPLER_REGISTER), matching the s1 prefilter
    // sampler.
    let samplers = [linear_clamp(0), linear_clamp(1), repeat, linear_clamp(3)];

    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    serialize_desc_and_create(device, &desc, "rt reflections root sig")
}

// PSO for one RT-reflection fullscreen variant. Writes `HDR_FORMAT`; no depth,
// no blend, no vertex input.
fn create_rt_pso(
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
        .map_err(|e| format!("create rt reflections PSO: {e}"))
}

// Resources

// Hardware-ray-traced-reflection resources held by `DxContext` when the world's
// `PostProcessConfig` enables `ray_traced_reflections` AND the GPU supports the
// DXR tier AND the DXC compile + acceleration-structure build succeed; otherwise
// the context leaves this `None` and the graph falls back to `SsrResolve`.
pub(in crate::directx) struct RtReflectionsResources {
    // Resolved authored tunables; turned into a per-frame `RtParams` push.
    pub(in crate::directx) settings: RtReflectionSettings,

    // Reflection output: the HDR scene with reflections composited in. Becomes
    // the "scene" SRV the TAA / bloom / composite passes consume (it owns its
    // own slot rather than reusing the optional SSR resolve output, because RT
    // can be authored with SSR resolve off).
    pub(in crate::directx) output: ID3D12Resource,
    output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,

    // Per-frame `RtParams` UBO (144-byte), persistently mapped.
    params_ubo_resources: Vec<ID3D12Resource>,
    params_ubo_ptrs: Vec<*mut u8>,

    // One root signature shared by both PSOs; flat-tint + textured (bindless).
    root_sig: ID3D12RootSignature,
    flat_pso: ID3D12PipelineState,
    textured_pso: ID3D12PipelineState,
}

#[allow(clippy::too_many_arguments)]
impl RtReflectionsResources {
    // Build the RT-reflection resources. Returns `Err` when the DXC compile
    // fails (DXC absent / shader error) so the caller can fall back to SSR; the
    // accel-structure build + DXR capability are gated separately by the caller.
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        settings: RtReflectionSettings,
        output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let output = create_rt_target(device, width, height, HDR_FORMAT)?;
        write_format_rtv(device, &output, output_rtv, HDR_FORMAT);
        write_format_srv(device, &output, output_srv.0, HDR_FORMAT);

        let params_size = align256(RT_PARAMS_UBO_SIZE);
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
                .map_err(|e| format!("map rt params ubo: {e}"))?;
            params_ubo_ptrs.push(ptr as *mut u8);
            params_ubo_resources.push(buf);
        }

        let shaders = compile_rt_shaders(hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_rt_root_signature(device))?;
        let flat_pso = dump_on_err(
            info_queue,
            create_rt_pso(device, &root_sig, &shaders.vs, &shaders.flat_ps),
        )?;
        let textured_pso = dump_on_err(
            info_queue,
            create_rt_pso(device, &root_sig, &shaders.vs, &shaders.textured_ps),
        )?;

        Ok(Self {
            settings,
            output,
            output_rtv,
            output_srv_gpu: output_srv.1,
            params_ubo_resources,
            params_ubo_ptrs,
            root_sig,
            flat_pso,
            textured_pso,
        })
    }

    // Rebuild the output target at a new resolution. The descriptor *slot* stays
    // put; only the backing resource changes, so the post stack's scene binding
    // (which points at the SRV slot's GPU handle) stays valid.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
        srv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu_base: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<(), String> {
        let srv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: srv_cpu_base.ptr + (self.output_srv_gpu.ptr - srv_gpu_base.ptr) as usize,
        };
        self.output = create_rt_target(device, width, height, HDR_FORMAT)?;
        write_format_rtv(device, &self.output, self.output_rtv, HDR_FORMAT);
        write_format_srv(device, &self.output, srv_cpu, HDR_FORMAT);
        Ok(())
    }
}

// Replacement RT PSOs returned by [`rebuild_rt_reflections_pipelines`]. The
// caller swaps them into the live resources only if both builds succeeded.
pub(in crate::directx) struct RebuiltRtPipelines {
    pub flat_pso: ID3D12PipelineState,
    pub textured_pso: ID3D12PipelineState,
}

// Rebuild both RT PSOs against fresh shader source, reusing the existing root
// signature. Returns the new PSOs for the caller to swap into the live
// `RtReflectionsResources`.
pub(in crate::directx) fn rebuild_rt_reflections_pipelines(
    device: &ID3D12Device,
    rt: &RtReflectionsResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltRtPipelines, String> {
    let shaders = compile_rt_shaders(hot_reload)?;
    let flat_pso = dump_on_err(
        info_queue,
        create_rt_pso(device, &rt.root_sig, &shaders.vs, &shaders.flat_ps),
    )?;
    let textured_pso = dump_on_err(
        info_queue,
        create_rt_pso(device, &rt.root_sig, &shaders.vs, &shaders.textured_ps),
    )?;
    Ok(RebuiltRtPipelines {
        flat_pso,
        textured_pso,
    })
}

// Swap freshly compiled RT PSOs into the live resources after a hot-reload.
pub(in crate::directx) fn swap_rt_reflections_pipelines(
    rt: &mut RtReflectionsResources,
    rebuilt: RebuiltRtPipelines,
) {
    rt.flat_pso = rebuilt.flat_pso;
    rt.textured_pso = rebuilt.textured_pso;
}

// Encoder

impl DxContext {
    // Encode the RT-reflection resolve: a fullscreen triangle that traces each
    // glossy pixel's reflection ray against the scene TLAS and composites the
    // reflected colour into `rt_reflections.output`. The output then becomes the
    // "scene" the TAA / bloom / composite passes consume via `scene_srv_for_post`.
    // No-op when any required resource is missing (the graph only schedules this
    // pass when RT is live, so the guards are defensive).
    pub(in crate::directx) fn encode_rt_reflections(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        fov_y_radians: f32,
        aspect: f32,
        cam_pos: [f32; 3],
    ) {
        let (rt, accel, gbuffer) = match (&self.rt_reflections, &self.rt_accel, &self.gbuffer) {
            (Some(r), Some(a), Some(g)) => (r, a, g),
            _ => return,
        };
        // The trace writes reflected radiance + weight into `rt.output`; the
        // reflection composite (below) blurs + composites it over the scene.
        let reflection_srv = rt.output_srv_gpu;

        // Build + upload this frame's RtParams. `inv_view_rot` is the view->world
        // rotation (the transpose of the view matrix's orthonormal 3x3), same as
        // the SSR resolve; `params` then fills in the camera-position translation
        // column to complete the camera-to-world transform.
        let v = self.view_matrix;
        let inv_view_rot = [
            [v[0][0], v[1][0], v[2][0], 0.0],
            [v[0][1], v[1][1], v[2][1], 0.0],
            [v[0][2], v[1][2], v[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let params = rt.settings.params(
            fov_y_radians,
            aspect,
            inv_view_rot,
            cam_pos,
            self.fog.sun_dir,
            self.fog.sun_color,
            self.env_map.prefilter_mip_count as f32,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                &params as *const RtParams as *const u8,
                rt.params_ubo_ptrs[frame_idx],
                std::mem::size_of::<RtParams>(),
            );
        }
        let params_gva = unsafe { rt.params_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Textured hit shading needs the bindless albedo/normal pool, which only
        // the GPU-cull bindless path populates; otherwise fall back to the
        // flat-tint variant (mirrors Metal's bindless-arg-buffer gate).
        let textured = self.cull.main_bindless_pso.is_some();
        let pso = if textured {
            &rt.textured_pso
        } else {
            &rt.flat_pso
        };

        let out_to_rt = transition_barrier(
            &rt.output,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { cmd.ResourceBarrier(&[out_to_rt]) };

        let w = self.render_width;
        let h = self.render_height;
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&rt.output_rtv), false, None);
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

            cmd.SetPipelineState(pso);
            cmd.SetGraphicsRootSignature(&rt.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, params_gva);
            // Root SRVs: TLAS / vertex / index / geometry table (by GPU virtual
            // address; inline ray tracing reads the TLAS through a root SRV).
            cmd.SetGraphicsRootShaderResourceView(1, accel.tlas_gva());
            cmd.SetGraphicsRootShaderResourceView(
                2,
                self.geometry.vertex_buffer.GetGPUVirtualAddress(),
            );
            cmd.SetGraphicsRootShaderResourceView(
                3,
                self.geometry.index_buffer.GetGPUVirtualAddress(),
            );
            cmd.SetGraphicsRootShaderResourceView(4, accel.geom_table_gva());
            // Texture tables.
            cmd.SetGraphicsRootDescriptorTable(5, self.hdr.srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(6, gbuffer.normal_depth_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(7, gbuffer.roughness_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(8, self.prefilter_cube_srv_gpu());
            if textured {
                cmd.SetGraphicsRootDescriptorTable(9, self.cull.bindless_pool_gpu);
            }
            // Skinned-geometry root SRVs: the deformed (posed) vertex buffer +
            // the u16 skinned index buffer the trace fetches a skinned hit from.
            // Both return a valid 1-element dummy GVA when there is no skinned
            // geometry, so the binding is always live.
            cmd.SetGraphicsRootShaderResourceView(10, accel.deformed_verts_gva());
            cmd.SetGraphicsRootShaderResourceView(11, accel.skinned_index_gva());
            // Reflection-probe miss fallback (probe_common.hlsl): the cube array table
            // at t10 + the per-frame ProbeSet CBV at b4. count == 0 keeps the sky path,
            // so a probe-less world is byte-identical to before.
            cmd.SetGraphicsRootDescriptorTable(12, self.probe_cube_table_gpu());
            cmd.SetGraphicsRootConstantBufferView(
                13,
                self.probe_set_cbvs[frame_idx].GetGPUVirtualAddress(),
            );
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }

        let out_to_psr = transition_barrier(
            &rt.output,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[out_to_psr]) };

        // Blur the reflection by roughness and composite it over the scene into the
        // reflection-composite output (the scene the post stack then consumes).
        self.encode_reflection_composite(cmd, reflection_srv);
    }
}
