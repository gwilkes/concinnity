// src/directx/glass.rs
//
// GlassPanel: the generic producer for the engine's transparent pass on the
// D3D12 backend. Each panel is a flat world-space quad (built once at init)
// drawn in the PassId::Transparent slot after SSR resolve and before TAA. The
// pass snapshots the pre-transparent scene, sorts the panels back-to-front by
// camera distance, and draws each one; the fragment shader refracts the
// snapshot, tints it, and adds a Fresnel rim (see shaders/glass.hlsl).
//
// Mirrors src/metal/glass.rs. Water is a separate (Metal-only) producer and is
// not ported here; the transparent slot on DX is glass-only.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::assets::GlassPanel;
use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{
    compile_hlsl, main_input_layout, serialize_desc_and_create, shader_source,
};
use crate::directx::texture::{
    HDR_FORMAT, create_buffer, create_hdr_resolve_target, transition_barrier, upload_buffer,
};
use crate::geometry::glass_quad::build_glass_quad;
use crate::gfx::mesh_payload::Vertex;

pub const GLASS_HLSL: &str = include_str!("shaders/glass.hlsl");

// Per-frame view inputs to the transparent pass. Mirrors the `TransparentView`
// cbuffer in glass.hlsl and `metal::uniforms::TransparentView`. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::directx) struct TransparentViewGpu {
    pub(in crate::directx) vp: [[f32; 4]; 4],
    pub(in crate::directx) inv_vp: [[f32; 4]; 4],
    pub(in crate::directx) camera_pos: [f32; 4],
    pub(in crate::directx) viewport: [f32; 2],
    pub(in crate::directx) time: f32,
    pub(in crate::directx) _pad: f32,
}

// Per-panel uniforms bound before each draw. Mirrors the `GlassParams` cbuffer
// in glass.hlsl and `metal::uniforms::GlassParams`. 64 bytes. Vec3 fields ride
// in float4s (.w unused) so the layout is byte-identical regardless of HLSL
// packing.
#[derive(Copy, Clone)]
#[repr(C)]
struct GlassParamsGpu {
    centre: [f32; 4],
    normal: [f32; 4],
    tint: [f32; 4],
    opacity: f32,
    refraction_strength: f32,
    fresnel_power: f32,
    _pad: f32,
}

// Per-panel GPU state: the static world-space quad VB + IB plus the per-panel
// uniform CBV. The quad is pre-transformed at build time and the params never
// change at runtime, so there is no per-frame work beyond projection.
struct GlassPanelRecord {
    #[allow(dead_code)]
    vertex_buffer: ID3D12Resource,
    vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    #[allow(dead_code)]
    index_buffer: ID3D12Resource,
    index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
    index_count: u32,
    #[allow(dead_code)]
    params_cbuffer: ID3D12Resource,
    params_cbuffer_gva: u64,
    visible: bool,
    // World-space centre, used for the back-to-front camera-distance sort.
    centre: [f32; 3],
}

// Owned by `DxContext` when the world declared any `GlassPanel`. Holds the
// shared pipeline, the per-panel records, the per-frame view CBV ring, and the
// scene-snapshot the fragment shader refracts. The depth SRV is the main-pass
// depth slot shared with the decal pass; the scene-copy SRV is the transparent
// pass's own heap slot.
pub(in crate::directx) struct GlassResources {
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    pub(in crate::directx) pso: ID3D12PipelineState,
    panels: Vec<GlassPanelRecord>,

    // Per-frame view UBO (single 160-byte block), persistently mapped.
    view_ubo_resources: Vec<ID3D12Resource>,
    view_ubo_ptrs: Vec<*mut u8>,

    // Pre-transparent scene snapshot. `encode_transparent` copies the scene
    // target into this each frame before the draws so refraction reads a stable
    // copy instead of the attachment being written.
    scene_copy: ID3D12Resource,
    scene_copy_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    scene_copy_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Main-depth SRV (shared with the decal pass); bound at t1 for the manual
    // occlusion test.
    depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

// The mapped view-ring pointers are POD raw pointers; the upload buffers stay
// alive through the `Vec<ID3D12Resource>` field and the pointers are written on
// the render thread only. Mirrors `RaymarchResources`.
unsafe impl Send for GlassResources {}
unsafe impl Sync for GlassResources {}

// Build the per-panel `GlassParamsGpu` from an authored panel. Pure; unit
// tested. Mirrors `metal::glass::glass_params_from`.
fn glass_params_from(panel: &GlassPanel) -> GlassParamsGpu {
    let n = panel.normal; // already unit-length from GlassPanel::from_args
    GlassParamsGpu {
        centre: [panel.centre[0], panel.centre[1], panel.centre[2], 0.0],
        normal: [n[0], n[1], n[2], 0.0],
        tint: [panel.tint[0], panel.tint[1], panel.tint[2], 0.0],
        opacity: panel.opacity,
        refraction_strength: panel.refraction_strength,
        fresnel_power: panel.fresnel_power,
        _pad: 0.0,
    }
}

// World-space distance from the camera to a panel centre. Larger = farther =
// drawn first. Pure; unit tested.
fn sort_distance(centre: [f32; 3], cam: [f32; 3]) -> f32 {
    let dx = centre[0] - cam[0];
    let dy = centre[1] - cam[1];
    let dz = centre[2] - cam[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Indices of the visible panels, ordered farthest-camera-distance first. Pure;
// unit tested. Invisible panels are excluded; the visible set is sorted via the
// shared `gfx::transparent::back_to_front_order`.
fn ordered_visible(centres: &[[f32; 3]], visible: &[bool], cam: [f32; 3]) -> Vec<usize> {
    let live: Vec<usize> = (0..centres.len()).filter(|&i| visible[i]).collect();
    let dists: Vec<f32> = live
        .iter()
        .map(|&i| sort_distance(centres[i], cam))
        .collect();
    crate::gfx::transparent::back_to_front_order(&dists)
        .into_iter()
        .map(|oi| live[oi])
        .collect()
}

// Compile the glass vertex + fragment shaders, prepending the MSAA define so
// the depth SRV declaration matches the resource's sample count. Used at init
// and by shader hot-reload.
pub(in crate::directx) fn compile_glass_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let define_line = if msaa_samples > 1 {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    let body = shader_source(hot_reload, "glass.hlsl", GLASS_HLSL);
    let src = format!("{define_line}{body}");
    let vs = compile_hlsl(&src, "vs_main", "vs_5_1")?;
    let ps = compile_hlsl(&src, "ps_main", "ps_5_1")?;
    Ok((vs, ps))
}

// Rebuild the glass PSO against fresh shader source. Called from the DirectX
// shader hot-reload pass; the root signature is reused.
pub(in crate::directx) fn rebuild_glass_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = compile_glass_shaders(msaa_samples, hot_reload)?;
    dump_on_err(info_queue, create_glass_pso(device, root_sig, &vs, &ps))
}

// Root-signature layout (binds 1:1 with the HLSL register declarations):
//   [0] root CBV b0   TransparentView (per-frame)
//   [1] root CBV b1   GlassParams     (per-panel)
//   [2] table  t0     scene-copy SRV  (Texture2D<float4>)
//   [3] table  t1     scene depth SRV (Texture2D[MS]<float>)
//   static sampler s0 : linear clamp
fn create_glass_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let scene_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let depth_range = D3D12_DESCRIPTOR_RANGE {
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
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &scene_range,
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
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    serialize_desc_and_create(device, &desc, "glass root sig")
}

// PSO for the glass pass. Writes the single-sample post-SSR scene target with
// src-alpha / inv-src-alpha blending. No depth attachment (the fragment shader
// does the manual occlusion test) and no face culling (the shader is
// two-sided). Standard 5-attribute vertex layout shared with the main pass.
fn create_glass_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
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
        .map_err(|e| format!("create glass PSO: {e}"))
}

// Write the scene-copy SRV (single-sample HDR Texture2D). Mirrors the raymarch
// scene-snapshot SRV; kept local so `resize_to` can re-point the descriptor.
fn write_scene_copy_srv(
    device: &ID3D12Device,
    scene_copy: &ID3D12Resource,
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
    unsafe { device.CreateShaderResourceView(scene_copy, Some(&desc), srv_cpu) };
}

impl GlassResources {
    // Build the glass pipeline + per-panel quad buffers + per-panel uniform
    // CBVs + the per-frame view ring + the scene snapshot. Called from
    // `DxContext::new` when the world declares any `GlassPanel`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        command_queue: &ID3D12CommandQueue,
        msaa_samples: u32,
        panels: &[GlassPanel],
        scene_copy_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        scene_copy_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        width: u32,
        height: u32,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let (vs, ps) = compile_glass_shaders(msaa_samples, hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_glass_root_signature(device))?;
        let pso = dump_on_err(info_queue, create_glass_pso(device, &root_sig, &vs, &ps))?;

        // Per-panel quad buffers + per-panel params CBV (built once; panels are
        // immutable at runtime).
        let mut records: Vec<GlassPanelRecord> = Vec::with_capacity(panels.len());
        for panel in panels {
            let (verts, idxs) = build_glass_quad(panel.centre, panel.normal, panel.half_size);

            // Flatten into the standard Vertex layout. Tangent is a placeholder
            // (the glass shader rebuilds its frame from the panel normal) and
            // per-vertex colour is unused.
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
            let vbytes = unsafe {
                std::slice::from_raw_parts(
                    packed.as_ptr() as *const u8,
                    std::mem::size_of_val(packed.as_slice()),
                )
            };
            let ibytes = unsafe {
                std::slice::from_raw_parts(
                    idxs.as_ptr() as *const u8,
                    std::mem::size_of_val(idxs.as_slice()),
                )
            };
            let vertex_buffer = upload_buffer(
                device,
                command_queue,
                vbytes,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            )?;
            let index_buffer = upload_buffer(
                device,
                command_queue,
                ibytes,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            )?;
            let vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: unsafe { vertex_buffer.GetGPUVirtualAddress() },
                SizeInBytes: vbytes.len() as u32,
                StrideInBytes: std::mem::size_of::<Vertex>() as u32,
            };
            let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: unsafe { index_buffer.GetGPUVirtualAddress() },
                SizeInBytes: ibytes.len() as u32,
                Format: DXGI_FORMAT_R16_UINT,
            };

            // Static per-panel params CBV.
            let params = glass_params_from(panel);
            let cb_size = align256(std::mem::size_of::<GlassParamsGpu>() as u64);
            let params_cbuffer = create_buffer(
                device,
                cb_size,
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut p = std::ptr::null_mut::<std::ffi::c_void>();
            unsafe { params_cbuffer.Map(0, None, Some(&mut p)) }
                .map_err(|e| format!("map glass params cb: {e}"))?;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &params as *const GlassParamsGpu as *const u8,
                    p as *mut u8,
                    std::mem::size_of::<GlassParamsGpu>(),
                );
                // Persistently mapped, never unmap.
            }
            let params_cbuffer_gva = unsafe { params_cbuffer.GetGPUVirtualAddress() };

            records.push(GlassPanelRecord {
                vertex_buffer,
                vertex_buffer_view,
                index_buffer,
                index_buffer_view,
                index_count: idxs.len() as u32,
                params_cbuffer,
                params_cbuffer_gva,
                visible: panel.visible,
                centre: panel.centre,
            });
        }

        // Per-frame view UBO ring.
        let view_size = align256(std::mem::size_of::<TransparentViewGpu>() as u64);
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
                .map_err(|e| format!("map glass view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        // Pre-transparent scene snapshot. Created in PIXEL_SHADER_RESOURCE;
        // `encode_transparent` flips it to COPY_DEST for the snapshot copy and
        // back each frame.
        let scene_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_copy_srv(device, &scene_copy, scene_copy_srv_cpu);

        Ok(Self {
            root_sig,
            pso,
            panels: records,
            view_ubo_resources,
            view_ubo_ptrs,
            scene_copy,
            scene_copy_srv_cpu,
            scene_copy_srv_gpu,
            depth_srv_gpu,
        })
    }

    // Recreate the scene snapshot at new render-target dimensions and rewrite
    // its SRV in place. The descriptor slot does not move, so the encoder's GPU
    // handle stays valid. Mirrors `RaymarchResources::resize_to`.
    pub(in crate::directx) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        self.scene_copy = create_hdr_resolve_target(device, width.max(1), height.max(1))?;
        write_scene_copy_srv(device, &self.scene_copy, self.scene_copy_srv_cpu);
        Ok(())
    }

    // True when any panel is currently visible. Drives
    // `FrameGraphInputs::transparent_enabled`.
    pub(in crate::directx) fn any_visible(&self) -> bool {
        self.panels.iter().any(|p| p.visible)
    }
}

impl DxContext {
    // Encode the transparent (glass) pass: snapshot the scene for refraction,
    // then draw every visible panel back-to-front into the post-SSR scene
    // target with SRC_ALPHA blending. No-op when no glass / no visible panels.
    pub(in crate::directx) fn encode_transparent(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        view: &TransparentViewGpu,
    ) -> Result<(), String> {
        let glass = match &self.glass {
            Some(g) => g,
            None => return Ok(()),
        };
        let cam = [view.camera_pos[0], view.camera_pos[1], view.camera_pos[2]];
        let centres: Vec<[f32; 3]> = glass.panels.iter().map(|p| p.centre).collect();
        let visible: Vec<bool> = glass.panels.iter().map(|p| p.visible).collect();
        let order = ordered_visible(&centres, &visible, cam);
        if order.is_empty() {
            return Ok(());
        }

        // Upload this frame's view UBO.
        unsafe {
            std::ptr::copy_nonoverlapping(
                view as *const TransparentViewGpu as *const u8,
                glass.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<TransparentViewGpu>(),
            );
        }
        let view_gva = unsafe { glass.view_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Pick the post-SSR scene target this pass blends into, mirroring
        // `scene_srv_for_post`'s precedence (the upscaler runs later, so it is
        // not consulted here). All rest in PIXEL_SHADER_RESOURCE after the
        // preceding pass and carry ALLOW_RENDER_TARGET.
        let (scene_res, scene_rtv): (&ID3D12Resource, D3D12_CPU_DESCRIPTOR_HANDLE) =
            if let Some(resolve) = self.ssr.as_ref().and_then(|s| s.resolve.as_ref()) {
                (&resolve.output, resolve.output_rtv)
            } else if let Some(hdr_resolve) = &self.hdr.resolve {
                (
                    hdr_resolve,
                    self.hdr
                        .resolve_rtv
                        .expect("hdr_resolve_rtv set when hdr_resolve is Some"),
                )
            } else {
                (&self.hdr.color, self.hdr.color_rtv)
            };

        // Snapshot the scene into `scene_copy` so refraction reads a stable copy
        // of what it is also blending into.
        let scene_to_copy = transition_barrier(
            scene_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let copy_to_dst = transition_barrier(
            &glass.scene_copy,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        unsafe { cmd.ResourceBarrier(&[scene_to_copy, copy_to_dst]) };
        unsafe { cmd.CopyResource(&glass.scene_copy, scene_res) };
        let scene_to_rt = transition_barrier(
            scene_res,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        let copy_to_psr = transition_barrier(
            &glass.scene_copy,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[scene_to_rt, copy_to_psr]) };

        // Main depth -> PIXEL_SHADER_RESOURCE so the fragment shader can Load it
        // for the manual occlusion test; restored to DEPTH_WRITE after the pass.
        let depth_to_psr = transition_barrier(
            &self.depth_resource,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[depth_to_psr]) };

        let w = self.render_width;
        let h = self.render_height;
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&scene_rtv), false, None);
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
            cmd.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

            cmd.SetPipelineState(&glass.pso);
            cmd.SetGraphicsRootSignature(&glass.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootDescriptorTable(2, glass.scene_copy_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(3, glass.depth_srv_gpu);
        }

        for &i in &order {
            let p = &glass.panels[i];
            unsafe {
                cmd.IASetVertexBuffers(0, Some(&[p.vertex_buffer_view]));
                cmd.IASetIndexBuffer(Some(&p.index_buffer_view));
                cmd.SetGraphicsRootConstantBufferView(1, p.params_cbuffer_gva);
                cmd.DrawIndexedInstanced(p.index_count, 1, 0, 0, 0);
            }
            self.inc_draw_calls(1);
        }

        // Restore: scene target back to PIXEL_SHADER_RESOURCE for TAA / bloom /
        // composite; main depth back to DEPTH_WRITE for next frame.
        let scene_back = transition_barrier(
            scene_res,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[scene_back]) };
        let depth_back = transition_barrier(
            &self.depth_resource,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        unsafe { cmd.ResourceBarrier(&[depth_back]) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn transparent_view_layout_matches_hlsl() {
        assert_eq!(size_of::<TransparentViewGpu>(), 160);
        assert_eq!(offset_of!(TransparentViewGpu, vp), 0);
        assert_eq!(offset_of!(TransparentViewGpu, inv_vp), 64);
        assert_eq!(offset_of!(TransparentViewGpu, camera_pos), 128);
        assert_eq!(offset_of!(TransparentViewGpu, viewport), 144);
        assert_eq!(offset_of!(TransparentViewGpu, time), 152);
        assert_eq!(offset_of!(TransparentViewGpu, _pad), 156);
    }

    #[test]
    fn glass_params_layout_matches_hlsl() {
        assert_eq!(size_of::<GlassParamsGpu>(), 64);
        assert_eq!(offset_of!(GlassParamsGpu, centre), 0);
        assert_eq!(offset_of!(GlassParamsGpu, normal), 16);
        assert_eq!(offset_of!(GlassParamsGpu, tint), 32);
        assert_eq!(offset_of!(GlassParamsGpu, opacity), 48);
        assert_eq!(offset_of!(GlassParamsGpu, refraction_strength), 52);
        assert_eq!(offset_of!(GlassParamsGpu, fresnel_power), 56);
        assert_eq!(offset_of!(GlassParamsGpu, _pad), 60);
    }

    #[test]
    fn glass_params_from_maps_fields() {
        let panel = GlassPanel {
            centre: [1.0, 2.0, 3.0],
            normal: [0.0, 0.0, 1.0],
            tint: [0.6, 0.85, 0.9],
            opacity: 0.45,
            refraction_strength: 0.04,
            fresnel_power: 4.0,
            ..Default::default()
        };
        let p = glass_params_from(&panel);
        assert_eq!(p.centre, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(p.normal, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(p.tint, [0.6, 0.85, 0.9, 0.0]);
        assert_eq!(p.opacity, 0.45);
        assert_eq!(p.refraction_strength, 0.04);
        assert_eq!(p.fresnel_power, 4.0);
        assert_eq!(p._pad, 0.0);
    }

    #[test]
    fn sort_distance_is_euclidean_and_monotone() {
        let cam = [0.0, 0.0, 0.0];
        let near = sort_distance([0.0, 0.0, 1.0], cam);
        let far = sort_distance([0.0, 0.0, 5.0], cam);
        assert!((near - 1.0).abs() < 1e-5);
        assert!((far - 5.0).abs() < 1e-5);
        assert!(far > near);
    }

    #[test]
    fn ordered_visible_excludes_hidden_and_sorts_back_to_front() {
        // Panel 1 is hidden; 0 (dist 5) and 2 (dist 3) are visible. Farthest
        // first => [0, 2]; the hidden panel never appears.
        let centres = [[0.0, 0.0, 5.0], [0.0, 0.0, 9.0], [0.0, 0.0, 3.0]];
        let visible = [true, false, true];
        let order = ordered_visible(&centres, &visible, [0.0, 0.0, 0.0]);
        assert_eq!(order, vec![0, 2]);
    }
}
