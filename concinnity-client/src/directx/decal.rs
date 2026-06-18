// src/directx/decal.rs
//
// Projected (deferred) decals for the D3D12 backend. Each decal is drawn as a
// unit cube (positions in `[-0.5, 0.5]^3`) transformed by its world model
// matrix and the camera VP; the fragment shader samples the main pass's depth
// attachment to reconstruct the world-space sample point at each pixel and
// tests it against the decal's local bounding box, stamping the texture onto
// whatever fills the box.
//
// Runs after the main HDR resolve and before SSR resolve / TAA, so decals
// are reflected and tracked by the temporal history just like the rest of
// the scene. Mirrors src/metal/decal.rs.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::{DxContext, FRAMES, align256, dump_on_err};
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::{
    HDR_FORMAT, ScopedBarrier, create_buffer, upload_buffer, write_rgba8_srv,
};
use crate::gfx::decal::DecalRecord;

// HLSL sources

pub const DECAL_VERT_HLSL: &str = include_str!("shaders/decal_vert.hlsl");
pub const DECAL_FRAG_HLSL: &str = include_str!("shaders/decal_frag.hlsl");

// Compile the decal vertex + fragment shaders, prepending the MSAA define so
// the depth SRV declaration in the fragment shader matches the resource's
// sample count. Used by [`DecalResources::new`] at init and by shader hot-
// reload to rebuild the decal PSO.
pub(in crate::directx) fn compile_decal_shaders(
    msaa_samples: u32,
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let define_line = if msaa_samples > 1 {
        "#define USE_MSAA 1\n"
    } else {
        "#define USE_MSAA 0\n"
    };
    let vs_body = shader_source(hot_reload, "decal_vert.hlsl", DECAL_VERT_HLSL);
    let ps_body = shader_source(hot_reload, "decal_frag.hlsl", DECAL_FRAG_HLSL);
    let vs_src = format!("{define_line}{vs_body}");
    let ps_src = format!("{define_line}{ps_body}");
    let vs = compile_hlsl(&vs_src, "main", "vs_5_1")?;
    let ps = compile_hlsl(&ps_src, "main", "ps_5_1")?;
    Ok((vs, ps))
}

// Rebuild the decal PSO against fresh shader source. Called from the
// DirectX shader hot-reload pass. The root signature is reused; the new
// PSO is returned for the caller to swap in atomically.
pub(in crate::directx) fn rebuild_decal_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    msaa_samples: u32,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<ID3D12PipelineState, String> {
    let (vs, ps) = compile_decal_shaders(msaa_samples, hot_reload)?;
    dump_on_err(info_queue, create_decal_pso(device, root_sig, &vs, &ps))
}

// Cap on the number of active decals: the SRV heap reserves a fixed block
// of `MAX_DECALS` per-decal albedo descriptors at init, so runtime adds past
// this many return an error. 256 is well under the 1024 SRV slot heap cap
// the existing backend allocates.
pub(in crate::directx) const MAX_DECALS: usize = 256;

// Eight unit-cube corners in `[-0.5, 0.5]^3`. Matches the Metal vertex list.
const CUBE_VERTS: [f32; 24] = [
    -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5,
    0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
];

// 36 indices forming 12 triangles wound CCW outward. Matches the Metal
// index list so the rasterised cube exactly mirrors the reference.
const CUBE_INDICES: [u16; 36] = [
    // -Z face                +Z face
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, // -Y                     +Y
    0, 1, 5, 0, 5, 4, 3, 6, 2, 3, 7, 6, // -X                     +X
    0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
];

// Per-frame view inputs to the decal pass. Mirrors the `DecalView` cbuffer
// in `decal_vert.hlsl` / `decal_frag.hlsl`. 144 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
struct DecalView {
    vp: [[f32; 4]; 4],
    inv_vp: [[f32; 4]; 4],
    viewport: [f32; 2],
    _pad: [f32; 2],
}

// Per-decal uniforms pushed before each draw. Mirrors the `DecalParams`
// cbuffer in the decal shaders. 160 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
struct DecalParams {
    model: [[f32; 4]; 4],
    inv_model: [[f32; 4]; 4],
    tint: [f32; 4],
    fade_pow: f32,
    _p0: f32,
    _p1: f32,
    _p2: f32,
}

// Root-signature layout (binds 1:1 with the HLSL register declarations):
//   [0] root CBV b0   DecalView    (per-frame)
//   [1] root CBV b1   DecalParams  (per-decal)
//   [2] table  t0     scene depth SRV (Texture2D[MS]<float>)
//   [3] table  t1     decal albedo SRV
//   static sampler s0 : linear clamp
fn create_decal_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let depth_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
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
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
    };
    serialize_desc_and_create(device, &desc, "decal root sig")
}

fn decal_input_layout() -> [D3D12_INPUT_ELEMENT_DESC; 1] {
    [D3D12_INPUT_ELEMENT_DESC {
        SemanticName: windows::core::s!("POSITION"),
        SemanticIndex: 0,
        Format: DXGI_FORMAT_R32G32B32_FLOAT,
        InputSlot: 0,
        AlignedByteOffset: 0,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    }]
}

// PSO for the decal pass. Writes the resolved HDR target with src-alpha /
// inv-src-alpha blending: the fragment shader emits `tint * tex.rgb` at the
// computed fade-weighted alpha and the blend composites it onto the scene.
// No depth attachment; the unit-cube + reconstructed-position clip in the
// fragment shader does the volumetric culling.
fn create_decal_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let layout = decal_input_layout();
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
            // Cull front faces: the camera may be inside a decal box. With
            // back-face culling on (the default) entering the volume would
            // make the unit cube disappear; culling the front face keeps the
            // back faces rasterised in both cases.
            CullMode: D3D12_CULL_MODE_FRONT,
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
        .map_err(|e| format!("create decal PSO: {e}"))
}

// Owned by `DxContext` exactly once: the decal pipeline, the unit-cube
// vertex / index buffers, and the per-frame uniform ring (one big upload
// buffer per frame split into per-decal regions). `decals` plus the freelist
// live on `DxContext` itself (mirroring the Metal context layout).
pub(in crate::directx) struct DecalResources {
    pub(in crate::directx) root_sig: ID3D12RootSignature,
    pub(in crate::directx) pso: ID3D12PipelineState,

    // Resources held to keep the GPU memory alive while the views below
    // reference them; the encoder binds through the views.
    #[allow(dead_code)]
    pub(in crate::directx) vertex_buffer: ID3D12Resource,
    pub(in crate::directx) vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    #[allow(dead_code)]
    pub(in crate::directx) index_buffer: ID3D12Resource,
    pub(in crate::directx) index_buffer_view: D3D12_INDEX_BUFFER_VIEW,

    // Per-frame view UBO (single 144-byte block), persistently mapped.
    pub(in crate::directx) view_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) view_ubo_ptrs: Vec<*mut u8>,
    // Per-frame `MAX_DECALS`-slot params ring. Each slot is `align256(160)`
    // = 256 bytes wide so the per-decal CBV GPU address is naturally aligned.
    pub(in crate::directx) params_ubo_resources: Vec<ID3D12Resource>,
    pub(in crate::directx) params_ubo_ptrs: Vec<*mut u8>,
    pub(in crate::directx) params_stride: u64,

    // Heap slot of the first per-decal albedo SRV; slot `i` is the SRV for
    // decal id `i`. Written by `add_decal` / refreshed by `update_texture_slot`
    // when a streamed texture lands.
    pub(in crate::directx) decal_srv_base_slot: usize,
    // Heap slot of the main-depth SRV. Bound at decal pass t0; the resource
    // is transitioned to PIXEL_SHADER_RESOURCE around the pass.
    pub(in crate::directx) depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

impl DecalResources {
    // Build the decal pipeline + unit-cube buffers + per-frame uniform
    // rings. Called from `DxContext::new` when the world declares any
    // `Decal` OR unconditionally so runtime `add_decal` works from a world
    // that started empty; the cost is one PSO + a few small buffers.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        command_queue: &ID3D12CommandQueue,
        msaa_samples: u32,
        decal_srv_base_slot: usize,
        depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let (vs, ps) = compile_decal_shaders(msaa_samples, hot_reload)?;

        let root_sig = dump_on_err(info_queue, create_decal_root_signature(device))?;
        let pso = dump_on_err(info_queue, create_decal_pso(device, &root_sig, &vs, &ps))?;

        // Unit-cube vertex + index buffers.
        let vbytes = unsafe {
            std::slice::from_raw_parts(
                CUBE_VERTS.as_ptr() as *const u8,
                std::mem::size_of_val(&CUBE_VERTS),
            )
        };
        let ibytes = unsafe {
            std::slice::from_raw_parts(
                CUBE_INDICES.as_ptr() as *const u8,
                std::mem::size_of_val(&CUBE_INDICES),
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
            StrideInBytes: 12,
        };
        let index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { index_buffer.GetGPUVirtualAddress() },
            SizeInBytes: ibytes.len() as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        // Per-frame view UBO.
        let view_size = align256(std::mem::size_of::<DecalView>() as u64);
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
                .map_err(|e| format!("map decal view ubo: {e}"))?;
            view_ubo_ptrs.push(ptr as *mut u8);
            view_ubo_resources.push(buf);
        }

        // Per-frame per-decal params ring. One CBV is 256-aligned, so size
        // each slot to align256(sizeof(DecalParams)).
        let params_stride = align256(std::mem::size_of::<DecalParams>() as u64);
        let params_total = params_stride * MAX_DECALS as u64;
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
                .map_err(|e| format!("map decal params ubo: {e}"))?;
            params_ubo_ptrs.push(ptr as *mut u8);
            params_ubo_resources.push(buf);
        }

        Ok(Self {
            root_sig,
            pso,
            vertex_buffer,
            vertex_buffer_view,
            index_buffer,
            index_buffer_view,
            view_ubo_resources,
            view_ubo_ptrs,
            params_ubo_resources,
            params_ubo_ptrs,
            params_stride,
            decal_srv_base_slot,
            depth_srv_gpu,
        })
    }
}

// Helpers for writing the main-depth SRV (so the runtime can rebuild it on
// resize if a future change adds that path) and the per-decal albedo SRVs.

// Write the main depth resource's SRV. MSAA: `Texture2DMS<float>`;
// otherwise plain `Texture2D<float>`. Format is the typed view of
// `R32_TYPELESS` (the depth resource's underlying format): `R32_FLOAT`.
pub(in crate::directx) fn write_main_depth_srv(
    device: &ID3D12Device,
    depth: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    sample_count: u32,
) {
    let srv_desc = if sample_count > 1 {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DMS,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DMS: D3D12_TEX2DMS_SRV {
                    UnusedField_NothingToDefine: 0,
                },
            },
        }
    } else {
        D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        }
    };
    unsafe { device.CreateShaderResourceView(depth, Some(&srv_desc), srv_cpu) };
}

// Encoder

#[allow(clippy::too_many_arguments)]
impl DxContext {
    // Encode the projected-decal pass. Called between the main HDR resolve
    // and the SSR resolve so a decal is reflected by SSR and tracked by
    // TAA's history buffer like the rest of the scene.
    //
    // `vp` is the same jittered view-projection the main pass rasterised
    // with; the inverse drives the world-space reconstruction in the
    // fragment shader.
    pub(in crate::directx) fn encode_decals(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        vp: [[f32; 4]; 4],
        frustum: &crate::gfx::frustum::Frustum,
    ) {
        let decals = match &self.decal.state {
            Some(s) => s,
            None => return,
        };
        if self.decal.records.iter().all(|slot| slot.is_none()) {
            return;
        }
        // Frustum-cull first so a frame where every live decal lands
        // off-screen skips the pass, including the depth-transition
        // barriers. Tombstoned (None) slots are always invisible.
        let visible_count = self
            .decal
            .records
            .iter()
            .filter(|slot| {
                slot.as_ref()
                    .map(|d| {
                        let (mn, mx) = d.aabb();
                        frustum.intersects_aabb(mn, mx)
                    })
                    .unwrap_or(false)
            })
            .count();
        if visible_count == 0 {
            return;
        }

        // Upload this frame's view UBO.
        let inv_vp = super::math::mat4_inverse(vp);
        let viewport = [self.render_width as f32, self.render_height as f32];
        let view_uni = DecalView {
            vp,
            inv_vp,
            viewport,
            _pad: [0.0; 2],
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const DecalView as *const u8,
                decals.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<DecalView>(),
            );
        }
        let view_gva = unsafe { decals.view_ubo_resources[frame_idx].GetGPUVirtualAddress() };
        let params_base_gva =
            unsafe { decals.params_ubo_resources[frame_idx].GetGPUVirtualAddress() };

        // Main depth -> PIXEL_SHADER_RESOURCE so the fragment can sample it; the
        // guard restores DEPTH_WRITE on drop (function end) so next frame's main
        // pass can clear/write it again. Declared before the scene guard so it
        // drops *after* it (LIFO): scene restored to PSR, then depth, matching
        // the original teardown order.
        let _depth_rmw = ScopedBarrier::new(
            cmd,
            &self.depth_resource,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );

        // hdr_resolve / hdr_color was transitioned to PIXEL_SHADER_RESOURCE by
        // encode_main_pass. The decal pass writes it directly as an RTV, so the
        // guard flips it to RENDER_TARGET now and back to PSR on drop for the
        // SSR resolve / TAA / bloom / composite passes.
        let (scene_res, scene_rtv): (&ID3D12Resource, D3D12_CPU_DESCRIPTOR_HANDLE) =
            if let Some(hdr_resolve) = &self.hdr.resolve {
                (
                    hdr_resolve,
                    self.hdr
                        .resolve_rtv
                        .expect("hdr_resolve_rtv set when hdr_resolve is Some"),
                )
            } else {
                // MSAA off: `hdr_color` is the resolved scene.
                (&self.hdr.color, self.hdr.color_rtv)
            };
        let _scene_rmw = ScopedBarrier::new(
            cmd,
            scene_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );

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
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[decals.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&decals.index_buffer_view));

            cmd.SetPipelineState(&decals.pso);
            cmd.SetGraphicsRootSignature(&decals.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetGraphicsRootConstantBufferView(0, view_gva);
            cmd.SetGraphicsRootDescriptorTable(2, decals.depth_srv_gpu);
        }

        let last_tex = self.descriptors.textures.len().saturating_sub(1);
        for (i, slot) in self.decal.records.iter().enumerate() {
            let d = match slot {
                Some(d) => d,
                None => continue,
            };
            let (mn, mx) = d.aabb();
            if !frustum.intersects_aabb(mn, mx) {
                continue;
            }
            let params = DecalParams {
                model: d.model,
                inv_model: d.inv_model,
                tint: d.tint,
                fade_pow: 2.0,
                _p0: 0.0,
                _p1: 0.0,
                _p2: 0.0,
            };
            // Upload into this frame's per-decal slot.
            let dst = unsafe {
                decals.params_ubo_ptrs[frame_idx].add((i as u64 * decals.params_stride) as usize)
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &params as *const DecalParams as *const u8,
                    dst,
                    std::mem::size_of::<DecalParams>(),
                );
            }
            let params_gva = params_base_gva + i as u64 * decals.params_stride;
            let tex_slot = d.texture_slot.min(last_tex);
            // The decal's albedo SRV was written into the heap at
            // `decal_srv_base_slot + i` by `add_decal`, pointing at the
            // texture-pool resource referenced by `tex_slot`. Bind that
            // descriptor for t1.
            let albedo_srv_gpu = self.decal_albedo_srv_gpu(i);
            // tex_slot is consumed by add_decal when it writes the SRV; the
            // local clamp here only guards against a record whose authored
            // slot escapes the pool length (e.g. a decal authored after a
            // texture eviction). Drop tex_slot from the iteration since the
            // SRV already encodes the right resource.
            let _ = tex_slot;
            unsafe {
                cmd.SetGraphicsRootConstantBufferView(1, params_gva);
                cmd.SetGraphicsRootDescriptorTable(3, albedo_srv_gpu);
                cmd.DrawIndexedInstanced(36, 1, 0, 0, 0);
            }
            self.inc_draw_calls(1);
        }

        // `_scene_rmw` then `_depth_rmw` drop here (LIFO): scene RT -> PSR for
        // the SSR resolve / TAA / bloom / composite passes, then main depth
        // PSR -> DEPTH_WRITE for next frame.
    }

    // GPU descriptor handle for decal `i`'s albedo SRV (written into the
    // SRV heap at `decals_state.decal_srv_base_slot + i` by `add_decal`).
    pub(in crate::directx) fn decal_albedo_srv_gpu(&self, i: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let base = self
            .decal
            .state
            .as_ref()
            .map(|s| s.decal_srv_base_slot)
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
}

// Runtime mutation (RenderBackend::add_decal / remove_decal)

// Runtime-mutation surface driven only by the cn-debug command server
// ([debug/runtime_spawn.rs]). This module tree is compiled into both the FFI
// library crate (`concinnity_editor`) and the `concinnity` binary; the cn-debug
// chain is reachable from the binary's `main` but not from the library crate's
// (FFI) roots, so dead-code flags these in the lib build. Not backend-specific.
// `allow` (not `expect`) because the same source is live in the binary, where
// an `expect` would be unfulfilled. Suppressing the methods also marks them live
// roots, so the freelist / SRV-slot fields they touch stay un-flagged on their
// own.
#[allow(
    dead_code,
    reason = "cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots, live in the concinnity binary"
)]
impl DxContext {
    // Append a runtime decal. Writes the per-decal albedo SRV into the
    // reserved heap region; the encoder reads it next frame. Reuses
    // tombstoned slots from a prior `remove_decal` before growing the vec.
    pub fn add_decal(&mut self, record: DecalRecord) -> Result<usize, String> {
        let state = self
            .decal
            .state
            .as_ref()
            .ok_or_else(|| "add_decal: decal pipeline unavailable".to_string())?;
        let base_slot = state.decal_srv_base_slot;

        let last_tex = self.descriptors.textures.len().saturating_sub(1);
        let tex_idx = record.texture_slot.min(last_tex);

        // Write the SRV for the chosen texture into this decal's heap slot.
        // The slot may be reused from a prior tombstone, in which case the
        // old descriptor is just overwritten.
        let id = if let Some(slot) = self.decal.free_slots.pop() {
            self.decal.records[slot] = Some(record);
            slot
        } else {
            if self.decal.records.len() >= MAX_DECALS {
                return Err(format!("add_decal: MAX_DECALS ({MAX_DECALS}) exceeded"));
            }
            self.decal.records.push(Some(record));
            self.decal.records.len() - 1
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

    // Tombstone a runtime decal slot. The id becomes invalid; the next
    // `add_decal` may reuse it. Returns an error when the id is out of
    // range or already tombstoned.
    pub fn remove_decal(&mut self, decal_id: usize) -> Result<(), String> {
        let slot = self
            .decal
            .records
            .get_mut(decal_id)
            .ok_or_else(|| format!("remove_decal: id {decal_id} out of range"))?;
        if slot.is_none() {
            return Err(format!("remove_decal: id {decal_id} already removed"));
        }
        *slot = None;
        self.decal.free_slots.push(decal_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // DecalView must match the `DecalView` cbuffer (b0) in the decal shaders:
    // two column-major float4x4 then viewport and a 2-float pad (144 B total).
    #[test]
    fn decal_view_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<DecalView>(), 144);
        assert_eq!(std::mem::offset_of!(DecalView, vp), 0);
        assert_eq!(std::mem::offset_of!(DecalView, inv_vp), 64);
        assert_eq!(std::mem::offset_of!(DecalView, viewport), 128);
        assert_eq!(std::mem::offset_of!(DecalView, _pad), 136);
    }

    // DecalParams must match the `DecalParams` cbuffer (b1): model and
    // inv_model, the float4 tint, fade_pow, then three end pads (160 B total).
    #[test]
    fn decal_params_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<DecalParams>(), 160);
        assert_eq!(std::mem::offset_of!(DecalParams, model), 0);
        assert_eq!(std::mem::offset_of!(DecalParams, inv_model), 64);
        assert_eq!(std::mem::offset_of!(DecalParams, tint), 128);
        assert_eq!(std::mem::offset_of!(DecalParams, fade_pow), 144);
        assert_eq!(std::mem::offset_of!(DecalParams, _p0), 148);
        assert_eq!(std::mem::offset_of!(DecalParams, _p1), 152);
        assert_eq!(std::mem::offset_of!(DecalParams, _p2), 156);
    }
}
