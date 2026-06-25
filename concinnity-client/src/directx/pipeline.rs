// src/directx/pipeline.rs
//
// Cross-cutting D3D12 pipeline helpers shared by every pass:
//   * Shader-compile + root-signature serialisation helpers (`compile_hlsl`,
//     `serialize_and_create_root_sig`, `serialize_desc_and_create`).
//   * Vertex input layouts referenced by main + shadow + velocity + SSAO
//     pre-pass + text pipelines (`main_input_layout`, `skinned_input_layout`,
//     `text_input_layout`).
//   * The text overlay pipeline (`create_text_root_signature`,
//     `create_text_pso`) and the composite (post-process) pipeline
//     (`create_composite_root_signature`, `create_composite_pso`).
//
// Mirrors src/metal/pipeline.rs (trimmed in the audit to the equivalent set:
// shared helpers + text + composite). Per-effect pipelines live in their
// own files: bloom/TAA/SSAO in directx/post/, cull at directx/cull.rs,
// main + shadow in directx/init/pipelines.rs.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

// Shared shader-compile + root-sig helpers

// Resolve the HLSL source for one of the runtime-bundled built-in shaders.
// With `hot_reload` off this just returns `embedded` -- the same byte stream
// the binary has always compiled via `include_str!`. With `hot_reload` on
// (set by `cn debug` via `crate::app::dev_flags`) the helper first tries
// `<CARGO_MANIFEST_DIR>/src/directx/shaders/<name>` so a saved edit to the
// `.hlsl` file in this checkout is picked up on the next call; if the disk
// read fails (binary moved, file removed, IO error) it transparently falls
// back to the embedded source. The embedded fallback means a shipped binary
// keeps working no matter where it is run from. Mirrors
// `crate::metal::pipeline::shader_source`.
//
// Returning `Cow` keeps the no-hot-reload case allocation-free.
pub(in crate::directx) fn shader_source(
    hot_reload: bool,
    name: &str,
    embedded: &'static str,
) -> std::borrow::Cow<'static, str> {
    if hot_reload {
        let path = format!(
            "{}/src/directx/shaders/{}",
            env!("CARGO_MANIFEST_DIR"),
            name
        );
        match std::fs::read_to_string(&path) {
            Ok(s) => return std::borrow::Cow::Owned(s),
            Err(e) => {
                tracing::debug!(
                    "hot-reload: falling back to embedded source for {} ({})",
                    name,
                    e
                );
            }
        }
    }
    std::borrow::Cow::Borrowed(embedded)
}

// Generated HLSL prelude declaring the shared reflection roughness cut as a
// compile-time constant, single-sourced from `concinnity_core::gfx::ssr`. Prepended
// to the SSR / RT / reflection-composite shaders so the resolve gates and the
// composite blur ramp cannot drift from one another (mirrors Metal's
// `reflection_constants_prelude`). Compile-folds, so zero runtime cost.
pub(in crate::directx) fn reflection_cut_prelude() -> String {
    format!(
        "static const float REFLECTION_ROUGHNESS_CUT = {:?};\n",
        crate::gfx::ssr::REFLECTION_ROUGHNESS_CUT
    )
}

pub(super) fn compile_hlsl(source: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
    let src_c = std::ffi::CString::new(source).map_err(|e| format!("hlsl src cstr: {e}"))?;
    let entry_c = std::ffi::CString::new(entry).map_err(|e| format!("hlsl entry cstr: {e}"))?;
    let target_c = std::ffi::CString::new(target).map_err(|e| format!("hlsl target cstr: {e}"))?;

    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut error: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;

    // Force column-major matrix storage globally. Every built-in HLSL shader
    // already sets `#pragma pack_matrix(column_major)` at the top of its
    // source; this flag is defensive belt-and-suspenders against any future
    // shader that forgets the pragma, and propagates the same default to
    // user-supplied HLSL compiled via `build/shader.rs`.
    // The bindless main fragment shader declares an unbounded
    // `Texture2D tex_pool[] : register(t0, space1)` array; FXC refuses
    // unbounded descriptor tables without this opt-in flag.
    let flags = if cfg!(debug_assertions) {
        windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_DEBUG
            | windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_SKIP_OPTIMIZATION
            | windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
            | windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES
    } else {
        windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_OPTIMIZATION_LEVEL3
            | windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
            | windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES
    };

    let result = unsafe {
        D3DCompile(
            src_c.as_ptr() as *const std::ffi::c_void,
            source.len(),
            None,
            None,
            None,
            windows::core::PCSTR(entry_c.as_ptr() as *const u8),
            windows::core::PCSTR(target_c.as_ptr() as *const u8),
            flags,
            0,
            &mut blob,
            Some(&mut error),
        )
    };

    if result.is_err() {
        let msg = error
            .as_ref()
            .map(|e| {
                let ptr = unsafe { e.GetBufferPointer() } as *const u8;
                let len = unsafe { e.GetBufferSize() };
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) })
                    .into_owned()
            })
            .unwrap_or_else(|| "unknown compile error".to_string());
        return Err(format!("compile {target}: {msg}"));
    }

    let b = blob.ok_or_else(|| format!("compile {target}: no blob"))?;
    let ptr = unsafe { b.GetBufferPointer() } as *const u8;
    let len = unsafe { b.GetBufferSize() };
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

pub(super) fn serialize_and_create_root_sig(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER],
    label: &str,
) -> Result<ID3D12RootSignature, String> {
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, label)
}

pub(super) fn serialize_desc_and_create(
    device: &ID3D12Device,
    desc: &D3D12_ROOT_SIGNATURE_DESC,
    label: &str,
) -> Result<ID3D12RootSignature, String> {
    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut error: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    unsafe {
        windows::Win32::Graphics::Direct3D12::D3D12SerializeRootSignature(
            desc,
            windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            Some(&mut error),
        )
    }
    .map_err(|e| {
        let msg = error
            .as_ref()
            .map(|b| {
                let p = unsafe { b.GetBufferPointer() } as *const u8;
                let n = unsafe { b.GetBufferSize() };
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(p, n) }).into_owned()
            })
            .unwrap_or_default();
        format!("serialize {label}: {e} {msg}")
    })?;

    let b = blob.ok_or_else(|| format!("{label}: no blob after serialize"))?;
    let ptr = unsafe { b.GetBufferPointer() };
    let len = unsafe { b.GetBufferSize() };
    let sig_bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };

    unsafe { device.CreateRootSignature(0, sig_bytes) }.map_err(|e| format!("create {label}: {e}"))
}

// Shared vertex input layouts
//
// Used by main + shadow + velocity + SSAO pre-pass + text pipelines. Kept
// here because multiple per-effect pipelines reference them.

// Vertex input elements for the main pass (56-byte Vertex struct).
pub(super) fn main_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    // SAFETY: the PSTR literals live for 'static; this is standard D3D12 usage.
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
            SemanticName: windows::core::s!("TANGENT"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 24,
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
            SemanticName: windows::core::s!("TEXCOORD"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 48,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

// Vertex input elements for the skinned pass (80-byte SkinnedVertex struct):
// the 56-byte static attributes plus ushort4 joint indices (offset 56) and
// float4 blend weights (offset 64).
pub(super) fn skinned_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    let mut layout = main_input_layout();
    layout.push(D3D12_INPUT_ELEMENT_DESC {
        SemanticName: windows::core::s!("BLENDINDICES"),
        SemanticIndex: 0,
        Format: DXGI_FORMAT_R16G16B16A16_UINT,
        InputSlot: 0,
        AlignedByteOffset: 56,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    });
    layout.push(D3D12_INPUT_ELEMENT_DESC {
        SemanticName: windows::core::s!("BLENDWEIGHT"),
        SemanticIndex: 0,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        InputSlot: 0,
        AlignedByteOffset: 64,
        InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
        InstanceDataStepRate: 0,
    });
    layout
}

// Vertex input elements for the text pass (32-byte TextVertex struct).
fn text_input_layout() -> Vec<D3D12_INPUT_ELEMENT_DESC> {
    vec![
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("TEXCOORD"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 8,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 16,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ]
}

// Composite (post-process) pipeline
//
// A vertex-buffer-less fullscreen triangle samples the off-screen FP16 HDR
// scene target, composites the bloom mip, applies an exposure multiplier, the
// Narkowicz ACES tonemap + gamma 2.2 encode, a single FXAA 3.11-style edge
// pass, a 3D-LUT colour grade, and a radial vignette, then writes the
// swapchain backbuffer. Mirrors the Vulkan COMPOSITE_*_GLSL and the Metal post
// pipeline. `COMPOSITE_VERT_HLSL` is also reused by the bloom + TAA resolve
// passes (each as their fullscreen-triangle VS).

pub(super) const COMPOSITE_VERT_HLSL: &str = include_str!("shaders/composite_vert.hlsl");
pub(super) const COMPOSITE_FRAG_HLSL: &str = include_str!("shaders/composite_frag.hlsl");

// Compile the composite (post-process) pass shaders. Returns (vs, ps).
pub(super) fn compile_composite_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let vs = compile_hlsl(
        &shader_source(hot_reload, "composite_vert.hlsl", COMPOSITE_VERT_HLSL),
        "main",
        "vs_5_1",
    )?;
    let ps = compile_hlsl(
        &shader_source(hot_reload, "composite_frag.hlsl", COMPOSITE_FRAG_HLSL),
        "main",
        "ps_5_1",
    )?;
    Ok((vs, ps))
}

// Root signature for the composite pass: a 1-SRV descriptor table at t0 (the
// scene target: the HDR resolve, or the TAA output when TAA is on), a 1-SRV
// table at t1 (bloom mip 0), six 32-bit root constants at b0
// (`PostProcessParams`), a 1-SRV descriptor table at t2 (the 3D colour-grading
// LUT), and a static linear-clamp sampler at s0. The scene SRV is its own
// table (separate from bloom mip 0) so the runtime can re-point it at the
// per-frame TAA output without the two needing to be heap-contiguous. Clamp
// keeps the FXAA neighbour taps from wrapping at screen edges and the LUT taps
// inside the cube.
pub(super) fn create_composite_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let scene_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let bloom_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 1, // t1
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    // The 3D colour-grading LUT SRV is a separate, non-contiguous heap slot
    // (it sits after the bloom mips), so it needs its own descriptor table.
    let lut_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 2, // t2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // [0] Descriptor table: scene SRV (t0)
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
        // [1] Descriptor table: bloom mip 0 SRV (t1)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &bloom_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [2] Root constants: PostProcessParams (8 floats: 6 post tunables +
        // the `hdr_output` + `pq_output` HDR-branch toggles) at b0.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 8,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] Descriptor table: 3D colour-grading LUT SRV (t2)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &lut_range,
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
        ShaderRegister: 0, // s0
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
    serialize_desc_and_create(device, &desc, "composite root sig")
}

// PSO for the composite pass: a vertex-buffer-less fullscreen triangle that
// samples the HDR scene target and writes the single-sample swapchain
// backbuffer. No input layout, no depth.
pub(super) fn create_composite_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
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
        // No input layout; the vertex shader generates the triangle from
        // SV_VertexID.
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = rtv_format;
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
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
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
        .map_err(|e| format!("create composite PSO: {e}"))
}

// Text overlay pipeline
//
// Drawn after the composite into the single-sample swapchain backbuffer with
// straight alpha-blending. Per-call vertex + index buffers are uploaded
// dynamically by `encode_composite_and_text`.

pub(super) const TEXT_VERT_HLSL: &str = include_str!("shaders/text_vert.hlsl");
pub(super) const TEXT_FRAG_HLSL: &str = include_str!("shaders/text_frag.hlsl");

// Compile the text overlay shaders.
pub(super) fn compile_text_shaders(hot_reload: bool) -> Result<(Vec<u8>, Vec<u8>), String> {
    let text_vs = compile_hlsl(
        &shader_source(hot_reload, "text_vert.hlsl", TEXT_VERT_HLSL),
        "main",
        "vs_5_1",
    )?;
    let text_ps = compile_hlsl(
        &shader_source(hot_reload, "text_frag.hlsl", TEXT_FRAG_HLSL),
        "main",
        "ps_5_1",
    )?;
    Ok((text_vs, text_ps))
}

pub(super) fn create_text_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let atlas_srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let text_sampler_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // s0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };

    let params = [
        // [0] Root constants: win_width, win_height, pad, pad = 4 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Descriptor table: atlas SRV (t0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &atlas_srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [2] Descriptor table: text sampler (s0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &text_sampler_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
    ];

    serialize_and_create_root_sig(device, &params, "text root sig")
}

pub(super) fn create_text_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
) -> Result<ID3D12PipelineState, String> {
    let layout = text_input_layout();
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
        DSVFormat: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
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
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
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
        .map_err(|e| format!("create text PSO: {e}"))
}
