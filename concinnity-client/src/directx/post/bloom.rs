// src/directx/post/bloom.rs
//
// Bloom post-process: prefilter + downsample chain + additive upsample chain.
// Owns the per-mip render targets, the three PSOs they share (all using the
// fullscreen-triangle composite VS), the root signature, and the
// `encode_bloom` per-frame encoder.
//
// Mirrors src/metal/post/bloom.rs: same mip-count clamp (4..=6), same
// Karis 13-tap prefilter, same plain 13-tap downsample + 9-tap tent upsample.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::render_types::PostProcessParams;

use crate::directx::context::DxContext;
use crate::directx::pipeline::{
    COMPOSITE_VERT_HLSL, compile_hlsl, serialize_desc_and_create, shader_source,
};
use crate::directx::texture::{HDR_FORMAT, transition_barrier};

// HLSL sources

// Prefilter: HDR scene -> bloom mip 0. Karis 13-tap downsample, then an
// exposure multiply and a quadratic soft-knee luminance threshold.
pub const BLOOM_PREFILTER_HLSL: &str = include_str!("../shaders/bloom_prefilter.hlsl");

// Downsample: bloom mip i-1 -> mip i. Plain (non-Karis) 13-tap.
pub const BLOOM_DOWNSAMPLE_HLSL: &str = include_str!("../shaders/bloom_downsample.hlsl");

// Upsample: bloom mip i+1 -> mip i. 9-tap tent filter; the result is additively
// blended onto the destination mip by the pipeline blend state.
pub const BLOOM_UPSAMPLE_HLSL: &str = include_str!("../shaders/bloom_upsample.hlsl");

// Shader compilation

// Compiled bloom-chain shader bytecode. All three passes share the
// fullscreen-triangle vertex shader (`COMPOSITE_VERT_HLSL`).
pub(in crate::directx) struct BloomShaders {
    pub vs: Vec<u8>,
    pub prefilter_ps: Vec<u8>,
    pub downsample_ps: Vec<u8>,
    pub upsample_ps: Vec<u8>,
}

// Compile the bloom prefilter / downsample / upsample shaders.
pub(in crate::directx) fn compile_bloom_shaders(hot_reload: bool) -> Result<BloomShaders, String> {
    Ok(BloomShaders {
        vs: compile_hlsl(
            &shader_source(hot_reload, "composite_vert.hlsl", COMPOSITE_VERT_HLSL),
            "main",
            "vs_5_1",
        )?,
        prefilter_ps: compile_hlsl(
            &shader_source(hot_reload, "bloom_prefilter.hlsl", BLOOM_PREFILTER_HLSL),
            "main",
            "ps_5_1",
        )?,
        downsample_ps: compile_hlsl(
            &shader_source(hot_reload, "bloom_downsample.hlsl", BLOOM_DOWNSAMPLE_HLSL),
            "main",
            "ps_5_1",
        )?,
        upsample_ps: compile_hlsl(
            &shader_source(hot_reload, "bloom_upsample.hlsl", BLOOM_UPSAMPLE_HLSL),
            "main",
            "ps_5_1",
        )?,
    })
}

// Root signature + PSO

// Root signature for the bloom-chain passes: one SRV descriptor table at t0
// (the pass's source image), six 32-bit root constants at b0
// (`PostProcessParams`, read only by the prefilter), and a static linear-clamp
// sampler at s0. Shared by the prefilter, downsample, and upsample PSOs.
pub(in crate::directx) fn create_bloom_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // [0] Descriptor table: source image SRV (t0)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [1] Root constants: PostProcessParams (6 floats) at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 6,
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
    serialize_desc_and_create(device, &desc, "bloom root sig")
}

// PSO for a bloom-chain pass: a vertex-buffer-less fullscreen triangle that
// samples one source mip and writes an `HDR_FORMAT` bloom mip. No input
// layout, no depth. `additive` enables one-to-one additive blending, set for
// the upsample passes so each coarser mip accumulates onto the finer one.
pub(in crate::directx) fn create_bloom_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    additive: bool,
) -> Result<ID3D12PipelineState, String> {
    let blend_rt = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: additive.into(),
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ONE,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ONE,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        ..Default::default()
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
                arr[0] = blend_rt;
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };

    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create bloom PSO: {e}"))
}

// Targets

// Number of mip levels in the bloom chain for an HDR target of the given
// resolution. Clamped to 4..=6: enough octaves for a wide soft glow without
// spending a dozen render passes on sub-pixel mips. Mirrors `bloom_mip_count`
// in vulkan/texture.rs.
pub(in crate::directx) fn bloom_mip_count(width: u32, height: u32) -> u32 {
    let min_dim = width.min(height).max(1);
    // mip 0 is already half-res, so subtract one octave before clamping.
    let levels = (min_dim as f32).log2().floor() as i32 - 1;
    levels.clamp(4, 6) as u32
}

// Bloom mip chain: the mip render targets paired with their (width, height).
type BloomMips = (Vec<ID3D12Resource>, Vec<(u32, u32)>);

// Extent of bloom mip 0 (`bloom_top`): half the output resolution, floored at
// one texel. The transient pool sizes the placed `bloom_top` to this so it
// matches what the chain expects for `mips[0]`.
pub(in crate::directx) fn bloom_top_extent(width: u32, height: u32) -> (u32, u32) {
    ((width.max(1) >> 1).max(1), (height.max(1) >> 1).max(1))
}

// Create the bloom mip chain for an HDR target of `width`x`height`. `mips[i]`
// has resolution `(width >> (i+1), height >> (i+1))`, floored at one texel, so
// `mips[0]` is half-res. `mips[0]` (`bloom_top`) is the transient pool's placed
// resource passed in as `top` (so the graph can alias its memory); the finer
// mips are committed single-sample `HDR_FORMAT` colour targets usable as both a
// render target and a sampled texture, created in the PIXEL_SHADER_RESOURCE
// state so the composite pass can bind `mips[0]` even when bloom is disabled and
// the bloom passes never run.
pub(in crate::directx) fn create_bloom_mips(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    top: ID3D12Resource,
) -> Result<BloomMips, String> {
    let full_w = width.max(1);
    let full_h = height.max(1);
    let count = bloom_mip_count(full_w, full_h);
    create_bloom_mips_at(device, full_w, full_h, count as usize, top)
}

// Same shape as [`create_bloom_mips`], but with an explicit `count` so the
// resize handler can recreate the chain at the new resolution while keeping
// the SRV/RTV-heap-slot layout (which was sized for the init-time count)
// stable. The trailing mips fall to `1×1` once `(w >> i) < 1`, harmless,
// the bloom passes still sample them and the composite ignores them.
pub(in crate::directx) fn create_bloom_mips_at(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    count: usize,
    top: ID3D12Resource,
) -> Result<BloomMips, String> {
    let full_w = width.max(1);
    let full_h = height.max(1);
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: HDR_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
    };
    let mut mips = Vec::with_capacity(count);
    let mut extents = Vec::with_capacity(count);
    // mip 0 (`bloom_top`) is the pool-owned placed resource; the finer octaves
    // below stay committed.
    let (tw, th) = bloom_top_extent(full_w, full_h);
    mips.push(top);
    extents.push((tw, th));
    for i in 1..count {
        let mw = (full_w >> (i + 1)).max(1);
        let mh = (full_h >> (i + 1)).max(1);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: mw as u64,
            Height: mh,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: HDR_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut res_opt: Option<ID3D12Resource> = None;
        unsafe {
            device.CreateCommittedResource(
                &heap_props,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                Some(&clear_value),
                &mut res_opt,
            )
        }
        .map_err(|e| format!("create bloom mip {i}: {e}"))?;
        mips.push(res_opt.ok_or_else(|| format!("bloom mip {i} returned None"))?);
        extents.push((mw, mh));
    }
    Ok((mips, extents))
}

// Write an `HDR_FORMAT` single-sample Texture2D render-target view at the
// given heap slot, used for the bloom mips.
pub(in crate::directx) fn write_color_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    unsafe { device.CreateRenderTargetView(resource, Some(&rtv_desc), rtv_cpu) };
}

// Encoder

// The bloom chain orchestration lives once in `gfx::fullscreen`; this impl binds
// + draws each sub-pass in D3D12. `Args` is the scene-colour SRV the prefilter
// samples (post-TAA when TAA is on, the HDR scene SRV otherwise). Each sub-pass
// transitions its destination mip to RENDER_TARGET for the draw and back to
// PIXEL_SHADER_RESOURCE so the next pass (or composite) can sample it; every mip
// therefore ends the frame back in its created state.
impl crate::gfx::fullscreen::BloomEncoder for DxContext {
    type Rec = ID3D12GraphicsCommandList;
    type Args = D3D12_GPU_DESCRIPTOR_HANDLE;

    fn bloom_mip_count(&self) -> usize {
        self.bloom.mips.len()
    }

    fn begin_bloom(&self, cmd: &Self::Rec, _scene_srv: &Self::Args) {
        unsafe {
            cmd.SetGraphicsRootSignature(&self.bloom.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            // The bloom shaders build the fullscreen triangle from SV_VertexID.
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
        }
    }

    fn bloom_prefilter(&self, cmd: &Self::Rec, scene_srv: &Self::Args) {
        self.bloom_run_pass(cmd, 0, *scene_srv, &self.bloom.pso_prefilter);
    }

    fn bloom_downsample(&self, cmd: &Self::Rec, _scene_srv: &Self::Args, dst: usize) {
        self.bloom_run_pass(
            cmd,
            dst,
            self.bloom.mip_srv_gpus[dst - 1],
            &self.bloom.pso_downsample,
        );
    }

    fn bloom_upsample(&self, cmd: &Self::Rec, _scene_srv: &Self::Args, dst: usize) {
        self.bloom_run_pass(
            cmd,
            dst,
            self.bloom.mip_srv_gpus[dst + 1],
            &self.bloom.pso_upsample,
        );
    }
}

impl DxContext {
    // Encode the bloom prefilter, downsample, and additive upsample passes via
    // the shared `gfx::fullscreen` driver. On return `bloom_mips[0]` holds the
    // accumulated soft glow the composite pass samples. Called only when
    // `post_process.bloom_intensity > 0`, and after the HDR resolve (and the TAA
    // resolve, if any) so the prefilter can sample `scene_srv`.
    pub(in crate::directx) fn encode_bloom(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        crate::gfx::fullscreen::encode_bloom_chain(self, cmd, scene_srv);
    }

    // One fullscreen-triangle bloom sub-pass: sample `src_srv`, render into
    // bloom mip `dst` with `pso` bound, wrapped in the RT<->SRV barrier pair.
    fn bloom_run_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        dst: usize,
        src_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
        pso: &ID3D12PipelineState,
    ) {
        let (mw, mh) = self.bloom.mip_extents[dst];
        let post = self.post_process;
        let to_rt = transition_barrier(
            &self.bloom.mips[dst],
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        );
        unsafe { cmd.ResourceBarrier(&[to_rt]) };
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&self.bloom.mip_rtvs[dst]), false, None);
            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: mw as f32,
                Height: mh as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: mw as i32,
                bottom: mh as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
            cmd.SetPipelineState(pso);
            cmd.SetGraphicsRootDescriptorTable(0, src_srv);
            cmd.SetGraphicsRoot32BitConstants(
                1,
                6,
                &post as *const PostProcessParams as *const std::ffi::c_void,
                0,
            );
            cmd.DrawInstanced(3, 1, 0, 0);
        }
        let to_psr = transition_barrier(
            &self.bloom.mips[dst],
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[to_psr]) };
    }
}

#[cfg(test)]
mod tests {
    use super::bloom_mip_count;

    #[test]
    fn bloom_mip_count_clamps_to_four_to_six() {
        // Common HD resolutions land in the wide-glow sweet spot (6 octaves).
        assert_eq!(bloom_mip_count(1920, 1080), 6);
        assert_eq!(bloom_mip_count(1280, 720), 6);
        // Smaller resolutions earn fewer octaves before the clamp.
        assert_eq!(bloom_mip_count(64, 64), 5);
        // Floor: ridiculously small resolutions still get four octaves.
        assert_eq!(bloom_mip_count(16, 16), 4);
        assert_eq!(bloom_mip_count(1, 1), 4);
        assert_eq!(bloom_mip_count(0, 0), 4);
    }
}
