// src/directx/post/taa.rs
//
// Temporal anti-aliasing for the D3D12 backend: the resolve pass that
// reprojects the accumulated history through the unified G-buffer pre-pass's
// motion buffer (post/gbuffer.rs) and neighbourhood-clips it. Owns the
// ping-pong history targets, the resolve pipeline, the CPU-side temporal state,
// and the `encode_taa` per-frame encoder.
//
// Mirrors src/metal/post/taa.rs.

use std::cell::Cell;

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::fullscreen::{FullscreenPass, encode_fullscreen};

use crate::directx::context::{DxContext, dump_on_err};
use crate::directx::pipeline::{
    COMPOSITE_VERT_HLSL, compile_hlsl, create_composite_pso, serialize_desc_and_create,
    shader_source,
};
use crate::directx::post::gbuffer::GbufferResources;
use crate::directx::texture::{HDR_FORMAT, create_rt_target, write_format_rtv, write_hdr_srv};

// HLSL sources

// TAA resolve fragment: reproject the accumulated history through the motion
// buffer, clip it to the current 3x3 neighbourhood in YCoCg, and blend.
// Mirrors TAA_FRAG_GLSL in vulkan/pipeline.rs.
pub const TAA_FRAG_HLSL: &str = include_str!("../shaders/taa_frag.hlsl");

// Shader compilation

// Compiled TAA resolve shader bytecode. The resolve pass reuses the
// fullscreen-triangle `COMPOSITE_VERT_HLSL`; the motion it reprojects through
// comes from the unified G-buffer pre-pass.
struct TaaShaders {
    resolve_vs: Vec<u8>,
    resolve_ps: Vec<u8>,
}

// Compile the TAA resolve shaders.
fn compile_taa_shaders(hot_reload: bool) -> Result<TaaShaders, String> {
    Ok(TaaShaders {
        resolve_vs: compile_hlsl(
            &shader_source(hot_reload, "composite_vert.hlsl", COMPOSITE_VERT_HLSL),
            "main",
            "vs_5_1",
        )?,
        resolve_ps: compile_hlsl(
            &shader_source(hot_reload, "taa_frag.hlsl", TAA_FRAG_HLSL),
            "main",
            "ps_5_1",
        )?,
    })
}

// Root signature

// Root signature for the TAA resolve pass: three 1-SRV descriptor tables at
// t0 (current scene), t1 (velocity), t2 (history), each its own table since
// the history SRV alternates between two ping-pong heap slots, plus one
// 32-bit root constant at b0 (`history_valid`) and a static linear-clamp
// sampler at s0.
fn create_taa_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let make_range = |reg: u32| D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: reg,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let scene_range = make_range(0);
    let velocity_range = make_range(1);
    let history_range = make_range(2);
    let params = [
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
                    pDescriptorRanges: &velocity_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &history_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        // [3] Root constant: history_valid at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 1,
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
    serialize_desc_and_create(device, &desc, "taa root sig")
}

// Resources

// Temporal anti-aliasing resources. Owns the ping-pong history images, the
// resolve pipeline, and the CPU-side frame counter (jitter sequence +
// history-validity gate). `record_frame` is `&self`, so `frame` is held behind
// a `Cell`. The motion buffer the resolve reprojects through comes from the
// unified G-buffer pre-pass (post/gbuffer.rs).
pub(in crate::directx) struct TaaResources {
    // Ping-pong history / output images. Frame N writes `history[N % 2]` and
    // samples `history[1 - N % 2]` as the accumulated history; bloom + the
    // composite then sample `history[N % 2]`. Two images because a same-image
    // read+write in one pass is illegal.
    pub(in crate::directx) history: [ID3D12Resource; 2],
    pub(in crate::directx) history_rtv: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
    pub(in crate::directx) history_srv_gpu: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
    pub(in crate::directx) taa_root_sig: ID3D12RootSignature,
    pub(in crate::directx) taa_pso: ID3D12PipelineState,
    // CPU temporal state. `frame` drives the Halton jitter sequence and gates
    // history validity.
    pub(in crate::directx) frame: Cell<u32>,
}

#[allow(clippy::too_many_arguments)]
impl TaaResources {
    // Build the TAA history + resolve resources. The two `history_*` handles are
    // the descriptor-heap slots reserved for TAA after the bloom mips + LUT.
    pub(in crate::directx) fn new(
        device: &ID3D12Device,
        width: u32,
        height: u32,
        history_rtv: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        history_srv: [(D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE); 2],
        info_queue: Option<&ID3D12InfoQueue>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        // Ping-pong HDR history images.
        let history = [
            create_rt_target(device, width, height, HDR_FORMAT)?,
            create_rt_target(device, width, height, HDR_FORMAT)?,
        ];
        for i in 0..2 {
            write_hdr_srv(device, &history[i], history_srv[i].0);
            write_format_rtv(device, &history[i], history_rtv[i], HDR_FORMAT);
        }

        // Pipelines.
        let shaders = compile_taa_shaders(hot_reload)?;
        let taa_root_sig = dump_on_err(info_queue, create_taa_root_signature(device))?;
        // The resolve pass is a fullscreen-triangle tonemap-shaped pass writing
        // an HDR_FORMAT target, same PSO shape as the composite.
        let taa_pso = dump_on_err(
            info_queue,
            create_composite_pso(
                device,
                &taa_root_sig,
                &shaders.resolve_vs,
                &shaders.resolve_ps,
                HDR_FORMAT,
            ),
        )?;

        Ok(Self {
            history,
            history_rtv,
            history_srv_gpu: [history_srv[0].1, history_srv[1].1],
            taa_root_sig,
            taa_pso,
            frame: Cell::new(0),
        })
    }

    // Ping-pong slot this frame writes; the other slot is the history it
    // samples. `bloom` + the composite read this slot's image afterwards.
    pub(in crate::directx) fn output_index(&self) -> usize {
        (self.frame.get() % 2) as usize
    }

    // Rebuild the ping-pong history resources at a new resolution. The
    // descriptor *slots* (history_rtv[] / history_srv_gpu[]) stay where they
    // were; only the resources backing them change, so the live composite /
    // bloom-prefilter bindings keep working without a re-bind. Resets `frame`
    // to zero so the resolve pass treats the next frame as the
    // first-after-resize (history is unreliable across a resize: the
    // reprojection coordinates were generated at the old resolution).
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

        // Ping-pong HDR history.
        self.history = [
            create_rt_target(device, width, height, HDR_FORMAT)?,
            create_rt_target(device, width, height, HDR_FORMAT)?,
        ];
        for i in 0..2 {
            write_hdr_srv(device, &self.history[i], srv_cpu(self.history_srv_gpu[i]));
            write_format_rtv(device, &self.history[i], self.history_rtv[i], HDR_FORMAT);
        }

        // History is invalid post-resize. Reset the frame counter so the
        // resolve pass takes the "first frame" branch and seeds history from
        // the current scene instead of reprojecting through stale motion.
        self.frame.set(0);

        Ok(())
    }
}

// Replacement TAA resolve PSO returned by [`rebuild_taa_pipelines`]. The caller
// swaps it in atomically only if the build succeeded.
pub(in crate::directx) struct RebuiltTaaPipelines {
    pub taa_pso: ID3D12PipelineState,
}

// Rebuild the TAA resolve PSO against fresh shader source. Reuses the existing
// root signature; returns the new PSO for the caller to swap into the live
// `TaaResources`.
pub(in crate::directx) fn rebuild_taa_pipelines(
    device: &ID3D12Device,
    taa: &TaaResources,
    hot_reload: bool,
    info_queue: Option<&ID3D12InfoQueue>,
) -> Result<RebuiltTaaPipelines, String> {
    let shaders = compile_taa_shaders(hot_reload)?;
    let taa_pso = dump_on_err(
        info_queue,
        create_composite_pso(
            device,
            &taa.taa_root_sig,
            &shaders.resolve_vs,
            &shaders.resolve_ps,
            HDR_FORMAT,
        ),
    )?;
    Ok(RebuiltTaaPipelines { taa_pso })
}

// Encoders

impl DxContext {
    // Encode the TAA history-resolve pass: a fullscreen triangle that
    // reprojects the accumulated history through the motion buffer, clips it
    // to the current frame's neighbourhood, and blends. Writes the
    // `history[frame % 2]` ping-pong slot, reading the other slot as history.
    // Called only when `self.taa` is `Some`, after the unified G-buffer
    // pre-pass.
    pub(in crate::directx) fn encode_taa(&self, cmd: &ID3D12GraphicsCommandList) {
        // Motion comes from the unified G-buffer pre-pass, which is always built
        // when TAA is on. Resolving the resources up front, before the encoder,
        // keeps the driver from ever leaving the barrier bracket half-open. The
        // write slot is read now (it is stable across the encode: `taa.frame`
        // only ticks after Composite).
        let Some(taa) = &self.taa else { return };
        let Some(gbuffer) = &self.gbuffer else { return };
        let cur = taa.output_index();
        encode_fullscreen(
            &TaaResolvePass {
                ctx: self,
                taa,
                gbuffer,
                cur,
            },
            cmd,
        );
    }
}

// Encoder for the TAA resolve fullscreen pass: the resolved resources + the
// ping-pong write slot. The PSR<->RENDER_TARGET barrier bracket + render-target
// bind live in `DxContext::begin/end_fullscreen_rt` (post/fullscreen.rs); only
// the TAA-specific bind + draw is here. Constructed + driven by `encode_taa`
// through `gfx::fullscreen::encode_fullscreen`.
struct TaaResolvePass<'a> {
    ctx: &'a DxContext,
    taa: &'a TaaResources,
    gbuffer: &'a GbufferResources,
    cur: usize,
}

impl FullscreenPass for TaaResolvePass<'_> {
    type Rec = ID3D12GraphicsCommandList;

    fn begin(&self, cmd: &Self::Rec) {
        self.ctx.begin_fullscreen_rt(
            cmd,
            &self.taa.history[self.cur],
            self.taa.history_rtv[self.cur],
        );
    }

    fn draw(&self, cmd: &Self::Rec) {
        let hist = 1 - self.cur;
        // History is invalid on the first frame; the scene then passes straight
        // through.
        let history_valid: f32 = if self.taa.frame.get() > 0 { 1.0 } else { 0.0 };
        unsafe {
            cmd.SetPipelineState(&self.taa.taa_pso);
            cmd.SetGraphicsRootSignature(&self.taa.taa_root_sig);
            // [0] scene (the SSR output when SSR is on, else the resolved HDR
            // target), [1] velocity, [2] history.
            cmd.SetGraphicsRootDescriptorTable(0, self.ctx.scene_srv_for_post());
            cmd.SetGraphicsRootDescriptorTable(1, self.gbuffer.velocity_srv_gpu);
            cmd.SetGraphicsRootDescriptorTable(2, self.taa.history_srv_gpu[hist]);
            // [3] history_valid root constant.
            cmd.SetGraphicsRoot32BitConstants(
                3,
                1,
                &history_valid as *const f32 as *const std::ffi::c_void,
                0,
            );
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            // The resolve VS builds the fullscreen triangle from SV_VertexID.
            cmd.IASetVertexBuffers(0, None);
            cmd.IASetIndexBuffer(None);
            cmd.DrawInstanced(3, 1, 0, 0);
        }
    }

    fn end(&self, cmd: &Self::Rec) {
        self.ctx.end_fullscreen_rt(cmd, &self.taa.history[self.cur]);
    }
}
