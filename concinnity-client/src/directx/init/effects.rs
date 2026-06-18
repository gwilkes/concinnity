// src/directx/init/effects.rs
//
// Post-process pipeline + target construction extracted from DxContext::new:
// bloom (always built), TAA history-resolve (gated on `taa_enabled`), and
// SSAO (gated on `ssao_settings.is_some()`). Each gated block pays zero cost
// when its setting is off.

use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::dump_on_err;
use crate::directx::post::bloom::{
    bloom_top_extent, compile_bloom_shaders, create_bloom_mips, create_bloom_pso,
    create_bloom_root_signature, write_color_rtv,
};
use crate::directx::post::rt_reflections::RtReflectionsResources;
use crate::directx::post::ssao::SsaoResources;
use crate::directx::post::ssgi::SsgiResources;
use crate::directx::post::ssr::SsrResources;
use crate::directx::post::taa::TaaResources;
use crate::directx::texture::{
    HDR_FORMAT, create_fallback_white_resource, write_hdr_srv, write_rgba8_srv,
};
use crate::directx::transient_pool::{TransientResourcePool, transient_slots};

pub(super) struct EffectsBundle {
    // Graph-owned transient render targets (the resources the aliasing planner
    // manages: `bloom_top` always, `ao_output` when SSAO is on). Built first so
    // the bloom chain and SSAO read their placed resources back by label.
    pub transient_pool: TransientResourcePool,
    pub bloom_mips: Vec<ID3D12Resource>,
    pub bloom_mip_rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub bloom_mip_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
    pub bloom_mip_extents: Vec<(u32, u32)>,
    pub bloom_root_sig: ID3D12RootSignature,
    pub bloom_pso_prefilter: ID3D12PipelineState,
    pub bloom_pso_downsample: ID3D12PipelineState,
    pub bloom_pso_upsample: ID3D12PipelineState,
    pub taa: Option<TaaResources>,
    pub ssao: Option<SsaoResources>,
    pub ssao_white: ID3D12Resource,
    pub ssao_white_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub ssr: Option<SsrResources>,
    pub ssgi: Option<SsgiResources>,
    // Built only when RT reflections are authored AND the GPU supports the DXR
    // tier AND the DXC compile succeeds; `None` otherwise (the graph then falls
    // back to SSR). The acceleration structure is built separately in mod.rs.
    pub rt_reflections: Option<RtReflectionsResources>,
}

// Slot handles for the bloom mip chain. The caller knows the descriptor-heap
// layout (RTV base, SRV slot base for the first mip) and supplies a closure
// per kind to mint per-mip handles. This keeps the heap layout in mod.rs.
pub(super) struct BloomSlots<'a> {
    pub rtv_for: &'a dyn Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_cpu_for: &'a dyn Fn(usize) -> D3D12_CPU_DESCRIPTOR_HANDLE,
    pub srv_gpu_for: &'a dyn Fn(usize) -> D3D12_GPU_DESCRIPTOR_HANDLE,
}

pub(super) struct TaaSlots {
    pub history_rtv: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
    pub history_srv: [(D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE); 2],
}

pub(super) struct SsaoSlots {
    pub ao_raw_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ao_raw_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub ao_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ao_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub white_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
}

pub(super) struct SsrSlots {
    pub output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
}

pub(super) struct SsgiSlots {
    pub gi_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub gi_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
}

pub(super) struct RtReflectionsSlots {
    pub output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_effects(
    device: &ID3D12Device,
    command_queue: &ID3D12CommandQueue,
    info_queue: Option<&ID3D12InfoQueue>,
    // Drawable resolution; sizes the bloom mip chain (which samples the
    // upscaler's output-res result / composite-bound scene).
    width: u32,
    height: u32,
    // Off-screen scene render resolution; sizes the TAA velocity + history,
    // SSAO, and SSR targets, which all live upstream of the upscaler. Equals
    // `width`/`height` when temporal upscaling is off.
    render_width: u32,
    render_height: u32,
    taa_enabled: bool,
    ssao_settings: Option<crate::gfx::ssao::SsaoSettings>,
    ssr_settings: Option<crate::gfx::ssr::SsrSettings>,
    ssgi_settings: Option<crate::gfx::ssgi::SsgiSettings>,
    // RT-reflection tunables (`Some` only when the world authored
    // `ray_traced_reflections`); `rt_supported` reports the DXR-tier capability
    // queried at init. The resources build only when both hold and the DXC
    // compile succeeds; any failure leaves `rt_reflections` `None` (SSR fallback).
    rt_reflection_settings: Option<crate::gfx::rt_reflections::RtReflectionSettings>,
    rt_supported: bool,
    bloom: BloomSlots<'_>,
    taa_slots: TaaSlots,
    ssao_slots: SsaoSlots,
    ssr_slots: SsrSlots,
    ssgi_slots: SsgiSlots,
    rt_slots: RtReflectionsSlots,
    hot_reload: bool,
) -> Result<EffectsBundle, String> {
    // Transient pool: the graph-owned transient render targets. `bloom_top`
    // (bloom mip 0) is always managed; `ao_output` is placed only when SSAO is
    // on (else `resource_for` returns None and the main pass binding 6 falls
    // back to `ssao_white`). Built first so the bloom chain and SSAO read their
    // placed resources back by label.
    let transient_pool = TransientResourcePool::build(
        device,
        command_queue,
        &transient_slots(
            ssao_settings.is_some(),
            (render_width, render_height),
            bloom_top_extent(width, height),
        ),
    )?;

    // Bloom mip chain + per-mip RTV/SRV writes. `mips[0]` (`bloom_top`) is the
    // pool's placed resource; the finer mips are committed.
    let bloom_top = transient_pool
        .resource_for("bloom_top")
        .ok_or("transient pool missing bloom_top")?
        .clone();
    let (bloom_mips, bloom_mip_extents) = create_bloom_mips(device, width, height, bloom_top)?;
    let mut bloom_mip_rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE> = Vec::with_capacity(bloom_mips.len());
    let mut bloom_mip_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE> =
        Vec::with_capacity(bloom_mips.len());
    for (i, mip) in bloom_mips.iter().enumerate() {
        let rtv = (bloom.rtv_for)(i);
        write_color_rtv(device, mip, rtv);
        bloom_mip_rtvs.push(rtv);
        let srv_cpu = (bloom.srv_cpu_for)(i);
        write_hdr_srv(device, mip, srv_cpu);
        bloom_mip_srv_gpus.push((bloom.srv_gpu_for)(i));
    }

    // Bloom pipelines (prefilter / downsample / upsample). All three write
    // HDR_FORMAT mips; the upsample blends additively so each coarser mip
    // accumulates onto the finer one.
    let bloom_root_sig = dump_on_err(info_queue, create_bloom_root_signature(device))?;
    let bloom_shaders = compile_bloom_shaders(hot_reload)?;
    let bloom_pso_prefilter = dump_on_err(
        info_queue,
        create_bloom_pso(
            device,
            &bloom_root_sig,
            &bloom_shaders.vs,
            &bloom_shaders.prefilter_ps,
            HDR_FORMAT,
            false,
        ),
    )?;
    let bloom_pso_downsample = dump_on_err(
        info_queue,
        create_bloom_pso(
            device,
            &bloom_root_sig,
            &bloom_shaders.vs,
            &bloom_shaders.downsample_ps,
            HDR_FORMAT,
            false,
        ),
    )?;
    let bloom_pso_upsample = dump_on_err(
        info_queue,
        create_bloom_pso(
            device,
            &bloom_root_sig,
            &bloom_shaders.vs,
            &bloom_shaders.upsample_ps,
            HDR_FORMAT,
            true,
        ),
    )?;

    // TAA history-resolve resources. Sized at render-res; the motion it
    // reprojects through comes from the unified G-buffer pre-pass.
    let taa = if taa_enabled {
        Some(TaaResources::new(
            device,
            render_width,
            render_height,
            taa_slots.history_rtv,
            taa_slots.history_srv,
            info_queue,
            hot_reload,
        )?)
    } else {
        None
    };

    // SSAO: 1x1 white fallback always populated so the main pass binds a
    // pass-through occlusion when SSAO is off. The real SSAO targets sit in
    // the three slots before it.
    let ssao_white = create_fallback_white_resource(device, command_queue)?;
    write_rgba8_srv(device, &ssao_white, ssao_slots.white_srv.0);
    let ssao_white_srv_gpu = ssao_slots.white_srv.1;
    let ssao = if let Some(settings) = ssao_settings {
        let ao_resource = transient_pool
            .resource_for("ao_output")
            .ok_or("transient pool missing ao_output while SSAO is enabled")?;
        Some(SsaoResources::new(
            device,
            render_width,
            render_height,
            settings,
            ssao_slots.ao_raw_rtv,
            ssao_slots.ao_raw_srv,
            ssao_slots.ao_rtv,
            ssao_slots.ao_srv,
            ao_resource,
            info_queue,
            hot_reload,
        )?)
    } else {
        None
    };

    // SSR: a fullscreen resolve reading the unified G-buffer pre-pass. The
    // resources are built whenever SSR *or* SSGI *or* RT reflections are on (all
    // reuse the G-buffer); `ssr_settings` (the resolve half) stays `None` for a
    // SSGI-only or RT-only build, which leaves the reserved output slot unwritten.
    let ssr =
        if ssr_settings.is_some() || ssgi_settings.is_some() || rt_reflection_settings.is_some() {
            Some(SsrResources::new(
                device,
                render_width,
                render_height,
                ssr_settings,
                ssr_slots.output_rtv,
                ssr_slots.output_srv,
                info_queue,
                hot_reload,
            )?)
        } else {
            None
        };

    // SSGI: hemisphere-gather + depth-aware blur over the unified G-buffer
    // pre-pass. The gather target lives here; the composite blends straight
    // into the scene.
    let ssgi = if let Some(settings) = ssgi_settings {
        Some(SsgiResources::new(
            device,
            render_width,
            render_height,
            settings,
            ssgi_slots.gi_rtv,
            ssgi_slots.gi_srv,
            info_queue,
            hot_reload,
        )?)
    } else {
        None
    };

    // RT reflections: build the pipelines + output target only when authored AND
    // the GPU supports the DXR tier. The DXC compile can still fail (DXC DLL
    // absent / shader error); that is non-fatal -> leave `None` and the graph
    // falls back to SSR (whose resolve is also off in an RT-only world, so the
    // scene simply renders without reflections). The acceleration structure is
    // built separately in mod.rs and gates `rt_reflections_active` alongside this.
    let rt_reflections = match (rt_reflection_settings, rt_supported) {
        (Some(settings), true) => match RtReflectionsResources::new(
            device,
            render_width,
            render_height,
            settings,
            rt_slots.output_rtv,
            rt_slots.output_srv,
            info_queue,
            hot_reload,
        ) {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::warn!("RT reflections unavailable, falling back to SSR: {e}");
                None
            }
        },
        _ => None,
    };

    Ok(EffectsBundle {
        transient_pool,
        bloom_mips,
        bloom_mip_rtvs,
        bloom_mip_srv_gpus,
        bloom_mip_extents,
        bloom_root_sig,
        bloom_pso_prefilter,
        bloom_pso_downsample,
        bloom_pso_upsample,
        taa,
        ssao,
        ssao_white,
        ssao_white_srv_gpu,
        ssr,
        ssgi,
        rt_reflections,
    })
}
