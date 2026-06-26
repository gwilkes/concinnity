// src/directx/quality.rs
//
// Runtime application of the Quality-group settings (TAA / SSAO / SSR / SSGI /
// auto-exposure). Each gates a render pass whose GPU resources (pipelines,
// render targets) are built once at init from the world's PostProcessConfig, so
// applying a change at runtime means building or tearing down those resources.
//
// Unlike the Vulkan backend, DirectX picks its scene-input + occlusion sources
// DYNAMICALLY every frame (`scene_srv_for_post`, `ssao_ao_srv_gpu`, and each
// feature's render-graph gate read the live `Option` state at encode time), so a
// toggle needs no descriptor rewrite and no swapchain rebuild: it just builds the
// effect into its pre-reserved fixed heap slot (the slots are reserved
// unconditionally at init, see `init/mod.rs`) or drops it. The next frame's draw
// adapts. The one coupling is SSAO's `ao_output`, which lives in the transient
// pool only while SSAO is on and shares a heap region with `bloom_top`; toggling
// it rebuilds the pool + the bloom mip chain, mirroring the resize path.
//
// Ray-traced reflections toggle the same way, with two extra costs: turning them
// on builds the scene acceleration structure (`build_rt_accel`, a one-shot
// fence-waited BLAS + TLAS over current geometry) plus the DXR reflection pass,
// so a live enable hitches once proportional to triangle count; and RT is only
// live-toggleable on a GPU that reports the DXR 1.1 tier (queried once at init
// into `rt_capable`), else the toggle no-ops with a warning and RT stays whatever
// it launched as. The RT output reserves its fixed heap slot unconditionally like
// the other five features, so an enable builds into it without shifting any other
// slot; the dynamic `scene_srv_for_post` picks the RT output over SSR each frame,
// so no rewire or rebuild is needed.

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::backend::QualitySettings;

use super::context::DxContext;
use super::post::bloom::{bloom_top_extent, create_bloom_mips_at, write_color_rtv};
use super::post::gbuffer::GbufferSlots;
use super::post::reflection_composite::ReflectionCompositeSlots;
use super::texture::write_hdr_srv;

// The fixed descriptor-heap slots the live-toggleable effects build into. Minted
// once at init (the slots are reserved unconditionally, independent of the
// init-time quality gates) and stashed on `DxContext` so a live toggle can
// construct a launched-off feature into its slot without re-deriving the heap
// layout. The handles are stable for the context's lifetime: a resize rewrites
// the resources behind them but never moves the slots.
#[derive(Clone, Copy)]
pub(in crate::directx) struct QualitySlotHandles {
    pub taa_history_rtv: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
    pub taa_history_srv: [(D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE); 2],
    pub ssao_ao_raw_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ssao_ao_raw_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub ssao_ao_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ssao_ao_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub ssr_output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ssr_output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub ssgi_gi_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub ssgi_gi_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub rt_output_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub rt_output_srv: (D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_GPU_DESCRIPTOR_HANDLE),
    pub refl_composite: ReflectionCompositeSlots,
    pub gbuffer: GbufferSlots,
}

impl DxContext {
    // Bring the toggle-controlled features to match `q`, applied between frames
    // (the GraphicsSystem reads the SettingCommand before the next draw_frame).
    // A build failure logs and leaves the prior state intact.
    pub(crate) fn apply_quality_settings(&mut self, q: QualitySettings) {
        if let Err(e) = self.apply_quality_settings_inner(q) {
            tracing::error!("apply_quality_settings: rebuild failed: {e}");
        }
    }

    fn apply_quality_settings_inner(&mut self, q: QualitySettings) -> Result<(), String> {
        // Every build / teardown below frees or replaces GPU resources a prior
        // frame may still reference; drain the device first so the swap is safe.
        self.wait_idle();

        let hot_reload = self.hot_reload.enabled;
        let render_w = self.render_width;
        let render_h = self.render_height;
        let slots = self.quality_slots;

        // RT is gated on the GPU reporting the DXR 1.1 tier: a non-DXR GPU cannot
        // build the acceleration structure, so the toggle no-ops with a warning
        // and RT stays whatever it launched as (persisted for the next launch).
        let desired_rt = q.rt_reflections.is_some() && self.rt_capable;
        if q.rt_reflections.is_some() && !self.rt_capable {
            tracing::warn!(
                "ray-traced reflections requested but the GPU does not report DXR \
                 tier 1.1; keeping SSR"
            );
        }

        let desired_ssr = q.ssr.is_some();
        let desired_ssgi = q.ssgi.is_some();
        let desired_ssao = q.ssao.is_some();
        let desired_ae = q.auto_exposure.is_some();
        // TAA resources are forced present while temporal upscaling is active
        // (the upscaler consumes the velocity pre-pass); the TAA resolve is then
        // dropped from the graph. Mirrors the init `taa_enabled` derivation.
        let upscale_on = self.upscale.backend.is_some();
        let desired_taa = q.taa || upscale_on;

        // The SSR pre-pass resources (`SsrResources`) exist whenever SSR, SSGI,
        // or RT is on (SSGI / RT reuse the SSR resolve's G-buffer plumbing); its
        // `resolve` half is built only when SSR itself is authored. The unified
        // G-buffer pre-pass is needed by any screen-space consumer (RT samples
        // its normal+depth and roughness). Mirrors the init `ssr_prepass_present`
        // / `gbuffer_enabled` gates.
        let ssr_needed = desired_ssr || desired_ssgi || desired_rt;
        let gbuffer_needed = ssr_needed || desired_ssao || desired_taa;

        // Unified G-buffer pre-pass (shared dependency): build it before any
        // consumer that samples it. Kept alive once built (a later toggle-off of
        // the last consumer leaves it resident until the next launch / resize),
        // which is harmless: with no consumer the graph omits its readers.
        if gbuffer_needed && self.gbuffer.is_none() {
            let gbuffer = super::post::gbuffer::GbufferResources::new(
                &self.device,
                render_w,
                render_h,
                self.n_clusters > 0,
                // Skinned variant builds lazily in `upload_skinned`, as at init.
                false,
                slots.gbuffer,
                self.info_queue.as_ref(),
                hot_reload,
            )?;
            self.gbuffer = Some(gbuffer);
        }

        // TAA.
        if desired_taa && self.taa.is_none() {
            let taa = super::post::taa::TaaResources::new(
                &self.device,
                render_w,
                render_h,
                slots.taa_history_rtv,
                slots.taa_history_srv,
                self.info_queue.as_ref(),
                hot_reload,
            )?;
            self.taa = Some(taa);
        } else if !desired_taa && self.taa.is_some() {
            self.taa = None;
        }

        // SSR pre-pass + resolve. `q.ssr` (Option) drives the resolve half:
        // `Some` builds it (SSR authored), `None` leaves it off (SSGI-only /
        // RT-only build the pre-pass but no resolve), matching init.
        if ssr_needed && self.ssr.is_none() {
            let ssr = super::post::ssr::SsrResources::new(
                &self.device,
                render_w,
                render_h,
                q.ssr,
                slots.ssr_output_rtv,
                slots.ssr_output_srv,
                self.info_queue.as_ref(),
                hot_reload,
            )?;
            self.ssr = Some(ssr);
        } else if !ssr_needed && self.ssr.is_some() {
            self.ssr = None;
        }

        // SSGI (samples the unified G-buffer; `gbuffer_needed` keeps it alive).
        if desired_ssgi && self.ssgi.is_none() {
            let settings = q.ssgi.expect("desired_ssgi implies ssgi settings");
            let ssgi = super::post::ssgi::SsgiResources::new(
                &self.device,
                render_w,
                render_h,
                settings,
                slots.ssgi_gi_rtv,
                slots.ssgi_gi_srv,
                self.info_queue.as_ref(),
                hot_reload,
            )?;
            self.ssgi = Some(ssgi);
        } else if !desired_ssgi && self.ssgi.is_some() {
            self.ssgi = None;
        }

        // Ray-traced reflections. Turning on builds the scene acceleration
        // structure (one-shot, fence-waited) + the DXR pass into the fixed RT
        // output slot; the unified G-buffer it samples is built above
        // (`gbuffer_needed` folds in `desired_rt`). Turning off drops both
        // (their COM resources release on drop); `scene_srv_for_post` falls back
        // to the SSR resolve / HDR dynamically next frame, so no rewire is needed.
        if desired_rt && self.rt_reflections.is_none() {
            self.build_rt_runtime(q.rt_reflections.expect("desired_rt implies settings"))?;
        } else if !desired_rt && self.rt_reflections.is_some() {
            self.rt_reflections = None;
            self.rt_accel = None;
        }

        // Reflection composite: the on-screen target the SSR/RT resolve writes its
        // radiance+weight into, then blurs/blends over the scene by roughness.
        // Present whenever a resolve can composite (SSR resolve or RT), mirroring
        // the init `ssr_settings || rt_reflection_settings` gate, and built into the
        // unconditionally-reserved refl_composite slots. Without this reconcile a
        // live RT/SSR enable on a world that authored neither leaves it `None`, so
        // `encode_reflection_composite` early-returns and the resolve's reflection
        // is computed but never shown (`scene_srv_for_post` / glass / the forward
        // `reflections_enabled` fade all gate on its presence).
        let refl_composite_needed = desired_ssr || desired_rt;
        if refl_composite_needed && self.reflection_composite.is_none() {
            let rc = super::post::reflection_composite::ReflectionCompositeResources::new(
                &self.device,
                render_w,
                render_h,
                q.reflection_blur_scale,
                slots.refl_composite,
                self.info_queue.as_ref(),
                hot_reload,
            )?;
            self.reflection_composite = Some(rc);
        } else if !refl_composite_needed && self.reflection_composite.is_some() {
            self.reflection_composite = None;
        }

        // Auto-exposure. Needs no descriptor-heap slots (own root UAVs +
        // readback ring). When it turns off the static authored EV drives
        // exposure again (the GraphicsSystem re-pushes `update_post_process`
        // after this call), so only the GPU + adaptation state is swapped here.
        if desired_ae && self.auto_exposure.resources.is_none() {
            let resources =
                super::auto_exposure::AutoExposureResources::new(&self.device, hot_reload)?;
            self.auto_exposure.resources = Some(resources);
            self.auto_exposure.state = q
                .auto_exposure
                .as_ref()
                .map(crate::gfx::auto_exposure::AutoExposureState::new);
            self.auto_exposure.settings = q.auto_exposure;
            self.auto_exposure.bias_ev = q.auto_exposure_bias_ev;
        } else if !desired_ae && self.auto_exposure.resources.is_some() {
            self.auto_exposure.resources = None;
            self.auto_exposure.settings = None;
            self.auto_exposure.state = None;
        }

        // SSAO. Its blurred `ao_output` is a transient-pool resource that only
        // exists while SSAO is on and shares a heap region with `bloom_top`, so a
        // toggle rebuilds the pool (relocating `bloom_top` -> rebuild the bloom
        // mip chain) and, on a turn-on, constructs the SSAO resources from the
        // freshly pooled `ao_output`. The main pass's occlusion binding
        // (`ssao_ao_srv_gpu`) falls back to the 1x1 white slot when off, so a
        // turn-off needs no rewire beyond dropping the resources.
        let ssao_was = self.ssao.resources.is_some();
        if desired_ssao && !ssao_was {
            self.rebuild_transient_pool_and_bloom(true)?;
            let ao_resource = self
                .transient_pool
                .resource_for("ao_output")
                .ok_or("transient pool missing ao_output after SSAO enable")?
                .clone();
            let settings = q.ssao.expect("desired_ssao implies ssao settings");
            let ssao = super::post::ssao::SsaoResources::new(
                &self.device,
                render_w,
                render_h,
                settings,
                slots.ssao_ao_raw_rtv,
                slots.ssao_ao_raw_srv,
                slots.ssao_ao_rtv,
                slots.ssao_ao_srv,
                &ao_resource,
                self.info_queue.as_ref(),
                hot_reload,
            )?;
            self.ssao.resources = Some(ssao);
        } else if !desired_ssao && ssao_was {
            // Drop before the pool rebuild removes the `ao_output` it points at.
            self.ssao.resources = None;
            self.rebuild_transient_pool_and_bloom(false)?;
        }

        Ok(())
    }

    // Build the RT acceleration structure + reflection pass at runtime (a live
    // toggle-on). Mirrors the init RT block: an empty scene, an AS-build error,
    // or a shader-compile failure leaves both `rt_accel` / `rt_reflections`
    // `None` and the renderer stays on SSR (a soft failure, returns `Ok`). The
    // caller has ensured the unified G-buffer pre-pass exists and drained the
    // device (`wait_idle`). The skinned BLAS is seeded on the first dynamic
    // frame, so the skin pipeline is attached here (a build failure is non-fatal:
    // static geometry still reflects, just without skinned hits).
    fn build_rt_runtime(
        &mut self,
        settings: crate::gfx::rt_reflections::RtReflectionSettings,
    ) -> Result<(), String> {
        let hot_reload = self.hot_reload.enabled;
        let mut accel = match super::raytrace::build_rt_accel(
            &self.device,
            &self.command_queue,
            &self.geometry.vertex_buffer,
            &self.geometry.index_buffer,
            &self.draw_objects,
            &self.instanced.clusters,
            self.rt_static_vertex_count,
            self.descriptors.textures.len() as u32,
            self.descriptors.normal_map_textures.len() as u32,
        ) {
            Ok(Some(accel)) => accel,
            Ok(None) => {
                tracing::info!(
                    "RT reflections enabled but no resident triangle geometry to trace; keeping SSR"
                );
                return Ok(());
            }
            Err(e) => {
                tracing::warn!("RT acceleration-structure build failed (keeping SSR): {e}");
                return Ok(());
            }
        };
        match super::raytrace::build_rt_skin_pipeline(&self.device, hot_reload) {
            Ok(skin) => accel.set_skin_pipeline(skin),
            Err(e) => tracing::warn!(
                "RT skin pipeline build failed (skinned meshes absent from reflections): {e}"
            ),
        }
        let slots = self.quality_slots;
        let rt = match super::post::rt_reflections::RtReflectionsResources::new(
            &self.device,
            self.render_width,
            self.render_height,
            settings,
            slots.rt_output_rtv,
            slots.rt_output_srv,
            self.info_queue.as_ref(),
            hot_reload,
        ) {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!("RT reflections pass build failed (keeping SSR): {e}");
                return Ok(());
            }
        };
        self.rt_accel = Some(accel);
        self.rt_reflections = Some(rt);
        Ok(())
    }

    // Rebuild the transient pool with the given SSAO gate (adds / removes
    // `ao_output`), then rebuild the bloom mip chain whose `mips[0]`
    // (`bloom_top`) is a pooled resource the pool rebuild relocates. The finer
    // mips are committed and unaffected, but the count is held fixed so the heap
    // layout past the bloom block stays anchored. Mirrors `handle_resize`'s
    // transient-pool + bloom steps (the device is idle when this is reached).
    fn rebuild_transient_pool_and_bloom(&mut self, ssao_on: bool) -> Result<(), String> {
        self.transient_pool.rebuild(
            &self.device,
            &self.command_queue,
            &super::transient_pool::transient_slots(
                ssao_on,
                (self.render_width, self.render_height),
                bloom_top_extent(self.output_width, self.output_height),
            ),
        )?;
        let bloom_count = self.bloom.mips.len();
        if bloom_count > 0 {
            let bloom_top = self
                .transient_pool
                .resource_for("bloom_top")
                .ok_or("transient pool missing bloom_top after rebuild")?
                .clone();
            let (mips, extents) = create_bloom_mips_at(
                &self.device,
                self.output_width,
                self.output_height,
                bloom_count,
                bloom_top,
            )?;
            self.bloom.mips = mips;
            self.bloom.mip_extents = extents;
            // Rewrite each mip's RTV + SRV into its existing (fixed) slot. The
            // SRV CPU handle is derived from the stored GPU handle the same way
            // the resize path does (the slot never moves).
            let srv_cpu_base = unsafe {
                self.descriptors
                    .srv_heap
                    .GetCPUDescriptorHandleForHeapStart()
            };
            let srv_gpu_base = unsafe {
                self.descriptors
                    .srv_heap
                    .GetGPUDescriptorHandleForHeapStart()
            };
            let srv_cpu_of = |gpu: D3D12_GPU_DESCRIPTOR_HANDLE| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: srv_cpu_base.ptr + (gpu.ptr - srv_gpu_base.ptr) as usize,
            };
            for i in 0..bloom_count {
                write_color_rtv(&self.device, &self.bloom.mips[i], self.bloom.mip_rtvs[i]);
                write_hdr_srv(
                    &self.device,
                    &self.bloom.mips[i],
                    srv_cpu_of(self.bloom.mip_srv_gpus[i]),
                );
            }
        }
        Ok(())
    }
}
