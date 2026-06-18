// src/directx/hot_reload.rs
//
// Filesystem watcher driving D3D12 shader hot-reload. A background notify
// watcher tails `<CARGO_MANIFEST_DIR>/src/directx/shaders/` and, on any modify
// event for a `.hlsl` file, flips a shared `Arc<AtomicBool>`. The main thread
// polls that flag at the top of `draw_frame` and calls
// `DxContext::reload_shaders` when it's set. Same flag is also set by the
// `reload-shaders` debug WebSocket command, so the two trigger paths converge.
//
// Entirely a dev-loop concern; only constructed when `DxContext::new` is
// called with `hot_reload = true`. Production `cn run` never instantiates it.
// Mirrors src/metal/hot_reload.rs.

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::context::DxContext;

// Rebuild a feature's PSO(s) into a temporary only when the feature is live,
// propagating any compile/create error out of the enclosing `reload_shaders`.
// `$cond` is the liveness check (`self.x.is_some()`); `$build` is the build
// expression (which re-accesses `self.x` and may use `?` internally for the
// shader compile). Expands to `Some(build?)` when live, `None` otherwise, so
// the swap phase below can `if let (Some(rebuilt), Some(x)) = ...` uniformly.
// Mirrors `metal/hot_reload.rs::rebuild_if_live!`.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// Live watcher handle. Held by `DxContext` purely to keep the watcher
// thread alive; dropping it stops the watcher. The flag itself is shared
// via [`DxContext::shader_reload_pending`].
pub(crate) struct WatcherHandle {
    // notify keeps its own listener thread alive for as long as the handle
    // exists; we never read this field after construction.
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    // The shader source directory the watcher is observing. Kept for
    // diagnostics; log lines reference it on init.
    #[allow(dead_code)]
    watched_dir: PathBuf,
}

// Spawn a `notify` watcher over the D3D12 shader source directory and wire
// it to flip `flag` on any `.hlsl` file modify event. The path is derived
// from `CARGO_MANIFEST_DIR` at compile time so the watcher works no matter
// where the binary is launched from, but only as long as the source tree
// still exists at that path. A shipped binary should never be hot-reload-
// enabled, so the missing-path case logs and returns `None` instead of
// failing the whole context init.
pub(crate) fn spawn(flag: Arc<AtomicBool>) -> Option<WatcherHandle> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("directx")
        .join("shaders");
    if !dir.is_dir() {
        tracing::warn!(
            "hot-reload: shader source dir {} not found; watcher disabled (debug \
             command still works)",
            dir.display()
        );
        return None;
    }

    // Suppress event bursts: editors (vim, VSCode) frequently emit several
    // close-write / rename events per save. Coalesce by a small debounce so
    // one save triggers exactly one reload.
    let debounce = Duration::from_millis(150);
    let last_fire = std::sync::Mutex::new(Instant::now() - debounce);
    let flag_for_cb = Arc::clone(&flag);
    let mut watcher = match notify::recommended_watcher(move |res: notify::Result<Event>| {
        let event = match res {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("hot-reload watcher error: {e}");
                return;
            }
        };
        if !is_relevant(&event) {
            return;
        }
        let mut last = match last_fire.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        if now.duration_since(*last) < debounce {
            return;
        }
        *last = now;
        tracing::info!(
            "hot-reload: detected change to {:?}, scheduling shader rebuild",
            event.paths
        );
        flag_for_cb.store(true, Ordering::SeqCst);
    }) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("hot-reload: failed to create notify watcher: {e}");
            return None;
        }
    };

    if let Err(e) = watcher.watch(&dir, RecursiveMode::NonRecursive) {
        tracing::warn!(
            "hot-reload: failed to watch {} ({}); watcher disabled",
            dir.display(),
            e
        );
        return None;
    }

    tracing::info!("hot-reload: watching {} for .hlsl changes", dir.display());
    Some(WatcherHandle {
        watcher,
        watched_dir: dir,
    })
}

// True when this notify event is a modify of a `.hlsl` file we care about.
// Filters out unrelated paths (e.g. swap files, sub-directory churn) and the
// non-mutating events notify emits (e.g. access/metadata).
fn is_relevant(event: &Event) -> bool {
    if !matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    event
        .paths
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "hlsl"))
}

impl DxContext {
    // True when the shared shader-reload flag is set. Cheap atomic load;
    // called at the top of `draw_frame`. Returns false when hot-reload is
    // off so the production path never enters the reload branch.
    pub(super) fn shader_reload_requested(&self) -> bool {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders`
    // regardless of outcome so a failed rebuild does not loop forever.
    pub(super) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.hot_reload.reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in D3D12 PSO from disk-resident source. Each PSO
    // is constructed into a temporary first; only when every rebuild
    // succeeds does the context atomically swap them in. Any HLSL compile
    // or PSO-create failure logs the underlying message and leaves the
    // live pipelines untouched; a typo in a shader edit won't crash the
    // running session.
    //
    // Covers every runtime-bundled PSO: composite, text, bloom (prefilter
    // / downsample / upsample), GPU-cull compute, auto-exposure (build +
    // average), projected-decal, glass (transparent), volumetric-fog, the
    // unified G-buffer pre-pass (static / instanced / skinned), SSAO (kernel,
    // blur), SSR (resolve), TAA (resolve), and the bindless main pipeline when
    // it is live. The world-loaded
    // main / shadow / instanced pipelines (`main_pso`, `shadow_pso`,
    // `main_instanced_pso`) and the skinned-main / skinned-shadow PSOs
    // are out of scope here; their fragment payload comes from the
    // world's `ShaderStage` library and is reloaded through a separate
    // `update_world_shader_pipelines` path (not yet implemented on
    // DirectX).
    pub(super) fn reload_shaders(&mut self) -> Result<(), String> {
        if !self.hot_reload.enabled {
            return Ok(());
        }
        let device = &self.device;
        let info_queue = self.info_queue.as_ref();
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.

        // Composite (always live).
        let (composite_vs, composite_ps) = super::pipeline::compile_composite_shaders(hr)?;
        let composite_pso = super::context::dump_on_err(
            info_queue,
            super::pipeline::create_composite_pso(
                device,
                &self.composite_root_sig,
                &composite_vs,
                &composite_ps,
                self.swap_format,
            ),
        )?;

        // Text (only when the world declared text atlases).
        let text_pso = rebuild_if_live!(self.text_pso.is_some(), {
            let (text_vs, text_ps) = super::pipeline::compile_text_shaders(hr)?;
            super::context::dump_on_err(
                info_queue,
                super::pipeline::create_text_pso(
                    device,
                    &self.text_root_sig,
                    &text_vs,
                    &text_ps,
                    self.swap_format,
                    1,
                ),
            )
        });

        // Bloom (always live).
        let bloom_shaders = super::post::bloom::compile_bloom_shaders(hr)?;
        let bloom_prefilter = super::context::dump_on_err(
            info_queue,
            super::post::bloom::create_bloom_pso(
                device,
                &self.bloom.root_sig,
                &bloom_shaders.vs,
                &bloom_shaders.prefilter_ps,
                super::texture::HDR_FORMAT,
                false,
            ),
        )?;
        let bloom_downsample = super::context::dump_on_err(
            info_queue,
            super::post::bloom::create_bloom_pso(
                device,
                &self.bloom.root_sig,
                &bloom_shaders.vs,
                &bloom_shaders.downsample_ps,
                super::texture::HDR_FORMAT,
                false,
            ),
        )?;
        let bloom_upsample = super::context::dump_on_err(
            info_queue,
            super::post::bloom::create_bloom_pso(
                device,
                &self.bloom.root_sig,
                &bloom_shaders.vs,
                &bloom_shaders.upsample_ps,
                super::texture::HDR_FORMAT,
                true,
            ),
        )?;

        // Bindless main + cull (when the world drives the built-in shader).
        let bindless_main_pso = rebuild_if_live!(
            self.cull.main_bindless_root_sig.is_some() && self.cull.main_bindless_pso.is_some(),
            {
                let (bvs, bps) = super::init::pipelines::compile_main_bindless_shaders(hr)?;
                super::context::dump_on_err(
                    info_queue,
                    super::init::pipelines::create_main_pso(
                        device,
                        self.cull.main_bindless_root_sig.as_ref().unwrap(),
                        &bvs,
                        &bps,
                        super::texture::HDR_FORMAT,
                        self.hdr.msaa_samples,
                    ),
                )
            }
        );
        let cull_pso = rebuild_if_live!(
            self.cull.cull_root_sig.is_some() && self.cull.cull_pso.is_some(),
            {
                let cs = super::cull::compile_cull_shader(hr)?;
                super::context::dump_on_err(
                    info_queue,
                    super::cull::create_cull_pso(
                        device,
                        self.cull.cull_root_sig.as_ref().unwrap(),
                        &cs,
                    ),
                )
            }
        );
        // Phase-2 cull PSO (two-pass occlusion), rebuilt against the same root
        // signature when it's live.
        let cull_pso_phase2 = rebuild_if_live!(
            self.cull.cull_root_sig.is_some() && self.cull.cull_pso_phase2.is_some(),
            {
                let cs2 = super::cull::compile_cull_shader_phase2(hr)?;
                super::context::dump_on_err(
                    info_queue,
                    super::cull::create_cull_pso(
                        device,
                        self.cull.cull_root_sig.as_ref().unwrap(),
                        &cs2,
                    ),
                )
            }
        );

        // Hi-Z (only when cull pipeline is live; same gating condition).
        // Rebuilds all three compute kernels (init_single, init_msaa,
        // downsample) against the existing root signature so the live cull
        // root binding stays valid.
        let hiz_rebuilt = if let Some(hiz) = self.cull.hiz.as_ref() {
            let (init_single_cs, init_msaa_cs, downsample_cs) =
                super::hiz::compile_hiz_shaders(hr)?;
            let init_single_pso = super::context::dump_on_err(
                info_queue,
                super::auto_exposure::create_compute_pso(
                    device,
                    &hiz.root_sig,
                    &init_single_cs,
                    "hiz init_single",
                ),
            )?;
            let init_msaa_pso = super::context::dump_on_err(
                info_queue,
                super::auto_exposure::create_compute_pso(
                    device,
                    &hiz.root_sig,
                    &init_msaa_cs,
                    "hiz init_msaa",
                ),
            )?;
            let downsample_pso = super::context::dump_on_err(
                info_queue,
                super::auto_exposure::create_compute_pso(
                    device,
                    &hiz.root_sig,
                    &downsample_cs,
                    "hiz downsample",
                ),
            )?;
            Some((init_single_pso, init_msaa_pso, downsample_pso))
        } else {
            None
        };

        // Auto-exposure (gated on the post-process config).
        let (auto_exp_build, auto_exp_average) = if let Some(ae) =
            self.auto_exposure.resources.as_ref()
        {
            let (build_cs, average_cs) = super::auto_exposure::compile_auto_exposure_shaders(hr)?;
            let build_pso = super::context::dump_on_err(
                info_queue,
                super::auto_exposure::create_compute_pso(
                    device,
                    ae.build_root_sig(),
                    &build_cs,
                    "auto-exposure build",
                ),
            )?;
            let average_pso = super::context::dump_on_err(
                info_queue,
                super::auto_exposure::create_compute_pso(
                    device,
                    ae.average_root_sig(),
                    &average_cs,
                    "auto-exposure average",
                ),
            )?;
            (Some(build_pso), Some(average_pso))
        } else {
            (None, None)
        };

        // Decal (always built when DecalResources exists, which is unconditional).
        let decal_pso = rebuild_if_live!(
            self.decal.state.is_some(),
            super::decal::rebuild_decal_pso(
                device,
                &self.decal.state.as_ref().unwrap().root_sig,
                self.hdr.msaa_samples,
                hr,
                info_queue,
            )
        );

        // Glass (only when the world declared any GlassPanel).
        let glass_pso = rebuild_if_live!(
            self.glass.is_some(),
            super::glass::rebuild_glass_pso(
                device,
                &self.glass.as_ref().unwrap().root_sig,
                self.hdr.msaa_samples,
                hr,
                info_queue,
            )
        );

        // Fog (only when the world declared a VolumetricFog). Both the
        // render PSO (fragment volume sampler) and the compute PSO
        // (froxel-volume kernel) rebuild from the same `fog.metal`-style
        // source pair.
        let fog_pso = rebuild_if_live!(
            self.fog.resources.is_some(),
            super::fog::rebuild_fog_pso(
                device,
                &self.fog.resources.as_ref().unwrap().root_sig,
                self.hdr.msaa_samples,
                hr,
                info_queue,
            )
        );
        let fog_froxel_pso = rebuild_if_live!(
            self.fog.resources.is_some(),
            super::fog::rebuild_fog_froxel_pso(
                device,
                &self.fog.resources.as_ref().unwrap().froxel_root_sig,
                hr,
                info_queue,
            )
        );

        // SSAO (only when PostProcessConfig opted in).
        let ssao_rebuilt = rebuild_if_live!(
            self.ssao.resources.is_some(),
            super::post::ssao::rebuild_ssao_pipelines(
                device,
                self.ssao.resources.as_ref().unwrap(),
                hr,
                info_queue
            )
        );

        // SSR (only when PostProcessConfig opted in).
        let ssr_rebuilt = rebuild_if_live!(
            self.ssr.is_some(),
            super::post::ssr::rebuild_ssr_pipelines(
                device,
                self.ssr.as_ref().unwrap(),
                hr,
                info_queue
            )
        );

        // Unified G-buffer pre-pass (built when any screen-space consumer is on).
        let gbuffer_rebuilt = rebuild_if_live!(
            self.gbuffer.is_some(),
            super::post::gbuffer::rebuild_gbuffer_pipelines(
                device,
                self.gbuffer.as_ref().unwrap(),
                hr,
                info_queue
            )
        );

        // SSGI (only when PostProcessConfig.indirect_lighting == ssgi).
        let ssgi_rebuilt = rebuild_if_live!(
            self.ssgi.is_some(),
            super::post::ssgi::rebuild_ssgi_pipelines(
                device,
                self.ssgi.as_ref().unwrap(),
                hr,
                info_queue
            )
        );

        // TAA (only when PostProcessConfig.taa).
        let taa_rebuilt = rebuild_if_live!(
            self.taa.is_some(),
            super::post::taa::rebuild_taa_pipelines(
                device,
                self.taa.as_ref().unwrap(),
                hr,
                info_queue
            )
        );

        // RT reflections (only when DXR + DXC compile + accel build all succeeded
        // at init). The shader compiles through DXC (SM 6.5).
        let rt_rebuilt = rebuild_if_live!(
            self.rt_reflections.is_some(),
            super::post::rt_reflections::rebuild_rt_reflections_pipelines(
                device,
                self.rt_reflections.as_ref().unwrap(),
                hr,
                info_queue
            )
        );

        // All builds succeeded; swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.composite_pso = composite_pso;
        if let Some(p) = text_pso {
            self.text_pso = Some(p);
        }
        self.bloom.pso_prefilter = bloom_prefilter;
        self.bloom.pso_downsample = bloom_downsample;
        self.bloom.pso_upsample = bloom_upsample;
        if let Some(p) = bindless_main_pso {
            self.cull.main_bindless_pso = Some(p);
        }
        if let Some(p) = cull_pso {
            self.cull.cull_pso = Some(p);
        }
        if let Some(p) = cull_pso_phase2 {
            self.cull.cull_pso_phase2 = Some(p);
        }
        if let (Some((init_s, init_m, ds)), Some(hiz)) = (hiz_rebuilt, self.cull.hiz.as_mut()) {
            hiz.swap_pipelines(init_s, init_m, ds);
        }
        if let (Some(build), Some(average), Some(ae)) = (
            auto_exp_build,
            auto_exp_average,
            self.auto_exposure.resources.as_mut(),
        ) {
            ae.swap_pipelines(build, average);
        }
        if let (Some(pso), Some(decals)) = (decal_pso, self.decal.state.as_mut()) {
            decals.pso = pso;
        }
        if let (Some(pso), Some(glass)) = (glass_pso, self.glass.as_mut()) {
            glass.pso = pso;
        }
        if let (Some(pso), Some(fog)) = (fog_pso, self.fog.resources.as_mut()) {
            fog.pso = pso;
        }
        if let (Some(pso), Some(fog)) = (fog_froxel_pso, self.fog.resources.as_mut()) {
            fog.froxel_pso = pso;
        }
        if let (Some(rebuilt), Some(ssao)) = (ssao_rebuilt, self.ssao.resources.as_mut()) {
            swap_ssao_pipelines(ssao, rebuilt);
        }
        if let (Some(rebuilt), Some(ssr)) = (ssr_rebuilt, self.ssr.as_mut()) {
            swap_ssr_pipelines(ssr, rebuilt);
        }
        if let (Some(rebuilt), Some(gbuffer)) = (gbuffer_rebuilt, self.gbuffer.as_mut()) {
            gbuffer.pso = rebuilt.pso;
            if let Some(p) = rebuilt.instanced_pso {
                gbuffer.instanced_pso = Some(p);
            }
            if let Some(p) = rebuilt.skinned_pso {
                gbuffer.skinned_pso = Some(p);
            }
        }
        if let (Some(rebuilt), Some(ssgi)) = (ssgi_rebuilt, self.ssgi.as_mut()) {
            super::post::ssgi::swap_ssgi_pipelines(ssgi, rebuilt);
        }
        if let (Some(rebuilt), Some(rt)) = (rt_rebuilt, self.rt_reflections.as_mut()) {
            super::post::rt_reflections::swap_rt_reflections_pipelines(rt, rebuilt);
        }
        if let (Some(rebuilt), Some(taa)) = (taa_rebuilt, self.taa.as_mut()) {
            swap_taa_pipelines(taa, rebuilt);
        }
        Ok(())
    }
}

// Per-resource swap helpers; keep the field assignments in one place so the
// reload pass reads as a list of "swap this subsystem" intents.

fn swap_ssao_pipelines(
    ssao: &mut super::post::ssao::SsaoResources,
    rebuilt: super::post::ssao::RebuiltSsaoPipelines,
) {
    ssao.kernel_pso = rebuilt.kernel_pso;
    ssao.blur_pso = rebuilt.blur_pso;
}

fn swap_ssr_pipelines(
    ssr: &mut super::post::ssr::SsrResources,
    rebuilt: super::post::ssr::RebuiltSsrPipelines,
) {
    if let (Some(pso), Some(resolve)) = (rebuilt.resolve_pso, ssr.resolve.as_mut()) {
        resolve.resolve_pso = pso;
    }
}

fn swap_taa_pipelines(
    taa: &mut super::post::taa::TaaResources,
    rebuilt: super::post::taa::RebuiltTaaPipelines,
) {
    taa.taa_pso = rebuilt.taa_pso;
}
