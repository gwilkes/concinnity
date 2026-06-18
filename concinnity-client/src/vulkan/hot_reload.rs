// src/vulkan/hot_reload.rs
//
// Filesystem watcher driving Vulkan shader hot-reload. A background notify
// watcher tails `<CARGO_MANIFEST_DIR>/src/vulkan/shaders/` and, on any modify
// event for a known shader-source extension, flips a shared
// `Arc<AtomicBool>`. The main thread polls that flag at the top of
// `draw_frame` and calls `VkContext::reload_shaders` when it is set. The
// same flag is also set by the `reload-shaders` debug WebSocket command, so
// the two trigger paths converge.
//
// Entirely a dev-loop concern, only constructed when `VkContext::new` is
// called with `hot_reload = true`. Production `cn run` never instantiates
// it. Mirrors `directx/hot_reload.rs` and `metal/hot_reload.rs`.

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ash::vk;

use super::auto_exposure::{AutoExposureResources, compile_auto_exposure_shaders};
use super::context::VkContext;
use super::pipeline::{
    compile_bindless_shaders, compile_composite_shaders, compile_cull_shader,
    compile_cull_shader_phase2, compile_skinned_shaders, compile_text_shaders,
    create_composite_pipeline, create_cull_pipeline, create_instanced_pipeline,
    create_main_pipeline, create_skinned_pipeline, create_text_pipeline, resolve_instanced_shader,
    resolve_main_shaders,
};
use super::post::bloom::{compile_bloom_shaders, create_bloom_pipeline};
use super::post::ssao::rebuild_ssao_pipelines;
use super::post::ssr::rebuild_ssr_pipelines;
use super::post::taa::rebuild_taa_pipelines;

// Rebuild a feature's pipeline(s) into a temporary only when the feature is
// live, propagating any compile/create error out of the enclosing
// `reload_shaders`. `$cond` is the liveness check (`self.x.is_some()`); `$build`
// is the build expression (which re-accesses `self.x` and may use `?` internally
// for the shader compile). Expands to `Some(build?)` when live, `None`
// otherwise, so the swap phase below can pair each rebuilt `Some(_)` with its
// live target uniformly. Mirrors `directx/hot_reload.rs::rebuild_if_live!`.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// Shader-source extensions the watcher reacts to. GLSL sources land as
// `.vert` / `.frag` / `.comp`; the helper rejects every other event so
// editor swap files, README updates, and tmp files don't trigger a
// rebuild.
const SHADER_EXTENSIONS: &[&str] = &["vert", "frag", "comp"];

// Live watcher handle. Held by `VkContext` purely to keep the watcher
// thread alive; dropping it stops the watcher. The flag itself is shared
// via [`VkContext::shader_reload_pending`].
pub(crate) struct WatcherHandle {
    // notify keeps its own listener thread alive for as long as the handle
    // exists; we never read this field after construction.
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    // The shader source directory the watcher is observing. Kept for
    // diagnostics: log lines reference it on init.
    #[allow(dead_code)]
    watched_dir: PathBuf,
}

// Spawn a `notify` watcher over the Vulkan shader source directory and
// wire it to flip `flag` on any modify event for a known shader extension.
// The path is derived from `CARGO_MANIFEST_DIR` at compile time so the
// watcher works no matter where the binary is launched from, but only as
// long as the source tree still exists at that path. A shipped binary
// should never be hot-reload-enabled, so the missing-path case logs and
// returns `None` instead of failing the whole context init.
pub(crate) fn spawn(flag: Arc<AtomicBool>) -> Option<WatcherHandle> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("vulkan")
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

    tracing::info!(
        "hot-reload: watching {} for {} changes",
        dir.display(),
        SHADER_EXTENSIONS.join("/"),
    );
    Some(WatcherHandle {
        watcher,
        watched_dir: dir,
    })
}

// True when this notify event is a modify of a known shader file. Filters
// out unrelated paths (e.g. swap files, sub-directory churn) and the
// non-mutating events notify emits (e.g. access/metadata).
fn is_relevant(event: &Event) -> bool {
    if !matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    event.paths.iter().any(|p| {
        p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
            SHADER_EXTENSIONS
                .iter()
                .any(|&se| se.eq_ignore_ascii_case(e))
        })
    })
}

impl VkContext {
    // True when the shared shader-reload flag is set. Cheap atomic load;
    // called at the top of `draw_frame`. Returns false when hot-reload is
    // off so the production path never enters the reload branch.
    pub(in crate::vulkan) fn shader_reload_requested(&self) -> bool {
        self.shader_reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders`
    // regardless of outcome so a failed rebuild does not loop forever.
    pub(in crate::vulkan) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.shader_reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in Vulkan pipeline from disk-resident source.
    // Each pipeline is constructed into a temporary first; only when every
    // rebuild succeeds does the context swap them in (after destroying
    // the displaced ones). Any GLSL compile or pipeline-create failure
    // logs the underlying message and leaves the live pipelines untouched:
    // a typo in a shader edit won't crash the running session.
    //
    // Covers every runtime-bundled pipeline whose source lives in
    // `vulkan/shaders/`: composite, text, bloom (prefilter / downsample /
    // upsample), bindless main (when live), GPU-cull compute, auto-exposure
    // (build + average), projected-decal, volumetric-fog, SSAO (prepass
    // static / instanced / skinned, kernel, blur), SSR (prepass static /
    // instanced / skinned, resolve), and TAA (velocity static / instanced,
    // resolve). The world-loaded main / shadow / instanced / skinned
    // pipelines remain out of scope; same split as DirectX. The caller
    // has already `device_wait_idle`'d so swapping pipelines out from
    // under in-flight command buffers is safe.
    pub(in crate::vulkan) fn reload_shaders(&mut self) -> Result<(), String> {
        if !self.hot_reload {
            return Ok(());
        }
        let device = self.device.clone();
        let device = &device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.

        // Composite (always live).
        let (composite_vs, composite_ps) = compile_composite_shaders(hr)?;
        let composite_pipeline = create_composite_pipeline(
            device,
            self.composite_render_pass,
            self.composite_pipeline_layout,
            &composite_vs,
            &composite_ps,
        )?;

        // Text (only when the world declared text atlases).
        let text_pipeline = rebuild_if_live!(self.text_pipeline.is_some(), {
            let (tv, tf) = compile_text_shaders(hr)?;
            create_text_pipeline(
                device,
                self.composite_render_pass,
                self.text_pipeline_layout,
                &tv,
                &tf,
                vk::SampleCountFlags::TYPE_1,
            )
        });

        // Bloom (always live, 3 pipelines).
        let bloom_shaders = compile_bloom_shaders(hr)?;
        let bloom_prefilter = create_bloom_pipeline(
            device,
            self.bloom_write_pass,
            self.bloom_pipeline_layout,
            &bloom_shaders.vert,
            &bloom_shaders.prefilter,
            false,
        )?;
        let bloom_downsample = create_bloom_pipeline(
            device,
            self.bloom_write_pass,
            self.bloom_pipeline_layout,
            &bloom_shaders.vert,
            &bloom_shaders.downsample,
            false,
        )?;
        let bloom_upsample = create_bloom_pipeline(
            device,
            self.bloom_blend_pass,
            self.bloom_pipeline_layout,
            &bloom_shaders.vert,
            &bloom_shaders.upsample,
            true,
        )?;

        // Bindless main + cull (when the world drives the built-in shader).
        let bindless_main_pipeline = rebuild_if_live!(
            self.cull.bindless_pipeline_layout.is_some() && self.cull.bindless_pipeline.is_some(),
            {
                let (bvs, bps) = compile_bindless_shaders(hr, self.textures.len())?;
                create_main_pipeline(
                    device,
                    self.main_render_pass,
                    self.cull.bindless_pipeline_layout.unwrap(),
                    &bvs,
                    &bps,
                    self.msaa_samples,
                    self.swapchain_format,
                )
            }
        );
        let cull_pipeline = rebuild_if_live!(
            self.cull.cull_pipeline_layout.is_some() && self.cull.cull_pipeline.is_some(),
            {
                let cs = compile_cull_shader(hr)?;
                create_cull_pipeline(device, self.cull.cull_pipeline_layout.unwrap(), &cs)
            }
        );
        // Phase-2 cull (two-pass occlusion), rebuilt alongside phase 1 from the
        // same source with the `CULL_PHASE2` define + the shared layout.
        let cull_pipeline_phase2 = rebuild_if_live!(
            self.cull.cull_pipeline_layout.is_some() && self.cull.cull_pipeline_phase2.is_some(),
            {
                let cs = compile_cull_shader_phase2(hr)?;
                create_cull_pipeline(device, self.cull.cull_pipeline_layout.unwrap(), &cs)
            }
        );
        // Hi-Z build kernels (live alongside the cull pipeline).
        let hiz_pipelines = rebuild_if_live!(
            self.cull.hiz.is_some(),
            self.cull
                .hiz
                .as_ref()
                .unwrap()
                .recompile_pipelines(device, hr)
        );

        // Auto-exposure (gated on the post-process config). Builds the histogram
        // + average compute pipelines; the trailing `.map` tuples them so the
        // whole build is one Result expression for the macro.
        let auto_exposure_pipelines = rebuild_if_live!(self.auto_exposure.is_some(), {
            let ae = self.auto_exposure.as_ref().unwrap();
            let (build_cs, average_cs) = compile_auto_exposure_shaders(hr)?;
            let build = AutoExposureResources::create_compute_pipeline(
                device,
                ae.build_pipeline_layout(),
                &build_cs,
            )?;
            AutoExposureResources::create_compute_pipeline(
                device,
                ae.average_pipeline_layout(),
                &average_cs,
            )
            .map(|average| (build, average))
        });

        // Decal (always built when DecalResources exists, which is
        // unconditional in `VkContext::new`).
        let decal_pipeline = rebuild_if_live!(
            self.decals_state.is_some(),
            super::decal::rebuild_decal_pipeline(
                device,
                self.decals_state.as_ref().unwrap(),
                self.msaa_samples != vk::SampleCountFlags::TYPE_1,
                hr,
            )
        );

        // Fog (only when the world declared a VolumetricFog). Rebuilds both the
        // fullscreen render pipeline and the froxel-volume compute kernel; the
        // trailing `.map` tuples them into one Result for the macro.
        let fog_pipelines = rebuild_if_live!(self.fog_resources.is_some(), {
            let fog = self.fog_resources.as_ref().unwrap();
            let render = super::fog::rebuild_fog_pipeline(
                device,
                fog,
                self.msaa_samples != vk::SampleCountFlags::TYPE_1,
                hr,
            )?;
            super::fog::rebuild_fog_froxel_pipeline(device, fog, hr).map(|froxel| (render, froxel))
        });

        // SSAO (only when PostProcessConfig opted in). Rebuilds prepass
        // static / instanced / skinned + kernel + blur in one shot.
        let ssao_rebuilt = rebuild_if_live!(
            self.ssao.is_some(),
            rebuild_ssao_pipelines(device, self.ssao.as_ref().unwrap(), hr)
        );

        // SSR (only when PostProcessConfig opted in). Rebuilds prepass
        // static / instanced / skinned + resolve in one shot.
        let ssr_rebuilt = rebuild_if_live!(
            self.ssr.is_some(),
            rebuild_ssr_pipelines(device, self.ssr.as_ref().unwrap(), hr)
        );

        // SSGI (only when indirect_lighting: ssgi). Rebuilds gather + composite.
        let ssgi_rebuilt = rebuild_if_live!(
            self.ssgi.is_some(),
            crate::vulkan::post::ssgi::rebuild_ssgi_pipelines(
                device,
                self.ssgi.as_ref().unwrap(),
                hr,
            )
        );

        // RT reflections (only when the world opted in + the GPU supports it).
        // Rebuilds the flat + textured ray-query pipelines.
        let rt_rebuilt = rebuild_if_live!(
            self.rt_reflections.is_some(),
            crate::vulkan::post::rt_reflections::rebuild_rt_pipelines(
                device,
                self.rt_reflections.as_ref().unwrap(),
                hr,
            )
        );

        // TAA (only when PostProcessConfig opted in). Rebuilds the resolve
        // pipeline; the velocity channel lives on the unified G-buffer pre-pass.
        let taa_rebuilt = rebuild_if_live!(
            self.taa.is_some(),
            rebuild_taa_pipelines(device, self.taa.as_ref().unwrap(), hr)
        );

        // Unified G-buffer pre-pass (only when any screen-space consumer is on).
        // Rebuilds prepass static / instanced / skinned in one shot (the skinned
        // variant only when a `SkinnedMesh` is live).
        let gbuffer_rebuilt = rebuild_if_live!(
            self.gbuffer.is_some(),
            crate::vulkan::post::gbuffer::rebuild_gbuffer_pipelines(
                device,
                self.gbuffer.as_ref().unwrap(),
                hr,
            )
        );

        // Particles (only when ≥1 emitter is live or has ever been
        // added at runtime). Rebuilds the compute + render pipelines in
        // one shot.
        let particle_rebuilt = rebuild_if_live!(
            self.particle_resources.is_some(),
            self.particle_resources
                .as_ref()
                .unwrap()
                .rebuild_pipelines(device, hr)
        );

        // All builds succeeded: destroy the displaced pipelines and swap
        // the freshly compiled ones in. The caller's `wait_idle` above this
        // method guarantees no command buffer still references them.
        unsafe {
            device.destroy_pipeline(self.composite_pipeline, None);
        }
        self.composite_pipeline = composite_pipeline;

        if let Some(new_pipeline) = text_pipeline {
            if let Some(old) = self.text_pipeline.take() {
                unsafe { device.destroy_pipeline(old, None) };
            }
            self.text_pipeline = Some(new_pipeline);
        }

        unsafe {
            device.destroy_pipeline(self.bloom_pipeline_prefilter, None);
            device.destroy_pipeline(self.bloom_pipeline_downsample, None);
            device.destroy_pipeline(self.bloom_pipeline_upsample, None);
        }
        self.bloom_pipeline_prefilter = bloom_prefilter;
        self.bloom_pipeline_downsample = bloom_downsample;
        self.bloom_pipeline_upsample = bloom_upsample;

        if let Some(new_pipeline) = bindless_main_pipeline {
            if let Some(old) = self.cull.bindless_pipeline.take() {
                unsafe { device.destroy_pipeline(old, None) };
            }
            self.cull.bindless_pipeline = Some(new_pipeline);
        }
        if let Some(new_pipeline) = cull_pipeline {
            if let Some(old) = self.cull.cull_pipeline.take() {
                unsafe { device.destroy_pipeline(old, None) };
            }
            self.cull.cull_pipeline = Some(new_pipeline);
        }
        if let Some(new_pipeline) = cull_pipeline_phase2 {
            if let Some(old) = self.cull.cull_pipeline_phase2.take() {
                unsafe { device.destroy_pipeline(old, None) };
            }
            self.cull.cull_pipeline_phase2 = Some(new_pipeline);
        }
        if let (Some((init, downsample)), Some(hiz)) = (hiz_pipelines, self.cull.hiz.as_mut()) {
            hiz.swap_pipelines(device, init, downsample);
        }

        if let (Some((build, average)), Some(ae)) =
            (auto_exposure_pipelines, self.auto_exposure.as_mut())
        {
            ae.swap_pipelines(device, build, average);
        }

        if let (Some(new_pipeline), Some(decals)) = (decal_pipeline, self.decals_state.as_mut()) {
            unsafe { device.destroy_pipeline(decals.pipeline, None) };
            decals.pipeline = new_pipeline;
        }
        if let (Some((render, froxel)), Some(fog)) = (fog_pipelines, self.fog_resources.as_mut()) {
            unsafe {
                device.destroy_pipeline(fog.pipeline, None);
                device.destroy_pipeline(fog.froxel_pipeline, None);
            }
            fog.pipeline = render;
            fog.froxel_pipeline = froxel;
        }
        if let (Some(rebuilt), Some(ssao)) = (ssao_rebuilt, self.ssao.as_mut()) {
            ssao.swap_pipelines(device, rebuilt);
        }
        if let (Some(rebuilt), Some(ssr)) = (ssr_rebuilt, self.ssr.as_mut()) {
            ssr.swap_pipelines(device, rebuilt);
        }
        if let (Some(rebuilt), Some(ssgi)) = (ssgi_rebuilt, self.ssgi.as_mut()) {
            ssgi.swap_pipelines(device, rebuilt);
        }
        if let (Some(rebuilt), Some(rt)) = (rt_rebuilt, self.rt_reflections.as_mut()) {
            rt.swap_pipelines(device, rebuilt);
        }
        if let (Some(rebuilt), Some(taa)) = (taa_rebuilt, self.taa.as_mut()) {
            taa.swap_pipelines(device, rebuilt);
        }
        if let (Some(rebuilt), Some(gb)) = (gbuffer_rebuilt, self.gbuffer.as_mut()) {
            gb.swap_pipelines(device, rebuilt);
        }
        if let (Some((cp, rp)), Some(p)) = (particle_rebuilt, self.particle_resources.as_mut()) {
            p.swap_pipelines(device, cp, rp);
        }
        Ok(())
    }

    // Rebuild the world-loaded shader pipelines (legacy main, optional
    // instanced, optional skinned main) from freshly compiled SPIR-V bytes.
    // Driven by asset hot-reload (`cn debug` only) when a captured
    // `ShaderStage` source file is saved or the debug-WS `reload-assets`
    // command fires. Mirrors the rebuild-then-swap safety pattern of
    // `reload_shaders`: every replacement is constructed into a temporary
    // first and the swap (destroy displaced + store fresh) only runs when
    // every build succeeds, so a typo in a shader edit leaves the live
    // pipelines untouched and the session keeps rendering. Sibling of
    // `DxContext::update_world_shader_pipelines` /
    // `MtlContext::update_world_shader_pipelines`.
    //
    // `vert_bytes` + `frag_bytes` are always required: a custom-shader world
    // renders through the legacy per-draw `main_pipeline` (the bindless
    // variant is forced off at init when the world supplies SPIR-V, and stays
    // engine-internal, rebuilt by `reload_shaders`). The instanced pipeline is
    // rebuilt only when one is live, pairing the world's instanced vertex
    // stage with the fresh fragment. The skinned main pipeline keeps its
    // engine-internal 80-byte vertex shader and only swaps the fragment
    // (`compile_skinned_shaders` compiles the skinned VS from inline GLSL and
    // treats the supplied bytes as the fragment, exactly as `upload_skinned`
    // does at init). The shadow pipelines (static + skinned) are
    // engine-internal GLSL, reloaded by `reload_shaders`, so `_shadow_bytes`
    // is unused, same as Metal / DirectX.
    //
    // Reached only through the bin's `cn debug` runtime-mutation path (dead
    // from the FFI lib crate's roots, live in the concinnity binary), like the
    // other runtime-mutation methods on `VkContext`.
    #[allow(dead_code)]
    pub fn update_world_shader_pipelines(
        &mut self,
        vert_bytes: Option<&[u8]>,
        frag_bytes: Option<&[u8]>,
        _shadow_bytes: Option<&[u8]>,
        vert_instanced_bytes: Option<&[u8]>,
    ) -> Result<(), String> {
        let vert = vert_bytes.ok_or_else(|| {
            "update_world_shader_pipelines: vertex shader bytes are required".to_string()
        })?;
        let frag = frag_bytes.ok_or_else(|| {
            "update_world_shader_pipelines: fragment shader bytes are required".to_string()
        })?;

        let device = self.device.clone();
        let device = &device;
        let hr = self.hot_reload;

        // Resolve the world bytes to SPIR-V. The hot-reload recompile always
        // hands us SPIR-V, so `resolve_main_shaders` passes them through; the
        // GLSL fallback only matters at init. Build every replacement into a
        // temporary first so a `?` early-return leaves the live pipelines
        // untouched, mirroring `reload_shaders`.
        let (vert_spv, frag_spv) = resolve_main_shaders(hr, vert, frag)?;
        let new_main = create_main_pipeline(
            device,
            self.main_render_pass,
            self.main_pipeline_layout,
            &vert_spv,
            &frag_spv,
            self.msaa_samples,
            self.swapchain_format,
        )?;

        // Instanced pipeline: rebuilt only when one is live. Needs the world's
        // instanced vertex stage paired with the fresh fragment, reusing the
        // live instanced pipeline layout.
        let new_instanced = if let (Some(_), Some(layout)) =
            (self.instanced.pipeline, self.instanced.pipeline_layout)
        {
            let inst = vert_instanced_bytes.ok_or_else(|| {
                "update_world_shader_pipelines: instanced vertex shader bytes are required \
                 when an instanced pipeline is live"
                    .to_string()
            })?;
            let inst_spv = resolve_instanced_shader(hr, inst, true)?.ok_or_else(|| {
                "update_world_shader_pipelines: instanced shader payload missing".to_string()
            })?;
            Some(create_instanced_pipeline(
                device,
                self.main_render_pass,
                layout,
                &inst_spv,
                &frag_spv,
                self.msaa_samples,
                self.swapchain_format,
            )?)
        } else {
            None
        };

        // Skinned main pipeline: rebuilt only when one is live. Keeps its
        // engine-internal skinned vertex shader; only the fragment changes.
        let new_skinned = if let (Some(_), Some(layout)) =
            (self.skinned.pipeline, self.skinned.pipeline_layout)
        {
            let (skinned_vs, _skinned_shadow_vs, frag_ps) = compile_skinned_shaders(hr, frag)?;
            Some(create_skinned_pipeline(
                device,
                self.main_render_pass,
                layout,
                &skinned_vs,
                &frag_ps,
                self.msaa_samples,
            )?)
        } else {
            None
        };

        // All builds succeeded. Drain the GPU before destroying the displaced
        // pipelines so no in-flight command buffer still references them: the
        // debug hot-reload drive does not `wait_idle` for us, unlike the
        // built-in `reload_shaders` path the draw loop guards. Mirrors the
        // internal `wait_idle` in `upload_skinned` / `update_skinned_mesh_geometry`.
        self.wait_idle();
        unsafe { device.destroy_pipeline(self.main_pipeline, None) };
        self.main_pipeline = new_main;
        if let Some(p) = new_instanced {
            if let Some(old) = self.instanced.pipeline.take() {
                unsafe { device.destroy_pipeline(old, None) };
            }
            self.instanced.pipeline = Some(p);
        }
        if let Some(p) = new_skinned {
            if let Some(old) = self.skinned.pipeline.take() {
                unsafe { device.destroy_pipeline(old, None) };
            }
            self.skinned.pipeline = Some(p);
        }
        Ok(())
    }
}
