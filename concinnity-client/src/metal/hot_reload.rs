// src/metal/hot_reload.rs
//
// Filesystem watcher driving Metal shader hot-reload. A background notify
// watcher tails `<CARGO_MANIFEST_DIR>/src/metal/shaders/` and, on any modify
// event for a `.metal` file, flips a shared `Arc<AtomicBool>`. The main thread
// polls that flag at the top of `draw_frame` and calls
// `MtlContext::reload_shaders` when it's set. Same flag is also set by the
// `reload-shaders` debug WebSocket command, so the two trigger paths converge.
//
// All entirely a dev-loop concern: only constructed when
// `MtlContext::new` is called with `hot_reload = true`. Production `cn run`
// never instantiates it.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use notify::{Event, EventKind, RecursiveMode, Watcher};
use objc2::rc::Retained;
use objc2_metal::{MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::auto_exposure::build_auto_exposure_pipelines;
use super::context::MtlContext;
use super::cull::{build_cull_pipeline, build_shadow_cull_pipeline};
use super::decal::build_decal_pipeline;
use super::fog::build_fog_pipeline;
use super::hiz::build_hiz_pipelines;
use super::init::pipelines::{
    MainPipelineBundle, build_instanced_pipeline, build_main_pipeline,
    build_shadow_bindless_pipeline, build_shadow_pipeline, make_vertex_descriptor,
};
use super::pipeline::{build_post_pipeline, build_text_pipeline};
use super::post::{
    build_bloom_pipelines, build_gbuffer_bindless_pipeline, build_gbuffer_prepass_pipeline,
    build_reflection_blur_pipeline, build_reflection_composite_pipeline,
    build_rt_reflection_pipeline, build_ssao_pipeline, build_ssgi_composite_pipeline,
    build_ssgi_gather_pipeline, build_ssr_pipeline, build_taa_pipeline,
};
use super::resources::skinning::{
    build_skinned_main_pipeline, build_skinned_shadow_pipeline, make_skinned_vertex_descriptor,
};

// Rebuild a built-in pipeline only when it is currently live. Expands to
// `if $cond { Some($build?) } else { None }`: the rebuild-then-swap pattern
// `reload_shaders` repeats for every optional pipeline: a `None` field stays
// `None`, and any compile error (the `?`) aborts the whole reload before the
// swap, leaving the live pipelines untouched.
macro_rules! rebuild_if_live {
    ($cond:expr_2021, $build:expr_2021 $(,)?) => {
        if $cond { Some($build?) } else { None }
    };
}

// Live watcher handle. Held by `MtlContext` purely to keep the watcher
// thread alive; dropping it stops the watcher. The flag itself is shared
// via [`MtlContext::shader_reload_pending`].
pub(crate) struct WatcherHandle {
    // We don't read `_watcher` after construction; notify keeps its own
    // listener thread for as long as the handle is alive.
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    // The shader source directory the watcher is observing. Kept for
    // diagnostics: log lines reference it on init.
    #[allow(dead_code)]
    watched_dir: PathBuf,
}

// Spawn a `notify` watcher over the Metal shader source directory and wire
// it to flip `flag` on any `.metal` file modify event. The path is derived
// from `CARGO_MANIFEST_DIR` at compile time so the watcher works no matter
// where the binary is launched from, but only as long as the source tree
// still exists at that path. A shipped binary should never be hot-reload-
// enabled, so the missing-path case logs and returns `None` instead of
// failing the whole context init.
pub(crate) fn spawn(flag: Arc<AtomicBool>) -> Option<WatcherHandle> {
    let dir: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("metal")
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

    tracing::info!("hot-reload: watching {} for .metal changes", dir.display());
    Some(WatcherHandle {
        watcher,
        watched_dir: dir,
    })
}

// True when this notify event is a modify of a `.metal` file we care about.
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
        .any(|p| p.extension().is_some_and(|e| e == "metal"))
}

// Static-vertex-layout descriptor used by the velocity / SSAO / SSR pre-pass
// rebuilds during hot-reload. Matches the layout `MtlContext::new` builds at
// init; kept in sync by construction since both touch the 56-byte static
// `Vertex` struct.
fn static_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let vdesc = MTLVertexDescriptor::new();
    unsafe {
        let set = |idx: usize, fmt: MTLVertexFormat, offset: usize| {
            let attr = vdesc.attributes().objectAtIndexedSubscript(idx);
            attr.setFormat(fmt);
            attr.setOffset(offset);
            attr.setBufferIndex(1);
        };
        set(0, MTLVertexFormat::Float3, 0);
        set(1, MTLVertexFormat::Float3, 12);
        set(2, MTLVertexFormat::Float3, 24);
        set(3, MTLVertexFormat::Float3, 36);
        set(4, MTLVertexFormat::Float2, 48);
        let layout = vdesc.layouts().objectAtIndexedSubscript(1);
        layout.setStride(std::mem::size_of::<crate::gfx::mesh_payload::Vertex>());
        layout.setStepFunction(MTLVertexStepFunction::PerVertex);
    }
    vdesc
}

impl MtlContext {
    // True when the shared shader-reload flag is set. Cheap atomic load; called
    // at the top of `draw_frame`. Returns false when hot-reload is off so
    // the production path never enters the reload branch.
    pub(super) fn shader_reload_requested(&self) -> bool {
        self.shader_reload_pending
            .as_ref()
            .map(|f| f.load(Ordering::SeqCst))
            .unwrap_or(false)
    }

    // Clear the pending-reload flag. Called after `reload_shaders` regardless
    // of outcome so a failed rebuild does not loop forever.
    pub(super) fn clear_shader_reload_flag(&self) {
        if let Some(flag) = &self.shader_reload_pending {
            flag.store(false, Ordering::SeqCst);
        }
    }

    // Rebuild every built-in Metal renderer pipeline from disk-resident source.
    // Each pipeline is constructed into a temporary first; only when every
    // rebuild succeeds does the context atomically swap them in. Any compile
    // or link error logs the underlying message and leaves the live pipelines
    // untouched: a typo in a shader edit won't crash the running session.
    //
    // Covers the skinned velocity / SSAO / SSR pre-pass variants too: they
    // compile from the same on-disk `velocity.metal` / `ssao.metal` /
    // `ssr.metal` sources as their static + instanced siblings, just with a
    // different vertex entry point. The skinned main + skinned shadow
    // pipelines are not built here: their entry points live in the world's
    // vertex / fragment / shadow `ShaderStage` library bytes, so they
    // reload through [`Self::update_world_shader_pipelines`] alongside the
    // static main pipeline.
    pub(super) fn reload_shaders(&mut self) -> Result<(), String> {
        if !self.hot_reload {
            return Ok(());
        }
        let device = &self.device;
        let hr = true;

        // Build every replacement into a temporary first. A `?` early-return
        // here means we never overwrite a live pipeline with a failed build:
        // any compile error leaves the running session rendering with the
        // previous shader source.
        let post = build_post_pipeline(device, self.swap_pixel_format, hr)?;
        let bloom = build_bloom_pipelines(device, hr)?;

        let text = rebuild_if_live!(
            self.text_pipeline_state.is_some(),
            build_text_pipeline(device, self.swap_pixel_format, hr)
        );
        let taa = rebuild_if_live!(
            self.taa.pipeline_state.is_some(),
            build_taa_pipeline(device, hr)
        );
        let cull = rebuild_if_live!(
            self.cull.pipeline.is_some(),
            build_cull_pipeline(device, hr)
        );
        // Hi-Z build kernels are engine built-ins (independent of the world
        // shader); rebuild them whenever a Hi-Z resource exists so a saved
        // edit to `hiz_build.metal` is picked up. The texture + mip views are
        // kept: only the pipelines swap.
        let hiz = rebuild_if_live!(self.cull.hiz.is_some(), build_hiz_pipelines(device, hr));
        let auto_ev = rebuild_if_live!(
            self.auto_exposure.pipelines.is_some(),
            build_auto_exposure_pipelines(device, hr)
        );
        let decal = rebuild_if_live!(
            self.decal.pipeline.is_some(),
            build_decal_pipeline(device, hr)
        );
        let fog = rebuild_if_live!(self.fog.pipeline.is_some(), build_fog_pipeline(device, hr));

        // The G-buffer pre-pass + SSAO/SSR resolve variants need the static layout.
        let static_vdesc = static_vertex_descriptor();
        let ssao_kernel = rebuild_if_live!(
            self.ssao.kernel_pipeline.is_some(),
            build_ssao_pipeline(device, "ssao_fragment", hr)
        );
        let ssao_blur = rebuild_if_live!(
            self.ssao.blur_pipeline.is_some(),
            build_ssao_pipeline(device, "ssao_blur_fragment", hr)
        );
        let gbuffer_prepass = rebuild_if_live!(
            self.gbuffer.prepass_pipeline.is_some(),
            build_gbuffer_prepass_pipeline(device, &static_vdesc, "gbuffer_prepass_vertex", hr)
        );
        let gbuffer_instanced = rebuild_if_live!(
            self.gbuffer.instanced_pipeline.is_some(),
            build_gbuffer_prepass_pipeline(
                device,
                &static_vdesc,
                "gbuffer_prepass_vertex_instanced",
                hr,
            )
        );
        // GPU-driven bindless G-buffer pipeline: builds its own
        // two-stream vertex descriptor internally.
        let gbuffer_bindless = rebuild_if_live!(
            self.gbuffer.bindless_pipeline.is_some(),
            build_gbuffer_bindless_pipeline(device, hr)
        );
        let ssr_resolve = rebuild_if_live!(
            self.ssr.resolve_pipeline.is_some(),
            build_ssr_pipeline(device, hr)
        );
        let reflection_composite = rebuild_if_live!(
            self.ssr.composite_pipeline.is_some(),
            build_reflection_composite_pipeline(device, hr)
        );
        let reflection_blur = rebuild_if_live!(
            self.ssr.blur_pipeline.is_some(),
            build_reflection_blur_pipeline(device, hr)
        );
        let ssgi_gather = rebuild_if_live!(
            self.ssgi.gather_pipeline.is_some(),
            build_ssgi_gather_pipeline(device, hr)
        );
        let ssgi_composite = rebuild_if_live!(
            self.ssgi.composite_pipeline.is_some(),
            build_ssgi_composite_pipeline(device, hr)
        );
        let rt_reflections = rebuild_if_live!(
            self.rt.pipeline.is_some(),
            build_rt_reflection_pipeline(device, "rt_reflections_fragment", hr)
        );
        let rt_reflections_textured = rebuild_if_live!(
            self.rt.pipeline_textured.is_some(),
            build_rt_reflection_pipeline(device, "rt_reflections_fragment_textured", hr)
        );

        // Skinned pre-pass variants compile from the same on-disk shader
        // sources as the static variants: only the vertex entry point and
        // 80-byte vertex layout differ. `upload_skinned` only builds these
        // when the matching static pre-pass exists, so the per-field
        // `is_some()` check here is the same gate.
        let skinned_vdesc = if self.gbuffer.skinned_pipeline.is_some()
            || self.skinned.shadow_pipeline_state.is_some()
        {
            Some(make_skinned_vertex_descriptor())
        } else {
            None
        };
        let gbuffer_skinned = rebuild_if_live!(
            self.gbuffer.skinned_pipeline.is_some(),
            build_gbuffer_prepass_pipeline(
                device,
                skinned_vdesc.as_ref().expect("skinned vdesc just built"),
                "gbuffer_prepass_vertex_skinned",
                hr,
            )
        );

        // Shadow pass shaders are engine-internal (compiled from
        // `shadow_map.metal`), so they rebuild here alongside the other
        // built-ins rather than in `update_world_shader_pipelines`. The static
        // shadow pipeline shares the 56-byte static layout; the skinned one
        // rides the 80-byte skinned layout.
        let shadow = rebuild_if_live!(
            self.shadow_pipeline_state.is_some(),
            build_shadow_pipeline(device, &static_vdesc, hr)
        );
        let skinned_shadow = rebuild_if_live!(
            self.skinned.shadow_pipeline_state.is_some(),
            build_skinned_shadow_pipeline(
                device,
                skinned_vdesc.as_ref().expect("skinned vdesc just built"),
                hr,
            )
        );

        // GPU-driven cascaded-shadow pipelines: the frustum-only
        // shadow cull kernel (from cull.metal) + the depth-only bindless shadow
        // render pipeline (from shadow_map.metal). Both engine-internal, so they
        // rebuild here. Gated on the live shadow-bindless path.
        let shadow_cull = rebuild_if_live!(
            self.cull.shadow_pipeline.is_some(),
            build_shadow_cull_pipeline(device, hr)
        );
        let shadow_bindless = rebuild_if_live!(
            self.cull.shadow_bindless_pipeline.is_some(),
            build_shadow_bindless_pipeline(device, &static_vdesc, hr)
        );

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        self.post_pipeline_state = post;
        self.bloom_pipelines = bloom;
        if let Some(p) = text {
            self.text_pipeline_state = Some(p);
        }
        if let Some(p) = taa {
            self.taa.pipeline_state = Some(p);
        }
        if let Some(p) = cull {
            self.cull.pipeline = Some(p.state);
            self.cull.icb_arg_encoder = Some(p.icb_arg_encoder);
            // The phase-2 (two-pass occlusion) pipeline + ICB arg encoder come
            // from the same rebuilt library; swap them in lockstep.
            self.cull.pipeline_phase2 = Some(p.state_phase2);
            self.cull.icb_2_arg_encoder = Some(p.icb2_arg_encoder);
            // Force ICB rebuild on next frame so its argument-buffer encoding
            // re-binds to the new arg encoders the new cull kernels produced.
            // The status buffer + phase-2 ICB are rebuilt by the same
            // `ensure_icb_capacity` pass that rebuilds the phase-1 ICB.
            self.cull.icb = None;
            self.cull.icb_arg_buffer = None;
            self.cull.icb_capacity = 0;
            self.cull.icb_2 = None;
            self.cull.icb_2_arg_buffer = None;
            self.cull.status_buffer = None;
        }
        if let Some((init_pipeline, downsample_pipeline)) = hiz
            && let Some(h) = self.cull.hiz.as_mut()
        {
            h.swap_pipelines(init_pipeline, downsample_pipeline);
        }
        if let Some(p) = auto_ev {
            self.auto_exposure.pipelines = Some(p);
        }
        if let Some(p) = decal {
            self.decal.pipeline = Some(p);
        }
        if let Some(p) = fog {
            self.fog.pipeline = Some(p);
        }
        if let Some(p) = ssao_kernel {
            self.ssao.kernel_pipeline = Some(p);
        }
        if let Some(p) = ssao_blur {
            self.ssao.blur_pipeline = Some(p);
        }
        if let Some(p) = gbuffer_prepass {
            self.gbuffer.prepass_pipeline = Some(p);
        }
        if let Some(p) = gbuffer_instanced {
            self.gbuffer.instanced_pipeline = Some(p);
        }
        if let Some(p) = gbuffer_bindless {
            self.gbuffer.bindless_pipeline = Some(p);
        }
        if let Some(p) = ssr_resolve {
            self.ssr.resolve_pipeline = Some(p);
        }
        if let Some(p) = reflection_composite {
            self.ssr.composite_pipeline = Some(p);
        }
        if let Some(p) = reflection_blur {
            self.ssr.blur_pipeline = Some(p);
        }
        if let Some(p) = ssgi_gather {
            self.ssgi.gather_pipeline = Some(p);
        }
        if let Some(p) = ssgi_composite {
            self.ssgi.composite_pipeline = Some(p);
        }
        if let Some(p) = rt_reflections {
            self.rt.pipeline = Some(p);
        }
        if let Some(p) = rt_reflections_textured {
            self.rt.pipeline_textured = Some(p);
        }
        if let Some(p) = gbuffer_skinned {
            self.gbuffer.skinned_pipeline = Some(p);
        }
        if let Some(p) = shadow {
            self.shadow_pipeline_state = Some(p);
        }
        if let Some(p) = skinned_shadow {
            self.skinned.shadow_pipeline_state = Some(p);
        }
        if let Some((p, enc)) = shadow_cull {
            self.cull.shadow_pipeline = Some(p);
            self.cull.shadow_icb_arg_encoder = Some(enc);
            // Force the shadow ICB rebuild on the next frame so its argument
            // buffer re-binds to the freshly compiled kernel's arg encoder
            // (mirrors the main cull ICB reset above).
            self.cull.shadow_icb = None;
            self.cull.shadow_icb_arg_buffer = None;
            self.cull.shadow_icb_capacity = 0;
        }
        if let Some(p) = shadow_bindless {
            self.cull.shadow_bindless_pipeline = Some(p);
        }
        Ok(())
    }

    // Rebuild the world-loaded shader pipelines (main, optional instanced,
    // optional shadow) from freshly compiled metallib bytes. Driven by
    // asset hot-reload (`cn debug` only) when a captured `ShaderStage`
    // source file is saved or `reload-assets` is fired. Mirrors the
    // rebuild-then-swap safety pattern of [`Self::reload_shaders`]: every
    // replacement is constructed into a temporary first, and the atomic
    // swap only runs when every build succeeds: a typo in a shader edit
    // leaves the live pipelines untouched and the session keeps rendering.
    //
    // `vert_bytes` and `frag_bytes` are always required (the main pipeline
    // is required for the world to render). `vert_instanced_bytes` is honoured
    // only when the instanced pipeline is currently live. Hot-reload cannot
    // introduce a new shader-stage kind (or drop one): that would need
    // draw-list / asset graph changes that this path doesn't support.
    //
    // Skinned variants ride the same library bytes: when the world declared
    // a `SkinnedMesh` (so `upload_skinned` ran and `skinned_pipeline_state`
    // is live), this also rebuilds the main skinned pipeline. The shadow
    // pipelines (static + skinned) and the skinned velocity / SSAO / SSR
    // pre-pass pipelines compile from disk-resident engine-internal source
    // (not from world library bytes), so they are covered by
    // [`Self::reload_shaders`]: no work here. `_shadow_bytes` is retained for
    // the cross-backend signature but unused (the shadow shader is internal).
    pub(super) fn update_world_shader_pipelines(
        &mut self,
        vert_bytes: Option<&[u8]>,
        frag_bytes: Option<&[u8]>,
        _shadow_bytes: Option<&[u8]>,
        vert_instanced_bytes: Option<&[u8]>,
    ) -> Result<(), String> {
        let vert_bytes = vert_bytes
            .ok_or_else(|| "vertex shader bytes are required for the main pipeline".to_string())?;
        let frag_bytes = frag_bytes.ok_or_else(|| {
            "fragment shader bytes are required for the main pipeline".to_string()
        })?;

        // Build everything into temporaries first. Any `?` early-return
        // leaves the live pipelines untouched, mirroring `reload_shaders`.
        let vert_desc = make_vertex_descriptor();
        let new_main = build_main_pipeline(
            &self.device,
            &vert_desc,
            vert_bytes,
            frag_bytes,
            self.hot_reload,
        )?;

        // Instanced pipeline depends on both the instanced vertex bytes and
        // the (potentially fresh) fragment bytes. Rebuilt only when an
        // instanced pipeline is currently live AND the caller supplied new
        // instanced vertex bytes: a world without an instanced stage keeps
        // `instanced_pipeline_state == None` and skips this branch.
        let new_instanced = if self.instanced_pipeline_state.is_some() {
            let inst_bytes = vert_instanced_bytes.ok_or_else(|| {
                "instanced vertex shader bytes are required when an instanced pipeline is live"
                    .to_string()
            })?;
            // `has_clusters = true` forces the build path (the builder
            // short-circuits on `empty bytes || !has_clusters`).
            let ps =
                build_instanced_pipeline(&self.device, &vert_desc, inst_bytes, frag_bytes, true)?
                    .ok_or_else(|| {
                    "build_instanced_pipeline returned None on a forced rebuild".to_string()
                })?;
            Some(ps)
        } else {
            None
        };

        // Skinned main pipeline rides the same vert + frag library bytes as
        // the static main pipeline (just a different vertex entry point and
        // the 80-byte skinned vertex layout). Rebuilt only when a skinned
        // pipeline is currently live: a world without a `SkinnedMesh`
        // never called `upload_skinned`, so this stays `None`.
        let skinned_vdesc = if self.skinned.pipeline_state.is_some() {
            Some(make_skinned_vertex_descriptor())
        } else {
            None
        };
        let new_skinned_main = if self.skinned.pipeline_state.is_some() {
            let vdesc = skinned_vdesc.as_ref().expect("skinned vdesc just built");
            Some(build_skinned_main_pipeline(
                &self.device,
                vdesc,
                vert_bytes,
                frag_bytes,
            )?)
        } else {
            None
        };

        // All builds succeeded: swap into the live context. After this
        // point the next frame's draw calls bind the freshly compiled
        // pipelines.
        let MainPipelineBundle {
            pipeline_state,
            bindless,
            cull_pipeline,
            cull_icb_arg_encoder,
            cull_pipeline_phase2,
            cull_icb2_arg_encoder,
            bindless_tex_arg_encoder,
        } = new_main;
        self.pipeline_state = pipeline_state;
        // Swap the bindless flag + dependent state. The flag is the
        // bindless-vs-legacy switch for the static draw loop; if it changed
        // (e.g. the user toggled `fragment_main_bindless` on or off in
        // their shader), the ICB also has to be rebuilt because the new
        // arg encoder produces a different encoding shape.
        self.bindless = bindless;
        self.cull.pipeline = cull_pipeline;
        self.cull.icb_arg_encoder = cull_icb_arg_encoder;
        // Second-pass (two-pass occlusion) pipeline + ICB arg encoder swap with the
        // rest of the bundle. `two_pass_occlusion` is left as the init-time
        // resolution: a shader edit that drops `fragment_main_bindless` leaves
        // `cull_pipeline_phase2` None, and `ensure_icb_capacity` then skips the
        // phase-2 ICB while the graph builder skips the phase-2 nodes.
        self.cull.pipeline_phase2 = cull_pipeline_phase2;
        self.cull.icb_2_arg_encoder = cull_icb2_arg_encoder;
        self.bindless_tex_arg_encoder = bindless_tex_arg_encoder;
        // Force a fresh ICB on the next frame so its argument-buffer encoding
        // re-binds to the new encoder. Matches the `cull` swap in
        // `reload_shaders`; the phase-2 ICB + status buffer rebuild alongside.
        self.cull.icb = None;
        self.cull.icb_arg_buffer = None;
        self.cull.icb_capacity = 0;
        self.cull.icb_2 = None;
        self.cull.icb_2_arg_buffer = None;
        self.cull.status_buffer = None;

        if let Some(ps) = new_instanced {
            self.instanced_pipeline_state = Some(ps);
        }
        if let Some(ps) = new_skinned_main {
            self.skinned.pipeline_state = Some(ps);
        }
        Ok(())
    }
}
