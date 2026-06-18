// src/metal/quality.rs
//
// Runtime application of the Quality-group settings (TAA / SSAO / SSR / RT
// reflections / SSGI / auto-exposure). Each gates a render pass whose GPU
// resources (pipelines, render targets, the ray-tracing acceleration structure)
// are built once at init from the world's PostProcessConfig, so applying a
// change at runtime means rebuilding those resources, not flipping a uniform.
//
// The rebuild reuses `init::effects::build_quality_effects` -- the exact path
// `MtlContext::new` runs -- so a live toggle produces resources byte-identical
// to a launch with the same config. Only the toggle-controlled subset is rebuilt;
// bloom, decals, fog, particles, and the uploaded geometry are untouched (so no
// particle-sim reset and no multi-second geometry re-upload).

#![allow(clippy::incompatible_msrv)]

use crate::gfx::backend::QualitySettings;

use super::context::MtlContext;
use super::init::effects::{QualityEffectsBundle, build_quality_effects};
use super::init::pipelines::make_vertex_descriptor;
use super::post::build_gbuffer_prepass_pipeline;
use super::raytrace::{build_rt_accel, raytracing_supported};
use super::resources::skinning::make_skinned_vertex_descriptor;

impl MtlContext {
    // Rebuild the toggle-controlled effects in place to match `q`, applied
    // between frames (the GraphicsSystem drain runs before the next
    // `draw_frame`). A build failure logs and leaves the prior state intact.
    pub(crate) fn apply_quality_settings(&mut self, q: QualitySettings) {
        // RT reflections only when the GPU supports hardware ray tracing;
        // otherwise the toggle persists + value-syncs but renders nothing,
        // matching the init-time fallback.
        let rt_settings = q
            .rt_reflections
            .filter(|_| raytracing_supported(&self.device));

        // TAA is bypassed while the MetalFX upscaler is active (the scaler does
        // its own temporal accumulation); the velocity pre-pass + G-buffer are
        // needed when TAA is effectively on OR the upscaler is active. Mirrors
        // the `effective_taa_enabled` / `velocity_needed` derivation in
        // `MtlContext::new`. Render dimensions come from the live HDR targets
        // (render-resolution, already post-upscale).
        let upscaling_active = self.upscale.scaler.is_some();
        let taa_effective = q.taa && !upscaling_active;
        let needs_velocity = taa_effective || upscaling_active;
        let has_instanced = self.instanced_pipeline_state.is_some();
        let render_w = self.hdr_targets.width;
        let render_h = self.hdr_targets.height;

        let bundle = match build_quality_effects(
            &self.device,
            &make_vertex_descriptor(),
            render_w,
            render_h,
            taa_effective,
            needs_velocity,
            has_instanced,
            &q.ssao,
            &q.ssr,
            &q.ssgi,
            &rt_settings,
            &q.auto_exposure,
            q.auto_exposure_bias_ev,
            self.hot_reload,
        ) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("apply_quality_settings: effect rebuild failed: {e}");
                return;
            }
        };

        let QualityEffectsBundle {
            taa_pipeline_state,
            taa_targets,
            ssao,
            transient_pool,
            ssr,
            mut gbuffer,
            ssgi,
            rt_pipeline,
            rt_pipeline_textured,
            rt_skin_pipeline,
            auto_exposure_pipelines,
            auto_exposure_histogram,
            auto_exposure_output,
            auto_exposure_state,
            auto_exposure_bias_ev,
        } = bundle;

        // Re-attach the 80-byte skinned G-buffer pre-pass pipeline when the world
        // has skinned meshes and the G-buffer is now built (`build_quality_effects`
        // leaves it `None`, like the init path, which fills it in `upload_skinned`).
        if gbuffer.targets.is_some() && self.skinned.vertex_buffer.is_some() {
            match build_gbuffer_prepass_pipeline(
                &self.device,
                &make_skinned_vertex_descriptor(),
                "gbuffer_prepass_vertex_skinned",
                self.hot_reload,
            ) {
                Ok(p) => gbuffer.skinned_pipeline = Some(p),
                Err(e) => {
                    tracing::error!("apply_quality_settings: skinned G-buffer pipeline: {e}")
                }
            }
        }

        // Swap the screen-space feature state in. The old `Retained` targets drop
        // here; any in-flight command buffer still referencing them holds its own
        // Metal retain until the GPU retires the frame, so the swap is safe
        // between frames. The render graph is rebuilt from these gates every
        // frame (no cached graph to invalidate).
        self.taa.enabled = taa_effective;
        self.taa.pipeline_state = taa_pipeline_state;
        self.taa.targets = taa_targets;
        self.taa.dst = 0;
        // History is stale after a rebuild; the first frame passes through.
        self.taa.history_valid = false;
        self.ssao = ssao;
        self.transient_pool = transient_pool;
        self.ssr = ssr;
        self.gbuffer = gbuffer;
        self.ssgi = ssgi;

        // RT resolve pipelines come from the rebuild; the acceleration structure
        // is built here (it needs the resident geometry buffers) when RT turns
        // on, and dropped when it turns off. Skinned geometry is seeded into the
        // BVH by the next frame's per-frame update, matching the init path.
        self.rt.settings = rt_settings;
        self.rt.pipeline = rt_pipeline;
        self.rt.pipeline_textured = rt_pipeline_textured;
        self.rt.skin_pipeline = rt_skin_pipeline;
        if self.rt.settings.is_some() {
            if self.rt.accel.is_none() {
                match build_rt_accel(
                    &self.device,
                    &self.command_queue,
                    &self.vertex_buffer,
                    &self.index_buffer,
                    &self.draw_objects,
                    &self.instanced_clusters,
                    self.textures.len(),
                    self.normal_map_textures.len(),
                    None,
                ) {
                    Ok(Some(a)) => {
                        tracing::info!(
                            "ray-traced reflections: built BVH over {} static objects",
                            a.blas.len()
                        );
                        self.rt.accel = Some(a);
                    }
                    Ok(None) => tracing::warn!(
                        "ray-traced reflections toggled on but the scene has no static geometry; no BVH built"
                    ),
                    Err(e) => tracing::error!("apply_quality_settings: RT accel build: {e}"),
                }
            }
        } else {
            self.rt.accel = None;
        }
        // Reset the failure streak so a later toggle-on starts clean.
        self.rt.update_failed = false;

        // Auto-exposure. When it turns off the static path uses
        // `self.post_process.exposure` (the authored / slider EV), already set,
        // so only the GPU state is swapped here.
        self.auto_exposure.settings = q.auto_exposure;
        self.auto_exposure.state = auto_exposure_state;
        self.auto_exposure.bias_ev = auto_exposure_bias_ev;
        self.auto_exposure.pipelines = auto_exposure_pipelines;
        self.auto_exposure.histogram = auto_exposure_histogram;
        self.auto_exposure.output = auto_exposure_output;
    }
}
