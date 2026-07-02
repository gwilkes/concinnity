// src/metal/draw/mod.rs
//
// `MtlContext::draw_frame` -- the per-frame orchestration. The per-pass GPU
// encoders live in sibling files:
//
//   shadow.rs    cascaded shadow map (depth-only, one render pass per cascade)
//   main.rs      main HDR pass + bindless/legacy/instanced/skinned static
//                geometry
//   composite.rs ACES tonemap + FXAA composite + text overlay
//
// Other passes (SSAO, SSR pre + resolve, decals, fog, velocity, TAA, bloom,
// auto-exposure) live in their own files at the `metal/` level alongside
// `decal.rs`, `fog.rs`, `post.rs`, etc., and are invoked through the
// `self.encode_*` methods defined there.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

mod composite;
mod main;
mod shadow;

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer as _, MTLCommandQueue as _, MTLDevice as _};

use crate::gfx::render_graph::FrameGraphInputs;
use crate::gfx::render_types::TextDrawCall;

use super::context::MtlContext;
use super::graph_exec::GraphFrameParams;
use super::math::{mat4_mul, perspective};
use super::uniforms::*;

// One term of the Halton low-discrepancy sequence. Used to drive the
// sub-pixel projection jitter so successive frames sample slightly different
// positions for the TAA pass to accumulate.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

// Diagnostic toggle (`CN_RT_NOSKIN=1`): keep skinned geometry out of the RT
// reflection BVH entirely (static + instanced clusters only). Read once. Lets us
// confirm whether the skinned trace path is what page-faults the reflection pass.
fn rt_no_skin() -> bool {
    static NO_SKIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NO_SKIN.get_or_init(|| std::env::var("CN_RT_NOSKIN").is_ok())
}
impl MtlContext {
    // Pump the NSEvent queue and encode one frame to the GPU.
    //
    // NSEvent processing must happen here rather than in the run loop because
    // window close, resize, and key events are only delivered after NSApp
    // dequeues them. CFRunLoopRunInMode alone does not dispatch NSEvents.
    //
    // The whole frame runs inside a fresh autorelease pool. The render loop is
    // a tight Rust loop with no Cocoa run-loop pool of its own, so without this
    // the autoreleased per-frame Metal objects (command buffers, which retain
    // every resource they reference until they are released, plus encoders,
    // descriptors, and `NSArray`s) would accumulate in the never-drained outer
    // pool. That keeps each frame's transient buffers / acceleration structures
    // alive even after they are replaced on the context, so
    // `device.currentAllocatedSize()` climbs every frame (faster at higher FPS)
    // until unified memory is exhausted and the GPU faults / the host panics.
    // Draining per frame frees each frame's command buffers once the GPU
    // retires them, bounding VRAM to the work actually in flight.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_frame(
        &mut self,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
        world_hidden: bool,
    ) -> Result<(), String> {
        objc2::rc::autoreleasepool(|_| {
            self.draw_frame_inner(
                elapsed,
                fov_y_radians,
                near,
                far,
                cam_pos,
                text_calls,
                world_hidden,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_frame_inner(
        &mut self,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
        world_hidden: bool,
    ) -> Result<(), String> {
        let mtm = objc2::MainThreadMarker::new()
            .ok_or("draw_frame must be called from the main thread")?;

        // Reset this frame's render stats; the draw counters below accumulate
        // into `frame_stats`, and `render_stats()` reports them (plus the GPU
        // frame time) to the profiler overlay. `objects` is the total scene
        // size: static draw objects, every instanced-cluster instance, and
        // skinned meshes.
        self.frame_stats = crate::gfx::profile::RenderStats::default();
        let instanced_total: usize = self
            .instanced_clusters
            .iter()
            .map(|c| c.instances.len())
            .sum();
        self.frame_stats.objects =
            (self.draw_objects.len() + instanced_total + self.skinned.draw_objects.len()) as u32;
        // Live skinned count: authored meshes plus runtime-spawned instances,
        // excluding the hidden pre-reserved pool slots. `objects` above counts
        // the whole pool and so stays flat across skinned spawn/despawn; this
        // tracks the visible count, so a spawn bumps it and a despawn drops it.
        self.frame_stats.skinned_visible = self
            .skinned
            .draw_objects
            .iter()
            .filter(|o| o.visible)
            .count() as u32;
        self.frame_stats.skinned_pool_free = self.skinned_pool.total_free() as u32;
        // Current GPU memory footprint. On Apple Silicon's unified memory this
        // is the Metal device's allocation within system RAM.
        self.frame_stats.vram_bytes = self.device.currentAllocatedSize() as u64;

        // Rotate the per-frame sample-buffer slot if per-pass GPU timing is
        // available. Every `pass_timing.attach_*` call this frame writes
        // into the same slot's buffer; the completion handler resolves it
        // after the frame retires.
        let pass_timing_slot = self
            .pass_timing
            .as_mut()
            .map(|p| p.begin_frame())
            .unwrap_or(0);

        // Drain all pending NSEvents so the window stays responsive. The
        // preview tab leaves `pump_events` false so SwiftUI owns event
        // delivery (pumping there would dequeue mouse clicks meant for the
        // tab bar before they reach their targets); the windowed CLI path
        // and the blocking-in-view play path opt in.
        if self.pump_events {
            self.pump_ns_events(mtm);
            if self.window_closed() {
                return Ok(());
            }
        }

        // Frames-in-flight gate: block until the GPU has retired an older frame
        // so the CPU never queues more than `frames_in_flight` frames ahead,
        // bounding how many sets of per-frame transient buffers pile up. Taken
        // here, before drawable prep and all per-frame buffer building. The slot
        // is handed to the frame command buffer's completion handler below
        // (released on GPU retirement); if this frame is abandoned before commit
        // (the drawable isn't ready, or a `?` fails mid-encode) `frame_slot`
        // drops and releases the slot synchronously, keeping the count balanced.
        let frame_slot = self.frame_pacing.acquire();

        // Asynchronous reflection-probe bake. One probe at a time, advanced across
        // frames, captures real geometry into `probe_maps` (the sky `env_map` is
        // left untouched) so glossy surfaces reflect their surroundings instead of a
        // foreign HDR; unbaked probes fall back to the sky until installed. The
        // render thread never blocks on it: the six faces are submitted without
        // `waitUntilCompleted` (into a reserved ring slot the frame never overwrites)
        // and the prefilter convolution runs on a worker thread. Runs AFTER
        // `acquire()` so its reserved-slot retire-pool collection sees a fence-
        // consistent frame id. Non-fatal: a failure keeps the current state.
        // Skipped while the world is hidden: a probe bake feeds reflections no
        // pass will sample this frame.
        if !world_hidden && let Err(e) = self.bake_pending_probes(elapsed, near, far) {
            tracing::warn!("reflection probe bake failed, keeping current environment: {e}");
        }

        // tell MTKView to prepare its drawable for this frame
        self.mtk_view.draw();

        let drawable = match self.mtk_view.currentDrawable() {
            Some(d) => d,
            // drawable not yet available -- skip this frame silently
            None => return Ok(()),
        };
        self.was_visible = true;

        // This frame's transient-buffer ring slot. The fence guarantees the
        // frame that last used `frame_ring_index - frames_in_flight` has retired
        // on the GPU, so overwriting this slot's buffers can't race an in-flight
        // read. Advanced once per built frame; skipped frames (no drawable) bail
        // above without consuming a slot.
        let frame_id = self.frame_ring_index;
        let ring_slot = (frame_id % self.frames_in_flight as u64) as usize;
        self.frame_ring_index = frame_id.wrapping_add(1);

        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or("failed to get command buffer")?;

        // Shader hot-reload: if either the filesystem watcher or the debug
        // `reload-shaders` command set the flag, rebuild every built-in
        // pipeline from disk-resident source before the frame's passes start
        // using them. The flag is cleared regardless of outcome so a failed
        // rebuild (typo in a shader edit) doesn't loop, and the previous
        // pipelines stay live so the session keeps rendering.
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }

        // Update auto-exposure from the previous frame's GPU-measured average
        // log-luminance, *before* any pass reads `self.post_process.exposure`
        // (the bloom prefilter and composite both consume it). A no-op when
        // auto-exposure is disabled -- the static authored EV then drives the
        // exposure multiplier unchanged.
        self.update_auto_exposure(elapsed);

        // Transient per-frame GPU buffers holding each skinned object's joint
        // matrices. Built once and reused across the shadow cascades and the
        // main pass. Empty when no SkinnedMesh is in the world.
        let skinned_joint_bufs = self.build_joint_buffers(ring_slot)?;

        // Previous-frame joint matrices, used only by the velocity pre-pass to
        // capture skinned deformation. Empty when TAA is disabled.
        let prev_skinned_joint_bufs = if self.taa.enabled {
            self.build_prev_joint_buffers(ring_slot)?
        } else {
            Vec::new()
        };

        // Compute per-frame cascade VPs + splits from current camera + light.
        // The aspect/near/far are taken from the same params used by the main
        // perspective below so cascades match the visible camera frustum.
        let cascade_aspect = {
            let s = self.mtk_view.drawableSize();
            if s.height == 0.0 {
                1.0
            } else {
                (s.width / s.height) as f32
            }
        };
        if self.shadow_pipeline_state.is_some() {
            let fresh = crate::gfx::csm::compute_shadow_uniforms(
                self.view_matrix,
                cam_pos,
                fov_y_radians,
                cascade_aspect,
                near,
                (self.shadow_distance as f32).min(far),
                self.shadow_light_dir,
                self.shadow_map_size,
                self.shadow_cascades,
            );
            // Pick this frame's cascades and refresh only their VPs; cascades
            // skipped this frame keep the VP their slice was rendered with so
            // the Main pass samples each slice consistently. Splits depend only
            // on the camera near/far range (not position), so always take fresh.
            let mask = self.next_shadow_cascade_mask();
            self.shadow_render_mask = mask;
            self.shadow_uniforms.cascade_splits = fresh.cascade_splits;
            self.shadow_uniforms.active_cascades = fresh.active_cascades;
            for i in 0..crate::gfx::render_types::NUM_SHADOW_CASCADES {
                if mask & (1u32 << i) != 0 {
                    self.shadow_uniforms.light_vps[i] = fresh.light_vps[i];
                }
            }
        }

        // Main pass prep: resize off-screen targets, then the visible set.
        // Resize the HDR targets if the drawable size changed (window resize
        // or initial layout). The drawable was just refreshed by mtk_view.draw().
        let draw_size = self.mtk_view.drawableSize();
        // Geometry-less worlds keep their off-screen targets pinned at 1x1
        // (see MtlContext::new); the composite pass still uses the full drawable.
        let (want_w, want_h) = if self.geometry_less {
            (1, 1)
        } else {
            (
                draw_size.width.max(1.0) as u32,
                draw_size.height.max(1.0) as u32,
            )
        };
        self.resize_targets_if_needed(want_w, want_h)?;

        // Render resolution: where the 3D scene + most post passes draw.
        // Equals `want_w/h` (the drawable size) when no upscaler is active;
        // otherwise it's smaller, so the upscaler reconstructs back up to
        // drawable size.
        let render_w = self.hdr_targets.width;
        let render_h = self.hdr_targets.height;

        // View-projection + GPU-driven cull.
        // The projection / jitter / VP are resolved here, ahead of the main
        // render encoder, because the cull compute pass needs the frustum
        // before the render pass begins.
        let aspect = cascade_aspect;
        let proj = perspective(fov_y_radians, aspect, near, far);
        // This frame's un-jittered VP, captured before the graph runs so the
        // two-pass phase-2 cull (`encode_cull_phase2`, dispatched inside
        // `execute_graph`) can project AABBs through it against the pyramid the
        // mid-frame `HizBuild` rebuilds from this frame's depth. The same value
        // becomes `cull_prev_view_proj` at end-of-frame for next frame's phase 1.
        self.cull.cur_view_proj = mat4_mul(proj, self.view_matrix);
        // When TAA or the MetalFX upscaler is on, offset the projection by
        // a sub-pixel Halton jitter so the temporal accumulator has fresh
        // sample positions each frame. The jitter is a pure NDC x/y shift,
        // so depth is unaffected. `proj[2][0/1]` are the z-coefficients of
        // clip x/y; subtracting the jitter there shifts post-divide NDC by
        // exactly the jitter amount (clip.w == -view_z). Pixel-space
        // jitter (`±0.5` per axis) is stashed for MetalFX, which expects
        // its input in pixel coords; TAA reads NDC directly.
        let needs_jitter = self.taa.enabled || self.upscale.scaler.is_some();
        let proj_render = if needs_jitter {
            let idx = self.taa.frame % 8 + 1;
            let jx_pix = halton(idx, 2) - 0.5;
            let jy_pix = halton(idx, 3) - 0.5;
            let jx = jx_pix * 2.0 / render_w as f32;
            let jy = jy_pix * 2.0 / render_h as f32;
            if self.upscale.scaler.is_some() {
                self.upscale
                    .jitter
                    .store(jx_pix, jy_pix, std::sync::atomic::Ordering::Release);
            }
            let mut p = proj;
            p[2][0] -= jx;
            p[2][1] -= jy;
            p
        } else {
            proj
        };
        let vp = mat4_mul(proj_render, self.view_matrix);
        // Inverse of the (jittered) view-projection, computed once here and
        // threaded through `GraphFrameParams` to every pass that reconstructs a
        // world-space position from depth (fog, decals, raymarch, transparent),
        // instead of each pass re-inverting `vp` independently.
        let inv_vp = super::math::mat4_inverse(vp);
        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp);

        // Resolve the visible set for the legacy CPU draw path, the TAA
        // velocity pre-pass, and the SSAO pre-pass: BVH-culled cullable
        // objects, then the always-draw fallback (skybox, rooms, dynamic
        // props). The bindless static pass instead consumes the GPU-culled
        // indirect command buffer, so it never walks this list.
        // mem::take swaps out the persistent scratch buffer so its heap
        // allocation is reused across frames; it's put back below before we
        // return Ok (the error path silently loses the capacity, which is
        // acceptable since draw_frame errors are exceptional).
        // mem::take swaps out the persistent scratch buffer so its heap
        // allocation is reused across frames; it's put back below before we
        // return Ok. While the world is hidden behind an opaque menu, the
        // surviving Main pass is fed an empty scene -- no visible set, no
        // bindless object / cull / texture buffers, no instanced clusters, and
        // no acceleration-structure refresh -- so it runs as a bare clear that
        // the opaque overlay then covers. The masked graph drops every other
        // world pass, so none of this work would be consumed anyway.
        let mut visible = std::mem::take(&mut self.visible_scratch);
        visible.clear();
        let (object_buffer, cull_draw_args, bindless_tex_args, prepared_instances) = if world_hidden
        {
            (
                None,
                None,
                None,
                super::instanced::PreparedInstances {
                    clusters: Vec::new(),
                },
            )
        } else {
            // Resolve the visible set for the legacy CPU draw path, the TAA
            // velocity pre-pass, and the SSAO pre-pass: BVH-culled cullable
            // objects, then the always-draw fallback (skybox, rooms, dynamic
            // props). The bindless static pass instead consumes the GPU-culled
            // indirect command buffer, so it never walks this list.
            self.cull_bvh
                .query(&frustum, cam_pos, |idx| visible.push(idx));
            visible.sort_unstable();
            visible.extend_from_slice(&self.always_draw);

            // Per-frame GPU buffer prep for the bindless path.
            // The object data + indirect-args + bindless texture argbuf are
            // all per-frame Metal buffers the bindless Main pass + Cull
            // compute pass consume. They must outlive the command buffer,
            // hence the bindings kept here through to `cmd_buf.commit()`.
            let object_buffer = if self.bindless {
                self.build_object_buffer(ring_slot)?
            } else {
                None
            };
            let cull_draw_args = if object_buffer.is_some() {
                let draw_args = self.build_draw_args_buffer(cam_pos, ring_slot)?;
                if draw_args.is_some() {
                    self.ensure_icb_capacity(self.cull_count())?;
                    // GPU-driven cascaded shadow: size the per-cascade
                    // shadow ICB to NUM_SHADOW_CASCADES * cull_count. A no-op when
                    // the shadow-bindless path is inactive (no shadow cull encoder).
                    self.ensure_shadow_icb_capacity(self.cull_count())?;
                    // Per-planar-slot mirror cull ICBs: one per distinct reflection
                    // plane, each sized to cull_count. A no-op (clears the slots) when
                    // the world has no planar set (RT on, or no flat reflectors).
                    let mirror_slots = self
                        .planar_reflection
                        .as_ref()
                        .map(|s| s.planes.len())
                        .unwrap_or(0);
                    self.ensure_mirror_icb_capacity(mirror_slots, self.cull_count())?;
                }
                draw_args
            } else {
                None
            };
            let bindless_tex_args = if object_buffer.is_some() {
                self.build_bindless_texture_args(ring_slot)?
            } else {
                None
            };

            // Cull + LOD-bucket + upload every instanced cluster ONCE here, on the
            // main thread before the pass fan-out. The main / SSR / SSAO / velocity
            // passes share this prepared set instead of each repeating the cull and
            // re-uploading the instance matrices to their own transient buffer.
            let prepared_instances = self.prepare_instanced_draws(ring_slot, cam_pos, &frustum)?;

            // Keep the RT acceleration structure current with this frame's
            // transforms before any pass reads `rt_accel`. The default `Auto` mode
            // rebuilds the TLAS only when a participating prop actually moved; a
            // fully static scene pays just a matrix compare here. Non-fatal: a
            // transient rebuild failure keeps last frame's BVH rather than stopping
            // the renderer.
            self.rt_dynamic_update(frame_id);

            (
                object_buffer,
                cull_draw_args,
                bindless_tex_args,
                prepared_instances,
            )
        };

        // Per-frame pass uniforms hoisted upfront.
        // Every pass that needs a struct of per-frame params builds its
        // uniforms here so a single GraphFrameParams below can carry
        // the union into `execute_graph`.
        let ssao_params = self
            .ssao
            .settings
            .map(|settings| settings.params(fov_y_radians, aspect));
        let ssr_params = self.ssr.settings.map(|settings| {
            let v = self.view_matrix;
            let inv_view_rot = [
                [v[0][0], v[1][0], v[2][0], 0.0],
                [v[0][1], v[1][1], v[2][1], 0.0],
                [v[0][2], v[1][2], v[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
            settings.params(
                fov_y_radians,
                aspect,
                inv_view_rot,
                cam_pos,
                prefilter_mip_count,
            )
        });
        let ssgi_params = self
            .ssgi
            .settings
            .map(|settings| settings.params(fov_y_radians, aspect));
        // RT-reflection params: built only when the acceleration structure is
        // live (so they stay in lockstep with `rt_reflections_enabled`). Carries
        // the camera-to-world transform + sun the kernel shades hits with, like
        // SSR's params plus the world-space camera + sun.
        let rt_reflection_params =
            self.rt
                .settings
                .filter(|_| self.rt.accel.is_some())
                .map(|settings| {
                    let v = self.view_matrix;
                    let inv_view_rot = [
                        [v[0][0], v[1][0], v[2][0], 0.0],
                        [v[0][1], v[1][1], v[2][1], 0.0],
                        [v[0][2], v[1][2], v[2][2], 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    let prefilter_mip_count = self.env_map.prefilter_mip_count as f32;
                    let sun = &self.light_uniforms.directional[0];
                    let sun_color = [
                        sun.color[0] * sun.intensity,
                        sun.color[1] * sun.intensity,
                        sun.color[2] * sun.intensity,
                    ];
                    settings.params(
                        fov_y_radians,
                        aspect,
                        inv_view_rot,
                        cam_pos,
                        sun.direction,
                        sun_color,
                        prefilter_mip_count,
                    )
                });
        let fog_params = self.fog.settings.map(|fog| {
            // Sun = the first directional light; falls back to the
            // LightUniforms::DEFAULT direction if the world declared none.
            let sun = &self.light_uniforms.directional[0];
            let sun_color = [
                sun.color[0] * sun.intensity,
                sun.color[1] * sun.intensity,
                sun.color[2] * sun.intensity,
            ];
            // Fog renders into hdr_resolve, which is render-resolution
            // when the upscaler is on. The fog shader uses the viewport
            // to reconstruct world position from screen UV, so it must
            // match the actual render target's pixel grid.
            let viewport = [render_w as f32, render_h as f32];
            // Reconstruct the froxel volume with the UN-jittered view-projection.
            // Fog is volumetric, so its screen-space contribution does not follow
            // the surface motion vectors TAA reprojects by. Feeding it the jittered
            // inv_vp shifts the whole volume sub-pixel every frame; on a large
            // smooth low-contrast surface, where the fog is the dominant
            // high-frequency signal, TAA cannot reconcile that per-frame shift with
            // the jitter-free history, so the fog flickers (a moving moire). The
            // un-jittered inv_vp keeps the volume stable frame to frame; its offset
            // versus the jittered depth buffer is far below the coarse froxel grid.
            let fog_inv_vp = super::math::mat4_inverse(mat4_mul(proj, self.view_matrix));
            fog.params(fog_inv_vp, cam_pos, sun.direction, sun_color, viewport)
        });
        // FogFroxel volume extras: view matrix + volume dimensions + near/far
        // so the compute kernel can place each froxel in world-space and the
        // fragment shader can map a scene depth into the volume's Z axis.
        let fog_froxel_params =
            self.fog
                .settings
                .map(|fog| crate::gfx::render_types::FogFroxelParams {
                    view: self.view_matrix,
                    froxel_dims: [
                        crate::gfx::render_graph::FOG_FROXEL_X,
                        crate::gfx::render_graph::FOG_FROXEL_Y,
                        crate::gfx::render_graph::FOG_FROXEL_Z,
                    ],
                    _pad_align: 0,
                    z_near: near.max(1e-3),
                    z_far: fog.max_distance,
                    _pad: [0.0; 2],
                });
        // Velocity (motion vectors in the G-buffer pre-pass) is needed whenever
        // temporal reconstruction runs: that's TAA or the MetalFX upscaler.
        let velocity_active = self.taa.enabled || self.upscale.scaler.is_some();
        let vel_uniforms = if velocity_active {
            Some(VelocityUniforms {
                jittered_vp: vp,
                cur_vp: mat4_mul(proj, self.view_matrix),
                prev_vp: self.prev_view_proj,
            })
        } else {
            None
        };
        let taa_uniforms = if self.taa.enabled {
            Some(TaaUniforms {
                history_valid: if self.taa.history_valid { 1.0 } else { 0.0 },
                _pad0: 0.0,
                _pad1: [0.0; 2],
            })
        } else {
            None
        };

        // `scene_input` is the engine-owned texture the post-decoration
        // stack treats as the pre-TAA scene: the SSR (or RT-reflection)
        // resolve output when on, else the raw `hdr_resolve`. Both SSR resolve
        // and the RT-reflection pass write `ssr_targets.output`. `scene_color`
        // is what Bloom + Composite read:
        //   - the upscaler's output (drawable-res) when MetalFX is on,
        //   - the TAA resolve target when TAA is on,
        //   - otherwise just the pre-TAA scene (no temporal stage).
        let scene_input = if self.ssr.settings.is_some() || self.rt.accel.is_some() {
            self.ssr
                .targets
                .as_ref()
                .ok_or("reflections enabled but SSR targets missing")?
                .output
                .clone()
        } else {
            self.hdr_targets.hdr_resolve.clone()
        };
        let scene_color = if let Some(u) = &self.upscale.scaler {
            u.output.clone()
        } else if self.taa.enabled {
            self.taa.targets[self.taa.dst].clone()
        } else {
            scene_input.clone()
        };

        // The transparent pass runs when any translucent producer is live.
        // Drives both the graph-input gate (whether the slot is inserted) and
        // the `scene_pre_taa` supply below (the pass reads + writes it). With
        // SSR off `scene_input` aliases `hdr_resolve`, which is the correct
        // RMW target: the transparent encoder blits a scene copy first, so the
        // self-read for refraction is safe.
        let transparent_active = (self.water_pipeline.is_some() && !self.water_surfaces.is_empty())
            || (self.glass_pipeline.is_some() && !self.glass_panels.is_empty());

        // Single render-graph dispatch for the full frame.
        // The merged graph contains every Metal pass that once ran
        // inline through `draw_frame`. The compile pass derives
        // execution order, per-pass barriers, and resource lifetimes
        // from the RAW + WAW + WAR edges over the version-chained
        // read / write declarations in `build_frame_graph`. Composite
        // is the presenter and runs last; the drawable is fetched at
        // frame start above and stays alive through `presentDrawable`
        // below.
        let graph_inputs = FrameGraphInputs {
            shadow_enabled: self.shadow_pipeline_state.is_some(),
            shadow_map_size: self.shadow_map_size,
            hdr_width: self.hdr_targets.width,
            hdr_height: self.hdr_targets.height,
            hdr_sample_count: super::context::HDR_SAMPLE_COUNT,
            bindless_cull_enabled: object_buffer.is_some() && cull_draw_args.is_some(),
            auto_exposure_enabled: self.auto_exposure.pipelines.is_some(),
            // Gated on the pipelines existing: a scene-less world builds none
            // (its 1x1 bloom targets stay untouched black).
            bloom_enabled: self.post_process.bloom_intensity > 0.0
                && self.bloom_pipelines.is_some(),
            // Velocity runs whenever its targets exist: that's TAA on or
            // the upscaler on. The graph builder adds the Velocity pass
            // when this flag is true; TaaResolve / Upscale then declare a
            // read edge on it for ordering.
            velocity_enabled: velocity_active,
            taa_enabled: self.taa.enabled,
            ssr_enabled: self.ssr.settings.is_some(),
            particles_enabled: self.particle.pipelines.is_some()
                && !self.particle.records.is_empty()
                && !self.particle.emitter_state.is_empty(),
            fog_enabled: self.fog.pipeline.is_some() && self.fog.settings.is_some(),
            decals_enabled: self.decal.pipeline.is_some() && !self.decal.records.is_empty(),
            // The SSR depth + normal + roughness pre-pass also feeds SSGI and
            // the RT-reflection kernel, so it runs when SSR, SSGI, *or* RT
            // reflections are on (RT keys off the live acceleration structure).
            ssr_prepass_enabled: self.ssr.settings.is_some()
                || self.ssgi.settings.is_some()
                || self.rt.accel.is_some(),
            ssao_enabled: self.ssao.settings.is_some(),
            upscale_enabled: self.upscale.scaler.is_some(),
            // Transparent pass runs when at least one translucent producer
            // (`WaterSurface` or `GlassPanel`) exists; the executor
            // short-circuits an empty draw list, but gating here keeps the
            // graph builder from inserting the slot at all.
            transparent_enabled: transparent_active,
            // Raymarch runs when at least one `SdfVolume` is live; the
            // per-volume pipeline cache is populated in lockstep with
            // the volume vec at init. Tightened to the real `is_some()
            // && !empty()` predicate once the context fields land
            // alongside `encode_raymarch` (see metal/raymarch.rs).
            raymarch_enabled: !self.raymarch_volumes.is_empty(),
            // Two-pass Hi-Z occlusion. Resolved from
            // `PostProcessConfig.occlusion_two_pass` (and gated at init on the
            // bindless cull path existing). The graph builder further ANDs this
            // with `bindless_cull_enabled` for this frame, so a frame with no
            // static geometry simply runs single-pass. When on, the builder
            // inserts HizBuild → Cull2 → Main2 between Main and the post chain.
            two_pass_occlusion_enabled: self.cull.two_pass_occlusion,
            // SSGI runs when `indirect_lighting: "ssgi"` resolved settings.
            // The builder inserts the Ssgi RMW pass after Raymarch on the
            // hdr_resolve chain; the gather reads the SSR pre-pass G-buffer
            // (forced on above via `ssr_prepass_enabled`).
            ssgi_enabled: self.ssgi.settings.is_some(),
            // RT reflections run when the scene acceleration structure is live
            // (RT requested + GPU supports it + scene has geometry). The builder
            // inserts the RtReflections pass in the SsrResolve slot and, when
            // both are on, picks it over SsrResolve (RT takes precedence; SSR is
            // the cross-backend fallback).
            rt_reflections_enabled: self.rt.accel.is_some(),
            // Metal collapses the SSR / SSAO / velocity pre-passes into one
            // GBufferPrepass node; the other backends keep them separate.
            unified_gbuffer_prepass: true,
            // An opaque menu backdrop hides the scene: the builder masks every
            // world pass off, collapsing to Main (a bare clear, fed the empty
            // scene above) -> Composite (presents the overlay).
            world_hidden,
        };
        let graph = crate::gfx::render_graph::build_frame_graph(&graph_inputs)
            .map_err(|e| format!("frame graph: {}", e))?;
        // This frame's skinned deformed-vertex buffer (skinned fold), cloned into
        // a local so `params` owns a handle rather than borrowing `self.skinned`
        // across the `&mut self` execute_graph call (every other GraphFrameParams
        // buffer is likewise a local). `Some` only when the fold is active
        // (n_skinned > 0, set in upload_skinned under bindless + static geometry);
        // the Cull pass writes it via encode_main_skin and the Main / Main2
        // skinned ICB tail binds it.
        let deformed_this_frame = if self.n_skinned > 0 {
            self.skinned.deformed.get(ring_slot).cloned()
        } else {
            None
        };
        // The previous frame's deformed slot (one behind in the ring), read by
        // the GPU-driven G-buffer skinned tail for per-vertex skin motion. The
        // priming gate (`deformed_primed`) covers the unposed first frame.
        let deformed_prev_frame = if self.n_skinned > 0 {
            let prev_slot = (ring_slot + self.frames_in_flight - 1) % self.frames_in_flight;
            self.skinned.deformed.get(prev_slot).cloned()
        } else {
            None
        };
        // Per-frame parallel prev_model buffer for the GPU-driven G-buffer pass.
        // Built only when that path will run (bindless object buffer +
        // a G-buffer consumer + the bindless G-buffer pipeline), indexed
        // identically to the object buffer.
        let prev_model_buffer = if object_buffer.is_some()
            && self.gbuffer.targets.is_some()
            && self.gbuffer.bindless_pipeline.is_some()
        {
            self.build_gbuffer_prev_models(ring_slot, velocity_active)?
        } else {
            None
        };
        let params = GraphFrameParams {
            cmd_buf: &cmd_buf,
            cam_pos,
            skinned_joint_bufs: &skinned_joint_bufs,
            scene_color: Some(&scene_color),
            text_calls,
            world_hidden,
            elapsed,
            vp,
            inv_vp,
            visible: &visible,
            frustum: &frustum,
            prepared_instances: &prepared_instances,
            object_buffer: object_buffer.as_ref(),
            bindless_tex_args: bindless_tex_args.as_ref(),
            deformed_skinned: deformed_this_frame.as_ref(),
            deformed_prev: deformed_prev_frame.as_ref(),
            prev_model_buffer: prev_model_buffer.as_ref(),
            draw_args_buffer: cull_draw_args.as_ref(),
            vel_uniforms: vel_uniforms.as_ref(),
            prev_skinned_joint_bufs: &prev_skinned_joint_bufs,
            taa_uniforms: taa_uniforms.as_ref(),
            scene_pre_taa: if self.taa.enabled
                || self.upscale.scaler.is_some()
                || transparent_active
            {
                Some(&scene_input)
            } else {
                None
            },
            ssr_params: ssr_params.as_ref(),
            fog_params: fog_params.as_ref(),
            fog_froxel_params: fog_froxel_params.as_ref(),
            ssao_params: ssao_params.as_ref(),
            ssgi_params: ssgi_params.as_ref(),
            rt_reflection_params: rt_reflection_params.as_ref(),
        };
        self.execute_graph(&graph, &params)?;

        // Hi-Z pyramid: reduce this frame's main depth buffer into the mip
        // chain the *next* frame's phase-1 cull dispatch consults. Encoded on
        // the outer command buffer here, after `execute_graph` (which already
        // committed every pass's cmd buf, so the depth attachment is written)
        // and before present, so it stays off the per-pass worker fan-out
        // while still executing after the last main pass. A no-op when no Hi-Z
        // resource was built (bindless cull pipeline not active). Under
        // two-pass occlusion this reduces the *final* (post-Main2) depth, and
        // the graph's mid-frame `HizBuild` already rebuilt the pyramid that fed
        // this frame's Cull2: this end-of-frame build supersedes it for the
        // next frame. The un-jittered VP captured at the top of the frame
        // becomes `cull_prev_view_proj` so next frame's projection lines up
        // (distinct from the velocity pre-pass's `prev_view_proj`, which only
        // advances when velocity runs).
        if self.cull.hiz.is_some() {
            self.encode_hiz_build(&cmd_buf);
            self.cull.hiz_valid = true;
            self.cull.prev_view_proj = self.cull.cur_view_proj;
        }

        cmd_buf.presentDrawable(ProtocolObject::from_ref(&*drawable));

        // Retain this drawable's colour texture so the headless `screenshot`
        // command can blit the last presented frame back to the host. Only
        // under `hot_reload` (the `cn debug` path that runs the WS server able
        // to request a capture, and the only path where the MTKView has
        // `framebufferOnly` switched off so this texture is blit-readable);
        // production keeps this `None`. Reading it next frame is safe: the
        // composite pass that wrote it committed earlier on the same queue, so
        // same-queue FIFO order guarantees it is fully rendered, and a
        // read-only blit may run alongside the compositor's scan-out.
        if self.hot_reload {
            use objc2_quartz_core::CAMetalDrawable;
            self.last_present_texture = Some(drawable.texture());
        }

        // Record this frame's GPU execution time for the profiler overlay.
        // The completion handler fires on a GPU callback thread once the
        // command buffer retires, so the result is read back a frame or two
        // later via the shared atomic. GPUStartTime / GPUEndTime are only
        // valid inside the handler.
        //
        // If per-pass timing is active, the same completion handler also
        // resolves the frame's `MTLCounterSampleBuffer` slot and publishes
        // each pass's microseconds into `pass_times_us`. The handler holds
        // a `Retained` clone of the sample buffer, so the buffer outlives
        // the borrow `self.pass_timing` came from.
        {
            // Hand this frame's in-flight slot to the GPU completion handler;
            // `into_gpu_release` suppresses the guard's Drop so the slot is
            // released exactly once, when the GPU retires the command buffer.
            let frame_sem = frame_slot.into_gpu_release();
            let gpu_time = std::sync::Arc::clone(&self.gpu_time_us);
            let pass_times = std::sync::Arc::clone(&self.pass_times_us);
            let render_fault_logged = std::sync::Arc::clone(&self.render_fault_logged);
            let pass_buffer = self
                .pass_timing
                .as_ref()
                .map(|p| p.buffer_for(pass_timing_slot));
            // Which passes actually ran this frame. The sample buffer is reused
            // across frames and never cleared, so a pass absent this frame (e.g.
            // every world pass behind an opaque menu) would otherwise resolve to
            // its last run's stale timestamps; the handler zeroes those slots.
            let active_mask = self
                .pass_timing
                .as_ref()
                .map(|p| p.attached_mask())
                .unwrap_or(0);
            let handler = block2::RcBlock::new(
                move |cb: std::ptr::NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
                    let cb = unsafe { cb.as_ref() };
                    // A faulted frame render buffer is the usual origin of a
                    // `SubmissionsIgnored` cascade seen later on the RT build.
                    // Log its own error once so the real first fault is visible.
                    use objc2_metal::MTLCommandBufferStatus;
                    if cb.status() == MTLCommandBufferStatus::Error
                        && !render_fault_logged.swap(true, std::sync::atomic::Ordering::Relaxed)
                    {
                        tracing::error!("frame render command buffer faulted: {:?}", cb.error());
                    }
                    // Whole-frame GPU time. This handler's command buffer is
                    // only one slice of a multi-buffer frame, so its own
                    // GPUStartTime/GPUEndTime span under-reports the frame.
                    // Prefer the counter-sample span (earliest pass start to
                    // latest pass end); fall back to this buffer's span when
                    // per-pass timing is unavailable.
                    let span = cb.GPUEndTime() - cb.GPUStartTime();
                    let mut frame_us = (span * 1.0e6).clamp(0.0, f64::from(u32::MAX)) as u32;
                    if let Some(buf) = &pass_buffer {
                        let per_pass = super::pass_timing::resolve(buf);
                        for (i, (slot, micros)) in
                            pass_times.iter().zip(per_pass.iter()).enumerate()
                        {
                            // Report a pass's time only if it ran this frame;
                            // otherwise its sample-buffer slot holds stale data.
                            let value = if active_mask & (1u64 << i) != 0 {
                                *micros
                            } else {
                                0
                            };
                            slot.store(value, std::sync::atomic::Ordering::Relaxed);
                        }
                        if let Some(span_us) = super::pass_timing::frame_span_us(buf) {
                            frame_us = span_us;
                        }
                    }
                    gpu_time.store(frame_us, std::sync::atomic::Ordering::Relaxed);
                    // Release this frame's in-flight slot now the GPU is done
                    // with the command buffer, freeing the CPU to queue the
                    // next frame. Fires on success and on GPU fault alike, so
                    // the semaphore can never leak a slot.
                    frame_sem.signal();
                },
            );
            // SAFETY: addCompletedHandler copies the block (Block_copy), so
            // the RcBlock is free to drop when this scope ends.
            unsafe {
                cmd_buf.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
            }
        }

        cmd_buf.commit();

        // Advance temporal state for the next frame whenever the velocity
        // pre-pass runs: that's TAA *or* the MetalFX upscaler. The
        // un-jittered VP becomes `prev_vp` so the velocity shader can
        // diff against it; this frame's per-object transforms are
        // snapshotted so per-object motion vectors stay correct after a
        // prop update. TAA-specific bookkeeping (history-target ping-pong)
        // only runs when TAA itself is on.
        if velocity_active {
            self.prev_view_proj = mat4_mul(proj, self.view_matrix);
            self.taa.frame = self.taa.frame.wrapping_add(1);
            if self.taa.enabled {
                self.taa.dst = 1 - self.taa.dst;
                self.taa.history_valid = true;
            }
            for (prev, obj) in self
                .prev_draw_models
                .iter_mut()
                .zip(self.draw_objects.iter())
            {
                *prev = obj.model;
            }
            self.skinned
                .prev_joint_matrices
                .clone_from(&self.skinned.joint_matrices);
        }

        self.visible_scratch = visible;
        Ok(())
    }

    // Update the RT acceleration structure to this frame's transforms. The
    // per-frame skinned path (`update_rt_skinned` -> `rebuild_skinned`) keeps the
    // persistent static/cluster BLAS and rebuilds only the skinned BLAS + TLAS +
    // geometry table from the current pose; the non-skinned `rebuild_tlas` path
    // and the one-time seed rebuild the TLAS (or the whole BVH). All paths
    // allocate fresh and retire the outgoing structures through a deferred-free
    // pool keyed on `frame_id`, so a prior in-flight frame keeps reading the old
    // structures. The skinned skin-compute + BLAS/TLAS build are committed without
    // waiting and ordered against the trace by same-queue commit order (both cmd
    // bufs are committed here, before the trace cmd buf in `execute_graph`, on the
    // shared queue). A no-op when RT is off or the scene is static (`Off`).
    //
    // `Auto` (the default) rebuilds the TLAS only when a participating
    // transform actually changed; `Rebuild` / `Tlas` force their work every
    // frame and exist only as diagnostics.
    // Keep the RT acceleration structure current with this frame's transforms
    // and skinned pose. Non-fatal: a per-frame rebuild can fail transiently
    // (e.g. a momentary acceleration-structure allocation hiccup under the
    // per-frame skinned rebuild), and a reflection-BVH update failure must
    // never stop the whole renderer. On failure the previous frame's BVH is
    // kept (the reflection is at most one frame stale, imperceptible) and the
    // failure is logged once per streak (and once on recovery), not at frame
    // rate. The actual work is in `rt_dynamic_update_inner`.
    fn rt_dynamic_update(&mut self, frame_id: u64) {
        match self.rt_dynamic_update_inner(frame_id) {
            Ok(()) => {
                if self.rt.update_failed {
                    tracing::info!("ray-traced reflections: BVH update recovered");
                    self.rt.update_failed = false;
                }
            }
            Err(e) => {
                if !self.rt.update_failed {
                    tracing::warn!(
                        "ray-traced reflections: keeping last frame's BVH, update failed: {e}"
                    );
                    self.rt.update_failed = true;
                }
            }
        }
    }

    fn rt_dynamic_update_inner(&mut self, frame_id: u64) -> Result<(), String> {
        use super::raytrace::RtDynamicMode;
        if !self.rt.dynamic_mode.is_dynamic() {
            return Ok(());
        }
        // RT reflections are not enabled this run (no settings, or the GPU lacks
        // ray tracing): there is no BVH to keep current, and a lingering topology
        // flag must not trigger a build. Clear it and bail.
        if self.rt.settings.is_none() {
            self.rt.topology_dirty = false;
            return Ok(());
        }
        let albedo_count = self.textures.len();
        let normal_count = self.normal_map_textures.len();

        // Free resources parked by prior skinned rebuilds that the frames-in-
        // flight fence now guarantees no in-flight frame can still read.
        let depth = self.frames_in_flight;
        if let Some(accel) = self.rt.accel.as_mut() {
            accel.retire_completed(frame_id, depth);
        }

        // Did a streamed chunk, cloned prop, or participation-changing material
        // edit alter the RT-relevant draw set since the last update? Consume the
        // flag; the BLAS topology must be refreshed below rather than ignored (the
        // `Auto` dirty check only watches the transforms of the prior set).
        let topology_changed = std::mem::take(&mut self.rt.topology_dirty);

        // Skinned meshes deform every frame, so their BLAS (baked from the posed
        // vertices) must be rebuilt each frame: a TLAS-only rebuild can't
        // re-skin. But the static + cluster BLAS never change under a rigid
        // transform, so only the skinned tail (+ TLAS + geometry table) needs
        // rebuilding. `rebuild_skinned` does exactly that, keeping the persistent
        // static BLAS; a full `rebuild_rt_accel` is used only to seed the BVH the
        // first frame after `upload_skinned` (the init build is static-only) or
        // when the `Rebuild` diagnostic forces a from-scratch build every frame.
        let has_skinned = !rt_no_skin()
            && !self.skinned.draw_objects.is_empty()
            && self.rt.skin_pipeline.is_some();
        if has_skinned {
            if self.rt.accel.is_none() || self.rt.dynamic_mode == RtDynamicMode::Rebuild {
                return self.rebuild_rt_accel(albedo_count, normal_count);
            }
            // Fold any added/removed draw geometry into the static head (BLAS only,
            // async), then the skinned path rebuilds the TLAS + table over the
            // refreshed head + the fresh skinned tail.
            if topology_changed {
                self.refresh_rt_topology(albedo_count, normal_count, false, frame_id)?;
            }
            return self.update_rt_skinned(albedo_count, normal_count, frame_id);
        }
        // No skinned geometry.
        if self.rt.accel.is_none() {
            // A topology change can introduce the first participating geometry
            // (e.g. the first streamed chunk in a world that began empty): seed
            // the BVH from scratch. Otherwise nothing to keep current.
            if topology_changed {
                return self.rebuild_rt_accel(albedo_count, normal_count);
            }
            return Ok(());
        }
        // The `Rebuild` diagnostic rebuilds every BLAS every frame, which already
        // absorbs any topology change.
        if self.rt.dynamic_mode == RtDynamicMode::Rebuild {
            return self.rebuild_rt_accel(albedo_count, normal_count);
        }
        if topology_changed {
            // Incrementally refresh the draw-object BLAS head AND rebuild the TLAS
            // over the refreshed set, all async on one command buffer (the
            // transform dirty check only sees the prior set, so the rebuild is
            // forced). `build_tlas = true` does the TLAS inline -- no separate
            // `rebuild_rt_tlas` follow-up.
            self.refresh_rt_topology(albedo_count, normal_count, true, frame_id)?;
            if self.rt.accel.as_ref().is_some_and(|a| a.is_empty()) {
                // The refresh removed the last draw + cluster geometry; drop the
                // BVH so a later add re-seeds it instead of building a degenerate
                // zero-instance TLAS.
                self.rt.accel = None;
            }
            return Ok(());
        }
        match self.rt.dynamic_mode {
            RtDynamicMode::Auto => {
                // Cheap shared-borrow dirty check; rebuild only if something moved.
                let dirty = self
                    .rt
                    .accel
                    .as_ref()
                    .expect("rt_accel is Some (checked above)")
                    .transforms_dirty(&self.draw_objects);
                if dirty {
                    self.rebuild_rt_tlas(albedo_count, normal_count)?;
                }
            }
            RtDynamicMode::Tlas => self.rebuild_rt_tlas(albedo_count, normal_count)?,
            // Handled above / filtered out by the `is_dynamic` guard.
            RtDynamicMode::Rebuild | RtDynamicMode::Off => {}
        }
        Ok(())
    }

    // Incrementally refresh the RT draw-object BLAS head to match the current
    // draw set (added/removed chunks, cloned props, participation-changing
    // material edits), reusing every unchanged BLAS, async. `build_tlas` also
    // rebuilds the TLAS + geometry table inline (the no-skinned path); when clear,
    // the caller's `rebuild_skinned` rebuilds the TLAS over the refreshed head +
    // skinned tail. Borrows the accel mutably while reading the device / queue /
    // shared buffers / draw list, so the cheap handles are cloned and the draw
    // list is lifted out (an O(1) `Vec` swap) to keep the borrows disjoint, then
    // restored.
    fn refresh_rt_topology(
        &mut self,
        albedo_count: usize,
        normal_count: usize,
        build_tlas: bool,
        frame_id: u64,
    ) -> Result<(), String> {
        let device = self.device.clone();
        let queue = self.command_queue.clone();
        let vbuf = self.vertex_buffer.clone();
        let ibuf = self.index_buffer.clone();
        let exclude_seethrough = self.seethrough_meshes_enabled();
        let draw_objects = std::mem::take(&mut self.draw_objects);
        let res = self
            .rt
            .accel
            .as_mut()
            .expect("rt_accel is Some (checked by caller)")
            .refresh_static_topology(
                &device,
                &queue,
                &vbuf,
                &ibuf,
                &draw_objects,
                albedo_count,
                normal_count,
                exclude_seethrough,
                build_tlas,
                frame_id,
            );
        self.draw_objects = draw_objects;
        res
    }

    // Full BVH rebuild (fresh BLAS + TLAS + table) from the current draw list,
    // instanced clusters, and skinned pose. The proven hazard-free path (fresh
    // allocations): used by the `Rebuild` diagnostic mode and, every frame, by
    // any scene with skinned geometry (its deformed vertices change per frame).
    // Replaces `rt_accel` only on a successful non-empty build, so a transient
    // failure or an emptied scene leaves the previous BVH in place. The
    // immutable borrows of `self` all end when the build returns, before the
    // assignment, so there is no aliasing.
    pub(in crate::metal) fn rebuild_rt_accel(
        &mut self,
        albedo_count: usize,
        normal_count: usize,
    ) -> Result<(), String> {
        use super::raytrace::SkinnedRtInputs;
        let skinned = match (
            &self.skinned.vertex_buffer,
            &self.skinned.index_buffer,
            &self.rt.skin_pipeline,
        ) {
            (Some(svb), Some(sib), Some(pipe))
                if !self.skinned.draw_objects.is_empty() && !rt_no_skin() =>
            {
                Some(SkinnedRtInputs {
                    objects: &self.skinned.draw_objects,
                    vertex_buffer: svb,
                    index_buffer: sib,
                    joint_matrices: &self.skinned.joint_matrices,
                    skin_pipeline: pipe.as_ref(),
                })
            }
            _ => None,
        };
        let built = super::raytrace::build_rt_accel(
            &self.device,
            &self.command_queue,
            &self.vertex_buffer,
            &self.index_buffer,
            &self.draw_objects,
            &self.instanced_clusters,
            albedo_count,
            normal_count,
            skinned,
            self.seethrough_meshes_enabled(),
        )?;
        if let Some(accel) = built {
            self.rt.accel = Some(accel);
        }
        Ok(())
    }

    // Per-frame skinned RT update: rebuild only the skinned BLAS + TLAS +
    // geometry table (keeping the persistent static/cluster BLAS) from the
    // current pose and transforms. The accel is borrowed mutably while the
    // skinned inputs are borrowed immutably, so the cheap handles are cloned and
    // the draw list is lifted out (an O(1) `Vec` swap) to keep the borrows
    // disjoint, then restored. A no-op (keeps last frame's BVH) if the required
    // skinned resources are missing.
    fn update_rt_skinned(
        &mut self,
        albedo_count: usize,
        normal_count: usize,
        frame_id: u64,
    ) -> Result<(), String> {
        use super::raytrace::SkinnedRtInputs;
        let device = self.device.clone();
        let queue = self.command_queue.clone();
        let (Some(svb), Some(sib), Some(pipe)) = (
            self.skinned.vertex_buffer.clone(),
            self.skinned.index_buffer.clone(),
            self.rt.skin_pipeline.clone(),
        ) else {
            return Ok(());
        };
        let draw_objects = std::mem::take(&mut self.draw_objects);
        let skinned = SkinnedRtInputs {
            objects: &self.skinned.draw_objects,
            vertex_buffer: &svb,
            index_buffer: &sib,
            joint_matrices: &self.skinned.joint_matrices,
            skin_pipeline: pipe.as_ref(),
        };
        let res = self
            .rt
            .accel
            .as_mut()
            .expect("rt_accel is Some (checked by caller)")
            .rebuild_skinned(
                &device,
                &queue,
                &draw_objects,
                skinned,
                albedo_count,
                normal_count,
                frame_id,
            );
        self.draw_objects = draw_objects;
        res
    }

    // Rebuild just the TLAS + geometry table (fresh allocations, static BLAS)
    // from the current draw-object transforms. `rebuild_tlas` borrows the accel
    // mutably while reading the device / queue / draw list, so clone the two
    // cheap handles and lift the draw list out (an O(1) `Vec` swap) to keep the
    // borrows from aliasing, then put the draw list back.
    fn rebuild_rt_tlas(&mut self, albedo_count: usize, normal_count: usize) -> Result<(), String> {
        let device = self.device.clone();
        let queue = self.command_queue.clone();
        let draw_objects = std::mem::take(&mut self.draw_objects);
        let res = self
            .rt
            .accel
            .as_mut()
            .expect("rt_accel is Some (checked by caller)")
            .rebuild_tlas(&device, &queue, &draw_objects, albedo_count, normal_count);
        self.draw_objects = draw_objects;
        res
    }

    // Rebuild the off-screen render targets whose footprint follows the
    // drawable size: HDR colour + depth + resolve, bloom chain, TAA history +
    // velocity (when TAA is on), SSAO targets (when SSAO is on), SSR targets
    // (when SSR is on). Called at the top of every frame; only the targets
    // that actually changed dimensions are recreated.
    //
    // When MetalFX upscaling is on, "render resolution" (where the 3D scene
    // draws) is `output * upscale_scale`, smaller than the drawable. Bloom
    // and the MetalFX output texture stay at the drawable (output)
    // resolution so the final composite reads cleanly into the swapchain.
    fn resize_targets_if_needed(&mut self, want_w: u32, want_h: u32) -> Result<(), String> {
        // Output (drawable) dimensions are the `want_w/h` arg; render
        // dimensions match output when no upscaler is active, otherwise
        // they're the upscaler's input size (clamped to device-supported
        // scale range at scaler-build time).
        let (render_w, render_h) = if self.upscale.scaler.is_some() {
            (
                ((want_w as f32) * self.upscale.scale).max(1.0) as u32,
                ((want_h as f32) * self.upscale.scale).max(1.0) as u32,
            )
        } else {
            (want_w, want_h)
        };

        let render_changed =
            render_w != self.hdr_targets.width || render_h != self.hdr_targets.height;
        if render_changed {
            self.hdr_targets = super::texture::create_hdr_targets(
                &self.device,
                render_w,
                render_h,
                super::context::HDR_SAMPLE_COUNT,
            )?;
        }
        // The planar reflection targets are render-resolution (they re-render the
        // scene from the mirrored camera at the same resolution the reflectors
        // sample). The plane set carries over; only the targets are reallocated.
        if render_changed && let Some(set) = self.planar_reflection.as_ref() {
            let planes = set.planes.clone();
            self.planar_reflection = Some(super::planar::create_planar_set(
                &self.device,
                render_w,
                render_h,
                super::context::HDR_SAMPLE_COUNT,
                &planes,
            )?);
        }
        // The bloom chain reads `scene_color`: at drawable size when the
        // upscaler runs, otherwise at native (= render) resolution. Sized
        // off `want_w/h` either way.
        if want_w != self.bloom_targets.width || want_h != self.bloom_targets.height {
            self.bloom_targets = super::post::create_bloom_targets(&self.device, want_w, want_h)?;
        }
        // The TAA history + velocity buffers are render-resolution. Stale
        // history can't be reprojected into the new resolution, so mark
        // it invalid: the next frame passes straight through and
        // accumulation restarts.
        if render_changed && self.taa.enabled {
            self.taa.targets =
                super::post::create_taa_targets(&self.device, render_w, render_h)?.to_vec();
            self.taa.history_valid = false;
        }
        // The SSAO occlusion targets are render-resolution. Its depth + normal
        // input now comes from the unified G-buffer pre-pass (below), so SSAO
        // owns no G-buffer of its own.
        if render_changed && self.ssao.settings.is_some() {
            self.ssao.targets = Some(super::post::create_ssao_targets(
                &self.device,
                render_w,
                render_h,
            )?);
            // `ao_output` (the blurred occlusion) lives in the transient pool;
            // rebuild it at the new render resolution too.
            self.transient_pool.rebuild(
                &self.device,
                &super::transient_pool::transient_specs(true, render_w, render_h),
            )?;
        }
        // The SSR resolve-output target is render-resolution. Rebuilt when SSR,
        // SSGI, *or* RT reflections are on (RT reuses `ssr_targets.output`). The
        // acceleration structure is resolution-independent, so it is not
        // rebuilt here.
        if render_changed
            && (self.ssr.settings.is_some()
                || self.ssgi.settings.is_some()
                || self.rt.settings.is_some())
        {
            self.ssr.targets = Some(super::post::create_ssr_targets(
                &self.device,
                render_w,
                render_h,
                self.ssr.blur_scale,
            )?);
        }
        // The unified G-buffer targets (normal+depth / roughness / velocity /
        // sampleable depth) are render-resolution. Rebuilt when any consumer
        // (SSR / SSGI / RT / SSAO / TAA / upscaler) is on: the same gate as the
        // GBufferPrepass node.
        if render_changed
            && (self.ssr.settings.is_some()
                || self.ssgi.settings.is_some()
                || self.rt.settings.is_some()
                || self.ssao.settings.is_some()
                || self.taa.enabled
                || self.upscale.scaler.is_some())
        {
            self.gbuffer.targets = Some(super::post::create_gbuffer_targets(
                &self.device,
                render_w,
                render_h,
            )?);
        }
        // The SSGI gather target is render-resolution scaled by `gi_scale`
        // (the composite bilateral-upsamples it back to full resolution).
        if render_changed && let Some(s) = self.ssgi.settings {
            let (gw, gh) = s.gi_dimensions(render_w, render_h);
            self.ssgi.targets = Some(super::post::create_ssgi_targets(&self.device, gw, gh)?);
        }
        // The Hi-Z pyramid matches the render (depth) resolution. Rebuild it
        // and mark it invalid so the next cull dispatch ignores the now-stale
        // pyramid (the projection coordinates were generated at the old
        // resolution); the next frame's build refills it.
        if render_changed && let Some(hiz) = self.cull.hiz.as_mut() {
            hiz.resize_to(&self.device, render_w, render_h)?;
            self.cull.hiz_valid = false;
        }
        // MetalFX scaler is bound to specific (input, output) sizes at
        // construction; a resize requires a fresh scaler instance. Reset
        // the temporal history on the first frame after the rebuild so
        // the new scaler doesn't pull from a stale buffer.
        if self.upscale.scaler.is_some() {
            let output_changed = match &self.upscale.scaler {
                Some(u) => want_w != u.output_width || want_h != u.output_height,
                None => false,
            };
            if output_changed {
                let new_scaler = super::post::MetalFXUpscaler::new(
                    &self.device,
                    want_w,
                    want_h,
                    self.upscale.scale,
                )?;
                self.upscale.scaler = Some(new_scaler);
                self.upscale
                    .reset_pending
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        Ok(())
    }
}
