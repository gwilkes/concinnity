// src/metal/graph_exec.rs
//
// Metal-side executor for the render graph. `MtlContext::execute_graph`
// walks a `CompiledGraph` and dispatches each pass by `PassId` to the
// existing `encode_*` method. Every Metal pass that ever ran inline is
// now in the graph. Composite plus Shadow, Main, Cull, AutoExposure,
// Bloom, Velocity, TaaResolve, SsrResolve, ParticlesDraw, Fog, Decals,
// SsrPrepass, and SsaoBlur are the dispatchable PassIds. PassIds
// `ParticlesSim`, `SsaoPrepass`, and `SsaoKernel` are timing-only:
// their per-pass timing slots fire from `pass_timing.attach_*` calls
// inside the bundled `encode_particles` / `encode_ssao` Rust functions,
// but they must never appear as graph nodes (the executor rejects them
// with a clear error if mis-added). `ParticlesDraw` dispatches the
// bundled `encode_particles` (which internally encodes both
// ParticlesSim compute + ParticlesDraw render), so PassId::ParticlesSim
// has no separate graph node but keeps its per-pass timing slot.
//
// Per-pass command buffers. Each non-composite pass now
// runs on its own freshly-minted `MTLCommandBuffer`, committed
// immediately. The `Composite` pass keeps using the outer cmd_buf that
// `draw_frame` owns (so `presentDrawable` + the completion handler
// stay attached to the cmd buf that actually writes to the drawable).
// On a single command queue, commit order = GPU execution order, so
// the topologically-sorted `graph.passes` iteration order is also the
// GPU order: no `MTLEvent` wait/signal pairs are needed. It also
// sidesteps the `MTLParallelRenderCommandEncoder`
// abort that reliably trips G14X (M2/M3 Pro/Max-class GPUs) on
// macOS 26.4 after ~20-90 s of rendering, regardless of how few
// sub-encoders we minted.
//
// The executor is a `&mut self` method on `MtlContext` taking the
// concrete per-frame params.
//
// Per-pass barriers (`pass.barriers_before`) are not applied on Metal: the
// DX/VK seam (`barrier_translate` + a per-resource registry +
// `emit_graph_barriers`) emits explicit resource-state TRANSITIONS, and Metal
// has none to emit.
//
//   1. Hazards are tracked automatically. Every resource here uses the default
//      tracked hazard mode (no `MTLHeap` / untracked resources), so Metal
//      inserts the cross-encoder and cross-command-buffer read/write
//      dependencies itself. There is no `ResourceBarrier` / pipeline-barrier
//      analogue to translate a `(class, ResourceState)` into.
//   2. Cross-pass ordering is free. Each pass commits its own command buffer in
//      topological order on one queue (commit order = GPU execution order, see
//      below), so the producer -> consumer ordering `barriers_before` encodes is
//      already guaranteed by submission order.
//   3. The only Metal "barrier-analogue" is `useResource` residency, and it is
//      a DIFFERENT concern the graph cannot drive: it is per-encoder (every
//      encoder reaching a resource INDIRECTLY -- through an ICB, an argument
//      buffer, or an acceleration structure -- must declare it), not a one-shot
//      producer -> consumer transition, and it covers resources the graph does
//      not model (the bindless texture pool, the env maps, the accel BLAS). So
//      it stays inline and comprehensive at each indirect-access encoder: the
//      ICB write residency + the bindless main pass `use_bindless_textures`
//      (cull.rs), and the RT trace (rt_reflections.rs / raytrace.rs).
//
// The Vulkan / DirectX executors consume the same `BarrierOp` list; Metal reads
// the graph for ordering + resource lifetimes only. (If untracked / heap
// resources are ever introduced, the point-1 assumption breaks and explicit
// `MTLFence`s become necessary.)

#![allow(clippy::incompatible_msrv)]

use std::sync::atomic::Ordering;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue as _, MTLTexture};

use crate::gfx::frustum::Frustum;
use crate::gfx::render_graph::{CompiledGraph, PassId};
use crate::gfx::render_types::{
    FogFroxelParams, FogParams, RtParams, SsaoParams, SsgiParams, SsrParams, TextDrawCall,
};

use super::context::MtlContext;
use super::parallel_encoder::{ParallelCtxRef, SendableCmdBuf};
use super::uniforms::{GBufferView, TaaUniforms, VelocityUniforms};

// Per-frame params the executor threads into each pass's `encode_*`
// method. The set is the union of what every currently-migrated pass
// needs; fields that a given pass does not consume are simply ignored.
// `scene_color` is `Option` because the pre-graph runs before TAA /
// SSR resolve has produced it; the post-graph (which contains
// Composite) supplies it.
//
// `Send + Sync` is asserted so the struct can be shared by reference into
// the rayon::scope fan-out inside `execute_graph`. Workers only read from
// it; the only non-Sync field is `cmd_buf`, which workers do not use
// (each worker mints its own `MTLCommandBuffer`). `cmd_buf` is consumed
// solely on the main thread for the Composite pass and the present /
// completion handler attached by `draw_frame`.
pub(in crate::metal) struct GraphFrameParams<'a> {
    pub cmd_buf: &'a ProtocolObject<dyn MTLCommandBuffer>,
    pub cam_pos: [f32; 3],
    pub skinned_joint_bufs: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
    pub scene_color: Option<&'a Retained<ProtocolObject<dyn MTLTexture>>>,
    pub text_calls: &'a [TextDrawCall],
    // An opaque menu backdrop hides the scene: the Main pass runs as a bare
    // clear (it is the only surviving world pass in the masked graph), skipping
    // every geometry sub-path so nothing of the world draws behind the menu.
    pub world_hidden: bool,
    // Main-pass params. Built by draw_frame between the un-migrated
    // pre-main legacy work and the pre-graph dispatch, then handed in
    // here so the executor can call encode_main_pass with the same
    // shape it had inline.
    pub elapsed: f32,
    pub vp: [[f32; 4]; 4],
    // Inverse of `vp`, computed once in `draw_frame` and shared by every pass
    // that reconstructs world-space position from depth (fog / decals /
    // raymarch / transparent) instead of each re-inverting `vp`.
    pub inv_vp: [[f32; 4]; 4],
    pub visible: &'a [u32],
    pub frustum: &'a Frustum,
    // Instanced clusters culled + LOD-bucketed + uploaded once this frame,
    // shared by the main / SSR / SSAO / velocity passes (see
    // `metal/instanced.rs`). Empty when the scene has no clusters.
    pub prepared_instances: &'a super::instanced::PreparedInstances,
    pub object_buffer: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub bindless_tex_args: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    // This frame's skinned deformed-vertex buffer. `Some` only
    // when the skinned fold is active (bindless + static geometry + a folded
    // SkinnedMesh). The Cull pass's `encode_main_skin` writes it; the Main /
    // Main2 skinned ICB tail binds it as the vertex buffer for that draw range.
    pub deformed_skinned: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    // The previous-frame skinned deformed-vertex buffer: the slot one
    // frame behind `deformed_skinned` in the per-frame deformed ring. The
    // GPU-driven G-buffer skinned tail reads it as the previous vertex stream to
    // emit a per-vertex skin motion vector. `Some` only when the fold is active;
    // the priming gate (`deformed_primed`) handles the unposed first frame.
    pub deformed_prev: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Per-frame parallel `prev_model` buffer for the GPU-driven G-buffer pass:
    // one `float4x4` per cull record, read by the bindless G-buffer
    // VS at `[[base_instance]]`. `Some` only when the GPU-driven pre-pass runs.
    pub prev_model_buffer: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    // GPU-cull output the bindless Main pass consumes via
    // executeCommandsInBuffer. `Some` only when bindless cull ran this
    // frame (i.e. matches `FrameGraphInputs::bindless_cull_enabled`).
    pub draw_args_buffer: Option<&'a Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Per-pixel motion-vector pass uniforms. `Some` only when the
    // `Velocity` pass is in the graph this frame (matches
    // `FrameGraphInputs::velocity_enabled`).
    pub vel_uniforms: Option<&'a VelocityUniforms>,
    // Skinned joint matrices from the previous frame, used only by the
    // Velocity pass to emit motion vectors for animated meshes. Empty
    // when TAA is off.
    pub prev_skinned_joint_bufs: &'a [Retained<ProtocolObject<dyn MTLBuffer>>],
    // TAA-resolve pass uniforms. `Some` only when the `TaaResolve` pass
    // is in the graph this frame (matches `FrameGraphInputs::taa_enabled`).
    pub taa_uniforms: Option<&'a TaaUniforms>,
    // Pre-TAA scene texture that `TaaResolve` reads (the SSR resolve
    // output when SSR is on, otherwise the raw `hdr_resolve`). `Some`
    // only when the `TaaResolve` pass is in the graph this frame.
    pub scene_pre_taa: Option<&'a Retained<ProtocolObject<dyn MTLTexture>>>,
    // SSR ray-march params (96 B, copy-able). `Some` only when the
    // `SsrResolve` pass is in the graph this frame (matches
    // `FrameGraphInputs::ssr_enabled`).
    pub ssr_params: Option<&'a SsrParams>,
    // Volumetric-fog pass uniforms. `Some` only when the `Fog` pass is
    // in the graph this frame (matches `FrameGraphInputs::fog_enabled`).
    pub fog_params: Option<&'a FogParams>,
    // Metal-only froxel-volume extras (view matrix + volume dims +
    // near/far). `Some` only when `Fog` is in the graph this frame; the
    // FogFroxel compute pass + the Fog fragment shader sample path both
    // consume it.
    pub fog_froxel_params: Option<&'a FogFroxelParams>,
    // SSAO kernel + blur params. `Some` only when the `SsaoBlur` pass
    // is in the graph this frame (matches `FrameGraphInputs::ssao_enabled`).
    pub ssao_params: Option<&'a SsaoParams>,
    // SSGI gather + composite params. `Some` only when the `Ssgi` pass is
    // in the graph this frame (matches `FrameGraphInputs::ssgi_enabled`).
    pub ssgi_params: Option<&'a SsgiParams>,
    // RT-reflection params (camera + sun + tunables). `Some` only when the
    // `RtReflections` pass is in the graph this frame (matches
    // `FrameGraphInputs::rt_reflections_enabled`).
    pub rt_reflection_params: Option<&'a RtParams>,
}

// SAFETY: see the type-level docs. The non-Sync `cmd_buf` field is used
// exclusively on the main thread (Composite + present + completion
// handler); every worker spawned by `execute_graph` mints its own cmd buf.
// All other fields are Sync (POD or thread-safe Apple Metal handles).
unsafe impl<'a> Send for GraphFrameParams<'a> {}
unsafe impl<'a> Sync for GraphFrameParams<'a> {}

impl MtlContext {
    // Walk a compiled render graph and dispatch each pass to its
    // existing per-backend encoder. `params` carries the per-frame state
    // that cannot live on `&mut self` (the command buffer is per-frame;
    // the scene-colour texture is computed each frame after SSR / TAA
    // resolve).
    //
    // Any pass not matched by the match arm below returns an error so
    // a caller that drops an unhandled PassId into the graph by
    // mistake fails loudly rather than silently no-op'ing.
    pub(in crate::metal) fn execute_graph(
        &mut self,
        graph: &CompiledGraph,
        params: &GraphFrameParams<'_>,
    ) -> Result<(), String> {
        // Per-frame particle-state mutations live on `&mut self` and have
        // to happen before the read-only `encode_particles` path runs. We
        // build the (dt, frame_index, per-emitter spawn budgets) tuple
        // once here and stash it for the match arm below.
        let particle_frame = self.prepare_particle_pass(params.elapsed);
        self.draw_calls_accum.store(0, Ordering::Relaxed);

        let composite_idx = graph
            .passes
            .iter()
            .position(|p| matches!(p.id, PassId::Composite));

        // Pre-allocate per-pass slots. Workers encode into their own
        // freshly-minted `MTLCommandBuffer` in parallel and hand the
        // encoded-but-uncommitted buffer back through the matching slot.
        // The main thread then commits each slot in topological pass
        // order, so the single command queue's commit order = the
        // graph's pass order = the GPU's execution order. This is
        // important on Apple Silicon: an earlier draft of this code
        // had workers commit their own cmd bufs in arbitrary
        // thread-schedule order with an `MTLEvent` wait/signal chain
        // enforcing GPU ordering. That left the renderer drawing into
        // a black drawable: Apple's command queue executes cmd bufs
        // FIFO in commit order regardless of events, so committing out
        // of order broke the dependency chain (later passes ran while
        // earlier passes' writes were still queued behind them).
        let worker_slots: std::sync::Mutex<Vec<Option<SendableCmdBuf>>> =
            std::sync::Mutex::new((0..graph.passes.len()).map(|_| None).collect());
        let first_error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

        // Cloned before the parallel borrow so the commit loop's per-pass
        // fault-logging handlers can share the throttle without re-borrowing
        // `self` while `ctx_ref` is live.
        let pass_fault_count = std::sync::Arc::clone(&self.pass_fault_count);
        let ctx_ref = ParallelCtxRef::new(self);
        crate::jobs::pool().install(|| {
            rayon::scope(|scope| {
                for (idx, pass) in graph.passes.iter().enumerate() {
                    if Some(idx) == composite_idx {
                        continue;
                    }
                    // No Metal-side barrier work: Apple's implicit
                    // hazard tracking covers it. Touched here so the
                    // unused field doesn't lint when this loop is the
                    // only consumer.
                    let _ = &pass.barriers_before;
                    let pass_id = pass.id;
                    let particle_ref = particle_frame.as_ref();
                    let first_error_ref = &first_error;
                    let worker_slots_ref = &worker_slots;
                    scope.spawn(move |_| {
                        let ctx = ctx_ref.as_ctx();
                        let cmd_buf = match ctx.command_queue.commandBuffer() {
                            Some(cb) => cb,
                            None => {
                                let mut e = first_error_ref.lock().unwrap();
                                if e.is_none() {
                                    *e = Some(
                                        "graph executor: failed to mint per-pass cmd buf".into(),
                                    );
                                }
                                return;
                            }
                        };
                        match ctx.encode_pass_into(pass_id, &cmd_buf, params, particle_ref) {
                            Ok(count) => {
                                ctx.draw_calls_accum.fetch_add(count, Ordering::Relaxed);
                                let mut lock = worker_slots_ref.lock().unwrap();
                                lock[idx] = Some(SendableCmdBuf(cmd_buf));
                            }
                            Err(e) => {
                                let mut lock = first_error_ref.lock().unwrap();
                                if lock.is_none() {
                                    *lock = Some(e);
                                }
                            }
                        }
                    });
                }
            });
        });

        if let Some(err) = first_error.into_inner().unwrap_or(None) {
            return Err(err);
        }

        // Commit every worker-encoded cmd buf in topological pass
        // order. Single command queue, FIFO execution: the order
        // these commit in is the order the GPU runs them in.
        let slots = worker_slots
            .into_inner()
            .map_err(|_| "graph executor: worker slot mutex poisoned".to_string())?;
        for (idx, slot) in slots.into_iter().enumerate() {
            if let Some(cb) = slot {
                // Diagnostic: each pass commits its own command buffer, so a
                // GPU fault confined to one pass (e.g. the RT reflection trace)
                // surfaces only here: the outer composite buffer just sees a
                // downstream victim. Attach a handler naming the faulting pass
                // so the *original* fault in a cascade is identifiable.
                let pass_id = graph.passes.get(idx).map(|p| p.id);
                let throttle = std::sync::Arc::clone(&pass_fault_count);
                let handler = block2::RcBlock::new(
                    move |cbh: std::ptr::NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                        let cbh = unsafe { cbh.as_ref() };
                        if cbh.status() == objc2_metal::MTLCommandBufferStatus::Error
                            && throttle.fetch_add(1, Ordering::Relaxed) < 8
                        {
                            tracing::error!(
                                "render-graph pass {:?} command buffer faulted: {:?}",
                                pass_id,
                                cbh.error()
                            );
                        }
                    },
                );
                // SAFETY: addCompletedHandler copies the block, so the RcBlock
                // may drop at end of iteration.
                unsafe {
                    cb.0.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
                }
                cb.0.commit();
            }
        }

        // Composite stays on the outer cmd buf that `draw_frame` owns,
        // so `presentDrawable` + the completion handler attach to the
        // same cmd buf that writes to the drawable. It's committed by
        // `draw_frame` after this returns; since every non-composite
        // cmd buf above has already committed, the queue order places
        // the outer cmd buf strictly after them: composite reads
        // every prior write correctly without any explicit MTLEvent
        // wait.
        if composite_idx.is_some() {
            let count = self.encode_pass_into(
                PassId::Composite,
                params.cmd_buf,
                params,
                particle_frame.as_ref(),
            )?;
            self.draw_calls_accum.fetch_add(count, Ordering::Relaxed);
        }

        self.frame_stats.draw_calls += self.draw_calls_accum.load(Ordering::Relaxed);
        Ok(())
    }

    // Dispatch a single pass into a freshly-minted `MTLCommandBuffer`. Takes
    // Build the per-frame `RaymarchView` from the graph params. Shared by the
    // `Raymarch` pass (live SDF surface) and the `Shadow` pass (SDF shadow
    // casters) so both agree on the camera VP / time / viewport: the shadow
    // fragment only reads `time`, but constructing the full view keeps the
    // binding identical to the main pass.
    fn build_raymarch_view(&self, params: &GraphFrameParams<'_>) -> super::raymarch::RaymarchView {
        super::raymarch::RaymarchView {
            vp: params.vp,
            inv_vp: params.inv_vp,
            cam_pos: [params.cam_pos[0], params.cam_pos[1], params.cam_pos[2], 0.0],
            viewport: [
                self.hdr_targets.width as f32,
                self.hdr_targets.height as f32,
            ],
            time: params.elapsed,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
        }
    }

    // `&self` so it can be called from both the main-thread Composite path
    // and the rayon-spawned per-pass workers.
    fn encode_pass_into(
        &self,
        pass_id: PassId,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        params: &GraphFrameParams<'_>,
        particle_frame: Option<&(f32, u32, Vec<u32>)>,
    ) -> Result<u32, String> {
        Ok(match pass_id {
            PassId::Cull => {
                let object_buffer = params.object_buffer.ok_or(
                    "graph executor: Cull pass requires object_buffer but none was supplied",
                )?;
                let draw_args_buffer = params.draw_args_buffer.ok_or(
                    "graph executor: Cull pass requires draw_args_buffer but none was supplied",
                )?;
                // Skinned fold: pre-skin into this frame's deformed
                // buffer in the Cull command buffer (before the cull dispatch).
                // Committed before Main, so Metal hazard-tracks the deformed
                // write ahead of the main pass's vertex read. Only when the fold
                // is active (deformed buffer supplied).
                if let Some(deformed) = params.deformed_skinned {
                    self.encode_main_skin(cmd_buf, deformed, params.skinned_joint_bufs)?;
                }
                self.encode_cull(
                    cmd_buf,
                    object_buffer,
                    draw_args_buffer,
                    params.frustum,
                    params.cam_pos,
                )?;
                // GPU-driven cascaded shadow: fill the per-cascade
                // shadow ICB in this same Cull command buffer (committed before
                // the Shadow render pass's command buffer), so the cross-command-
                // buffer FIFO order makes the shadow ICB ready when Shadow reads
                // it -- the same ordering the main cull -> Main ICB relies on. A
                // no-op when the shadow-bindless path is inactive.
                self.encode_shadow_culls(cmd_buf, object_buffer, draw_args_buffer)?;
                0
            }
            PassId::HizBuild => {
                // Two-pass occlusion: rebuild the Hi-Z pyramid mid-frame
                // from this frame's phase-1 depth so Cull2 re-tests against
                // up-to-date occluders. Same `encode_hiz_build` the
                // end-of-frame (next-frame) build uses, just dispatched
                // here as a graph node ordered after Main.
                self.encode_hiz_build(cmd_buf);
                0
            }
            PassId::Cull2 => {
                let object_buffer = params.object_buffer.ok_or(
                    "graph executor: Cull2 pass requires object_buffer but none was supplied",
                )?;
                let draw_args_buffer = params.draw_args_buffer.ok_or(
                    "graph executor: Cull2 pass requires draw_args_buffer but none was supplied",
                )?;
                self.encode_cull_phase2(
                    cmd_buf,
                    object_buffer,
                    draw_args_buffer,
                    params.frustum,
                    params.cam_pos,
                )?
            }
            PassId::Main2 => self.encode_main_pass_phase2(
                cmd_buf,
                params.elapsed,
                params.vp,
                params.cam_pos,
                params.object_buffer,
                params.bindless_tex_args,
                params.deformed_skinned,
            )?,
            PassId::Shadow => {
                // Build the raymarch view only when a volume opts into
                // shadow casting; otherwise pass `None` so the shadow encoder
                // skips the SDF caster sub-pass with zero overhead. The view
                // matches the matching `PassId::Raymarch` build later this
                // frame (same camera VP / time / viewport). Mirrors DirectX.
                let raymarch_view = if self.any_raymarch_shadow_casters() {
                    Some(self.build_raymarch_view(params))
                } else {
                    None
                };
                self.encode_shadow_pass(
                    cmd_buf,
                    params.skinned_joint_bufs,
                    params.cam_pos,
                    params.object_buffer,
                    params.deformed_skinned,
                    raymarch_view.as_ref(),
                )?
            }
            PassId::Main => self.encode_main_pass(
                cmd_buf,
                params.elapsed,
                params.vp,
                params.cam_pos,
                params.visible,
                params.prepared_instances,
                params.skinned_joint_bufs,
                params.object_buffer,
                params.bindless_tex_args,
                params.deformed_skinned,
                params.world_hidden,
            )?,
            PassId::AutoExposure => self.encode_auto_exposure(cmd_buf)?,
            PassId::Bloom => {
                let scene_color = params.scene_color.ok_or(
                    "graph executor: Bloom pass requires scene_color but none was supplied",
                )?;
                self.encode_bloom(cmd_buf, scene_color)?
            }
            PassId::Velocity | PassId::SsrPrepass => {
                // Merged into GBufferPrepass on Metal: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never these.
                return Err(format!(
                    "graph executor (metal): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::GBufferPrepass => {
                // The jittered VP rasterises; the un-jittered cur/prev VPs (from
                // vel_uniforms, when velocity is active) drive the motion vector.
                let gview = match params.vel_uniforms {
                    Some(v) => GBufferView {
                        jittered_vp: v.jittered_vp,
                        cur_vp: v.cur_vp,
                        prev_vp: v.prev_vp,
                        view: self.view_matrix,
                    },
                    // Velocity inactive: cur == prev so the motion channel is a
                    // harmless zero (no consumer reads it).
                    None => GBufferView {
                        jittered_vp: params.vp,
                        cur_vp: params.vp,
                        prev_vp: params.vp,
                        view: self.view_matrix,
                    },
                };
                self.encode_gbuffer_prepass(
                    cmd_buf,
                    &gview,
                    params.visible,
                    params.cam_pos,
                    params.prepared_instances,
                    params.skinned_joint_bufs,
                    params.prev_skinned_joint_bufs,
                    params.vel_uniforms.is_some(),
                    params.object_buffer,
                    params.prev_model_buffer,
                    params.deformed_skinned,
                    params.deformed_prev,
                )?
            }
            PassId::TaaResolve => {
                let taa_uniforms = params.taa_uniforms.ok_or(
                    "graph executor: TaaResolve pass requires taa_uniforms but none was supplied",
                )?;
                let scene_pre_taa = params.scene_pre_taa.ok_or(
                    "graph executor: TaaResolve pass requires scene_pre_taa but none was supplied",
                )?;
                self.encode_taa(cmd_buf, taa_uniforms, scene_pre_taa)?
            }
            PassId::SsrResolve => {
                let ssr_params = params.ssr_params.ok_or(
                    "graph executor: SsrResolve pass requires ssr_params but none was supplied",
                )?;
                self.encode_ssr_resolve(cmd_buf, ssr_params)?
            }
            PassId::Ssgi => {
                let ssgi_params = params.ssgi_params.ok_or(
                    "graph executor: Ssgi pass requires ssgi_params but none was supplied",
                )?;
                self.encode_ssgi(cmd_buf, ssgi_params)?
            }
            PassId::RtReflections => {
                let rt_params = params.rt_reflection_params.ok_or(
                    "graph executor: RtReflections pass requires rt_reflection_params but none was supplied",
                )?;
                self.encode_rt_reflections(cmd_buf, rt_params, params.bindless_tex_args)?
            }
            PassId::SsaoBlur => {
                // PassId::SsaoBlur dispatches the bundled `encode_ssao` (GTAO
                // kernel + depth-aware blur). It reads the unified G-buffer
                // pre-pass output, so SSAO runs no geometry redraw of its own;
                // per-pass timing for the sub-passes is wired inline inside
                // `encode_ssao`.
                let ssao_params = params.ssao_params.ok_or(
                    "graph executor: SsaoBlur pass requires ssao_params but none was supplied",
                )?;
                self.encode_ssao(cmd_buf, ssao_params)?
            }
            PassId::SsaoPrepass | PassId::SsaoKernel => {
                // Bundled inside `encode_ssao` (dispatched via
                // PassId::SsaoBlur). These PassIds keep their
                // per-pass timing slots via inline
                // `pass_timing.attach_render` calls inside
                // encode_ssao, but they must not appear as their
                // own graph nodes: same pattern as
                // PassId::ParticlesSim.
                return Err(format!(
                    "graph executor: pass {} is bundled inside SsaoBlur \
                         (encode_ssao encodes all three SSAO sub-passes); it \
                         should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::ReflectionComposite => {
                // Encoded inline at the tail of SsrResolve / RtReflections (it
                // blurs + composites the reflection target they wrote). Keeps a
                // timing slot via an inline `attach_render`, but is never a graph
                // node of its own -- same pattern as the bundled SSAO sub-passes.
                return Err(format!(
                    "graph executor: pass {} is encoded inline by SsrResolve / \
                         RtReflections; it should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::Decals => {
                self.encode_decals(cmd_buf, params.vp, params.inv_vp, params.frustum)?
            }
            PassId::Fog => {
                let fog_params = params
                    .fog_params
                    .ok_or("graph executor: Fog pass requires fog_params but none was supplied")?;
                let fog_froxel_params = params.fog_froxel_params.ok_or(
                    "graph executor: Fog pass requires fog_froxel_params but none was supplied",
                )?;
                self.encode_fog(cmd_buf, fog_params, fog_froxel_params)?
            }
            PassId::FogFroxel => {
                let fog_params = params.fog_params.ok_or(
                    "graph executor: FogFroxel pass requires fog_params but none was supplied",
                )?;
                let fog_froxel_params = params.fog_froxel_params.ok_or(
                        "graph executor: FogFroxel pass requires fog_froxel_params but none was supplied",
                    )?;
                self.encode_fog_froxel(cmd_buf, fog_params, fog_froxel_params)?
            }
            PassId::ParticlesDraw => {
                // Bundles ParticlesSim (compute) + ParticlesDraw
                // (render); see `encode_particles` for the per-pass
                // timing wiring. Only `PassId::ParticlesDraw` is a
                // graph node: `PassId::ParticlesSim` remains a
                // timing-only PassId. Per-frame particle-state
                // mutations (`particle_last_elapsed`,
                // `particle_frame_index`, per-emitter spawn budget)
                // were run on `&mut self` before this loop via
                // `prepare_particle_pass`; the read-only encode here
                // consumes the precomputed tuple.
                if let Some((dt, frame_index, budgets)) = particle_frame {
                    self.encode_particles(
                        cmd_buf,
                        *dt,
                        *frame_index,
                        budgets,
                        params.vp,
                        params.frustum,
                    )?
                } else {
                    0
                }
            }
            PassId::Composite => {
                let scene_color = params.scene_color.ok_or(
                    "graph executor: Composite pass requires scene_color but none was supplied",
                )?;
                self.encode_composite_and_text(cmd_buf, scene_color, params.text_calls)?
            }
            // ParticlesSim is bundled inside `encode_particles`
            // (dispatched via `PassId::ParticlesDraw`), so it has
            // no separate graph node. It keeps its per-pass timing
            // slot via the inline `pass_timing.attach_compute` call.
            PassId::ParticlesSim => {
                return Err(format!(
                    "graph executor: pass {} is bundled inside ParticlesDraw \
                         (encode_particles encodes both); it should not appear as \
                         its own graph node",
                    pass_id.name()
                ));
            }
            PassId::Upscale => {
                let scene_pre_taa = params.scene_pre_taa.ok_or(
                    "graph executor: Upscale pass requires scene_pre_taa but none was supplied",
                )?;
                self.encode_upscale(cmd_buf, scene_pre_taa)?
            }
            PassId::Transparent => {
                let scene_pre_taa = params.scene_pre_taa.ok_or(
                    "graph executor: Transparent pass requires scene_pre_taa but none was supplied",
                )?;
                let inv_vp = params.inv_vp;
                let view = super::uniforms::TransparentView {
                    vp: params.vp,
                    inv_vp,
                    camera_pos: [params.cam_pos[0], params.cam_pos[1], params.cam_pos[2], 0.0],
                    viewport: [
                        self.hdr_targets.width as f32,
                        self.hdr_targets.height as f32,
                    ],
                    time: params.elapsed,
                    _pad: 0.0,
                };
                // Planar reflection: when RT is off and the world has flat
                // reflectors (water surfaces / glass panes), re-render the scene
                // mirrored across each distinct reflector plane into its planar
                // target (reusing this frame's cull ICB + bindless buffers) so the
                // surface samples a sharp scene reflection instead of the blurry
                // probe cube. RT on uses the per-pixel trace instead, so the planar
                // pass is skipped. Encoded before `encode_transparent` on the same
                // command buffer, which samples the resolves.
                let planar_live = self.rt.accel.is_none()
                    && self
                        .planar_reflection
                        .as_ref()
                        .is_some_and(|s| !s.targets.is_empty());
                if planar_live {
                    self.encode_planar_reflections(cmd_buf, params)?;
                }

                // Gather every translucent producer's draws, then let the
                // shared encoder sort them back-to-front and issue them.
                let mut draws = Vec::new();
                self.collect_water_transparent_draws(
                    &view,
                    params.bindless_tex_args.is_some(),
                    planar_live,
                    &mut draws,
                );
                self.collect_glass_transparent_draws(
                    &view,
                    params.bindless_tex_args.is_some(),
                    planar_live,
                    &mut draws,
                );
                // Transparent glass MESHES (Layer 2): imported `transparent`
                // materials traced per-pixel when RT is live. Inert otherwise
                // (those meshes render opaque in the main pass).
                self.collect_mesh_transparent_draws(
                    &view,
                    params.bindless_tex_args.is_some(),
                    &mut draws,
                );
                self.encode_transparent(
                    cmd_buf,
                    &view,
                    scene_pre_taa,
                    &draws,
                    params.rt_reflection_params,
                    params.bindless_tex_args,
                )?
            }
            PassId::Raymarch => {
                let view = self.build_raymarch_view(params);
                self.encode_raymarch(cmd_buf, &view, params.frustum)?
            }
        })
    }
}
