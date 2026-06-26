// Frame recording for the Vulkan backend.
// Encodes the shadow pass, main scene pass, and post stack into a single
// command buffer. Called from VkContext::draw_frame each tick. Every
// pass dispatches through the render-graph executor; see
// `vulkan/graph_exec.rs` and `vulkan/composite.rs`.

use ash::vk;

use crate::gfx::render_graph::{FrameGraphInputs, build_frame_graph};
use crate::gfx::render_types::{LightUniforms, ShadowUniforms, TextDrawCall};

use super::context::VkContext;
use super::graph_exec::GraphFrameParams;
use super::math::{mat4_mul, perspective};

// ViewUniforms layout (160 bytes, std140) must match GLSL ViewBlock exactly.
// `view_mat` is the camera view matrix used to compute view-space depth in
// the vertex shader (for shadow cascade selection). cam_pos is stored as
// three individual floats to avoid std140 vec3 alignment bumping subsequent
// fields.
#[derive(Copy, Clone)]
#[repr(C)]
pub(super) struct ViewUniforms {
    pub vp: [[f32; 4]; 4],
    pub view_mat: [[f32; 4]; 4],
    pub elapsed: f32,
    // 1.0 when an SSR / RT reflection composite owns the sharp specular this frame,
    // so the forward bindless shader fades its glossy-dielectric probe specular to
    // avoid double-counting; 0.0 keeps the full forward reflection (and at probe
    // bakes, where no resolve runs). Repurposes the former offset-132 pad.
    pub reflections_enabled: f32,
    pub cam_x: f32,
    pub cam_y: f32,
    pub cam_z: f32,
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = IBL off.
    pub prefilter_mip_count: f32,
    pub _ep0: f32,
    pub _ep1: f32,
}

// One term of the Halton low-discrepancy sequence, drives the sub-pixel
// projection jitter so successive TAA frames sample slightly different
// positions. Mirrors `halton` in metal/draw.rs.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0_f32;
    let mut f = 1.0_f32;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

impl VkContext {
    // Rebuild this frame's `GpuObjectData` storage buffer for the bindless
    // static pass: one 144-byte record per build-time `DrawObject`, indexed
    // by object id. Streamed `VoxelWorld` chunks (past `n_objects`) are
    // skipped: they render through the legacy pipeline. The pool indices
    // address the deduplicated `[albedo..] ++ [normal..]` texture pool:
    // albedo = `texture_slot`, normal = `albedo_count + normal_map_slot`,
    // both clamped to the pool. Rebuilt every frame so `update_model` /
    // `update_visibility` edits are reflected; a no-op when bindless is off.
    fn build_object_buffer(&self, frame_idx: usize) {
        let Some(&ptr) = self.cull.object_buffer_ptrs.get(frame_idx) else {
            return;
        };
        self.build_object_records_into(ptr);
    }

    // Write the bindless `GpuObjectData` records (static + streamed-chunk +
    // skinned-tail) into a mapped buffer at `ptr`. Factored out of
    // `build_object_buffer` so the reflection-probe capture can build the same
    // records into its own bake-owned buffer (the instance tail is left untouched,
    // so a bake buffer must be zeroed first -- a zero record is a disabled draw the
    // cull kernel skips, which is how the probe omits instanced geometry in V1).
    pub(in crate::vulkan) fn build_object_records_into(&self, ptr: *mut u8) {
        use crate::gfx::render_types::{GpuObjectData, pack_object_record, pack_skinned_record};
        let albedo_count = self.textures.len();
        let last_tex = albedo_count.saturating_sub(1);
        let last_nm = self.normal_map_textures.len().saturating_sub(1);
        let stride = std::mem::size_of::<GpuObjectData>();
        for (i, obj) in self.draw_objects.iter().take(self.n_objects).enumerate() {
            let albedo = obj.texture_slot.min(last_tex) as u32;
            let normal = (albedo_count + obj.normal_map_slot.min(last_nm)) as u32;
            let rec = pack_object_record(obj, albedo, normal);
            // SAFETY: the buffer was sized for `n_objects + n_instances + n_skinned`
            // records (the instance tail is written once at init) and the loop is
            // bounded by `take(n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }

        // Streamed chunks: one record each in the reserved region at
        // `[chunk_record_base() + k]`, packed like a static object (chunk geometry
        // already lives in the shared VB/IB with the chunk's `base_vertex`, so they
        // ride the static + instance prefix indirect draw). Per-chunk flat-pool
        // texture indices give per-chunk materials. A non-resident / unused slot's
        // stale record here is never read -- `build_draw_args_buffer` disables it,
        // and the cull kernel skips `objects[i]` for a disabled record.
        let chunk_base = self.chunk_record_base();
        self.for_each_chunk_record(|k, obj| {
            let albedo = obj.texture_slot.min(last_tex) as u32;
            let normal = (albedo_count + obj.normal_map_slot.min(last_nm)) as u32;
            let rec = pack_object_record(obj, albedo, normal);
            // SAFETY: the chunk reserve is `[chunk_base, chunk_base + n_chunk)` and
            // `for_each_chunk_record` caps `k < n_chunk`, so the write is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        });

        // Skinned objects: one record each in the reserved tail at
        // `[skinned_record_base(), cull_count())`. `model = obj.model` (applied
        // after the per-frame skin deform), flat-pool texture indices like a static
        // object, and a padded bind-pose AABB so the cull kernel can frustum/Hi-Z
        // test them. Drawn by the main pass's 2nd indirect draw. `take(n_skinned)`
        // no-ops when the fold is inactive.
        let skinned_base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.n_skinned)
            .enumerate()
        {
            let albedo = obj.texture_slot.min(last_tex) as u32;
            let normal = (albedo_count + obj.normal_map_slot.min(last_nm)) as u32;
            let rec = pack_skinned_record(obj, albedo, normal);
            // SAFETY: the buffer reserved `n_skinned` records past
            // `skinned_record_base()` at init; the loop is bounded by
            // `self.skinned.draw_objects.len() == self.n_skinned`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add((skinned_base + k) * stride),
                    stride,
                );
            }
        }
    }

    // Rebuild this frame's `GpuDrawArgs` storage buffer for the GPU-cull
    // compute kernel: one 16-byte record per build-time `DrawObject`, carrying
    // the indexed-draw arguments the kernel encodes plus the per-frame
    // cull-decision bits (`update_visibility` / streaming residency). Streamed
    // chunks (past `n_objects`) are skipped; a no-op when bindless is off.
    // The per-object `(index_offset, index_count)` is the active LOD slice
    // picked by camera distance, so the bindless main pass renders the
    // chosen LOD with no shader-side change. Mirrors `directx/cull.rs`.
    fn build_draw_args_buffer(&self, frame_idx: usize, cam_pos: [f32; 3]) {
        let Some(&ptr) = self.cull.draw_args_buffer_ptrs.get(frame_idx) else {
            return;
        };
        self.build_draw_args_records_into(ptr, cam_pos);
    }

    // Write the GPU-cull `GpuDrawArgs` records (static + streamed-chunk +
    // skinned-tail, the per-object active-LOD slice picked by distance from
    // `cam_pos`) into a mapped buffer at `ptr`. Factored out of
    // `build_draw_args_buffer` so the reflection-probe capture can build the same
    // args into its own bake-owned buffer against the probe eye. The instance tail
    // is left untouched (a zeroed bake buffer keeps it disabled = skipped).
    pub(in crate::vulkan) fn build_draw_args_records_into(&self, ptr: *mut u8, cam_pos: [f32; 3]) {
        use crate::gfx::render_types::{GpuDrawArgs, draw_args_flags};
        let stride = std::mem::size_of::<GpuDrawArgs>();
        for (i, obj) in self.draw_objects.iter().take(self.n_objects).enumerate() {
            // Per-frame active LOD pick. Objects with no alternates fall
            // straight through to LOD0.
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                flags: draw_args_flags(obj.visible, obj.resident, obj.cullable()),
            };
            // SAFETY: the buffer was sized for `n_objects + n_instances + n_skinned`
            // records (the instance tail is written once at init) and the loop is
            // bounded by `take(n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuDrawArgs as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }

        // Streamed chunks: one draw-arg each in the reserved region at
        // `[chunk_record_base() + k]`. Chunk geometry lives in the shared VB/IB, so
        // the args carry the chunk's own `base_vertex` + index slice and the chunk
        // rides the static + instance prefix indirect draw. Chunks are non-cullable
        // (NaN AABB), so a resident chunk draws unconditionally; a freed slot's
        // `resident` clear disables it. The unused reserve tail is disabled.
        let chunk_base = self.chunk_record_base();
        let n_resident_chunks = self.for_each_chunk_record(|k, obj| {
            // Chunks have no LOD alternates; `active_lod(0.0)` returns the base slice
            // (and avoids a NaN camera distance from the chunk's NaN AABB).
            let (index_offset, index_count) = obj.active_lod(0.0);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                flags: draw_args_flags(obj.visible, obj.resident, obj.cullable()),
            };
            // SAFETY: `for_each_chunk_record` caps `k < n_chunk`, so
            // `chunk_base + k < skinned_record_base()`, in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuDrawArgs as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        });
        // Disable the unused chunk reserve tail so vacated / never-used slots draw
        // nothing (the cull kernel skips `objects[i]` for an ENABLED-clear record).
        let disabled = GpuDrawArgs {
            index_count: 0,
            index_offset: 0,
            base_vertex: 0,
            flags: 0,
        };
        for k in n_resident_chunks..self.n_chunk {
            // SAFETY: `k < n_chunk`, so `chunk_base + k < skinned_record_base()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &disabled as *const GpuDrawArgs as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        }

        // Skinned objects: one record each in the reserved tail. The main pass's
        // 2nd indirect draw binds the per-frame deformed VB + the skinned u16 IB,
        // so `base_vertex = 0` (the deformed buffer mirrors global skinned indexing)
        // and the active-LOD slice is the element offset into the skinned IB.
        // Skinned objects carry a finite padded bind-pose AABB (`pack_skinned_record`),
        // so they are cullable + resident. `take(n_skinned)` no-ops when inactive.
        let skinned_base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.n_skinned)
            .enumerate()
        {
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: 0,
                flags: draw_args_flags(obj.visible, true, true),
            };
            // SAFETY: the buffers reserved `n_skinned` records past
            // `skinned_record_base()` at init; the loop is bounded by
            // `self.skinned.draw_objects.len() == self.n_skinned`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuDrawArgs as *const u8,
                    ptr.add((skinned_base + k) * stride),
                    stride,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_frame(
        &mut self,
        cmd: vk::CommandBuffer,
        image_index: u32,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
        frame_idx: usize,
    ) -> Result<Vec<vk::CommandBuffer>, String> {
        let device = self.device.clone();
        let device = &device;
        // The scene rasterises at render resolution (== swapchain extent unless
        // upscaling). Cascade / projection aspect, the HDR graph dims, and the
        // sub-pixel jitter all derive from this, not the display extent.
        let extent = self.render_extent;

        // Profiler-overlay timestamp pair. The pool slot for this frame is
        // reset (the matching `get_query_pool_results` already ran at the top
        // of `draw_frame`, after the fence wait that gated the previous trip's
        // writes), then the start tick is recorded as the first cmd-buffer
        // op. The matching end tick is written just before
        // `end_command_buffer` returns control. Mirrors the DirectX
        // EndQuery(TIMESTAMP) + ResolveQueryData pattern.
        // Outer "start" command buffer: the leading timestamp (the frame's
        // first GPU op), recorded into its own buffer so it can be submitted
        // before the per-pass buffers. The query-pool reset must precede every
        // pass, hence it lives here at the head of the batch.
        let start_cmd = self.commands.start_command_buffers[frame_idx];
        unsafe {
            device
                .reset_command_buffer(start_cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset start cmd buf: {e}"))?;
            device
                .begin_command_buffer(
                    start_cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("begin start cmd buf: {e}"))?;
        }
        if let Some(pool) = self.timestamp_query_pool {
            // Reset this frame's whole timestamp block (whole-frame pair + every
            // per-pass pair) here, before any per-pass buffer writes into it.
            // `start_cmd` is submitted first, so the reset precedes every write in
            // queue order. The whole-frame start goes in the block's first slot;
            // each pass writes its own pair (see graph_exec); the whole-frame end
            // is the block's second slot, written in the end buffer below.
            let block_base = super::pass_timing::frame_block_base(frame_idx);
            let (wf_start, _) = super::pass_timing::whole_frame_pair(frame_idx);
            unsafe {
                device.cmd_reset_query_pool(
                    start_cmd,
                    pool,
                    block_base,
                    super::pass_timing::SLOTS_PER_FRAME as u32,
                );
                device.cmd_write_timestamp(
                    start_cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    wf_start,
                );
            }
        }
        // Hardware ray-traced reflections: rebuild the TLAS for moved props (when
        // the dynamic mode + dirty gate call for it) onto the start buffer, which
        // is submitted before every per-pass trace, then re-point this frame's RT
        // descriptor set at the live TLAS + geometry table. A no-op when RT is
        // off. Recorded here (not on a per-pass worker) because it needs
        // `&mut self`; the start buffer carries it within the frame's submit batch.
        self.rt_dynamic_update(start_cmd, frame_idx);
        unsafe {
            device
                .end_command_buffer(start_cmd)
                .map_err(|e| format!("end start cmd buf: {e}"))?;
        }

        // Recompute cascade VPs + splits from the current camera + light, and
        // push the result to the shadow UBO so both passes see the same data.
        let cascade_aspect = if extent.height == 0 {
            1.0
        } else {
            extent.width as f32 / extent.height as f32
        };
        if self.shadow.pipeline.is_some() {
            let fresh = crate::gfx::csm::compute_shadow_uniforms(
                self.view_matrix,
                cam_pos,
                fov_y_radians,
                cascade_aspect,
                near,
                crate::gfx::csm::DEFAULT_SHADOW_DISTANCE.min(far),
                self.shadow.light_dir,
                self.shadow.map_size,
            );
            // Advance the cascade schedule and refresh only this frame's
            // cascades' light VPs; skipped cascades keep the VP + depth their
            // slice was last rendered with, so the Main pass samples each cascade
            // consistently. Splits depend only on the camera range (not which
            // cascades render), so always refresh. encode_shadow_pass
            // re-rasterizes only the masked slices.
            let update = self.shadow.update;
            let mask = self.shadow.scheduler.next_mask(update);
            self.shadow.render_mask = mask;
            self.shadow.uniforms.cascade_splits = fresh.cascade_splits;
            for i in 0..crate::gfx::render_types::NUM_SHADOW_CASCADES {
                if mask & (1u32 << i) != 0 {
                    self.shadow.uniforms.light_vps[i] = fresh.light_vps[i];
                }
            }
            upload_shadow_uniforms(device, self.shadow.ubo_memory, &self.shadow.uniforms)?;
        }

        // Push this frame's skinning matrices into the per-frame joint buffers
        // before the skinned shadow + main passes read them. No-op when no
        // SkinnedMesh is declared.
        self.upload_joint_matrices(frame_idx);

        // Auto-exposure: step the EMA from a previous frame's GPU
        // measurement before any pipeline reads `post_process.exposure`.
        // The fence wait at the top of `draw_frame` already gated the
        // GPU work that wrote this slot's readback, so the value is
        // committed. No-op when auto-exposure is disabled.
        self.update_auto_exposure(elapsed, frame_idx);

        //  Per-frame seed inputs for the shared backend-agnostic frame
        //  builder ([gfx/render_graph/frame.rs](../../gfx/render_graph/frame.rs)).
        //  Decals landed 2026-05-24; Fog followed; AutoExposure landed
        //  2026-05-25; Particles landed 2026-05-25. The flags track
        //  whether each pipeline is built: the encoders skip cheaply
        //  when there is nothing live to draw.
        let seed_inputs = FrameGraphInputs {
            shadow_enabled: self.shadow.pipeline.is_some(),
            shadow_map_size: self.shadow.map_size,
            hdr_width: extent.width,
            hdr_height: extent.height,
            hdr_sample_count: self.msaa_samples.as_raw(),
            bindless_cull_enabled: self.cull.cull_pipeline.is_some() && self.cull_count() > 0,
            auto_exposure_enabled: self.auto_exposure.is_some(),
            bloom_enabled: self.post_process.bloom_intensity > 0.0,
            // Velocity (motion vectors) runs for TAA *or* temporal upscaling
            // (FSR consumes them); TAA resources are forced built under
            // upscaling, so `taa.is_some()` already implies it, but spell out
            // the upscale case too.
            velocity_enabled: self.taa.is_some() || self.upscale.is_some(),
            // TAA resolve and Upscale are mutually exclusive (both do temporal
            // accumulation and share the graph slot). Drop TAA when upscaling.
            taa_enabled: self.taa.is_some() && self.upscale.is_none(),
            // The SSR *resolve* (reflection compositing). `self.ssr` may exist
            // for a SSGI-only build (it owns the shared pre-pass G-buffer), so
            // gate the resolve node on the dedicated flag, not `ssr.is_some()`.
            ssr_enabled: self.ssr_resolve_active,
            particles_enabled: self.particle_resources.is_some()
                && self.particles.iter().any(|p| p.is_some()),
            // Gated on both the resources (built at init when the world declared
            // a VolumetricFog) and the live settings, so runtime
            // `update_fog_settings(None)` drops the FogFroxel + Fog passes from
            // the graph entirely. Mirrors Metal's `pipeline && settings` gate.
            fog_enabled: self.fog_resources.is_some() && self.fog_settings.is_some(),
            decals_enabled: self.decals_state.is_some() && self.decals.iter().any(|d| d.is_some()),
            // The SSR pre-pass G-buffer is shared with SSGI, so it runs whenever
            // `self.ssr` exists (built for SSR resolve *or* SSGI).
            ssr_prepass_enabled: self.ssr.is_some(),
            ssao_enabled: self.ssao.is_some(),
            // Gated on the resources (built at init when at least one `.glsl`
            // SdfVolume survived the filter) AND a currently-visible volume, so
            // an all-hidden world drops the pass from the graph.
            raymarch_enabled: self.raymarch.as_ref().is_some_and(|r| r.any_visible()),
            // Temporal upscaling (FSR via FidelityFX). `Some` only when the
            // world opted in AND the FFX VK runtime + context built; the shared
            // builder then runs `Upscale` in the `TaaResolve` slot, reading the
            // post-SSR scene + velocity and writing the swapchain-res scene the
            // bloom + composite stack samples.
            upscale_enabled: self.upscale.is_some(),
            // Transparent / translucent pass: on when the world declared
            // visible `GlassPanel`s (the only producer on this backend; water
            // is Metal-only). The shared builder then seeds the Transparent
            // node and the executor draws the glass over the post-SSR scene.
            transparent_enabled: self.glass.as_ref().is_some_and(|g| g.any_visible()),
            // Two-pass Hi-Z occlusion: inserts HizBuild -> Cull2 -> Main2 after
            // Main when the world requested `occlusion_two_pass` and the bindless
            // GPU-cull path + phase-2 resources are live. `two_pass_occlusion_active`
            // is the single gate the executor's phase-2 arms + the phase-1
            // render-pass selection share, so the graph shape matches what the
            // executor dispatches. The graph builder further ANDs this with
            // `bindless_cull_enabled`, which is already implied here.
            two_pass_occlusion_enabled: self.two_pass_occlusion_active(),
            // Screen-space global illumination. `Some` only when the world
            // selected `indirect_lighting: ssgi`; the graph then inserts the
            // `Ssgi` node on the hdr_resolve RMW chain (which forces the SSR
            // pre-pass on, since `self.ssr` is built for SSGI too).
            ssgi_enabled: self.ssgi.is_some(),
            // Hardware ray-traced reflections (`VK_KHR_ray_query`). On only when
            // the world requested it, the GPU exposed the ray-query extensions,
            // and the acceleration structure built; the shared builder then emits
            // `RtReflections` in the `SsrResolve` slot (RT takes precedence; SSR
            // is the non-RT-GPU fallback).
            rt_reflections_enabled: self.rt_reflections_active(),
            // Unified geometry pre-pass: one `GBufferPrepass` node rasterises the
            // normal+depth / roughness / velocity MRT every screen-space consumer
            // (SSR / SSAO / SSGI / TAA / FSR) reads, replacing the separate SSR /
            // SSAO / velocity pre-passes. On exactly when the merged buffer was
            // built (any of those consumers is live); the shared builder then
            // emits the single node and skips `SsrPrepass` / `Velocity`. Mirrors
            // DirectX's `unified_gbuffer_prepass: self.gbuffer.is_some()`.
            unified_gbuffer_prepass: self.gbuffer.is_some(),
        };

        //  Camera projection + per-frame view state. Computed before the main
        //  render pass begins so the GPU-cull compute dispatch (which Vulkan
        //  forbids inside a render pass) can read this frame's frustum.
        let aspect = if extent.height == 0 {
            1.0
        } else {
            extent.width as f32 / extent.height as f32
        };
        let proj = perspective(fov_y_radians, aspect, near, far);
        // Un-jittered camera VP, fed to the velocity pre-pass so the stored
        // motion vector is free of the sub-pixel projection jitter.
        let cur_vp = mat4_mul(proj, self.view_matrix);
        // When TAA is on, offset the projection by a sub-pixel Halton jitter so
        // the accumulation has fresh sample positions each frame. The jitter is
        // a pure NDC x/y shift (depth is unaffected): `proj[2][0/1]` are the
        // z-coefficients of clip x/y, so subtracting the jitter there shifts
        // post-divide NDC by exactly the jitter amount (clip.w == -view_z).
        // Mirrors the jitter in metal/draw.rs.
        let render_proj = if let Some(up) = self.upscale.as_ref() {
            // The active backend prescribes the jitter sequence (render-pixel
            // units): FSR queries its FFX-tuned offsets, DLSS / XeSS use the
            // shared Halton-2/3. The same offset feeds the dispatch via
            // `set_jitter`. Phase index = the TAA frame counter, which advances
            // every frame here (TAA resources are forced built under upscaling).
            let phase = self.taa.as_ref().map(|t| t.taa_frame).unwrap_or(0);
            let [jx_px, jy_px] = up.jitter_offset(phase);
            up.set_jitter([jx_px, jy_px]);
            let mut p = proj;
            p[2][0] -= jx_px * 2.0 / extent.width.max(1) as f32;
            p[2][1] -= jy_px * 2.0 / extent.height.max(1) as f32;
            p
        } else if let Some(taa_frame) = self.taa.as_ref().map(|t| t.taa_frame) {
            let idx = taa_frame % 8 + 1;
            let jx = (halton(idx, 2) - 0.5) * 2.0 / extent.width.max(1) as f32;
            let jy = (halton(idx, 3) - 0.5) * 2.0 / extent.height.max(1) as f32;
            let mut p = proj;
            p[2][0] -= jx;
            p[2][1] -= jy;
            p
        } else {
            proj
        };
        let vp_mat = mat4_mul(render_proj, self.view_matrix);
        // Update view UBO for this frame.
        let view_uni = ViewUniforms {
            vp: vp_mat,
            view_mat: self.view_matrix,
            elapsed,
            // Hand glossy dielectric specular to the SSR / RT resolve when its
            // composite owns the scene image this frame (the composite is present
            // iff a resolve is active), else the forward shader keeps it all.
            reflections_enabled: if self.reflection_composite.is_some() {
                1.0
            } else {
                0.0
            },
            cam_x: cam_pos[0],
            cam_y: cam_pos[1],
            cam_z: cam_pos[2],
            prefilter_mip_count: self.prefilter_mip_count as f32,
            _ep0: 0.0,
            _ep1: 0.0,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &view_uni as *const ViewUniforms as *const u8,
                self.uniforms.view_ubo_ptrs[frame_idx],
                std::mem::size_of::<ViewUniforms>(),
            );
        }

        // Reflection-probe set (global set 0 binding 7): EMPTY (count 0 = sky
        // reflection) until a probe bakes, so the forward shader keeps the sky
        // path. Uploaded every frame so a later install is picked up immediately.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.probe_set as *const super::probe_uniforms::ProbeSet as *const u8,
                self.uniforms.probe_set_ubo_ptrs[frame_idx],
                std::mem::size_of::<super::probe_uniforms::ProbeSet>(),
            );
        }

        let frustum = crate::gfx::frustum::Frustum::from_view_projection(vp_mat);

        // Compute-cull host-side prep: rebuild this frame's
        // `GpuObjectData` + `GpuDrawArgs` storage buffers with the
        // latest per-object state. These are mapped-memory writes and
        // run unconditionally on the CPU side so the cull compute pass
        // (dispatched by the graph executor below) sees fresh data.
        // The compute dispatch itself runs through the graph as
        // `PassId::Cull`; the toposort orders Cull → Main via the
        // `draw_args` buffer RAW edge declared on Main.
        if seed_inputs.bindless_cull_enabled {
            self.build_object_buffer(frame_idx);
            self.build_draw_args_buffer(frame_idx, cam_pos);
        }

        // CPU visibility list (BVH-culled cullables + always_draw fallback).
        // Computed before the main render pass so the SSAO pre-pass below can
        // walk the same set, and so velocity / TAA later can reuse it without
        // a second BVH walk. `mem::take` swaps out the persistent scratch
        // buffer so its heap allocation is reused across frames; it's put
        // back below before we return Ok (error path loses capacity, fine
        // since record_frame errors are exceptional).
        let mut visible = std::mem::take(&mut self.visible_scratch);
        visible.clear();
        self.cull_bvh
            .query(&frustum, cam_pos, |idx| visible.push(idx));
        visible.sort_unstable();
        visible.extend_from_slice(&self.always_draw);

        //  Single merged frame graph dispatched in one
        //  `execute_graph` call. The toposort orders Cull → Main via
        //  the `draw_args` buffer RAW edge, SsaoBlur / Shadow → Main
        //  via their texture RAW edges, and the post-stack chain
        //  (SsrResolve → TaaResolve → Bloom → Composite) via the
        //  scene_color version chain. Velocity pins before TaaResolve
        //  via the `velocity` texture read. Shadow's encoder owns
        //  its DEPTH → SHADER_READ post-loop transition; Cull's owns
        //  its SHADER_WRITE → INDIRECT_COMMAND_READ memory barrier;
        //  each render-pass attachment pins the per-pass layouts.
        //  Deriving `vkCmdPipelineBarrier` from `pass.barriers_before`
        //  (so encoders can shed their inline barriers) is a
        //  follow-up. The pre-graph dispatch site sits at the natural
        //  Main location: after the per-frame view/projection +
        //  visible-set compute (Vulkan forbids compute inside a
        //  render pass, so Cull must come before any render-pass
        //  dispatch from the executor).
        let graph = build_frame_graph(&seed_inputs).map_err(|e| format!("frame graph: {e}"))?;
        let params = GraphFrameParams {
            cmd,
            image_index,
            frame_idx,
            text_calls,
            visible: &visible,
            frustum: &frustum,
            cam_pos,
            vp_mat,
            cur_vp,
            fov_y_radians,
            aspect,
            elapsed,
            near,
            far,
        };
        // Each non-composite pass is recorded into its own command buffer
        // (returned here in graph order); Composite + the post-graph work below
        // record into `cmd` (the outer "end" buffer). The whole frame is
        // submitted as `[start, ...pass_bufs, end]` by `draw_frame`.
        let pass_bufs = self.execute_graph(&graph, &params)?;

        // Hi-Z occlusion: reduce this frame's main depth into the depth-mip
        // pyramid that next frame's `Cull` dispatch consults. Runs inline on
        // the frame's command buffer after the graph (so the Main pass has
        // written depth and any decal / fog pass has restored it to
        // DEPTH_STENCIL_ATTACHMENT_OPTIMAL). A no-op when GPU-cull is off.
        self.encode_hiz_build(cmd, frame_idx);

        // The cascade slices rest sampled (SHADER_READ_ONLY_OPTIMAL) between
        // frames; next frame's Shadow producer barrier (graph-driven) performs
        // the SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT reset over every
        // cascade layer, so no inline end-of-frame restore is needed here.

        // Advance the TAA jitter sequence (`taa_frame > 0` also validates the
        // history for next frame). The motion-vector temporal state lives on the
        // unified G-buffer (advanced below); TAA only consumes its velocity view.
        if let Some(taa) = &mut self.taa {
            taa.taa_frame = taa.taa_frame.wrapping_add(1);
        }

        // Advance the unified G-buffer's velocity-channel temporal state in
        // lockstep with TAA's: this frame's un-jittered VP becomes next frame's
        // `prev_vp`, and every object transform is snapshotted so the next
        // GBufferPrepass can diff against it. Owned by `GbufferResources` so the
        // motion vector works for any consumer (TAA or FSR), exactly mirroring
        // the TAA advance above. Mirrors DirectX's `prev_view_proj`/`prev_models`
        // bookkeeping in `record_frame`.
        if let Some(gb) = &mut self.gbuffer {
            gb.prev_view_proj = cur_vp;
            gb.prev_models
                .resize(self.draw_objects.len(), super::math::IDENTITY4);
            for (prev, obj) in gb.prev_models.iter_mut().zip(self.draw_objects.iter()) {
                *prev = obj.model;
            }
        }

        // Advance Hi-Z temporal state: this frame's un-jittered VP becomes next
        // frame's occlusion-test projection, and the pyramid `encode_hiz_build`
        // just wrote is now valid for next frame's cull (kept independent of
        // TAA, which may be off while Hi-Z is on).
        if self.cull.hiz.is_some() {
            self.cull.hiz_prev_view_proj = cur_vp;
            self.cull.hiz_valid = true;
        }

        self.visible_scratch = visible;

        // Drain the parallel-safe draw-call accumulator (bumped by every pass
        // encoder, including those fanned onto rayon workers) into this frame's
        // `frame_stats` for the profiler overlay. All recording is done by here.
        let mut stats = self.frame_stats.get();
        stats.draw_calls = self
            .draw_calls_accum
            .load(std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(stats);

        // End-of-frame timestamp for the profiler overlay. Pairs with the
        // TOP_OF_PIPE write near the top of the function (the block's first pair).
        if let Some(pool) = self.timestamp_query_pool {
            let (_, wf_end) = super::pass_timing::whole_frame_pair(frame_idx);
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    wf_end,
                );
            }
        }

        // The per-frame submission order, excluding the outer "end" buffer
        // (`cmd`) the caller appends: the `start` buffer (leading timestamp)
        // followed by the per-pass buffers in graph order. Submission order is
        // GPU order on the single graphics queue, so every encoder's inline
        // barrier synchronises against the prior pass across buffer boundaries.
        let mut submit = Vec::with_capacity(pass_bufs.len() + 1);
        submit.push(start_cmd);
        submit.extend(pass_bufs);
        Ok(submit)
    }
}

// A buffer+memory pair that must be destroyed after the GPU finishes using it.
//
// `frame` records the frame-in-flight slot whose command buffer references
// the buffer; it is only safe to destroy once that slot's fence has been
// waited on again (i.e. the next time `draw_frame` reuses the same slot).
pub(super) struct DeferredBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub frame: usize,
}

impl DeferredBuffer {
    pub(super) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

// Helper: update the shared ShadowUniforms UBO (called at init and when lights change).
pub(super) fn upload_shadow_uniforms(
    device: &ash::Device,
    shadow_ubo_memory: vk::DeviceMemory,
    su: &ShadowUniforms,
) -> Result<(), String> {
    let size = std::mem::size_of::<ShadowUniforms>() as u64;
    unsafe {
        let ptr = device
            .map_memory(shadow_ubo_memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map shadow ubo: {e}"))? as *mut u8;
        std::ptr::copy_nonoverlapping(su as *const ShadowUniforms as *const u8, ptr, size as usize);
        device.unmap_memory(shadow_ubo_memory);
    }
    Ok(())
}

// Helper: upload LightUniforms to the shared light UBO.
pub(super) fn upload_light_uniforms(
    device: &ash::Device,
    light_ubo_memory: vk::DeviceMemory,
    lu: &LightUniforms,
) -> Result<(), String> {
    let size = std::mem::size_of::<LightUniforms>() as u64;
    unsafe {
        let ptr = device
            .map_memory(light_ubo_memory, 0, size, vk::MemoryMapFlags::empty())
            .map_err(|e| format!("map light ubo: {e}"))? as *mut u8;
        std::ptr::copy_nonoverlapping(lu as *const LightUniforms as *const u8, ptr, size as usize);
        device.unmap_memory(light_ubo_memory);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ViewUniforms must match the std140 `ViewBlock` UBO in the main-pass
    // shaders: two mat4 then elapsed/pad and the camera position as three
    // scalars, prefilter mip count, and two end pads (160 B total).
    #[test]
    fn view_uniforms_layout_matches_glsl() {
        assert_eq!(std::mem::size_of::<ViewUniforms>(), 160);
        assert_eq!(std::mem::offset_of!(ViewUniforms, vp), 0);
        assert_eq!(std::mem::offset_of!(ViewUniforms, view_mat), 64);
        assert_eq!(std::mem::offset_of!(ViewUniforms, elapsed), 128);
        assert_eq!(std::mem::offset_of!(ViewUniforms, reflections_enabled), 132);
        assert_eq!(std::mem::offset_of!(ViewUniforms, cam_x), 136);
        assert_eq!(std::mem::offset_of!(ViewUniforms, cam_y), 140);
        assert_eq!(std::mem::offset_of!(ViewUniforms, cam_z), 144);
        assert_eq!(std::mem::offset_of!(ViewUniforms, prefilter_mip_count), 148);
        assert_eq!(std::mem::offset_of!(ViewUniforms, _ep0), 152);
        assert_eq!(std::mem::offset_of!(ViewUniforms, _ep1), 156);
    }
}
