// src/vulkan/cull.rs
//
// Compute-driven cull pass for the Vulkan backend. One compute
// invocation per build-time `DrawObject` frustum / distance-tests the
// object and writes its `VkDrawIndexedIndirectCommand` (with
// `instance_count` 0 for culled / disabled objects) into this frame's
// indirect buffer. The bindless Main pass then consumes the buffer via
// one `cmd_draw_indexed_indirect`. Must run outside any render pass
// (Vulkan disallows compute dispatch inside a render pass), which is
// why the graph dispatch site in `vulkan/draw.rs::record_frame` sits
// before `cmd_begin_render_pass` for Shadow / Main.
//
// The shape mirrors `metal/cull.rs::encode_cull`; the graph executor
// in [`graph_exec.rs`](graph_exec.rs) dispatches `PassId::Cull` here.
//
// CPU-side per-frame buffer rebuilds (`build_object_buffer` +
// `build_draw_args_buffer`) stay in `record_frame`: they're host
// writes to mapped GPU memory, not part of the GPU command stream the
// graph orders.

use ash::vk;

use crate::gfx::frustum::Frustum;

use super::context::VkContext;
use super::hiz::CullHizParams;

// Push constants for the GPU-cull compute kernel (112 bytes, std430).
// Must match the `CullParams` push-constant block in CULL_COMPUTE_GLSL:
// six already-normalised frustum planes (xyz = normal, w = d), the
// camera position, and the build-time object count (`cam_pos` +
// `object_count` share one std430 16-byte slot). Size pinned by
// `CULL_PUSH_CONSTANT_BYTES`.
#[derive(Copy, Clone)]
#[repr(C)]
struct CullParams {
    planes: [[f32; 4]; 6],
    cam_pos: [f32; 3],
    object_count: u32,
}

impl VkContext {
    // Total records the GPU-driven cull + bindless main pass processes: the
    // build-time static objects, the instanced-cluster instances folded in after
    // them, then the skinned objects (`n_objects + n_instances + n_skinned`). The
    // cull dispatch + the `GpuObjectData` / `GpuDrawArgs` / indirect buffers all
    // count this; the main pass draws the static+instance prefix and the skinned
    // tail with two `cmd_draw_indexed_indirect` calls. With no instanced props /
    // skinned meshes (or a non-bindless world) the extra terms are 0, leaving it
    // equal to the static `n_objects`. Mirrors `directx/cull.rs::cull_count`.
    pub(in crate::vulkan) fn cull_count(&self) -> usize {
        self.n_objects + self.n_instances + self.n_chunk + self.n_skinned
    }

    // Buffer index of the first streamed-chunk record. The chunk reserve is
    // `[chunk_record_base(), skinned_record_base())`; resident chunks pack into the
    // front each frame and the unused tail is disabled. Chunks ride the
    // static+instance prefix indirect draw (their geometry lives in the shared
    // VB/IB), so this is just the instance tail. Mirrors `directx/cull.rs`.
    pub(in crate::vulkan) fn chunk_record_base(&self) -> usize {
        self.n_objects + self.n_instances
    }

    // Buffer index of the first skinned record: the static + instance + chunk
    // prefix the first indirect draw covers ends here, and the skinned tail
    // `[skinned_record_base(), cull_count())` is the second indirect draw. The
    // chunk reserve sits inside the prefix, so the skinned base is past it.
    pub(in crate::vulkan) fn skinned_record_base(&self) -> usize {
        self.n_objects + self.n_instances + self.n_chunk
    }

    // Walk the resident streamed-chunk draw objects -- the build-time-geometry tail
    // past `n_objects` that are NOT runtime clones -- invoking `emit` with the
    // chunk's reserve index `k` (into `[chunk_record_base() + k]`) + the DrawObject.
    // Chunk geometry already lives in the shared VB/IB, so chunks fold into the
    // static+instance prefix indirect draw as plain records (with their own
    // `base_vertex` + flat-pool material). Runtime clones (in `clone_slot_by_draw_idx`)
    // are skipped -- they keep the legacy per-object path. Bounded by the chunk
    // reserve `n_chunk`. Returns the number of chunk records emitted (so the caller
    // can disable the unused reserve tail). Mirrors `directx/draw_iter.rs`.
    pub(in crate::vulkan) fn for_each_chunk_record<F>(&self, mut emit: F) -> usize
    where
        F: FnMut(usize, &crate::gfx::render_types::DrawObject),
    {
        if self.n_chunk == 0 {
            return 0;
        }
        let mut k = 0;
        for (i, obj) in self.draw_objects.iter().enumerate().skip(self.n_objects) {
            if self.clone_slot_by_draw_idx.contains_key(&i) {
                continue; // runtime clone -> legacy per-object path
            }
            if k >= self.n_chunk {
                break;
            }
            emit(k, obj);
            k += 1;
        }
        k
    }

    // True when two-pass Hi-Z occlusion runs this frame: the world requested
    // `occlusion_two_pass`, the phase-2 cull pipeline + Hi-Z + second indirect
    // buffers are built, and the bindless GPU-cull path is active with
    // build-time geometry. This is the exact condition under which the shared
    // graph inserts the HizBuild / Cull2 / Main2 chain, so the frame-graph seed
    // (`record_frame`), the phase-1 render-pass selection (`encode_main_pass`),
    // and the executor's phase-2 arms all gate on it identically. Mirrors
    // `directx/cull.rs::two_pass_occlusion_active`.
    pub(in crate::vulkan) fn two_pass_occlusion_active(&self) -> bool {
        self.cull.occlusion_two_pass
            && self.cull.cull_pipeline_phase2.is_some()
            && self.cull.hiz.is_some()
            && self.cull.cull_pipeline.is_some()
            && !self.cull.indirect_buffers2.is_empty()
            && self.cull_count() > 0
    }

    // Dispatch the compute-driven cull pass for frame slot
    // `frame_idx`. Ends with a memory barrier ordering the kernel's
    // SSBO writes (`SHADER_WRITE`) before the bindless main pass's
    // `cmd_draw_indexed_indirect` reads (`INDIRECT_COMMAND_READ`).
    // A no-op when the cull pipeline isn't built (geometry-less
    // worlds or a world that opted out of bindless cull).
    //
    // The caller (`record_frame`) must rebuild this frame's
    // `object_buffer` + `draw_args_buffer` host-side before this runs;
    // those are mapped-memory writes that don't belong in the GPU
    // command stream.
    pub(in crate::vulkan) fn encode_cull(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frustum: &Frustum,
        cam_pos: [f32; 3],
    ) {
        let (Some(pipeline), Some(layout)) =
            (self.cull.cull_pipeline, self.cull.cull_pipeline_layout)
        else {
            return;
        };
        let device = &self.device;

        // Pack the six already-normalised frustum planes for the kernel.
        let mut params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos,
            object_count: self.cull_count() as u32,
        };
        for (i, p) in frustum.planes.iter().enumerate() {
            params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }
        // SAFETY: `CullParams` is `repr(C)` and `CULL_PUSH_CONSTANT_BYTES` wide.
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const CullParams as *const u8,
                std::mem::size_of::<CullParams>(),
            )
        };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&self.cull.cull_sets[frame_idx]),
                &[],
            );
            // Hi-Z occlusion set (set 1): the depth pyramid sampler + this
            // frame's CullHizParams (previous-frame VP + pyramid dims +
            // validity gate). `Some` whenever the cull pipeline is (same
            // gating). `hiz_enabled` is 0 until a pyramid at the current
            // resolution exists, so the kernel falls back to frustum + distance.
            if let Some(hiz) = self.cull.hiz.as_ref() {
                let params = CullHizParams {
                    prev_view_proj: self.cull.hiz_prev_view_proj,
                    hiz_size: [hiz.width as f32, hiz.height as f32],
                    hiz_mip_count: hiz.mip_count,
                    hiz_enabled: u32::from(self.cull.hiz_valid),
                };
                std::ptr::copy_nonoverlapping(
                    &params as *const CullHizParams as *const u8,
                    hiz.cull_ubo_ptrs[frame_idx],
                    std::mem::size_of::<CullHizParams>(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    1,
                    std::slice::from_ref(&hiz.read_sets[frame_idx]),
                    &[],
                );
            }
            device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            // One invocation per build-time object, 64-wide local groups.
            device.cmd_dispatch(cmd, (self.cull_count() as u32).div_ceil(64), 1, 1);
            // Order the kernel's indirect-buffer writes before the main pass'
            // `cmd_draw_indexed_indirect` reads them.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Per-cascade GPU cull for the GPU-driven shadow pass. One dispatch per
    // re-rendered cascade frustum + distance tests every record (static +
    // instances + skinned) against that cascade's light frustum (extracted from
    // `light_vps[c]`; no Hi-Z) and writes the surviving `DrawIndexedIndirectCommand`s
    // into that cascade's indirect buffer via the per-(frame, cascade) shadow cull
    // set. Ends with one memory barrier ordering the kernel's writes before the
    // shadow pass's `cmd_draw_indexed_indirect` reads. Must run outside any render
    // pass, so the caller dispatches it at the top of `encode_shadow_pass` before
    // the per-cascade render passes begin. A no-op when the GPU-driven shadow
    // resources are absent or `cull_count() == 0`. Mirrors
    // `directx/cull.rs::encode_shadow_culls`.
    pub(in crate::vulkan) fn encode_shadow_culls(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        render_mask: u32,
        cam_pos: [f32; 3],
    ) {
        let (Some(pipeline), Some(layout)) = (
            self.cull.shadow_cull_pipeline,
            self.cull.shadow_cull_pipeline_layout,
        ) else {
            return;
        };
        let Some(sets) = self.cull.shadow_cull_sets.get(frame_idx) else {
            return;
        };
        if self.cull_count() == 0 {
            return;
        }
        let device = &self.device;
        let object_count = self.cull_count() as u32;

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            // `sets` has one entry per cascade (NUM_SHADOW_CASCADES), allocated in
            // `init`; iterate it so cascade `c` uses its own output set + frustum.
            for (c, &set) in sets.iter().enumerate() {
                if render_mask & (1u32 << c) == 0 {
                    continue;
                }
                let frustum = Frustum::from_view_projection(self.shadow.uniforms.light_vps[c]);
                let mut params = CullParams {
                    planes: [[0.0; 4]; 6],
                    cam_pos,
                    object_count,
                };
                for (i, p) in frustum.planes.iter().enumerate() {
                    params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
                }
                let push_bytes = std::slice::from_raw_parts(
                    &params as *const CullParams as *const u8,
                    std::mem::size_of::<CullParams>(),
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout,
                    0,
                    std::slice::from_ref(&set),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
                device.cmd_dispatch(cmd, object_count.div_ceil(64), 1, 1);
            }
            // Order every cascade's indirect-buffer writes before the shadow
            // pass's `cmd_draw_indexed_indirect` reads.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }

    // Dispatch the phase-2 (two-pass occlusion) cull for frame slot
    // `frame_idx`. Runs after `HizBuild` has rebuilt the Hi-Z pyramid from this
    // frame's phase-1 depth; re-tests only the objects phase-1 cull marked
    // `STATUS_HIZ_CANDIDATE` against the fresh pyramid (projected through this
    // frame's un-jittered VP) and writes a draw for any that turn out visible
    // into the phase-2 indirect buffer `Main2` consumes. A no-op unless the
    // phase-2 pipeline + sets are built (two-pass occlusion active). Mirrors
    // `directx/cull.rs::encode_cull_phase2`.
    pub(in crate::vulkan) fn encode_cull_phase2(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        frustum: &Frustum,
        cam_pos: [f32; 3],
        cur_vp: [[f32; 4]; 4],
    ) {
        let (Some(pipeline), Some(layout), Some(hiz)) = (
            self.cull.cull_pipeline_phase2,
            self.cull.cull_pipeline_layout,
            self.cull.hiz.as_ref(),
        ) else {
            return;
        };
        if self.cull.cull_sets2.is_empty() || self.cull_count() == 0 {
            return;
        }
        let device = &self.device;

        // Frustum planes + camera position are unused by the phase-2 kernel
        // (candidates already passed those in phase 1) but the push-constant
        // block layout is shared with phase 1, so pack them anyway.
        let mut params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos,
            object_count: self.cull_count() as u32,
        };
        for (i, p) in frustum.planes.iter().enumerate() {
            params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }
        // SAFETY: `CullParams` is `repr(C)` and `CULL_PUSH_CONSTANT_BYTES` wide.
        let push_bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const CullParams as *const u8,
                std::mem::size_of::<CullParams>(),
            )
        };

        // Project AABBs through this frame's un-jittered VP against the pyramid
        // `HizBuild` just rebuilt from this frame's depth. `hiz_enabled = 1`:
        // HizBuild always precedes this dispatch, so a valid pyramid is
        // guaranteed (the kernel still guards defensively).
        let hiz_params = CullHizParams {
            prev_view_proj: cur_vp,
            hiz_size: [hiz.width as f32, hiz.height as f32],
            hiz_mip_count: hiz.mip_count,
            hiz_enabled: 1,
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                &hiz_params as *const CullHizParams as *const u8,
                hiz.cull_ubo2_ptrs[frame_idx],
                std::mem::size_of::<CullHizParams>(),
            );

            // Order phase-1's `cull_status` writes (an earlier compute dispatch
            // on this queue) before this kernel's reads. `cull_status` has no
            // layout/state to transition, so this memory barrier is the only
            // thing flushing the phase-1 writes to the phase-2 reads.
            let status_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&status_barrier),
                &[],
                &[],
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                std::slice::from_ref(&self.cull.cull_sets2[frame_idx]),
                &[],
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                1,
                std::slice::from_ref(&hiz.read_sets2[frame_idx]),
                &[],
            );
            device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
            device.cmd_dispatch(cmd, (self.cull_count() as u32).div_ceil(64), 1, 1);

            // Order the kernel's phase-2 indirect-buffer writes before `Main2`'s
            // `cmd_draw_indexed_indirect` reads them.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CullParams must match the `CullParams` push-constant block in cull.comp
    // (std430): six frustum planes, then cam_pos sharing its 16-byte slot with
    // object_count (112 B total). Pinned by CULL_PUSH_CONSTANT_BYTES.
    #[test]
    fn cull_params_layout_matches_glsl() {
        assert_eq!(std::mem::size_of::<CullParams>(), 112);
        assert_eq!(
            std::mem::size_of::<CullParams>() as u32,
            super::super::pipeline::CULL_PUSH_CONSTANT_BYTES
        );
        assert_eq!(std::mem::offset_of!(CullParams, planes), 0);
        assert_eq!(std::mem::offset_of!(CullParams, cam_pos), 96);
        assert_eq!(std::mem::offset_of!(CullParams, object_count), 108);
    }
}
