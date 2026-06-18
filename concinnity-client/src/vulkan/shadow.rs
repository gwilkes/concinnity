// src/vulkan/shadow.rs
//
// Shadow pass for the Vulkan backend: one depth-only render pass per
// cascade slice of the shadow-map array. Both of shadow_map's transitions are
// graph-driven: shadow_map is the render graph's `shadow_map` resource, so the
// executor emits (over every cascade layer) the Shadow producer barrier
// (`SHADER_READ_ONLY_OPTIMAL` -> `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`, the
// cross-frame reset for this frame's shadow loop) before this pass and the Main
// consumer barrier (`DEPTH_STENCIL_ATTACHMENT_OPTIMAL` -> `SHADER_READ_ONLY_OPTIMAL`,
// letting the main pass sample the cascades) before the Main pass. The map rests
// sampled between frames, so there is no inline reset.
//
// When the bindless GPU-cull path is active (`shadow_bindless_pipeline` built +
// build-time geometry present) the pass is GPU-driven: a per-cascade cull
// dispatch writes one indirect buffer per cascade and each cascade is issued
// with one `cmd_draw_indexed_indirect` (static + instance prefix) + one for the
// skinned tail, instead of the CPU per-object loop. Streamed chunks / runtime
// clones (records past `n_objects`) keep the legacy per-object loop; a
// non-bindless world (custom shader) keeps the legacy path entirely.
//
// The shape mirrors `metal/draw/shadow.rs::encode_shadow_pass`; the
// graph executor in [`graph_exec.rs`](graph_exec.rs) dispatches
// `PassId::Shadow` here.

use ash::{Device, vk};

use super::context::VkContext;

// Push constants for the legacy shadow pass (80 bytes): model matrix + cascade
// index. The runtime loops the shadow pass once per cascade and pushes a
// different `cascade_idx` each iteration; the shader uses it to index
// `sg.light_vps[push.cascade_idx]`.
#[derive(Copy, Clone)]
#[repr(C)]
struct ShadowPush {
    model: [[f32; 4]; 4],
    cascade_idx: u32,
    _pad: [u32; 3],
}

impl VkContext {
    // Encode the cascaded-shadow-map render passes for frame slot
    // `frame_idx`: one render pass per cascade slice, drawing every
    // visible static / instanced / skinned caster into the array layer
    // for that cascade. Ends with a single barrier transitioning every
    // cascade slice from depth-attachment to shader-read so the main
    // pass can sample them.
    //
    // A no-op when no shadow pipeline is built (geometry-less worlds
    // or a world that opted out of CSM). The caller must compute +
    // upload `shadow_uniforms` and `upload_joint_matrices` before this
    // runs so the shadow vertex shader sees the current cascade VPs
    // and the skinned caster pass sees the current joint matrices.
    pub(in crate::vulkan) fn encode_shadow_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cam_pos: [f32; 3],
        elapsed: f32,
    ) {
        let (Some(shadow_pipeline), Some(shadow_pl)) =
            (self.shadow.pipeline, self.shadow.pipeline_layout)
        else {
            return;
        };

        // Raymarched SDF shadow casters share these cascade DSVs: upload this
        // frame's animation time once (no-op without casters) so the from-light
        // SDF march lines up with the lit-side surface.
        self.upload_raymarch_shadow_view(frame_idx, elapsed);
        let device = self.device.clone();
        let device = &device;

        let sm = self.shadow.map_size;
        let shadow_extent = vk::Extent2D {
            width: sm,
            height: sm,
        };

        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        // Cascades to re-render this frame; draw_frame computed the mask from the
        // update policy. A skipped cascade's render pass is omitted entirely, so
        // its slice keeps the depth + VP from when it was last rendered (the
        // graph-driven producer/consumer barriers still round-trip every layer,
        // preserving the contents). The 0 sentinel falls back to all cascades.
        let all_cascades = (1u32 << crate::gfx::render_types::NUM_SHADOW_CASCADES) - 1;
        let render_mask = if self.shadow.render_mask == 0 {
            all_cascades
        } else {
            self.shadow.render_mask
        };

        let gpu_driven = self.cull.shadow_bindless_pipeline.is_some() && self.cull_count() > 0;

        // GPU-driven cull prologue: dispatch every re-rendered cascade's cull
        // before opening any render pass (Vulkan disallows compute inside a
        // render pass). Each writes that cascade's indirect buffer.
        if gpu_driven {
            self.encode_shadow_culls(cmd, frame_idx, render_mask, cam_pos);
        }

        for (cascade_idx, &shadow_fb) in self.shadow.framebuffers.iter().enumerate() {
            if render_mask & (1u32 << cascade_idx) == 0 {
                continue;
            }
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.shadow.render_pass)
                .framebuffer(shadow_fb)
                .render_area(vk::Rect2D::default().extent(shadow_extent))
                .clear_values(std::slice::from_ref(&clear_depth));

            unsafe {
                device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

                // Negative-height viewport: Y-flips NDC so Y-up matches Metal.
                let vp = vk::Viewport {
                    x: 0.0,
                    y: sm as f32,
                    width: sm as f32,
                    height: -(sm as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
                let scissor = vk::Rect2D::default().extent(shadow_extent);
                device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            }

            if gpu_driven {
                self.encode_shadow_cascade_indirect(device, cmd, frame_idx, cascade_idx, cam_pos);
            } else {
                self.encode_shadow_cascade_legacy(
                    device,
                    cmd,
                    frame_idx,
                    cascade_idx,
                    cam_pos,
                    shadow_pipeline,
                    shadow_pl,
                );
            }

            // Raymarched SDF shadow casters into this cascade's DSV, after the
            // rasterised casters and within the same render pass (no re-clear);
            // the LESS depth test keeps the nearer occluder.
            unsafe {
                self.encode_sdf_shadow_cascade(cmd, frame_idx, cascade_idx);
                device.cmd_end_render_pass(cmd);
            }
        }

        // shadow_map's transitions are fully graph-driven (over every cascade
        // layer): the Shadow producer barrier (SHADER_READ_ONLY ->
        // DEPTH_STENCIL_ATTACHMENT, the cross-frame reset) runs before this pass
        // and the Main consumer barrier (DEPTH_STENCIL_ATTACHMENT ->
        // SHADER_READ_ONLY) before the Main pass. Neither is emitted here, and
        // the map rests sampled between frames (no inline reset).
    }

    // GPU-driven cascade body (inside the cascade's render pass): the depth-only
    // bindless pipeline issues this cascade's static + instance prefix and the
    // skinned tail with two `cmd_draw_indexed_indirect` calls over the cascade's
    // cull-written indirect buffer, then the legacy chunk/clone casters.
    fn encode_shadow_cascade_indirect(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cascade_idx: usize,
        cam_pos: [f32; 3],
    ) {
        let (Some(sb_pipeline), Some(sb_layout)) = (
            self.cull.shadow_bindless_pipeline,
            self.cull.shadow_bindless_pipeline_layout,
        ) else {
            return;
        };
        let Some(indirect) = self
            .cull
            .shadow_indirect_buffers
            .get(frame_idx)
            .and_then(|c| c.get(cascade_idx).copied())
        else {
            return;
        };
        let stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32;
        let prefix = self.skinned_record_base() as u32;
        let cascade = cascade_idx as u32;

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sb_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                sb_layout,
                0,
                &[
                    self.shadow.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                ],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                sb_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                &cascade.to_ne_bytes(),
            );

            // Static + instance prefix against the static VB/IB.
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
            if prefix > 0 {
                device.cmd_draw_indexed_indirect(cmd, indirect, 0, prefix, stride);
                self.inc_draw_calls(1);
            }

            // Skinned tail against the deformed VB + skinned u16 IB.
            if self.n_skinned > 0
                && let Some(deformed) = self.skinned.deformed.get(frame_idx)
            {
                device.cmd_bind_vertex_buffers(
                    cmd,
                    0,
                    std::slice::from_ref(&deformed.buffer),
                    &[0],
                );
                device.cmd_bind_index_buffer(
                    cmd,
                    self.skinned.index_buffer,
                    0,
                    vk::IndexType::UINT16,
                );
                device.cmd_draw_indexed_indirect(
                    cmd,
                    indirect,
                    (self.skinned_record_base() * stride as usize) as u64,
                    self.n_skinned as u32,
                    stride,
                );
                self.inc_draw_calls(1);
            }
        }

        // Legacy depth-only casters for draws past the bindless record range
        // (streamed chunks + runtime clones, not in the GpuObjectData buffer).
        self.encode_shadow_legacy_extra(device, cmd, frame_idx, cascade_idx, cam_pos);
    }

    // Legacy per-object casters for runtime clones past the bindless record range
    // (`i >= n_objects` AND in `clone_slot_by_draw_idx`). Streamed VoxelWorld chunks
    // now fold into the GPU-driven cull records (drawn by the per-cascade indirect
    // draw), so they are skipped here. Mirrors the legacy static loop, appending
    // into this cascade's depth (no re-clear). A no-op for worlds with no clones.
    fn encode_shadow_legacy_extra(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cascade_idx: usize,
        cam_pos: [f32; 3],
    ) {
        if self.clone_slot_by_draw_idx.is_empty() {
            return;
        }
        let (Some(shadow_pipeline), Some(shadow_pl)) =
            (self.shadow.pipeline, self.shadow.pipeline_layout)
        else {
            return;
        };
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pl,
                0,
                std::slice::from_ref(&self.shadow.global_sets[frame_idx]),
                &[],
            );
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
            for (i, obj) in self.draw_objects.iter().enumerate() {
                if i < self.n_objects || !obj.visible || !obj.resident {
                    continue;
                }
                if !self.clone_slot_by_draw_idx.contains_key(&i) {
                    continue; // streamed chunk -> folded into the cull records
                }
                let d = crate::gfx::lod::camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                let push = ShadowPush {
                    model: obj.model,
                    cascade_idx: cascade_idx as u32,
                    _pad: [0; 3],
                };
                device.cmd_push_constants(
                    cmd,
                    shadow_pl,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const ShadowPush as *const u8,
                        std::mem::size_of::<ShadowPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
                self.inc_draw_calls(1);
            }
        }
    }

    // Legacy CPU-driven cascade body (inside the cascade's render pass): per-object
    // `cmd_draw_indexed` for static + instanced (iterated per instance) + skinned
    // casters. Used for non-bindless worlds (custom shader) or worlds with no
    // build-time geometry.
    #[allow(clippy::too_many_arguments)]
    fn encode_shadow_cascade_legacy(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        cascade_idx: usize,
        cam_pos: [f32; 3],
        shadow_pipeline: vk::Pipeline,
        shadow_pl: vk::PipelineLayout,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, shadow_pipeline);

            // Global shadow descriptor: ShadowUniforms UBO.
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                shadow_pl,
                0,
                std::slice::from_ref(&self.shadow.global_sets[frame_idx]),
                &[],
            );

            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);

            for obj in &self.draw_objects {
                // A non-resident streamed mesh has no geometry in the
                // shared buffers yet -- skip it everywhere.
                if !obj.visible || !obj.resident {
                    continue;
                }
                // Pick the LOD by camera distance: the shadow pass uses
                // the same slice the main pass will, so silhouettes track
                // when the runtime swaps to a coarser LOD.
                let d = crate::gfx::lod::camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                let push = ShadowPush {
                    model: obj.model,
                    cascade_idx: cascade_idx as u32,
                    _pad: [0; 3],
                };
                device.cmd_push_constants(
                    cmd,
                    shadow_pl,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    std::slice::from_raw_parts(
                        &push as *const ShadowPush as *const u8,
                        std::mem::size_of::<ShadowPush>(),
                    ),
                );
                device.cmd_draw_indexed(
                    cmd,
                    index_count as u32,
                    1,
                    index_offset as u32,
                    obj.base_vertex,
                    0,
                );
                self.inc_draw_calls(1);
            }

            // Instanced clusters in the shadow pass: iterate instances
            // individually using the regular shadow pipeline. Cheap to
            // ship; visually identical to an instanced shadow shader. Walk
            // the same per-LOD buckets the Main pass uses (computed by
            // `prepare_instanced_clusters`) so shadow silhouettes track the
            // per-instance LOD the camera picked.
            for cluster_idx in 0..self.instanced.clusters.len() {
                let Some(buckets) = self.instanced.lod_buckets.get(cluster_idx) else {
                    continue;
                };
                for bucket in buckets {
                    for &model in &bucket.instances {
                        let push = ShadowPush {
                            model,
                            cascade_idx: cascade_idx as u32,
                            _pad: [0; 3],
                        };
                        device.cmd_push_constants(
                            cmd,
                            shadow_pl,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            std::slice::from_raw_parts(
                                &push as *const ShadowPush as *const u8,
                                std::mem::size_of::<ShadowPush>(),
                            ),
                        );
                        device.cmd_draw_indexed(
                            cmd,
                            bucket.index_count as u32,
                            1,
                            bucket.index_offset as u32,
                            0,
                            0,
                        );
                        self.inc_draw_calls(1);
                    }
                }
            }

            // Skinned meshes: deformed depth, drawn after the static
            // and instanced casters within the same cascade render
            // pass (no re-clear, so skinned depth appends).
            if let (Some(sk_pipeline), Some(sk_pl)) = (
                self.shadow.skinned_pipeline,
                self.shadow.skinned_pipeline_layout,
            ) && !self.skinned.draw_objects.is_empty()
            {
                let (sk_vbuf, sk_ibuf) = self.skinned_geometry();
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sk_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sk_pl,
                    0,
                    std::slice::from_ref(&self.shadow.global_sets[frame_idx]),
                    &[],
                );
                device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&sk_vbuf), &[0]);
                device.cmd_bind_index_buffer(cmd, sk_ibuf, 0, vk::IndexType::UINT16);
                for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
                    if !obj.visible {
                        continue;
                    }
                    // Match the Main pass's per-object LOD pick so shadow
                    // silhouettes track the active skinned LOD.
                    let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                    let (index_offset, index_count) = obj.active_lod(d);
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_pl,
                        1,
                        std::slice::from_ref(&self.skinned.joint_sets[frame_idx][i]),
                        &[],
                    );
                    let push = ShadowPush {
                        model: obj.model,
                        cascade_idx: cascade_idx as u32,
                        _pad: [0; 3],
                    };
                    device.cmd_push_constants(
                        cmd,
                        sk_pl,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const ShadowPush as *const u8,
                            std::mem::size_of::<ShadowPush>(),
                        ),
                    );
                    device.cmd_draw_indexed(cmd, index_count as u32, 1, index_offset as u32, 0, 0);
                    self.inc_draw_calls(1);
                }
            }
        }
    }
}
