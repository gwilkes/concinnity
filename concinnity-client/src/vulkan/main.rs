// src/vulkan/main.rs
//
// Main scene pass for the Vulkan backend: linear-light HDR off-screen
// render that draws every visible static / instanced / skinned object
// into the multisampled HDR colour + depth attachments (resolved into
// `hdr_resolve` for the post stack). One Vulkan render pass with three
// sub-passes-by-pipeline:
//
//   1. Bindless build-time statics via `cmd_draw_indexed_indirect` over
//      the GPU-cull-written indirect buffer.
//   2. Legacy per-draw fallback (custom shaders + streamed `VoxelWorld`
//      chunks past `n_objects`).
//   3. Instanced clusters via per-instance storage buffers.
//   4. Skinned meshes via LBS vertex shader + per-object joint sets.
//
// The shape mirrors `metal/draw/main.rs::encode_main_pass`; the graph
// executor in [`graph_exec.rs`](graph_exec.rs) dispatches
// `PassId::Main` here. The Shadow → Main `shadow_map` read edge in
// the frame graph pins Shadow before Main via toposort; the encoder
// itself only deals with the HDR pass.

use ash::vk;

use crate::gfx::frustum::Frustum;

use super::context::VkContext;

// Push constants for the main pass (112 bytes, std430).
// Matches ModelUniforms(64) + MaterialUniforms(48) packed together.
#[derive(Copy, Clone)]
#[repr(C)]
struct MainPush {
    model: [[f32; 4]; 4],
    roughness: f32,
    metallic: f32,
    _mpad0: f32,
    _mpad1: f32,
    tint: [f32; 3],
    _mpad2: f32,
    emissive: [f32; 3],
    _mpad3: f32,
}

impl VkContext {
    // Recompute every instanced cluster's per-LOD-bucket partition for the
    // current camera and memcpy the bucket-ordered instance matrices into this
    // frame's mapped per-cluster SSBOs, storing the partition in
    // `instanced.lod_buckets` for every instanced draw site to read.
    //
    // Run on `&mut self` from `execute_graph` BEFORE the render-graph fan-out,
    // mirroring `prepare_particle_pass`: the SSAO / SSR / velocity pre-passes
    // run earlier in the graph than Main but share the same per-cluster upload
    // buffer, so the upload has to happen up front (every pre-pass + Main then
    // see this frame's bucket order). With LOD bucketing the buffer order
    // depends on `cam_pos`, so it cannot be deferred into the Main pass the way
    // the old single-LOD upload was. Mirrors `DxContext::build_instance_upload`.
    pub(in crate::vulkan) fn prepare_instanced_clusters(
        &mut self,
        frame_idx: usize,
        cam_pos: [f32; 3],
    ) {
        if self.instanced.clusters.is_empty() || self.instanced.ptrs.is_empty() {
            return;
        }
        // Re-shape on a runtime cluster-count change (asset hot-reload), then
        // clear each row in place to reuse its heap allocation.
        if self.instanced.lod_buckets.len() != self.instanced.clusters.len() {
            self.instanced
                .lod_buckets
                .resize(self.instanced.clusters.len(), Vec::new());
        }
        const STRIDE: usize = std::mem::size_of::<[[f32; 4]; 4]>();
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            let row = &mut self.instanced.lod_buckets[cluster_idx];
            row.clear();
            if cluster.instances.is_empty() {
                continue;
            }
            let upload_ptr = self.instanced.ptrs[frame_idx][cluster_idx];
            let buckets = cluster.lod_buckets(cam_pos);
            row.reserve(buckets.len());
            let mut prefix_instances: usize = 0;
            for bucket in buckets {
                let count = bucket.instances.len();
                // SAFETY: the per-cluster SSBO was sized at init for every
                // instance the cluster declared; the bucket lengths sum to that
                // count, so the bucket-ordered write stays in bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bucket.instances.as_ptr() as *const u8,
                        upload_ptr.add(prefix_instances * STRIDE),
                        count * STRIDE,
                    );
                }
                prefix_instances += count;
                row.push(bucket);
            }
        }
    }

    // Encode the main HDR scene pass for frame slot `frame_idx`. Draws
    // every visible static / instanced / skinned object in the
    // `visible` set into the multisampled colour + depth attachments
    // of `framebuffers[frame_idx]`; the render pass resolves into
    // `hdr_resolve` (the post-stack input) on `cmd_end_render_pass`.
    //
    // Bindless draws come from the GPU-culled indirect buffer the
    // cull compute kernel wrote earlier this frame; legacy /
    // instanced / skinned draws iterate the CPU-culled `visible`
    // list. Cluster culling repeats the frustum / distance check
    // here because instanced clusters aren't in the BVH.
    pub(in crate::vulkan) fn encode_main_pass(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
        visible: &[u32],
        frustum: &Frustum,
        cam_pos: [f32; 3],
        world_hidden: bool,
    ) {
        let device = self.device.clone();
        let device = &device;
        let extent = self.render_extent;

        // Opaque menu backdrop, MSAA path: skip the main render pass entirely.
        // Beginning it would clear the MSAA colour+depth and, on
        // `end_render_pass`, resolve the (undrawn) MSAA colour into hdr_resolve:
        // a full-render-resolution resolve of a frame nothing presents (the
        // composite samples the post-stack scene + the opaque overlay on top).
        // On an immediate-mode GPU that resolve is the bulk of the paused
        // frame's GPU cost. Skipping it drops the main pass to a lone layout
        // barrier (~0us, so it falls off the passes HUD like Metal / DirectX),
        // and the barrier leaves hdr_resolve sampled-ready for the plain-world
        // composite path (no TAA / reflections sample it directly). Its
        // contents are irrelevant under the opaque overlay, matching the
        // DirectX paused path (which likewise skips the resolve). The
        // single-sample path below has no resolve to skip (hdr_resolve is the
        // colour attachment), so it keeps its cheap clear-only render pass.
        if world_hidden && self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            super::texture::transition_image_layout(
                device,
                cmd,
                self.hdr_resolve_images[frame_idx].image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
            );
            return;
        }

        // Clears for the main HDR attachments. With MSAA on, the
        // resolve attachment doesn't need a clear (resolve overwrites);
        // a `ClearValue::default()` placeholder keeps the slice index
        // aligned with `main_render_pass`'s attachment count.
        let [r, g, b, a] = self.clear_color;
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [r, g, b, a],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clears: &[vk::ClearValue] = if self.msaa_samples != vk::SampleCountFlags::TYPE_1 {
            &[clear_color, clear_depth, vk::ClearValue::default()]
        } else {
            &[clear_color, clear_depth]
        };

        // Under two-pass occlusion, phase 1 renders into a variant render pass
        // that STORE's the MSAA colour (so `Main2` can load + composite onto
        // it) and leaves the colour in COLOR_ATTACHMENT_OPTIMAL. The clears are
        // identical (phase 1 still CLEAR's), so only the render pass differs.
        // The raymarch pass needs the same STORE-colour treatment (it loads +
        // re-resolves the MSAA colour after AutoExposure), so when raymarch is
        // active but two-pass occlusion is not, switch to its store-colour pass.
        let render_pass = if self.two_pass_occlusion_active() {
            self.cull
                .main_render_pass_phase1
                .unwrap_or(self.main_render_pass)
        } else if let Some(rp) = self.raymarch.as_ref().and_then(|r| r.main_store_color_pass) {
            rp
        } else {
            self.main_render_pass
        };
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(self.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent))
            .clear_values(clears);

        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };

        // Opaque menu backdrop, single-sample path: the render pass above
        // already cleared hdr_resolve (the colour attachment) and there is no
        // resolve to skip, so just end immediately; nothing of the world draws
        // behind the menu. (The MSAA path returned earlier without beginning a
        // render pass at all.)
        if world_hidden {
            unsafe { device.cmd_end_render_pass(cmd) };
            return;
        }

        // Viewport: negative height flips Y to match Metal coordinate system.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        unsafe {
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }

        // Geometry buffers, pipeline-layout-independent, so bound once for
        // both the bindless and legacy main sub-passes below.
        unsafe {
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);
        }

        // Build-time static objects render through the bindless pipeline
        // driven by the GPU-culled indirect command buffer the cull
        // compute pass wrote above. One `cmd_draw_indexed_indirect`
        // issues every build-time object's draw; culled / disabled objects
        // were written with `instance_count = 0` (a no-op). Each draw is
        // stateless apart from the object id, which rides `first_instance`
        // (Vulkan's `gl_InstanceIndex` includes it); model/material/textures
        // are fetched from the per-frame GpuObjectData SSBO + the bindless
        // texture pool. Streamed VoxelWorld chunks keep the legacy per-draw
        // pipeline below.
        let use_bindless = self.cull.bindless_pipeline.is_some() && self.cull_count() > 0;
        if use_bindless {
            let pipeline = self.cull.bindless_pipeline.unwrap();
            let layout = self.cull.bindless_pipeline_layout.unwrap();
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout,
                    0,
                    &[
                        self.descriptors.global_sets[frame_idx],
                        self.cull.bindless_sets[frame_idx],
                    ],
                    &[],
                );
                // Indirect draw #1: the static + instance prefix
                // `[0, skinned_record_base())` against the static VB/IB bound above.
                // The skinned tail is drawn by a second indirect draw below (deformed
                // VB + skinned IB), reading the same indirect buffer from
                // `skinned_record_base()` on.
                device.cmd_draw_indexed_indirect(
                    cmd,
                    self.cull.indirect_buffers[frame_idx],
                    0,
                    self.skinned_record_base() as u32,
                    std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            }
            // GPU expands the indirect buffer to up to `n_objects` draw
            // commands inside, but the call count surfaced to the profiler
            // is the host-side draw. Mirrors DirectX / Metal.
            self.inc_draw_calls(1);
        }

        // Legacy per-draw main pass. Draws every visible object when the
        // bindless pass is inactive (custom shader / no build-time geometry);
        // otherwise only runtime clones (streamed VoxelWorld chunks now fold into
        // the bindless indirect draw as their own records). Shared with the
        // instanced + skinned passes' fragment shader.
        let legacy_needed = !use_bindless || !self.clone_slot_by_draw_idx.is_empty();
        if legacy_needed {
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.main_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.main_pipeline_layout,
                    0,
                    std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                    &[],
                );
            }
            for &draw_idx in visible {
                let i = draw_idx as usize;
                if use_bindless && i < self.n_objects {
                    continue; // build-time object, already drawn bindless
                }
                if use_bindless
                    && i >= self.n_objects
                    && !self.clone_slot_by_draw_idx.contains_key(&i)
                {
                    continue; // streamed chunk, already drawn bindless (folded record)
                }
                let obj = &self.draw_objects[i];
                if !obj.visible || !obj.resident {
                    continue;
                }

                // Per-object descriptor set (albedo + normal map). Streamed
                // `VoxelWorld` chunks are appended past the build-time object
                // count and share one descriptor set bound to the world's
                // chunk material; build-time objects use their own baked
                // (albedo, normal) set; runtime clones (from
                // `clone_static_draw_object`) carry their own per-clone set
                // stored in `clone_object_sets` and looked up via
                // `clone_slot_by_draw_idx`.
                let obj_set = if i >= self.n_objects {
                    if let Some(&offset) = self.clone_slot_by_draw_idx.get(&i) {
                        match self.clone_object_sets.get(offset) {
                            Some(&s) => s,
                            None => continue,
                        }
                    } else {
                        match self.chunk_stream.object_set {
                            Some(s) => s,
                            None => continue,
                        }
                    }
                } else {
                    self.descriptors.object_sets
                        [i.min(self.descriptors.object_sets.len().saturating_sub(1))]
                };

                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.main_pipeline_layout,
                        1,
                        std::slice::from_ref(&obj_set),
                        &[],
                    );
                }

                // Per-frame active LOD pick. Streamed VoxelWorld chunks
                // (past `n_objects`) never declare LOD alternates, so the
                // pick collapses to LOD0 for them.
                let d = crate::gfx::lod::camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                let push = MainPush {
                    model: obj.model,
                    roughness: obj.material.roughness,
                    metallic: obj.material.metallic,
                    _mpad0: 0.0,
                    _mpad1: 0.0,
                    tint: obj.material.tint,
                    _mpad2: 0.0,
                    emissive: obj.material.emissive,
                    _mpad3: 0.0,
                };
                unsafe {
                    device.cmd_push_constants(
                        cmd,
                        self.main_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const MainPush as *const u8,
                            std::mem::size_of::<MainPush>(),
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
                }
                self.inc_draw_calls(1);
            }
        }

        // Instanced clusters main pass. Skipped when the bindless merge is active:
        // each instance is then a `GpuObjectData` record at `n_objects + k` in the
        // cull buffers, drawn by the bindless `cmd_draw_indexed_indirect` above.
        if let (Some(inst_pipeline), Some(inst_pipeline_layout)) =
            (self.instanced.pipeline, self.instanced.pipeline_layout)
            && !self.instanced.clusters.is_empty()
            && !use_bindless
        {
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, inst_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    inst_pipeline_layout,
                    0,
                    std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                    &[],
                );
            }

            for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
                if cluster.instances.is_empty() {
                    continue;
                }
                if cluster.cullable() {
                    if !frustum.intersects_aabb(cluster.cluster_bb_min, cluster.cluster_bb_max) {
                        continue;
                    }
                    if cluster.cull_distance > 0.0 {
                        let d2 = crate::gfx::frustum::aabb_distance_sq(
                            cam_pos,
                            cluster.cluster_bb_min,
                            cluster.cluster_bb_max,
                        );
                        if d2 > cluster.cull_distance * cluster.cull_distance {
                            continue;
                        }
                    }
                }

                // Per-cluster (albedo, normal) sampler + this frame's
                // bucket-ordered instance SSBO. The matrices were uploaded by
                // `prepare_instanced_clusters`; issue one draw per LOD bucket,
                // offsetting into the SSBO via `first_instance` (the instanced
                // VS reads `instances[gl_InstanceIndex]`).
                let Some(buckets) = self.instanced.lod_buckets.get(cluster_idx) else {
                    continue;
                };
                unsafe {
                    // Set 1: per-cluster (albedo, normal) sampler.
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        inst_pipeline_layout,
                        1,
                        std::slice::from_ref(&self.instanced.object_sets[cluster_idx]),
                        &[],
                    );
                    // Set 2: per-instance storage buffer for this frame.
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        inst_pipeline_layout,
                        2,
                        std::slice::from_ref(&self.instanced.sets[frame_idx][cluster_idx]),
                        &[],
                    );

                    let push = MainPush {
                        model: [[0.0; 4]; 4], // ignored by instanced VS
                        roughness: cluster.material.roughness,
                        metallic: cluster.material.metallic,
                        _mpad0: 0.0,
                        _mpad1: 0.0,
                        tint: cluster.material.tint,
                        _mpad2: 0.0,
                        emissive: cluster.material.emissive,
                        _mpad3: 0.0,
                    };
                    device.cmd_push_constants(
                        cmd,
                        inst_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const MainPush as *const u8,
                            std::mem::size_of::<MainPush>(),
                        ),
                    );
                    let mut first_instance: u32 = 0;
                    for bucket in buckets {
                        let count = bucket.instances.len() as u32;
                        device.cmd_draw_indexed(
                            cmd,
                            bucket.index_count as u32,
                            count,
                            bucket.index_offset as u32,
                            0,
                            first_instance,
                        );
                        first_instance += count;
                    }
                }
                self.inc_draw_calls(buckets.len() as u32);
            }
        }

        // Skinned meshes main pass. When the GPU-driven bindless fold is active,
        // skinned objects ride the same cull buffers as static + instances and are
        // drawn (as rigid deformed geometry) by a 2nd `cmd_draw_indexed_indirect`
        // over this frame's deformed-vertex buffer + the skinned u16 index buffer,
        // reading the cull-written indirect buffer from `skinned_record_base()`. The
        // `encode_skin` compute pass (Cull graph arm) has already posed the deformed
        // buffer. Otherwise the legacy per-draw skinned pass runs (custom-shader
        // worlds, or a pure-skinned world with no static geometry to engage bindless).
        if use_bindless && self.n_skinned > 0 {
            if let (Some(bindless_pipeline), Some(bindless_layout), Some(deformed)) = (
                self.cull.bindless_pipeline,
                self.cull.bindless_pipeline_layout,
                self.skinned.deformed.get(frame_idx),
            ) {
                unsafe {
                    device.cmd_bind_pipeline(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        bindless_pipeline,
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        bindless_layout,
                        0,
                        &[
                            self.descriptors.global_sets[frame_idx],
                            self.cull.bindless_sets[frame_idx],
                        ],
                        &[],
                    );
                    // Bind the deformed verts (base_vertex = 0, global skinned
                    // indexing) + the skinned u16 IB.
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
                    // Indirect draw #2: the skinned tail
                    // `[skinned_record_base(), cull_count())`, byte-offset into the
                    // same indirect command buffer.
                    let cmd_stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>();
                    device.cmd_draw_indexed_indirect(
                        cmd,
                        self.cull.indirect_buffers[frame_idx],
                        (self.skinned_record_base() * cmd_stride) as u64,
                        self.n_skinned as u32,
                        cmd_stride as u32,
                    );
                }
                self.inc_draw_calls(1);
            }
        } else if let (Some(sk_pipeline), Some(sk_pl)) =
            (self.skinned.pipeline, self.skinned.pipeline_layout)
            && !self.skinned.draw_objects.is_empty()
        {
            let (sk_vbuf, sk_ibuf) = self.skinned_geometry();
            unsafe {
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sk_pipeline);
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    sk_pl,
                    0,
                    std::slice::from_ref(&self.descriptors.global_sets[frame_idx]),
                    &[],
                );
                device.cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&sk_vbuf), &[0]);
                device.cmd_bind_index_buffer(cmd, sk_ibuf, 0, vk::IndexType::UINT16);
            }
            for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
                if !obj.visible {
                    continue;
                }
                // Pick the LOD by camera distance to the object's placement
                // (skinned meshes deform every frame, so they have no static
                // AABB); the shadow + SSR / SSAO pre-passes pick the same slice.
                let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                let push = MainPush {
                    model: obj.model,
                    roughness: obj.material.roughness,
                    metallic: obj.material.metallic,
                    _mpad0: 0.0,
                    _mpad1: 0.0,
                    tint: obj.material.tint,
                    _mpad2: 0.0,
                    emissive: obj.material.emissive,
                    _mpad3: 0.0,
                };
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_pl,
                        1,
                        std::slice::from_ref(&self.skinned.object_sets[i]),
                        &[],
                    );
                    device.cmd_bind_descriptor_sets(
                        cmd,
                        vk::PipelineBindPoint::GRAPHICS,
                        sk_pl,
                        2,
                        std::slice::from_ref(&self.skinned.joint_sets[frame_idx][i]),
                        &[],
                    );
                    device.cmd_push_constants(
                        cmd,
                        sk_pl,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        std::slice::from_raw_parts(
                            &push as *const MainPush as *const u8,
                            std::mem::size_of::<MainPush>(),
                        ),
                    );
                    device.cmd_draw_indexed(cmd, index_count as u32, 1, index_offset as u32, 0, 0);
                }
                self.inc_draw_calls(1);
            }
        }

        // End the main scene pass. The render pass leaves the HDR resolve
        // image in SHADER_READ_ONLY_OPTIMAL for the composite pass to sample.
        unsafe { device.cmd_end_render_pass(cmd) };
    }

    // Phase-2 main pass for two-pass occlusion. Loads (does not clear) the
    // phase-1 HDR colour + depth, re-runs the bindless indirect draw (static
    // objects + merged instances, both Hi-Z-tested through the cull buffer) over
    // the phase-2 indirect buffer `Cull2` wrote, and resolves the combined scene
    // into `hdr_resolve` for the post stack. Skinned geometry is not Hi-Z-culled,
    // so it was fully drawn in phase 1 and is not repeated here. A no-op unless
    // the bindless path is active with build-time geometry. Mirrors
    // `directx/draw/main.rs::encode_main_pass_phase2`.
    //
    // `bindless_pipeline` was created against `main_render_pass` but is used
    // here with `main_render_pass_phase2`; that is valid because the two passes
    // are render-pass-compatible (identical attachment count / formats / sample
    // counts, only load/store ops + layouts differ). Keep them so if the
    // phase-2 attachment set ever diverges, build a phase-2-specific pipeline.
    pub(in crate::vulkan) fn encode_main_pass_phase2(
        &self,
        cmd: vk::CommandBuffer,
        frame_idx: usize,
    ) {
        let (Some(render_pass), Some(pipeline), Some(layout)) = (
            self.cull.main_render_pass_phase2,
            self.cull.bindless_pipeline,
            self.cull.bindless_pipeline_layout,
        ) else {
            return;
        };
        if self.n_objects == 0 || self.cull.indirect_buffers2.is_empty() {
            return;
        }
        let device = self.device.clone();
        let device = &device;
        let extent = self.render_extent;

        // The phase-2 render pass LOADs the phase-1 colour + depth. Because it
        // shares the main render pass's subpass dependency (so the shared
        // framebuffer + bindless pipeline stay render-pass-compatible), that
        // dependency does not order the phase-1 writes before this LOAD. Emit
        // the ordering explicitly here: phase-1 Main's colour + depth writes
        // (and HizBuild's depth restore) -> this pass's attachment load. Spans
        // the command-buffer boundary via submission order under parallel
        // recording.
        let load_barrier = vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&load_barrier),
                &[],
                &[],
            );
        }

        // LOAD render pass: no clears (loadOp = LOAD / DONT_CARE), so no clear
        // values are required.
        let rp_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(self.framebuffers[frame_idx])
            .render_area(vk::Rect2D::default().extent(extent));
        unsafe { device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE) };

        // Same negative-height viewport flip as the phase-1 main pass so the
        // disoccluded geometry rasterises into identical pixels.
        let vp = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D::default().extent(extent);
        unsafe {
            device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.geometry.vertex_buffer),
                &[0],
            );
            device.cmd_bind_index_buffer(cmd, self.geometry.index_buffer, 0, vk::IndexType::UINT32);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[
                    self.descriptors.global_sets[frame_idx],
                    self.cull.bindless_sets[frame_idx],
                ],
                &[],
            );
            // Indirect draw #1: the static + instance prefix against the static
            // VB/IB bound above.
            device.cmd_draw_indexed_indirect(
                cmd,
                self.cull.indirect_buffers2[frame_idx],
                0,
                self.skinned_record_base() as u32,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
        self.inc_draw_calls(1);

        // Indirect draw #2: the skinned tail against the deformed VB + skinned IB.
        // The pipeline + descriptor sets bound above persist (same bindless
        // pipeline), so only the vertex/index buffers rebind.
        if self.n_skinned > 0
            && let Some(deformed) = self.skinned.deformed.get(frame_idx)
        {
            let cmd_stride = std::mem::size_of::<vk::DrawIndexedIndirectCommand>();
            unsafe {
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
                    self.cull.indirect_buffers2[frame_idx],
                    (self.skinned_record_base() * cmd_stride) as u64,
                    self.n_skinned as u32,
                    cmd_stride as u32,
                );
            }
            self.inc_draw_calls(1);
        }

        // End the phase-2 pass. The render pass resolves the combined phase-1 +
        // phase-2 MSAA colour into `hdr_resolve` and leaves it
        // SHADER_READ_ONLY_OPTIMAL, so the post stack reads the combined scene.
        unsafe { device.cmd_end_render_pass(cmd) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // MainPush must match the `PushBlock` push constant in the main-pass
    // shaders (std430): the model matrix, roughness/metallic with two pads,
    // then tint and emissive vec3s each followed by a pad (112 B total).
    #[test]
    fn main_push_layout_matches_glsl() {
        assert_eq!(std::mem::size_of::<MainPush>(), 112);
        assert_eq!(std::mem::offset_of!(MainPush, model), 0);
        assert_eq!(std::mem::offset_of!(MainPush, roughness), 64);
        assert_eq!(std::mem::offset_of!(MainPush, metallic), 68);
        assert_eq!(std::mem::offset_of!(MainPush, _mpad0), 72);
        assert_eq!(std::mem::offset_of!(MainPush, _mpad1), 76);
        assert_eq!(std::mem::offset_of!(MainPush, tint), 80);
        assert_eq!(std::mem::offset_of!(MainPush, _mpad2), 92);
        assert_eq!(std::mem::offset_of!(MainPush, emissive), 96);
        assert_eq!(std::mem::offset_of!(MainPush, _mpad3), 108);
    }
}
