// src/directx/draw/main.rs
//
// Main HDR scene pass: SSAO pre-pass + GPU-driven bindless static draw +
// legacy per-draw fallback + instanced clusters + skinned meshes. Renders
// linear-light HDR into `hdr_color`; the composite pass tonemaps that down
// onto the swapchain backbuffer. Ends by transitioning (or MSAA-resolving)
// `hdr_color` to `PIXEL_SHADER_RESOURCE` so post-process passes can sample
// it.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::texture::{HDR_FORMAT, transition_barrier};

// Root constants for the main pass (112 bytes = 28 DWORDs).
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

impl DxContext {
    // Rebuild this frame's `StructuredBuffer<GpuObjectData>` for the bindless
    // static pass: one 144-byte record per build-time `DrawObject`, indexed by
    // object id. Streamed `VoxelWorld` chunks (past `n_objects`) are skipped;
    // they render through the legacy pipeline. Rebuilt every frame so
    // `update_model` / `update_visibility` edits are reflected; a no-op when
    // the bindless pass is inactive.
    pub(super) fn build_object_buffer(&self, frame_idx: usize) {
        use crate::gfx::render_types::{GpuObjectData, pack_object_record, pack_skinned_record};
        let Some(&ptr) = self.cull.object_buffer_ptrs.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<GpuObjectData>();
        // Flat deduplicated bindless pool indices, identical to Vulkan/Metal:
        // albedo = texture_slot, normal = albedo_count + normal_map_slot, both
        // clamped to the pool. The bindless main pass + RT hit shader bind the
        // flat pool base, so a shared texture resolves to one descriptor.
        let albedo_count = self.descriptors.textures.len();
        let last_tex = albedo_count.saturating_sub(1);
        let last_nm = self.descriptors.normal_map_textures.len().saturating_sub(1);
        for (i, obj) in self.draw_objects.iter().take(self.n_objects).enumerate() {
            let albedo = obj.texture_slot.min(last_tex) as u32;
            let normal = (albedo_count + obj.normal_map_slot.min(last_nm)) as u32;
            let rec = pack_object_record(obj, albedo, normal);
            // SAFETY: the buffer was sized for `n_objects` records and the
            // loop is bounded by `take(n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuObjectData as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }

        // Streamed chunks: one record each in the reserved region at
        // `[chunk_record_base() + k]`, packed exactly like a static object (chunk
        // geometry already lives in the shared VB/IB with the chunk's `base_vertex`,
        // so they ride the static + instance prefix `ExecuteIndirect`). Per-chunk
        // flat-pool texture indices give per-chunk materials. A non-resident (freed)
        // chunk slot's stale object record here is never read -- `build_draw_args_buffer`
        // disables it (ENABLED clear), and the cull kernel skips `objects[i]` for a
        // disabled record. The unused reserve tail is likewise never read.
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
        // after the per-frame skin deform), flat-pool texture indices like a
        // static object, and a padded bind-pose AABB so the cull kernel can
        // frustum/Hi-Z test them. Drawn by the main pass's 2nd `ExecuteIndirect`.
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

    // Recompute every instanced cluster's per-LOD-bucket partition for the
    // current camera, memcpy bucket-ordered instance matrices into this
    // frame's mapped upload buffers, and store the layout in
    // `instance_bucket_layouts` so every instanced draw site (main,
    // shadow's iter-per-instance path, SSAO / SSR / TAA-velocity
    // pre-passes) reads the same partition.
    //
    // Called once per frame from `record_frame` BEFORE `execute_graph`
    // dispatches any pass, because SSAO / SSR / TAA-velocity pre-passes
    // run earlier in the graph than main but share the same per-frame
    // upload buffer. Doing the upload here means every pre-pass + main
    // see the **current** frame's bucket layout; the legacy code (which
    // uploaded inside the main pass) only worked because instances didn't
    // move frame-to-frame, so reading the previous frame's buffer
    // happened to match. With LOD bucketing the buffer order depends on
    // `cam_pos`, so we have to upload up-front.
    pub(super) fn build_instance_upload(&self, frame_idx: usize, cam_pos: [f32; 3]) {
        let mut layouts = self.instanced.bucket_layouts.write().unwrap();
        // Re-shape the outer Vec when cluster count changed (runtime asset
        // hot-reload), then clear every row in place to reuse heap.
        if layouts.len() != self.instanced.clusters.len() {
            layouts.clear();
            layouts.resize(self.instanced.clusters.len(), Vec::new());
        } else {
            for row in layouts.iter_mut() {
                row.clear();
            }
        }
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            if cluster.instances.is_empty() {
                continue;
            }
            let upload_ptr = self.instanced.upload_ptrs[frame_idx][cluster_idx];
            const STRIDE: usize = std::mem::size_of::<[[f32; 4]; 4]>();

            let buckets = cluster.lod_buckets(cam_pos);
            let row = &mut layouts[cluster_idx];
            row.reserve(buckets.len());
            let mut prefix_instances: usize = 0;
            for bucket in buckets {
                let count = bucket.instances.len();
                let bucket_bytes = count * STRIDE;
                let byte_offset = prefix_instances * STRIDE;
                // SAFETY: the upload buffer was sized at init for every
                // instance the cluster declared; the sum of bucket
                // lengths matches that count, so the write stays in
                // bounds.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bucket.instances.as_ptr() as *const u8,
                        upload_ptr.add(byte_offset),
                        bucket_bytes,
                    );
                }
                row.push(crate::directx::context::InstanceBucketLayout {
                    instance_byte_offset: byte_offset as u64,
                    instance_count: count as u32,
                    index_offset: bucket.index_offset,
                    index_count: bucket.index_count,
                    instances: bucket.instances,
                });
                prefix_instances += count;
            }
        }
    }

    // Encode the SSAO pre-pass (if enabled), the bindless + legacy main pass,
    // the instanced clusters pass, and the skinned meshes pass into `cmd`.
    // Finishes by resolving (MSAA) or transitioning (no MSAA) the HDR target
    // to `PIXEL_SHADER_RESOURCE` so the velocity / TAA / bloom / composite
    // passes can sample it.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn encode_main_pass(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        width: u32,
        height: u32,
        view_gva: u64,
        light_gva: u64,
        shadow_ubo_gva: u64,
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
        visible: &[u32],
    ) {
        let depth_dsv = self.depth_dsv;

        unsafe {
            cmd.OMSetRenderTargets(1, Some(&self.hdr.color_rtv), false, Some(&depth_dsv));
            cmd.ClearRenderTargetView(self.hdr.color_rtv, &self.clear_color, None);
            cmd.ClearDepthStencilView(depth_dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);
        }

        // Pipeline-independent main-pass state: topology, geometry buffers,
        // and the shader-visible descriptor heaps. Survives root-signature
        // changes, so it is set once before either sub-pass binds its pipeline.
        unsafe {
            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
        }

        let last_obj = self.draw_objects.len().saturating_sub(1);

        // SSAO (GTAO) ran ahead of this pass via the pre-graph
        // (`PassId::SsaoBlur` dispatches the bundled prepass + kernel +
        // blur). The RAW edge `ao_output` → Main pins SsaoBlur → Main
        // in the toposort, so by the time we get here the blurred
        // occlusion target is filled. The main fragment shaders sample
        // it via the standard SRV binding at the bindless slot the
        // SSAO encoder updates.

        // Build-time static objects render through the bindless pipeline
        // driven by a GPU-culled indirect command buffer.
        // A compute kernel frustum/distance-tests every build-time object and
        // writes one ExecuteIndirect command per object (survivors get a real
        // draw, culled / disabled objects an instance_count-0 no-op); the
        // bindless main pass then issues the whole buffer with a single
        // ExecuteIndirect; the CPU never walks the static draw list. Each
        // draw is stateless apart from the per-command object-id b0 root
        // constant, with model/material/textures fetched from the per-frame
        // GpuObjectData buffer + the bindless texture pool. Streamed
        // VoxelWorld chunks keep the legacy per-draw pipeline below.
        let use_bindless = self.cull.main_bindless_pso.is_some() && self.cull_count() > 0;
        if use_bindless {
            // The per-frame per-object SRV-pool record buffer
            // (`build_object_buffer`) and the cull compute dispatch
            // (`encode_cull`) both ran ahead of this pass via the
            // pre-graph: `build_object_buffer` as inline CPU prep
            // before the executor dispatch, and `encode_cull` through
            // the executor's `PassId::Cull` arm. The toposort's
            // RAW edge from `draw_args` to Main pins Cull → Main, so
            // by the time we get here `indirect_cmd_buffers[frame_idx]`
            // is filled with this frame's ExecuteIndirect commands.

            let bindless_pso = self.cull.main_bindless_pso.as_ref().unwrap();
            let bindless_root = self.cull.main_bindless_root_sig.as_ref().unwrap();
            let cull_sig = self.cull.cull_command_signature.as_ref().unwrap();
            let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
            let object_gva =
                unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };

            // Main bindless pass: issue the GPU-culled command buffer. The b0
            // object-id root constant is set per command by the command
            // signature, so it is not bound here.
            unsafe {
                cmd.SetPipelineState(bindless_pso);
                cmd.SetGraphicsRootSignature(bindless_root);
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                // [5] is the bindless texture pool (per-object SRV region base).
                cmd.SetGraphicsRootDescriptorTable(5, self.cull.bindless_pool_gpu);
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                // [8] root SRV: this frame's StructuredBuffer<GpuObjectData>.
                cmd.SetGraphicsRootShaderResourceView(8, object_gva);
                // [9] descriptor table: blurred SSAO occlusion (or 1x1 white
                // fallback when SSAO is disabled).
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
                // ExecuteIndirect #1: the static + instance prefix
                // `[0, skinned_record_base())` against the static VB/IB (bound
                // above). The skinned tail is drawn by a second ExecuteIndirect
                // below (different bound VB/IB), reading the same indirect buffer
                // from `skinned_record_base()` on.
                cmd.ExecuteIndirect(
                    cull_sig,
                    self.skinned_record_base() as u32,
                    indirect,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            // One CPU draw issued; the kernel-written ICB runs N indirect
            // commands inside, but the call count surfaced to the profiler
            // is the host-side draw. Mirrors Metal's bindless main pass.
            self.inc_draw_calls(1);
        }

        // Legacy per-draw main pass. Draws every visible object when the
        // bindless pass is inactive (custom shader / no build-time geometry);
        // otherwise only runtime clones (streamed VoxelWorld chunks now fold into
        // the bindless `ExecuteIndirect` as their own records). Also still used by
        // the instanced + skinned passes below.
        let legacy_needed = !use_bindless || !self.clone.slot_by_draw_idx.is_empty();
        if legacy_needed {
            unsafe {
                cmd.SetPipelineState(&self.main_pso);
                cmd.SetGraphicsRootSignature(&self.main_root_sig);
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                // [8] SSAO occlusion SRV (or 1x1 white fallback).
                cmd.SetGraphicsRootDescriptorTable(8, self.ssao_ao_srv_gpu());
            }
            // Shared static-object traversal (gate + LOD pick); the closure
            // owns this pass's per-draw bindings + draw.
            self.draw_static_objects(visible, cam_pos, |obj, i, index_offset, index_count| {
                if use_bindless && i < self.n_objects {
                    return; // build-time object, already drawn bindless
                }
                let is_clone = self.clone.slot_by_draw_idx.contains_key(&i);
                if use_bindless && i >= self.n_objects && !is_clone {
                    return; // streamed chunk, already drawn bindless (folded record)
                }
                // Descriptor table [5]: albedo + normal SRVs for this object.
                // Three sources for runtime-added draws past `n_objects`:
                //   - Runtime clones (`clone_static_draw_object`) have their
                //     own (albedo, normal) SRV pair baked into the clone pool
                //     at init; the draw_idx → clone_offset lookup finds it.
                //   - Streamed `VoxelWorld` chunks (only reached on a non-bindless
                //     world) share one baked pair at `chunk_srv_base_slot`.
                //   - Build-time draws use the pre-baked per-object pair.
                let obj_srv_gpu = if let Some(&clone_offset) = self.clone.slot_by_draw_idx.get(&i) {
                    self.clone_srv_gpu(clone_offset)
                } else if i >= self.n_objects {
                    self.chunk_srv_gpu()
                } else {
                    self.object_srv_gpu(i.min(last_obj))
                };
                unsafe {
                    cmd.SetGraphicsRootDescriptorTable(5, obj_srv_gpu);

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
                    cmd.SetGraphicsRoot32BitConstants(
                        0,
                        28,
                        &push as *const MainPush as *const std::ffi::c_void,
                        0,
                    );
                    cmd.DrawIndexedInstanced(
                        index_count as u32,
                        1,
                        index_offset as u32,
                        obj.base_vertex,
                        0,
                    );
                }
                self.inc_draw_calls(1);
            });
        }

        // Instanced clusters main pass. Skipped when the bindless merge is active:
        // each instance is folded into the GPU-driven `GpuObjectData` buffer as a
        // record at `n_objects + k` and drawn by the bindless `ExecuteIndirect`
        // above (with per-instance culling). The gbuffer pre-pass + shadow still
        // use this legacy path for instances.
        if let (Some(inst_pso), Some(inst_root_sig)) = (
            self.instanced.pso.as_ref(),
            self.instanced.root_sig.as_ref(),
        ) && !self.instanced.clusters.is_empty()
            && !use_bindless
        {
            unsafe {
                cmd.SetPipelineState(inst_pso);
                cmd.SetGraphicsRootSignature(inst_root_sig);

                // Re-bind the per-frame CBVs since we switched root sig.
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);

                cmd.SetDescriptorHeaps(&[
                    Some(self.descriptors.srv_heap.clone()),
                    Some(self.descriptors.sampler_heap.clone()),
                ]);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                // [9] SSAO occlusion SRV (or 1x1 white fallback).
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
            }

            // Shared cluster cull + bucket iteration; the closures own
            // this pass's per-cluster material/SRV bind and per-bucket
            // instance-SRV bump + draw. The shared layout means the bucket
            // each instance lands in matches the other instanced passes.
            self.draw_instanced_clusters(
                frame_idx,
                frustum,
                cam_pos,
                |cluster_idx, cluster| {
                    // Per-cluster (albedo, normal) SRV pair allocated at init.
                    let cluster_srv_gpu = self.cluster_srv_gpu(cluster_idx);
                    unsafe {
                        cmd.SetGraphicsRootDescriptorTable(5, cluster_srv_gpu);

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
                        cmd.SetGraphicsRoot32BitConstants(
                            0,
                            28,
                            &push as *const MainPush as *const std::ffi::c_void,
                            0,
                        );
                    }
                },
                |bucket, inst_gva_base| {
                    unsafe {
                        // Root SRV at param [8]: per-instance matrices.
                        // Bumping the GVA past prior buckets points the
                        // structured-buffer SRV at this bucket's slice;
                        // SV_InstanceID then indexes from 0 within it.
                        cmd.SetGraphicsRootShaderResourceView(
                            8,
                            inst_gva_base + bucket.instance_byte_offset,
                        );
                        cmd.DrawIndexedInstanced(
                            bucket.index_count as u32,
                            bucket.instance_count,
                            bucket.index_offset as u32,
                            0,
                            0,
                        );
                    }
                    self.inc_draw_calls(1);
                },
            );
        }

        // Skinned meshes main pass. When the GPU-driven bindless fold is active,
        // skinned objects ride the same cull buffers as static + instances and are
        // drawn (as rigid deformed geometry) by a 2nd `ExecuteIndirect` over this
        // frame's deformed-vertex buffer + the skinned u16 index buffer, reading
        // the cull-written indirect buffer from `skinned_record_base()`. The
        // `encode_skin` compute pass (Cull graph arm) has already posed the
        // deformed buffer and left it in VERTEX_AND_CONSTANT_BUFFER. Otherwise the
        // legacy per-draw skinned pass runs (custom-shader worlds, or a
        // pure-skinned world with no static geometry to engage bindless).
        if use_bindless && self.n_skinned > 0 {
            if let (Some(bindless_pso), Some(bindless_root), Some(cull_sig), Some(deformed_vbv)) = (
                self.cull.main_bindless_pso.as_ref(),
                self.cull.main_bindless_root_sig.as_ref(),
                self.cull.cull_command_signature.as_ref(),
                self.skinned.deformed_vbvs.get(frame_idx),
            ) {
                let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
                let object_gva =
                    unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };
                unsafe {
                    cmd.SetPipelineState(bindless_pso);
                    cmd.SetGraphicsRootSignature(bindless_root);
                    cmd.IASetPrimitiveTopology(
                        windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                    );
                    // Bind the deformed verts + skinned u16 IB; the records carry
                    // base_vertex = 0 (the deformed buffer mirrors global skinned
                    // indexing) and index offsets into the skinned IB.
                    cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv]));
                    cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                    cmd.SetDescriptorHeaps(&[
                        Some(self.descriptors.srv_heap.clone()),
                        Some(self.descriptors.sampler_heap.clone()),
                    ]);
                    cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                    cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                    cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                    cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                    cmd.SetGraphicsRootDescriptorTable(5, self.cull.bindless_pool_gpu);
                    cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                    cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                    cmd.SetGraphicsRootShaderResourceView(8, object_gva);
                    cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
                    // ExecuteIndirect #2: skinned tail
                    // `[skinned_record_base(), cull_count())`, byte-offset into the
                    // same indirect command buffer.
                    cmd.ExecuteIndirect(
                        cull_sig,
                        self.n_skinned as u32,
                        indirect,
                        (self.skinned_record_base()
                            * crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize)
                            as u64,
                        None::<&ID3D12Resource>,
                        0,
                    );
                }
                self.inc_draw_calls(1);
            }
        } else if let (Some(skinned_pso), Some(skinned_root_sig)) =
            (self.skinned.pso.as_ref(), self.skinned.root_sig.as_ref())
            && !self.skinned.draw_objects.is_empty()
        {
            unsafe {
                cmd.SetPipelineState(skinned_pso);
                cmd.SetGraphicsRootSignature(skinned_root_sig);
                cmd.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                cmd.IASetVertexBuffers(0, Some(&[self.skinned.vertex_buffer_view]));
                cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));

                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);

                cmd.SetDescriptorHeaps(&[
                    Some(self.descriptors.srv_heap.clone()),
                    Some(self.descriptors.sampler_heap.clone()),
                ]);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                // [9] SSAO occlusion SRV (or 1x1 white fallback).
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
            }

            // Shared skinned traversal (gate + LOD pick); the closure owns
            // the per-object joint SRV + draw. Skinned meshes with no
            // authored alternates collapse to LOD0.
            self.draw_skinned_objects(cam_pos, |obj, i, index_offset, index_count| {
                unsafe {
                    cmd.SetGraphicsRootDescriptorTable(5, self.skinned_srv_gpu(i));
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
                    cmd.SetGraphicsRoot32BitConstants(
                        0,
                        28,
                        &push as *const MainPush as *const std::ffi::c_void,
                        0,
                    );
                    // Root SRV at param [8]: this object's joint matrices.
                    cmd.SetGraphicsRootShaderResourceView(8, self.skinned_joint_gva(frame_idx, i));
                    cmd.DrawIndexedInstanced(index_count as u32, 1, index_offset as u32, 0, 0);
                }
                self.inc_draw_calls(1);
            });
        }

        // Resolve the HDR scene target so the post stack can sample it. Under
        // two-pass occlusion the resolve is deferred to `Main2` (which re-runs
        // the disoccluded geometry on top of this pass's colour + depth), so the
        // post stack sees the combined phase-1 + phase-2 scene. Skipping it here
        // leaves `hdr_color` in RENDER_TARGET (and `hdr_resolve`, when present,
        // in PIXEL_SHADER_RESOURCE) exactly as `Main2` expects to load them.
        if !self.two_pass_occlusion_active() {
            self.finish_hdr_target(cmd);
        }
    }

    // Resolve (MSAA) or transition (no MSAA) the HDR scene target to
    // `PIXEL_SHADER_RESOURCE` so the velocity / TAA / bloom / composite passes
    // can sample it. With MSAA on, resolve the multisampled `hdr_color` into the
    // single-sample `hdr_resolve` and restore `hdr_color` to RENDER_TARGET for
    // next frame; with MSAA off, the composite samples `hdr_color` directly. The
    // HDR SRV the composite binds was created at init on whichever of the two is
    // sampled here. Shared by the phase-1 main pass and the phase-2 `Main2`
    // pass; entry state must be `hdr_color` = RENDER_TARGET (+ `hdr_resolve` =
    // PIXEL_SHADER_RESOURCE), which is the resting state after either pass's
    // draws and the state `Main2` inherits when phase 1 deferred the resolve.
    pub(in crate::directx) fn finish_hdr_target(&self, cmd: &ID3D12GraphicsCommandList) {
        if let Some(hdr_resolve) = &self.hdr.resolve {
            let color_to_src = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            );
            let resolve_to_dst = transition_barrier(
                hdr_resolve,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
            );
            unsafe { cmd.ResourceBarrier(&[color_to_src, resolve_to_dst]) };
            unsafe { cmd.ResolveSubresource(hdr_resolve, 0, &self.hdr.color, 0, HDR_FORMAT) };
            let resolve_to_psr = transition_barrier(
                hdr_resolve,
                D3D12_RESOURCE_STATE_RESOLVE_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            // Restore the multisampled target to RENDER_TARGET for next frame.
            let color_to_rt = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            unsafe { cmd.ResourceBarrier(&[resolve_to_psr, color_to_rt]) };
        } else {
            let color_to_psr = transition_barrier(
                &self.hdr.color,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            unsafe { cmd.ResourceBarrier(&[color_to_psr]) };
        }
    }

    // Phase-2 main pass for two-pass occlusion (`Main2`). Loads (does not clear)
    // the HDR colour + depth `encode_main_pass` (phase 1) wrote and re-runs the
    // bindless indirect draw through this frame's second indirect buffer (the
    // phase-2 cull's output), depth-compositing the disoccluded geometry with
    // phase 1. Static + instances + skinned all ride the shared cull buffers, so
    // any of them that were occlusion candidates are re-tested by the phase-2 cull
    // and redrawn here with the same two-`ExecuteIndirect` split as phase 1 (the
    // static+instance prefix against the static VB/IB, the skinned tail against the
    // deformed VB + skinned IB). Finishes by resolving the HDR target (the resolve
    // phase 1 deferred), so the post-decoration stack reads the combined result.
    // Only dispatched when `two_pass_occlusion_active`
    // (the graph gates the Main2 node on it), so all phase-2 resources are
    // present; the resolve still runs even if there is nothing to redraw.
    // Mirrors `metal/draw/main.rs::encode_main_pass_phase2`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn encode_main_pass_phase2(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        width: u32,
        height: u32,
        view_gva: u64,
        light_gva: u64,
        shadow_ubo_gva: u64,
    ) {
        let depth_dsv = self.depth_dsv;

        // Load (do not clear) the phase-1 colour + depth: Main2 composites the
        // disoccluded geometry on top.
        unsafe {
            cmd.OMSetRenderTargets(1, Some(&self.hdr.color_rtv), false, Some(&depth_dsv));

            let vp = D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            };
            cmd.RSSetViewports(&[vp]);
            let scissor = RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            };
            cmd.RSSetScissorRects(&[scissor]);

            cmd.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            cmd.IASetVertexBuffers(0, Some(&[self.geometry.vertex_buffer_view]));
            cmd.IASetIndexBuffer(Some(&self.geometry.index_buffer_view));
            cmd.SetDescriptorHeaps(&[
                Some(self.descriptors.srv_heap.clone()),
                Some(self.descriptors.sampler_heap.clone()),
            ]);
        }

        // Re-issue the bindless pass over the phase-2 indirect buffer. Same
        // bindings + the same static-prefix / skinned-tail split as the phase-1
        // bindless branch.
        if let (Some(bindless_pso), Some(bindless_root), Some(cull_sig), Some(indirect)) = (
            self.cull.main_bindless_pso.as_ref(),
            self.cull.main_bindless_root_sig.as_ref(),
            self.cull.cull_command_signature.as_ref(),
            self.cull.indirect_cmd_buffers_2.get(frame_idx),
        ) && self.cull_count() > 0
        {
            let object_gva =
                unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };
            unsafe {
                cmd.SetPipelineState(bindless_pso);
                cmd.SetGraphicsRootSignature(bindless_root);
                cmd.SetGraphicsRootConstantBufferView(1, view_gva);
                cmd.SetGraphicsRootConstantBufferView(2, light_gva);
                cmd.SetGraphicsRootConstantBufferView(3, shadow_ubo_gva);
                cmd.SetGraphicsRootDescriptorTable(4, self.shadow.srv_gpu);
                cmd.SetGraphicsRootDescriptorTable(5, self.cull.bindless_pool_gpu);
                cmd.SetGraphicsRootDescriptorTable(6, self.descriptors.shadow_sampler_gpu);
                cmd.SetGraphicsRootDescriptorTable(7, self.descriptors.linear_sampler_gpu);
                cmd.SetGraphicsRootShaderResourceView(8, object_gva);
                cmd.SetGraphicsRootDescriptorTable(9, self.ssao_ao_srv_gpu());
                // ExecuteIndirect #1: static + instance prefix against the static
                // VB/IB (bound above).
                cmd.ExecuteIndirect(
                    cull_sig,
                    self.skinned_record_base() as u32,
                    indirect,
                    0,
                    None::<&ID3D12Resource>,
                    0,
                );
            }
            self.inc_draw_calls(1);

            // ExecuteIndirect #2: skinned tail against the deformed VB + skinned
            // IB. The PSO + root signature + root descriptors set above persist
            // (same bindless pipeline), so only the vertex/index buffers rebind.
            if self.n_skinned > 0
                && let Some(deformed_vbv) = self.skinned.deformed_vbvs.get(frame_idx)
            {
                unsafe {
                    cmd.IASetVertexBuffers(0, Some(&[*deformed_vbv]));
                    cmd.IASetIndexBuffer(Some(&self.skinned.index_buffer_view));
                    cmd.ExecuteIndirect(
                        cull_sig,
                        self.n_skinned as u32,
                        indirect,
                        (self.skinned_record_base()
                            * crate::directx::cull::INDIRECT_COMMAND_STRIDE as usize)
                            as u64,
                        None::<&ID3D12Resource>,
                        0,
                    );
                }
                self.inc_draw_calls(1);
            }
        }

        // Resolve the combined phase-1 + phase-2 scene (the resolve phase 1
        // deferred). Always runs, even with nothing disoccluded, so the post
        // stack always reads a resolved target.
        self.finish_hdr_target(cmd);
    }
}
