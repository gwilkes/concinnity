// src/vulkan/resources/skinning.rs
//
// Skinned-mesh upload, per-frame joint upload, and helpers for VkContext.
// Builds the skinned pipelines + per-(frame, object) joint storage buffers
// once at init; per-frame `update_skinned_pose` + `upload_joint_matrices`
// keep the matrices fresh from the gameplay-side pose update.

use ash::vk;

use crate::gfx::mesh_payload::SkinnedVertex;
use crate::gfx::render_types::*;

use super::super::context::*;
use super::super::math::*;
use super::super::pipeline::{
    compile_skinned_shaders, create_skinned_pipeline, create_skinned_shadow_pipeline,
};
use super::super::texture::create_buffer;
use super::{alloc_descriptor_sets, create_descriptor_set_layout};

impl VkContext {
    // Upload skinned-mesh geometry and build the skinned render pipelines.
    pub fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u16],
        draw_objects: Vec<SkinnedDrawObject>,
        frag_bytes: &[u8],
    ) -> Result<(), String> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }
        self.wait_idle();
        let frames = self.frames_in_flight.max(1);
        let n = draw_objects.len();

        let (skinned_vs, skinned_shadow_vs, frag_spv) =
            compile_skinned_shaders(self.hot_reload, frag_bytes)?;

        let joint_set_layout = create_descriptor_set_layout(
            &self.device,
            &[(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                vk::ShaderStageFlags::VERTEX,
            )],
        )?;

        let main_pc = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(112);
        let main_set_layouts = [
            self.descriptors.global_set_layout,
            self.descriptors.object_set_layout,
            joint_set_layout,
        ];
        let skinned_pipeline_layout = unsafe {
            self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&main_set_layouts)
                    .push_constant_ranges(std::slice::from_ref(&main_pc)),
                None,
            )
        }
        .map_err(|e| format!("skinned pipeline layout: {e}"))?;
        let skinned_pipeline = create_skinned_pipeline(
            &self.device,
            self.main_render_pass,
            skinned_pipeline_layout,
            &skinned_vs,
            &frag_spv,
            self.msaa_samples,
        )?;

        let (skinned_shadow_pipeline, skinned_shadow_pipeline_layout) =
            if let (Some(_), Some(shadow_global)) =
                (self.shadow.pipeline, self.shadow.global_set_layout)
            {
                let shadow_pc = vk::PushConstantRange::default()
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .offset(0)
                    .size(80);
                let shadow_set_layouts = [shadow_global, joint_set_layout];
                let layout = unsafe {
                    self.device.create_pipeline_layout(
                        &vk::PipelineLayoutCreateInfo::default()
                            .set_layouts(&shadow_set_layouts)
                            .push_constant_ranges(std::slice::from_ref(&shadow_pc)),
                        None,
                    )
                }
                .map_err(|e| format!("skinned shadow pipeline layout: {e}"))?;
                let pipeline = create_skinned_shadow_pipeline(
                    &self.device,
                    self.shadow.render_pass,
                    layout,
                    &skinned_shadow_vs,
                )?;
                (Some(pipeline), Some(layout))
            } else {
                (None, None)
            };

        let vtx_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                std::mem::size_of_val(indices),
            )
        };
        // When ray-traced reflections are live the skinned VB/IB feed the RT
        // skinning path: the skin compute kernel reads the bind-pose VB as a
        // storage buffer, and the IB is both the skinned BLAS index input
        // (device-addressed) and the hit-shader's u16 index SSBO. These flags
        // require the ray-query extensions, so add them whenever the device is
        // RT-capable (not only when RT is on at launch) so a later live toggle
        // finds the skinned buffers already usable, mirroring how the static
        // VB/IB gate their RT flags at init. Inert when RT is never built.
        let skinned_vb_rt = if self.rt_capable {
            vk::BufferUsageFlags::STORAGE_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let skinned_ib_rt = if self.rt_capable {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
        } else {
            vk::BufferUsageFlags::empty()
        };
        let (skinned_vbuf, skinned_vmem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            vtx_bytes.len() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | skinned_vb_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let (skinned_ibuf, skinned_imem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            idx_bytes.len() as u64,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | skinned_ib_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        self.write_geometry_region(skinned_vbuf, 0, vtx_bytes)?;
        self.write_geometry_region(skinned_ibuf, 0, idx_bytes)?;

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count((n * 2) as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((n * frames) as u32),
        ];
        let pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets((n + n * frames) as u32)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|e| format!("skinned descriptor pool: {e}"))?;

        let object_layouts: Vec<_> = (0..n).map(|_| self.descriptors.object_set_layout).collect();
        let object_sets = alloc_descriptor_sets(&self.device, pool, &object_layouts)?;
        let last_tex = self.textures.len().saturating_sub(1);
        let last_nm = self.normal_map_textures.len().saturating_sub(1);
        for (&set, obj) in object_sets.iter().zip(draw_objects.iter()) {
            let albedo_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.textures[obj.texture_slot.min(last_tex)].view)
                .sampler(self.linear_sampler);
            let nm_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.normal_map_textures[obj.normal_map_slot.min(last_nm)].view)
                .sampler(self.linear_sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&albedo_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&nm_info)),
            ];
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }

        // Per-(frame, object) joint storage buffers seeded with identity
        // matrices so any not-yet-overwritten slot reads as identity.
        let joint_buf_bytes = (MAX_JOINTS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let identity_seed: Vec<[[f32; 4]; 4]> = vec![IDENTITY4; MAX_JOINTS];
        let mut joint_buffers: Vec<Vec<vk::Buffer>> = Vec::with_capacity(frames);
        let mut joint_memories: Vec<Vec<vk::DeviceMemory>> = Vec::with_capacity(frames);
        let mut joint_ptrs: Vec<Vec<*mut u8>> = Vec::with_capacity(frames);
        let mut joint_sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let mut bufs: Vec<vk::Buffer> = Vec::with_capacity(n);
            let mut mems: Vec<vk::DeviceMemory> = Vec::with_capacity(n);
            let mut ptrs: Vec<*mut u8> = Vec::with_capacity(n);
            for _ in 0..n {
                let (buf, mem) = create_buffer(
                    &self.instance,
                    &self.device,
                    self.physical_device,
                    joint_buf_bytes,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?;
                let ptr = unsafe {
                    self.device
                        .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                }
                .map_err(|e| format!("map skinned joint buffer: {e}"))?
                    as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        identity_seed.as_ptr() as *const u8,
                        ptr,
                        joint_buf_bytes as usize,
                    );
                }
                bufs.push(buf);
                mems.push(mem);
                ptrs.push(ptr);
            }
            let layouts: Vec<_> = (0..n).map(|_| joint_set_layout).collect();
            let sets = alloc_descriptor_sets(&self.device, pool, &layouts)?;
            for (i, &set) in sets.iter().enumerate() {
                let info = vk::DescriptorBufferInfo::default()
                    .buffer(bufs[i])
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&info));
                unsafe {
                    self.device
                        .update_descriptor_sets(std::slice::from_ref(&write), &[])
                };
            }
            joint_buffers.push(bufs);
            joint_memories.push(mems);
            joint_ptrs.push(ptrs);
            joint_sets.push(sets);
        }

        self.skinned.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY4; o.joint_count.max(1)])
            .collect();

        self.skinned.pipeline = Some(skinned_pipeline);
        self.skinned.pipeline_layout = Some(skinned_pipeline_layout);
        self.shadow.skinned_pipeline = skinned_shadow_pipeline;
        self.shadow.skinned_pipeline_layout = skinned_shadow_pipeline_layout;
        self.skinned.joint_set_layout = Some(joint_set_layout);
        self.skinned.descriptor_pool = Some(pool);
        self.skinned.vertex_buffer = skinned_vbuf;
        self.skinned.vertex_buffer_memory = skinned_vmem;
        self.skinned.vertex_buffer_bytes = vtx_bytes.len() as u64;
        self.skinned.index_buffer = skinned_ibuf;
        self.skinned.index_buffer_memory = skinned_imem;
        self.skinned.index_buffer_bytes = idx_bytes.len() as u64;
        self.skinned.object_sets = object_sets;
        self.skinned.joint_buffers = joint_buffers;
        self.skinned.joint_memories = joint_memories;
        self.skinned.joint_ptrs = joint_ptrs;
        self.skinned.joint_sets = joint_sets;
        self.skinned.draw_objects = draw_objects;

        // GPU-driven main-pass skinning fold: when the bindless cull path is active,
        // build the `rt_skin` compute pipeline + per-frame deformed-vertex buffers +
        // their descriptor sets, and set `self.n_skinned` (which engages the fold so
        // `cull_count()` reserves the skinned tail). A build failure leaves it 0 and
        // the legacy skinned main pass runs. Mirrors the DirectX `upload_skinned`.
        if self.cull.bindless_pipeline.is_some()
            && self.cull_count() > 0
            && let Err(e) = self.build_main_skin(vertices.len())
        {
            tracing::warn!(
                "skinned: main-pass skin fold build failed ({e}); skinned meshes \
                 use the legacy main pass"
            );
        }

        if let Some(gb) = self.gbuffer.as_mut() {
            gb.ensure_skinned_gbuffer_pso(&self.device, joint_set_layout)?;
        }
        Ok(())
    }

    // Replace a `SkinnedMesh` draw slot's vertex + index data in place.
    // Driven by asset hot-reload (`cn debug` only). The shared skinned VB
    // / IB were sized once at `upload_skinned` to hold every skinned
    // mesh's geometry, so the new payload must fit within this slot's
    // existing region (size-changing reloads route through
    // `rebuild_skinned_geometry`). `vertex_base` is the slot's vertex
    // offset *in vertices*; `indices` are mesh-relative and get rebased
    // by `vertex_base` before being written into the shared IB. Skinned
    // IBs stay `R16_UINT` so the rebased indices must fit in u16.
    // Mirrors `DxContext::update_skinned_mesh_geometry`. Reached only through
    // the bin's `cn debug` runtime-mutation path (dead in the FFI lib, live in
    // the bin).
    #[allow(dead_code)]
    pub fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u16,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .draw_objects
            .get(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_mesh_geometry: skinned object {} out of range",
                    skinned_index
                )
            })?;
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_skinned_mesh_geometry: skinned {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_skinned_geometry)",
                skinned_index,
                obj.index_count,
                indices.len()
            ));
        }
        if self.skinned.vertex_buffer == vk::Buffer::null()
            || self.skinned.index_buffer == vk::Buffer::null()
        {
            return Err(
                "update_skinned_mesh_geometry: no skinned vertex/index buffer (was \
                 upload_skinned called?)"
                    .to_string(),
            );
        }
        let v_byte_off =
            (vertex_base as usize).saturating_mul(std::mem::size_of::<SkinnedVertex>());
        let v_byte_len = std::mem::size_of_val(vertices);
        let v_buf_len = self.skinned.vertex_buffer_bytes as usize;
        if v_byte_off + v_byte_len > v_buf_len {
            return Err(format!(
                "update_skinned_mesh_geometry: vertex region [{}, {}) overruns skinned \
                 vertex buffer length {}",
                v_byte_off,
                v_byte_off + v_byte_len,
                v_buf_len
            ));
        }
        let i_byte_off = (obj.index_offset * std::mem::size_of::<u16>()) as u64;
        let rebased: Vec<u16> = indices
            .iter()
            .map(|&i| i.checked_add(vertex_base))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                format!(
                    "update_skinned_mesh_geometry: index rebase by {} overflows u16 \
                     (skinned slot {})",
                    vertex_base, skinned_index
                )
            })?;

        self.wait_idle();

        let vert_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        self.write_geometry_region(self.skinned.vertex_buffer, v_byte_off as u64, vert_bytes)?;
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                rebased.as_ptr() as *const u8,
                std::mem::size_of_val(rebased.as_slice()),
            )
        };
        self.write_geometry_region(self.skinned.index_buffer, i_byte_off, idx_bytes)?;
        Ok(())
    }

    // Update a skinned slot's joint count to match a re-imported `.glb`
    // skeleton. The per-(frame, object) joint storage buffers were sized
    // for `MAX_JOINTS` matrices at init, so no GPU resource needs to grow:
    // only the CPU-side `SkinnedDrawObject::joint_count` and the parallel
    // `skinned_joint_matrices` slot are touched. Shrinking truncates the
    // matrix slot; growing seeds the new entries to identity so the
    // shader sees a valid pose until the next `update_skinned_pose` runs.
    // Counts above `MAX_JOINTS` are clamped (the storage buffer is fixed
    // at that size). Driven by asset hot-reload. Mirrors
    // `DxContext::update_skinned_skeleton`. Reached only through the bin's
    // `cn debug` runtime-mutation path (dead in the FFI lib, live in the bin).
    #[allow(dead_code)]
    pub fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .draw_objects
            .get_mut(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_skeleton: skinned object {} out of range",
                    skinned_index
                )
            })?;
        let capped = new_joint_count.min(MAX_JOINTS);
        obj.joint_count = capped;
        let size = capped.max(1);
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.resize(size, IDENTITY4);
        }
        Ok(())
    }

    // Replace the skinning matrices for one skinned object.
    pub fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.clear();
            slot.extend_from_slice(matrices);
            if slot.is_empty() {
                slot.push(IDENTITY4);
            }
        }
    }

    // Copy this frame's skinning matrices into the per-frame joint buffers.
    pub(in crate::vulkan) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_bufs) = self.skinned.joint_ptrs.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.skinned.joint_matrices.iter().enumerate() {
            let Some(&dst) = frame_bufs.get(i) else {
                continue;
            };
            let count = mats.len().min(MAX_JOINTS);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mats.as_ptr() as *const u8,
                    dst,
                    count * std::mem::size_of::<[[f32; 4]; 4]>(),
                );
            }
        }
    }

    // Bind the skinned vertex + index buffers for the skinned passes.
    pub(in crate::vulkan) fn skinned_geometry(&self) -> (vk::Buffer, vk::Buffer) {
        (self.skinned.vertex_buffer, self.skinned.index_buffer)
    }
}
