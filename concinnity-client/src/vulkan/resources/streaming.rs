// src/vulkan/resources/streaming.rs
//
// VoxelWorld chunk streaming for VkContext: appends a headroom region to the
// shared vertex/index buffers, builds the chunk descriptor set from the
// world's chunk material, then allocates / frees per-chunk geometry from that
// headroom on demand.

use ash::vk;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::*;

use super::super::context::*;
use super::super::texture::{self, create_buffer};
use super::alloc_descriptor_sets;

impl VkContext {
    // Grow the shared vertex/index buffers by a headroom region for streamed
    // `VoxelWorld` chunks, seed the chunk sub-allocators with it, and build
    // the shared chunk (albedo, normal) descriptor set from the world's chunk
    // material.
    pub fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
        texture_slot: usize,
        normal_map_slot: usize,
    ) -> Result<(), String> {
        self.wait_idle();
        let old_v = self.geometry.vertex_buffer_bytes;
        let old_i = self.geometry.index_buffer_bytes;
        let new_v = old_v + chunk_vtx_bytes as u64;
        let new_i = old_i + chunk_idx_bytes as u64;

        let (new_vbuf, new_vmem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            new_v,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let (new_ibuf, new_imem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            new_i,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        texture::one_shot_submit(
            &self.device,
            self.commands.command_pool,
            self.graphics_queue,
            |cmd| {
                let vcopy = vk::BufferCopy::default().size(old_v);
                let icopy = vk::BufferCopy::default().size(old_i);
                unsafe {
                    self.device.cmd_copy_buffer(
                        cmd,
                        self.geometry.vertex_buffer,
                        new_vbuf,
                        std::slice::from_ref(&vcopy),
                    );
                    self.device.cmd_copy_buffer(
                        cmd,
                        self.geometry.index_buffer,
                        new_ibuf,
                        std::slice::from_ref(&icopy),
                    );
                }
            },
        )?;

        unsafe {
            self.device
                .destroy_buffer(self.geometry.vertex_buffer, None);
            self.device
                .free_memory(self.geometry.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.geometry.index_buffer, None);
            self.device
                .free_memory(self.geometry.index_buffer_memory, None);
        }
        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.vertex_buffer_memory = new_vmem;
        self.geometry.index_buffer = new_ibuf;
        self.geometry.index_buffer_memory = new_imem;
        self.geometry.vertex_buffer_bytes = new_v;
        self.geometry.index_buffer_bytes = new_i;

        self.chunk_stream
            .vtx_alloc
            .free(old_v, chunk_vtx_bytes as u64, 0);
        self.chunk_stream
            .idx_alloc
            .free(old_i, chunk_idx_bytes as u64, 0);

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(2)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }
            .map_err(|e| format!("chunk descriptor pool: {e}"))?;
        let set = alloc_descriptor_sets(&self.device, pool, &[self.descriptors.object_set_layout])?
            .into_iter()
            .next()
            .ok_or("chunk descriptor set: allocation returned none")?;
        let tex_slot = texture_slot.min(self.textures.len().saturating_sub(1));
        let nm_slot = normal_map_slot.min(self.normal_map_textures.len().saturating_sub(1));
        self.chunk_stream.texture_slot = Some(tex_slot);
        self.chunk_stream.normal_map_slot = Some(nm_slot);
        let albedo_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.textures[tex_slot].view)
            .sampler(self.linear_sampler);
        let nm_info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(self.normal_map_textures[nm_slot].view)
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
        self.chunk_stream.descriptor_pool = Some(pool);
        self.chunk_stream.object_set = Some(set);
        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // add (or recycle) a `DrawObject` for it; returns the draw-list index.
    #[allow(clippy::too_many_arguments)]
    pub fn add_chunk_mesh(
        &mut self,
        vertices: &[Vertex],
        indices: &[u16],
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: MaterialUniforms,
        frame: u64,
    ) -> Result<usize, String> {
        if vertices.is_empty() || indices.is_empty() {
            return Err("add_chunk_mesh: empty chunk geometry".to_string());
        }
        self.chunk_stream.vtx_alloc.reclaim(frame);
        self.chunk_stream.idx_alloc.reclaim(frame);

        let v_len = std::mem::size_of_val(vertices);
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .chunk_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "add_chunk_mesh: no free chunk vertex space for {} bytes",
                    v_len
                )
            })? as usize;
        let i_off = match self.chunk_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.chunk_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "add_chunk_mesh: no free chunk index space for {} bytes",
                    i_len
                ));
            }
        };

        self.wait_idle();

        let vert_bytes =
            unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, v_len) };
        self.write_geometry_region(self.geometry.vertex_buffer, v_off as u64, vert_bytes)?;
        let widened: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        let idx_bytes = unsafe { std::slice::from_raw_parts(widened.as_ptr() as *const u8, i_len) };
        self.write_geometry_region(self.geometry.index_buffer, i_off as u64, idx_bytes)?;

        let base_vertex = (v_off / std::mem::size_of::<Vertex>()) as i32;
        let obj = DrawObject {
            vertex_offset: v_off,
            vertex_count: vertices.len(),
            index_offset: i_off / std::mem::size_of::<u32>(),
            index_count: indices.len(),
            base_vertex,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            lod_alternates: Vec::new(),
        };

        let draw_idx = if let Some(slot) = self.chunk_stream.free_slots.pop() {
            self.draw_objects[slot] = obj;
            slot
        } else {
            self.draw_objects.push(obj);
            let idx = self.draw_objects.len() - 1;
            self.always_draw.push(idx as u32);
            idx
        };
        // Seed the streamed-chunk previous transform onto the unified G-buffer's
        // velocity bookkeeping so a chunk that streams in does not ghost from
        // IDENTITY on its first frame.
        if let Some(gb) = &mut self.gbuffer
            && draw_idx < gb.prev_models.len()
        {
            gb.prev_models[draw_idx] = model;
        }
        Ok(draw_idx)
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    pub fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("remove_chunk_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.chunk_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.chunk_stream.idx_alloc.free(i_off, i_len, retire_frame);
        let obj = &mut self.draw_objects[draw_idx];
        obj.visible = false;
        obj.resident = false;
        self.chunk_stream.free_slots.push(draw_idx);
        Ok(())
    }

    // Rewrite a resident chunk's model matrix.
    pub fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get_mut(draw_idx)
            .ok_or_else(|| format!("set_chunk_model: draw object {} out of range", draw_idx))?;
        obj.model = model;
        Ok(())
    }
}
