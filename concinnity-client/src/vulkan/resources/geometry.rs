// src/vulkan/resources/geometry.rs
//
// Streamed-mesh upload + eviction for VkContext, plus the shared
// `write_geometry_region` helper that copies a sub-region into the static
// vertex / index buffers via a host-visible staging buffer + one-shot command
// buffer. Used by mesh streaming (here), chunk streaming, and skinned upload.

use ash::vk;

use crate::gfx::mesh_payload::Vertex;

use super::super::context::*;
use super::super::texture::{self, create_buffer};

impl VkContext {
    // Copy `data` into a sub-region of a DEVICE_LOCAL geometry buffer.
    //
    // `dest` is the vertex or index buffer (both created with `TRANSFER_DST`).
    // The copy goes through a host-visible staging buffer and a one-shot
    // command buffer, mirroring `upload_geometry_buffer`'s init path. The
    // caller must `wait_idle` first so no in-flight command buffer still reads
    // `dest` while the transfer writes it.
    pub(in crate::vulkan) fn write_geometry_region(
        &self,
        dest: vk::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let size = data.len() as u64;
        let (staging, staging_mem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        unsafe {
            let ptr = self
                .device
                .map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| format!("map mesh staging: {e}"))? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(staging_mem);
        }
        let result = texture::one_shot_submit(
            &self.device,
            self.commands.command_pool,
            self.graphics_queue,
            |cmd| {
                let copy = vk::BufferCopy::default().dst_offset(offset).size(size);
                unsafe {
                    self.device
                        .cmd_copy_buffer(cmd, staging, dest, std::slice::from_ref(&copy))
                };
            },
        );
        unsafe {
            self.device.destroy_buffer(staging, None);
            self.device.free_memory(staging_mem, None);
        }
        result
    }

    // Upload a streamed mesh's geometry into the shared vertex and index
    // buffers, place it via the sub-allocators, and mark the draw resident.
    pub fn upload_mesh(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        frame: u64,
    ) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("upload_mesh: draw object {} out of range", draw_idx))?;
        let (vertex_count, index_count) = (obj.vertex_count, obj.index_count);
        if vertices.len() != vertex_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} vertices, got {}",
                draw_idx,
                vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != index_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} indices, got {}",
                draw_idx,
                index_count,
                indices.len()
            ));
        }

        self.geometry.mesh_vtx_alloc.reclaim(frame);
        self.geometry.mesh_idx_alloc.reclaim(frame);
        let v_len = std::mem::size_of_val(vertices);
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .geometry
            .mesh_vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "upload_mesh: draw {}: no free vertex space for {} bytes",
                    draw_idx, v_len
                )
            })? as usize;
        let i_off = match self.geometry.mesh_idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.geometry
                    .mesh_vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "upload_mesh: draw {}: no free index space for {} bytes",
                    draw_idx, i_len
                ));
            }
        };

        self.wait_idle();

        let vert_bytes =
            unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, v_len) };
        self.write_geometry_region(self.geometry.vertex_buffer, v_off as u64, vert_bytes)?;
        let base = (v_off / std::mem::size_of::<Vertex>()) as u32;
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = unsafe { std::slice::from_raw_parts(rebased.as_ptr() as *const u8, i_len) };
        self.write_geometry_region(self.geometry.index_buffer, i_off as u64, idx_bytes)?;

        let obj = &mut self.draw_objects[draw_idx];
        obj.vertex_offset = v_off;
        obj.index_offset = i_off / std::mem::size_of::<u32>();
        obj.resident = true;
        Ok(())
    }

    // Replace a build-time static mesh's vertex + index data in place.
    // Driven by asset hot-reload (`cn debug` only). Reuses the slot's
    // existing region in the shared VB / IB (allocated by the build-time
    // layout, not `mesh_vtx_alloc`), so the new geometry must match the
    // slot's `vertex_count` / `index_count` exactly; size-changing reloads
    // route through `rebuild_static_geometry` instead. LOD alternates are
    // uploaded into their existing per-LOD slices the same way LOD0 is.
    // `wait_idle` first guarantees no in-flight command buffer reads the
    // region we're about to overwrite. Mirrors
    // `DxContext::update_mesh_geometry`. Reached only through the bin's
    // `cn debug` runtime-mutation path (dead in the FFI lib, live in the bin).
    #[allow(dead_code)]
    pub fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let obj = self.draw_objects.get(draw_idx).ok_or_else(|| {
            format!(
                "update_mesh_geometry: draw object {} out of range",
                draw_idx
            )
        })?;
        if vertices.len() != obj.vertex_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} vertices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.index_count,
                indices.len()
            ));
        }
        if lod_alternates.len() != obj.lod_alternates.len() {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} LOD alternate(s), got {} \
                 (LOD-count changes need rebuild_static_geometry)",
                draw_idx,
                obj.lod_alternates.len(),
                lod_alternates.len()
            ));
        }
        for (lod_idx, ((_, alt_idx), slice)) in lod_alternates
            .iter()
            .zip(obj.lod_alternates.iter())
            .enumerate()
        {
            if alt_idx.len() != slice.index_count {
                return Err(format!(
                    "update_mesh_geometry: draw {} LOD{} expects {} indices, got {} \
                     (LOD size changes need rebuild_static_geometry)",
                    draw_idx,
                    lod_idx + 1,
                    slice.index_count,
                    alt_idx.len()
                ));
            }
        }

        let v_off = obj.vertex_offset as u64;
        let i_off_bytes = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        // Static draws keep indices absolute (base_vertex == 0), so rebase
        // mesh-relative u16 indices onto the slot's vertex_offset and widen
        // to u32 before writing, matching the shared u32 index buffer.
        let base = (obj.vertex_offset / std::mem::size_of::<Vertex>()) as u32;
        let lod_byte_offsets: Vec<u64> = obj
            .lod_alternates
            .iter()
            .map(|s| (s.index_offset * std::mem::size_of::<u32>()) as u64)
            .collect();

        self.wait_idle();

        let vert_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        self.write_geometry_region(self.geometry.vertex_buffer, v_off, vert_bytes)?;
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                rebased.as_ptr() as *const u8,
                std::mem::size_of_val(rebased.as_slice()),
            )
        };
        self.write_geometry_region(self.geometry.index_buffer, i_off_bytes, idx_bytes)?;
        // LOD alternates were laid out at init alongside LOD0 in the same
        // shared IB; each alternate shares LOD0's vertex region, so rebase
        // onto the same `base`.
        for ((_, alt_idx), &alt_off_bytes) in lod_alternates.iter().zip(lod_byte_offsets.iter()) {
            let alt_rebased: Vec<u32> = alt_idx.iter().map(|&i| u32::from(i) + base).collect();
            let alt_bytes = unsafe {
                std::slice::from_raw_parts(
                    alt_rebased.as_ptr() as *const u8,
                    std::mem::size_of_val(alt_rebased.as_slice()),
                )
            };
            self.write_geometry_region(self.geometry.index_buffer, alt_off_bytes, alt_bytes)?;
        }
        // Refresh per-LOD switch distances so JSON-side tweaks to
        // `lod_distances` propagate without a process restart.
        let slot = &mut self.draw_objects[draw_idx];
        for ((switch_distance, _), slice) in
            lod_alternates.iter().zip(slot.lod_alternates.iter_mut())
        {
            slice.switch_distance = *switch_distance;
        }
        Ok(())
    }

    // Return a streamed mesh's geometry region to the sub-allocators and mark
    // the draw non-resident so it is skipped in every pass.
    pub fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("evict_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.geometry
            .mesh_vtx_alloc
            .free(v_off, v_len, retire_frame);
        self.geometry
            .mesh_idx_alloc
            .free(i_off, i_len, retire_frame);
        self.draw_objects[draw_idx].resident = false;
        Ok(())
    }

    // Seed the streamed-mesh sub-allocators with the reserved headroom block
    // (byte ranges in the shared vertex / index buffers), for the
    // shrinkable-seed path.
    //
    // The streamed geometry is not baked into the buffers at build time;
    // instead the buffers carry one zeroed headroom region (sized to the
    // cap-many resident meshes) at these offsets, which `compact_for_streaming`
    // appended before init. `retire_frame 0`: nothing has been drawn yet, so
    // the space is allocatable immediately -- mirrors `setup_chunk_streaming`'s
    // seeding. From then on `upload_mesh` / `evict_mesh` place and free streamed
    // meshes within it. Mirrors `DxContext::seed_mesh_streaming`.
    pub fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        self.geometry.mesh_vtx_alloc.free(vtx_offset, vtx_bytes, 0);
        self.geometry.mesh_vtx_alloc.reclaim(0);
        self.geometry.mesh_idx_alloc.free(idx_offset, idx_bytes, 0);
        self.geometry.mesh_idx_alloc.reclaim(0);
    }
}
