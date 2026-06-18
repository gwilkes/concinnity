// src/vulkan/resources/geometry_rebuild.rs
//
// Hot-reload rebuild of the shared static + skinned vertex / index buffers
// when a re-imported `.glb` no longer fits in its init-time slot. Mirrors
// `directx/geometry_rebuild.rs` and `metal/resources/geometry.rs +
// metal/resources/skinning.rs`'s rebuild paths.
//
// Vulkan's DEVICE_LOCAL buffers are not host-visible, so the rebuild forces
// a CPU round-trip: a one-shot `cmd_copy_buffer` reads the live VB/IB into
// HOST_VISIBLE staging, the new contents are spliced CPU-side, fresh
// DEVICE_LOCAL buffers are allocated at the post-rebuild size, and the
// rebuilt data is uploaded via the existing `write_geometry_region` helper.
// Streamed-mesh sub-allocators (`mesh_vtx_alloc`, `mesh_idx_alloc`) are
// **not** preserved: the rebuilt buffer is sized exactly for the current
// draws, so any subsequent `upload_mesh` will fail allocation. `cn debug`-
// only by design; matches DirectX + Metal.

use std::collections::HashMap;

use ash::vk;

use crate::gfx::backend::{DrawGeometryUpdate, SkinnedDrawGeometryUpdate, SkinnedSlotLayout};
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::LodSlice;

use super::super::context::VkContext;
use super::super::texture::{create_buffer, one_shot_submit};

impl VkContext {
    // Rebuild the shared static-mesh vertex + index buffers, swapping in
    // fresh geometry for the draws named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a `Mesh` re-import has a different
    // vertex / index / LOD-alternate count than its init-time slot. Walks
    // every `DrawObject` in order: for each draw in `changes`, the new
    // vertices / indices / LOD alternates are appended to a fresh CPU
    // buffer; for unchanged draws, the current geometry is read back from
    // the live `vertex_buffer` / `index_buffer` and copied with index
    // rebasing. New DEVICE_LOCAL buffers are created at the post-rebuild
    // size, the contents uploaded through staging, and the old buffers /
    // memory dropped after the swap commits.
    //
    // Streamed-mesh sub-allocators are reset to empty since the rebuilt
    // buffer leaves no headroom; matches the DirectX + Metal scope split.
    // Mirrors `DxContext::rebuild_static_geometry`. Reached only through the
    // bin's `cn debug` runtime-mutation path (dead in the FFI lib, live in the
    // bin).
    #[allow(dead_code)]
    pub fn rebuild_static_geometry(
        &mut self,
        changes: Vec<DrawGeometryUpdate>,
    ) -> Result<(), String> {
        // Stop GPU + CPU pipelines so the readback + swap can run safely.
        self.wait_idle();

        let mut change_map: HashMap<usize, DrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.draw_idx, c)).collect();

        // Read the live VB / IB back to CPU memory through a HOST_VISIBLE
        // staging buffer (DEVICE_LOCAL is not host-mappable).
        let old_v_bytes = self.geometry.vertex_buffer_bytes;
        let old_i_bytes = self.geometry.index_buffer_bytes;
        let old_vertices: Vec<Vertex> =
            readback_typed(self, self.geometry.vertex_buffer, old_v_bytes)?;
        let old_indices: Vec<u32> = readback_typed(self, self.geometry.index_buffer, old_i_bytes)?;

        // Build rebuilt buffers + per-draw layouts CPU-side. Byte-for-byte
        // mirror of the DirectX walk; the read-only walk avoids aliasing
        // `&mut self` (it's applied after the loop).
        let mut new_vertices: Vec<Vertex> = Vec::new();
        let mut new_indices: Vec<u32> = Vec::new();
        type DrawLayout = (usize, usize, usize, usize, i32, Vec<LodSlice>);
        let mut new_layouts: Vec<DrawLayout> = Vec::with_capacity(self.draw_objects.len());

        for (draw_idx, obj) in self.draw_objects.iter().enumerate() {
            let new_v_byte_off = new_vertices.len() * std::mem::size_of::<Vertex>();
            let new_i_elem_off = new_indices.len();
            let new_base_u32 = new_vertices.len() as u32;
            // base_vertex == 0 means "absolute indices" (the static draws);
            // non-zero means "mesh-relative indices, GPU adds base_vertex
            // at fetch time" (voxel chunks). Preserve each draw's
            // semantics on rebuild.
            let absolute_indices = obj.base_vertex == 0;
            let new_base_vertex = if absolute_indices {
                0
            } else {
                (new_v_byte_off / std::mem::size_of::<Vertex>()) as i32
            };

            if let Some(change) = change_map.remove(&draw_idx) {
                new_vertices.extend_from_slice(&change.vertices);
                if absolute_indices {
                    new_indices.extend(change.indices.iter().map(|i| u32::from(*i) + new_base_u32));
                } else {
                    new_indices.extend(change.indices.iter().map(|i| u32::from(*i)));
                }
                let mut new_lods: Vec<LodSlice> = Vec::with_capacity(change.lod_alternates.len());
                for (switch_distance, alt_idx) in &change.lod_alternates {
                    let alt_off = new_indices.len();
                    if absolute_indices {
                        new_indices.extend(alt_idx.iter().map(|i| u32::from(*i) + new_base_u32));
                    } else {
                        new_indices.extend(alt_idx.iter().map(|i| u32::from(*i)));
                    }
                    new_lods.push(LodSlice {
                        index_offset: alt_off,
                        index_count: alt_idx.len(),
                        switch_distance: *switch_distance,
                    });
                }
                new_layouts.push((
                    new_v_byte_off,
                    change.vertices.len(),
                    new_i_elem_off,
                    change.indices.len(),
                    new_base_vertex,
                    new_lods,
                ));
            } else {
                // Unchanged draw: copy current geometry verbatim, rebasing
                // absolute indices onto the new vertex region.
                let v_start = obj.vertex_offset / std::mem::size_of::<Vertex>();
                let v_end = v_start + obj.vertex_count;
                if v_end > old_vertices.len() {
                    return Err(format!(
                        "rebuild_static_geometry: draw {} vertex region [{}, {}) out \
                         of bounds (buffer has {} vertices)",
                        draw_idx,
                        v_start,
                        v_end,
                        old_vertices.len()
                    ));
                }
                new_vertices.extend_from_slice(&old_vertices[v_start..v_end]);
                let old_base_u32 = if absolute_indices {
                    v_start as u32
                } else {
                    obj.base_vertex as u32
                };
                let i_end = obj.index_offset + obj.index_count;
                if i_end > old_indices.len() {
                    return Err(format!(
                        "rebuild_static_geometry: draw {} index region [{}, {}) out \
                         of bounds (buffer has {} indices)",
                        draw_idx,
                        obj.index_offset,
                        i_end,
                        old_indices.len()
                    ));
                }
                if absolute_indices {
                    for &idx in &old_indices[obj.index_offset..i_end] {
                        new_indices.push(idx.wrapping_sub(old_base_u32) + new_base_u32);
                    }
                } else {
                    new_indices.extend_from_slice(&old_indices[obj.index_offset..i_end]);
                }
                let mut new_lods: Vec<LodSlice> = Vec::with_capacity(obj.lod_alternates.len());
                for slice in &obj.lod_alternates {
                    let alt_end = slice.index_offset + slice.index_count;
                    if alt_end > old_indices.len() {
                        return Err(format!(
                            "rebuild_static_geometry: draw {} LOD slice [{}, {}) out \
                             of bounds (buffer has {} indices)",
                            draw_idx,
                            slice.index_offset,
                            alt_end,
                            old_indices.len()
                        ));
                    }
                    let alt_off = new_indices.len();
                    if absolute_indices {
                        for &idx in &old_indices[slice.index_offset..alt_end] {
                            new_indices.push(idx.wrapping_sub(old_base_u32) + new_base_u32);
                        }
                    } else {
                        new_indices.extend_from_slice(&old_indices[slice.index_offset..alt_end]);
                    }
                    new_lods.push(LodSlice {
                        index_offset: alt_off,
                        index_count: slice.index_count,
                        switch_distance: slice.switch_distance,
                    });
                }
                new_layouts.push((
                    new_v_byte_off,
                    obj.vertex_count,
                    new_i_elem_off,
                    obj.index_count,
                    new_base_vertex,
                    new_lods,
                ));
            }
        }

        if !change_map.is_empty() {
            tracing::warn!(
                "rebuild_static_geometry: {} change(s) targeted draw indices not in \
                 draw_objects (ignored)",
                change_map.len()
            );
        }

        if new_vertices.is_empty() || new_indices.is_empty() {
            return Err(
                "rebuild_static_geometry: post-rebuild buffers would be empty (no \
                 static draws to ship)"
                    .into(),
            );
        }

        // Allocate new DEVICE_LOCAL buffers + ship the rebuilt contents
        // through staging (write_geometry_region's one-shot pattern).
        let new_v_bytes = std::mem::size_of_val(new_vertices.as_slice()) as u64;
        let new_i_bytes = std::mem::size_of_val(new_indices.as_slice()) as u64;
        let (new_vbuf, new_vmem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            new_v_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let (new_ibuf, new_imem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            new_i_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let vert_bytes = unsafe {
            std::slice::from_raw_parts(new_vertices.as_ptr() as *const u8, new_v_bytes as usize)
        };
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(new_indices.as_ptr() as *const u8, new_i_bytes as usize)
        };
        self.write_geometry_region(new_vbuf, 0, vert_bytes)?;
        self.write_geometry_region(new_ibuf, 0, idx_bytes)?;

        // Commit the swap: destroy old buffers + memory, apply per-draw
        // layouts, and reset the streaming sub-allocators (the rebuilt
        // buffer leaves no headroom). `wait_idle` above gated every
        // in-flight read.
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
        self.geometry.vertex_buffer_bytes = new_v_bytes;
        self.geometry.index_buffer = new_ibuf;
        self.geometry.index_buffer_memory = new_imem;
        self.geometry.index_buffer_bytes = new_i_bytes;
        self.geometry.mesh_vtx_alloc = crate::gfx::range_alloc::RangeAllocator::new();
        self.geometry.mesh_idx_alloc = crate::gfx::range_alloc::RangeAllocator::new();
        for (i, (v_off, v_count, i_off, i_count, base_v, lods)) in
            new_layouts.into_iter().enumerate()
        {
            let obj = &mut self.draw_objects[i];
            obj.vertex_offset = v_off;
            obj.vertex_count = v_count;
            obj.index_offset = i_off;
            obj.index_count = i_count;
            obj.base_vertex = base_v;
            obj.lod_alternates = lods;
        }
        Ok(())
    }

    // Rebuild the shared skinned-mesh vertex + index buffers, swapping in
    // fresh geometry for the slots named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a `SkinnedMesh` re-import has a
    // different vertex / index count than its init-time slot. Walks every
    // `SkinnedDrawObject` in order: for each slot in `changes`, the new
    // vertices / indices are appended to a fresh CPU buffer; for unchanged
    // slots, the current geometry is read back from the live skinned
    // buffers and copied with index rebasing from the slot's old
    // `vertex_base` onto its new one. New DEVICE_LOCAL buffers are created
    // at the post-rebuild size, the contents uploaded through staging, and
    // the old buffers / memory dropped after the swap commits. Returns a
    // `SkinnedSlotLayout` per slot (in `skinned_index` order) so the
    // asset-hot-reload caller can refresh its `SkinnedMeshSourceEntry`s.
    //
    // The skinned IB stays `R16_UINT`; the skinned pipelines, shadow /
    // SSAO / SSR variants, and per-slot metadata (`texture_slot` /
    // `normal_map_slot` / `material` / `joint_count`) are untouched.
    // Skeleton-shape changes route through `update_skinned_skeleton`, not
    // this call. Mirrors `DxContext::rebuild_skinned_geometry`. Reached only
    // through the bin's `cn debug` runtime-mutation path (dead in the FFI lib,
    // live in the bin).
    #[allow(dead_code)]
    pub fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<SkinnedSlotLayout>, String> {
        if self.skinned.vertex_buffer == vk::Buffer::null()
            || self.skinned.index_buffer == vk::Buffer::null()
        {
            return Err(
                "rebuild_skinned_geometry: no skinned vertex/index buffer (was \
                 upload_skinned called?)"
                    .into(),
            );
        }

        self.wait_idle();

        let mut change_map: HashMap<usize, SkinnedDrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.skinned_index, c)).collect();

        // Read back the live skinned buffers via HOST_VISIBLE staging.
        let old_v_bytes = self.skinned.vertex_buffer_bytes;
        let old_i_bytes = self.skinned.index_buffer_bytes;
        let old_vertices: Vec<SkinnedVertex> =
            readback_typed(self, self.skinned.vertex_buffer, old_v_bytes)?;
        let old_indices: Vec<u16> = readback_typed(self, self.skinned.index_buffer, old_i_bytes)?;

        let mut new_vertices: Vec<SkinnedVertex> = Vec::new();
        let mut new_indices: Vec<u16> = Vec::new();
        let mut layouts: Vec<SkinnedSlotLayout> =
            Vec::with_capacity(self.skinned.draw_objects.len());
        // Captured per-slot new layout (applied to `skinned_draw_objects`
        // after the read-only walk to avoid aliasing `self`).
        let mut new_per_slot: Vec<(usize, u16, usize, usize, usize)> =
            Vec::with_capacity(self.skinned.draw_objects.len());

        for (skinned_index, obj) in self.skinned.draw_objects.iter().enumerate() {
            let new_v_base_usize = new_vertices.len();
            let new_v_base: u16 = u16::try_from(new_v_base_usize).map_err(|_| {
                format!(
                    "rebuild_skinned_geometry: post-rebuild vertex base {} for slot \
                     {} overflows u16 (skinned IB is u16)",
                    new_v_base_usize, skinned_index
                )
            })?;
            let new_i_off = new_indices.len();

            if let Some(change) = change_map.remove(&skinned_index) {
                let new_v_count = change.vertices.len();
                let new_i_count = change.indices.len();
                new_vertices.extend_from_slice(&change.vertices);
                for &local in &change.indices {
                    let absolute = local.checked_add(new_v_base).ok_or_else(|| {
                        format!(
                            "rebuild_skinned_geometry: index rebase by {} for slot \
                             {} overflows u16",
                            new_v_base, skinned_index
                        )
                    })?;
                    new_indices.push(absolute);
                }
                layouts.push(SkinnedSlotLayout {
                    skinned_index,
                    vertex_base: new_v_base,
                    vertex_count: new_v_count,
                    index_count: new_i_count,
                });
                new_per_slot.push((
                    skinned_index,
                    new_v_base,
                    new_v_count,
                    new_i_off,
                    new_i_count,
                ));
            } else {
                // Unchanged slot: copy current geometry verbatim, rebasing
                // its absolute indices from old vertex_base onto the new
                // one.
                let v_start = obj.vertex_base as usize;
                let v_end = v_start + obj.vertex_count;
                if v_end > old_vertices.len() {
                    return Err(format!(
                        "rebuild_skinned_geometry: slot {} vertex region [{}, {}) \
                         out of bounds (buffer has {} vertices)",
                        skinned_index,
                        v_start,
                        v_end,
                        old_vertices.len()
                    ));
                }
                new_vertices.extend_from_slice(&old_vertices[v_start..v_end]);
                let i_end = obj.index_offset + obj.index_count;
                if i_end > old_indices.len() {
                    return Err(format!(
                        "rebuild_skinned_geometry: slot {} index region [{}, {}) \
                         out of bounds (buffer has {} indices)",
                        skinned_index,
                        obj.index_offset,
                        i_end,
                        old_indices.len()
                    ));
                }
                let old_base = obj.vertex_base;
                for &abs in &old_indices[obj.index_offset..i_end] {
                    let local = abs.checked_sub(old_base).ok_or_else(|| {
                        format!(
                            "rebuild_skinned_geometry: stale index {} below \
                             vertex_base {} on slot {}",
                            abs, old_base, skinned_index
                        )
                    })?;
                    let absolute = local.checked_add(new_v_base).ok_or_else(|| {
                        format!(
                            "rebuild_skinned_geometry: rebasing index {} onto \
                             vertex_base {} overflows u16 on slot {}",
                            local, new_v_base, skinned_index
                        )
                    })?;
                    new_indices.push(absolute);
                }
                layouts.push(SkinnedSlotLayout {
                    skinned_index,
                    vertex_base: new_v_base,
                    vertex_count: obj.vertex_count,
                    index_count: obj.index_count,
                });
                new_per_slot.push((
                    skinned_index,
                    new_v_base,
                    obj.vertex_count,
                    new_i_off,
                    obj.index_count,
                ));
            }
        }

        if !change_map.is_empty() {
            tracing::warn!(
                "rebuild_skinned_geometry: {} change(s) targeted skinned indices not \
                 in skinned_draw_objects (ignored)",
                change_map.len()
            );
        }

        if new_vertices.is_empty() || new_indices.is_empty() {
            return Err(
                "rebuild_skinned_geometry: post-rebuild buffers would be empty (no \
                 skinned draws to ship)"
                    .into(),
            );
        }

        // Allocate new DEVICE_LOCAL skinned buffers + ship through staging. When
        // the device is RT-capable the new buffers must carry the same RT usage
        // flags `upload_skinned` added, or the RT skinning path loses its inputs
        // after a size-changing skinned reload (the flags ride along whenever
        // capable so a live RT toggle keeps working across reloads).
        let new_v_bytes = std::mem::size_of_val(new_vertices.as_slice()) as u64;
        let new_i_bytes = std::mem::size_of_val(new_indices.as_slice()) as u64;
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
        let (new_vbuf, new_vmem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            new_v_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | skinned_vb_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let (new_ibuf, new_imem) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            new_i_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | skinned_ib_rt,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let vert_bytes = unsafe {
            std::slice::from_raw_parts(new_vertices.as_ptr() as *const u8, new_v_bytes as usize)
        };
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(new_indices.as_ptr() as *const u8, new_i_bytes as usize)
        };
        self.write_geometry_region(new_vbuf, 0, vert_bytes)?;
        self.write_geometry_region(new_ibuf, 0, idx_bytes)?;

        // Commit: destroy old buffers, apply per-slot layouts.
        unsafe {
            self.device.destroy_buffer(self.skinned.vertex_buffer, None);
            self.device
                .free_memory(self.skinned.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.skinned.index_buffer, None);
            self.device
                .free_memory(self.skinned.index_buffer_memory, None);
        }
        self.skinned.vertex_buffer = new_vbuf;
        self.skinned.vertex_buffer_memory = new_vmem;
        self.skinned.vertex_buffer_bytes = new_v_bytes;
        self.skinned.index_buffer = new_ibuf;
        self.skinned.index_buffer_memory = new_imem;
        self.skinned.index_buffer_bytes = new_i_bytes;
        for (skinned_index, v_base, v_count, i_off, i_count) in new_per_slot {
            let obj = &mut self.skinned.draw_objects[skinned_index];
            obj.vertex_base = v_base;
            obj.vertex_count = v_count;
            obj.index_offset = i_off;
            obj.index_count = i_count;
        }
        Ok(layouts)
    }
}

// Read a DEVICE_LOCAL buffer's full contents back to CPU memory as a typed
// `Vec<T>`. Allocates a HOST_VISIBLE staging buffer, runs a one-shot
// `cmd_copy_buffer` from `src` into it (`wait_idle` already gated the source
// side; the one-shot's internal fence wait gates the destination), maps,
// and `copy_nonoverlapping`s into the Vec. `T`'s stride must match the
// buffer's stride exactly. Only reached through the (bin-only) geometry-rebuild
// path, so dead in the FFI lib.
#[allow(dead_code)]
fn readback_typed<T: Copy>(ctx: &VkContext, src: vk::Buffer, bytes: u64) -> Result<Vec<T>, String> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    let stride = std::mem::size_of::<T>() as u64;
    if !bytes.is_multiple_of(stride) {
        return Err(format!(
            "readback_typed: buffer size {} not a multiple of T stride {}",
            bytes, stride
        ));
    }
    let count = (bytes / stride) as usize;
    let (staging, staging_mem) = create_buffer(
        &ctx.instance,
        &ctx.device,
        ctx.physical_device,
        bytes,
        vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let result = one_shot_submit(
        &ctx.device,
        ctx.commands.command_pool,
        ctx.graphics_queue,
        |cmd| {
            let copy = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(bytes);
            unsafe {
                ctx.device
                    .cmd_copy_buffer(cmd, src, staging, std::slice::from_ref(&copy))
            };
        },
    );
    if let Err(e) = result {
        unsafe {
            ctx.device.destroy_buffer(staging, None);
            ctx.device.free_memory(staging_mem, None);
        }
        return Err(e);
    }

    let mut out: Vec<T> = Vec::with_capacity(count);
    unsafe {
        let ptr = match ctx
            .device
            .map_memory(staging_mem, 0, bytes, vk::MemoryMapFlags::empty())
        {
            Ok(ptr) => ptr as *const T,
            Err(e) => {
                ctx.device.destroy_buffer(staging, None);
                ctx.device.free_memory(staging_mem, None);
                return Err(format!("readback_typed map: {e}"));
            }
        };
        std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), count);
        out.set_len(count);
        ctx.device.unmap_memory(staging_mem);
        ctx.device.destroy_buffer(staging, None);
        ctx.device.free_memory(staging_mem, None);
    }
    Ok(out)
}
