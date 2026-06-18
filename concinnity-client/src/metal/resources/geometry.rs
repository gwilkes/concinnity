// src/metal/resources/geometry.rs
//
// Hot-reload rebuild of the shared static-mesh vertex + index buffers when
// re-imported `.glb` source no longer fits each draw's init-time slot.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2_metal::{MTLBuffer as _, MTLDevice as _, MTLResourceOptions};

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::LodSlice;
use crate::metal::context::MtlContext;

impl MtlContext {
    // Rebuild the shared static-mesh vertex + index buffers, swapping in
    // new geometry for the draws named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a Mesh's re-imported `.glb` no
    // longer fits in its init-time slot. Walks every `DrawObject` in
    // order: for each draw in `changes`, the new vertices / indices /
    // LOD alternates are appended to a fresh CPU buffer; for unchanged
    // draws, the current geometry is read back from the live
    // `vertex_buffer` / `index_buffer` (both `StorageModeShared` so the
    // pointers are CPU-readable) and copied with index rebasing. New
    // `MTLBuffer`s are created at the post-rebuild size and swapped in
    // after `wait_idle` so no in-flight command buffer touches the old
    // resource pair. Streaming sub-allocators (`mesh_vtx_alloc`,
    // `mesh_idx_alloc`) are reset to the new buffer's full extent -- any
    // streaming uploads in flight would be invalidated by the swap, so
    // the caller is expected to gate this on a quiet renderer (the
    // asset-hot-reload path already is -- it runs at frame start before
    // the streaming poll).
    pub fn rebuild_static_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::DrawGeometryUpdate>,
    ) -> Result<(), String> {
        use std::collections::HashMap;

        // Stop the GPU + CPU pipelines so we can safely read the old
        // buffers and atomically swap. Costs a frame-time stall but only
        // fires under `cn debug` and only when the source `.glb` size
        // actually changed.
        self.wait_idle();

        let mut change_map: HashMap<usize, crate::gfx::backend::DrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.draw_idx, c)).collect();

        // Read views over the current shared buffers. `StorageModeShared`
        // means `contents()` is a CPU-addressable pointer aliasing the GPU
        // data; safe after `wait_idle`.
        let old_v_len = self.vertex_buffer.length() / std::mem::size_of::<Vertex>();
        let old_v_slice: &[Vertex] = unsafe {
            let ptr = self.vertex_buffer.contents().as_ptr() as *const Vertex;
            std::slice::from_raw_parts(ptr, old_v_len)
        };
        let old_i_len = self.index_buffer.length() / std::mem::size_of::<u32>();
        let old_i_slice: &[u32] = unsafe {
            let ptr = self.index_buffer.contents().as_ptr() as *const u32;
            std::slice::from_raw_parts(ptr, old_i_len)
        };

        let mut new_vertices: Vec<Vertex> = Vec::new();
        let mut new_indices: Vec<u32> = Vec::new();
        // Captured per-draw new layout (applied to `draw_objects` after the
        // read-only walk to avoid aliasing `self`).
        type DrawLayout = (usize, usize, usize, usize, i32, Vec<LodSlice>);
        let mut new_layouts: Vec<DrawLayout> = Vec::with_capacity(self.draw_objects.len());

        for (draw_idx, obj) in self.draw_objects.iter().enumerate() {
            let new_v_byte_off = new_vertices.len() * std::mem::size_of::<Vertex>();
            let new_i_elem_off = new_indices.len();
            let new_base_u32 = new_vertices.len() as u32;
            // base_vertex == 0 means "absolute indices" (the static draws);
            // non-zero means "mesh-relative indices, GPU adds base_vertex
            // at fetch time" (voxel chunks). We preserve each draw's
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
                // Unchanged draw -- copy current geometry verbatim, rebasing
                // indices onto the new vertex region.
                let v_start = obj.vertex_offset / std::mem::size_of::<Vertex>();
                let v_end = v_start + obj.vertex_count;
                if v_end > old_v_slice.len() {
                    return Err(format!(
                        "rebuild_static_geometry: draw {} vertex region [{}, {}) out \
                         of bounds (buffer has {} vertices)",
                        draw_idx,
                        v_start,
                        v_end,
                        old_v_slice.len()
                    ));
                }
                new_vertices.extend_from_slice(&old_v_slice[v_start..v_end]);
                let old_base_u32 = if absolute_indices {
                    v_start as u32
                } else {
                    obj.base_vertex as u32
                };
                let i_end = obj.index_offset + obj.index_count;
                if i_end > old_i_slice.len() {
                    return Err(format!(
                        "rebuild_static_geometry: draw {} index region [{}, {}) out \
                         of bounds (buffer has {} indices)",
                        draw_idx,
                        obj.index_offset,
                        i_end,
                        old_i_slice.len()
                    ));
                }
                if absolute_indices {
                    for &idx in &old_i_slice[obj.index_offset..i_end] {
                        new_indices.push(idx.wrapping_sub(old_base_u32) + new_base_u32);
                    }
                } else {
                    new_indices.extend_from_slice(&old_i_slice[obj.index_offset..i_end]);
                }
                let mut new_lods: Vec<LodSlice> = Vec::with_capacity(obj.lod_alternates.len());
                for slice in &obj.lod_alternates {
                    let alt_end = slice.index_offset + slice.index_count;
                    if alt_end > old_i_slice.len() {
                        return Err(format!(
                            "rebuild_static_geometry: draw {} LOD slice [{}, {}) out \
                             of bounds (buffer has {} indices)",
                            draw_idx,
                            slice.index_offset,
                            alt_end,
                            old_i_slice.len()
                        ));
                    }
                    let alt_off = new_indices.len();
                    if absolute_indices {
                        for &idx in &old_i_slice[slice.index_offset..alt_end] {
                            new_indices.push(idx.wrapping_sub(old_base_u32) + new_base_u32);
                        }
                    } else {
                        new_indices.extend_from_slice(&old_i_slice[slice.index_offset..alt_end]);
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

        // Create new MTL buffers sized to the rebuilt layout.
        let new_vertex_buffer = unsafe {
            let v_bytes = std::mem::size_of_val(new_vertices.as_slice());
            let ptr = std::ptr::NonNull::new(new_vertices.as_ptr() as *mut _)
                .ok_or("rebuild_static_geometry: vertex slice pointer is null")?;
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    v_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("rebuild_static_geometry: failed to create new vertex buffer")?
        };
        let new_index_buffer = unsafe {
            let i_bytes = std::mem::size_of_val(new_indices.as_slice());
            let ptr = std::ptr::NonNull::new(new_indices.as_ptr() as *mut _)
                .ok_or("rebuild_static_geometry: index slice pointer is null")?;
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    i_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("rebuild_static_geometry: failed to create new index buffer")?
        };

        // Apply the new per-draw layout.
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

        self.vertex_buffer = new_vertex_buffer;
        self.index_buffer = new_index_buffer;
        Ok(())
    }
}
