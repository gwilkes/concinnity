// src/directx/geometry_rebuild.rs
//
// Hot-reload rebuild of the shared static + skinned vertex / index buffers
// when a re-imported `.glb` no longer fits in its init-time slot. Mirrors
// metal/resources/geometry.rs + metal/resources/skinning.rs's
// `rebuild_skinned_geometry`; the DEFAULT-heap buffers force a CPU
// round-trip (READBACK staging for the old contents, UPLOAD staging for
// the new ones) where Metal can just read `StorageModeShared` `contents()`.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT};

use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::LodSlice;

use super::context::DxContext;
use super::texture::{create_buffer, one_shot_submit, transition_barrier};

// cn-debug-only asset hot-reload geometry rebuild; dead from the FFI lib
// crate's roots, live in the concinnity binary. The two module-level helper fns
// below (`read_typed_vec` / `write_upload_buffer`) are walked through these
// suppressed-but-live-root methods, so they need no attribute of their own.
// See the note on the analogous block in [directx/decal.rs].
#[allow(
    dead_code,
    reason = "cn-debug-only hot-reload surface; dead from the FFI lib crate's roots, live in the concinnity binary"
)]
impl DxContext {
    // Rebuild the shared static-mesh vertex + index buffers, swapping in
    // fresh geometry for the draws named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a `Mesh` re-import has a different
    // vertex / index / LOD-alternate count than its init-time slot. Walks
    // every `DrawObject` in order: for each draw in `changes`, the new
    // vertices / indices / LOD alternates are appended to a fresh CPU
    // buffer; for unchanged draws, the current geometry is read back from
    // the live `vertex_buffer` / `index_buffer` (DEFAULT heap → READBACK
    // staging buffer + one-shot GPU copy + Map) and copied with index
    // rebasing. New DEFAULT-heap buffers are created at the post-rebuild
    // size and the contents uploaded through UPLOAD-heap staging; the old
    // buffers are dropped only after the swap commits.
    //
    // Streamed-mesh sub-allocators (`mesh_vtx_alloc`, `mesh_idx_alloc`) are
    // **not** preserved; the rebuilt buffer is sized exactly for the
    // current draws, so any subsequent `upload_mesh` will fail allocation.
    // `cn debug`-only assumption: the caller (asset hot-reload) runs this
    // at frame start, when the renderer is otherwise quiet.
    pub fn rebuild_static_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::DrawGeometryUpdate>,
    ) -> Result<(), String> {
        use std::collections::HashMap;

        // Stop the GPU + CPU pipelines so the readback + swap can run safely.
        // Costs a frame-time stall but only fires under `cn debug` when the
        // source `.glb` size actually changed.
        self.wait_idle();

        let mut change_map: HashMap<usize, crate::gfx::backend::DrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.draw_idx, c)).collect();

        // Read the current shared buffers back to CPU memory via READBACK
        // staging (DEFAULT-heap resources are not CPU-mappable). One one-shot
        // submit transitions both buffers to COPY_SOURCE, copies each to its
        // matching staging buffer, then transitions back; `wait_idle` above
        // gates the source side; the one-shot's internal fence wait gates
        // the destination.
        let old_v_bytes = self.geometry.vertex_buffer_view.SizeInBytes as u64;
        let old_i_bytes = self.geometry.index_buffer_view.SizeInBytes as u64;
        let v_readback = create_buffer(
            &self.device,
            old_v_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let i_readback = create_buffer(
            &self.device,
            old_i_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let v_src = transition_barrier(
                &self.geometry.vertex_buffer,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[v_src, i_src]);
            cmd.CopyBufferRegion(&v_readback, 0, &self.geometry.vertex_buffer, 0, old_v_bytes);
            cmd.CopyBufferRegion(&i_readback, 0, &self.geometry.index_buffer, 0, old_i_bytes);
            let v_back = transition_barrier(
                &self.geometry.vertex_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_back = transition_barrier(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_back, i_back]);
        })?;

        // Map the readback staging buffers and copy into typed CPU Vecs.
        // The static IB is `u32` (set when the buffer was created in
        // init/mod.rs); the VB stride is `size_of::<Vertex>()`.
        let old_v_count = (old_v_bytes as usize) / std::mem::size_of::<Vertex>();
        let old_i_count = (old_i_bytes as usize) / std::mem::size_of::<u32>();
        let old_vertices: Vec<Vertex> = read_typed_vec(&v_readback, old_v_count)?;
        let old_indices: Vec<u32> = read_typed_vec(&i_readback, old_i_count)?;

        // Build the rebuilt buffers + per-draw layout CPU-side. Mirrors the
        // Metal walk byte-for-byte except for the readback source.
        let mut new_vertices: Vec<Vertex> = Vec::new();
        let mut new_indices: Vec<u32> = Vec::new();
        type DrawLayout = (usize, usize, usize, usize, i32, Vec<LodSlice>);
        let mut new_layouts: Vec<DrawLayout> = Vec::with_capacity(self.draw_objects.len());

        for (draw_idx, obj) in self.draw_objects.iter().enumerate() {
            let new_v_byte_off = new_vertices.len() * std::mem::size_of::<Vertex>();
            let new_i_elem_off = new_indices.len();
            let new_base_u32 = new_vertices.len() as u32;
            // base_vertex == 0 means "absolute indices" (the static draws);
            // non-zero means "mesh-relative indices, GPU adds base_vertex at
            // fetch time" (voxel chunks). Preserve each draw's semantics on
            // rebuild.
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

        // Allocate the new DEFAULT-heap buffers + UPLOAD-heap staging copies
        // and ship the rebuilt contents in a single one-shot submit. The new
        // resources start in COMMON; CopyBufferRegion promotes them to
        // COPY_DEST implicitly, then a barrier puts each into its read state.
        let new_v_bytes = std::mem::size_of_val(new_vertices.as_slice()) as u64;
        let new_i_bytes = std::mem::size_of_val(new_indices.as_slice()) as u64;
        let new_vbuf = create_buffer(
            &self.device,
            new_v_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let new_ibuf = create_buffer(
            &self.device,
            new_i_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let v_upload = create_buffer(
            &self.device,
            new_v_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let i_upload = create_buffer(
            &self.device,
            new_i_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        write_upload_buffer(&v_upload, unsafe {
            std::slice::from_raw_parts(new_vertices.as_ptr() as *const u8, new_v_bytes as usize)
        })?;
        write_upload_buffer(&i_upload, unsafe {
            std::slice::from_raw_parts(new_indices.as_ptr() as *const u8, new_i_bytes as usize)
        })?;
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            cmd.CopyBufferRegion(&new_vbuf, 0, &v_upload, 0, new_v_bytes);
            cmd.CopyBufferRegion(&new_ibuf, 0, &i_upload, 0, new_i_bytes);
            let v_dst = transition_barrier(
                &new_vbuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_dst = transition_barrier(
                &new_ibuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_dst, i_dst]);
        })?;

        // Commit the swap: rewrite per-draw layouts, point the live views at
        // the new buffers, then drop the old buffer references. After this
        // line the old `vertex_buffer` / `index_buffer` resources are
        // unreachable; the next-frame fence wait has already happened, so
        // the COM refcount drop is safe.
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
        self.geometry.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { new_vbuf.GetGPUVirtualAddress() },
            SizeInBytes: new_v_bytes as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        self.geometry.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { new_ibuf.GetGPUVirtualAddress() },
            SizeInBytes: new_i_bytes as u32,
            Format: DXGI_FORMAT_R32_UINT,
        };
        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.index_buffer = new_ibuf;
        Ok(())
    }

    // Rebuild the shared skinned-mesh vertex + index buffers, swapping in
    // fresh geometry for the slots named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a `SkinnedMesh` re-import has a
    // different vertex / index count than its init-time slot. Walks every
    // `SkinnedDrawObject` in order: for each slot in `changes`, the new
    // vertices / indices are appended to a fresh CPU buffer; for unchanged
    // slots, the current geometry is read back from the live skinned
    // buffers (DEFAULT heap → READBACK staging + one-shot GPU copy + Map)
    // and copied with index rebasing from the slot's old `vertex_base`
    // onto its new one. New DEFAULT-heap buffers are created at the
    // post-rebuild size and the contents uploaded through UPLOAD staging;
    // the old buffers are dropped only after the swap commits. Returns a
    // `SkinnedSlotLayout` per slot (in `skinned_index` order) so the
    // asset-hot-reload caller can refresh its `SkinnedMeshSourceEntry`s.
    //
    // The skinned IB stays `DXGI_FORMAT_R16_UINT`; the skinned pipelines,
    // shadow / velocity / SSAO / SSR variants, and per-slot metadata
    // (`texture_slot` / `normal_map_slot` / `material` / `joint_count`)
    // are untouched. Skeleton-shape changes (joint-count mismatch) route
    // through `update_skinned_skeleton`, not this call.
    pub fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<crate::gfx::backend::SkinnedSlotLayout>, String> {
        use std::collections::HashMap;

        let v_buf = self.skinned.vertex_buffer.as_ref().cloned().ok_or(
            "rebuild_skinned_geometry: no skinned vertex buffer (was upload_skinned called?)",
        )?;
        let i_buf = self.skinned.index_buffer.as_ref().cloned().ok_or(
            "rebuild_skinned_geometry: no skinned index buffer (was upload_skinned called?)",
        )?;

        self.wait_idle();

        let mut change_map: HashMap<usize, crate::gfx::backend::SkinnedDrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.skinned_index, c)).collect();

        // Read the live skinned buffers back to CPU memory via READBACK
        // staging. Same one-shot pattern as `rebuild_static_geometry`.
        let old_v_bytes = self.skinned.vertex_buffer_view.SizeInBytes as u64;
        let old_i_bytes = self.skinned.index_buffer_view.SizeInBytes as u64;
        let v_readback = create_buffer(
            &self.device,
            old_v_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let i_readback = create_buffer(
            &self.device,
            old_i_bytes,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let v_src = transition_barrier(
                &v_buf,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                &i_buf,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[v_src, i_src]);
            cmd.CopyBufferRegion(&v_readback, 0, &v_buf, 0, old_v_bytes);
            cmd.CopyBufferRegion(&i_readback, 0, &i_buf, 0, old_i_bytes);
            let v_back = transition_barrier(
                &v_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_back = transition_barrier(
                &i_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_back, i_back]);
        })?;

        // Skinned VB stride is `size_of::<SkinnedVertex>()`; skinned IB is
        // `u16` (matches the format set in `upload_skinned`).
        let old_v_count = (old_v_bytes as usize) / std::mem::size_of::<SkinnedVertex>();
        let old_i_count = (old_i_bytes as usize) / std::mem::size_of::<u16>();
        let old_vertices: Vec<SkinnedVertex> = read_typed_vec(&v_readback, old_v_count)?;
        let old_indices: Vec<u16> = read_typed_vec(&i_readback, old_i_count)?;

        // Walk every skinned slot, appending new or unchanged-and-rebased
        // geometry into the fresh CPU buffers.
        let mut new_vertices: Vec<SkinnedVertex> = Vec::new();
        let mut new_indices: Vec<u16> = Vec::new();
        let mut layouts: Vec<crate::gfx::backend::SkinnedSlotLayout> =
            Vec::with_capacity(self.skinned.draw_objects.len());
        // Captured per-slot new layout (applied to `skinned_draw_objects`
        // after the read-only walk to avoid aliasing `self`).
        let mut new_per_slot: Vec<(usize, u16, usize, usize, usize)> =
            Vec::with_capacity(self.skinned.draw_objects.len());

        for (skinned_index, obj) in self.skinned.draw_objects.iter().enumerate() {
            let new_v_base_usize = new_vertices.len();
            let new_v_base: u16 = match u16::try_from(new_v_base_usize) {
                Ok(v) => v,
                Err(_) => {
                    return Err(format!(
                        "rebuild_skinned_geometry: post-rebuild vertex base {} for \
                         slot {} overflows u16 (skinned IB is u16)",
                        new_v_base_usize, skinned_index
                    ));
                }
            };
            let new_i_off = new_indices.len();

            if let Some(change) = change_map.remove(&skinned_index) {
                let new_v_count = change.vertices.len();
                let new_i_count = change.indices.len();
                // Each mesh-relative index must stay in u16 after rebase.
                let last_base_for_overflow = new_v_count
                    .checked_sub(1)
                    .and_then(|max_local| u16::try_from(max_local).ok())
                    .unwrap_or(0);
                if new_v_base.checked_add(last_base_for_overflow).is_none() {
                    return Err(format!(
                        "rebuild_skinned_geometry: vertex region for slot {} \
                         would push max absolute index past u16",
                        skinned_index
                    ));
                }
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
                layouts.push(crate::gfx::backend::SkinnedSlotLayout {
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
                // its absolute indices from the old vertex_base onto the new
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
                // `idx - old_base + new_v_base`: both subtraction and
                // addition are bounded by the slot's vertex_count, which we
                // just placed at new_v_base.
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
                layouts.push(crate::gfx::backend::SkinnedSlotLayout {
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

        // Allocate new DEFAULT-heap buffers + UPLOAD staging copies and ship
        // the rebuilt contents in a single one-shot submit.
        let new_v_bytes = std::mem::size_of_val(new_vertices.as_slice()) as u64;
        let new_i_bytes = std::mem::size_of_val(new_indices.as_slice()) as u64;
        let new_vbuf = create_buffer(
            &self.device,
            new_v_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let new_ibuf = create_buffer(
            &self.device,
            new_i_bytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let v_upload = create_buffer(
            &self.device,
            new_v_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let i_upload = create_buffer(
            &self.device,
            new_i_bytes,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        write_upload_buffer(&v_upload, unsafe {
            std::slice::from_raw_parts(new_vertices.as_ptr() as *const u8, new_v_bytes as usize)
        })?;
        write_upload_buffer(&i_upload, unsafe {
            std::slice::from_raw_parts(new_indices.as_ptr() as *const u8, new_i_bytes as usize)
        })?;
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            cmd.CopyBufferRegion(&new_vbuf, 0, &v_upload, 0, new_v_bytes);
            cmd.CopyBufferRegion(&new_ibuf, 0, &i_upload, 0, new_i_bytes);
            let v_dst = transition_barrier(
                &new_vbuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_dst = transition_barrier(
                &new_ibuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_dst, i_dst]);
        })?;

        // Commit: rewrite per-slot layouts, repoint the live views at the
        // new buffers, and drop the old buffer COM references.
        for (skinned_index, v_base, v_count, i_off, i_count) in new_per_slot {
            let obj = &mut self.skinned.draw_objects[skinned_index];
            obj.vertex_base = v_base;
            obj.vertex_count = v_count;
            obj.index_offset = i_off;
            obj.index_count = i_count;
        }
        self.skinned.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { new_vbuf.GetGPUVirtualAddress() },
            SizeInBytes: new_v_bytes as u32,
            StrideInBytes: std::mem::size_of::<SkinnedVertex>() as u32,
        };
        self.skinned.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { new_ibuf.GetGPUVirtualAddress() },
            SizeInBytes: new_i_bytes as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };
        self.skinned.vertex_buffer = Some(new_vbuf);
        self.skinned.index_buffer = Some(new_ibuf);
        Ok(layouts)
    }
}

// Map a READBACK-heap buffer and copy `count` `T`s out into a CPU Vec. The
// caller has already gated the GPU writes (via `one_shot_submit`'s internal
// fence wait), so the memcpy sees fully committed bytes. `T` must match the
// buffer's stride exactly.
fn read_typed_vec<T: Copy>(src: &ID3D12Resource, count: usize) -> Result<Vec<T>, String> {
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { src.Map(0, None, Some(&mut ptr)) }
        .map_err(|e| format!("rebuild readback map: {e}"))?;
    let mut out: Vec<T> = Vec::with_capacity(count);
    unsafe {
        std::ptr::copy_nonoverlapping(ptr as *const T, out.as_mut_ptr(), count);
        out.set_len(count);
        src.Unmap(0, None);
    }
    Ok(out)
}

// Map an UPLOAD-heap buffer and copy `bytes` into it. Standard UPLOAD-heap
// idiom: Map (CPU writes), copy_nonoverlapping, Unmap (driver flushes).
fn write_upload_buffer(dest: &ID3D12Resource, bytes: &[u8]) -> Result<(), String> {
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { dest.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("rebuild upload map: {e}"))?;
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
        dest.Unmap(0, None);
    }
    Ok(())
}
