// src/metal/resources/streaming.rs
//
// Per-mesh upload / eviction into the shared static-mesh vertex + index
// buffers via the sub-allocators, plus in-place per-slot updates for asset
// hot-reload.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use crate::gfx::mesh_payload::Vertex;
use crate::metal::context::{MtlContext, bytes_of_slice, write_buffer_region, zero_buffer_region};

impl MtlContext {
    // Upload a streamed mesh's geometry into the shared vertex and index
    // buffers, place it via the sub-allocators, and mark the draw resident.
    //
    // The mesh-streaming subsystem calls this to bring a mesh resident after
    // init. The geometry is placed wherever the allocators find free space
    // (not the build-time region), so `DrawObject::vertex_offset` /
    // `index_offset` are rewritten here. `vertices` / `indices` must match the
    // fixed `vertex_count` / `index_count` recorded by `build_draw_list`.
    //
    // `indices` are mesh-relative (0-based); they are rebased onto the chosen
    // vertex region before upload. `frame` is the current frame: deferred
    // frees that have retired by then are reclaimed first, so freed space
    // becomes reusable. The chosen region was not drawn while the mesh was
    // non-resident, so no in-flight command buffer reads it -- the write is
    // race-free.
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
        if vertices.len() != obj.vertex_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} vertices, got {}",
                draw_idx,
                obj.vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != obj.index_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} indices, got {}",
                draw_idx,
                obj.index_count,
                indices.len()
            ));
        }

        // Reclaim frees whose in-flight frames have retired, then place the
        // geometry. A zero-length mesh would not occupy the buffers, but
        // build_draw_list never emits one, so treat it as a hard error.
        self.mesh_vtx_alloc.reclaim(frame);
        self.mesh_idx_alloc.reclaim(frame);
        let v_len = std::mem::size_of_val(vertices);
        // The shared index buffer is u32-typed; the input `indices` are u16 and
        // get widened on write below, so size the allocation against the u32
        // stride. Sizing against the u16 source would alloc half the bytes the
        // write needs and corrupt whatever sub-allocation followed.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self.mesh_vtx_alloc.alloc(v_len as u64).ok_or_else(|| {
            format!(
                "upload_mesh: draw {}: no free vertex space for {} bytes",
                draw_idx, v_len
            )
        })? as usize;
        let i_off = match self.mesh_idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                // hand the vertex region back so a half-failed upload leaks no
                // space (frame 0: it was never written or drawn)
                self.mesh_vtx_alloc.free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "upload_mesh: draw {}: no free index space for {} bytes",
                    draw_idx, i_len
                ));
            }
        };

        // Vertices copy verbatim. Indices are mesh-relative, so rebase them to
        // the vertex region the allocator chose: v_off is always a multiple of
        // size_of::<Vertex>() (every seed region and allocation is), so the
        // base is an exact vertex index.
        write_buffer_region(&self.vertex_buffer, v_off, bytes_of_slice(vertices))?;
        // Static IB is u32 (per-scene total can exceed u16); per-mesh indices
        // are u16 (each mesh fits in u16, enforced by the build-time splitter).
        let base = (v_off / std::mem::size_of::<Vertex>()) as u32;
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        write_buffer_region(&self.index_buffer, i_off, bytes_of_slice(&rebased))?;

        let obj = &mut self.draw_objects[draw_idx];
        obj.vertex_offset = v_off;
        obj.index_offset = i_off / std::mem::size_of::<u32>();
        obj.resident = true;
        Ok(())
    }

    // Seed the streamed-mesh sub-allocators with the reserved headroom block
    // (byte ranges in the shared vertex / index buffers), for the
    // shrinkable-seed path.
    //
    // The streamed geometry is not baked into the buffers at build time;
    // instead the buffers carry one zeroed headroom region (sized to the
    // cap-many resident meshes) at these offsets. `retire_frame 0`: nothing
    // has been drawn yet, so the space is allocatable immediately -- mirrors
    // `setup_chunk_streaming`'s seeding. From then on `upload_mesh` /
    // `evict_mesh` place and free streamed meshes within it.
    pub fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        self.mesh_vtx_alloc.free(vtx_offset, vtx_bytes, 0);
        self.mesh_vtx_alloc.reclaim(0);
        self.mesh_idx_alloc.free(idx_offset, idx_bytes, 0);
        self.mesh_idx_alloc.reclaim(0);
    }

    // Clear a streamed mesh's geometry region to zero, return its space to the
    // sub-allocators, and mark the draw non-resident so it is skipped in every
    // pass.
    //
    // `retire_frame` is the frame from which the freed region may be reused:
    // pass `current_frame + frames_in_flight` for a runtime eviction so a
    // still-in-flight command buffer never has its geometry overwritten, and
    // `0` at init, where nothing has been drawn. A later `upload_mesh` brings
    // the mesh back, wherever the allocators then place it. Zeroing makes the
    // region carry no geometry, so a stray draw renders nothing rather than
    // stale triangles.
    pub fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("evict_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset;
        let v_len = obj.vertex_count * std::mem::size_of::<Vertex>();
        let i_off = obj.index_offset * std::mem::size_of::<u32>();
        let i_len = obj.index_count * std::mem::size_of::<u32>();
        zero_buffer_region(&self.vertex_buffer, v_off, v_len)?;
        zero_buffer_region(&self.index_buffer, i_off, i_len)?;
        self.mesh_vtx_alloc
            .free(v_off as u64, v_len as u64, retire_frame);
        self.mesh_idx_alloc
            .free(i_off as u64, i_len as u64, retire_frame);
        self.draw_objects[draw_idx].resident = false;
        Ok(())
    }

    // Overwrite a `Mesh` draw slot's vertex / index data in place. Driven by
    // asset hot-reload (`cn debug` only).
    //
    // Unlike the steady-state streaming paths (`upload_mesh` / `evict_mesh`),
    // which gate reuse of a region on `current_frame + frames_in_flight` so no
    // in-flight command buffer can still be reading it, this rewrites a live
    // region with no such fence. That is sound only because hot-reload is a
    // human-paced `cn debug` action: the editor stalls for the reload, so a
    // frame racing this write is not plausibly in flight. Production (non-debug)
    // callers must not take this path.
    //
    // The new geometry is written at the
    // draw object's existing offsets in the shared vertex / index buffers, so
    // every other draw sharing those offsets (a `Prop`-instanced clone of the
    // same `Mesh` always gets its own copy) is updated by the per-`draw_idx`
    // caller loop, not by this call. New `verts` / `idxs` must match the
    // slot's init-time count; this is the in-place fast path. A reload that
    // changes the count cannot fit the fixed slot, so the hot-reload driver
    // routes it through [`Self::rebuild_static_geometry`] instead, which
    // repacks the shared buffers from scratch. Each entry in
    // `lod_alternates` is written to the matching slot's pre-allocated LOD
    // region; the per-LOD index counts must match init-time counts too, and
    // the per-LOD `switch_distance`s are re-stored so JSON-side tweaks to
    // `lod_distances` propagate without restart.
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
        let v_off = obj.vertex_offset;
        let i_off_bytes = obj.index_offset * std::mem::size_of::<u32>();
        // Static draws keep indices absolute (base_vertex == 0), so rebase
        // the mesh-relative indices onto the slot's vertex_offset before
        // writing. v_off is always a multiple of size_of::<Vertex>() since
        // every region the build_draw_list appender produced started on a
        // vertex boundary.
        let base = (v_off / std::mem::size_of::<Vertex>()) as u32;
        // Snapshot the per-LOD index offsets while `obj` is still borrowed
        // so the buffer writes below can drop the borrow before mutating
        // each slice's switch_distance.
        let lod_byte_offsets: Vec<usize> = obj
            .lod_alternates
            .iter()
            .map(|s| s.index_offset * std::mem::size_of::<u32>())
            .collect();
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        write_buffer_region(&self.vertex_buffer, v_off, bytes_of_slice(vertices))?;
        write_buffer_region(&self.index_buffer, i_off_bytes, bytes_of_slice(&rebased))?;
        // LOD alternate slots were laid out at init-time alongside LOD0 in
        // the same shared index buffer. Rebase each alternate onto the same
        // `base` as LOD0 since LOD decimation shares the LOD0 vertex region.
        for ((_, alt_idx), &alt_off_bytes) in lod_alternates.iter().zip(lod_byte_offsets.iter()) {
            let alt_rebased: Vec<u32> = alt_idx.iter().map(|&i| u32::from(i) + base).collect();
            write_buffer_region(
                &self.index_buffer,
                alt_off_bytes,
                bytes_of_slice(&alt_rebased),
            )?;
        }
        // Refresh the per-LOD switch distances so JSON-side tweaks to
        // `lod_distances` propagate without a process restart.
        let slot = &mut self.draw_objects[draw_idx];
        for ((switch_distance, _), slice) in
            lod_alternates.iter().zip(slot.lod_alternates.iter_mut())
        {
            slice.switch_distance = *switch_distance;
        }
        Ok(())
    }
}
