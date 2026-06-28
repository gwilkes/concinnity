// src/metal/streaming.rs
//
// VoxelWorld chunk streaming for MtlContext: sub-allocator setup and the
// add / remove / move-chunk-mesh operations driven after init.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2_metal::{MTLBuffer, MTLDevice as _, MTLResourceOptions};

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::DrawObject;

use super::context::*;

impl MtlContext {
    // `VoxelWorld` chunks and seed the chunk sub-allocators with it.
    //
    // Called once at init by `GraphicsSystem` when a `VoxelWorld` is present.
    // The build-time geometry is copied verbatim into the start of the new
    // (larger) buffers; chunks are placed in the appended headroom by
    // `add_chunk_mesh`. This runs before the first frame, so no in-flight
    // command buffer references the replaced buffers.
    pub fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
    ) -> Result<(), String> {
        let old_v_len = self.vertex_buffer.length();
        let old_i_len = self.index_buffer.length();

        let new_vbuf = self
            .device
            .newBufferWithLength_options(
                old_v_len + chunk_vtx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or("setup_chunk_streaming: failed to allocate chunk vertex buffer")?;
        let new_ibuf = self
            .device
            .newBufferWithLength_options(
                old_i_len + chunk_idx_bytes,
                MTLResourceOptions::StorageModeShared,
            )
            .ok_or("setup_chunk_streaming: failed to allocate chunk index buffer")?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        copy_buffer_prefix(&self.vertex_buffer, &new_vbuf, old_v_len);
        copy_buffer_prefix(&self.index_buffer, &new_ibuf, old_i_len);
        self.vertex_buffer = new_vbuf;
        self.index_buffer = new_ibuf;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.chunk_vtx_alloc
            .free(old_v_len as u64, chunk_vtx_bytes as u64, 0);
        self.chunk_idx_alloc
            .free(old_i_len as u64, chunk_idx_bytes as u64, 0);
        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // add (or recycle) a `DrawObject` for it; returns the draw-list index.
    //
    // The chunk is non-cullable and joins the `always_draw` set: the streaming
    // window already bounds the resident chunk count, so the renderer draws
    // every resident chunk. `frame` reclaims retired deferred frees first.
    #[allow(clippy::too_many_arguments)]
    pub fn add_chunk_mesh(
        &mut self,
        vertices: &[Vertex],
        indices: &[u16],
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: crate::gfx::render_types::MaterialUniforms,
        frame: u64,
    ) -> Result<usize, String> {
        if vertices.is_empty() || indices.is_empty() {
            return Err("add_chunk_mesh: empty chunk geometry".to_string());
        }
        self.chunk_vtx_alloc.reclaim(frame);
        self.chunk_idx_alloc.reclaim(frame);

        let v_len = std::mem::size_of_val(vertices);
        // The shared index buffer is u32-typed; the input `indices` are u16 and
        // get widened on write below, so size the allocation against the u32
        // stride. Sizing against the u16 source would alloc half the bytes the
        // write needs and corrupt the next chunk's indices.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self.chunk_vtx_alloc.alloc(v_len as u64).ok_or_else(|| {
            format!(
                "add_chunk_mesh: no free chunk vertex space for {} bytes",
                v_len
            )
        })? as usize;
        let i_off = match self.chunk_idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.chunk_vtx_alloc.free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "add_chunk_mesh: no free chunk index space for {} bytes",
                    i_len
                ));
            }
        };

        // Vertices copy verbatim. Indices stay mesh-relative (0-based): a chunk
        // can land far past the 65 535-vertex u16 index range, so rather than
        // rebasing the indices the draw passes the vertex region's base as
        // `baseVertex`. v_off is a multiple of size_of::<Vertex>() (the
        // headroom start and every alloc are), so the base is an exact index.
        // The shared index_buffer is u32-typed, so widen the per-mesh u16
        // indices before writing.
        write_buffer_region(&self.vertex_buffer, v_off, bytes_of_slice(vertices))?;
        let indices_u32: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        write_buffer_region(&self.index_buffer, i_off, bytes_of_slice(&indices_u32))?;
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
            // Non-cullable: degenerate AABB disables frustum/distance culling.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            // Streamed `VoxelWorld` chunks do not run through the build-time
            // per-draw LOD decimator: distance LOD is handled by the streaming
            // window instead, which meshes a near chunk at full voxel detail
            // and a distant one as a coarse impostor (`ChunkDetail`), each a
            // single resolution. So no per-draw `lod_alternates` here.
            lod_alternates: Vec::new(),
        };

        // Recycle a vacated slot when one is free, else append. ensure_always_draw
        // adds a slot recycled from a culled static prop (not yet a member); a
        // slot reused from another chunk is already in always_draw and is left
        // alone.
        let draw_idx = match self.draw_slots.allocate() {
            crate::gfx::draw_slot::SlotAlloc::Reuse(slot) => {
                self.draw_objects[slot] = obj;
                self.prev_draw_models[slot] = model;
                slot
            }
            crate::gfx::draw_slot::SlotAlloc::Append(slot) => {
                self.draw_objects.push(obj);
                self.prev_draw_models.push(model);
                self.always_draw_member.push(false);
                slot
            }
        };
        self.ensure_always_draw(draw_idx);
        // A new resident chunk changes the RT-relevant draw set; the next RT
        // update folds it into the BVH (building just this chunk's BLAS).
        self.rt.topology_dirty = true;
        Ok(draw_idx)
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    //
    // `retire_frame` is `current_frame + frames_in_flight` so an in-flight
    // command buffer never has the freed region overwritten by a later
    // `add_chunk_mesh`. The slot stays in `draw_objects` / `always_draw` but
    // is marked non-resident and invisible, so every pass skips it.
    pub fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get_mut(draw_idx)
            .ok_or_else(|| format!("remove_chunk_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset;
        let v_len = obj.vertex_count * std::mem::size_of::<Vertex>();
        let i_off = obj.index_offset * std::mem::size_of::<u32>();
        let i_len = obj.index_count * std::mem::size_of::<u32>();
        obj.visible = false;
        obj.resident = false;
        zero_buffer_region(&self.vertex_buffer, v_off, v_len)?;
        zero_buffer_region(&self.index_buffer, i_off, i_len)?;
        self.chunk_vtx_alloc
            .free(v_off as u64, v_len as u64, retire_frame);
        self.chunk_idx_alloc
            .free(i_off as u64, i_len as u64, retire_frame);
        self.draw_slots.free(draw_idx);
        // The removed chunk leaves the RT-relevant draw set; the next RT update
        // drops its BLAS (deferred-freed once in-flight traces retire).
        self.rt.topology_dirty = true;
        Ok(())
    }

    // Rewrite a resident chunk's model matrix.
    //
    // Used by camera-relative rendering: when the camera crosses into a new
    // chunk the render origin follows it, so every resident chunk is rebased
    // onto the new origin. Only the model
    // matrix changes -- the geometry stays where it was uploaded. The
    // previous-frame model (`prev_draw_models`) is left untouched so the TAA
    // velocity pre-pass still diffs against the origin the chunk was last
    // rendered with: the rebase is exact, so a stationary chunk shows zero
    // motion across an origin shift.
    pub fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get_mut(draw_idx)
            .ok_or_else(|| format!("set_chunk_model: draw object {} out of range", draw_idx))?;
        obj.model = model;
        Ok(())
    }
}
