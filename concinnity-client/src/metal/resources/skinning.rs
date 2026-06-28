// src/metal/resources/skinning.rs
//
// Skinned-mesh GPU resources: pipeline + buffer setup (`upload_skinned`),
// per-frame pose updates, hot-reload of skinned geometry, and skeleton
// joint-count changes.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer as _, MTLComputePipelineState, MTLDevice, MTLLibrary as _, MTLPixelFormat,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLVertexDescriptor,
    MTLVertexFormat, MTLVertexStepFunction,
};

use crate::gfx::mesh_payload::SkinnedVertex;
use crate::gfx::render_types::SkinnedDrawObject;
use crate::metal::context::{HDR_SAMPLE_COUNT, MtlContext, bytes_of_slice, write_buffer_region};
use crate::metal::math::IDENTITY4;
use crate::metal::pipeline::{load_library, ns_str, shader_source};
use crate::metal::post::build_gbuffer_prepass_pipeline;
use crate::metal::post::fullscreen::compile_library;

// All skinned-mesh rendering state grouped into one feature unit: the main +
// shadow pipelines, the shared skinned vertex / index buffers, the per-mesh
// draw objects, and the current + previous joint-palette matrices. All
// `None` / empty until `upload_skinned` runs; with no `SkinnedMesh` in the
// world the skinned passes are skipped entirely. (The G-buffer pre-pass
// skinned pipeline lives on `GBufferState` with its siblings.)
pub(crate) struct SkinnedState {
    pub pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Depth-only skinned pipeline for the shadow pass. `None` when shadows are
    // disabled even if `pipeline_state` is set.
    pub shadow_pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Shared vertex buffer holding every skinned mesh's `SkinnedVertex` data.
    pub vertex_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Shared index buffer for skinned geometry.
    pub index_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // One entry per skinned mesh.
    pub draw_objects: Vec<SkinnedDrawObject>,
    // Current skinning matrices per skinned object, parallel to `draw_objects`.
    // Rewritten each frame by `update_skinned_pose` and uploaded per frame.
    pub joint_matrices: Vec<Vec<[[f32; 4]; 4]>>,
    // Previous frame's skinning matrices, parallel to `joint_matrices`. Lets
    // the velocity pre-pass capture per-vertex skinned deformation.
    pub prev_joint_matrices: Vec<Vec<[[f32; 4]; 4]>>,
    // GPU-driven fold: the `rt_skin` compute pipeline that deforms
    // bind-pose vertices into the per-frame `deformed` buffer, built here
    // independently of RT (which keeps its own pipeline) so a skinned world with
    // no ray tracing still gets the pre-skin. `None` until `upload_skinned` runs
    // on a bindless world with static geometry.
    pub skin_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    // One deformed-vertex buffer per frame-in-flight (56-byte static `Vertex`
    // layout, mirroring the skinned VB's global indexing so the u16 skinned
    // index buffer addresses it directly with `base_vertex = 0`). The per-frame
    // skin compute writes this frame's slot; the main-pass skinned ICB tail
    // reads it. `StorageModeShared`, NOT Private: written and read in separate
    // command buffers under the parallel per-pass encoder, where a Private
    // buffer GPU-page-faults (the RT deformed buffer hit the same and uses
    // Shared). Empty until `upload_skinned` allocates them.
    pub deformed: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // First-frame priming gate for the GPU-driven G-buffer velocity:
    // the previous-frame deformed buffer is unposed on frame 0 (and after a
    // deformed-ring rebuild in `upload_skinned`), so reading it would emit a
    // garbage skinned motion vector for one frame. While `false`, the G-buffer
    // skinned tail binds the CURRENT deformed buffer as the previous one (zero
    // skinned motion); set `true` after the first skinned-tail draw. Atomic, not
    // `Cell`: the G-buffer pass encodes on a render-graph worker thread (the
    // parallel per-pass encoder shares `&self` across rayon workers), so any
    // interior mutation reachable from `encode_pass_into` must be atomic, like
    // `draw_calls_accum`. Reset to `false` on the main thread in `upload_skinned`.
    pub deformed_primed: std::sync::atomic::AtomicBool,
}

// Skinned vertex layout: the 56-byte static attributes (pos / normal /
// tangent / colour / uv) plus `ushort4` joint indices (offset 56) and
// `float4` weights (offset 64). 80-byte stride; matches `SkinnedVertex` in
// [`crate::gfx::mesh_payload`]. Shared between init (one-shot in
// [`MtlContext::upload_skinned`]) and the hot-reload pipeline rebuild path
// so both produce byte-for-byte identical descriptors.
pub(crate) fn make_skinned_vertex_descriptor() -> Retained<MTLVertexDescriptor> {
    let vdesc = MTLVertexDescriptor::new();
    unsafe {
        let set = |idx: usize, fmt: MTLVertexFormat, offset: usize| {
            let attr = vdesc.attributes().objectAtIndexedSubscript(idx);
            attr.setFormat(fmt);
            attr.setOffset(offset);
            attr.setBufferIndex(1);
        };
        set(0, MTLVertexFormat::Float3, 0);
        set(1, MTLVertexFormat::Float3, 12);
        set(2, MTLVertexFormat::Float3, 24);
        set(3, MTLVertexFormat::Float3, 36);
        set(4, MTLVertexFormat::Float2, 48);
        set(5, MTLVertexFormat::UShort4, 56);
        set(6, MTLVertexFormat::Float4, 64);
        let layout = vdesc.layouts().objectAtIndexedSubscript(1);
        layout.setStride(std::mem::size_of::<SkinnedVertex>());
        layout.setStepFunction(MTLVertexStepFunction::PerVertex);
    }
    vdesc
}

// Build the main skinned pipeline: pairs `vertex_main_skinned` (from the
// world's vertex library bytes) with `fragment_main` (from the world's
// fragment library bytes), targeting the off-screen HDR MSAA surface:
// byte-for-byte identical state to the static main pipeline aside from the
// vertex entry point + 80-byte vertex descriptor. Shared by
// [`MtlContext::upload_skinned`] and the hot-reload pipeline rebuild path.
pub(crate) fn build_skinned_main_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vdesc: &MTLVertexDescriptor,
    vert_lib_bytes: &[u8],
    frag_lib_bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let vert_library = load_library(device, vert_lib_bytes)
        .map_err(|e| format!("skinned: failed to load vertex metallib: {}", e))?;
    let frag_library = load_library(device, frag_lib_bytes)
        .map_err(|e| format!("skinned: failed to load fragment metallib: {}", e))?;
    let skinned_vert_fn = vert_library
        .newFunctionWithName(&ns_str("vertex_main_skinned"))
        .ok_or("vertex_main_skinned not found in metallib")?;
    let frag_fn = frag_library
        .newFunctionWithName(&ns_str("fragment_main"))
        .ok_or("fragment_main not found in metallib")?;

    let desc = MTLRenderPipelineDescriptor::new();
    desc.setVertexDescriptor(Some(vdesc));
    desc.setVertexFunction(Some(&skinned_vert_fn));
    desc.setFragmentFunction(Some(&frag_fn));
    desc.setRasterSampleCount(HDR_SAMPLE_COUNT as usize);
    unsafe {
        desc.colorAttachments()
            .objectAtIndexedSubscript(0)
            .setPixelFormat(MTLPixelFormat::RGBA16Float);
    }
    desc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&desc)
        .map_err(|e| format!("failed to create skinned pipeline state: {:?}", e))
}

// Build the skinned shadow pipeline: depth-only, no fragment function, no
// MSAA, compiled from the engine-internal `shadow_map.metal` source (entry
// `shadow_vertex_main_skinned`). Mirrors
// [`crate::metal::init::pipelines::build_shadow_pipeline`] but on the 80-byte
// skinned vertex layout. Shared by [`MtlContext::upload_skinned`] and the
// internal-shader hot-reload pipeline rebuild path.
pub(crate) fn build_skinned_shadow_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vdesc: &MTLVertexDescriptor,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn MTLRenderPipelineState>>, String> {
    let msl = shader_source(hot_reload, "shadow_map.metal");
    let shadow_library = compile_library(device, msl.as_ref(), "shadow_map")?;
    let shadow_fn = shadow_library
        .newFunctionWithName(&ns_str("shadow_vertex_main_skinned"))
        .ok_or("shadow_vertex_main_skinned not found in shadow library")?;
    let sdesc = MTLRenderPipelineDescriptor::new();
    sdesc.setVertexDescriptor(Some(vdesc));
    sdesc.setVertexFunction(Some(&shadow_fn));
    sdesc.setRasterSampleCount(1);
    sdesc.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
    device
        .newRenderPipelineStateWithDescriptor_error(&sdesc)
        .map_err(|e| format!("failed to create skinned shadow pipeline state: {:?}", e))
}

impl MtlContext {
    // Rebuild the shared skinned-mesh vertex + index buffers, swapping in
    // new geometry for the slots named in `changes`. Driven by asset
    // hot-reload (`cn debug` only) when a SkinnedMesh's re-imported `.glb`
    // no longer fits in its init-time slot. Walks every
    // `SkinnedDrawObject` in order: for each slot in `changes`, the new
    // vertices / indices are appended to a fresh CPU buffer; for unchanged
    // slots, the current geometry is read back from the live
    // `skinned_vertex_buffer` / `skinned_index_buffer` (both
    // `StorageModeShared` so the pointers are CPU-readable) and copied
    // with index rebasing. New `MTLBuffer`s are created at the post-rebuild
    // size and swapped in after `wait_idle` so no in-flight command buffer
    // touches the old resource pair. The skinned pipelines and shadow /
    // velocity / SSAO / SSR variants are untouched -- only the per-slot
    // `vertex_base` / `vertex_count` / `index_offset` / `index_count` on
    // each `SkinnedDrawObject` (and the two GPU buffers themselves) move.
    // Skeleton-shape changes (joint-count mismatch) are still rejected one
    // level above this call (in `reload_assets`) -- they would need the
    // original `vert_lib_bytes` / `frag_lib_bytes` / `shadow_lib_bytes`
    // which `upload_skinned` consumes and drops.
    pub fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<crate::gfx::backend::SkinnedSlotLayout>, String> {
        use std::collections::HashMap;

        let v_buf = self.skinned.vertex_buffer.as_ref().ok_or(
            "rebuild_skinned_geometry: no skinned vertex buffer (was upload_skinned called?)",
        )?;
        let i_buf = self.skinned.index_buffer.as_ref().ok_or(
            "rebuild_skinned_geometry: no skinned index buffer (was upload_skinned called?)",
        )?;

        // Stop the GPU + CPU pipelines so we can safely read the old buffers
        // and atomically swap. Costs a frame-time stall but only fires under
        // `cn debug` and only when the source `.glb` size actually changed.
        self.wait_idle();

        let mut change_map: HashMap<usize, crate::gfx::backend::SkinnedDrawGeometryUpdate> =
            changes.into_iter().map(|c| (c.skinned_index, c)).collect();

        let old_v_len = v_buf.length() / std::mem::size_of::<SkinnedVertex>();
        let old_v_slice: &[SkinnedVertex] = unsafe {
            let ptr = v_buf.contents().as_ptr() as *const SkinnedVertex;
            std::slice::from_raw_parts(ptr, old_v_len)
        };
        let old_i_len = i_buf.length() / std::mem::size_of::<u16>();
        let old_i_slice: &[u16] = unsafe {
            let ptr = i_buf.contents().as_ptr() as *const u16;
            std::slice::from_raw_parts(ptr, old_i_len)
        };

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
                // Each mesh-relative index must stay in u16 after rebase. The
                // post-rebuild total vertex count also has to fit u16; the
                // check above on new_v_base catches that boundary too.
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
                // Unchanged slot -- copy current geometry verbatim, rebasing
                // its absolute indices from the old vertex_base onto the new
                // one.
                let v_start = obj.vertex_base as usize;
                let v_end = v_start + obj.vertex_count;
                if v_end > old_v_slice.len() {
                    return Err(format!(
                        "rebuild_skinned_geometry: slot {} vertex region [{}, {}) \
                         out of bounds (buffer has {} vertices)",
                        skinned_index,
                        v_start,
                        v_end,
                        old_v_slice.len()
                    ));
                }
                new_vertices.extend_from_slice(&old_v_slice[v_start..v_end]);
                let i_end = obj.index_offset + obj.index_count;
                if i_end > old_i_slice.len() {
                    return Err(format!(
                        "rebuild_skinned_geometry: slot {} index region [{}, {}) \
                         out of bounds (buffer has {} indices)",
                        skinned_index,
                        obj.index_offset,
                        i_end,
                        old_i_slice.len()
                    ));
                }
                let old_base = obj.vertex_base;
                // `idx - old_base + new_v_base` -- both subtraction and
                // addition are bounded by the slot's vertex_count (which we
                // just placed at new_v_base).
                for &abs in &old_i_slice[obj.index_offset..i_end] {
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

        // Create new MTL buffers sized to the rebuilt layout.
        let new_vertex_buffer = unsafe {
            let v_bytes = std::mem::size_of_val(new_vertices.as_slice());
            let ptr = std::ptr::NonNull::new(new_vertices.as_ptr() as *mut _)
                .ok_or("rebuild_skinned_geometry: vertex slice pointer is null")?;
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    v_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("rebuild_skinned_geometry: failed to create new vertex buffer")?
        };
        let new_index_buffer = unsafe {
            let i_bytes = std::mem::size_of_val(new_indices.as_slice());
            let ptr = std::ptr::NonNull::new(new_indices.as_ptr() as *mut _)
                .ok_or("rebuild_skinned_geometry: index slice pointer is null")?;
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    i_bytes,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("rebuild_skinned_geometry: failed to create new index buffer")?
        };

        // Apply the new per-slot layout.
        for (skinned_index, v_base, v_count, i_off, i_count) in new_per_slot {
            let obj = &mut self.skinned.draw_objects[skinned_index];
            obj.vertex_base = v_base;
            obj.vertex_count = v_count;
            obj.index_offset = i_off;
            obj.index_count = i_count;
        }

        self.skinned.vertex_buffer = Some(new_vertex_buffer);
        self.skinned.index_buffer = Some(new_index_buffer);
        Ok(layouts)
    }

    // Overwrite a `SkinnedMesh` draw slot's vertex + index data in the
    // shared skinned vertex / index buffers in place. Driven by asset
    // hot-reload (`cn debug` only).
    //
    // Like [`MtlContext::update_mesh_geometry`], this rewrites a live buffer
    // region with no in-flight fence. There is no steady-state skinned
    // streamer (skinned meshes upload once at init and are never evicted), so
    // the only caller is the human-paced `cn debug` hot-reload, which stalls
    // for the reload: no frame racing this write is plausibly in flight.
    // Production (non-debug) callers must not take this path.
    //
    // The slot's vertex region starts at
    // `vertex_base * size_of::<SkinnedVertex>()` and is `vertices.len()`
    // vertices wide; the index region lives at the slot's init-time
    // `index_offset` / `index_count`. Indices are rebased onto `vertex_base`
    // before writing (the existing init-time draws followed the same
    // rebasing convention). New `verts` / `idxs` must match the slot's
    // init-time count -- size-changing reloads route through
    // [`Self::rebuild_skinned_geometry`] instead. The shader
    // libraries + pipelines stay untouched on every skinned reload path;
    // joint-count changes resize the per-slot joint-matrix buffers via
    // [`Self::update_skinned_skeleton`].
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
        let v_buf = self.skinned.vertex_buffer.as_ref().ok_or(
            "update_skinned_mesh_geometry: no skinned vertex buffer (was upload_skinned called?)",
        )?;
        let i_buf = self.skinned.index_buffer.as_ref().ok_or(
            "update_skinned_mesh_geometry: no skinned index buffer (was upload_skinned called?)",
        )?;
        // Check the vertex region fits inside the live buffer. The shared
        // buffer was sized once at upload_skinned to hold every skinned
        // mesh's vertices; vertex_base + vertices.len() must stay within
        // that region. Overflow would corrupt a neighbouring slot.
        let v_byte_off = (vertex_base as usize) * std::mem::size_of::<SkinnedVertex>();
        let v_byte_len = std::mem::size_of_val(vertices);
        if v_byte_off + v_byte_len > v_buf.length() {
            return Err(format!(
                "update_skinned_mesh_geometry: vertex region [{}, {}) overruns skinned \
                 vertex buffer length {}",
                v_byte_off,
                v_byte_off + v_byte_len,
                v_buf.length()
            ));
        }
        let i_byte_off = obj.index_offset * std::mem::size_of::<u16>();
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
        write_buffer_region(v_buf, v_byte_off, bytes_of_slice(vertices))?;
        write_buffer_region(i_buf, i_byte_off, bytes_of_slice(&rebased))?;
        Ok(())
    }

    // Build the GPU pipelines + buffers for skeletally animated meshes.
    //
    // Called once by `GraphicsSystem` after `MtlContext::new`, only when the
    // world declares at least one `SkinnedMesh`. The skinned vertex shader
    // (`vertex_main_skinned`) compiles from the same world vertex/fragment
    // metallibs as the static main pipeline; the skinned shadow shader
    // (`shadow_vertex_main_skinned`) compiles from the engine-internal
    // `shadow_map.metal` source. With no skinned meshes this is never called
    // and every skinned pass is skipped.
    //
    // `_shadow_lib_bytes` is retained for the cross-backend `RenderBackend`
    // signature but unused on Metal: the shadow shader is engine-internal here.
    pub fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u16],
        draw_objects: Vec<SkinnedDrawObject>,
        vert_lib_bytes: &[u8],
        frag_lib_bytes: &[u8],
        _shadow_lib_bytes: &[u8],
    ) -> Result<(), String> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        let vdesc = make_skinned_vertex_descriptor();
        let skinned_ps =
            build_skinned_main_pipeline(&self.device, &vdesc, vert_lib_bytes, frag_lib_bytes)?;

        // Skinned shadow pipeline: built only when the static shadow pass is
        // active, so a skinned mesh casts a correctly deformed shadow.
        let skinned_shadow_ps = if self.shadow_pipeline_state.is_some() {
            Some(build_skinned_shadow_pipeline(
                &self.device,
                &vdesc,
                self.hot_reload,
            )?)
        } else {
            None
        };

        // Unified G-buffer pre-pass pipeline for skinned geometry. Built here for
        // the same reason (80-byte skinned layout), when any consumer (SSR / SSGI
        // / RT / SSAO / TAA / upscaler) is on.
        if self.ssr.settings.is_some()
            || self.ssgi.settings.is_some()
            || self.rt.settings.is_some()
            || self.ssao.settings.is_some()
            || self.taa.enabled
            || self.upscale.scaler.is_some()
        {
            self.gbuffer.skinned_pipeline = Some(build_gbuffer_prepass_pipeline(
                &self.device,
                &vdesc,
                "gbuffer_prepass_vertex_skinned",
                self.hot_reload,
            )?);
        }

        let skinned_vertex_buffer = unsafe {
            let ptr = std::ptr::NonNull::new(vertices.as_ptr() as *mut _)
                .ok_or("skinned vertex slice is empty")?;
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    std::mem::size_of_val(vertices),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("failed to create skinned vertex buffer")?
        };
        let skinned_index_buffer = unsafe {
            let ptr = std::ptr::NonNull::new(indices.as_ptr() as *mut _)
                .ok_or("skinned index slice is empty")?;
            self.device
                .newBufferWithBytes_length_options(
                    ptr,
                    std::mem::size_of_val(indices),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or("failed to create skinned index buffer")?
        };

        // Seed each object's joint matrices to identity (bind pose) so the
        // mesh renders undeformed until the first `update_skinned_pose`. The
        // previous-frame copy starts identical so the velocity pre-pass sees
        // zero skinned motion on the first frame.
        self.skinned.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY4; o.joint_count.max(1)])
            .collect();
        self.skinned.prev_joint_matrices = self.skinned.joint_matrices.clone();

        // GPU-driven skinned fold: when bindless is active AND the
        // world has static geometry (so the cull + bindless ICB run), build the
        // per-frame pre-skin so skinned objects draw as rigid deformed geometry
        // through the unified cull, exactly like DX/VK. A pure-skinned or
        // non-bindless world leaves these unset and keeps the legacy skinned VS
        // draw (the main-pass gate falls back when `!bindless || draw_objects
        // empty`). The skin pipeline is built independently of RT; RT keeps its
        // own skin pipeline + deformed buffer.
        if self.bindless && !self.draw_objects.is_empty() {
            let skin_pipeline =
                crate::metal::raytrace::build_rt_skin_pipeline(&self.device, self.hot_reload)?;
            // One deformed buffer per frame-in-flight (the skin write and the
            // main-pass read live in separate command buffers, so a per-frame
            // ring lets frames pipeline without the next frame's skin racing this
            // frame's draw). Sized to every skinned vertex (global indexing,
            // base 0). Shared storage: a Private buffer page-faults in this
            // cross-command-buffer producer/consumer pattern (see the RT path).
            let stride = crate::metal::raytrace::VERTEX_STRIDE;
            let deformed_bytes = (vertices.len() * stride).max(stride);
            let mut deformed = Vec::with_capacity(self.frames_in_flight);
            for _ in 0..self.frames_in_flight {
                let buf = self
                    .device
                    .newBufferWithLength_options(
                        deformed_bytes,
                        MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or("failed to allocate skinned deformed-vertex buffer")?;
                deformed.push(buf);
            }
            self.skinned.skin_pipeline = Some(skin_pipeline);
            self.skinned.deformed = deformed;
            // Fresh deformed ring: the previous-frame slots are unposed until a
            // frame writes them, so re-arm the G-buffer velocity priming gate.
            // Main-thread store (this runs outside the per-pass fan-out).
            self.skinned
                .deformed_primed
                .store(false, std::sync::atomic::Ordering::Relaxed);
            // The count `cull_count()` reads: now the skinned records ride the
            // unified cull + bindless ICB, and the legacy skinned main draw is
            // gated off.
            self.n_skinned = draw_objects.len();
        }

        self.skinned.pipeline_state = Some(skinned_ps);
        self.skinned.shadow_pipeline_state = skinned_shadow_ps;
        self.skinned.vertex_buffer = Some(skinned_vertex_buffer);
        self.skinned.index_buffer = Some(skinned_index_buffer);
        self.skinned.draw_objects = draw_objects;
        Ok(())
    }

    // Replace the skinning matrices for one skinned object. Called each frame
    // from `GraphicsSystem` with the pose `AnimationSystem` computed. Out-of-
    // range indices are ignored.
    pub fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.clear();
            slot.extend_from_slice(matrices);
            if slot.is_empty() {
                slot.push(IDENTITY4);
            }
        }
    }

    // Update a skinned slot's joint count and resize its per-slot joint
    // matrix buffers. Driven by asset hot-reload (`cn debug` only) when a
    // re-imported `.glb`'s skeleton has a different joint count than the
    // slot was initialised with. The shared skinned pipelines stay
    // untouched -- the shaders read `joints` through a pointer and use
    // vertex-encoded joint indices, so a new joint count only requires the
    // CPU-side per-slot Vec to be resized (and `SkinnedDrawObject.joint_count`
    // updated). New entries are seeded to identity so the slot renders
    // undeformed until the next `update_skinned_pose` writes the new pose.
    // `prev_skinned_joint_matrices` is resized in lockstep so the velocity
    // pre-pass sees zero skinned motion on the post-reload frame.
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
        let capped = new_joint_count.min(crate::gfx::render_types::MAX_JOINTS);
        obj.joint_count = capped;
        let size = capped.max(1);
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.resize(size, IDENTITY4);
        }
        if let Some(slot) = self.skinned.prev_joint_matrices.get_mut(skinned_index) {
            slot.resize(size, IDENTITY4);
        }
        Ok(())
    }

    // Seed the skinned instance pool from `(template_index, instance_index)`
    // pairs built at load, each instance being a hidden bind-pose copy of its
    // template. Called once after `upload_skinned`, before any runtime skinned
    // spawn. With no skinned mesh opting into runtime spawning the list is
    // empty and the pool stays empty.
    pub fn seed_skinned_instance_pool(&mut self, reservations: Vec<(usize, usize)>) {
        for (template, instance) in reservations {
            self.skinned_pool.reserve(template, instance);
        }
    }

    // Claim a free pre-reserved copy of the skinned object at
    // `template_skinned_index`, reveal it at `model`, and reset its palette to
    // the bind pose so it does not flash its previous occupant's last frame
    // (the owning `SkeletonPose`'s first pose push replaces it next frame).
    // Returns the claimed slot's skinned index, or `None` when the template
    // reserved no pool or the pool is exhausted.
    pub fn spawn_skinned_instance(
        &mut self,
        template_skinned_index: usize,
        model: [[f32; 4]; 4],
    ) -> Option<usize> {
        let slot = self.skinned_pool.acquire(template_skinned_index)?;
        let obj = self.skinned.draw_objects.get_mut(slot)?;
        obj.model = model;
        obj.visible = true;
        if let Some(palette) = self.skinned.joint_matrices.get_mut(slot) {
            palette.iter_mut().for_each(|m| *m = IDENTITY4);
        }
        if let Some(palette) = self.skinned.prev_joint_matrices.get_mut(slot) {
            palette.iter_mut().for_each(|m| *m = IDENTITY4);
        }
        Some(slot)
    }

    // Hide a skinned object and, if it was a pre-reserved instance, return its
    // slot to the pool so a later spawn can claim it. An authored template slot
    // is simply hidden (it owns no pool entry). A no-op if the index is out of
    // range.
    pub fn retire_skinned_draw_object(&mut self, skinned_index: usize) {
        if let Some(obj) = self.skinned.draw_objects.get_mut(skinned_index) {
            obj.visible = false;
        }
        self.skinned_pool.release(skinned_index);
    }

    // Push a skinned object's model-to-world matrix. The per-frame cull rebuild
    // reads `obj.model` directly (the skinned records are rebuilt every frame),
    // so this only writes the field. A no-op if the index is out of range.
    pub fn update_skinned_model(&mut self, skinned_index: usize, model: [[f32; 4]; 4]) {
        if let Some(obj) = self.skinned.draw_objects.get_mut(skinned_index) {
            obj.model = model;
        }
    }
}
