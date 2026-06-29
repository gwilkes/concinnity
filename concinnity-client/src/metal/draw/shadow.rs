// src/metal/draw/shadow.rs
//
// Cascaded shadow-map pass: one depth-only render per CSM cascade slice.
// Draws static objects, instanced clusters, and (when present) skinned
// meshes into each slice of `shadow_map`. Skipped entirely when no shadow
// pipeline is configured.
//
// Within each cascade the three sub-paths run sequentially on a single
// `MTLRenderCommandEncoder`. Each cascade is its own render pass (`setSlice`
// targets a different shadow_map array slice). The per-path helpers below
// stay split out so the dispatch shape is ready once a Metal-side parallel
// path proves safe; see [`encode_main_pass`](../draw/main.rs) for the
// matching note on why the earlier `MTLParallelRenderCommandEncoder` landing
// was reverted.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _, MTLIndexType, MTLLoadAction,
    MTLPrimitiveType, MTLRenderCommandEncoder as _, MTLRenderPassDescriptor, MTLStoreAction,
};

use crate::gfx::render_types::{NUM_SHADOW_CASCADES, ShadowPassPush, ShadowUniforms};
use crate::metal::context::MtlContext;
use crate::metal::scoped_encoder::ScopedEncoder;
use crate::metal::uniforms::ModelUniforms;

impl MtlContext {
    // Choose which shadow cascades to re-render this frame and advance the
    // round-robin clock. Delegates to the shared `ShadowCascadeScheduler`
    // (`gfx::shadow_schedule`, unit-tested there). Called once per frame from
    // draw_frame; the result is stashed in `shadow_render_mask` for
    // encode_shadow_pass and used to gate which cascade VPs refresh.
    pub(in crate::metal) fn next_shadow_cascade_mask(&mut self) -> u32 {
        self.shadow_scheduler
            .next_mask(self.shadow_update, self.shadow_cascades)
    }

    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::metal) fn encode_shadow_pass(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
        cam_pos: [f32; 3],
        // GPU-driven shadow: the per-frame `GpuObjectData` buffer the
        // bindless shadow VS reads each cascade's model matrix from (by the
        // `[[base_instance]]` record id the shadow cull baked). `Some` exactly
        // when the bindless cull path ran this frame; `None` routes the pass to
        // the legacy per-cascade CPU loop.
        object_buffer: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // This frame's skinned deformed-vertex buffer (skinned fold). Bound as
        // the vertex buffer for the skinned tail of each cascade's indirect
        // draw. `Some` only when the skinned fold is active.
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
        // When `Some`, raymarched SDF casters draw into each cascade after the
        // rasterised + skinned draws (and before the Main pass samples the
        // shadow map). Built by the graph executor: same matrix / time /
        // camera the main raymarch pass will use later this frame, so the
        // shadow cast and the live surface agree. `None` when no volume opts
        // into `cast_shadows`. Mirrors the DirectX shadow pass.
        raymarch_view: Option<&crate::metal::raymarch::RaymarchView>,
    ) -> Result<u32, String> {
        let Some(shadow_pipeline) = self.shadow_pipeline_state.clone() else {
            return Ok(0);
        };
        // GPU-driven path: when the bindless cull ran this frame and the
        // depth-only bindless shadow pipeline exists, every cascade's casters
        // (static + instances + skinned tail) draw through the shadow ICB the
        // `encode_shadow_culls` compute prologue filled. Otherwise fall back to
        // the legacy per-cascade CPU loop (custom-shader / non-bindless /
        // pure-skinned worlds). `object_buffer.is_some()` is exactly the main
        // bindless gate (`bindless && !draw_objects.is_empty()`).
        let gpu_driven = self.cull.shadow_bindless_pipeline.is_some() && object_buffer.is_some();
        let mut total_draws: u32 = 0;

        // Cascades to re-render this frame; draw_frame computed the mask from
        // the update policy. A skipped cascade keeps the depth and VP from when
        // it was last rendered, so the Main pass still samples it consistently.
        // Defensive fallback to all cascades if no mask was set this frame.
        let all = (1u32 << NUM_SHADOW_CASCADES) - 1;
        let mask = if self.shadow_render_mask == 0 {
            all
        } else {
            self.shadow_render_mask
        };
        let rendered: Vec<usize> = (0..NUM_SHADOW_CASCADES)
            .filter(|i| mask & (1u32 << i) != 0)
            .collect();
        let first_rendered = rendered.first().copied();
        let last_rendered = rendered.last().copied();

        for &cascade_idx in &rendered {
            let shadow_pass_desc = MTLRenderPassDescriptor::new();
            let depth_attach = shadow_pass_desc.depthAttachment();
            depth_attach.setTexture(Some(self.shadow_map.as_ref()));
            depth_attach.setSlice(cascade_idx);
            depth_attach.setLoadAction(MTLLoadAction::Clear);
            depth_attach.setStoreAction(MTLStoreAction::Store);
            depth_attach.setClearDepth(1.0);

            // Per-pass GPU timing spans the first to the last cascade actually
            // rendered this frame (the set varies with the update policy):
            // attach the start sample to the first and the end to the last.
            if let Some(t) = &self.pass_timing {
                let id = super::super::pass_timing::PassId::Shadow;
                let is_first = Some(cascade_idx) == first_rendered;
                let is_last = Some(cascade_idx) == last_rendered;
                if is_first && is_last {
                    t.attach_render(&shadow_pass_desc, id);
                } else if is_first {
                    t.attach_render_first(&shadow_pass_desc, id);
                } else if is_last {
                    t.attach_render_last(&shadow_pass_desc, id);
                }
            }

            // Loop-local guard: each cascade's encoder ends when the guard drops
            // at the end of this iteration, before the next cascade opens one.
            let shadow_enc = ScopedEncoder::new(
                cmd_buf
                    .renderCommandEncoderWithDescriptor(&shadow_pass_desc)
                    .ok_or("failed to get shadow render encoder")?,
                "shadow cascade",
            );

            // Slope-scale bias scales with cascade to compensate for the
            // larger texel footprint of distant cascades.
            let slope_bias = 1.0 + cascade_idx as f32 * 0.5;
            let push = ShadowPassPush {
                cascade_idx: cascade_idx as u32,
                _pad: [0; 3],
            };

            if gpu_driven {
                // One (static+instance prefix) or two (+ skinned tail) indirect
                // draws over this cascade's slice of the shadow ICB. `object_buffer`
                // is `Some` here (gates `gpu_driven`).
                total_draws += self.encode_shadow_cascade_indirect(
                    &shadow_enc,
                    &push,
                    slope_bias,
                    cascade_idx,
                    object_buffer.expect("gpu_driven implies object_buffer"),
                    deformed_skinned,
                );
            } else {
                let count_static = self.encode_shadow_static_into(
                    &shadow_enc,
                    &shadow_pipeline,
                    &push,
                    slope_bias,
                    cam_pos,
                );
                let count_instanced = self.encode_shadow_instanced_into(
                    &shadow_enc,
                    &shadow_pipeline,
                    &push,
                    slope_bias,
                    cam_pos,
                );
                let count_skinned = self.encode_shadow_skinned_into(
                    &shadow_enc,
                    &push,
                    slope_bias,
                    cam_pos,
                    skinned_joint_bufs,
                );
                total_draws += count_static + count_instanced + count_skinned;
            }
        }

        // Raymarched SDF shadow casters: depth-only draws into the same
        // per-cascade slices, run after the rasterised + skinned casters so
        // both layers compete via the slice's LESS depth test (nearest caster
        // wins per texel). No-op when no volume opts into `cast_shadows` or the
        // executor passed no view.
        if let Some(view) = raymarch_view {
            total_draws += self.encode_sdf_shadow_casters(cmd_buf, view)?;
        }

        Ok(total_draws)
    }

    // Apply the bindings every shadow sub-path needs (shadow pipeline,
    // depth state, ShadowUniforms at vertex buffer 0, the cascade push at
    // vertex buffer 7, and the shared vertex buffer at binding 1).
    fn bind_shadow_pass_shared(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        shadow_pipeline: &ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
        push: &ShadowPassPush,
        slope_bias: f32,
    ) {
        enc.setRenderPipelineState(shadow_pipeline);
        enc.setDepthStencilState(Some(&self.depth_state));
        enc.setDepthBias_slopeScale_clamp(0.005, slope_bias, 0.01);
        unsafe {
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(&self.shadow_uniforms).cast(),
                std::mem::size_of::<ShadowUniforms>(),
                0,
            );
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(push).cast(),
                std::mem::size_of::<ShadowPassPush>(),
                7,
            );
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 1);
        }
    }

    // GPU-driven shadow draws for one cascade: execute this cascade's
    // slice of the shadow ICB the `cull_encode_shadow` kernel filled. Mirrors the
    // main pass's two-range split (`execute_bindless_static_icb`): one indirect
    // draw for the static + instance prefix (static VB bound at 1, static u32 IB
    // resident), then one for the folded skinned tail (deformed VB rebound at 1,
    // skinned u16 IB resident). The depth-only bindless shadow VS reads each
    // record's model from the object buffer at vbuf 9 by `[[base_instance]]`.
    // Returns the indirect-draw count (1 or 2).
    fn encode_shadow_cascade_indirect(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        push: &ShadowPassPush,
        slope_bias: f32,
        cascade_idx: usize,
        object_buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        deformed_skinned: Option<&Retained<ProtocolObject<dyn MTLBuffer>>>,
    ) -> u32 {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        let (Some(pipeline), Some(icb)) = (
            self.cull.shadow_bindless_pipeline.as_ref(),
            self.cull.shadow_icb.as_ref(),
        ) else {
            return 0;
        };
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str(
            "shadow cascade indirect",
        ));
        enc.setRenderPipelineState(pipeline);
        enc.setDepthStencilState(Some(&self.depth_state));
        enc.setDepthBias_slopeScale_clamp(0.005, slope_bias, 0.01);
        unsafe {
            // ShadowUniforms (vbuf 0), cascade push (vbuf 7), object buffer
            // (vbuf 9), static vertex buffer (vbuf 1). The ICB commands inherit
            // these bindings; the cull baked base_instance = record id, so the VS
            // reads `objects[id].model`.
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(&self.shadow_uniforms).cast(),
                std::mem::size_of::<ShadowUniforms>(),
                0,
            );
            enc.setVertexBytes_length_atIndex(
                std::ptr::NonNull::from(push).cast(),
                std::mem::size_of::<ShadowPassPush>(),
                7,
            );
            enc.setVertexBuffer_offset_atIndex(Some(object_buffer), 0, 9);
            enc.setVertexBuffer_offset_atIndex(Some(&self.vertex_buffer), 0, 1);
        }

        // This cascade's command slots live at `[c*stride, c*stride + stride)`
        // in the shared shadow ICB (stride = the live cull_count, the same value
        // `encode_shadow_culls` used as `cascade_base`).
        let stride = self.cull_count();
        let base = self.skinned_record_base();
        let cascade_off = cascade_idx * stride;
        let mut draw_calls = 0u32;

        // Static + instance prefix.
        if base > 0 {
            enc.useResource_usage_stages(
                ProtocolObject::from_ref(&*self.index_buffer),
                MTLResourceUsage::Read,
                MTLRenderStages::Vertex,
            );
            let range = objc2_foundation::NSRange {
                location: cascade_off,
                length: base,
            };
            // SAFETY: [cascade_off, cascade_off + base) spans this cascade's
            // static + instance command slots (ensure_shadow_icb_capacity sized
            // the ICB for NUM_SHADOW_CASCADES * cull_count).
            unsafe {
                enc.executeCommandsInBuffer_withRange(icb.as_ref(), range);
            }
            draw_calls += 1;
        }

        // Folded skinned tail: deformed VB at binding 1, skinned u16 IB resident.
        if let Some(deformed) = deformed_skinned
            && self.n_skinned > 0
        {
            unsafe {
                enc.setVertexBuffer_offset_atIndex(Some(deformed), 0, 1);
            }
            if let Some(skinned_ib) = self.skinned.index_buffer.as_ref() {
                enc.useResource_usage_stages(
                    ProtocolObject::from_ref(&**skinned_ib),
                    MTLResourceUsage::Read,
                    MTLRenderStages::Vertex,
                );
            }
            let range = objc2_foundation::NSRange {
                location: cascade_off + base,
                length: self.n_skinned,
            };
            // SAFETY: [cascade_off + base, cascade_off + cull_count) spans this
            // cascade's folded skinned command slots.
            unsafe {
                enc.executeCommandsInBuffer_withRange(icb.as_ref(), range);
            }
            draw_calls += 1;
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode the static-geometry shadow draws.
    fn encode_shadow_static_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        shadow_pipeline: &ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
        push: &ShadowPassPush,
        slope_bias: f32,
        cam_pos: [f32; 3],
    ) -> u32 {
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("shadow static"));
        self.bind_shadow_pass_shared(enc, shadow_pipeline, push, slope_bias);

        let mut draw_calls: u32 = 0;
        for obj in &self.draw_objects {
            if !obj.visible || !obj.resident {
                continue;
            }
            let model_uniforms = ModelUniforms { model: obj.model };
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&model_uniforms).cast(),
                    std::mem::size_of::<ModelUniforms>(),
                    2,
                );
            }
            // Pick the LOD by camera distance -- the shadow pass uses the
            // same slice the main pass will, so silhouettes track when the
            // runtime swaps to a coarser LOD.
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let index_byte_offset = index_offset * std::mem::size_of::<u32>();
            unsafe {
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount_baseVertex_baseInstance(
                    MTLPrimitiveType::Triangle,
                    index_count,
                    MTLIndexType::UInt32,
                    &self.index_buffer,
                    index_byte_offset,
                    1,
                    obj.base_vertex as isize,
                    0,
                );
            }
            draw_calls += 1;
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode shadow draws for instanced clusters by iterating per-instance
    // using the (non-instanced) shadow pipeline. Cheap to ship and
    // visually identical to an instanced shadow shader. Off-screen
    // instances can still cast shadows onto visible surfaces, so no
    // cluster-level cull here.
    fn encode_shadow_instanced_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        shadow_pipeline: &ProtocolObject<dyn objc2_metal::MTLRenderPipelineState>,
        push: &ShadowPassPush,
        slope_bias: f32,
        cam_pos: [f32; 3],
    ) -> u32 {
        if self.instanced_clusters.is_empty() {
            return 0;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("shadow instanced"));
        self.bind_shadow_pass_shared(enc, shadow_pipeline, push, slope_bias);

        let mut draw_calls: u32 = 0;
        for cluster in &self.instanced_clusters {
            // Shadows only read each bucket's matrices (per-instance vertex
            // bytes), so borrow them: no LOD-bucket clone, which otherwise
            // recurred once per cascade.
            cluster.for_each_lod_bucket(cam_pos, |index_offset, index_count, instances| {
                let index_byte_offset = index_offset * std::mem::size_of::<u32>();
                for &model in instances {
                    let model_uniforms = ModelUniforms { model };
                    unsafe {
                        enc.setVertexBytes_length_atIndex(
                            std::ptr::NonNull::from(&model_uniforms).cast(),
                            std::mem::size_of::<ModelUniforms>(),
                            2,
                        );
                        enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                            MTLPrimitiveType::Triangle,
                            index_count,
                            MTLIndexType::UInt32,
                            &self.index_buffer,
                            index_byte_offset,
                        );
                    }
                    draw_calls += 1;
                }
            });
        }
        enc.popDebugGroup();
        draw_calls
    }

    // Encode shadow draws for skinned meshes (deformed depth, drawn last
    // in the cascade).
    fn encode_shadow_skinned_into(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        push: &ShadowPassPush,
        slope_bias: f32,
        cam_pos: [f32; 3],
        skinned_joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> u32 {
        let mut draw_calls: u32 = 0;
        let (Some(skinned_shadow_ps), Some(svb), Some(sib)) = (
            &self.skinned.shadow_pipeline_state,
            &self.skinned.vertex_buffer,
            &self.skinned.index_buffer,
        ) else {
            return draw_calls;
        };
        if self.skinned.draw_objects.is_empty() {
            return draw_calls;
        }
        enc.pushDebugGroup(&objc2_foundation::NSString::from_str("shadow skinned"));
        // The skinned-shadow path uses its own pipeline; we still need the
        // shared shadow uniforms + cascade push + depth state, so route
        // through `bind_shadow_pass_shared` first: `setRenderPipelineState`
        // below overwrites the pipeline binding with the skinned variant.
        self.bind_shadow_pass_shared(enc, skinned_shadow_ps, push, slope_bias);
        unsafe {
            enc.setVertexBuffer_offset_atIndex(Some(svb), 0, 1);
        }
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let model_uniforms = ModelUniforms { model: obj.model };
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let index_byte_offset = index_offset * std::mem::size_of::<u16>();
            unsafe {
                enc.setVertexBytes_length_atIndex(
                    std::ptr::NonNull::from(&model_uniforms).cast(),
                    std::mem::size_of::<ModelUniforms>(),
                    2,
                );
                enc.setVertexBuffer_offset_atIndex(Some(&skinned_joint_bufs[i]), 0, 8);
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    index_count,
                    MTLIndexType::UInt16,
                    sib,
                    index_byte_offset,
                );
            }
            draw_calls += 1;
        }
        enc.popDebugGroup();
        draw_calls
    }
}
