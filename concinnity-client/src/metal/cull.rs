// src/metal/cull.rs
//
// GPU-driven cull support for the Metal frame encoder: per-frame object /
// draw-args / joint buffer construction, the cull compute pass, and the
// bindless texture argument buffer.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLArgumentEncoder, MTLCommandBuffer as _, MTLComputePassDescriptor, MTLComputePipelineState,
    MTLDevice as _, MTLFunction as _, MTLLibrary as _, MTLRenderCommandEncoder as _,
    MTLRenderPipelineState,
};

use super::context::*;
use super::pipeline::{ns_str, shader_source};
use super::scoped_encoder::ScopedEncoder;
use super::uniforms::*;

// Re-export the camera-distance helper under the legacy local name so the
// existing draw_args builder reads naturally; the actual implementation
// lives on the backend-agnostic `gfx::lod` module.
use crate::gfx::lod::camera_distance as lod_camera_distance;

// All GPU-driven cull state grouped into one feature unit: the phase-1 +
// phase-2 cull pipelines, their indirect command buffers + argument
// encoders/buffers, the per-object status buffer, the two-pass-occlusion
// toggle, and the Hi-Z depth pyramid + the view-projection snapshots the
// occlusion test reprojects through. All `Some`/active only on the bindless
// path; non-bindless shaders keep the legacy per-draw CPU loop and leave
// every field `None` / default.
pub(crate) struct CullState {
    // GPU-driven cull pipeline. `Some` only when `bindless` is set.
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    // Indirect command buffer the cull kernel encodes draws into; rebuilt by
    // `ensure_icb_capacity` when the draw list outgrows it.
    pub icb: Option<Retained<ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>>,
    // Encoder that writes `icb` into the kernel's argument buffer.
    pub icb_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    // Argument buffer holding the encoded reference to `icb`.
    pub icb_arg_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Command capacity of `icb`; 0 until first built. `icb_2` + `status_buffer`
    // grow in lockstep with it.
    pub icb_capacity: usize,
    // Second-pass cull pipeline for two-pass occlusion. `Some` whenever
    // `pipeline` is; used only when `two_pass_occlusion` is on.
    pub pipeline_phase2: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    // Second indirect command buffer holding the phase-2 (disocclusion) draws.
    pub icb_2: Option<Retained<ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>>,
    // Argument encoder + buffer wiring `icb_2` into the phase-2 kernel.
    pub icb_2_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub icb_2_arg_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Per-object status buffer (one `u32` each): phase-1 cull writes it,
    // phase-2 cull reads it. Private storage, never CPU-touched.
    pub status_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Two-pass Hi-Z occlusion toggle, resolved from
    // `PostProcessConfig.occlusion_two_pass`.
    pub two_pass_occlusion: bool,
    // Hi-Z (depth-mip pyramid) used by the cull kernel for occlusion culling.
    // Built at the end of `draw_frame`, consumed by the next frame's cull
    // dispatch (projected through `prev_view_proj`).
    pub hiz: Option<super::hiz::HiZResources>,
    // Previous frame's un-jittered view-projection, captured every Hi-Z build.
    // The next frame's cull kernel projects AABBs through it. Distinct from the
    // TAA `prev_view_proj`.
    pub prev_view_proj: [[f32; 4]; 4],
    // This frame's un-jittered view-projection, captured before `execute_graph`
    // so the phase-2 cull can project AABBs against the freshly built pyramid.
    pub cur_view_proj: [[f32; 4]; 4],
    // `false` on the first frame and after a resize; while false the cull
    // kernel skips the Hi-Z test. Flipped `true` after the first build.
    pub hiz_valid: bool,
    // GPU-driven cascaded shadow. All `Some` only on the bindless
    // path with shadows enabled (`bindless && shadow_map_size > 0`); non-bindless
    // / custom-shader worlds leave them `None` and keep the legacy per-cascade
    // CPU shadow loop. The frustum-only `cull_encode_shadow` kernel
    // (`shadow_pipeline`) writes per-cascade indirect commands into one shadow
    // ICB holding `NUM_SHADOW_CASCADES * cull_count()` slots (cascade `c` at base
    // `c * cull_count()`); the depth-only `shadow_bindless_pipeline` then issues
    // each cascade's range. The ICB + its argument buffer grow in lockstep via
    // `ensure_shadow_icb_capacity`.
    pub shadow_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub shadow_bindless_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub shadow_icb: Option<Retained<ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>>>,
    pub shadow_icb_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub shadow_icb_arg_buffer: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Command capacity of `shadow_icb` (across all cascades); 0 until first built.
    pub shadow_icb_capacity: usize,
}

// Kernel buffer index the cull kernel expects its `ICBContainer` argument
// buffer at. The argument encoder that fills that argument buffer is built
// for this same index.
pub(super) const CULL_ICB_BUFFER_INDEX: usize = 4;

// The six flat bindless-pool texture indices the Metal bindless fragment shader
// (`fragment_main_bindless`, default.metal) reads for one surface. The shared
// pool is laid out as [albedo textures..][normal maps..] and the shader indexes
// it DIRECTLY (`tex_pool[obj.normal_index]`, no in-shader bias), so the CPU must
// bias every normal-region slot by `albedo_count` (primary + secondary normal),
// clamp each to its pool size, then clamp to the `BINDLESS_TEXTURE_COUNT` cap
// (the pool is a fixed-size MSL array, so an over-cap index would read past it).
// DX/VK bias the normal region IN-shader and leave these raw, which is why this
// convention lives in the Metal backend, not the shared `pack_object_record` /
// `instance_object_records` core helpers. Shared by the per-frame static fill
// (`build_object_buffer`) and the init-time instanced records
// (`metal_instance_records`) so a folded instance addresses the pool identically
// to a static object.
pub(super) struct FlatPoolIndices {
    pub albedo: u32,
    pub normal: u32,
    pub albedo_secondary: u32,
    pub normal_secondary: u32,
    pub emissive: u32,
    pub orm: u32,
}

pub(super) fn metal_flat_pool_indices(
    albedo_count: usize,
    normal_count: usize,
    texture_slot: usize,
    normal_map_slot: usize,
    material: &crate::gfx::render_types::MaterialUniforms,
) -> FlatPoolIndices {
    let cap = (super::context::BINDLESS_TEXTURE_COUNT as u32).saturating_sub(1);
    let last_albedo = albedo_count.saturating_sub(1);
    let last_normal = normal_count.saturating_sub(1);
    // Albedo-region slots index the pool directly; normal-region slots follow
    // the albedo block, so bias by the albedo count. Both clamp to the cap.
    let albedo_region = |slot: usize| (slot.min(last_albedo) as u32).min(cap);
    let normal_region = |slot: usize| ((albedo_count + slot.min(last_normal)) as u32).min(cap);
    FlatPoolIndices {
        albedo: albedo_region(texture_slot),
        normal: normal_region(normal_map_slot),
        albedo_secondary: albedo_region(material.albedo_secondary_index as usize),
        normal_secondary: normal_region(material.normal_secondary_index as usize),
        emissive: albedo_region(material.emissive_map_index as usize),
        orm: albedo_region(material.orm_map_index as usize),
    }
}

// Build the GPU-driven bindless instanced records: one `GpuObjectData` per
// cluster instance, in cluster-then-instance order, addressing the bindless pool
// with the SAME convention as `build_object_buffer`'s static fill (via
// `metal_flat_pool_indices`). Bounds are the cluster's mesh-local AABB
// transformed by each instance's model, so the cull kernel frustum/distance/
// Hi-Z-tests each instance independently. Built once at init (instances are
// placed at world load and never move) and appended to the per-frame object
// buffer after the static records. Deliberately NOT the shared core
// `instance_object_records`: that helper uses the DX/VK raw-index convention
// (their shaders bias the normal region), which would mis-address Metal's flat
// pool -- a cluster with a secondary normal map would sample an albedo texture.
pub(super) fn metal_instance_records(
    clusters: &[crate::gfx::render_types::InstancedCluster],
    albedo_count: usize,
    normal_count: usize,
) -> Vec<crate::gfx::render_types::GpuObjectData> {
    use crate::gfx::render_types::GpuObjectData;
    let total: usize = clusters.iter().map(|c| c.instances.len()).sum();
    let mut records = Vec::with_capacity(total);
    for cluster in clusters {
        let idx = metal_flat_pool_indices(
            albedo_count,
            normal_count,
            cluster.texture_slot,
            cluster.normal_map_slot,
            &cluster.material,
        );
        for &model in &cluster.instances {
            let (bb_min, bb_max) = crate::gfx::frustum::transform_aabb(
                cluster.local_bb_min,
                cluster.local_bb_max,
                model,
            );
            records.push(GpuObjectData {
                model,
                tint: cluster.material.tint,
                roughness: cluster.material.roughness,
                emissive: cluster.material.emissive,
                metallic: cluster.material.metallic,
                albedo_index: idx.albedo,
                normal_index: idx.normal,
                macro_variation: cluster.material.macro_variation,
                terrain_blend: cluster.material.terrain_blend,
                bb_min,
                cull_distance: cluster.cull_distance,
                bb_max,
                secondary_blend_sharpness: cluster.material.secondary_blend_sharpness,
                albedo_secondary_index: idx.albedo_secondary,
                normal_secondary_index: idx.normal_secondary,
                emissive_map_index: idx.emissive,
                orm_map_index: idx.orm,
            });
        }
    }
    records
}

// Build one GPU-driven bindless record for a skinned object. Reuses
// the core `pack_skinned_record` for the padded bind-pose AABB + model +
// material, then overwrites the texture indices with Metal's flat-pool
// convention: the core helper (like `pack_object_record`) leaves the secondary
// / normal-secondary / emissive / ORM indices raw for DX/VK's in-shader bias,
// but Metal's bindless shader indexes the flat pool directly, so they must be
// pre-biased + capped via `metal_flat_pool_indices` -- the same convention the
// static + instanced records use.
pub(super) fn metal_skinned_record(
    obj: &crate::gfx::render_types::SkinnedDrawObject,
    albedo_count: usize,
    normal_count: usize,
) -> crate::gfx::render_types::GpuObjectData {
    let idx = metal_flat_pool_indices(
        albedo_count,
        normal_count,
        obj.texture_slot,
        obj.normal_map_slot,
        &obj.material,
    );
    let mut rec = crate::gfx::render_types::pack_skinned_record(obj, idx.albedo, idx.normal);
    rec.albedo_secondary_index = idx.albedo_secondary;
    rec.normal_secondary_index = idx.normal_secondary;
    rec.emissive_map_index = idx.emissive;
    rec.orm_map_index = idx.orm;
    rec
}

impl MtlContext {
    // Build the current-pose joint-palette buffers for the skinned passes, one
    // per skinned object, from the per-object ring at `ring_slot`. Returns an
    // empty vec when there are no skinned meshes. The ring reuses persistent
    // shared buffers across frames (the frames-in-flight fence guarantees the
    // slot is no longer GPU-read before it is overwritten) instead of minting
    // a fresh buffer per object per frame.
    pub(super) fn build_joint_buffers(
        &mut self,
        ring_slot: usize,
    ) -> Result<Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        self.joint_ring
            .write_all(&self.device, ring_slot, &self.skinned.joint_matrices)
    }

    // Build the previous-pose joint-palette buffers the velocity pre-pass
    // reprojects from, in a separate ring so they never alias the current
    // pose within the same frame.
    pub(super) fn build_prev_joint_buffers(
        &mut self,
        ring_slot: usize,
    ) -> Result<Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        self.prev_joint_ring
            .write_all(&self.device, ring_slot, &self.skinned.prev_joint_matrices)
    }

    // Build the per-frame `GpuObjectData` buffer for the bindless static
    // pass: one record per `DrawObject`, indexed by the object id the draw
    // call passes as `[[base_instance]]`. Returns `None` when there is no
    // static geometry. Rebuilt every frame so `update_model` /
    // `update_visibility` changes are reflected; the committed command buffer
    // keeps the transient buffer alive until the GPU is done with it.
    pub(super) fn build_object_buffer(
        &mut self,
        ring_slot: usize,
    ) -> Result<Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        use crate::gfx::render_types::GpuObjectData;
        if self.draw_objects.is_empty() {
            return Ok(None);
        }
        let albedo_count = self.textures.len();
        let normal_count = self.normal_map_textures.len();
        // Reuse a persistent scratch Vec across frames; `mem::take` lifts it out
        // so the build loop borrows only `draw_objects` while the ring + device
        // borrows below stay on disjoint fields.
        let mut objects = std::mem::take(&mut self.object_scratch);
        objects.clear();
        for obj in &self.draw_objects {
            // Flat bindless-pool indices (normal region biased by the albedo
            // count, all clamped to the cap); the identical mapping the folded
            // instance records use, so static + instances address the pool the
            // same way.
            let idx = metal_flat_pool_indices(
                albedo_count,
                normal_count,
                obj.texture_slot,
                obj.normal_map_slot,
                &obj.material,
            );
            objects.push(GpuObjectData {
                model: obj.model,
                tint: obj.material.tint,
                roughness: obj.material.roughness,
                emissive: obj.material.emissive,
                metallic: obj.material.metallic,
                albedo_index: idx.albedo,
                normal_index: idx.normal,
                macro_variation: obj.material.macro_variation,
                terrain_blend: obj.material.terrain_blend,
                bb_min: obj.bb_min,
                cull_distance: obj.cull_distance,
                bb_max: obj.bb_max,
                secondary_blend_sharpness: obj.material.secondary_blend_sharpness,
                albedo_secondary_index: idx.albedo_secondary,
                normal_secondary_index: idx.normal_secondary,
                emissive_map_index: idx.emissive,
                orm_map_index: idx.orm,
            });
        }
        // Fold the instanced clusters into the same buffer: each instance's
        // pre-built record is appended after the static objects so one cull
        // dispatch + one indirect draw cover both (the ring auto-grows to the
        // written slice). The records are static (built once at init), so this
        // is a memcpy. `objects` was `mem::take`n, so this borrows only
        // `instance_records`, leaving the other fields free.
        if self.n_instances > 0 {
            objects.extend_from_slice(&self.instance_records);
        }
        // Append a record per skinned object: the compute-deformed
        // geometry draws as rigid static geometry, so it folds into the same
        // cull. Rebuilt every frame (the record's AABB + model follow obj.model,
        // which animates), unlike the cached static instance records.
        if self.n_skinned > 0 {
            for obj in &self.skinned.draw_objects {
                objects.push(metal_skinned_record(obj, albedo_count, normal_count));
            }
        }
        let result = self.object_ring.write(
            &self.device,
            ring_slot,
            super::context::bytes_of_slice(&objects),
        );
        self.object_scratch = objects;
        result.map(Some)
    }

    // Build the per-frame `GpuDrawArgs` buffer for the GPU-driven cull pass:
    // one record per `DrawObject` (same indexing as the `GpuObjectData`
    // buffer), carrying the indexed-draw arguments the cull kernel encodes
    // into the indirect command buffer plus the per-frame cull-decision bits.
    // Returns `None` when there is no static geometry. Rebuilt every frame so
    // `update_visibility` / streaming residency changes (*and* per-frame LOD
    // swaps driven by camera distance) take effect.
    pub(super) fn build_draw_args_buffer(
        &mut self,
        cam_pos: [f32; 3],
        ring_slot: usize,
    ) -> Result<Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        use crate::gfx::render_types::{GpuDrawArgs, draw_args_flags};
        if self.draw_objects.is_empty() {
            return Ok(None);
        }
        let mut args = std::mem::take(&mut self.draw_args_scratch);
        args.clear();
        for obj in &self.draw_objects {
            // Pick this frame's active LOD by camera distance: the bindless
            // main pass then renders the chosen slice with no shader-side
            // change. Objects with no alternates fall straight through to LOD0.
            let d = lod_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            args.push(GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                flags: draw_args_flags(obj.visible, obj.resident, obj.cullable()),
            });
        }
        // Append the instances' draw args in the SAME cluster-then-instance
        // order as `instance_records`, so cull index `draw_objects.len() + k`
        // reads matching object + draw-args records. Static (base LOD only),
        // so a memcpy; per-instance LOD would move this build per-frame.
        if self.n_instances > 0 {
            args.extend_from_slice(&self.instance_draw_args);
        }
        // Skinned draw args: one per skinned object, the active-LOD
        // slice into the skinned u16 index buffer with base_vertex 0 (the
        // deformed buffer mirrors global skinned indexing). Cullable + gated on
        // obj.visible; rebuilt every frame (pose-driven LOD + visibility). The
        // cull kernel routes records at/after `skinned_record_base()` through the
        // skinned index buffer (see encode_cull's `skinned_base`).
        if self.n_skinned > 0 {
            for obj in &self.skinned.draw_objects {
                let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
                let (index_offset, index_count) = obj.active_lod(d);
                args.push(GpuDrawArgs {
                    index_count: index_count as u32,
                    index_offset: index_offset as u32,
                    base_vertex: 0,
                    flags: draw_args_flags(obj.visible, true, true),
                });
            }
        }
        let result = self.draw_args_ring.write(
            &self.device,
            ring_slot,
            super::context::bytes_of_slice(&args),
        );
        self.draw_args_scratch = args;
        result.map(Some)
    }

    // Encode the GPU-driven cull compute pass: one thread per
    // `DrawObject` frustum/distance-tests the object and either encodes an
    // indexed draw into `cull_icb` or resets that command slot to a no-op.
    // The bindless main pass then issues the whole buffer with one
    // `executeCommandsInBuffer`. A no-op when the cull pipeline / ICB are not
    // set up (non-bindless contexts) or there is no geometry.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_cull(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        object_buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        draw_args_buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
    ) -> Result<u32, String> {
        use objc2_metal::{
            MTLComputeCommandEncoder as _, MTLComputePipelineState as _, MTLResourceUsage, MTLSize,
        };
        let (Some(pipeline), Some(icb), Some(arg_buf)) = (
            &self.cull.pipeline,
            &self.cull.icb,
            &self.cull.icb_arg_buffer,
        ) else {
            return Ok(0);
        };
        // Static objects + folded instances: the kernel tests one thread per
        // record and encodes survivors into the ICB.
        let object_count = self.cull_count();
        if object_count == 0 {
            return Ok(0);
        }

        // Pack the six already-normalised frustum planes for the kernel.
        let mut planes = [[0.0f32; 4]; 6];
        for (i, p) in frustum.planes.iter().enumerate() {
            planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }
        // Hi-Z occlusion metadata + binding. `hiz_enabled` is gated on the
        // per-context `hiz_valid` flag (false on the first frame and right
        // after a resize, before a valid pyramid has been built) and on whether
        // a `HiZResources` exists (built at init exactly when the cull pipeline
        // is). The texture is bound unconditionally when present so the kernel's
        // `texture(0)` always resolves; the `hiz_enabled` flag is what actually
        // gates the sample.
        let (hiz_tex, hiz_size, hiz_mip_count, hiz_enabled) = match self.cull.hiz.as_ref() {
            Some(h) => (
                Some(h.texture.as_ref()),
                [h.width as f32, h.height as f32],
                h.mip_count,
                if self.cull.hiz_valid { 1u32 } else { 0u32 },
            ),
            None => (None, [1.0, 1.0], 1, 0u32),
        };
        let cull_uniforms = CullUniforms {
            planes,
            cam_pos,
            object_count: object_count as u32,
            prev_view_proj: self.cull.prev_view_proj,
            hiz_size,
            hiz_mip_count,
            hiz_enabled,
            skinned_base: self.skinned_record_base() as u32,
            // Main cull writes at `tid` (cascade_base 0); the shadow cull is the
            // only path that offsets by cascade.
            cascade_base: 0,
            _pad_skin: [0; 2],
        };

        let cull_pass_desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.pass_timing {
            t.attach_compute(&cull_pass_desc, super::pass_timing::PassId::Cull);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&cull_pass_desc)
                .ok_or("failed to get compute encoder")?,
            "cull phase1",
        );
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(object_buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(draw_args_buffer), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&cull_uniforms).cast(),
                std::mem::size_of::<CullUniforms>(),
                2,
            );
            enc.setBuffer_offset_atIndex(Some(&self.index_buffer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(arg_buf), 0, CULL_ICB_BUFFER_INDEX);
            // Per-object cull status at buffer(5). Always allocated alongside
            // the ICB, so always bound; the kernel writes it unconditionally
            // and phase 2 reads it under two-pass occlusion (ignored under
            // single-pass). A missing buffer would be UB, so require it. The
            // ScopedEncoder ensures this `?` can't leak an open encoder.
            let status = self
                .cull
                .status_buffer
                .as_ref()
                .ok_or("cull status buffer missing")?;
            enc.setBuffer_offset_atIndex(Some(status), 0, 5);
            // Skinned u16 index buffer at buffer(6): the kernel bakes it into the
            // indirect command for records at/after `skinned_base`. Bound
            // unconditionally (Metal requires a buffer the kernel references to be
            // bound even under a never-taken branch); the static index buffer is a
            // harmless placeholder when no skinned mesh is folded (skinned_base ==
            // object_count then, so the skinned branch never fires).
            enc.setBuffer_offset_atIndex(Some(self.skinned_index_or_placeholder()), 0, 6);
            // Hi-Z depth pyramid at texture(0). Bound directly (not via an
            // argument buffer), so Metal tracks its residency automatically.
            // Always bound when present; `hiz_enabled` decides whether it's read.
            if let Some(tex) = hiz_tex {
                enc.setTexture_atIndex(Some(tex), 0);
            }
        }
        // The kernel writes draw commands into the ICB through the argument
        // buffer, so the ICB must be declared resident for the compute pass.
        enc.useResource_usage(ProtocolObject::from_ref(&**icb), MTLResourceUsage::Write);

        // One thread per draw object, non-uniform grid: no remainder branch
        // needed beyond the kernel's own bounds guard.
        let tg = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 64);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: object_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        Ok(0)
    }

    // Encode the phase-2 GPU cull for two-pass occlusion. Runs after the Hi-Z
    // pyramid has been rebuilt mid-frame from phase-1 depth (`encode_hiz_build`
    // dispatched as the `HizBuild` graph pass). One thread per `DrawObject`
    // re-tests the objects phase 1 marked `STATUS_HIZ_CANDIDATE` against the
    // fresh pyramid, projecting through *this* frame's view-projection
    // (`cull_cur_view_proj`), and encodes a draw into `cull_icb_2` for any
    // that turn out visible. `Main2` then issues `cull_icb_2`. A no-op when the
    // phase-2 pipeline / ICB are not set up (two-pass off, or non-bindless).
    pub(in crate::metal) fn encode_cull_phase2(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        object_buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        draw_args_buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
    ) -> Result<u32, String> {
        use objc2_metal::{
            MTLComputeCommandEncoder as _, MTLComputePipelineState as _, MTLResourceUsage, MTLSize,
        };
        let (Some(pipeline), Some(icb), Some(arg_buf), Some(status), Some(hiz)) = (
            &self.cull.pipeline_phase2,
            &self.cull.icb_2,
            &self.cull.icb_2_arg_buffer,
            &self.cull.status_buffer,
            self.cull.hiz.as_ref(),
        ) else {
            return Ok(0);
        };
        // Static objects + folded instances re-tested against the fresh Hi-Z
        // pyramid; same record count as phase 1.
        let object_count = self.cull_count();
        if object_count == 0 {
            return Ok(0);
        }

        // Frustum planes are unused by the phase-2 kernel (candidates already
        // passed the frustum test in phase 1) but the uniform layout is shared,
        // so pack them anyway for a clean struct.
        let mut planes = [[0.0f32; 4]; 6];
        for (i, p) in frustum.planes.iter().enumerate() {
            planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }
        // Project AABBs through this frame's un-jittered VP: it matches the
        // pyramid we just rebuilt from this frame's depth. `hiz_enabled = 1`:
        // the `HizBuild` pass always precedes this dispatch in the graph, so a
        // valid pyramid is guaranteed (the kernel still guards defensively).
        let cull_uniforms = CullUniforms {
            planes,
            cam_pos,
            object_count: object_count as u32,
            prev_view_proj: self.cull.cur_view_proj,
            hiz_size: [hiz.width as f32, hiz.height as f32],
            hiz_mip_count: hiz.mip_count,
            hiz_enabled: 1,
            skinned_base: self.skinned_record_base() as u32,
            cascade_base: 0,
            _pad_skin: [0; 2],
        };

        let cull_pass_desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.pass_timing {
            t.attach_compute(&cull_pass_desc, super::pass_timing::PassId::Cull2);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&cull_pass_desc)
                .ok_or("failed to get compute encoder")?,
            "cull phase2",
        );
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(object_buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(draw_args_buffer), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&cull_uniforms).cast(),
                std::mem::size_of::<CullUniforms>(),
                2,
            );
            enc.setBuffer_offset_atIndex(Some(&self.index_buffer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(arg_buf), 0, CULL_ICB_BUFFER_INDEX);
            enc.setBuffer_offset_atIndex(Some(status), 0, 5);
            // Skinned u16 index buffer at buffer(6); see encode_cull. Phase 2
            // of two-pass occlusion re-tests the same records, so the skinned
            // tail is handled here too.
            enc.setBuffer_offset_atIndex(Some(self.skinned_index_or_placeholder()), 0, 6);
            enc.setTexture_atIndex(Some(hiz.texture.as_ref()), 0);
        }
        // The kernel writes draw commands into the phase-2 ICB through the
        // argument buffer, so that ICB must be declared resident here too.
        enc.useResource_usage(ProtocolObject::from_ref(&**icb), MTLResourceUsage::Write);

        let tg = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 64);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: object_count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: tg,
                height: 1,
                depth: 1,
            },
        );
        Ok(0)
    }

    // Encode the GPU-driven cascaded-shadow cull: one
    // `cull_encode_shadow` dispatch per re-rendered cascade (gated by
    // `shadow_render_mask`), each frustum-testing every record against that
    // cascade's LIGHT frustum and encoding survivors into the cascade's slice of
    // the shared shadow ICB (`cascade_base = c * cull_count()`). Hi-Z + distance
    // are off (frustum only). A no-op when the shadow-bindless path is inactive
    // or there is no geometry.
    //
    // Runs as a compute prologue in the SAME command buffer as the main `Cull`
    // pass (dispatched right after `encode_cull` from the graph executor's Cull
    // arm), so the shadow ICB write lands in a command buffer committed before
    // the `Shadow` render pass's command buffer -- the exact cross-command-buffer
    // FIFO ordering the main cull -> main ICB already relies on. No explicit
    // barrier (Metal has none); residency is declared with `useResource`.
    // pub(in crate::metal) so the graph executor can dispatch it.
    pub(in crate::metal) fn encode_shadow_culls(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        object_buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        draw_args_buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
    ) -> Result<(), String> {
        use crate::gfx::render_types::NUM_SHADOW_CASCADES;
        use objc2_metal::{
            MTLComputeCommandEncoder as _, MTLComputePipelineState as _, MTLResourceUsage, MTLSize,
        };
        let (Some(pipeline), Some(icb), Some(arg_buf)) = (
            &self.cull.shadow_pipeline,
            &self.cull.shadow_icb,
            &self.cull.shadow_icb_arg_buffer,
        ) else {
            return Ok(());
        };
        let object_count = self.cull_count();
        if object_count == 0 {
            return Ok(());
        }
        // Same cascade set the shadow render pass refreshes this frame; a skipped
        // cascade keeps its prior depth slice, so its cull dispatch + ICB region
        // are left untouched.
        let all = (1u32 << NUM_SHADOW_CASCADES) - 1;
        let mask = if self.shadow_render_mask == 0 {
            all
        } else {
            self.shadow_render_mask
        };

        let cull_pass_desc = MTLComputePassDescriptor::new();
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&cull_pass_desc)
                .ok_or("failed to get shadow cull compute encoder")?,
            "shadow cull",
        );
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(object_buffer), 0, 0);
            enc.setBuffer_offset_atIndex(Some(draw_args_buffer), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&self.index_buffer), 0, 3);
            enc.setBuffer_offset_atIndex(Some(arg_buf), 0, CULL_ICB_BUFFER_INDEX);
            // Skinned u16 index buffer at buffer(6); the kernel bakes it into the
            // skinned-tail commands exactly like the main cull.
            enc.setBuffer_offset_atIndex(Some(self.skinned_index_or_placeholder()), 0, 6);
        }
        // The kernel writes draw commands into the shadow ICB through the
        // argument buffer, so it must be resident for the compute pass.
        enc.useResource_usage(ProtocolObject::from_ref(&**icb), MTLResourceUsage::Write);

        let tg = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 64);
        let skinned_base = self.skinned_record_base() as u32;
        for c in 0..NUM_SHADOW_CASCADES {
            if mask & (1u32 << c) == 0 {
                continue;
            }
            // Cascade light frustum: world-space planes from the cascade's light
            // view-projection (the caster-extent near push baked into light_vps
            // survives, so off-screen / tall casters are kept).
            let frustum = crate::gfx::frustum::Frustum::from_view_projection(
                self.shadow_uniforms.light_vps[c],
            );
            let mut planes = [[0.0f32; 4]; 6];
            for (i, p) in frustum.planes.iter().enumerate() {
                planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
            }
            let cull_uniforms = CullUniforms {
                planes,
                // Unused by the shadow kernel (no distance cull); kept zero.
                cam_pos: [0.0; 3],
                object_count: object_count as u32,
                // Unused (Hi-Z disabled); identity keeps the struct clean.
                prev_view_proj: super::math::IDENTITY4,
                hiz_size: [1.0, 1.0],
                hiz_mip_count: 1,
                hiz_enabled: 0,
                skinned_base,
                cascade_base: (c * object_count) as u32,
                _pad_skin: [0; 2],
            };
            unsafe {
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::from(&cull_uniforms).cast(),
                    std::mem::size_of::<CullUniforms>(),
                    2,
                );
            }
            enc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: object_count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Ok(())
    }

    // Build the per-frame `BindlessTextures` argument buffer for the bindless
    // static pass: the albedo + normal-map pool (every one of the
    // `BINDLESS_TEXTURE_COUNT` slots filled: overflow and trailing empty
    // slots fall back to the white albedo texture at slot 0) followed by the
    // shadow map and the two IBL cubes. The bindless fragment shader can only
    // reach textures through this argument buffer because discrete texture
    // bindings make it incompatible with indirect command buffers. A fresh
    // buffer is allocated each frame (like the object / draw-args buffers)
    // so a streamed texture swap is picked up and the GPU never reads a buffer
    // the next frame's CPU encode is rewriting. `None` for non-bindless
    // contexts. The committed command buffer keeps the buffer alive.
    pub(super) fn build_bindless_texture_args(
        &mut self,
        ring_slot: usize,
    ) -> Result<Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>, String> {
        use objc2_metal::MTLArgumentEncoder as _;
        // Clone the encoder handle (cheap refcount bump) so no borrow of `self`
        // is held while the ring (a different field) is borrowed mutably below.
        let enc = match &self.bindless_tex_arg_encoder {
            Some(e) => e.clone(),
            None => return Ok(None),
        };
        let len = enc.encodedLength().max(16);
        // Ring slot, grown to the encoder's `encodedLength()` instead of a fresh
        // allocation each frame. The argument encoder rewrites it in place; the
        // fence guarantees the prior user of this slot has retired on the GPU.
        let buf = self.bindless_tex_ring.slot(&self.device, ring_slot, len)?;
        // SAFETY: `buf` was sized to the encoder's `encodedLength()`, and every
        // texture index below is within the `BindlessTextures` layout.
        unsafe {
            enc.setArgumentBuffer_offset(Some(&buf), 0);
        }
        let count = super::context::BINDLESS_TEXTURE_COUNT;
        let albedo_count = self.textures.len();
        let normal_count = self.normal_map_textures.len();
        for i in 0..count {
            let tex = if i < albedo_count {
                self.textures[i].as_ref()
            } else if i < albedo_count + normal_count {
                self.normal_map_textures[i - albedo_count].as_ref()
            } else {
                self.textures[0].as_ref()
            };
            unsafe {
                enc.setTexture_atIndex(Some(tex), i);
            }
        }
        unsafe {
            enc.setTexture_atIndex(Some(self.shadow_map.as_ref()), count);
            enc.setTexture_atIndex(Some(self.env_map.irradiance.as_ref()), count + 1);
            enc.setTexture_atIndex(Some(self.env_map.prefilter.as_ref()), count + 2);
            // SSAO occlusion: the blurred AO when SSAO is on, else 1×1 white.
            enc.setTexture_atIndex(Some(self.ao_output_texture()), count + 3);
        }
        Ok(Some(buf))
    }
    // Declare every texture the bindless pass samples resident for the
    // indirect command buffer. The textures are referenced through the
    // `BindlessTextures` argument buffer rather than bound on the encoder, so
    // the indirect execution cannot see them unless they are explicitly used.
    pub(super) fn use_bindless_textures(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
    ) {
        use objc2_metal::{MTLRenderStages, MTLResourceUsage};
        for tex in self.textures.iter().chain(self.normal_map_textures.iter()) {
            encoder.useResource_usage_stages(
                ProtocolObject::from_ref(&**tex),
                MTLResourceUsage::Read,
                MTLRenderStages::Fragment,
            );
        }
        for tex in [
            &self.shadow_map,
            &self.env_map.irradiance,
            &self.env_map.prefilter,
        ] {
            encoder.useResource_usage_stages(
                ProtocolObject::from_ref(&**tex),
                MTLResourceUsage::Read,
                MTLRenderStages::Fragment,
            );
        }
        // SSAO occlusion travels in the BindlessTextures argument buffer too.
        encoder.useResource_usage_stages(
            ProtocolObject::from_ref(self.ao_output_texture()),
            MTLResourceUsage::Read,
            MTLRenderStages::Fragment,
        );
    }
}

// The GPU-driven cull stage: a compute pipeline plus the argument encoder
// that wires an `MTLIndirectCommandBuffer` into the kernel. The kernel
// reaches the ICB only through an argument buffer, so the encoder must be
// kept to (re)encode that argument buffer whenever the ICB is recreated.
//
// The phase-2 pipeline + its argument encoder drive two-pass occlusion: the
// `cull_encode_phase2` kernel re-tests phase-1's Hi-Z-occluded objects against
// the rebuilt pyramid and encodes survivors into a second ICB. Built from the
// same library whenever the bindless path is active; used only when
// `occlusion_two_pass` is on.
pub(super) struct CullPipeline {
    pub state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub icb_arg_encoder: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
    pub state_phase2: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub icb2_arg_encoder: Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
}

// Build the GPU-driven cull pipeline. The `cull_encode` kernel runs
// one thread per `DrawObject`: it frustum/distance-tests the object against
// `CullUniforms` and, for survivors, encodes an indexed draw into the
// indirect command buffer; culled or disabled objects have their command
// reset to a no-op. The render pass then issues the whole buffer with one
// `executeCommandsInBuffer`, so the CPU never walks the draw list.
//
// The frustum and distance maths mirror `gfx::frustum` exactly (the six
// planes are extracted CPU-side and handed in already normalised), so the
// GPU path culls identically to the CPU BVH path it replaces.
pub(super) fn build_cull_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<CullPipeline, String> {
    let msl = shader_source(hot_reload, "cull.metal");

    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("cull shader compile error: {:?}", e))?;

    let cull_fn = library
        .newFunctionWithName(&ns_str("cull_encode"))
        .ok_or("cull_encode not found in cull library")?;

    let state = device
        .newComputePipelineStateWithFunction_error(&cull_fn)
        .map_err(|e| format!("failed to create cull pipeline state: {:?}", e))?;

    // SAFETY: CULL_ICB_BUFFER_INDEX is the static buffer index the kernel
    // declares its argument-buffer parameter at.
    let icb_arg_encoder =
        unsafe { cull_fn.newArgumentEncoderWithBufferIndex(CULL_ICB_BUFFER_INDEX) };

    // Second-pass cull (two-pass occlusion): same library, the
    // `cull_encode_phase2` kernel. It declares its ICB argument buffer at the
    // same buffer index, so the encoder is built the same way but tied to the
    // second-pass function.
    let cull_fn_phase2 = library
        .newFunctionWithName(&ns_str("cull_encode_phase2"))
        .ok_or("cull_encode_phase2 not found in cull library")?;
    let state_phase2 = device
        .newComputePipelineStateWithFunction_error(&cull_fn_phase2)
        .map_err(|e| format!("failed to create phase-2 cull pipeline state: {:?}", e))?;
    // SAFETY: same static buffer index: `cull_encode_phase2` declares its
    // ICBContainer argument buffer at CULL_ICB_BUFFER_INDEX too.
    let icb2_arg_encoder =
        unsafe { cull_fn_phase2.newArgumentEncoderWithBufferIndex(CULL_ICB_BUFFER_INDEX) };

    Ok(CullPipeline {
        state,
        icb_arg_encoder,
        state_phase2,
        icb2_arg_encoder,
    })
}

// The GPU-driven shadow cull pipeline + the argument encoder that wires its
// shadow ICB into the `cull_encode_shadow` kernel.
pub(super) type ShadowCullPipeline = (
    Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    Retained<ProtocolObject<dyn MTLArgumentEncoder>>,
);

// Build the GPU-driven cascaded-shadow cull pipeline: the
// `cull_encode_shadow` kernel + the argument encoder that wires its shadow ICB
// into the kernel. Compiled from the same `cull.metal` source as the main cull
// (a separate compile keeps the call site shadow-gated rather than always
// building it on the bindless path). The render-side depth-only shadow pipeline
// is built separately in `init/pipelines.rs::build_shadow_bindless_pipeline`.
pub(super) fn build_shadow_cull_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<ShadowCullPipeline, String> {
    let msl = shader_source(hot_reload, "cull.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("shadow cull shader compile error: {:?}", e))?;
    let shadow_fn = library
        .newFunctionWithName(&ns_str("cull_encode_shadow"))
        .ok_or("cull_encode_shadow not found in cull library")?;
    let state = device
        .newComputePipelineStateWithFunction_error(&shadow_fn)
        .map_err(|e| format!("failed to create shadow cull pipeline state: {:?}", e))?;
    // SAFETY: same static buffer index as the main cull kernels:
    // `cull_encode_shadow` declares its ICBContainer argument buffer at
    // CULL_ICB_BUFFER_INDEX.
    let icb_arg_encoder =
        unsafe { shadow_fn.newArgumentEncoderWithBufferIndex(CULL_ICB_BUFFER_INDEX) };
    Ok((state, icb_arg_encoder))
}

#[cfg(test)]
mod tests {
    use super::metal_flat_pool_indices;
    use crate::gfx::render_types::MaterialUniforms;

    #[test]
    fn flat_pool_indices_bias_albedo_and_normal_regions() {
        // Pool: 4 albedo + 3 normal maps, laid out [a0 a1 a2 a3][n0 n1 n2].
        let material = MaterialUniforms {
            albedo_secondary_index: 1,
            normal_secondary_index: 2,
            emissive_map_index: 3,
            orm_map_index: 0,
            ..MaterialUniforms::DEFAULT
        };
        let idx = metal_flat_pool_indices(4, 3, 2, 1, &material);
        // Albedo-region slots index the pool directly.
        assert_eq!(idx.albedo, 2);
        assert_eq!(idx.albedo_secondary, 1);
        assert_eq!(idx.emissive, 3);
        assert_eq!(idx.orm, 0);
        // Normal-region slots are biased by the albedo count: the regression the
        // shared core helper missed for folded instances was the SECONDARY normal.
        assert_eq!(idx.normal, 4 + 1);
        assert_eq!(idx.normal_secondary, 4 + 2);
    }

    #[test]
    fn flat_pool_indices_clamp_out_of_range_slots() {
        // A stale/oversized slot clamps to the last valid entry of its region
        // rather than reading past the pool.
        let material = MaterialUniforms {
            albedo_secondary_index: 50,
            normal_secondary_index: 99,
            ..MaterialUniforms::DEFAULT
        };
        let idx = metal_flat_pool_indices(2, 2, 99, 99, &material);
        assert_eq!(idx.albedo, 1); // clamped to last albedo
        assert_eq!(idx.albedo_secondary, 1);
        assert_eq!(idx.normal, 2 + 1); // bias + clamp to last normal
        assert_eq!(idx.normal_secondary, 2 + 1);
    }

    #[test]
    fn flat_pool_indices_clamp_to_bindless_cap() {
        // A pathological over-capacity pool (albedo_count + normal_count beyond
        // BINDLESS_TEXTURE_COUNT) must still cap every index at the last pool
        // slot so the fixed-size MSL tex_pool array is never indexed past its end.
        let cap = super::super::context::BINDLESS_TEXTURE_COUNT;
        let idx = metal_flat_pool_indices(cap, cap, 5, 10, &MaterialUniforms::DEFAULT);
        assert_eq!(idx.albedo, 5); // below the cap, untouched
        assert_eq!(idx.normal, (cap - 1) as u32); // cap + 10 -> capped
    }
}
