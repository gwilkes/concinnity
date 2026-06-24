// src/metal/instanced.rs
//
// Shared GPU-instanced-cluster preparation and draw. The main, SSR pre-pass,
// SSAO pre-pass, and velocity pre-pass each used to run a verbatim-identical
// block: cluster frustum/distance cull → `lod_buckets` (which cloned every
// instance matrix) → a fresh `newBufferWithBytes` per bucket → bind at
// vertex(6) → `drawIndexedInstanced`. That was the same cull + LOD partition +
// instance upload done up to four times per frame for identical data, and the
// copy-paste had already drifted between passes.
//
// `prepare_instanced_draws` does the cull + LOD bucketing + instance upload
// ONCE per frame (on the main thread, before the pass fan-out), uploading into
// the per-frame `InstanceRing` instead of fresh allocations. It iterates buckets
// via `try_for_each_lod_bucket`, so a cluster with no LOD alternates (the common
// case) memcpy's its instance matrices straight into the ring with no clone. The
// four passes
// then call `draw_prepared_instances` with a small per-cluster closure for the
// only thing that actually differs between them (material/textures, roughness,
// or nothing), so the cull/upload/draw boilerplate lives in exactly one place.
//
// The same drift problem affected the static and skinned draw loops: the
// `visible`/`resident` filter, the camera-distance LOD pick, and the
// `drawIndexedPrimitives ... baseVertex` call were copy-pasted across the main
// pass and every pre-pass. `draw_static_objects` / `draw_skinned_objects` own
// that boilerplate; each pass passes a per-draw closure for the only thing that
// varies (the model / material / texture / joint bindings).

#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLIndexType, MTLPrimitiveType, MTLRenderCommandEncoder as _};

use crate::gfx::frustum::Frustum;
use crate::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};

use super::context::{MtlContext, bytes_of_slice};

// One LOD bucket of a cluster, prepared for this frame: a ready-to-bind
// instance-matrix buffer (from the per-frame ring) plus its index range and
// instance count.
pub(super) struct PreparedBucket {
    index_offset: usize,
    index_count: usize,
    instance_count: usize,
    instances: Retained<ProtocolObject<dyn MTLBuffer>>,
}

// One visible cluster prepared for this frame.
pub(super) struct PreparedCluster {
    // Index into [`MtlContext::instanced_clusters`], so a pass can read the
    // cluster's material / texture slots in its per-cluster closure.
    cluster_index: usize,
    buckets: Vec<PreparedBucket>,
}

// Every visible instanced cluster prepared once per frame and shared by the
// main / SSR / SSAO / velocity passes. Empty when the scene has no clusters or
// none survive culling.
pub(super) struct PreparedInstances {
    pub(super) clusters: Vec<PreparedCluster>,
}

impl MtlContext {
    // Cull every instanced cluster against the frustum + distance, partition
    // each survivor into LOD buckets, and upload each bucket's instance
    // matrices into this frame's `InstanceRing` slot. Returns the prepared set
    // the geometry passes share. Runs once per frame on the main thread before
    // the pass fan-out; the workers only read the result.
    pub(in crate::metal) fn prepare_instanced_draws(
        &mut self,
        ring_slot: usize,
        cam_pos: [f32; 3],
        frustum: &Frustum,
    ) -> Result<PreparedInstances, String> {
        if self.instanced_clusters.is_empty() {
            return Ok(PreparedInstances {
                clusters: Vec::new(),
            });
        }
        self.instance_ring.begin_frame(ring_slot);
        let mut clusters = Vec::new();
        // Bind the ring + device to locals up front: the per-bucket closure
        // below writes the ring while borrowing the cluster's instances, and
        // these are disjoint fields from `instanced_clusters`, so binding them
        // separately lets the closure capture them without colliding with the
        // cluster borrow.
        let instance_ring = &mut self.instance_ring;
        let device = &self.device;
        for ci in 0..self.instanced_clusters.len() {
            let cluster = &self.instanced_clusters[ci];
            if cluster.instances.is_empty() {
                continue;
            }
            if cluster.cullable()
                && !frustum.intersects_aabb(cluster.cluster_bb_min, cluster.cluster_bb_max)
            {
                continue;
            }
            if cluster.cull_distance > 0.0 && cluster.cullable() {
                let d2 = crate::gfx::frustum::aabb_distance_sq(
                    cam_pos,
                    cluster.cluster_bb_min,
                    cluster.cluster_bb_max,
                );
                if d2 > cluster.cull_distance * cluster.cull_distance {
                    continue;
                }
            }
            // The common no-alternates case hands the closure a borrow of the
            // cluster's own instance slice, memcpy'd straight into the ring with
            // no intermediate clone; clusters with LOD alternates regroup per
            // bucket (the one copy separate per-LOD draws require).
            let mut buckets = Vec::new();
            cluster.try_for_each_lod_bucket::<String>(
                cam_pos,
                |index_offset, index_count, instances| {
                    let buf = instance_ring.write(device, ring_slot, bytes_of_slice(instances))?;
                    buckets.push(PreparedBucket {
                        index_offset,
                        index_count,
                        instance_count: instances.len(),
                        instances: buf,
                    });
                    Ok(())
                },
            )?;
            clusters.push(PreparedCluster {
                cluster_index: ci,
                buckets,
            });
        }
        Ok(PreparedInstances { clusters })
    }

    // Issue the prepared instanced draws on `enc`. For each cluster the caller
    // supplies `per_cluster` to set the only per-cluster state that varies by
    // pass (material + textures for the main pass, roughness for SSR, nothing
    // for SSAO / velocity); the shared code binds each bucket's instance buffer
    // at vertex(6), and at vertex(7) too when `bind_prev` is set (the velocity
    // pass reuses the static instance set as the previous-frame transforms),
    // then issues one `drawIndexedInstanced` per bucket. Returns the draw count.
    pub(in crate::metal) fn draw_prepared_instances<F>(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        prepared: &PreparedInstances,
        bind_prev: bool,
        mut per_cluster: F,
    ) -> u32
    where
        F: FnMut(&ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>, &InstancedCluster),
    {
        let mut draws = 0u32;
        for pc in &prepared.clusters {
            let cluster = &self.instanced_clusters[pc.cluster_index];
            per_cluster(enc, cluster);
            for b in &pc.buckets {
                let index_byte_offset = b.index_offset * std::mem::size_of::<u32>();
                unsafe {
                    enc.setVertexBuffer_offset_atIndex(Some(&b.instances), 0, 6);
                    if bind_prev {
                        enc.setVertexBuffer_offset_atIndex(Some(&b.instances), 0, 7);
                    }
                    enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset_instanceCount(
                        MTLPrimitiveType::Triangle,
                        b.index_count,
                        MTLIndexType::UInt32,
                        &self.index_buffer,
                        index_byte_offset,
                        b.instance_count,
                    );
                }
                draws += 1;
            }
        }
        draws
    }

    // Draw the visible static draw objects, one indexed draw each. Owns the
    // `visible` iteration, the `obj.visible && obj.resident` filter, the
    // camera-distance LOD pick (`active_lod`), and the `baseVertex` indexed draw
    // into the shared u32 index buffer. `per_draw` receives the object and its
    // index into `draw_objects` (so a pass can look up parallel arrays like
    // `prev_draw_models`) and sets the only thing that varies by pass (the
    // model / material / texture bindings) before the draw is issued. The
    // caller binds the pipeline + shared vertex buffer + per-frame view uniforms
    // first. Returns the draw count.
    pub(in crate::metal) fn draw_static_objects<F>(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        visible: &[u32],
        cam_pos: [f32; 3],
        mut per_draw: F,
    ) -> u32
    where
        F: FnMut(&ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>, &DrawObject, usize),
    {
        let mut draws = 0u32;
        // See-through glass meshes (Layer 2) draw in the transparent pass when the
        // RT path is live, so skip them here. A no-op on non-RT worlds and worlds
        // with no see-through material. Bistro (bindless) renders through the ICB
        // path, not this loop, so this only covers the legacy main pass + the
        // CPU-driven capture path.
        let skip_seethrough = self.mesh_glass_active();
        for &draw_idx in visible {
            let obj = &self.draw_objects[draw_idx as usize];
            if !obj.visible || !obj.resident || (skip_seethrough && obj.material.see_through != 0) {
                continue;
            }
            per_draw(enc, obj, draw_idx as usize);
            // LOD by camera distance, matching the bindless path's GpuDrawArgs
            // pick so every pass rasterizes the same slice for an object.
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
            draws += 1;
        }
        draws
    }

    // Draw the visible skinned meshes, one indexed (u16) draw each. Owns the
    // `skinned_draw_objects` iteration, the `obj.visible` filter, the
    // skinned-camera-distance LOD pick, and the indexed draw into `sib` (the
    // shared skinned index buffer). `per_draw` receives the object's index `i`
    // (for `skinned_joint_bufs[i]`) and sets the per-object model / material /
    // joint bindings. The caller binds the skinned pipeline + skinned vertex
    // buffer first. (Skinned objects carry no `resident` flag, unlike static
    // `DrawObject`s, so only `visible` gates them.) Returns the draw count.
    pub(in crate::metal) fn draw_skinned_objects<F>(
        &self,
        enc: &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
        sib: &ProtocolObject<dyn MTLBuffer>,
        cam_pos: [f32; 3],
        mut per_draw: F,
    ) -> u32
    where
        F: FnMut(
            &ProtocolObject<dyn objc2_metal::MTLRenderCommandEncoder>,
            &SkinnedDrawObject,
            usize,
        ),
    {
        let mut draws = 0u32;
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            per_draw(enc, obj, i);
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let index_byte_offset = index_offset * std::mem::size_of::<u16>();
            unsafe {
                enc.drawIndexedPrimitives_indexCount_indexType_indexBuffer_indexBufferOffset(
                    MTLPrimitiveType::Triangle,
                    index_count,
                    MTLIndexType::UInt16,
                    sib,
                    index_byte_offset,
                );
            }
            draws += 1;
        }
        draws
    }
}
