// src/directx/draw_iter.rs
//
// Shared scene-traversal helpers for the geometry passes. The main pass and
// every geometry pre-pass (SSR, SSAO, velocity) walk the same visible set
// with the same frustum/distance cull and the same per-object LOD pick, then
// issue a draw. Only the per-draw GPU state (pipeline, root constants,
// descriptor bindings, draw args) differs between passes.
//
// These helpers own the common skeleton (the gate + LOD pick + cluster cull +
// bucket iteration) and hand each pass a closure for its per-draw specifics,
// so the cull/LOD logic lives in one place and every pass selects the same
// LOD slice and the same surviving instances (the pre-pass G-buffers stay
// pixel-aligned with what the main pass rasterized).

use crate::gfx::render_types::{DrawObject, InstancedCluster, SkinnedDrawObject};

use super::context::{DxContext, InstanceBucketLayout};

impl DxContext {
    // Walk the visible build-time / runtime static draw objects, gating on
    // `visible && resident` and picking the camera-distance LOD slice, then
    // invoke `emit` with the object, its draw index, and the chosen LOD's
    // `(index_offset, index_count)`. The caller binds the pipeline + shared
    // root state once before calling; `emit` sets only the per-draw state
    // (material/model constants, object SRV table) and issues the draw.
    pub(in crate::directx) fn draw_static_objects<F>(
        &self,
        visible: &[u32],
        cam_pos: [f32; 3],
        mut emit: F,
    ) where
        F: FnMut(&DrawObject, usize, usize, usize),
    {
        for &draw_idx in visible {
            let i = draw_idx as usize;
            let Some(obj) = self.draw_objects.get(i) else {
                continue;
            };
            if !obj.visible || !obj.resident {
                continue;
            }
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            emit(obj, i, index_offset, index_count);
        }
    }

    // Walk the resident streamed-chunk draw objects -- the build-time-geometry
    // tail past `n_objects` that are NOT runtime clones -- invoking `emit` with the
    // chunk's reserve index `k` (0-based, into `[chunk_record_base() + k]`) and the
    // `DrawObject`. Chunk geometry already lives in the shared VB/IB, so chunks
    // fold into the static+instance prefix indirect draw as plain records (with
    // their own `base_vertex` + flat-pool material). Runtime clones (in
    // `clone.slot_by_draw_idx`) are skipped -- they keep the legacy per-object
    // path. Bounded by the chunk reserve `n_chunk`; chunks past it (only possible
    // with an absurd streaming radius) are dropped. Returns the number of chunk
    // records emitted, so the caller can disable the unused reserve tail. Includes
    // non-resident (freed) chunk slots; `emit` reads `obj.visible` / `obj.resident`
    // for the per-record draw flags.
    pub(in crate::directx) fn for_each_chunk_record<F>(&self, mut emit: F) -> usize
    where
        F: FnMut(usize, &DrawObject),
    {
        if self.n_chunk == 0 {
            return 0;
        }
        let mut k = 0;
        for (i, obj) in self.draw_objects.iter().enumerate().skip(self.n_objects) {
            if self.clone.slot_by_draw_idx.contains_key(&i) {
                continue; // runtime clone -> legacy per-object path
            }
            if k >= self.n_chunk {
                break;
            }
            emit(k, obj);
            k += 1;
        }
        k
    }

    // Walk the skinned draw objects, gating on `visible` (skinned meshes are
    // not stream-evicted, so there is no `resident` field) and picking the
    // LOD slice by distance to the model-matrix translation, then invoke
    // `emit` with the object, its index, and the chosen LOD slice.
    pub(in crate::directx) fn draw_skinned_objects<F>(&self, cam_pos: [f32; 3], mut emit: F)
    where
        F: FnMut(&SkinnedDrawObject, usize, usize, usize),
    {
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            if !obj.visible {
                continue;
            }
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            emit(obj, i, index_offset, index_count);
        }
    }

    // Walk the instanced clusters: frustum + distance cull each cluster, read
    // the shared per-frame LOD bucket layout (filled by `build_instance_upload`
    // at the top of the frame), and for every surviving bucket invoke
    // `per_bucket` with the bucket and the cluster's instance-matrix upload
    // GPU virtual address. `per_cluster` runs once per surviving cluster
    // (before its buckets) so the caller can bind cluster-wide state
    // (material constants + the cluster's albedo/normal SRV table).
    //
    // All instanced geometry passes (main + SSR / SSAO / velocity pre-passes)
    // share this skeleton; they differ only in which root slot `per_bucket`
    // bumps for the per-bucket instance SRV and what `per_cluster` binds.
    pub(in crate::directx) fn draw_instanced_clusters<C, B>(
        &self,
        frame_idx: usize,
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
        mut per_cluster: C,
        mut per_bucket: B,
    ) where
        C: FnMut(usize, &InstancedCluster),
        B: FnMut(&InstanceBucketLayout, u64),
    {
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            if cluster.instances.is_empty() {
                continue;
            }
            // Cluster-wide frustum + distance cull.
            if cluster.cullable() {
                if !frustum.intersects_aabb(cluster.cluster_bb_min, cluster.cluster_bb_max) {
                    continue;
                }
                if cluster.cull_distance > 0.0 {
                    let d2 = crate::gfx::frustum::aabb_distance_sq(
                        cam_pos,
                        cluster.cluster_bb_min,
                        cluster.cluster_bb_max,
                    );
                    if d2 > cluster.cull_distance * cluster.cull_distance {
                        continue;
                    }
                }
            }

            // Per-cluster LOD bucket layout that `build_instance_upload` filled
            // into this frame's upload buffer. Held across the bucket loop so
            // every bucket reads a consistent partition; no draw closure
            // touches `instance_bucket_layouts`, so the read lock is safe.
            let buckets_borrow = self.instanced.bucket_layouts.read().unwrap();
            let Some(buckets) = buckets_borrow.get(cluster_idx) else {
                continue;
            };
            if buckets.is_empty() {
                continue;
            }
            let inst_buf = &self.instanced.upload_buffers[frame_idx][cluster_idx];
            let inst_gva_base = unsafe { inst_buf.GetGPUVirtualAddress() };

            per_cluster(cluster_idx, cluster);
            for bucket in buckets.iter() {
                per_bucket(bucket, inst_gva_base);
            }
        }
    }
}
