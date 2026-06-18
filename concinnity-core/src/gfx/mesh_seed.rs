// Shrinkable seed VRAM: planning + buffer compaction for streamed mesh
// geometry.
//
// Without this, `build_draw_list` bakes every streamed mesh into the shared
// vertex/index buffers, so the buffers are sized for the *whole* streamed set
// and streaming never shrinks GPU memory. The shrinkable-seed path instead
// keeps only the resident geometry baked in and reserves a smaller `seed`
// headroom for streamed meshes -- sized to the cap-many largest meshes the
// residency cap permits. The streamer places meshes into the headroom on
// upload and tolerates a transient `alloc` miss while freed regions await
// their retire frame.
//
// This is pure policy + buffer math: no backend types, no I/O, no threads. It
// lives alongside `gfx::range_alloc` for the same reason.

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawObject, InstancedCluster};

const VERTEX_STRIDE: usize = core::mem::size_of::<Vertex>();
const INDEX_STRIDE: usize = core::mem::size_of::<u32>();

// The streaming headroom reserved in the shared vertex / index buffers, in
// bytes. After init the renderer seeds the mesh sub-allocators with this one
// block instead of the individual build-time regions, so the buffers shrink
// from "every streamed mesh at once" to "the cap-many resident at once".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshSeedRegion {
    pub vtx_offset: u64,
    pub vtx_bytes: u64,
    pub idx_offset: u64,
    pub idx_bytes: u64,
}

// Decide the seed headroom (in bytes) for a streamed mesh set, or `None` when
// no shrink is possible.
//
// `mesh_byte_sizes[i] = (vertex_bytes, index_bytes)` for streamed mesh `i`,
// where `index_bytes` is the *u32* shared-buffer stride (per-mesh `u16`
// indices are widened on upload). `cap` is `StreamingConfig::mesh_cap`.
//
// The seed is sized to hold the `residency` largest meshes at once, where
// `residency = cap + margin` capped at the mesh count. The cap-many floor
// guarantees the steady-state resident set (the planner keeps at most `cap`
// resident) always fits, so a load can never *permanently* miss; the margin
// absorbs the transient where an eviction's freed region still awaits its
// retire frame while the replacement loads. Sizing each buffer by the
// largest-`residency` of *that* buffer's per-mesh bytes is a safe independent
// upper bound for each buffer.
//
// Returns `None` (the caller keeps the full set baked in, as before) when the
// cap -- plus margin -- can already hold every streamed mesh, so there is no
// VRAM to reclaim.
pub fn plan_seed_bytes(mesh_byte_sizes: &[(u64, u64)], cap: usize) -> Option<(u64, u64)> {
    let n = mesh_byte_sizes.len();
    let cap = cap.max(1);
    if cap >= n {
        // Every streamed mesh can be resident at once: the full set is already
        // the minimal seed.
        return None;
    }
    let margin = (cap / 4).max(1);
    let residency = (cap + margin).min(n);
    if residency >= n {
        // The margin already covers the whole set -- shrinking would not help.
        return None;
    }
    let mut vtx: Vec<u64> = mesh_byte_sizes.iter().map(|&(v, _)| v).collect();
    let mut idx: Vec<u64> = mesh_byte_sizes.iter().map(|&(_, i)| i).collect();
    // Largest `residency` of each, summed independently. Descending sort, then
    // take the front.
    vtx.sort_unstable_by(|a, b| b.cmp(a));
    idx.sort_unstable_by(|a, b| b.cmp(a));
    let seed_vtx: u64 = vtx.iter().take(residency).sum();
    let seed_idx: u64 = idx.iter().take(residency).sum();
    Some((seed_vtx, seed_idx))
}

// Copy one geometry region -- its vertices plus its LOD0 and alternate index
// ranges -- into the growing destination buffers, rebasing the region's
// absolute indices onto the moved vertex base. Returns the region's new
// vertex byte offset, its new LOD0 index element offset, and the new element
// offset of each alternate (in input order).
#[allow(clippy::too_many_arguments)] // distinct source/destination buffers and offsets; grouping would obscure intent
fn relocate_region(
    src_v: &[Vertex],
    src_i: &[u32],
    dst_v: &mut Vec<Vertex>,
    dst_i: &mut Vec<u32>,
    v_off_bytes: usize,
    v_count: usize,
    lod0: (usize, usize),
    alts: &[(usize, usize)],
) -> (usize, usize, Vec<usize>) {
    let old_vbase = v_off_bytes / VERTEX_STRIDE;
    let new_vbase = dst_v.len();
    dst_v.extend_from_slice(&src_v[old_vbase..old_vbase + v_count]);
    // Indices are absolute into the shared vertex buffer; moving the vertices
    // from old_vbase to new_vbase shifts every index by the same delta. Use
    // i64 so a cluster relocated *forward* (its geometry sat behind the props
    // it now trails) rebases correctly too.
    let delta = new_vbase as i64 - old_vbase as i64;
    let rebase = |i: u32| -> u32 { (i as i64 + delta) as u32 };

    let (i0_off, i0_count) = lod0;
    let new_i0_off = dst_i.len();
    for &idx in &src_i[i0_off..i0_off + i0_count] {
        dst_i.push(rebase(idx));
    }
    let mut new_alt_offsets = Vec::with_capacity(alts.len());
    for &(a_off, a_count) in alts {
        let new_a_off = dst_i.len();
        for &idx in &src_i[a_off..a_off + a_count] {
            dst_i.push(rebase(idx));
        }
        new_alt_offsets.push(new_a_off);
    }
    (new_vbase * VERTEX_STRIDE, new_i0_off, new_alt_offsets)
}

// Rewrite the shared vertex / index buffers so only resident geometry is
// baked in, then append a zeroed seed headroom for streamed meshes.
//
// Every resident `DrawObject` / `InstancedCluster` offset (and LOD-alternate
// offset) is rewritten to its new place, rebasing its absolute indices onto
// the moved vertex region. Each streamed draw (`streamed[i] == true`) is
// marked non-resident with placeholder offsets and its geometry is *not*
// copied -- it lives in the streamer's payload source and is uploaded on
// demand into the headroom.
//
// Run before backend init so the GPU buffers are created at the compacted
// size and the RT acceleration structure (built over resident draws) sees the
// final offsets. Returns the headroom region to seed into the mesh
// sub-allocators.
pub fn compact_for_streaming(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    draw_objects: &mut [DrawObject],
    clusters: &mut [InstancedCluster],
    streamed: &[bool],
    seed_vtx_bytes: u64,
    seed_idx_bytes: u64,
) -> MeshSeedRegion {
    let mut new_v: Vec<Vertex> = Vec::with_capacity(vertices.len());
    let mut new_i: Vec<u32> = Vec::with_capacity(indices.len());

    for (i, obj) in draw_objects.iter_mut().enumerate() {
        if streamed.get(i).copied().unwrap_or(false) {
            // Streamed draws are uploaded on demand; their geometry is not
            // baked in. Placeholder offsets -- `upload_mesh` assigns real ones
            // from the seeded headroom. (Alternates are already stripped from
            // streamable draws upstream; clear defensively so no stale offset
            // survives into the smaller buffer.)
            obj.vertex_offset = 0;
            obj.index_offset = 0;
            obj.resident = false;
            obj.lod_alternates.clear();
            continue;
        }
        let alts: Vec<(usize, usize)> = obj
            .lod_alternates
            .iter()
            .map(|s| (s.index_offset, s.index_count))
            .collect();
        let (new_v_off, new_i_off, new_alt_offs) = relocate_region(
            vertices,
            indices,
            &mut new_v,
            &mut new_i,
            obj.vertex_offset,
            obj.vertex_count,
            (obj.index_offset, obj.index_count),
            &alts,
        );
        obj.vertex_offset = new_v_off;
        obj.index_offset = new_i_off;
        for (slice, new_off) in obj.lod_alternates.iter_mut().zip(new_alt_offs) {
            slice.index_offset = new_off;
        }
    }

    // Clusters never stream, but their geometry sits after the props in the
    // shared buffer, so removing streamed-prop gaps shifts every cluster.
    for c in clusters.iter_mut() {
        let alts: Vec<(usize, usize)> = c
            .lod_alternates
            .iter()
            .map(|s| (s.index_offset, s.index_count))
            .collect();
        let (new_v_off, new_i_off, new_alt_offs) = relocate_region(
            vertices,
            indices,
            &mut new_v,
            &mut new_i,
            c.vertex_offset,
            c.vertex_count,
            (c.index_offset, c.index_count),
            &alts,
        );
        c.vertex_offset = new_v_off;
        c.index_offset = new_i_off;
        for (slice, new_off) in c.lod_alternates.iter_mut().zip(new_alt_offs) {
            slice.index_offset = new_off;
        }
    }

    // Append the zeroed seed headroom. Contents are irrelevant -- a streamed
    // draw is skipped until its geometry is uploaded over this region -- only
    // the size matters, so the buffers are born at resident + headroom bytes.
    let vtx_offset = (new_v.len() * VERTEX_STRIDE) as u64;
    let idx_offset = (new_i.len() * INDEX_STRIDE) as u64;
    let zero_v = Vertex {
        pos: [0.0; 3],
        normal: [0.0; 3],
        tangent: [0.0; 3],
        color: [0.0; 3],
        uv: [0.0; 2],
    };
    let seed_v_count = (seed_vtx_bytes as usize) / VERTEX_STRIDE;
    let seed_i_count = (seed_idx_bytes as usize) / INDEX_STRIDE;
    new_v.extend(std::iter::repeat_n(zero_v, seed_v_count));
    new_i.extend(std::iter::repeat_n(0u32, seed_i_count));

    *vertices = new_v;
    *indices = new_i;
    MeshSeedRegion {
        vtx_offset,
        vtx_bytes: seed_vtx_bytes,
        idx_offset,
        idx_bytes: seed_idx_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_types::{LodSlice, MaterialUniforms};

    fn vtx(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    // Append `n` vertices (tagged by `tag` so a relocated region is
    // identifiable) at the current end and return (vertex_byte_offset, base).
    fn push_verts(v: &mut Vec<Vertex>, n: usize, tag: f32) -> (usize, u32) {
        let base = v.len() as u32;
        for k in 0..n {
            v.push(vtx(tag + k as f32 / 100.0));
        }
        (base as usize * VERTEX_STRIDE, base)
    }

    // Append a triangle-fan-ish index list referencing the region's own
    // vertices (absolute, base + 0..n) and return (index_offset, count).
    fn push_idx(i: &mut Vec<u32>, base: u32, n: usize) -> (usize, usize) {
        let off = i.len();
        for k in 0..n {
            i.push(base + (k as u32 % 3));
        }
        (off, n)
    }

    fn draw(
        v_off: usize,
        v_count: usize,
        i_off: usize,
        i_count: usize,
        resident: bool,
        lods: Vec<LodSlice>,
    ) -> DrawObject {
        DrawObject {
            vertex_offset: v_off,
            vertex_count: v_count,
            index_offset: i_off,
            index_count: i_count,
            base_vertex: 0,
            model: [[0.0; 4]; 4],
            texture_slot: 0,
            normal_map_slot: 0,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident,
            bb_min: [0.0; 3],
            bb_max: [0.0; 3],
            cull_distance: 0.0,
            lod_alternates: lods,
        }
    }

    fn cluster(v_off: usize, v_count: usize, i_off: usize, i_count: usize) -> InstancedCluster {
        InstancedCluster {
            vertex_offset: v_off,
            vertex_count: v_count,
            index_offset: i_off,
            index_count: i_count,
            texture_slot: 0,
            normal_map_slot: 0,
            material: MaterialUniforms::DEFAULT,
            cluster_bb_min: [0.0; 3],
            cluster_bb_max: [0.0; 3],
            local_bb_min: [0.0; 3],
            local_bb_max: [0.0; 3],
            cull_distance: 0.0,
            instances: vec![[[0.0; 4]; 4]],
            lod_alternates: Vec::new(),
        }
    }

    #[test]
    fn plan_returns_none_when_cap_holds_every_mesh() {
        // 3 meshes, cap 4 -> all can be resident, nothing to shrink.
        let sizes = vec![(100, 40), (100, 40), (100, 40)];
        assert_eq!(plan_seed_bytes(&sizes, 4), None);
        // cap == count is also "no shrink".
        assert_eq!(plan_seed_bytes(&sizes, 3), None);
    }

    #[test]
    fn plan_returns_none_when_margin_covers_the_set() {
        // 5 meshes, cap 4 -> margin = max(1,1) = 1, residency = 5 = count.
        let sizes = vec![(100, 40); 5];
        assert_eq!(plan_seed_bytes(&sizes, 4), None);
    }

    #[test]
    fn plan_sizes_seed_to_the_largest_residency_meshes() {
        // 8 uniform meshes, cap 4 -> margin = 1, residency = 5.
        let sizes = vec![(100u64, 40u64); 8];
        let (sv, si) = plan_seed_bytes(&sizes, 4).expect("shrink");
        assert_eq!(sv, 100 * 5);
        assert_eq!(si, 40 * 5);
        // The seed is strictly smaller than the full set (8 meshes).
        assert!(sv < 100 * 8);
        assert!(si < 40 * 8);
    }

    #[test]
    fn plan_picks_the_biggest_meshes_independently_per_buffer() {
        // Skewed sizes: the largest-vtx meshes differ from the largest-idx ones.
        // cap 1 -> margin = 1, residency = 2.
        let sizes = vec![
            (1000, 4), // big vtx, small idx
            (4, 1000), // small vtx, big idx
            (10, 10),  // small both
            (10, 10),
        ];
        let (sv, si) = plan_seed_bytes(&sizes, 1).expect("shrink");
        // top-2 vtx: 1000 + 10; top-2 idx: 1000 + 10
        assert_eq!(sv, 1010);
        assert_eq!(si, 1010);
    }

    #[test]
    fn compact_removes_streamed_gaps_and_rewrites_resident_offsets() {
        // Layout: [resident A][streamed B][resident C], then [cluster D].
        let mut v: Vec<Vertex> = Vec::new();
        let mut i: Vec<u32> = Vec::new();
        let (a_voff, a_base) = push_verts(&mut v, 4, 1.0);
        let (a_ioff, a_ic) = push_idx(&mut i, a_base, 6);
        let (b_voff, b_base) = push_verts(&mut v, 10, 2.0);
        let (b_ioff, b_ic) = push_idx(&mut i, b_base, 12);
        let (c_voff, c_base) = push_verts(&mut v, 5, 3.0);
        let (c_ioff, c_ic) = push_idx(&mut i, c_base, 9);
        let (d_voff, d_base) = push_verts(&mut v, 6, 4.0);
        let (d_ioff, d_ic) = push_idx(&mut i, d_base, 6);

        let mut draws = vec![
            draw(a_voff, 4, a_ioff, a_ic, true, vec![]),
            draw(b_voff, 10, b_ioff, b_ic, true, vec![]),
            draw(c_voff, 5, c_ioff, c_ic, true, vec![]),
        ];
        // Mark only B as streamed; A and C are resident.
        let streamed = [false, true, false];
        let mut clusters = vec![cluster(d_voff, 6, d_ioff, d_ic)];

        // Snapshot the geometry we expect to survive (A, C, D). `Vertex` has no
        // `PartialEq`, so compare by the position tag `push_verts` stamped.
        let pos = |s: &[Vertex]| -> Vec<[f32; 3]> { s.iter().map(|x| x.pos).collect() };
        let a_verts = pos(&v[a_base as usize..a_base as usize + 4]);
        let c_verts = pos(&v[c_base as usize..c_base as usize + 5]);
        let d_verts = pos(&v[d_base as usize..d_base as usize + 6]);

        let region = compact_for_streaming(
            &mut v,
            &mut i,
            &mut draws,
            &mut clusters,
            &streamed,
            /*seed_vtx*/ 0,
            /*seed_idx*/ 0,
        );

        // Streamed B is non-resident with placeholder offsets.
        assert!(!draws[1].resident);
        assert_eq!(draws[1].vertex_offset, 0);
        assert_eq!(draws[1].index_offset, 0);

        // Resident A keeps its (now front-of-buffer) offsets; geometry intact.
        assert!(draws[0].resident);
        assert_eq!(draws[0].vertex_offset, 0);
        let a0 = draws[0].vertex_offset / VERTEX_STRIDE;
        assert_eq!(pos(&v[a0..a0 + 4]), a_verts);

        // Resident C moved up to fill B's gap; its indices still address its
        // own (relocated) vertices.
        let c0 = draws[2].vertex_offset / VERTEX_STRIDE;
        assert_eq!(pos(&v[c0..c0 + 5]), c_verts);
        for k in 0..c_ic {
            let idx = i[draws[2].index_offset + k] as usize;
            assert!(
                idx >= c0 && idx < c0 + 5,
                "C index {} out of its region",
                idx
            );
        }

        // Cluster D relocated; indices address its own vertices.
        let d0 = clusters[0].vertex_offset / VERTEX_STRIDE;
        assert_eq!(pos(&v[d0..d0 + 6]), d_verts);
        for k in 0..d_ic {
            let idx = i[clusters[0].index_offset + k] as usize;
            assert!(
                idx >= d0 && idx < d0 + 6,
                "D index {} out of its region",
                idx
            );
        }

        // The buffer shrank by exactly B's vertex + index region.
        assert_eq!(v.len(), 4 + 5 + 6); // A + C + D, no headroom this test
        assert_eq!(i.len(), a_ic + c_ic + d_ic);
        // Zero headroom requested -> region sits at the compacted tail.
        assert_eq!(region.vtx_offset, (v.len() * VERTEX_STRIDE) as u64);
        assert_eq!(region.vtx_bytes, 0);
    }

    #[test]
    fn compact_appends_seed_headroom_after_resident_geometry() {
        let mut v: Vec<Vertex> = Vec::new();
        let mut i: Vec<u32> = Vec::new();
        let (a_voff, a_base) = push_verts(&mut v, 3, 1.0);
        let (a_ioff, a_ic) = push_idx(&mut i, a_base, 3);
        let mut draws = vec![draw(a_voff, 3, a_ioff, a_ic, true, vec![])];
        let mut clusters: Vec<InstancedCluster> = Vec::new();
        let streamed = [false];

        let seed_v_bytes = (10 * VERTEX_STRIDE) as u64;
        let seed_i_bytes = (24 * INDEX_STRIDE) as u64;
        let region = compact_for_streaming(
            &mut v,
            &mut i,
            &mut draws,
            &mut clusters,
            &streamed,
            seed_v_bytes,
            seed_i_bytes,
        );

        // Headroom begins right after the 3 resident vertices / indices.
        assert_eq!(region.vtx_offset, (3 * VERTEX_STRIDE) as u64);
        assert_eq!(region.idx_offset, (3 * INDEX_STRIDE) as u64);
        assert_eq!(region.vtx_bytes, seed_v_bytes);
        assert_eq!(region.idx_bytes, seed_i_bytes);
        // Buffers are sized for resident + headroom.
        assert_eq!(v.len(), 3 + 10);
        assert_eq!(i.len(), 3 + 24);
        // Headroom start is vertex-aligned (offset is a whole multiple of stride).
        assert_eq!(region.vtx_offset as usize % VERTEX_STRIDE, 0);
    }

    #[test]
    fn compact_relocates_lod_alternate_index_ranges() {
        // [streamed A][resident B(with one LOD alt)].
        let mut v: Vec<Vertex> = Vec::new();
        let mut i: Vec<u32> = Vec::new();
        let (a_voff, a_base) = push_verts(&mut v, 8, 1.0);
        let (a_ioff, a_ic) = push_idx(&mut i, a_base, 12);
        let (b_voff, b_base) = push_verts(&mut v, 4, 2.0);
        let (b_ioff, b_ic) = push_idx(&mut i, b_base, 6); // LOD0
        let (b_alt_off, b_alt_ic) = push_idx(&mut i, b_base, 3); // LOD1 alt

        let mut draws = vec![
            draw(a_voff, 8, a_ioff, a_ic, true, vec![]),
            draw(
                b_voff,
                4,
                b_ioff,
                b_ic,
                true,
                vec![LodSlice {
                    index_offset: b_alt_off,
                    index_count: b_alt_ic,
                    switch_distance: 10.0,
                }],
            ),
        ];
        let mut clusters: Vec<InstancedCluster> = Vec::new();
        let streamed = [true, false];

        compact_for_streaming(&mut v, &mut i, &mut draws, &mut clusters, &streamed, 0, 0);

        let b0 = draws[1].vertex_offset / VERTEX_STRIDE;
        // LOD0 and the alternate both address B's relocated vertices.
        for k in 0..b_ic {
            let idx = i[draws[1].index_offset + k] as usize;
            assert!(idx >= b0 && idx < b0 + 4);
        }
        let alt = draws[1].lod_alternates[0];
        for k in 0..alt.index_count {
            let idx = i[alt.index_offset + k] as usize;
            assert!(idx >= b0 && idx < b0 + 4);
        }
        // The alternate's switch distance is preserved.
        assert_eq!(alt.switch_distance, 10.0);
    }
}
