// Build-time mesh decimator and runtime-side helpers for the multi-LOD
// pipeline. The decimator that lives here:
//
//   * `decimate_by_qem`: half-edge collapse driven by the Garland-Heckbert
//     quadric error metric. Each candidate edge is scored by the squared
//     distance the surviving endpoint would have to the planes of every
//     triangle that touched either endpoint; the cheapest edge is collapsed
//     first. Only the two endpoints are candidates for the survivor (no
//     optimal-point solve), so the LOD0 vertex set is preserved unchanged
//     and the runtime payload format only needs an extra *index* list per
//     level. Used by `build_lod_alternates` to bake LOD1..N.
//
// It writes into the original LOD0 vertex set, so the runtime's shared
// vertex buffer stays untouched and a LOD swap is a pure
// `(index_offset, index_count)` change.

// Distance from `cam_pos` to a skinned object's authored placement (the
// column-3 translation of its model matrix). Skinned objects deform every
// frame, so they have no static AABB: this is the cheap stand-in the
// per-frame LOD picks use.
#[allow(dead_code)] // Consumed by every backend's skinned draw path (client crate).
pub fn skinned_camera_distance(
    obj: &crate::gfx::render_types::SkinnedDrawObject,
    cam_pos: [f32; 3],
) -> f32 {
    let centre = obj.translation();
    let dx = centre[0] - cam_pos[0];
    let dy = centre[1] - cam_pos[1];
    let dz = centre[2] - cam_pos[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Distance from `cam_pos` to the centre of `obj`'s world AABB, used to pick
// the active LOD slice each frame. Dynamic props (sentinel non-finite AABB)
// fall back to the model-matrix translation so they still LOD by their
// authored placement.
pub fn camera_distance(obj: &crate::gfx::render_types::DrawObject, cam_pos: [f32; 3]) -> f32 {
    let centre = if obj.cullable() {
        [
            0.5 * (obj.bb_min[0] + obj.bb_max[0]),
            0.5 * (obj.bb_min[1] + obj.bb_max[1]),
            0.5 * (obj.bb_min[2] + obj.bb_max[2]),
        ]
    } else {
        [obj.model[3][0], obj.model[3][1], obj.model[3][2]]
    };
    let dx = centre[0] - cam_pos[0];
    let dy = centre[1] - cam_pos[1];
    let dz = centre[2] - cam_pos[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// Decimate `indices` to at most `target_tri_count` triangles using
// half-edge collapse driven by the Garland-Heckbert quadric error metric.
//
// Per-vertex quadrics are the sum of the plane-equation outer products of
// every triangle the vertex touches; the cost of collapsing edge `(a, b)`
// is `min(p_a^T (Q_a + Q_b) p_a, p_b^T (Q_a + Q_b) p_b)` and the survivor
// is the cheaper endpoint. Restricting the survivor to one of the two
// existing endpoints (the "half-edge" variant of the algorithm) leaves
// the vertex set unchanged, so the runtime payload format only carries
// new index lists per LOD; see the module docstring.
//
// Costs are evaluated once up front and consumed in order from a min-heap.
// When an edge is popped we check whether either endpoint has already
// been merged into another vertex; if so the edge is skipped. This is
// the "lazy" variant: quality is good enough for distance-keyed LOD
// swaps without the expense of recomputing every neighbour's cost after
// each collapse.
//
// Returns a new index list addressing the same `positions` slice; empty
// when the input is empty or every triangle degenerated.
pub fn decimate_by_qem(
    positions: &[[f32; 3]],
    indices: &[u16],
    target_tri_count: usize,
) -> Vec<u16> {
    let n_verts = positions.len();
    let n_tris = indices.len() / 3;
    if n_verts == 0 || n_tris == 0 || target_tri_count == 0 {
        return Vec::new();
    }

    // 1. Per-triangle plane equations and per-vertex quadrics.
    //    Q stored as the 10 unique entries of a 4x4 symmetric matrix:
    //    (a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2).
    let mut quadrics: Vec<Quadric> = vec![Quadric::ZERO; n_verts];
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(n_tris);
    for t in 0..n_tris {
        let i0 = indices[t * 3] as usize;
        let i1 = indices[t * 3 + 1] as usize;
        let i2 = indices[t * 3 + 2] as usize;
        if i0 >= n_verts || i1 >= n_verts || i2 >= n_verts {
            continue;
        }
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let n = face_normal_unnormalised(p0, p1, p2);
        let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if mag < 1e-12 {
            continue;
        }
        let inv = 1.0 / mag;
        let a = n[0] * inv;
        let b = n[1] * inv;
        let c = n[2] * inv;
        let d = -(a * p0[0] + b * p0[1] + c * p0[2]);
        let q = Quadric::from_plane(a, b, c, d);
        quadrics[i0] = quadrics[i0].add(q);
        quadrics[i1] = quadrics[i1].add(q);
        quadrics[i2] = quadrics[i2].add(q);
        tris.push([i0 as u32, i1 as u32, i2 as u32]);
    }
    if tris.is_empty() {
        return Vec::new();
    }

    // Fast path: target is at or above the post-filter triangle count. No
    // collapses needed, but the final walk still rewrites indices through
    // the (identity) remap so any degenerates we filtered above stay out.
    if target_tri_count >= tris.len() {
        let mut out: Vec<u16> = Vec::with_capacity(tris.len() * 3);
        for tri in &tris {
            out.push(tri[0] as u16);
            out.push(tri[1] as u16);
            out.push(tri[2] as u16);
        }
        return out;
    }

    // 2. Unique undirected edges + initial half-edge collapse cost.
    let mut edges: std::collections::HashSet<(u32, u32)> =
        std::collections::HashSet::with_capacity(tris.len() * 3);
    for tri in &tris {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            edges.insert((lo, hi));
        }
    }

    let mut heap: std::collections::BinaryHeap<HeapEdge> =
        std::collections::BinaryHeap::with_capacity(edges.len());
    for (lo, hi) in edges {
        let q = quadrics[lo as usize].add(quadrics[hi as usize]);
        let cost_lo = q.eval(positions[lo as usize]);
        let cost_hi = q.eval(positions[hi as usize]);
        let (survivor, deleted, cost) = if cost_lo <= cost_hi {
            (lo, hi, cost_lo)
        } else {
            (hi, lo, cost_hi)
        };
        heap.push(HeapEdge {
            cost,
            survivor,
            deleted,
        });
    }

    // 3. Pop in ascending cost; collapse `deleted → survivor` if both
    //    are still alive. Track collapses through a union-find `remap`.
    let mut remap: Vec<u32> = (0..n_verts as u32).collect();
    let mut tri_count = tris.len();
    while tri_count > target_tri_count {
        let Some(edge) = heap.pop() else { break };
        let survivor = resolve_remap(&mut remap, edge.survivor);
        let deleted = resolve_remap(&mut remap, edge.deleted);
        if survivor == deleted {
            continue;
        }
        remap[deleted as usize] = survivor;
        let merged = quadrics[deleted as usize];
        quadrics[survivor as usize] = quadrics[survivor as usize].add(merged);
        // Each manifold edge is shared by 1-2 triangles; assume 2 for
        // budget tracking. The final remap walk drops the actual
        // degenerates.
        tri_count = tri_count.saturating_sub(2);
    }

    // 4. Rewrite the original triangle list through `remap`, dropping
    //    triangles whose corners collapsed onto each other.
    let mut out: Vec<u16> = Vec::with_capacity(indices.len());
    let max_idx = u16::MAX as u32;
    for tri in &tris {
        let a = resolve_remap(&mut remap, tri[0]);
        let b = resolve_remap(&mut remap, tri[1]);
        let c = resolve_remap(&mut remap, tri[2]);
        if a == b || b == c || a == c {
            continue;
        }
        if a > max_idx || b > max_idx || c > max_idx {
            continue;
        }
        out.push(a as u16);
        out.push(b as u16);
        out.push(c as u16);
    }
    out
}

// Per-vertex Garland-Heckbert quadric stored as the 10 unique entries of
// a 4×4 symmetric matrix. Compact, branch-free add.
#[derive(Copy, Clone, Debug)]
struct Quadric {
    m: [f32; 10],
}

impl Quadric {
    const ZERO: Self = Self { m: [0.0; 10] };

    fn from_plane(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            m: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }

    fn add(self, other: Self) -> Self {
        let mut out = [0.0f32; 10];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.m[i] + other.m[i];
        }
        Self { m: out }
    }

    fn eval(&self, p: [f32; 3]) -> f32 {
        let [x, y, z] = p;
        let m = &self.m;
        x * x * m[0]
            + 2.0 * x * y * m[1]
            + 2.0 * x * z * m[2]
            + 2.0 * x * m[3]
            + y * y * m[4]
            + 2.0 * y * z * m[5]
            + 2.0 * y * m[6]
            + z * z * m[7]
            + 2.0 * z * m[8]
            + m[9]
    }
}

// One entry on the min-heap. `Ord` is inverted so `BinaryHeap` pops the
// lowest-cost edge first; ties broken by deterministic ids so collapse
// order is reproducible across runs.
#[derive(Copy, Clone, Debug)]
struct HeapEdge {
    cost: f32,
    survivor: u32,
    deleted: u32,
}

impl PartialEq for HeapEdge {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.survivor == other.survivor && self.deleted == other.deleted
    }
}

impl Eq for HeapEdge {}

impl Ord for HeapEdge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap. NaN sinks to the bottom (treated as max).
        let a = if self.cost.is_nan() {
            f32::INFINITY
        } else {
            self.cost
        };
        let b = if other.cost.is_nan() {
            f32::INFINITY
        } else {
            other.cost
        };
        b.partial_cmp(&a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| other.survivor.cmp(&self.survivor))
            .then_with(|| other.deleted.cmp(&self.deleted))
    }
}

impl PartialOrd for HeapEdge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// Path-compressed union-find lookup used to walk an index through the
// remap chain to its current survivor.
fn resolve_remap(remap: &mut [u32], mut v: u32) -> u32 {
    while remap[v as usize] != v {
        let parent = remap[v as usize];
        let grand = remap[parent as usize];
        remap[v as usize] = grand;
        v = grand;
    }
    v
}

fn face_normal_unnormalised(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ]
}

// Compute the LOD level's target triangle count given the LOD0 triangle
// count. Halves per level (level 1 → 50 %, level 2 → 25 %, ...) with a
// floor of 4 so the coarsest LOD still renders something. Mirrors the
// "halve triangles per LOD" rule of thumb the legacy clustering grid
// roughly approximated.
pub fn target_tri_count_for_level(lod0_tri_count: usize, level: u32) -> usize {
    if lod0_tri_count == 0 {
        return 0;
    }
    let shift = level.min(20);
    let target = lod0_tri_count >> shift;
    target.max(4)
}

// Default switch-distance for LOD level `i` (i ≥ 1) given the LOD0 mesh's
// bounding-sphere radius. Each level doubles the previous threshold so
// distant LODs swap in progressively further out. The base distance
// (LOD1) is `radius * 12`, picked so a 1-unit-radius prop swaps at 12 m,
// far enough that the cluster artefacts are not obvious in the showcase
// but close enough that the LOD pass shows visible work in the debug
// renderer.
pub fn default_distance_for_level(radius: f32, level: u32) -> f32 {
    let base = radius.max(0.25) * 12.0;
    let exp = level.saturating_sub(1);
    let scale = (1u32 << exp.min(20)) as f32;
    base * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(x: f32, y: f32, z: f32) -> [f32; 3] {
        [x, y, z]
    }

    #[test]
    fn legacy_output_indices_are_in_range_via_qem() {
        // 8 corners of a unit cube + 12 triangles.
        let verts = vec![
            pos(-1.0, -1.0, -1.0),
            pos(1.0, -1.0, -1.0),
            pos(1.0, 1.0, -1.0),
            pos(-1.0, 1.0, -1.0),
            pos(-1.0, -1.0, 1.0),
            pos(1.0, -1.0, 1.0),
            pos(1.0, 1.0, 1.0),
            pos(-1.0, 1.0, 1.0),
        ];
        let idxs: Vec<u16> = vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
            3, 3, 7, 4, 3, 4, 0,
        ];
        let out = decimate_by_qem(&verts, &idxs, 4);
        for &i in &out {
            assert!((i as usize) < verts.len());
        }
    }

    #[test]
    fn qem_returns_input_when_target_meets_or_exceeds_tri_count() {
        // 8-corner cube, 12 triangles. Target ≥ 12 means no work to do.
        let verts = vec![
            pos(-1.0, -1.0, -1.0),
            pos(1.0, -1.0, -1.0),
            pos(1.0, 1.0, -1.0),
            pos(-1.0, 1.0, -1.0),
            pos(-1.0, -1.0, 1.0),
            pos(1.0, -1.0, 1.0),
            pos(1.0, 1.0, 1.0),
            pos(-1.0, 1.0, 1.0),
        ];
        let idxs: Vec<u16> = vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
            3, 3, 7, 4, 3, 4, 0,
        ];
        let out = decimate_by_qem(&verts, &idxs, 12);
        assert_eq!(out, idxs);
        let out = decimate_by_qem(&verts, &idxs, 999);
        assert_eq!(out, idxs);
    }

    #[test]
    fn qem_reduces_triangle_count_toward_target() {
        // Subdivided plane: 100 quads = 200 triangles. QEM should collapse
        // co-planar interior vertices preferentially (their quadrics agree
        // on the plane equation, so the error of removing them is zero).
        let n = 10usize;
        let mut verts = Vec::with_capacity((n + 1) * (n + 1));
        for j in 0..=n {
            for i in 0..=n {
                verts.push(pos(i as f32, 0.0, j as f32));
            }
        }
        let stride = (n + 1) as u16;
        let mut idxs: Vec<u16> = Vec::with_capacity(n * n * 6);
        for j in 0..n {
            for i in 0..n {
                let a = (j as u16) * stride + i as u16;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                idxs.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
        let lod0_tris = idxs.len() / 3;
        let out = decimate_by_qem(&verts, &idxs, lod0_tris / 4);
        let out_tris = out.len() / 3;
        assert!(
            out_tris < lod0_tris,
            "expected fewer triangles after QEM: lod0={}, out={}",
            lod0_tris,
            out_tris
        );
        // Co-planar geometry: the algorithm should collapse aggressively
        // without losing any triangles to degeneracy beyond the target.
        assert!(
            out_tris >= 2,
            "should produce at least one quad worth: {}",
            out_tris
        );
    }

    #[test]
    fn qem_preserves_vertex_range() {
        let verts = vec![
            pos(-1.0, -1.0, -1.0),
            pos(1.0, -1.0, -1.0),
            pos(1.0, 1.0, -1.0),
            pos(-1.0, 1.0, -1.0),
            pos(-1.0, -1.0, 1.0),
            pos(1.0, -1.0, 1.0),
            pos(1.0, 1.0, 1.0),
            pos(-1.0, 1.0, 1.0),
        ];
        let idxs: Vec<u16> = vec![
            0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7,
            3, 3, 7, 4, 3, 4, 0,
        ];
        let out = decimate_by_qem(&verts, &idxs, 4);
        for &i in &out {
            assert!((i as usize) < verts.len());
        }
    }

    #[test]
    fn qem_empty_input_returns_empty() {
        assert!(decimate_by_qem(&[], &[], 4).is_empty());
        assert!(decimate_by_qem(&[pos(0.0, 0.0, 0.0)], &[], 4).is_empty());
        // target_tri_count = 0 always returns empty.
        let verts = vec![pos(0.0, 0.0, 0.0), pos(1.0, 0.0, 0.0), pos(0.0, 1.0, 0.0)];
        let idxs = vec![0u16, 1, 2];
        assert!(decimate_by_qem(&verts, &idxs, 0).is_empty());
    }

    #[test]
    fn qem_handles_degenerate_input_triangles() {
        // Two-collapsed-corners triangle: cluster decimator drops it
        // immediately, QEM filters it during the per-triangle plane fit
        // (zero face area → skipped).
        let verts = vec![pos(0.0, 0.0, 0.0), pos(0.0, 0.0, 0.0), pos(1.0, 0.0, 0.0)];
        let idxs = vec![0u16, 1, 2];
        let out = decimate_by_qem(&verts, &idxs, 1);
        assert!(out.is_empty());
    }

    #[test]
    fn target_tri_count_halves_per_level_with_floor() {
        assert_eq!(target_tri_count_for_level(800, 1), 400);
        assert_eq!(target_tri_count_for_level(800, 2), 200);
        assert_eq!(target_tri_count_for_level(800, 3), 100);
        // Floor at 4 so the coarsest LOD never collapses to a point.
        assert_eq!(target_tri_count_for_level(4, 7), 4);
        assert_eq!(target_tri_count_for_level(0, 1), 0);
    }

    #[test]
    fn default_distance_doubles_per_level() {
        let r = 1.0;
        let d1 = default_distance_for_level(r, 1);
        let d2 = default_distance_for_level(r, 2);
        let d3 = default_distance_for_level(r, 3);
        assert!(d1 > 0.0);
        assert!(
            (d2 - 2.0 * d1).abs() < 1e-3,
            "LOD2 should be 2× LOD1: {} vs {}",
            d1,
            d2
        );
        assert!(
            (d3 - 4.0 * d1).abs() < 1e-3,
            "LOD3 should be 4× LOD1: {} vs {}",
            d1,
            d3
        );
    }
}
