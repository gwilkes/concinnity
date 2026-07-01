// src/gfx/rt_topology.rs
//
// Backend-agnostic planner for the incremental RT acceleration-structure
// topology refresh. When the participating draw set changes at runtime (a
// cloned prop, a streamed chunk added/removed), the BLAS head must be brought
// back in line with the current set without rebuilding every BLAS: reuse every
// BLAS whose geometry slice is unchanged, build only the new ones, and retire
// the orphans. This module owns only the pure decision (which slot reuses which
// old BLAS, which are orphaned); the actual GPU allocation / build / retire is
// per-backend (directx/raytrace.rs, vulkan/raytrace.rs). Split out so the plan
// is unit-testable without a GPU.
//
// Consumed by the DirectX + Vulkan backends. The Metal backend predates this
// module and keeps its own equivalent copy (metal/raytrace.rs); a future
// cleanup could converge it here once Metal can be rebuilt alongside.

use crate::gfx::render_types::DrawObject;

// Identifies the geometry slice a draw-object BLAS traces, on the shared
// vertex/index buffers. Two draw objects with the same signature trace
// identical geometry, so a topology refresh can reuse the existing BLAS instead
// of building a new one. Sound because the shared buffer regions are stable
// once streaming is set up and a slot's bytes cannot be overwritten while its
// BLAS is live (the deferred free holds the region until the frames-in-flight
// fence retires it). `base_vertex` + `index_offset` + `index_count` are exactly
// the inputs the per-backend geometry descriptor uses; `vertex_offset` is
// carried too so a static draw (whose `base_vertex` is 0) still distinguishes
// distinct vertex regions.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GeomSig {
    base_vertex: i32,
    vertex_offset: usize,
    index_offset: usize,
    index_count: usize,
}

impl GeomSig {
    pub(crate) fn of(obj: &DrawObject) -> Self {
        Self {
            base_vertex: obj.base_vertex,
            vertex_offset: obj.vertex_offset,
            index_offset: obj.index_offset,
            index_count: obj.index_count,
        }
    }
}

// Per-new-slot decision for a topology refresh of the draw-object BLAS head.
pub(crate) struct TopologyPlan {
    // `reuse[j] == Some(k)`: new draw slot `j` reuses the old draw BLAS at index
    // `k` (its geometry is unchanged). `None`: build a fresh BLAS for slot `j`.
    pub(crate) reuse: Vec<Option<usize>>,
    // Old draw BLAS indices no longer referenced by any new slot -- retire them.
    pub(crate) retire: Vec<usize>,
}

// Decide, for the draw-object BLAS head only, which BLAS to reuse, which to
// build, and which to retire when the participating draw set changes. Matches
// old and new slots by `draw_objects` index AND geometry signature: a slot whose
// geometry moved (a chunk slot recycled for a different chunk) does not match, so
// it rebuilds. Pure so it is unit-testable without a GPU.
pub(crate) fn plan_topology_refresh(
    old_indices: &[usize],
    old_sigs: &[GeomSig],
    new_indices: &[usize],
    new_sigs: &[GeomSig],
) -> TopologyPlan {
    use std::collections::HashMap;
    // draw_objects index -> (position in the old draw BLAS head, its signature).
    // `object_indices` entries are unique (one per draw slot), so this is 1:1.
    let mut by_idx: HashMap<usize, (usize, GeomSig)> = HashMap::with_capacity(old_indices.len());
    for (k, (&idx, &sig)) in old_indices.iter().zip(old_sigs).enumerate() {
        by_idx.insert(idx, (k, sig));
    }
    let mut used = vec![false; old_indices.len()];
    let mut reuse = Vec::with_capacity(new_indices.len());
    for (&idx, &sig) in new_indices.iter().zip(new_sigs) {
        match by_idx.get(&idx) {
            Some(&(k, old_sig)) if old_sig == sig && !used[k] => {
                used[k] = true;
                reuse.push(Some(k));
            }
            _ => reuse.push(None),
        }
    }
    let retire = used
        .iter()
        .enumerate()
        .filter(|&(_, &u)| !u)
        .map(|(k, _)| k)
        .collect();
    TopologyPlan { reuse, retire }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A distinct geometry signature keyed off `tag` (used as the index offset),
    // so two slots with different tags never compare equal.
    fn sig(tag: usize) -> GeomSig {
        GeomSig {
            base_vertex: tag as i32,
            vertex_offset: tag * 100,
            index_offset: tag,
            index_count: 3,
        }
    }

    #[test]
    fn topology_plan_reuses_an_unchanged_set() {
        let old_i = [2usize, 5, 7];
        let old_s = [sig(2), sig(5), sig(7)];
        let plan = plan_topology_refresh(&old_i, &old_s, &old_i, &old_s);
        assert_eq!(plan.reuse, vec![Some(0), Some(1), Some(2)]);
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn topology_plan_builds_only_the_added_slot() {
        let old_i = [2usize, 5];
        let old_s = [sig(2), sig(5)];
        let new_i = [2usize, 5, 9];
        let new_s = [sig(2), sig(5), sig(9)];
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        // The two existing slots reuse; the new one (9) builds fresh.
        assert_eq!(plan.reuse, vec![Some(0), Some(1), None]);
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn topology_plan_retires_a_removed_slot() {
        let old_i = [2usize, 5, 7];
        let old_s = [sig(2), sig(5), sig(7)];
        let new_i = [2usize, 7];
        let new_s = [sig(2), sig(7)];
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        assert_eq!(plan.reuse, vec![Some(0), Some(2)]);
        assert_eq!(plan.retire, vec![1]); // slot 5's old BLAS is orphaned
    }

    #[test]
    fn topology_plan_rebuilds_a_recycled_slot_whose_geometry_moved() {
        // Same draw index, different geometry signature: a chunk slot recycled for
        // a different chunk. The old BLAS must NOT be reused; it is retired and a
        // fresh one is built.
        let old_i = [5usize];
        let old_s = [sig(5)];
        let new_i = [5usize];
        let new_s = [sig(8)]; // moved geometry under the same draw index
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        assert_eq!(plan.reuse, vec![None]);
        assert_eq!(plan.retire, vec![0]);
    }

    #[test]
    fn topology_plan_reuses_across_reorder_by_index() {
        // The participating set is the same but its order changed; each slot still
        // reuses its BLAS by draw index (the reuse points at the old position).
        let old_i = [2usize, 5];
        let old_s = [sig(2), sig(5)];
        let new_i = [5usize, 2];
        let new_s = [sig(5), sig(2)];
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        assert_eq!(plan.reuse, vec![Some(1), Some(0)]);
        assert!(plan.retire.is_empty());
    }
}
