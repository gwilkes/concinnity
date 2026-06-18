// src/gfx/chunk_window.rs
//
// Sliding-window streaming policy for an infinite voxel world.
//
// `gfx::streaming::StreamPlanner` streams a *fixed* pool of items known at
// init (textures, build-time meshes). An infinite chunk world is different:
// the item set is unbounded and only a bounded *window* around the camera is
// ever resident. This module is that policy -- given the camera's chunk and a
// view radius it decides which chunks to load (nearest first, budget-limited)
// and which to evict (those that have fallen well outside the window).
//
// Two concentric bands: chunks within `near_radius` stream at full voxel
// detail; chunks beyond it but within `far_radius` stream as cheap coarse
// "impostors" (a low-poly surface mesh). As the camera moves a chunk crosses
// the near/far boundary and is *re-detailed* -- evicted and reloaded at the new
// detail. A small detail hysteresis keeps a chunk pacing across the boundary
// from thrashing. When `far_radius == near_radius` (the default) the far band
// is empty, so the window behaves exactly as the original single-detail one.
//
// Like `gfx::streaming` and `gfx::chunk_coord` this is written against
// `core` + `alloc` only -- no threads, no I/O, no `std` collections (a
// `BTreeMap`, not a `HashMap`) -- so it can move into a future `no_std`
// runtime unchanged. The `std`-side driver (background generation thread,
// GPU upload) lives in `crate::app::chunk_stream`.

// `BTreeMap` is an `alloc` collection (re-exported here through `std`); a
// `HashMap` would pull in `std`-only hashing. `Vec` comes from the prelude.
use std::collections::BTreeMap;

use crate::gfx::chunk_coord::ChunkCoord;

// Extra chunk rings a chunk may drift beyond the view radius before it is
// evicted. The gap between the load radius and the evict radius is hysteresis:
// without it a chunk straddling the boundary would load and evict on
// alternating frames as the camera jitters across a chunk edge.
const EVICT_HYSTERESIS: i32 = 2;

// Extra rings a currently-full (Near) chunk may drift past `near_radius` before
// it is downgraded to a Far impostor. Without it a camera pacing back and forth
// across the near/far boundary would re-detail a chunk every step.
const DETAIL_HYSTERESIS: i32 = 1;

// Residency state of a chunk the window is currently tracking.
//
// A chunk not in the window's map is simply unloaded -- there is no explicit
// `Unloaded` state, since the grid is infinite and tracking every never-seen
// chunk would be unbounded.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChunkState {
    // A background generation+upload has been dispatched but not completed.
    Pending,
    // The chunk's mesh is resident on the GPU.
    Resident,
}

// Which representation a chunk is streamed at.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ChunkDetail {
    // Full voxel geometry for chunks within `near_radius`.
    Near,
    // A coarse distant-impostor surface mesh for chunks beyond `near_radius` but
    // within `far_radius`.
    Far,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Slot {
    state: ChunkState,
    detail: ChunkDetail,
}

// The load / evict decisions produced by one [`ChunkWindow::plan`] call.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ChunkPlan {
    // Chunks whose background load should be dispatched this frame, nearest
    // to the camera first, each tagged with the detail to generate it at.
    // Already marked [`ChunkState::Pending`].
    pub to_load: Vec<(ChunkCoord, ChunkDetail)>,
    // Chunks removed from the GPU this frame: those that fell outside the
    // evict radius, plus those crossing the near/far boundary (which reload at
    // the new detail). Already dropped from the window's tracking map.
    pub to_evict: Vec<ChunkCoord>,
}

// Decides which chunks stream in and out of the camera-centred view window,
// and at which detail.
//
// The window owns only residency *bookkeeping* -- it never generates a chunk
// or touches a GPU resource. Each frame the driver calls [`plan`] with the
// camera's current chunk, dispatches the loads, applies the evictions, and
// reports completed loads back via [`mark_resident`].
//
// [`plan`]: ChunkWindow::plan
// [`mark_resident`]: ChunkWindow::mark_resident
pub struct ChunkWindow {
    // Tracked chunks only: a chunk absent from the map is unloaded.
    states: BTreeMap<ChunkCoord, Slot>,
    // Chebyshev radius (in chunks) of the full-detail square window.
    near_radius: i32,
    // Chebyshev radius of the outer impostor window (>= near_radius).
    far_radius: i32,
    // Chebyshev radius past which a tracked chunk is evicted.
    evict_radius: i32,
    // Max chunk loads dispatched per `plan` call.
    load_budget: usize,
}

impl ChunkWindow {
    // A window with a full-detail radius of `near_radius`, an outer impostor
    // radius of `far_radius`, and a per-frame load budget of `load_budget`
    // chunks.
    //
    // `near_radius` is floored at 0 (a lone chunk), `far_radius` at
    // `near_radius` (so it never undercuts the full-detail band; equal means
    // "no impostors"), and `load_budget` at 1 so a stray 0 cannot wedge
    // streaming permanently.
    pub fn new(near_radius: i32, far_radius: i32, load_budget: usize) -> Self {
        let near_radius = near_radius.max(0);
        let far_radius = far_radius.max(near_radius);
        Self {
            states: BTreeMap::new(),
            near_radius,
            far_radius,
            evict_radius: far_radius + EVICT_HYSTERESIS,
            load_budget: load_budget.max(1),
        }
    }

    // The detail a chunk should currently be at, given its distance from the
    // camera and (for hysteresis) the detail it is currently tracked at.
    fn target_detail(
        &self,
        c: ChunkCoord,
        camera: ChunkCoord,
        current: Option<ChunkDetail>,
    ) -> ChunkDetail {
        let d = c.chebyshev_distance(camera);
        if d <= self.near_radius {
            ChunkDetail::Near
        } else if matches!(current, Some(ChunkDetail::Near))
            && d <= self.near_radius + DETAIL_HYSTERESIS
        {
            // A currently-full chunk keeps full detail through the hysteresis
            // band rather than re-detailing the instant it leaves near_radius.
            ChunkDetail::Near
        } else {
            ChunkDetail::Far
        }
    }

    // Decide this frame's chunk loads and evictions for a camera in chunk
    // `camera`.
    //
    // Evicts every tracked chunk now beyond the evict radius, then re-details
    // any tracked chunk that has crossed the near/far boundary (evict +
    // reload at the new detail), then dispatches the nearest in-window chunks
    // not yet tracked, up to the load budget, marking each `Pending`.
    pub fn plan(&mut self, camera: ChunkCoord) -> ChunkPlan {
        let mut to_evict: Vec<ChunkCoord> = Vec::new();

        // 1. Evict chunks that have drifted past the evict radius.
        let gone: Vec<ChunkCoord> = self
            .states
            .keys()
            .copied()
            .filter(|c| c.chebyshev_distance(camera) > self.evict_radius)
            .collect();
        for c in &gone {
            self.states.remove(c);
        }
        to_evict.extend_from_slice(&gone);

        // 2. Re-detail: a tracked chunk whose target detail no longer matches
        //    is dropped + evicted so the candidate scan reloads it at the new
        //    detail (a near<->far crossing as the camera moves).
        let redetail: Vec<ChunkCoord> = self
            .states
            .iter()
            .filter(|(c, slot)| self.target_detail(**c, camera, Some(slot.detail)) != slot.detail)
            .map(|(c, _)| *c)
            .collect();
        for c in &redetail {
            self.states.remove(c);
        }
        to_evict.extend_from_slice(&redetail);

        // 3. Collect in-window chunks (within far_radius) not yet tracked,
        //    nearest first.
        let mut candidates: Vec<ChunkCoord> = Vec::new();
        for dz in -self.far_radius..=self.far_radius {
            for dx in -self.far_radius..=self.far_radius {
                let c = camera.offset(dx, dz);
                if !self.states.contains_key(&c) {
                    candidates.push(c);
                }
            }
        }
        candidates.sort_unstable_by(|a, b| {
            a.sq_distance(camera)
                .cmp(&b.sq_distance(camera))
                // Stable tiebreak on the coordinate so the plan is deterministic.
                .then(a.cmp(b))
        });
        candidates.truncate(self.load_budget);

        let mut to_load = Vec::with_capacity(candidates.len());
        for &c in &candidates {
            let detail = self.target_detail(c, camera, None);
            self.states.insert(
                c,
                Slot {
                    state: ChunkState::Pending,
                    detail,
                },
            );
            to_load.push((c, detail));
        }

        to_evict.sort_unstable();
        ChunkPlan { to_load, to_evict }
    }

    // Mark a dispatched chunk resident once its mesh is on the GPU.
    //
    // A no-op if the chunk is no longer tracked -- the camera may have moved
    // far enough to evict it while its load was still in flight.
    pub fn mark_resident(&mut self, coord: ChunkCoord) {
        if let Some(slot) = self.states.get_mut(&coord) {
            slot.state = ChunkState::Resident;
        }
    }

    // Drop `coord` from tracking so a later [`plan`](Self::plan) will
    // re-dispatch it.
    //
    // The driver calls this when a dispatch could not be delivered to the
    // background worker, so the chunk is retried rather than stuck `Pending`.
    pub fn forget(&mut self, coord: ChunkCoord) {
        self.states.remove(&coord);
    }

    // Whether the window is still tracking `coord` (pending or resident).
    //
    // The driver checks this when a background load completes: a chunk
    // evicted mid-flight is no longer tracked and its mesh should be dropped.
    pub fn is_tracked(&self, coord: ChunkCoord) -> bool {
        self.states.contains_key(&coord)
    }

    // `(resident, pending)` chunk counts -- for diagnostics.
    pub fn counts(&self) -> (usize, usize) {
        let mut resident = 0;
        let mut pending = 0;
        for slot in self.states.values() {
            match slot.state {
                ChunkState::Resident => resident += 1,
                ChunkState::Pending => pending += 1,
            }
        }
        (resident, pending)
    }

    // `(near_resident, far_resident)` counts -- resident full chunks vs
    // resident impostors, for diagnostics / verifying the far band is active.
    pub fn counts_by_detail(&self) -> (usize, usize) {
        let mut near = 0;
        let mut far = 0;
        for slot in self.states.values() {
            if slot.state == ChunkState::Resident {
                match slot.detail {
                    ChunkDetail::Near => near += 1,
                    ChunkDetail::Far => far += 1,
                }
            }
        }
        (near, far)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(x: i32, z: i32) -> ChunkCoord {
        ChunkCoord::new(x, z)
    }

    // Coords in a plan's load list, detail dropped, for set-style assertions.
    fn load_coords(plan: &ChunkPlan) -> Vec<ChunkCoord> {
        plan.to_load.iter().map(|(c, _)| *c).collect()
    }

    #[test]
    fn plan_loads_nearest_in_window_chunks_within_budget() {
        // near=far=2 -> a 5x5 window of 25 chunks, impostors off; budget 4.
        let mut w = ChunkWindow::new(2, 2, 4);
        let plan = w.plan(cc(0, 0));
        assert!(plan.to_evict.is_empty());
        assert_eq!(plan.to_load.len(), 4);
        // The camera's own chunk is distance 0 -- it must be dispatched first.
        assert_eq!(plan.to_load[0], (cc(0, 0), ChunkDetail::Near));
        // Every dispatched chunk is within the load radius and full-detail.
        for (c, detail) in &plan.to_load {
            assert!(c.chebyshev_distance(cc(0, 0)) <= 2);
            assert_eq!(*detail, ChunkDetail::Near);
        }
    }

    #[test]
    fn plan_does_not_redispatch_tracked_chunks() {
        let mut w = ChunkWindow::new(3, 3, 100);
        let first = w.plan(cc(0, 0));
        // A generous budget loads the whole 7x7 window at once.
        assert_eq!(first.to_load.len(), 49);
        // Nothing left to dispatch on the next frame at the same position.
        let second = w.plan(cc(0, 0));
        assert!(second.to_load.is_empty());
        assert!(second.to_evict.is_empty());
    }

    #[test]
    fn plan_evicts_chunks_past_the_hysteresis_band() {
        let mut w = ChunkWindow::new(2, 2, 100);
        w.plan(cc(0, 0)); // load the 5x5 window around the origin
        // Move far enough that the origin chunk is past radius 2 + hysteresis 2.
        let plan = w.plan(cc(6, 0));
        assert!(plan.to_evict.contains(&cc(0, 0)));
    }

    #[test]
    fn evicted_chunk_can_be_reloaded_after_returning() {
        let mut w = ChunkWindow::new(1, 1, 100);
        w.plan(cc(0, 0));
        w.plan(cc(20, 0)); // evicts the origin window entirely
        assert!(!w.is_tracked(cc(0, 0)));
        let plan = w.plan(cc(0, 0));
        assert!(load_coords(&plan).contains(&cc(0, 0)));
    }

    #[test]
    fn mark_resident_promotes_a_pending_chunk() {
        let mut w = ChunkWindow::new(0, 0, 1);
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load, vec![(cc(0, 0), ChunkDetail::Near)]);
        assert_eq!(w.counts(), (0, 1));
        w.mark_resident(cc(0, 0));
        assert_eq!(w.counts(), (1, 0));
    }

    #[test]
    fn mark_resident_of_an_untracked_chunk_is_a_noop() {
        let mut w = ChunkWindow::new(0, 0, 1);
        w.mark_resident(cc(9, 9)); // never planned -- must not panic or insert
        assert_eq!(w.counts(), (0, 0));
        assert!(!w.is_tracked(cc(9, 9)));
    }

    #[test]
    fn forget_lets_a_chunk_be_redispatched() {
        let mut w = ChunkWindow::new(0, 0, 1);
        w.plan(cc(0, 0));
        assert!(w.is_tracked(cc(0, 0)));
        w.forget(cc(0, 0));
        assert!(!w.is_tracked(cc(0, 0)));
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load, vec![(cc(0, 0), ChunkDetail::Near)]);
    }

    #[test]
    fn zero_radius_and_budget_are_floored() {
        // near floored to 0 (just the camera chunk), far to near, budget to 1.
        let mut w = ChunkWindow::new(-5, -5, 0);
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load, vec![(cc(0, 0), ChunkDetail::Near)]);
    }

    #[test]
    fn far_band_chunks_load_as_impostors() {
        // near 1, far 3: the 3x3 core is full detail, the surrounding rings are
        // impostors. A generous budget loads the whole 7x7 window at once.
        let mut w = ChunkWindow::new(1, 3, 100);
        let plan = w.plan(cc(0, 0));
        assert_eq!(plan.to_load.len(), 49);
        for (c, detail) in &plan.to_load {
            let d = c.chebyshev_distance(cc(0, 0));
            let expected = if d <= 1 {
                ChunkDetail::Near
            } else {
                ChunkDetail::Far
            };
            assert_eq!(*detail, expected, "chunk {:?} at distance {}", c, d);
        }
    }

    #[test]
    fn crossing_the_boundary_redetails_a_chunk() {
        let mut w = ChunkWindow::new(1, 3, 100);
        // Load + resolve the whole window around the origin.
        let plan = w.plan(cc(0, 0));
        let crossing = cc(2, 0); // distance 2 -> Far impostor at the origin
        assert!(plan.to_load.contains(&(crossing, ChunkDetail::Far)));
        for (c, _) in plan.to_load.clone() {
            w.mark_resident(c);
        }
        let (near0, far0) = w.counts_by_detail();
        assert!(near0 > 0 && far0 > 0);

        // Step toward `crossing` so it falls inside near_radius: it must be
        // re-detailed (evicted) and re-dispatched as Near.
        let plan = w.plan(cc(1, 0));
        assert!(plan.to_evict.contains(&crossing));
        assert!(plan.to_load.contains(&(crossing, ChunkDetail::Near)));
    }

    #[test]
    fn detail_hysteresis_holds_a_full_chunk_through_the_band() {
        // near 2, far 5. A chunk at the origin starts Near (camera at origin).
        let mut w = ChunkWindow::new(2, 5, 200);
        for (c, _) in w.plan(cc(0, 0)).to_load {
            w.mark_resident(c);
        }
        // Camera steps to (3,0): origin chunk is now chebyshev distance 3 =
        // near_radius(2) + hysteresis(1), so it stays Near, no re-detail.
        let plan = w.plan(cc(3, 0));
        assert!(!plan.to_evict.contains(&cc(0, 0)));
        // Step once more to (4,0): distance 4 > 2 + 1, so it downgrades to Far.
        let plan = w.plan(cc(4, 0));
        assert!(plan.to_evict.contains(&cc(0, 0)));
        assert!(plan.to_load.contains(&(cc(0, 0), ChunkDetail::Far)));
    }

    #[test]
    fn equal_radii_disable_the_far_band() {
        // far == near -> every in-window chunk is Near, exactly as the original
        // single-detail window.
        let mut w = ChunkWindow::new(3, 3, 100);
        let plan = w.plan(cc(0, 0));
        assert!(plan.to_load.iter().all(|(_, d)| *d == ChunkDetail::Near));
        assert_eq!(w.counts_by_detail().1, 0); // never any far chunks
    }
}
