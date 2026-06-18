// src/gfx/streaming.rs
//
// Asset-streaming policy core.
//
// Pure decision logic: given each streamable item's priority score (camera
// distance), a per-frame load budget, and a cap on how many items may be
// resident at once, this decides *which* items to load and *which* to evict.
// It performs no I/O, spawns no threads, and touches no backend.
//
// This module is deliberately written against `core` + `alloc` constructs
// only (`Vec`, primitives, slices) so it can move into a future `no_std`
// client runtime unchanged. The `std`-side half -- the background fetch
// thread, the channels, and the GPU upload -- lives in
// `crate::app::texture_stream`. Keep that boundary: no `std::`-only types
// (threads, files, `HashMap`, `Instant`, ...) belong in this file.

// Residency state of a single streamable item.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StreamState {
    // Not on the GPU; eligible to be loaded.
    Unloaded,
    // A background load has been dispatched but has not completed.
    Pending,
    // On the GPU and ready to sample.
    Resident,
}

#[derive(Clone, Copy, Debug)]
struct Item {
    state: StreamState,
    // Priority score; lower = more urgent. The driver feeds squared camera
    // distance, so "closer to the camera" sorts first and no `sqrt` (which
    // lives in `std`, not `core`) is needed here.
    score: f32,
    // Frame this item was last referenced; the LRU tiebreak when two resident
    // items have an equal score during eviction.
    last_touch: u64,
}

// The load / evict decisions produced by one [`StreamPlanner::plan`] call.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct StreamPlan {
    // Item ids whose background load should be dispatched this frame.
    pub to_load: Vec<usize>,
    // Item ids that should be evicted from the GPU this frame.
    pub to_evict: Vec<usize>,
}

// Decides what to stream in and out of a fixed-size residency pool.
//
// The planner owns only residency *bookkeeping*: it never reads or writes a
// GPU resource. Each frame the driver updates scores, calls [`plan`], and
// reports completed loads back via [`mark_resident`].
//
// [`plan`]: StreamPlanner::plan
// [`mark_resident`]: StreamPlanner::mark_resident
pub struct StreamPlanner {
    items: Vec<Item>,
    // Max number of loads `plan` will dispatch in a single call.
    load_budget: usize,
    // Max number of items allowed Resident-or-Pending simultaneously. Once the
    // pool is full a load can only proceed by evicting a lower-priority item.
    resident_cap: usize,
}

impl StreamPlanner {
    // Create a planner tracking `count` items, all initially `Unloaded`.
    //
    // `load_budget` and `resident_cap` are both clamped to at least 1 so a
    // zero from a misconfigured asset cannot wedge streaming permanently.
    pub fn new(count: usize, load_budget: usize, resident_cap: usize) -> Self {
        Self {
            items: vec![
                Item {
                    state: StreamState::Unloaded,
                    score: 0.0,
                    last_touch: 0,
                };
                count
            ],
            load_budget: load_budget.max(1),
            resident_cap: resident_cap.max(1),
        }
    }

    // Number of tracked items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    // Residency state of item `id`, or `None` if `id` is out of range.
    pub fn state(&self, id: usize) -> Option<StreamState> {
        self.items.get(id).map(|i| i.state)
    }

    // Set item `id`'s priority score (lower = loaded sooner / evicted later).
    // Out-of-range ids are ignored.
    pub fn set_score(&mut self, id: usize, score: f32) {
        if let Some(item) = self.items.get_mut(id) {
            item.score = score;
        }
    }

    // Record that item `id` was referenced on `frame`. Refreshes the LRU
    // tiebreak used when evicting equally-scored resident items.
    pub fn touch(&mut self, id: usize, frame: u64) {
        if let Some(item) = self.items.get_mut(id) {
            item.last_touch = frame;
        }
    }

    // Report that a dispatched load for item `id` has completed and the
    // resource is now on the GPU.
    pub fn mark_resident(&mut self, id: usize, frame: u64) {
        if let Some(item) = self.items.get_mut(id) {
            item.state = StreamState::Resident;
            item.last_touch = frame;
        }
    }

    // Force item `id` back to `Unloaded` (e.g. after a failed load that
    // should be retried). Out-of-range ids are ignored.
    pub fn mark_unloaded(&mut self, id: usize) {
        if let Some(item) = self.items.get_mut(id) {
            item.state = StreamState::Unloaded;
        }
    }

    // `(resident, pending, unloaded)` item counts, for diagnostics.
    pub fn counts(&self) -> (usize, usize, usize) {
        let mut resident = 0;
        let mut pending = 0;
        let mut unloaded = 0;
        for item in &self.items {
            match item.state {
                StreamState::Resident => resident += 1,
                StreamState::Pending => pending += 1,
                StreamState::Unloaded => unloaded += 1,
            }
        }
        (resident, pending, unloaded)
    }

    // Decide which items to load and evict this frame.
    //
    // `Unloaded` items are considered best-score-first. While the pool has
    // spare capacity they are simply scheduled to load. Once the pool is full
    // a candidate can still load by evicting the worst-scored resident item,
    // but only when that resident is strictly farther than the candidate, so
    // equal-priority items never churn. At most `load_budget` loads are
    // scheduled per call.
    //
    // This method mutates planner state: scheduled items become `Pending` and
    // evicted items become `Unloaded`, so a later `plan` call in the same
    // frame (or the next frame) will not re-pick them.
    pub fn plan(&mut self) -> StreamPlan {
        let mut plan = StreamPlan::default();

        // Candidate loads: every Unloaded item, best (lowest) score first.
        let mut candidates: Vec<usize> = self
            .items
            .iter()
            .enumerate()
            .filter(|(_, it)| it.state == StreamState::Unloaded)
            .map(|(id, _)| id)
            .collect();
        candidates.sort_by(|&a, &b| {
            self.items[a]
                .score
                .partial_cmp(&self.items[b].score)
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Items occupying (or about to occupy) a pool slot.
        let occupied = |items: &[Item]| {
            items
                .iter()
                .filter(|it| it.state != StreamState::Unloaded)
                .count()
        };

        for &id in &candidates {
            if plan.to_load.len() >= self.load_budget {
                break;
            }
            let free = self.resident_cap.saturating_sub(occupied(&self.items));
            if free > 0 {
                self.items[id].state = StreamState::Pending;
                plan.to_load.push(id);
                continue;
            }
            // Pool is full: evict the worst resident if it is genuinely
            // farther than this candidate, otherwise stop; every remaining
            // candidate is even lower priority.
            match self.worst_resident(&plan.to_evict) {
                Some(victim) if self.items[victim].score > self.items[id].score => {
                    self.items[victim].state = StreamState::Unloaded;
                    self.items[id].state = StreamState::Pending;
                    plan.to_evict.push(victim);
                    plan.to_load.push(id);
                }
                _ => break,
            }
        }

        plan
    }

    // The resident item with the worst (highest) score, breaking ties toward
    // the least-recently-touched. `excluded` items are skipped so a single
    // `plan` call never evicts the same victim twice.
    fn worst_resident(&self, excluded: &[usize]) -> Option<usize> {
        let mut worst: Option<usize> = None;
        for (id, item) in self.items.iter().enumerate() {
            if item.state != StreamState::Resident || excluded.contains(&id) {
                continue;
            }
            match worst {
                None => worst = Some(id),
                Some(w) => {
                    let cur = &self.items[w];
                    let better_victim = item.score > cur.score
                        || (item.score == cur.score && item.last_touch < cur.last_touch);
                    if better_victim {
                        worst = Some(id);
                    }
                }
            }
        }
        worst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_planner_has_all_items_unloaded() {
        let p = StreamPlanner::new(3, 4, 8);
        assert_eq!(p.len(), 3);
        for id in 0..3 {
            assert_eq!(p.state(id), Some(StreamState::Unloaded));
        }
        assert_eq!(p.state(3), None);
        assert_eq!(p.counts(), (0, 0, 3));
    }

    #[test]
    fn zero_budget_and_cap_are_clamped_to_one() {
        let mut p = StreamPlanner::new(2, 0, 0);
        let plan = p.plan();
        // A budget/cap of 0 would wedge streaming; clamped to 1 it still moves.
        assert_eq!(plan.to_load.len(), 1);
    }

    #[test]
    fn plan_loads_nearest_items_first_within_budget() {
        let mut p = StreamPlanner::new(4, 2, 8);
        p.set_score(0, 30.0);
        p.set_score(1, 10.0);
        p.set_score(2, 20.0);
        p.set_score(3, 40.0);
        let plan = p.plan();
        // Budget is 2; the two lowest scores (ids 1 then 2) are picked in order.
        assert_eq!(plan.to_load, vec![1, 2]);
        assert!(plan.to_evict.is_empty());
        assert_eq!(p.state(1), Some(StreamState::Pending));
        assert_eq!(p.state(2), Some(StreamState::Pending));
        assert_eq!(p.state(0), Some(StreamState::Unloaded));
    }

    #[test]
    fn pending_items_are_not_re_dispatched() {
        let mut p = StreamPlanner::new(3, 1, 8);
        let first = p.plan();
        assert_eq!(first.to_load.len(), 1);
        let dispatched = first.to_load[0];
        let second = p.plan();
        assert!(!second.to_load.contains(&dispatched));
    }

    #[test]
    fn resident_cap_blocks_loading_when_no_eviction_is_worthwhile() {
        let mut p = StreamPlanner::new(3, 4, 2);
        // Two near items become resident.
        p.set_score(0, 1.0);
        p.set_score(1, 2.0);
        p.set_score(2, 99.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1);
        p.mark_resident(1, 1);
        // The far item cannot displace either resident; they are both closer.
        let plan = p.plan();
        assert!(plan.to_load.is_empty());
        assert!(plan.to_evict.is_empty());
    }

    #[test]
    fn closer_candidate_evicts_a_farther_resident() {
        let mut p = StreamPlanner::new(3, 4, 2);
        p.set_score(0, 50.0);
        p.set_score(1, 60.0);
        p.set_score(2, 99.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1);
        p.mark_resident(1, 1);
        // Item 2 walks closer than resident item 1.
        p.set_score(2, 10.0);
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![2]);
        assert_eq!(plan.to_evict, vec![1]);
        assert_eq!(p.state(1), Some(StreamState::Unloaded));
        assert_eq!(p.state(2), Some(StreamState::Pending));
    }

    #[test]
    fn eviction_breaks_score_ties_toward_least_recently_touched() {
        let mut p = StreamPlanner::new(3, 4, 2);
        p.set_score(0, 5.0);
        p.set_score(1, 5.0);
        p.set_score(2, 99.0); // far away initially, so 0 and 1 become resident
        let plan = p.plan();
        assert_eq!(plan.to_load, vec![0, 1]);
        p.mark_resident(0, 1);
        p.mark_resident(1, 1);
        // Item 0 is referenced more recently than item 1.
        p.touch(0, 100);
        p.touch(1, 50);
        // A closer candidate forces one eviction; the staler resident loses.
        p.set_score(2, 1.0);
        let plan = p.plan();
        assert_eq!(plan.to_evict, vec![1]);
    }

    #[test]
    fn counts_track_state_transitions() {
        let mut p = StreamPlanner::new(3, 1, 8);
        assert_eq!(p.counts(), (0, 0, 3));
        let plan = p.plan();
        let id = plan.to_load[0];
        assert_eq!(p.counts(), (0, 1, 2));
        p.mark_resident(id, 1);
        assert_eq!(p.counts(), (1, 0, 2));
        p.mark_unloaded(id);
        assert_eq!(p.counts(), (0, 0, 3));
    }

    #[test]
    fn empty_planner_plans_nothing() {
        let mut p = StreamPlanner::new(0, 4, 8);
        assert_eq!(p.len(), 0);
        assert_eq!(p.plan(), StreamPlan::default());
    }
}
