// src/gfx/shadow_schedule.rs
//
// Cross-backend cascade re-render scheduling for the cascaded shadow map. The
// shadow pass re-rasterizes all scene geometry into every cascade slice, so it
// is one of the heaviest passes; `ShadowUpdate::Hybrid` amortizes the far
// cascades across frames (near cascade every frame, one far cascade round-robin)
// while keeping each slice primed before it is sampled. Shared by all three
// backends so the policy lives once next to the CSM math in `csm.rs`.

use crate::assets::ShadowUpdate;
use crate::gfx::render_types::NUM_SHADOW_CASCADES;

// Round-robin clock + primed-set state for the cascade re-render schedule. One
// per renderer; `next_mask` advances it once per frame and returns which
// cascades to re-render. The caller refreshes only those cascades' light VPs and
// re-rasterizes only those slices, so skipped cascades keep the depth + VP they
// were last rendered with.
#[derive(Debug, Default)]
pub struct ShadowCascadeScheduler {
    // Advances once per shadow update; selects which far cascade Hybrid mode
    // refreshes this frame.
    clock: u32,
    // Bit `i` set once cascade `i` has been rendered since the scheduler was
    // created. Unprimed cascades are force-rendered so a slice is never sampled
    // before it holds valid depth.
    primed_mask: u32,
}

impl ShadowCascadeScheduler {
    // Choose which cascades to re-render this frame and advance the round-robin
    // clock. Delegates the selection to the pure `select_cascade_mask` and
    // applies its side effects: advance the clock and record the newly primed
    // set. Bit `i` set in the result means cascade `i` re-renders this frame.
    pub fn next_mask(&mut self, update: ShadowUpdate, active_cascades: u32) -> u32 {
        let (mask, primed) =
            select_cascade_mask(update, self.clock, self.primed_mask, active_cascades);
        self.clock = self.clock.wrapping_add(1);
        self.primed_mask = primed;
        mask
    }
}

// Pure cascade-selection step, split out of `next_mask` so the priming +
// round-robin policy is unit-testable without renderer state.
//
// Given the update policy, the current round-robin clock, and which cascades
// have already been primed, returns `(render_mask, new_primed_mask)`:
//
//   - `scheduled` is the policy's steady-state set (EveryFrame = all; Hybrid =
//     the near cascade plus one round-robin far cascade).
//   - any cascade not yet primed is force-rendered this frame so its slice is
//     never sampled before it holds valid depth. Because a cascade's bit is set
//     in `primed` the first frame it renders and never cleared, priming renders
//     each cascade exactly once on first access; it is never re-primed.
fn select_cascade_mask(
    update: ShadowUpdate,
    clock: u32,
    primed: u32,
    active_cascades: u32,
) -> (u32, u32) {
    // Only the first `active` cascades exist this frame; the rest are never
    // scheduled (and never sampled -- the shader bounds its lookup by the same
    // count). `primed` may carry bits for cascades that were active under a
    // higher count; `& all` masks them off so they never force a render.
    let active = active_cascades.clamp(1, NUM_SHADOW_CASCADES as u32);
    let all = (1u32 << active) - 1;
    let scheduled = match update {
        ShadowUpdate::EveryFrame => all,
        ShadowUpdate::Hybrid => {
            let far_count = (active - 1).max(1);
            let far = 1 + (clock % far_count);
            (1u32 | (1u32 << far)) & all
        }
    };
    let unprimed = all & !primed;
    let mask = (scheduled | unprimed) & all;
    (mask, primed | mask)
}

#[cfg(test)]
mod tests {
    use super::{ShadowCascadeScheduler, select_cascade_mask};
    use crate::assets::ShadowUpdate;
    use crate::gfx::render_types::NUM_SHADOW_CASCADES;

    const ALL: u32 = (1u32 << NUM_SHADOW_CASCADES) - 1;

    #[test]
    fn first_access_primes_every_cascade_at_once() {
        // From the unprimed state both policies render all cascades on the
        // very first frame, so no slice is sampled before it holds depth.
        for update in [ShadowUpdate::Hybrid, ShadowUpdate::EveryFrame] {
            let (mask, primed) = select_cascade_mask(update, 0, 0, NUM_SHADOW_CASCADES as u32);
            assert_eq!(mask, ALL, "{update:?} should prime all cascades on frame 0");
            assert_eq!(primed, ALL, "every cascade is primed after frame 0");
        }
    }

    #[test]
    fn fewer_active_cascades_only_schedule_the_live_slots() {
        // With active_cascades = 2, only cascades 0 and 1 are ever scheduled --
        // the inactive 2,3 never render even if their primed bits are stale from
        // a higher count. EveryFrame renders both; Hybrid (far_count = 1) also
        // renders both (no far cascade to amortize with only one far slot).
        let live = 0b11u32;
        for update in [ShadowUpdate::Hybrid, ShadowUpdate::EveryFrame] {
            // Stale primed bits for the now-inactive cascades must not leak in.
            let (mask, primed) = select_cascade_mask(update, 0, ALL, 2);
            assert_eq!(mask, live, "{update:?} scheduled an inactive cascade");
            assert_eq!(
                primed & !live,
                ALL & !live,
                "inactive primed bits untouched"
            );
        }
        // active = 1: only the near cascade ever renders.
        let (mask, _) = select_cascade_mask(ShadowUpdate::Hybrid, 3, 0, 1);
        assert_eq!(mask, 0b1);
    }

    #[test]
    fn priming_never_re_renders_a_cascade() {
        // Once primed, Hybrid drops back to its steady-state set: the near
        // cascade plus exactly one far cascade. The full-prime render only
        // happens while a cascade is still unprimed, never again.
        let far_count = (NUM_SHADOW_CASCADES as u32 - 1).max(1);
        for clock in 0..(far_count * 3) {
            let (mask, primed) =
                select_cascade_mask(ShadowUpdate::Hybrid, clock, ALL, NUM_SHADOW_CASCADES as u32);
            assert_eq!(primed, ALL, "already-primed set is unchanged");
            assert_eq!(mask & 1, 1, "near cascade refreshes every frame");
            let extra = (mask & !1u32).count_ones();
            assert_eq!(extra, 1, "exactly one far cascade refreshes per frame");
        }
    }

    #[test]
    fn hybrid_round_robins_every_far_cascade() {
        // Over `far_count` consecutive frames the steady-state Hybrid mask
        // touches each far cascade exactly once.
        let far_count = (NUM_SHADOW_CASCADES as u32 - 1).max(1);
        let mut union = 0u32;
        for clock in 0..far_count {
            let (mask, _) =
                select_cascade_mask(ShadowUpdate::Hybrid, clock, ALL, NUM_SHADOW_CASCADES as u32);
            union |= mask;
        }
        assert_eq!(union, ALL, "every cascade is refreshed within one round");
    }

    #[test]
    fn every_frame_always_renders_all() {
        for clock in 0..5 {
            let (mask, primed) = select_cascade_mask(
                ShadowUpdate::EveryFrame,
                clock,
                ALL,
                NUM_SHADOW_CASCADES as u32,
            );
            assert_eq!(mask, ALL);
            assert_eq!(primed, ALL);
        }
    }

    #[test]
    fn priming_is_monotonic_and_one_shot_per_cascade() {
        // Drive the policy frame by frame from the unprimed state and assert
        // the primed set only grows, and any cascade rendered purely for
        // priming (in the mask but not in that frame's scheduled set) is
        // rendered at most once across the whole run.
        let mut primed = 0u32;
        let mut prime_renders = [0u32; NUM_SHADOW_CASCADES];
        for clock in 0..(NUM_SHADOW_CASCADES as u32 + 4) {
            let far_count = (NUM_SHADOW_CASCADES as u32 - 1).max(1);
            let scheduled_near_far = 1u32 | (1u32 << (1 + clock % far_count));
            let before = primed;
            let (mask, next) = select_cascade_mask(
                ShadowUpdate::Hybrid,
                clock,
                primed,
                NUM_SHADOW_CASCADES as u32,
            );
            assert_eq!(next & before, before, "primed set must never lose a bit");
            for (c, count) in prime_renders.iter_mut().enumerate() {
                let bit = 1u32 << c;
                let primed_only = (mask & bit != 0) && (scheduled_near_far & bit == 0);
                if primed_only {
                    *count += 1;
                }
            }
            primed = next;
        }
        for (c, count) in prime_renders.iter().enumerate() {
            assert!(
                *count <= 1,
                "cascade {c} was force-primed {count} times (>1)"
            );
        }
        assert_eq!(primed, ALL, "all cascades primed after the run");
    }

    #[test]
    fn scheduler_advances_clock_and_accumulates_primes() {
        // The struct wrapper primes everything on frame 0, then steady-states
        // to near + one far cascade and rotates the far cascade each frame.
        let mut sched = ShadowCascadeScheduler::default();
        assert_eq!(
            sched.next_mask(ShadowUpdate::Hybrid, NUM_SHADOW_CASCADES as u32),
            ALL,
            "frame 0 primes all"
        );
        let far_count = (NUM_SHADOW_CASCADES as u32 - 1).max(1);
        let mut union = 0u32;
        for _ in 0..far_count {
            let mask = sched.next_mask(ShadowUpdate::Hybrid, NUM_SHADOW_CASCADES as u32);
            assert_eq!(mask & 1, 1, "near cascade refreshes every frame");
            assert_eq!((mask & !1u32).count_ones(), 1, "one far cascade per frame");
            union |= mask;
        }
        assert_eq!(union, ALL, "every cascade refreshed within one round");
    }
}
