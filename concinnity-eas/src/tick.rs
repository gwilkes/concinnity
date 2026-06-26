// concinnity-eas/src/tick.rs
//
// Monotonic change tick and the wrapping-relative comparison change detection
// uses. Ticks are u32 and wrap. A system records the tick it last ran at and
// asks whether a column changed since. A naive `>` breaks at wraparound, so the
// comparison treats the 32-bit difference as signed (a half-range window), and
// `clamp_to` keeps a stored tick from drifting more than half the range behind
// the current tick over a long session.

// Half the u32 range. A tick older than this relative to the current tick is
// clamped forward so the signed-window comparison never aliases.
pub const MAX_CHANGE_AGE: u32 = u32::MAX / 2;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Tick(pub u32);

impl Tick {
    pub const ZERO: Tick = Tick(0);

    pub fn get(self) -> u32 {
        self.0
    }

    // Advance to the next tick (wrapping) and return the new value.
    pub fn bump(&mut self) -> Tick {
        self.0 = self.0.wrapping_add(1);
        *self
    }

    // Whether `self` is strictly newer than `other`, robust to u32 wraparound.
    // The difference is interpreted as signed: positive means ahead, within the
    // half-range window the clamp guarantees.
    pub fn is_newer_than(self, other: Tick) -> bool {
        (self.0.wrapping_sub(other.0) as i32) > 0
    }

    // Pull a stored tick forward if it has fallen more than MAX_CHANGE_AGE
    // behind `now`, so the signed-window comparison stays valid no matter how
    // long the world runs. Recent ticks are returned unchanged.
    pub fn clamp_to(self, now: Tick) -> Tick {
        if now.0.wrapping_sub(self.0) > MAX_CHANGE_AGE {
            Tick(now.0.wrapping_sub(MAX_CHANGE_AGE))
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_zero() {
        assert_eq!(Tick::default(), Tick::ZERO);
    }

    #[test]
    fn bump_advances() {
        let mut t = Tick::ZERO;
        assert_eq!(t.bump(), Tick(1));
        assert_eq!(t.bump(), Tick(2));
        assert_eq!(t.get(), 2);
    }

    #[test]
    fn newer_than_is_strict() {
        assert!(Tick(2).is_newer_than(Tick(1)));
        assert!(!Tick(1).is_newer_than(Tick(1)));
        assert!(!Tick(1).is_newer_than(Tick(2)));
    }

    #[test]
    fn newer_than_survives_wraparound() {
        // Just after wrap, Tick(0) is newer than Tick(u32::MAX).
        assert!(Tick(0).is_newer_than(Tick(u32::MAX)));
        assert!(Tick(5).is_newer_than(Tick(u32::MAX - 2)));
        assert!(!Tick(u32::MAX).is_newer_than(Tick(0)));
    }

    #[test]
    fn clamp_pulls_stale_ticks_forward() {
        let now = Tick(MAX_CHANGE_AGE + 100);
        // A tick at 0 is too old: clamped to now - MAX_CHANGE_AGE.
        assert_eq!(Tick::ZERO.clamp_to(now), Tick(100));
        // Comparisons stay correct after clamping.
        assert!(now.is_newer_than(Tick::ZERO.clamp_to(now)));
    }

    #[test]
    fn clamp_leaves_recent_ticks_untouched() {
        let now = Tick(1000);
        assert_eq!(Tick(990).clamp_to(now), Tick(990));
    }
}
