// src/gfx/display_mode.rs
//
// The backend-agnostic display-mode list behind the "Resolution" settings row.
// A backend enumerates the modes (width x height at refresh rate) the display
// it renders to supports; this module holds the shared shaping: the row/list
// label format, the dedup + sort that turns a raw enumeration into the menu
// list, the persisted-choice -> list-index recovery, and the static fallback a
// backend without enumeration (or an embedded view with no window) uses so the
// row still drives the windowed resize path.
//
// How a chosen mode is applied stays per window mode: windowed resizes the
// window's content area to the resolution; fullscreen switches the display to
// the mode itself (resolution + refresh rate); borderless always covers the
// display's current mode, so the row is inert there.

// One display mode the hardware supports: pixel dimensions plus refresh rate.
// `refresh_hz` of 0 means unknown (some built-in panels report none); the label
// then omits the rate and a fullscreen apply keeps the display's current rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

impl DisplayMode {
    // The option text shown in the Resolution row and its dropdown list, e.g.
    // "2560 x 1440 (165Hz)"; a mode with an unknown rate reads "2560 x 1440".
    pub fn label(&self) -> String {
        if self.refresh_hz == 0 {
            format!("{} x {}", self.width, self.height)
        } else {
            format!("{} x {} ({}Hz)", self.width, self.height, self.refresh_hz)
        }
    }
}

// The menu list for a raw enumeration: duplicates collapsed, ordered by width,
// then height, then refresh rate ascending (each resolution's rate variants
// group together).
pub(crate) fn normalize(mut modes: Vec<DisplayMode>) -> Vec<DisplayMode> {
    modes.sort();
    modes.dedup();
    modes
}

// The list index for a (possibly persisted) choice. An exact match wins; a
// choice whose resolution is listed but whose rate is not (the display
// changed) snaps to that resolution's nearest rate; otherwise the nearest
// resolution by pixel count, so a stale persisted mode still lands somewhere
// sensible. Returns 0 for an empty list (callers guard, but stay total).
pub(crate) fn index_of(modes: &[DisplayMode], choice: DisplayMode) -> usize {
    if let Some(i) = modes.iter().position(|m| *m == choice) {
        return i;
    }
    let same_res = modes
        .iter()
        .enumerate()
        .filter(|(_, m)| m.width == choice.width && m.height == choice.height)
        .min_by_key(|(_, m)| m.refresh_hz.abs_diff(choice.refresh_hz))
        .map(|(i, _)| i);
    if let Some(i) = same_res {
        return i;
    }
    let choice_px = u64::from(choice.width) * u64::from(choice.height);
    modes
        .iter()
        .enumerate()
        .min_by_key(|(_, m)| {
            let px = u64::from(m.width) * u64::from(m.height);
            (
                px.abs_diff(choice_px),
                m.refresh_hz.abs_diff(choice.refresh_hz),
            )
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// The index in `modes` of the native mode to apply for `want`: an exact
// (resolution, rate) match wins; a `want` with an unknown rate (0) or a rate
// the display no longer offers takes the matching resolution's highest rate;
// `None` when no mode has that resolution (e.g. a stale persisted choice from
// another monitor), so the caller leaves the display alone. Used by the
// DirectX + Vulkan apply paths; Metal does the same matching natively over
// CGDisplayModes (`find_native_mode`), so this is dead on a Metal-only build.
#[cfg_attr(backend_metal, allow(dead_code))]
pub(crate) fn best_native_index(modes: &[DisplayMode], want: DisplayMode) -> Option<usize> {
    let mut best: Option<(u32, usize)> = None;
    for (i, m) in modes.iter().enumerate() {
        if m.width != want.width || m.height != want.height {
            continue;
        }
        if want.refresh_hz != 0 && m.refresh_hz == want.refresh_hz {
            return Some(i);
        }
        if best.as_ref().is_none_or(|(hz, _)| m.refresh_hz > *hz) {
            best = Some((m.refresh_hz, i));
        }
    }
    best.map(|(_, i)| i)
}

// The static list used when the backend cannot enumerate the display (DirectX /
// Vulkan today, or an embedded view with no window). Common resolutions with no
// rate, so the row keeps driving the windowed resize path.
pub(crate) fn fallback_modes() -> Vec<DisplayMode> {
    [(1280, 720), (1600, 900), (1920, 1080), (2560, 1440)]
        .into_iter()
        .map(|(width, height)| DisplayMode {
            width,
            height,
            refresh_hz: 0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mode(width: u32, height: u32, refresh_hz: u32) -> DisplayMode {
        DisplayMode {
            width,
            height,
            refresh_hz,
        }
    }

    #[test]
    fn label_includes_rate_only_when_known() {
        assert_eq!(mode(2560, 1440, 165).label(), "2560 x 1440 (165Hz)");
        assert_eq!(mode(1920, 1080, 60).label(), "1920 x 1080 (60Hz)");
        assert_eq!(mode(1280, 720, 0).label(), "1280 x 720");
    }

    #[test]
    fn normalize_dedups_and_groups_rates_per_resolution() {
        let raw = vec![
            mode(2560, 1440, 60),
            mode(1024, 768, 120),
            mode(1024, 768, 60),
            mode(2560, 1440, 165),
            mode(1024, 768, 60), // duplicate
            mode(1280, 720, 75),
        ];
        let list = normalize(raw);
        assert_eq!(
            list,
            vec![
                mode(1024, 768, 60),
                mode(1024, 768, 120),
                mode(1280, 720, 75),
                mode(2560, 1440, 60),
                mode(2560, 1440, 165),
            ]
        );
    }

    #[test]
    fn index_of_prefers_exact_then_rate_then_resolution() {
        let list = vec![
            mode(1280, 720, 60),
            mode(1280, 720, 120),
            mode(1920, 1080, 60),
            mode(2560, 1440, 165),
        ];
        // Exact match.
        assert_eq!(index_of(&list, mode(1920, 1080, 60)), 2);
        // Listed resolution, unlisted rate: nearest rate for that resolution.
        assert_eq!(index_of(&list, mode(1280, 720, 144)), 1);
        // A fallback-preset choice (rate 0) lands on the resolution's lowest rate.
        assert_eq!(index_of(&list, mode(1280, 720, 0)), 0);
        // Unlisted resolution: nearest by pixel count (1600x900 sits closer to
        // 1280x720 than to 1920x1080; 2048x1152 closer to 1920x1080).
        assert_eq!(index_of(&list, mode(1600, 900, 60)), 0);
        assert_eq!(index_of(&list, mode(2048, 1152, 60)), 2);
        // Empty list stays total.
        assert_eq!(index_of(&[], mode(1920, 1080, 60)), 0);
    }

    #[test]
    fn best_native_index_snaps_rate_and_rejects_unknown_resolution() {
        let list = vec![
            mode(1280, 720, 60),
            mode(1280, 720, 120),
            mode(1920, 1080, 60),
        ];
        // Exact (resolution, rate) match wins.
        assert_eq!(best_native_index(&list, mode(1280, 720, 120)), Some(1));
        // Unknown wanted rate (0, a fallback preset) takes the resolution's
        // highest rate.
        assert_eq!(best_native_index(&list, mode(1280, 720, 0)), Some(1));
        // A rate the display no longer offers also snaps to the highest.
        assert_eq!(best_native_index(&list, mode(1920, 1080, 144)), Some(2));
        // An unlisted resolution applies nothing (the display is left alone).
        assert_eq!(best_native_index(&list, mode(2560, 1440, 60)), None);
    }

    #[test]
    fn fallback_modes_are_sorted_rate_free_presets() {
        let list = fallback_modes();
        assert_eq!(normalize(list.clone()), list);
        assert!(list.len() > 2, "must expand as a dropdown row");
        assert!(list.iter().all(|m| m.refresh_hz == 0));
        assert!(list.iter().any(|m| (m.width, m.height) == (1920, 1080)));
    }
}
