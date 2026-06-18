// src/gfx/scroll_layout.rs
//
// Pure vertical-layout solver shared by the build pipeline and the client. A
// scrollable settings page is a vertical stack of rows; some rows belong to a
// collapsible group and disappear when that group is collapsed; the visible
// stack may be taller than the band it shows through, in which case it scrolls.
//
// This module owns only the geometry: given each row's natural height, its
// group membership, the per-group collapsed state, the band height, and the
// current scroll offset, it returns where each row ends up (a delta from its
// build-time position), whether it is visible, the total content height, the
// clamped scroll, and the scrollbar thumb size + offset. It knows nothing about
// AssetIds or rendering; the client maps these results onto concrete elements.

// One row in the stack. `group` is the index of the collapsible group whose
// collapsed state hides this row, or `None` for a row that is always shown
// (chrome-in-stack, e.g. a group header).
#[derive(Debug, Clone, Copy)]
pub struct RowSpec {
    pub height: f32,
    pub group: Option<usize>,
}

// Where one row ends up after solving. `dy` is the vertical offset from the
// row's build-time (all-expanded, unscrolled) position, so a client adds it to
// the element's authored y. A hidden row's `dy` is unspecified.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowPlacement {
    pub dy: f32,
    pub visible: bool,
}

// The full solution for one panel.
#[derive(Debug, Clone, PartialEq)]
pub struct Solved {
    pub rows: Vec<RowPlacement>,
    // Total height of the visible stack (sum of shown row heights).
    pub content_height: f32,
    // The scroll offset after clamping to `[0, max_scroll]`.
    pub scroll: f32,
    // Scrollbar thumb size as a fraction of the track, in `(0, 1]`. 1.0 means
    // the content fits and no scrolling is possible (the bar can be hidden).
    pub thumb_frac: f32,
    // Scrollbar thumb top as a fraction of the track, in `[0, 1 - thumb_frac]`.
    pub thumb_offset_frac: f32,
}

impl Solved {
    // Whether the content overflows the band (i.e. scrolling does anything and
    // the scrollbar is meaningful).
    pub fn scrollable(&self) -> bool {
        self.thumb_frac < 1.0
    }

    // The largest in-range scroll offset for this solution.
    pub fn max_scroll(&self) -> f32 {
        (self.content_height - self.band_height_from_thumb()).max(0.0)
    }

    // Recover the band height from the thumb fraction (band = content *
    // thumb_frac). Used only by `max_scroll`; exact for thumb_frac > 0.
    fn band_height_from_thumb(&self) -> f32 {
        if self.thumb_frac <= 0.0 {
            return self.content_height;
        }
        self.content_height * self.thumb_frac
    }
}

// Solve the vertical layout. `rows` are in top-to-bottom order with their
// build-time (all-expanded, unscrolled) positions implied by the running sum of
// heights. `collapsed[g]` hides every row whose `group == Some(g)`. `band_height`
// is the visible window; `scroll` is the requested offset (clamped here).
pub fn solve(rows: &[RowSpec], collapsed: &[bool], band_height: f32, scroll: f32) -> Solved {
    let is_hidden = |r: &RowSpec| {
        r.group
            .is_some_and(|g| collapsed.get(g).copied().unwrap_or(false))
    };

    // The build laid every row out as if all groups were expanded; recover each
    // row's build-time top from that all-expanded running sum so `dy` is the
    // delta the client adds to the authored position.
    let mut base_top = 0.0_f32;
    let mut base_tops = Vec::with_capacity(rows.len());
    for r in rows {
        base_tops.push(base_top);
        base_top += r.height;
    }

    // Total content height with the current collapsed state.
    let content_height: f32 = rows
        .iter()
        .filter(|r| !is_hidden(r))
        .map(|r| r.height)
        .sum();

    let max_scroll = (content_height - band_height).max(0.0);
    let scroll = scroll.clamp(0.0, max_scroll);

    // Walk the visible rows, accumulating the laid-out top, and emit each row's
    // delta from its build-time top minus the scroll.
    let mut laid_top = 0.0_f32;
    let mut placements = Vec::with_capacity(rows.len());
    for (i, r) in rows.iter().enumerate() {
        if is_hidden(r) {
            placements.push(RowPlacement {
                dy: 0.0,
                visible: false,
            });
            continue;
        }
        let dy = laid_top - base_tops[i] - scroll;
        placements.push(RowPlacement { dy, visible: true });
        laid_top += r.height;
    }

    // Thumb size = band / content (clamped to 1 when content fits); offset =
    // scroll / content. Guard a zero content height (empty panel).
    let (thumb_frac, thumb_offset_frac) = if content_height <= 0.0 {
        (1.0, 0.0)
    } else {
        let frac = (band_height / content_height).min(1.0);
        let offset = (scroll / content_height).clamp(0.0, 1.0 - frac);
        (frac, offset)
    };

    Solved {
        rows: placements,
        content_height,
        scroll,
        thumb_frac,
        thumb_offset_frac,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(height: f32, group: Option<usize>) -> RowSpec {
        RowSpec { height, group }
    }

    #[test]
    fn fits_in_band_no_scroll_no_offset() {
        // Three 50px rows = 150px content in a 300px band: nothing moves, the
        // thumb fills the track, and scrolling is a no-op.
        let rows = [row(50.0, None), row(50.0, None), row(50.0, None)];
        let s = solve(&rows, &[], 300.0, 0.0);
        assert_eq!(s.content_height, 150.0);
        assert!(!s.scrollable());
        assert_eq!(s.thumb_frac, 1.0);
        assert_eq!(s.scroll, 0.0);
        for p in &s.rows {
            assert!(p.visible);
            assert_eq!(p.dy, 0.0);
        }
    }

    #[test]
    fn overflow_clamps_scroll_and_shifts_rows_up() {
        // Ten 50px rows = 500px content in a 200px band: max scroll = 300.
        let rows: Vec<RowSpec> = (0..10).map(|_| row(50.0, None)).collect();
        let s = solve(&rows, &[], 200.0, 1000.0);
        assert_eq!(s.content_height, 500.0);
        assert_eq!(s.scroll, 300.0, "scroll clamps to content - band");
        assert!(s.scrollable());
        // Every visible row shifts up by the scroll amount.
        for p in &s.rows {
            assert!(p.visible);
            assert_eq!(p.dy, -300.0);
        }
        // Thumb = 200/500 = 0.4, offset = 300/500 = 0.6 (pinned to the bottom).
        assert!((s.thumb_frac - 0.4).abs() < 1e-5);
        assert!((s.thumb_offset_frac - 0.6).abs() < 1e-5);
    }

    #[test]
    fn collapsed_group_hides_body_and_pulls_rows_below_up() {
        // Layout: header(None), body0(g0), body1(g0), footer(None). All 40px.
        let rows = [
            row(40.0, None),
            row(40.0, Some(0)),
            row(40.0, Some(0)),
            row(40.0, None),
        ];
        // Expanded: nothing hidden, nothing moves.
        let open = solve(&rows, &[false], 1000.0, 0.0);
        assert_eq!(open.content_height, 160.0);
        for p in &open.rows {
            assert!(p.visible);
            assert_eq!(p.dy, 0.0);
        }
        // Collapsed: the two body rows hide and the footer rises by 80px (their
        // combined height); the header is unmoved.
        let shut = solve(&rows, &[true], 1000.0, 0.0);
        assert_eq!(shut.content_height, 80.0);
        assert!(shut.rows[0].visible && shut.rows[0].dy == 0.0); // header
        assert!(!shut.rows[1].visible); // body0
        assert!(!shut.rows[2].visible); // body1
        assert!(shut.rows[3].visible);
        assert_eq!(
            shut.rows[3].dy, -80.0,
            "footer rises by the collapsed height"
        );
    }

    #[test]
    fn collapse_then_scroll_compose() {
        // header + 6 body rows (g0) + footer, all 50px = 400px content.
        let mut rows = vec![row(50.0, None)];
        rows.extend((0..6).map(|_| row(50.0, Some(0))));
        rows.push(row(50.0, None));
        // Collapsed body: content = header + footer = 100px, fits a 200px band,
        // so no scroll regardless of the requested offset.
        let s = solve(&rows, &[true], 200.0, 500.0);
        assert_eq!(s.content_height, 100.0);
        assert_eq!(s.scroll, 0.0);
        assert!(!s.scrollable());
        // The footer rises by the 6 collapsed bodies (300px), unscrolled.
        assert_eq!(s.rows.last().unwrap().dy, -300.0);
    }

    #[test]
    fn empty_panel_is_stable() {
        let s = solve(&[], &[], 200.0, 10.0);
        assert_eq!(s.content_height, 0.0);
        assert_eq!(s.scroll, 0.0);
        assert_eq!(s.thumb_frac, 1.0);
        assert!(s.rows.is_empty());
    }

    #[test]
    fn out_of_range_group_index_treated_as_expanded() {
        // A row referencing a group with no collapsed entry stays visible.
        let rows = [row(50.0, Some(5))];
        let s = solve(&rows, &[false], 200.0, 0.0);
        assert!(s.rows[0].visible);
    }
}
