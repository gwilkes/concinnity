// src/gfx/dropdown.rs
//
// Reference-space layout for a settings dropdown's floating option list, shared
// by the client's input hit-test and its renderer so the two agree on where
// each option sits. A dropdown row's control button is the anchor; the list is
// a stack of equal-height option rows placed directly below it, or flipped
// above when it would spill past the bottom of the reference canvas. Purely
// geometric: no colors, fonts, or draw state.

use crate::gfx::overlay::UI_REFERENCE_SIZE;

// The placed list rectangle and the per-option rectangles inside it, all in
// reference-space `[x, y, width, height]`. `items` has one entry per option, in
// order, each the full-width row a click / hover tests against.
#[derive(Debug, Clone, PartialEq)]
pub struct DropdownLayout {
    pub list: [f32; 4],
    pub items: Vec<[f32; 4]>,
}

// Lay out a `count`-option list anchored to a control button `anchor`
// (`[x, y, width, height]`, reference space). Each option row is the anchor's
// height; the list matches the anchor's x + width. It opens downward from just
// below the anchor, flipping to open upward when opening down would overflow the
// reference canvas bottom, and clamps onto the canvas if it is taller than the
// space either way. A zero `count` yields an empty list.
pub fn layout(anchor: [f32; 4], count: usize) -> DropdownLayout {
    let [ax, ay, aw, ah] = anchor;
    let item_h = ah;
    let list_h = item_h * count as f32;
    let ref_h = UI_REFERENCE_SIZE[1];

    // Prefer opening downward; flip up if that overflows the bottom.
    let below_top = ay + ah;
    let above_top = ay - list_h;
    let mut top = if below_top + list_h <= ref_h || above_top < 0.0 {
        below_top
    } else {
        above_top
    };
    // Keep the list on the canvas even in the degenerate "taller than the
    // screen" case (top pinned to 0, the excess simply runs off the bottom).
    let max_top = (ref_h - list_h).max(0.0);
    top = top.clamp(0.0, max_top);

    let items = (0..count)
        .map(|i| [ax, top + i as f32 * item_h, aw, item_h])
        .collect();
    DropdownLayout {
        list: [ax, top, aw, list_h],
        items,
    }
}

// The index of the option row containing reference-space point `(px, py)`, or
// `None` if the point is outside every row.
pub fn item_at(layout: &DropdownLayout, px: f32, py: f32) -> Option<usize> {
    layout
        .items
        .iter()
        .position(|&[x, y, w, h]| px >= x && px < x + w && py >= y && py < y + h)
}

// Whether reference-space point `(px, py)` lies inside the list rectangle.
pub fn contains(layout: &DropdownLayout, px: f32, py: f32) -> bool {
    let [x, y, w, h] = layout.list;
    px >= x && px < x + w && py >= y && py < y + h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opens_downward_with_room_below() {
        // A row near the top: the 3-item list sits directly below it.
        let anchor = [400.0, 100.0, 200.0, 40.0];
        let l = layout(anchor, 3);
        assert_eq!(l.list, [400.0, 140.0, 200.0, 120.0]);
        assert_eq!(l.items.len(), 3);
        assert_eq!(l.items[0], [400.0, 140.0, 200.0, 40.0]);
        assert_eq!(l.items[1], [400.0, 180.0, 200.0, 40.0]);
        assert_eq!(l.items[2], [400.0, 220.0, 200.0, 40.0]);
    }

    #[test]
    fn flips_upward_when_it_would_overflow_the_bottom() {
        // A row low on the canvas (720 tall): opening down would spill past the
        // bottom, so the list opens upward, ending at the anchor's top.
        let anchor = [400.0, 680.0, 200.0, 40.0];
        let l = layout(anchor, 4); // list_h = 160
        // above_top = 680 - 160 = 520; below_top + 160 = 720+160 overflow.
        assert_eq!(l.list, [400.0, 520.0, 200.0, 160.0]);
        assert_eq!(l.items[0], [400.0, 520.0, 200.0, 40.0]);
        assert_eq!(l.items[3], [400.0, 640.0, 200.0, 40.0]);
    }

    #[test]
    fn clamps_onto_canvas_when_taller_than_the_screen() {
        // An absurd option count taller than the canvas pins the top to 0.
        let anchor = [0.0, 300.0, 100.0, 40.0];
        let l = layout(anchor, 40); // list_h = 1600 > 720
        assert_eq!(l.list[1], 0.0);
        assert_eq!(l.items[0][1], 0.0);
    }

    #[test]
    fn item_at_finds_the_row_under_a_point() {
        let l = layout([400.0, 100.0, 200.0, 40.0], 3);
        assert_eq!(item_at(&l, 500.0, 150.0), Some(0));
        assert_eq!(item_at(&l, 500.0, 190.0), Some(1));
        assert_eq!(item_at(&l, 500.0, 230.0), Some(2));
        // Outside the list horizontally / below it: no hit.
        assert_eq!(item_at(&l, 700.0, 150.0), None);
        assert_eq!(item_at(&l, 500.0, 300.0), None);
    }

    #[test]
    fn contains_matches_the_list_bounds() {
        let l = layout([400.0, 100.0, 200.0, 40.0], 3);
        assert!(contains(&l, 400.0, 140.0));
        assert!(contains(&l, 599.0, 259.0));
        assert!(!contains(&l, 399.0, 140.0));
        assert!(!contains(&l, 500.0, 260.0));
    }

    #[test]
    fn zero_options_is_empty() {
        let l = layout([0.0, 0.0, 100.0, 40.0], 0);
        assert!(l.items.is_empty());
        assert_eq!(l.list[3], 0.0);
    }
}
