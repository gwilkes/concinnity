// src/assets/layout_container.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Horizontal placement of a row's labels within the container's content width
/// (the width of the widest row). Ignored when a row is as wide as the content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Justify {
    /// Pack labels against the left edge (the default).
    #[default]
    Left,
    /// Center the row within the content width.
    Center,
    /// Pack labels against the right edge.
    Right,
    /// Spread the row across the full content width, distributing the slack
    /// evenly between labels. A single-label row falls back to `Left`.
    SpaceBetween,
}

/// One horizontal row of labels inside a `LayoutContainer`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LayoutRow {
    /// The [TextLabel](#textlabel)s in this row, laid out left to right. Their
    /// own `x`/`y` are ignored: the container positions them.
    pub cols: Vec<AssetId>,
    /// How this row is placed within the container's content width.
    pub justify: Justify,
}

impl Default for LayoutRow {
    fn default() -> Self {
        Self {
            cols: Vec::new(),
            justify: Justify::Left,
        }
    }
}

/// Positions a set of [TextLabel](#textlabel)s as a stack of rows, so a HUD does
/// not have to hand-place every chip. Each row lays its labels out left to
/// right; rows stack top to bottom. The container owns the labels' on-screen
/// position: the labels keep their own styling (font, colour, background,
/// padding) but their `x`/`y` are overwritten each frame.
///
/// Sizing is content-driven: a label is measured at its current text, so the
/// layout reflows as live HUD values change width. A row with a single label
/// sits on its own line beneath the previous row, which is how a wide chip
/// (e.g. a multi-pass timing line) ends up spanning the width of the row above.
///
/// Labels referenced by `cols` are matched by name; a label whose font is not
/// loaded, or which is hidden, is skipped and reserves no space.
///
/// ```jsonl
/// {"name":"hud_layout","type":"LayoutContainer","args":{"x":10,"y":10,"col_gap":6,"row_gap":6,"rows":[{"cols":["fps_chip","vram_chip","ev_chip","edr_chip"]},{"cols":["passes_chip"]}]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct LayoutContainer {
    /// Left edge of the container in window pixels.
    pub x: f32,
    /// Top edge of the container in window pixels.
    pub y: f32,
    /// Pixels between adjacent labels in a row, measured between their
    /// background boxes.
    pub col_gap: f32,
    /// Pixels between adjacent rows, measured between their background boxes.
    pub row_gap: f32,
    /// Rows of labels, top to bottom.
    pub rows: Vec<LayoutRow>,
    /// When false, the container leaves its labels where they are instead of
    /// repositioning them.
    pub visible: bool,
}

impl Default for LayoutContainer {
    fn default() -> Self {
        Self {
            x: 10.0,
            y: 10.0,
            col_gap: 6.0,
            row_gap: 6.0,
            rows: Vec::new(),
            visible: true,
        }
    }
}

/// A label's measured extent, used by [`LayoutContainer::layout`] to place it.
/// `w`/`h` are the full background-box size in pixels (the text extent grown by
/// `pad` on every side). `pad` is the horizontal inset from the box's left edge
/// to the text origin. `top_inset` is the vertical inset from the box's top edge
/// down to the text origin (the label's `y`); it is distinct from `pad` because
/// the box hugs the visible glyphs, which sit below the text origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabelBox {
    pub w: f32,
    pub h: f32,
    pub pad: f32,
    pub top_inset: f32,
}

/// The resolved top-left text origin for one label.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub id: AssetId,
    pub x: f32,
    pub y: f32,
}

impl LayoutContainer {
    /// Resolve the text origin for every label in the container.
    ///
    /// `size_of` returns a label's measured box, or `None` to drop it from the
    /// layout (unknown label, unloaded font, hidden). The result is pure
    /// geometry: callers measure with their font metrics, this places the
    /// boxes. Boxes within a row are laid edge-to-edge with `col_gap` between
    /// them; rows are stacked with `row_gap` between them.
    pub fn layout(&self, size_of: impl Fn(AssetId) -> Option<LabelBox>) -> Vec<Placement> {
        // Resolve each row to its measurable cells, dropping unknown labels.
        let rows: Vec<Vec<(AssetId, LabelBox)>> = self
            .rows
            .iter()
            .map(|row| {
                row.cols
                    .iter()
                    .filter_map(|&id| size_of(id).map(|b| (id, b)))
                    .collect()
            })
            .collect();

        let row_width = |cells: &[(AssetId, LabelBox)]| -> f32 {
            if cells.is_empty() {
                return 0.0;
            }
            let sum: f32 = cells.iter().map(|(_, b)| b.w).sum();
            sum + self.col_gap * (cells.len() - 1) as f32
        };

        // Content width is the widest row; narrower rows justify within it.
        let content_w = rows
            .iter()
            .map(|cells| row_width(cells))
            .fold(0.0_f32, f32::max);

        let mut out = Vec::new();
        let mut y_cursor = self.y;
        for (row_idx, cells) in rows.iter().enumerate() {
            if !cells.is_empty() {
                let rw = row_width(cells);
                let slack = (content_w - rw).max(0.0);
                let n = cells.len();
                let (start, gap) = match self.rows[row_idx].justify {
                    Justify::Left => (0.0, self.col_gap),
                    Justify::Right => (slack, self.col_gap),
                    Justify::Center => (slack / 2.0, self.col_gap),
                    Justify::SpaceBetween => {
                        if n > 1 {
                            (0.0, self.col_gap + slack / (n - 1) as f32)
                        } else {
                            (0.0, self.col_gap)
                        }
                    }
                };
                let mut x_cursor = self.x + start;
                for (id, b) in cells {
                    // Box occupies [x_cursor, x_cursor + b.w]; the text origin the
                    // renderer wants is inset from the box's top-left by the
                    // horizontal padding and the (possibly different) vertical
                    // inset, since the box hugs the visible glyphs.
                    out.push(Placement {
                        id: *id,
                        x: x_cursor + b.pad,
                        y: y_cursor + b.top_inset,
                    });
                    x_cursor += b.w + gap;
                }
            }
            let row_h = cells.iter().map(|(_, b)| b.h).fold(0.0_f32, f32::max);
            y_cursor += row_h + self.row_gap;
        }
        out
    }
}

impl Component for LayoutContainer {
    const NAME: &'static str = "LayoutContainer";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        // Positioning runs inside the GraphicsSystem's frame assembly, where the
        // font metrics live; ensure a GraphicsConfig (which gates that system)
        // is present even if a world somehow has a container but no labels.
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::{intern_all, reset_interner};

    // The vertical inset matches the horizontal padding here so the existing
    // placement expectations (text origin = box top-left + pad) hold; the
    // renderer can supply a different `top_inset` when the box hugs the glyphs.
    fn boxed(w: f32, h: f32, pad: f32) -> LabelBox {
        LabelBox {
            w,
            h,
            pad,
            top_inset: pad,
        }
    }

    /// A single left-justified row places boxes edge-to-edge with `col_gap`
    /// between them, and insets each origin by the label's padding.
    #[test]
    fn single_row_left_packs_with_gap() {
        let c = LayoutContainer {
            x: 10.0,
            y: 20.0,
            col_gap: 4.0,
            row_gap: 5.0,
            rows: vec![LayoutRow {
                cols: vec![AssetId(1), AssetId(2)],
                justify: Justify::Left,
            }],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(30.0, 16.0, 2.0)),
            AssetId(2) => Some(boxed(50.0, 16.0, 2.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        assert_eq!(p.len(), 2);
        // First box at container origin; origin inset by its padding.
        assert_eq!(
            p[0],
            Placement {
                id: AssetId(1),
                x: 12.0,
                y: 22.0
            }
        );
        // Second box starts after first box width + col_gap = 10 + 30 + 4 = 44.
        assert_eq!(
            p[1],
            Placement {
                id: AssetId(2),
                x: 46.0,
                y: 22.0
            }
        );
    }

    /// Unknown / unmeasurable labels are dropped and reserve no space.
    #[test]
    fn unknown_labels_are_skipped() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 10.0,
            row_gap: 0.0,
            rows: vec![LayoutRow {
                cols: vec![AssetId(1), AssetId(99), AssetId(2)],
                justify: Justify::Left,
            }],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(20.0, 10.0, 0.0)),
            AssetId(2) => Some(boxed(20.0, 10.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].id, AssetId(1));
        assert_eq!(p[0].x, 0.0);
        // The missing label leaves no gap: second visible box at 0 + 20 + 10.
        assert_eq!(p[1].id, AssetId(2));
        assert_eq!(p[1].x, 30.0);
    }

    /// A second row stacks below the first by the first row's box height plus
    /// the row gap. A lone label on that row starts at the container's left,
    /// occupying the row beneath the wider row above.
    #[test]
    fn second_row_stacks_below_and_spans() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 5.0,
            row_gap: 6.0,
            rows: vec![
                LayoutRow {
                    cols: vec![AssetId(1), AssetId(2)],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![AssetId(3)],
                    justify: Justify::Left,
                },
            ],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(40.0, 18.0, 0.0)),
            AssetId(2) => Some(boxed(40.0, 18.0, 0.0)),
            AssetId(3) => Some(boxed(120.0, 14.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        assert_eq!(p.len(), 3);
        // Row 2 label drops by row 1 box height (18) + row_gap (6) = 24.
        let passes = p.iter().find(|pl| pl.id == AssetId(3)).unwrap();
        assert_eq!(passes.x, 0.0);
        assert_eq!(passes.y, 24.0);
    }

    /// Centering a narrow row offsets it by half the slack to the widest row.
    #[test]
    fn center_justify_offsets_by_half_slack() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 0.0,
            row_gap: 0.0,
            rows: vec![
                LayoutRow {
                    cols: vec![AssetId(1)],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![AssetId(2)],
                    justify: Justify::Center,
                },
            ],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(1) => Some(boxed(100.0, 10.0, 0.0)),
            AssetId(2) => Some(boxed(40.0, 10.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        let narrow = p.iter().find(|pl| pl.id == AssetId(2)).unwrap();
        // slack = 100 - 40 = 60; centered offset = 30.
        assert_eq!(narrow.x, 30.0);
    }

    /// SpaceBetween spreads a short row across the content width, distributing
    /// slack into the gaps between labels.
    #[test]
    fn space_between_distributes_slack_into_gaps() {
        let c = LayoutContainer {
            x: 0.0,
            y: 0.0,
            col_gap: 0.0,
            row_gap: 0.0,
            rows: vec![
                // Widest row sets content width to 200.
                LayoutRow {
                    cols: vec![AssetId(10)],
                    justify: Justify::Left,
                },
                LayoutRow {
                    cols: vec![AssetId(1), AssetId(2), AssetId(3)],
                    justify: Justify::SpaceBetween,
                },
            ],
            visible: true,
        };
        let sizes = |id: AssetId| match id {
            AssetId(10) => Some(boxed(200.0, 10.0, 0.0)),
            AssetId(1) | AssetId(2) | AssetId(3) => Some(boxed(20.0, 10.0, 0.0)),
            _ => None,
        };
        let p = c.layout(sizes);
        let row = |id| p.iter().find(|pl: &&Placement| pl.id == id).unwrap().x;
        // Three 20px boxes in 200px → 140px slack over 2 gaps = 70px each.
        assert_eq!(row(AssetId(1)), 0.0);
        assert_eq!(row(AssetId(2)), 90.0); // 20 + 70
        assert_eq!(row(AssetId(3)), 180.0); // 90 + 20 + 70
    }

    /// Args round-trip through JSON the way the build pipeline reserializes
    /// them: label names intern to ids, justify parses kebab-case, and missing
    /// fields fall back to the defaults.
    #[test]
    fn args_deserialize_from_world_json() {
        reset_interner();
        intern_all(&["fps_chip", "vram_chip", "passes_chip"]);
        let json = r#"{
            "x": 8, "y": 8, "col_gap": 5, "row_gap": 5,
            "rows": [
                {"cols": ["fps_chip", "vram_chip"]},
                {"cols": ["passes_chip"], "justify": "space-between"}
            ]
        }"#;
        let c: LayoutContainer = serde_json::from_str(json).unwrap();
        assert_eq!(c.x, 8.0);
        assert_eq!(c.rows.len(), 2);
        assert_eq!(c.rows[0].cols, vec![AssetId(0), AssetId(1)]);
        assert_eq!(c.rows[0].justify, Justify::Left); // defaulted
        assert_eq!(c.rows[1].cols, vec![AssetId(2)]);
        assert_eq!(c.rows[1].justify, Justify::SpaceBetween);
    }
}
