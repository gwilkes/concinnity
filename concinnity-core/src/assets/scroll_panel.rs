// src/assets/scroll_panel.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// Runtime model that makes a band of UI rows scrollable and (optionally)
/// collapsible.
///
/// A `ScrollPanel` is emitted by the build (e.g. by a settings menu) and read
/// by the UI at runtime; it is not hand-authored. It names a content band (a
/// fixed rectangle in the menu's reference canvas), the ordered rows that live
/// inside it, the collapsible groups some rows belong to, and the scrollbar
/// thumb/track sprites. The UI lays the rows out each frame: a collapsed group's
/// body rows hide and the rows below them move up; when the visible stack is
/// taller than the band it scrolls (mouse wheel or thumb drag) and rows outside
/// the band are clipped.
///
/// All pixel fields are in the same reference-space coordinates as the View's
/// other UI (see the overlay scaling notes on [MainMenu](#mainmenu)).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ScrollPanel {
    /// [View](#view) this panel belongs to. Resolved automatically from the
    /// `<view>_*` naming convention; you don't set this directly. The panel is
    /// only live while its view is active.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub view: Option<AssetId>,
    /// Left edge of the content band in reference pixels.
    pub x: f32,
    /// Top edge of the content band in reference pixels.
    pub y: f32,
    /// Width of the content band in reference pixels.
    pub width: f32,
    /// Height of the content band (the visible window) in reference pixels.
    pub height: f32,
    /// The rows in the band, top to bottom.
    pub rows: Vec<ScrollRow>,
    /// Collapsible groups, referenced by index from [ScrollRow::group].
    pub groups: Vec<ScrollGroup>,
    /// Scrollbar thumb [Sprite](#sprite) the UI moves and resizes. `None` for a
    /// panel with no scrollbar.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub thumb: Option<AssetId>,
    /// Scrollbar track [Sprite](#sprite). Hidden along with the thumb when the
    /// content fits the band.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub track: Option<AssetId>,
    /// Left edge of the scrollbar track in reference pixels.
    pub track_x: f32,
    /// Top edge of the scrollbar track in reference pixels.
    pub track_y: f32,
    /// Width of the scrollbar track in reference pixels.
    pub track_w: f32,
    /// Height of the scrollbar track in reference pixels (the thumb travels
    /// within it).
    pub track_h: f32,
}

/// One row inside a [ScrollPanel](#scrollpanel): the elements that move
/// together, the row's height, and the collapsible group it belongs to.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ScrollRow {
    /// The [Sprite](#sprite)/[TextLabel](#textlabel) ids that make up this row
    /// and move (and clip) together. Click regions are matched to their row by
    /// position, so they are not listed here.
    pub elements: Vec<AssetId>,
    /// The row's authored top edge in reference pixels (its build-time, all
    /// groups expanded, unscrolled position).
    pub base_y: f32,
    /// The row's height in reference pixels (its vertical pitch in the stack).
    pub height: f32,
    /// Index into [ScrollPanel::groups] of the group whose collapsed state
    /// hides this row, or `-1` for a row that is always shown (a group header
    /// or an ungrouped row).
    pub group: i32,
}

impl Default for ScrollRow {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
            base_y: 0.0,
            height: 0.0,
            group: -1,
        }
    }
}

/// A collapsible group of rows inside a [ScrollPanel](#scrollpanel).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ScrollGroup {
    /// Whether the group starts collapsed (its body rows hidden).
    pub collapsed: bool,
    /// The header [TextLabel](#textlabel) whose text gets a `+`/`-` prefix to
    /// reflect the collapsed state. `None` leaves the header text unchanged.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub header: Option<AssetId>,
    /// The header's base title (e.g. `"Advanced"`); the UI shows `"+ Advanced"`
    /// when collapsed and `"- Advanced"` when expanded.
    pub title: String,
}

impl Component for ScrollPanel {
    const NAME: &'static str = "ScrollPanel";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::asset_id::{intern_all, reset_interner};

    #[test]
    fn bare_args_deserialize_with_defaults() {
        let p: ScrollPanel = serde_json::from_str("{}").unwrap();
        assert!(p.view.is_none());
        assert!(p.rows.is_empty());
        assert!(p.groups.is_empty());
        assert!(p.thumb.is_none());
    }

    #[test]
    fn rows_and_refs_resolve_from_names() {
        reset_interner();
        // Declaration order assigns ids: row0a=0, row0b=1, hdr=2, body=3,
        // thumb=4, track=5, view=6.
        intern_all(&["row0a", "row0b", "hdr", "body", "thumb", "track", "menu"]);
        let json = r#"{
            "view": "menu",
            "x": 10, "y": 20, "width": 300, "height": 200,
            "rows": [
                {"elements": ["row0a", "row0b"], "base_y": 20, "height": 40, "group": -1},
                {"elements": ["body"], "base_y": 60, "height": 40, "group": 0}
            ],
            "groups": [{"collapsed": true, "header": "hdr", "title": "Advanced"}],
            "thumb": "thumb", "track": "track",
            "track_x": 305, "track_y": 20, "track_w": 6, "track_h": 200
        }"#;
        let p: ScrollPanel = serde_json::from_str(json).unwrap();
        assert_eq!(p.view, Some(AssetId(6)));
        assert_eq!(p.rows.len(), 2);
        assert_eq!(p.rows[0].elements, vec![AssetId(0), AssetId(1)]);
        assert_eq!(p.rows[0].group, -1);
        assert_eq!(p.rows[1].elements, vec![AssetId(3)]);
        assert_eq!(p.rows[1].group, 0);
        assert_eq!(p.groups.len(), 1);
        assert!(p.groups[0].collapsed);
        assert_eq!(p.groups[0].header, Some(AssetId(2)));
        assert_eq!(p.groups[0].title, "Advanced");
        assert_eq!(p.thumb, Some(AssetId(4)));
        assert_eq!(p.track, Some(AssetId(5)));
    }

    #[test]
    fn round_trips_through_serde() {
        let p = ScrollPanel {
            view: Some(AssetId(2)),
            x: 1.0,
            y: 2.0,
            width: 3.0,
            height: 4.0,
            rows: vec![ScrollRow {
                elements: vec![AssetId(5)],
                base_y: 2.0,
                height: 40.0,
                group: 0,
            }],
            groups: vec![ScrollGroup {
                collapsed: false,
                header: Some(AssetId(7)),
                title: "Advanced".to_string(),
            }],
            thumb: Some(AssetId(8)),
            track: Some(AssetId(9)),
            track_x: 5.0,
            track_y: 6.0,
            track_w: 7.0,
            track_h: 8.0,
        };
        let v = serde_json::to_value(&p).unwrap();
        let back: ScrollPanel = serde_json::from_value(v).unwrap();
        assert_eq!(back.rows.len(), 1);
        assert_eq!(back.rows[0].elements, vec![AssetId(5)]);
        assert_eq!(back.groups[0].title, "Advanced");
    }
}
