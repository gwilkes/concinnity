// src/assets/hit_region.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// A responsive invisible rectangular region in screen space.
///
/// When clicked, fires an `action`. When hovered, it optionally restyles a
/// referenced [TextLabel](#textlabel) (colour and/or scale).
///
/// The cursor must be free (not captured for camera control) for events to fire.
///
/// ```jsonl
/// {
///   "name": "btn_start",
///   "type": "HitRegion",
///   "args": {
///     "x": 430, "y": 330, "width": 220, "height": 40,
///     "label": "scene_menu_start",
///     "hover_color": [1.0, 0.85, 0.3],
///     "hover_scale": 1.08,
///     "action": "scene:scene_game"
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct HitRegion {
    /// Left edge of the region in window pixels.
    pub x: f32,
    /// Top edge of the region in window pixels.
    pub y: f32,
    /// Width of the region in window pixels.
    pub width: f32,
    /// Height of the region in window pixels.
    pub height: f32,
    /// A [TextLabel](#textlabel) to style on hover. `None` = no label effect.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub label: Option<AssetId>,
    /// RGB colour applied to the label while hovered. `None` = no change.
    pub hover_color: Option<[f32; 3]>,
    /// Scale applied to the label while hovered. None = no change.
    pub hover_scale: Option<f32>,
    /// Action to fire on click. Recognised forms:
    /// `"scene:<name>"`, `"quit"`, `"view:show:<name>"`, `"view:hide"`,
    /// `"view:toggle:<name>"`.
    pub action: String,
    /// The [Sprite](#sprite) a [Slider](#slider) drag region moves along its
    /// track. `None` for ordinary regions. Set automatically when a `Slider`
    /// expands; you don't set this directly.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub drag_handle: Option<AssetId>,
    /// [View](#view) this region belongs to. Resolved automatically from the
    /// naming convention (a region named `<view>_*` belongs to view `<view>`);
    /// you don't set this directly. While a view is active, only its regions
    /// fire; when no view is active, only view-less regions fire.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub view: Option<AssetId>,
    /// Whether this region is inert. A disabled region never hovers or fires.
    /// Set by the engine at runtime (e.g. a settings row whose feature the GPU
    /// cannot provide is disabled and grayed out); you don't set this directly.
    #[serde(default)]
    pub disabled: bool,
}

impl Default for HitRegion {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 40.0,
            label: None,
            hover_color: None,
            hover_scale: None,
            action: String::new(),
            drag_handle: None,
            view: None,
            disabled: false,
        }
    }
}

impl Component for HitRegion {
    const NAME: &'static str = "HitRegion";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
