// src/assets/text_label.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Screen-space text drawn as a UI overlay on top of the 3D scene each frame.
///
/// Text is laid out using the referenced [Font](#font). The `content` field can
/// be updated every frame (e.g. by an [FpsCounter](#fpscounter)).
///
/// A `\n` in `content` starts a new line. When `background` has an alpha > 0, a
/// box is filled behind the glyphs, extended outward by `padding` pixels,
/// useful for HUD chips.
///
/// ```jsonl
/// {
///   "type": "TextLabel",
///   "name": "fps_text",
///   "args": {
///     "font": "fps_font",
///     "content": "FPS: --",
///     "x": 10,
///     "y": 10,
///     "color": [
///       1,
///       1,
///       1
///     ],
///     "scale": 1
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TextLabel {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [Font](#font) asset to use for rendering.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub font: Option<AssetId>,
    /// Text to display. Can be updated each frame.
    pub content: String,
    /// Horizontal position in pixels from the left edge of the window.
    pub x: f32,
    /// Vertical position in pixels from the top edge of the window.
    pub y: f32,
    /// Linear-space RGB text colour.
    pub color: [f32; 3],
    /// Uniform scale applied on top of the font's `size_px`. 1.0 = native size.
    pub scale: f32,
    /// When true, center the label in the viewport each frame; x and y are ignored.
    pub centered: bool,
    /// RGBA fill of a box drawn behind the text. An alpha of 0 (the default)
    /// draws no box; any alpha > 0 draws the box at that opacity.
    pub background: [f32; 4],
    /// Pixels the background box extends past the text on every side. Only
    /// meaningful when `background` is visible.
    pub padding: f32,
    /// When false, the label is hidden.
    pub visible: bool,
    /// [View](#view) this label belongs to. Resolved automatically from the
    /// naming convention (`<view>_*`); you don't set this directly. `None` means
    /// the label is always visible.
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub view: Option<AssetId>,
}

impl Default for TextLabel {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            font: None,
            content: String::new(),
            x: 10.0,
            y: 10.0,
            color: [1.0, 1.0, 1.0],
            scale: 1.0,
            centered: false,
            background: [0.0, 0.0, 0.0, 0.0],
            padding: 0.0,
            visible: true,
            view: None,
        }
    }
}

impl Component for TextLabel {
    const NAME: &'static str = "TextLabel";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}
