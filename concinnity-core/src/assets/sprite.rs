// src/assets/sprite.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, CompanionSpec, Component};

/// Screen-space 2D rectangle drawn as a UI overlay each frame.
///
/// Sprites are pixel-anchored quads with an RGBA tint. They draw alongside
/// [TextLabel](#textlabel)s, ordered behind labels so text sits on top.
///
/// Currently only the tint is drawn (solid-coloured rectangles). The `texture`
/// field is reserved for forward compatibility: a sprite with `texture` set
/// renders exactly as if it were unset.
///
/// ```jsonl
/// {
///   "name": "title_menu_bg",
///   "type": "Sprite",
///   "args": {
///     "x": 0, "y": 0, "width": 1280, "height": 720,
///     "tint": [0.04, 0.06, 0.10, 1.0]
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Sprite {
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Left edge in screen pixels from the window's top-left.
    pub x: f32,
    /// Top edge in screen pixels from the window's top-left.
    pub y: f32,
    /// Width in screen pixels.
    pub width: f32,
    /// Height in screen pixels.
    pub height: f32,
    /// [Texture](#texture) to draw (reserved; not yet sampled).
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub texture: Option<AssetId>,
    /// RGBA colour the rectangle is filled with, each channel in [0, 1].
    pub tint: [f32; 4],
    /// When true, the sprite acts as an in-engine cursor: it is drawn on top of
    /// the other overlays as an arrow pointer tracking the mouse, with the
    /// pointer at the arrow's tip. `tint` is the arrow fill (a contrasting
    /// outline is added automatically) and `height` its size; `width` is
    /// ignored so the arrow keeps its shape. The system cursor is hidden while
    /// a visible `follow_cursor` sprite exists.
    pub follow_cursor: bool,
    /// When false the sprite is skipped each frame.
    pub visible: bool,
    /// [View](#view) this sprite belongs to. Resolved automatically from the
    /// naming convention (`<view>_*`); you don't set this directly. `None`
    /// means the sprite is always visible (e.g. a scene background).
    #[serde(default, deserialize_with = "de_opt_asset_ref")]
    pub view: Option<AssetId>,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            texture: None,
            tint: [1.0, 1.0, 1.0, 1.0],
            follow_cursor: false,
            visible: true,
            view: None,
        }
    }
}

impl Component for Sprite {
    const NAME: &'static str = "Sprite";
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_with_all_fields() {
        let json = r#"{
            "x": 10, "y": 20, "width": 300, "height": 200,
            "tint": [0.5, 0.5, 0.5, 0.8], "visible": true
        }"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert_eq!(s.x, 10.0);
        assert_eq!(s.width, 300.0);
        assert_eq!(s.tint, [0.5, 0.5, 0.5, 0.8]);
        assert!(s.visible);
        assert!(s.texture.is_none());
    }

    #[test]
    fn deserializes_with_defaults() {
        let s: Sprite = serde_json::from_str("{}").unwrap();
        assert_eq!(s.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(s.visible);
        assert_eq!(s.width, 100.0);
        assert!(!s.follow_cursor);
    }

    #[test]
    fn follow_cursor_round_trips() {
        let json = r#"{"follow_cursor":true,"width":16,"height":16}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert!(s.follow_cursor);
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["follow_cursor"], true);
    }

    #[test]
    fn deserializes_with_texture_reference() {
        // The interner is global and not reset here; we just check the field
        // is populated when a string name is supplied (it interns lazily).
        let json = r#"{"texture":"tex_intro","tint":[1,1,1,1]}"#;
        let s: Sprite = serde_json::from_str(json).unwrap();
        assert!(s.texture.is_some());
    }
}
