// src/assets/slider.rs

use crate::ecs::{AssetOrigin, Component};

/// A settings row that sets a continuous value by dragging a handle along a
/// track.
///
/// `Slider` is a build-time shorthand for one row of a settings menu: a
/// left-aligned name, a draggable track with a handle, and a right-aligned
/// current value. It expands into a [TextLabel](#textlabel) for the name, a
/// [TextLabel](#textlabel) for the value, two [Sprite](#sprite)s (the track and
/// the handle), and a [HitRegion](#hitregion) covering the track that fires a
/// `"setting:<setting>:drag"` action. While the region is pressed the handle
/// follows the cursor and the value updates live.
///
/// The `setting` field names an engine setting the runtime knows how to map
/// from a fraction, apply, and format (e.g. `"exposure"`); its value range and
/// display format live in the engine, not here. The value label and handle
/// show a placeholder position at build time and are corrected to the live
/// value when the world starts.
///
/// ```jsonl
/// {"name":"sld_exposure","type":"Slider","args":{"setting":"exposure","label":"Exposure"}}
/// ```
///
/// Generated names are prefixed with this asset's `name` (`<name>_label`,
/// `<name>_value`, `<name>_track`, `<name>_handle`, `<name>_drag`), so they
/// never clash with hand-authored assets.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Slider {
    /// Engine setting this row controls (e.g. `"exposure"`). Must be a setting
    /// the runtime recognises as a slider; an unknown key renders but does
    /// nothing on drag.
    pub setting: String,
    /// Display name shown at the left of the row.
    pub label: String,
    /// Left edge of the row in window pixels.
    pub x: f32,
    /// Top edge of the row in window pixels.
    pub y: f32,
    /// Row width in window pixels (name sits at the left, track and value at
    /// the right).
    pub width: f32,
    /// Row height in window pixels (the draggable region's height).
    pub height: f32,
    /// [Font](#font) for the row text. Empty uses the built-in font.
    pub font: String,
    /// Pixel size of the row text when it uses the built-in font (that is, when
    /// `font` is empty). Ignored when `font` names a [Font](#font), which
    /// carries its own size.
    pub font_px: f32,
    /// Linear-space RGB color of the name text.
    pub text_color: [f32; 3],
    /// Linear-space RGB color of the value text.
    pub value_color: [f32; 3],
    /// Scale applied to the row text.
    pub text_scale: f32,
    /// RGBA color of the track bar behind the handle.
    pub track_color: [f32; 4],
    /// RGBA color of the draggable handle.
    pub handle_color: [f32; 4],
}

impl Default for Slider {
    fn default() -> Self {
        Self {
            setting: String::new(),
            label: String::new(),
            x: 0.0,
            y: 0.0,
            width: 360.0,
            height: 48.0,
            font: String::new(),
            font_px: 48.0,
            text_color: [0.85, 0.85, 0.85],
            value_color: [0.85, 0.85, 0.85],
            text_scale: 1.0,
            track_color: [0.28, 0.28, 0.32, 1.0],
            handle_color: [1.0, 0.85, 0.3, 1.0],
        }
    }
}

impl Component for Slider {
    const NAME: &'static str = "Slider";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
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

    #[test]
    fn bare_args_deserialize_with_defaults() {
        let s: Slider = serde_json::from_str("{}").unwrap();
        assert!(s.setting.is_empty());
        assert_eq!(s.width, 360.0);
        assert_eq!(s.text_scale, 1.0);
        assert_eq!(s.handle_color, [1.0, 0.85, 0.3, 1.0]);
    }

    #[test]
    fn explicit_setting_and_label_round_trip() {
        let json = r#"{"setting":"exposure","label":"Exposure"}"#;
        let s: Slider = serde_json::from_str(json).unwrap();
        assert_eq!(s.setting, "exposure");
        assert_eq!(s.label, "Exposure");
        let back = serde_json::to_value(&s).unwrap();
        assert_eq!(back["setting"], "exposure");
    }
}
