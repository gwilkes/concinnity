// src/assets/option_select.rs

use crate::ecs::{AssetOrigin, Component};

/// A settings row that cycles through a fixed set of values on click.
///
/// `OptionSelect` is a build-time shorthand for one row of a settings menu: a
/// left-aligned name, a right-aligned current value, and a clickable region
/// that advances the value. It expands into a [TextLabel](#textlabel) for the
/// name, a `TextLabel` for the value, and a [HitRegion](#hitregion) that fires a
/// `"setting:<setting>:next"` action.
///
/// The `setting` field names an engine setting the runtime knows how to read,
/// cycle, and apply (e.g. `"vsync"`); its option list lives in the engine, not
/// here. The value label shows a placeholder at build time and is corrected to
/// the live value when the world starts.
///
/// ```jsonl
/// {"name":"opt_vsync","type":"OptionSelect","args":{"setting":"vsync","label":"Vsync"}}
/// ```
///
/// Generated names are prefixed with this asset's `name` (`<name>_label`,
/// `<name>_value`, `<name>_btn`), so they never clash with hand-authored assets.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OptionSelect {
    /// Engine setting this row controls (e.g. `"vsync"`). Must be a setting the
    /// runtime recognises; an unknown key renders but does nothing on click.
    pub setting: String,
    /// Display name shown at the left of the row.
    pub label: String,
    /// Left edge of the row in window pixels.
    pub x: f32,
    /// Top edge of the row in window pixels.
    pub y: f32,
    /// Row width in window pixels (name sits at the left, value at the right).
    pub width: f32,
    /// Row height in window pixels (the clickable region's height).
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
    /// RGB color of the value text while the row is hovered.
    pub hover_color: [f32; 3],
    /// Scale of the value text while the row is hovered.
    pub hover_scale: f32,
    /// Width in pixels of the `<` previous-value click region. The `>`
    /// next-value region spans the rest of the row's right half (the value
    /// sits inside it), so a click on the value advances to the next option.
    pub stepper_width: f32,
}

impl Default for OptionSelect {
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
            hover_color: [1.0, 0.85, 0.3],
            hover_scale: 1.08,
            stepper_width: 40.0,
        }
    }
}

impl Component for OptionSelect {
    const NAME: &'static str = "OptionSelect";
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
        let o: OptionSelect = serde_json::from_str("{}").unwrap();
        assert!(o.setting.is_empty());
        assert_eq!(o.width, 360.0);
        assert_eq!(o.text_scale, 1.0);
    }

    #[test]
    fn explicit_setting_and_label_round_trip() {
        let json = r#"{"setting":"vsync","label":"Vsync"}"#;
        let o: OptionSelect = serde_json::from_str(json).unwrap();
        assert_eq!(o.setting, "vsync");
        assert_eq!(o.label, "Vsync");
        let back = serde_json::to_value(&o).unwrap();
        assert_eq!(back["setting"], "vsync");
    }
}
