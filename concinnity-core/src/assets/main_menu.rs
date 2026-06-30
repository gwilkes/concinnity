// src/assets/main_menu.rs

use crate::ecs::{AssetOrigin, Component};

/// A ready-made menu declared in a single line.
///
/// `MainMenu` is a build-time shorthand. It expands into the assets a menu is
/// built from: a [View](#view) layer, a dim backdrop [Sprite](#sprite), a
/// [TextLabel](#textlabel) and [HitRegion](#hitregion) for each item, an
/// optional [KeyBinding](#keybinding) that toggles the menu, and an optional
/// in-engine mouse cursor [Sprite](#sprite). So `world.jsonl` stays small.
///
/// The bare form gives a centered Return / Settings / Quit menu shown on load:
///
/// ```jsonl
/// {"name":"main_menu","type":"MainMenu"}
/// ```
///
/// **Items.** Each item has a `label` (the text) and an `action` fired on
/// click. `action` takes the same vocabulary as [HitRegion](#hitregion)
/// (`"scene:<name>"`, `"quit"`, `"view:show:<name>"`, `"view:hide"`,
/// `"view:toggle:<name>"`) plus two conveniences resolved against this menu:
/// - `"return"`: hide this menu (the same as `"view:hide"`).
/// - `"settings"`: open a generated settings sub-menu that has a Back button.
///
/// ```jsonl
/// {"name":"title","type":"MainMenu","args":{"items":[
///   {"label":"New Game","action":"scene:level_1"},
///   {"label":"Quit","action":"quit"}
/// ]}}
/// ```
///
/// **Generated names** are prefixed with the menu's `name` (`<name>_btn_0`,
/// `<name>_label_0`, `<name>_cursor`, ...), so they never clash with
/// hand-authored assets and you never reference them by hand.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MainMenu {
    /// Menu entries, top to bottom. Each one is a clickable button.
    pub items: Vec<MainMenuItem>,
    /// Optional heading drawn above the items. Empty draws no heading.
    pub title: String,
    /// Show the menu as soon as the world loads.
    pub initial: bool,
    /// Key that toggles the menu while the cursor is free. Empty binds no key.
    /// Only `"Escape"` is currently recognised by the runtime.
    pub toggle_key: String,
    /// RGBA fill drawn across the whole window behind the items. Defaults to
    /// opaque black: a fully opaque alpha (1.0) hides the scene completely, which
    /// lets the renderer skip the entire world render while the menu is open, so
    /// the frame costs only the menu overlay. Lower the alpha to keep the world
    /// visible behind a translucent fade (the world then keeps rendering); an
    /// alpha of 0 draws no backdrop at all.
    pub dim: [f32; 4],
    /// Horizontally center the menu and align it to the top of the window.
    /// When false, `x` is the column's center and `y` is the top of the first
    /// item.
    ///
    /// The menu is a screen overlay laid out against a fixed reference
    /// resolution and uniformly scaled to fill the window, so it keeps the same
    /// proportions at any window size. All pixel fields below are in that
    /// reference space, not raw window pixels.
    pub centered: bool,
    /// Column center x in reference-space pixels, used when `centered` is false.
    pub x: f32,
    /// Top of the first item in reference-space pixels, used when `centered` is
    /// false.
    pub y: f32,
    /// Width of each item's clickable region in pixels.
    pub button_width: f32,
    /// Height of each item's clickable region in pixels.
    pub button_height: f32,
    /// Pixels between adjacent items.
    pub row_gap: f32,
    /// [Font](#font) for the item text. Empty uses the built-in font.
    pub font: String,
    /// Pixel size of the item text when this menu emits its own built-in font
    /// (that is, when `font` is empty). Ignored when `font` names a
    /// [Font](#font), which carries its own size. In reference-space pixels.
    pub font_px: f32,
    /// Linear-space RGB color of the item text.
    pub text_color: [f32; 3],
    /// Scale applied to the item text.
    pub text_scale: f32,
    /// RGB color of an item's text while it is hovered.
    pub hover_color: [f32; 3],
    /// Multiplier applied to an item's text size while it is hovered. The
    /// default `1.0` keeps the size and position fixed, so only the color
    /// changes on hover; a value like `1.1` grows the hovered text by 10%.
    pub hover_scale: f32,
    /// Draw an in-engine arrow cursor while the menu is shown (the system
    /// cursor is hidden). When false the system cursor is used.
    pub cursor: bool,
    /// RGBA fill color of the arrow cursor. A contrasting outline is added
    /// automatically so it stays legible over any scene.
    pub cursor_color: [f32; 4],
    /// Arrow cursor height in pixels (its width follows the arrow's shape).
    pub cursor_size: f32,
}

/// One entry in a [MainMenu](#mainmenu).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MainMenuItem {
    /// Button text.
    pub label: String,
    /// Action fired on click. See [MainMenu](#mainmenu) for the vocabulary.
    pub action: String,
}

impl Default for MainMenu {
    fn default() -> Self {
        Self {
            items: vec![
                MainMenuItem {
                    label: "Return".to_string(),
                    action: "return".to_string(),
                },
                MainMenuItem {
                    label: "Settings".to_string(),
                    action: "settings".to_string(),
                },
                MainMenuItem {
                    label: "Quit".to_string(),
                    action: "quit".to_string(),
                },
            ],
            title: String::new(),
            initial: true,
            toggle_key: "Escape".to_string(),
            dim: [0.0, 0.0, 0.0, 1.0],
            centered: true,
            x: 640.0,
            y: 300.0,
            button_width: 360.0,
            button_height: 60.0,
            row_gap: 14.0,
            font: String::new(),
            font_px: 48.0,
            text_color: [0.85, 0.85, 0.85],
            text_scale: 1.1,
            hover_color: [1.0, 0.85, 0.3],
            hover_scale: 1.0,
            cursor: true,
            cursor_color: [1.0, 1.0, 1.0, 1.0],
            cursor_size: 22.0,
        }
    }
}

impl Component for MainMenu {
    const NAME: &'static str = "MainMenu";
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
    fn bare_args_default_to_return_settings_quit() {
        let m: MainMenu = serde_json::from_str("{}").unwrap();
        let labels: Vec<&str> = m.items.iter().map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["Return", "Settings", "Quit"]);
        assert!(m.initial);
        assert_eq!(m.toggle_key, "Escape");
        assert!(m.cursor);
    }

    #[test]
    fn explicit_items_replace_the_default() {
        let json = r#"{"items":[{"label":"Play","action":"scene:level_1"}]}"#;
        let m: MainMenu = serde_json::from_str(json).unwrap();
        assert_eq!(m.items.len(), 1);
        assert_eq!(m.items[0].label, "Play");
        assert_eq!(m.items[0].action, "scene:level_1");
    }
}
