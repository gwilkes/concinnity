// src/assets/window.rs

use crate::ecs::{AssetOrigin, Component};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub enum WindowMode {
    #[default]
    Windowed,
    Fullscreen,
    Borderless,
}

/// Declares the application window.
///
/// ```json
/// {
///   "name": "main_window",
///   "type": "Window",
///   "args": {
///     "title": "Game",
///     "width": 1280,
///     "height": 720,
///     "mode": "windowed",
///     "resizable": true
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Window {
    /// Window title shown in the title bar.
    pub title: String,
    /// Initial window width in pixels.
    pub width: u32,
    /// Initial window height in pixels.
    pub height: u32,
    /// How the window is displayed.
    pub mode: WindowMode,
    /// Whether the user can resize the window.
    pub resizable: bool,
}

impl Default for Window {
    fn default() -> Self {
        Self {
            title: "Concinnity".to_string(),
            width: 1024,
            height: 768,
            mode: WindowMode::Windowed,
            resizable: false,
        }
    }
}

/// Back-compat alias: external callers (`GraphicsSystem`, etc.) previously
/// stored a `WindowArgs`. The runtime and args structs are now one type.
pub type WindowArgs = Window;

impl Component for Window {
    const NAME: &'static str = "Window";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
