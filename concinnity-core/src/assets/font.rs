// src/assets/font.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// Rasterises a TrueType font into a glyph atlas at build time.
///
/// Reference a Font by name from a [TextLabel](#textlabel).
///
/// ```jsonl
/// {
///   "type": "Font",
///   "name": "fps_font",
///   "args": {
///     "path": "assets/fonts/JetBrainsMono-Regular.ttf",
///     "size_px": 20
///   }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Font {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the TTF file, relative to the project root.
    pub path: String,
    /// Rasterisation size in pixels. Determines the rendered glyph height.
    pub size_px: u32,
    /// Filled by inject_locator after the build step packs the payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for Font {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            path: String::new(),
            size_px: 20,
            locator: None,
        }
    }
}

impl Component for Font {
    const NAME: &'static str = "Font";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::build::SourceBacked for Font {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("path")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
