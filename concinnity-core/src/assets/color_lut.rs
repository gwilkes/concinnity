// src/assets/color_lut.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// A 3D colour-grading lookup table applied as a final post-process step. The
/// build bakes the source into a colour cube; the graded result is blended over
/// the image by [PostProcessConfig](#postprocessconfig)'s `lut_strength`.
///
/// A world declares at most one `ColorLut`; the first wins. When none is
/// present, colour grading is skipped regardless of `lut_strength`.
///
/// Two source formats are accepted, picked by file extension:
///   - `.cube`  Adobe Cube LUT (plain-text interchange format).
///   - `.png`   A horizontal slice strip: `(n*n)` wide by `n` tall.
///
/// ```jsonl
/// {"name":"grade","type":"ColorLut","args":{"source":"luts/cinematic_warm.cube"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct ColorLut {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source `.cube` or `.png` LUT file.
    pub source: String,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Component for ColorLut {
    const NAME: &'static str = "ColorLut";

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

impl crate::build::SourceBacked for ColorLut {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
