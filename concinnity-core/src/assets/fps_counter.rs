// src/assets/fps_counter.rs
//
// FpsCounter component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::fps_counter`.

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// Requests a frames-per-second counter; optionally writes it to a
/// [TextLabel](#textlabel).
///
/// Declaring an `FpsCounter` updates the named [TextLabel](#textlabel) with the
/// current rate once per second. Omit `label` to suppress on-screen display.
///
/// To display an FPS overlay, declare a [Font](#font), a
/// [TextLabel](#textlabel), and an `FpsCounter` that references the label:
///
/// ```jsonl
/// {"$include":"assets/fps_font.json"}
/// {"$include":"assets/fps_text.json"}
/// {"type":"FpsCounter","name":"fps","args":{"label":"fps_text"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct FpsCounter {
    /// A [TextLabel](#textlabel) to update with the current FPS each second.
    /// Leave unset to suppress on-screen display.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub label: Option<AssetId>,
}

impl Component for FpsCounter {
    const NAME: &'static str = "FpsCounter";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
