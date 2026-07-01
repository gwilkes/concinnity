// src/assets/debug_hud.rs
//
// DebugHud component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::debug_hud`.

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// Requests the developer debug HUD: a set of [TextLabel](#textlabel) chips
/// with diagnostic readouts, anchored to the top-right of the window and
/// toggled with F1 (hidden by default).
///
/// Each label field, when set, receives one chip: `passes_label` a multi-line
/// list of the heaviest rendering steps of the last frame, `mouse_label` the
/// cursor position in window pixels, and `camera_label` the live camera pose
/// (position, yaw, pitch) in the exact form a fixed viewpoint is reproduced
/// with. Chips whose stat is unavailable stay blank. The chips stack
/// vertically from the top-right corner in the order cursor, then camera, then
/// passes (passes is last because its height varies with the frame's step
/// count), so their on-screen position is fixed by the engine rather than the
/// authored coordinates.
///
/// The always-on frame-rate and GPU-memory readouts live on the separate
/// [StatHud](#stathud).
///
/// ```jsonl
/// {"type":"Font","name":"hud_font","args":{"size_px":20}}
/// {"type":"TextLabel","name":"mouse_chip","args":{"font":"hud_font","scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"passes_chip","args":{"font":"hud_font","scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"camera_chip","args":{"font":"hud_font","scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"DebugHud","name":"debug_hud","args":{"passes_label":"passes_chip","mouse_label":"mouse_chip","camera_label":"camera_chip"}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DebugHud {
    /// [TextLabel](#textlabel) that receives the per-step GPU-timing chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub passes_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the cursor-position chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub mouse_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the live camera-pose chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub camera_label: Option<AssetId>,
}

impl Component for DebugHud {
    const NAME: &'static str = "DebugHud";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
