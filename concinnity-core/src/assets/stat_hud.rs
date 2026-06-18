// src/assets/stat_hud.rs
//
// StatHud component (pure data). The runtime behavior that reads it lives in
// the client crate's `hud::stat_hud`.

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// Requests an on-screen performance HUD. Drives a set of
/// [TextLabel](#textlabel) chips with live engine stats, refreshed on a fixed
/// interval and toggled with F1.
///
/// Each label field, when set, receives one chip: `fps_label` the averaged
/// frame rate, `vram_label` the GPU-memory use, `ev_label` the auto-exposure
/// value, `edr_label` the HDR headroom multiplier, `passes_label` a multi-line
/// list of the heaviest rendering steps of the last frame, `mouse_label` the
/// cursor position in window pixels, and `camera_label` the live camera pose
/// (position, yaw, pitch) in the exact form a fixed viewpoint is reproduced
/// with. Chips whose stat is unavailable stay blank.
///
/// ```jsonl
/// {"type":"Font","name":"hud_font","args":{"size_px":20}}
/// {"type":"TextLabel","name":"fps_chip","args":{"font":"hud_font","x":10,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"vram_chip","args":{"font":"hud_font","x":92,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"ev_chip","args":{"font":"hud_font","x":192,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"edr_chip","args":{"font":"hud_font","x":272,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"passes_chip","args":{"font":"hud_font","x":10,"y":36,"scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"mouse_chip","args":{"font":"hud_font","x":352,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"TextLabel","name":"camera_chip","args":{"font":"hud_font","x":352,"y":36,"scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
/// {"type":"StatHud","name":"hud","args":{"fps_label":"fps_chip","vram_label":"vram_chip","ev_label":"ev_chip","edr_label":"edr_chip","passes_label":"passes_chip","mouse_label":"mouse_chip","camera_label":"camera_chip"}}
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct StatHud {
    /// [TextLabel](#textlabel) that receives the frame-rate chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub fps_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the GPU-memory chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub vram_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the auto-exposure chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub ev_label: Option<AssetId>,
    /// [TextLabel](#textlabel) that receives the HDR-headroom chip text.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub edr_label: Option<AssetId>,
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

impl Component for StatHud {
    const NAME: &'static str = "StatHud";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
