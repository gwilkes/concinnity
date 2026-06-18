// src/assets/view.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// A named overlay layer drawn on top of the active [Scene](#scene).
///
/// UI elements ([Sprite](#sprite), [TextLabel](#textlabel),
/// [HitRegion](#hitregion)) belong to a view by name prefix `<view_name>_*`,
/// mirroring the [Scene](#scene) → [Prop](#prop) convention. Views are shown /
/// hidden via [HitRegion](#hitregion) or [KeyBinding](#keybinding) actions:
/// - `view:show:<name>`
/// - `view:hide`
/// - `view:toggle:<name>`
///
/// When a view is active, its UI elements become visible and the underlying
/// scene's [HitRegion](#hitregion)s stop firing. Hiding the view restores the
/// scene exactly as it was. Only one view can be active at a time.
///
/// ```jsonl
/// {"name":"pause_menu","type":"View","args":{}}
/// // UI assets prefixed pause_menu_* belong to this view:
/// {"name":"pause_menu_dim","type":"Sprite","args":{"x":0,"y":0,"width":1280,"height":720,"tint":[0,0,0,0.55]}}
/// {"name":"pause_menu_btn_resume","type":"HitRegion","args":{"action":"view:hide", ...}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct View {
    #[serde(skip)]
    pub asset_id: AssetId,
    /// When true, this view is shown as soon as the world loads.
    pub initial: bool,
    /// Seconds to fade the view in when it's shown. 0 shows it instantly.
    pub fade_in_secs: f32,
}

impl Component for View {
    const NAME: &'static str = "View";
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_with_defaults() {
        let v: View = serde_json::from_str("{}").unwrap();
        assert_eq!(v.fade_in_secs, 0.0);
        assert!(!v.initial);
    }

    #[test]
    fn deserializes_with_initial_true() {
        let v: View = serde_json::from_str(r#"{"initial":true}"#).unwrap();
        assert!(v.initial);
    }
}
