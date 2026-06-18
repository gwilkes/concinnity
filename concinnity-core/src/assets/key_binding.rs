// src/assets/key_binding.rs

use crate::ecs::{AssetOrigin, Component};

/// Maps a keyboard key to an action string.
///
/// When the bound key is pressed, the action fires once per press (like a
/// [HitRegion](#hitregion) click). Bindings only run while the cursor is free:
/// they're inactive in worlds that capture the cursor for camera control.
///
/// The action vocabulary is the same as [HitRegion](#hitregion)'s:
/// - `"scene:<name>"`:       jump to the named [Scene](#scene)
/// - `"quit"`:               stop the application
/// - `"view:show:<name>"`:   show the named [View](#view) overlay
/// - `"view:hide"`:          hide the active [View](#view)
/// - `"view:toggle:<name>"`: toggle the named [View](#view)
///
/// Recognised key names are case-sensitive; currently `"Escape"` is supported.
///
/// ```jsonl
/// {"name":"esc_binding","type":"KeyBinding","args":{"key":"Escape","action":"view:toggle:pause_menu"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct KeyBinding {
    /// The key name to bind (e.g. `"Escape"`).
    pub key: String,
    /// The action to fire when the key is pressed.
    pub action: String,
}

impl Component for KeyBinding {
    const NAME: &'static str = "KeyBinding";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_escape_to_view_toggle() {
        let json = r#"{"key":"Escape","action":"view:toggle:pause_menu"}"#;
        let kb: KeyBinding = serde_json::from_str(json).unwrap();
        assert_eq!(kb.key, "Escape");
        assert_eq!(kb.action, "view:toggle:pause_menu");
    }

    #[test]
    fn deserializes_with_defaults_to_empty_strings() {
        let kb: KeyBinding = serde_json::from_str("{}").unwrap();
        assert!(kb.key.is_empty());
        assert!(kb.action.is_empty());
    }
}
