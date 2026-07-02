// src/assets/engine_defaults.rs

use crate::ecs::{AssetOrigin, Component};

/// Opts a world out of individual engine-injected defaults.
///
/// A rendering world is completed at build time with standard assets it does
/// not declare itself: the [DebugHud](#debughud) with its chip
/// [TextLabel](#textlabel)s and font, the [StatHud](#stathud) and its chips
/// when the world declares a [MainMenu](#mainmenu), and, when an
/// [EnvironmentMap](#environmentmap) is present, the sky mesh that displays
/// it. Declaring the same asset yourself replaces the injected one; declaring
/// `EngineDefaults` with a flag set to `false` removes it entirely.
///
/// The build records every injected asset in `world-lock.json`; copy an entry
/// from there (or from `cn explain <name>`) into `world.jsonl` to override it.
///
/// ```jsonl
/// {"name":"defaults","type":"EngineDefaults","args":{"debug_hud":false,"sky":false}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EngineDefaults {
    /// Inject the [StatHud](#stathud) with its chip labels and font when the
    /// world declares a [MainMenu](#mainmenu) but no `StatHud`.
    pub hud: bool,
    /// Inject the [DebugHud](#debughud) with its chip labels when the world
    /// declares no `DebugHud`.
    pub debug_hud: bool,
    /// Inject the sky mesh (a skybox [ProceduralMesh](#proceduralmesh),
    /// [Material](#material), and [Prop](#prop)) when the world has an
    /// [EnvironmentMap](#environmentmap) but no skybox mesh. Disable to use an
    /// `EnvironmentMap` for image-based lighting only, with the background
    /// left to `clear_color` or your own geometry.
    pub sky: bool,
}

impl Default for EngineDefaults {
    fn default() -> Self {
        Self {
            hud: true,
            debug_hud: true,
            sky: true,
        }
    }
}

impl Component for EngineDefaults {
    const NAME: &'static str = "EngineDefaults";
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
    fn bare_args_enable_every_default() {
        let d: EngineDefaults = serde_json::from_str("{}").unwrap();
        assert!(d.hud && d.debug_hud && d.sky);
    }

    #[test]
    fn individual_flags_opt_out() {
        let d: EngineDefaults = serde_json::from_str(r#"{"sky":false}"#).unwrap();
        assert!(!d.sky);
        assert!(d.hud && d.debug_hud);
    }
}
