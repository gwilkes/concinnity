// src/assets/scene_reel.rs

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, Component};

/// An ordered playlist of named [Scene](#scene)s.
///
/// The current scene's [Prop](#prop)s are shown, then it advances to the next
/// based on that scene's `duration_secs`. Timing and transition style are
/// declared on each [Scene](#scene) asset. Props not prefixed by any scene name
/// remain visible in all scenes.
///
/// ```jsonl
/// {"name":"day",  "type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
/// {"name":"night","type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
/// {"name":"day_sun",  "type":"Prop","args":{"model":"model_sun_disc","position":[0,80,-200]}}
/// {"name":"night_moon","type":"Prop","args":{"model":"model_moon_disc","position":[0,80,-200]}}
/// {"name":"reel","type":"SceneReel","args":{"looping":true,"scenes":["day","night"]}}
/// ```
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SceneReel {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Ordered list of [Scene](#scene) assets to play.
    pub scenes: Vec<AssetId>,
    /// When true, wraps back to the first scene after the last one ends.
    pub looping: bool,
    /// Index of the entry that is active at world start.
    pub start_index: u32,
}

impl Component for SceneReel {
    const NAME: &'static str = "SceneReel";
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

impl crate::check::cross_reference::CrossReferenced for SceneReel {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let mut refs = Vec::new();

        if let Some(entries) = args.get("scenes").and_then(|v| v.as_array()) {
            if entries.is_empty() {
                refs.push(CrossRef::Issue(format!(
                    "SceneReel '{}': scenes list is empty",
                    name
                )));
            }
            for (i, entry) in entries.iter().enumerate() {
                let scene_ref = entry.as_str().unwrap_or("");
                if scene_ref.is_empty() {
                    refs.push(CrossRef::Issue(format!(
                        "SceneReel '{}': scenes[{}] is not a valid scene name string",
                        name, i
                    )));
                } else {
                    refs.push(CrossRef::Resolve {
                        kind: RefKind::Scene,
                        target: scene_ref.to_string(),
                        error: format!(
                            "SceneReel '{}': scenes[{}] references unknown scene '{}', add a Scene asset with that name",
                            name, i, scene_ref
                        ),
                    });
                }
            }
        }

        refs
    }
}
