// src/assets/audio_emitter.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// A point source of sound in the world.
///
/// Plays its `clip` (an [AudioClip](#audioclip) reference) from `position`,
/// attenuated and panned relative to the camera. When `prop` names a
/// [Prop](#prop), the emitter tracks that prop's position every frame, so the
/// sound follows a moving object.
///
/// ```jsonl
/// {"name":"fire_sound","type":"AudioEmitter","args":{"clip":"fire_loop","position":[6.0,4.0,-6.0]}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct AudioEmitter {
    /// The [AudioClip](#audioclip) this emitter plays.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub clip: Option<AssetId>,
    /// World-space position of the sound source.
    pub position: [f32; 3],
    /// Linear gain multiplier applied to the clip.
    pub volume: f32,
    /// Whether the clip restarts when it ends.
    pub looping: bool,
    /// Optional [Prop](#prop) whose position the emitter tracks each frame.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub prop: Option<AssetId>,
}

impl Default for AudioEmitter {
    fn default() -> Self {
        Self {
            clip: None,
            position: [0.0; 3],
            volume: 1.0,
            looping: true,
            prop: None,
        }
    }
}

impl Component for AudioEmitter {
    const NAME: &'static str = "AudioEmitter";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn from_args(args: Self) -> Self {
        args
    }
}
