// src/assets/audio_clip.rs

use std::collections::HashSet;

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator, PipelineContext};

/// A baked audio clip: the sound an [AudioEmitter](#audioemitter) plays.
///
/// The build reads the `source` file (any format the engine can decode:
/// `.ogg`, `.wav`, `.flac`, `.mp3`) and packs it into the world.
///
/// An `AudioClip` is inert on its own: reference it from an
/// [AudioEmitter](#audioemitter)'s `clip` field to place the sound in the world.
///
/// ```jsonl
/// {"name":"fire_loop","type":"AudioClip","args":{"source":"audio/fire_crackle.ogg"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct AudioClip {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Path to the source audio file.
    pub source: String,
    /// Injected at load time from the compiled blob payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Component for AudioClip {
    const NAME: &'static str = "AudioClip";

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

/// Blob indices that hold an `AudioClip` payload.
///
/// `AudioSystem` reads these payloads at its `init`, but it inits *after* the
/// graphics systems, which free blob payloads once their own GPU uploads are
/// done. The graphics systems consult this so they leave the audio blobs
/// resident for `AudioSystem` to read.
pub fn audio_clip_blob_indices(ctx: &PipelineContext) -> HashSet<u32> {
    ctx.query::<AudioClip>()
        .filter_map(|c| c.locator.as_ref().map(|l| l.blob_index))
        .collect()
}

impl crate::build::SourceBacked for AudioClip {
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}
