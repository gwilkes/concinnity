// src/assets/animation.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};
use crate::gfx::skinning::{self, JointPose};

/// One keyframe in an animation track: a joint pose sampled at `time` seconds.
/// The pose fields (`translation`, `rotation_deg`, `scale`) are given directly
/// on the keyframe, each defaulting to the identity transform when omitted.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Keyframe {
    /// Time of this keyframe in seconds from the clip start.
    pub time: f32,
    /// The joint's transform at this keyframe.
    #[serde(flatten)]
    pub pose: JointPose,
}

/// An animation channel: a time-ordered list of keyframes for one joint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnimationTrack {
    /// Index of the joint in the target skeleton this track drives.
    pub joint: usize,
    /// Keyframes, expected in ascending time order.
    pub keyframes: Vec<Keyframe>,
}

/// A skeletal animation clip that animates one [SkinnedMesh](#skinnedmesh).
///
/// The clip plays every frame, sampling each track and deforming the target
/// mesh's skeleton. Joints with no track hold their bind pose.
///
/// Several `Animation` assets may target the same [SkinnedMesh](#skinnedmesh);
/// they are then blended into one pose, weighted by each clip's `weight` (a
/// normalised weighted average). A single clip plays at full strength
/// regardless of its `weight`.
///
/// **glTF import.** A clip may be authored entirely by hand (`tracks` filled
/// out, `source` left empty) or imported from the same `.glb` that backs the
/// target [SkinnedMesh](#skinnedmesh). Set `source` to the `.glb` path and the
/// build imports `duration` + `tracks` from it. `animation_index` picks one
/// clip when the file contains several (default 0); `animation_name` names it
/// for matching against the file's clip names: when set it takes precedence
/// over the index. Channels whose target node is not a joint of the file's
/// first skinned node are dropped. The same `.glb` should back the target
/// [SkinnedMesh](#skinnedmesh) so the joint indices agree.
///
/// ```jsonl
/// // Inline:
/// {"name":"flag_wave","type":"Animation","args":{"target":"flag","duration":2.0,"tracks":[{"joint":1,"keyframes":[{"time":0.0,"rotation_deg":[0,0,0]},{"time":1.0,"rotation_deg":[0,30,0]},{"time":2.0,"rotation_deg":[0,0,0]}]}]}}
/// // From glTF:
/// {"name":"hero_walk","type":"Animation","args":{"target":"hero","source":"models/hero.glb","animation_name":"Walk","looping":true}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Animation {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// The [SkinnedMesh](#skinnedmesh) asset this clip animates.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub target: Option<AssetId>,
    /// Optional path to a `.glb` file. When set, the build imports
    /// `duration` + `tracks` from it; inline-authored clips leave this empty.
    pub source: String,
    /// Index of the animation to import when `source` is set and the file
    /// contains several. Ignored when `animation_name` is non-empty.
    pub animation_index: u32,
    /// Name of the animation to import. When set, the matching glTF animation
    /// is looked up by name; takes precedence over `animation_index`.
    pub animation_name: String,
    /// Clip length in seconds. Overridden by glTF import.
    pub duration: f32,
    /// When true, playback wraps after `duration`.
    pub looping: bool,
    /// Blend weight used when several clips target the same
    /// [SkinnedMesh](#skinnedmesh). Ignored when this is the only clip on its
    /// target.
    pub weight: f32,
    /// When non-zero, the clip's contribution ramps from 0 to its declared
    /// `weight` over this many seconds after the world starts. Zero (the
    /// default) plays the clip at full `weight` from the first frame.
    pub fade_in_secs: f32,
    /// Per-joint keyframe channels.
    pub tracks: Vec<AnimationTrack>,
}

impl Default for Animation {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            target: None,
            source: String::new(),
            animation_index: 0,
            animation_name: String::new(),
            duration: 1.0,
            looping: true,
            weight: 1.0,
            fade_in_secs: 0.0,
            tracks: Vec::new(),
        }
    }
}

impl Animation {
    /// Convert this asset into the runtime `AnimationClip` consumed by the
    /// skinning math.
    pub fn to_clip(&self) -> skinning::AnimationClip {
        skinning::AnimationClip {
            duration: self.duration.max(1e-3),
            looping: self.looping,
            tracks: self
                .tracks
                .iter()
                .map(|t| skinning::JointTrack {
                    joint: t.joint,
                    keys: t
                        .keyframes
                        .iter()
                        .map(|k| skinning::Keyframe {
                            time: k.time,
                            pose: k.pose,
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

impl Component for Animation {
    const NAME: &'static str = "Animation";
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

impl crate::build::SourceBacked for Animation {
    // A glTF-sourced Animation needs its `.glb` fetched before the build's
    // desugar pass can expand it; an inline-authored clip has no source.
    fn source_path(args: &serde_json::Value, _platform: crate::build::Platform) -> Option<String> {
        args.get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{Platform, SourceBacked};

    #[test]
    fn deserialises_with_defaults() {
        let a: Animation = serde_json::from_str("{}").unwrap();
        assert_eq!(a.duration, 1.0);
        assert!(a.looping);
        assert_eq!(a.weight, 1.0);
        assert!(a.tracks.is_empty());
        assert_eq!(a.source, "");
        assert_eq!(a.animation_index, 0);
        assert_eq!(a.animation_name, "");
    }

    #[test]
    fn deserialises_glb_source_fields() {
        let json = r#"{
            "target":"hero",
            "source":"models/hero.glb",
            "animation_index":2,
            "animation_name":"Walk",
            "looping":false
        }"#;
        let a: Animation = serde_json::from_str(json).unwrap();
        assert_eq!(a.source, "models/hero.glb");
        assert_eq!(a.animation_index, 2);
        assert_eq!(a.animation_name, "Walk");
        assert!(!a.looping);
    }

    #[test]
    fn deserialises_inline_tracks() {
        let json = r#"{
            "target":"flag",
            "duration":2.0,
            "tracks":[{"joint":0,"keyframes":[{"time":0.0,"rotation_deg":[0,30,0]}]}]
        }"#;
        let a: Animation = serde_json::from_str(json).unwrap();
        assert_eq!(a.duration, 2.0);
        assert_eq!(a.tracks.len(), 1);
        assert_eq!(a.tracks[0].joint, 0);
    }

    #[test]
    fn source_backed_returns_path_only_when_set() {
        let with = serde_json::json!({"source": "x.glb"});
        let without = serde_json::json!({"source": ""});
        let missing = serde_json::json!({});
        assert_eq!(
            <Animation as SourceBacked>::source_path(&with, Platform::Metal),
            Some("x.glb".to_string())
        );
        assert!(<Animation as SourceBacked>::source_path(&without, Platform::Metal).is_none());
        assert!(<Animation as SourceBacked>::source_path(&missing, Platform::Metal).is_none());
    }

    #[test]
    fn to_clip_floors_duration_so_runtime_loop_does_not_divide_by_zero() {
        let a = Animation {
            duration: 0.0,
            ..Default::default()
        };
        let clip = a.to_clip();
        assert!(clip.duration >= 1e-3);
    }
}
