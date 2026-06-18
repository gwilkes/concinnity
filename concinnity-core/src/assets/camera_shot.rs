// src/assets/camera_shot.rs

use crate::ecs::{AssetOrigin, Component};

/// A reusable [Camera3D](#camera3d) preset: reference it from a
/// [Scene](#scene)'s `camera_shot`, or use it standalone.
///
/// Used standalone, it expands into a [Camera3D](#camera3d) with the same
/// parameters.
///
/// **Library presets** (JSON files in `assets/shots/`):
///
/// ```jsonl
/// // With SceneReel: camera switches per scene (declared on each Scene):
/// {"name":"wide", "type":"CameraShot","args":{"fov_y_degrees":80,"position":[0,1.75,8],"yaw":3.14}}
/// {"name":"close","type":"CameraShot","args":{"fov_y_degrees":55,"position":[0,1.5,3],"yaw":3.14}}
/// {"name":"intro", "type":"Scene","args":{"duration_secs":4.0,"camera_shot":"wide", "transition":"FadeBlack"}}
/// {"name":"detail","type":"Scene","args":{"duration_secs":4.0,"camera_shot":"close","transition":"FadeBlack"}}
/// {"name":"reel","type":"SceneReel","args":{"looping":true,"scenes":["intro","detail"]}}
///
/// // From library preset (standalone, replaces Camera3D):
/// {"name":"cam","type":"CameraShot","args":{"preset":"shot_outdoor_wide"}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraShot {
    /// Name of a built-in or file-backed preset (e.g. "shot_eye_level").
    /// Preset values are used as defaults; any inline fields override them.
    pub preset: String,
    /// Vertical field of view in degrees.
    pub fov_y_degrees: f32,
    /// Near clip plane distance in world units.
    pub near: f32,
    /// Far clip plane distance in world units.
    pub far: f32,
    /// World-space camera position.
    pub position: [f32; 3],
    /// Yaw rotation in radians (Y-axis, applied first).
    pub yaw: f32,
    /// Pitch rotation in radians (X-axis, applied second).
    pub pitch: f32,
}

impl Default for CameraShot {
    fn default() -> Self {
        Self {
            preset: String::new(),
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

impl Component for CameraShot {
    const NAME: &'static str = "CameraShot";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
