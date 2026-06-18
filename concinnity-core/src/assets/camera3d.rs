// src/assets/camera3d.rs

use crate::ecs::{AssetOrigin, Component};

/// First-person / fly-through controller settings carried on a `Camera3D`.
///
/// A `Camera3D` whose `controller` is set (the default) is driven each frame by
/// the internal camera controller, which turns mouse/keyboard input into a
/// camera orientation and a movement intent. Set `controller` to `null` for a
/// camera driven by something else (a `CameraShot` / `SceneReel` cutscene).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CameraController {
    /// Direct 6-DoF flight mode. WASD moves along the camera's full forward
    /// vector (yaw + pitch) and jump rises along world +Y; the controller
    /// writes the new position straight onto Camera3D, bypassing the physics
    /// step and the bounds box. Used for inspector / fly-through cameras (the
    /// default, e.g. the `cn add foo.glb` scaffold). Set `false` for the
    /// FPS-style ground walker.
    pub free_fly: bool,
    /// Walk / fly speed in world units per second.
    pub move_speed: f32,
    /// Sprint multiplier applied when the sprint key is held.
    pub sprint_multiplier: f32,
    /// Mouse look sensitivity in radians per pixel.
    pub mouse_sensitivity: f32,
    /// Margin kept between the camera and the bounds box (world units).
    pub player_radius: f32,
    /// AABB minimum corner the camera centre must stay inside [x, y, z].
    pub bounds_min: [f32; 3],
    /// AABB maximum corner the camera centre must stay inside [x, y, z].
    pub bounds_max: [f32; 3],
}

impl Default for CameraController {
    fn default() -> Self {
        const BIG: f32 = 1.0e9;
        Self {
            // A bare `Camera3D` is navigable out of the box as a free-fly
            // inspector: the `cn add foo.glb` scaffold relies on this. Worlds
            // that want the FPS ground walker set `free_fly: false`.
            free_fly: true,
            move_speed: 1.0,
            sprint_multiplier: 3.0,
            mouse_sensitivity: 0.0015,
            player_radius: 0.3,
            bounds_min: [-BIG, -BIG, -BIG],
            bounds_max: [BIG, BIG, BIG],
        }
    }
}

// A `Camera3D` with no explicit `controller` gets the default inspector
// controller, so an authored scene is navigable out of the box.
fn default_controller() -> Option<CameraController> {
    Some(CameraController::default())
}

/// Declares the 3D camera. One per scene.
///
/// ```jsonl
/// {
///   "name": "main_camera",
///   "type": "Camera3D",
///   "args": {
///     "fov_y_degrees": 80.0,
///     "near": 0.05,
///     "far": 500.0,
///     "position": [0.0, 4.0, 0.0]
///   }
/// }
/// ```
#[derive(Debug)]
pub struct Camera3D {
    pub fov_y_degrees: f32,
    pub near: f32,
    pub far: f32,
    /// Current view matrix, written each step by the active camera system.
    /// Column-major, matching the GLSL mat4 convention.
    pub view_matrix: [[f32; 4]; 4],
    /// Current world-space eye position, kept in sync with view_matrix.
    pub position: [f32; 3],
    /// Current yaw in radians.
    pub yaw: f32,
    /// Current pitch in radians.
    pub pitch: f32,
    /// World-space horizontal movement intent (units/second). Written by
    /// Camera3DSystem each frame, consumed by PhysicsSystem. Runtime-only.
    pub desired_move: [f32; 3],
    /// Set for one frame when the jump key is pressed. Runtime-only.
    pub jump_requested: bool,
    /// Set for one frame when the interact key is pressed. Runtime-only.
    pub interact_requested: bool,
    /// Controller settings, or `None` for an uncontrolled (cutscene) camera.
    /// Read once by the internal camera controller at init.
    pub controller: Option<CameraController>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Camera3DArgs {
    /// Vertical field-of-view in degrees.
    pub fov_y_degrees: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
    /// Initial eye position in world space [x, y, z].
    pub position: [f32; 3],
    /// Initial yaw in radians (0 = looking toward -Z).
    pub yaw: f32,
    /// Initial pitch in radians.
    pub pitch: f32,
    /// Input controller settings, or `null` to leave the camera uncontrolled
    /// (driven by a [CameraShot](#camerashot) / [SceneReel](#scenereel)
    /// cutscene). Omitted defaults to a free-fly inspector controller.
    #[serde(default = "default_controller")]
    pub controller: Option<CameraController>,
}

impl Default for Camera3DArgs {
    fn default() -> Self {
        Self {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            position: [0.0, 1.7, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            controller: default_controller(),
        }
    }
}

impl Component for Camera3D {
    const NAME: &'static str = "Camera3D";

    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Camera3DArgs;

    fn to_args(&self) -> Camera3DArgs {
        Camera3DArgs {
            fov_y_degrees: self.fov_y_degrees,
            near: self.near,
            far: self.far,
            position: self.position,
            yaw: self.yaw,
            pitch: self.pitch,
            controller: self.controller.clone(),
        }
    }

    fn from_args(args: Camera3DArgs) -> Self {
        Self {
            fov_y_degrees: args.fov_y_degrees,
            near: args.near,
            far: args.far,
            view_matrix: crate::gfx::camera::view_matrix(args.position, args.yaw, args.pitch),
            position: args.position,
            yaw: args.yaw,
            pitch: args.pitch,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller: args.controller,
        }
    }
}
