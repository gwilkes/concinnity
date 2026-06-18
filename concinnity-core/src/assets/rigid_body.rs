// src/assets/rigid_body.rs

use crate::ecs::{AssetOrigin, Component};

/// Gives a player [Camera3D](#camera3d) gravity, jumping, and a grounded
/// character body.
///
/// Every [Camera3D](#camera3d) already collides with the world as a capsule.
/// Adding a RigidBody upgrades that camera from a free-flying spectator to a
/// grounded character: it falls under gravity, lands on surfaces, climbs steps,
/// slides off steep slopes, and can jump. The capsule size is configured here
/// too.
///
/// ```json
/// { "name": "player_body", "type": "RigidBody", "args": { "jump_height": 1.4 } }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RigidBody {
    /// Multiplier applied to the global gravity constant. 1.0 = normal gravity.
    pub gravity_scale: f32,
    /// Radius of the player capsule used for collision, in world units.
    pub capsule_radius: f32,
    /// Total height of the player capsule. The camera eye sits at the top.
    pub capsule_height: f32,
    /// Apex height of a jump in world units. 0 disables jumping.
    pub jump_height: f32,
    /// Steepest slope the player can walk up, in degrees.
    pub max_slope_deg: f32,
    /// Tallest obstacle the controller auto-steps over, in world units.
    pub step_height: f32,
    /// True when the capsule is resting on a surface this frame.
    /// Written by PhysicsSystem.
    #[serde(skip)]
    pub is_grounded: bool,
}

impl Default for RigidBody {
    fn default() -> Self {
        Self {
            gravity_scale: 1.0,
            capsule_radius: 0.3,
            capsule_height: 1.7,
            jump_height: 1.1,
            max_slope_deg: 50.0,
            step_height: 0.3,
            is_grounded: true,
        }
    }
}

impl Component for RigidBody {
    const NAME: &'static str = "RigidBody";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(mut args: Self) -> Self {
        // Runtime state is always reset on construction.
        args.is_grounded = true;
        args
    }
}
