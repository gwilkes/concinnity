// src/assets/prop_body.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// Makes a companion [Prop](#prop) a dynamic physics body.
///
/// Attach a PropBody to give a [Prop](#prop) real physics: it falls, collides,
/// stacks, tumbles, and (with `pickup: true` on the prop) can be carried and
/// thrown. A Prop with a `collider` but no PropBody is a static, immovable
/// obstacle.
///
/// ```json
/// {
///   "name": "crate_a_body",
///   "type": "PropBody",
///   "args": { "prop_name": "crate_a", "mass": 4.0, "friction": 0.6 }
/// }
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PropBody {
    /// The [Prop](#prop) this body drives. Must match a Prop declared in the
    /// same world.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub prop_name: Option<AssetId>,
    /// Mass in kilograms. 0 lets the simulation derive mass from the collider
    /// shape and a default density.
    pub mass: f32,
    /// Friction coefficient used for contacts with this body.
    pub friction: f32,
    /// Bounciness in [0, 1]. 0 is fully inelastic.
    pub restitution: f32,
    /// Multiplier applied to world gravity for this body. 1.0 is normal.
    pub gravity_scale: f32,
    /// Linear velocity damping, modelling air drag.
    pub linear_damping: f32,
}

impl Default for PropBody {
    fn default() -> Self {
        Self {
            prop_name: None,
            mass: 0.0,
            friction: 0.5,
            restitution: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.05,
        }
    }
}

impl Component for PropBody {
    const NAME: &'static str = "PropBody";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }
}
