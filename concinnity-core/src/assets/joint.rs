// src/assets/joint.rs

use crate::ecs::asset_id::{AssetId, de_opt_asset_ref};
use crate::ecs::{AssetOrigin, Component};

/// The constraint shape a `Joint` declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JointKind {
    /// All 6 degrees of freedom locked. The bodies move and rotate as one
    /// rigid assembly relative to their anchors. Use to weld two props
    /// together.
    Fixed,
    /// Single rotational axis. Rotation around `axis` (in each body's local
    /// frame) is free; everything else is locked. The canonical door hinge.
    Revolute,
    /// Three rotational axes free, all translation locked. Ball-and-socket
    /// joint: the canonical rope link or a hip socket.
    Spherical,
    /// Single translational axis. Sliding along `axis` is free; rotation and
    /// the other two translational axes are locked. The canonical slider /
    /// piston.
    Prismatic,
}

impl JointKind {
    fn from_str_norm(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fixed" | "weld" => Some(Self::Fixed),
            "revolute" | "hinge" => Some(Self::Revolute),
            "spherical" | "ball" | "socket" => Some(Self::Spherical),
            "prismatic" | "slider" | "piston" => Some(Self::Prismatic),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Revolute => "revolute",
            Self::Spherical => "spherical",
            Self::Prismatic => "prismatic",
        }
    }
}

/// A physics constraint connecting two [Prop](#prop)s that own a `collider`.
///
/// The joint pins `anchor_a` on `body_a` to `anchor_b` on `body_b` and locks
/// the relative motion of the two bodies according to its `kind`. Anchors are
/// in each body's local frame: `[0, 0, 0]` is the body's own pivot.
///
/// To anchor a body to "the world" (no second prop), leave `body_b` empty: a
/// hidden static anchor is created at `anchor_b` (interpreted as world space in
/// that case) and the body joints to it. This is the pendulum / lamp / trapeze
/// pattern.
///
/// `axis` only applies to `revolute` and `prismatic`: it is the single free
/// axis (rotation or translation) in each body's local frame. The vector is
/// normalised on load; a zero axis falls back to `[0, 1, 0]`.
///
/// `limits_enabled` clamps the free axis: angle in degrees for revolute,
/// distance in world units for prismatic. `motor_target_velocity` and
/// `motor_max_force` drive the free axis when `motor_max_force > 0`; the
/// velocity is in degrees/sec for revolute, units/sec for prismatic.
///
/// ```jsonl
/// // Pendulum: a dynamic ball hanging 2 m below a world anchor, hinged on +Z.
/// {"name":"pendulum_joint","type":"Joint","args":{
///   "kind":"revolute","body_a":"pendulum_bob",
///   "anchor_a":[0,2,0],"anchor_b":[0,5,0],"axis":[0,0,1]
/// }}
///
/// // Door: hinged on a wall, swing limited to ±90°.
/// {"name":"door_hinge","type":"Joint","args":{
///   "kind":"revolute","body_a":"wall","body_b":"door",
///   "anchor_a":[1,1,0],"anchor_b":[-0.5,0,0],"axis":[0,1,0],
///   "limits_enabled":true,"limits":[-90,90]
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Joint {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// Constraint shape; defaults to "fixed".
    pub kind: String,
    /// First body: a [Prop](#prop) name. Required.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub body_a: Option<AssetId>,
    /// Second body: a [Prop](#prop) name. Empty means "world anchor", in which
    /// case `anchor_b` is interpreted as a world-space position.
    #[serde(deserialize_with = "de_opt_asset_ref")]
    pub body_b: Option<AssetId>,
    /// Attach point in `body_a`'s local frame.
    pub anchor_a: [f32; 3],
    /// Attach point in `body_b`'s local frame (or world space if `body_b` is
    /// empty).
    pub anchor_b: [f32; 3],
    /// Free axis for revolute/prismatic, in each body's local frame.
    pub axis: [f32; 3],
    /// Whether the `limits` clamp is enforced.
    pub limits_enabled: bool,
    /// `[min, max]` clamp on the free axis: degrees for revolute, world units
    /// for prismatic. Ignored unless `limits_enabled` is true.
    pub limits: [f32; 2],
    /// Motor target velocity: degrees/sec for revolute, world units/sec for
    /// prismatic. Ignored unless `motor_max_force > 0`.
    pub motor_target_velocity: f32,
    /// Motor force budget. The motor is inactive when this is 0.
    pub motor_max_force: f32,
}

impl Default for Joint {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            kind: "fixed".to_string(),
            body_a: None,
            body_b: None,
            anchor_a: [0.0, 0.0, 0.0],
            anchor_b: [0.0, 0.0, 0.0],
            axis: [0.0, 1.0, 0.0],
            limits_enabled: false,
            limits: [0.0, 0.0],
            motor_target_velocity: 0.0,
            motor_max_force: 0.0,
        }
    }
}

impl Joint {
    /// Parse `kind`; falls back to `Fixed` for unrecognised values so a typo
    /// degrades safely. Cross-reference validation flags bad kinds explicitly.
    pub fn parsed_kind(&self) -> JointKind {
        JointKind::from_str_norm(&self.kind).unwrap_or(JointKind::Fixed)
    }
}

impl Component for Joint {
    const NAME: &'static str = "Joint";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Normalise the kind string so `to_args` round-trips cleanly.
        if let Some(k) = JointKind::from_str_norm(&args.kind) {
            args.kind = k.as_str().to_string();
        }
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }
}

impl crate::check::cross_reference::CrossReferenced for Joint {
    fn cross_refs(
        name: &str,
        args: &serde_json::Value,
    ) -> Vec<crate::check::cross_reference::CrossRef> {
        use crate::check::cross_reference::{CrossRef, RefKind};
        let arg_str = |key: &str| args.get(key).and_then(|v| v.as_str()).unwrap_or("");
        let mut refs = Vec::new();

        let kind = arg_str("kind");
        if !kind.is_empty() && JointKind::from_str_norm(kind).is_none() {
            refs.push(CrossRef::Issue(format!(
                "Joint '{name}': unknown kind '{kind}' (expected one of fixed | revolute | spherical | prismatic)"
            )));
        }

        let body_a = arg_str("body_a");
        if body_a.is_empty() {
            refs.push(CrossRef::Issue(format!(
                "Joint '{name}': `body_a` is required, name of a Prop with a collider"
            )));
        } else {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: body_a.to_string(),
                error: format!("Joint '{name}': body_a '{body_a}' is not a declared Prop"),
            });
        }

        let body_b = arg_str("body_b");
        if !body_b.is_empty() {
            refs.push(CrossRef::Resolve {
                kind: RefKind::Prop,
                target: body_b.to_string(),
                error: format!("Joint '{name}': body_b '{body_b}' is not a declared Prop"),
            });
        }

        refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialises_with_defaults() {
        let j: Joint = serde_json::from_str("{}").unwrap();
        assert_eq!(j.kind, "fixed");
        assert_eq!(j.anchor_a, [0.0, 0.0, 0.0]);
        assert_eq!(j.axis, [0.0, 1.0, 0.0]);
        assert!(!j.limits_enabled);
        assert_eq!(j.motor_max_force, 0.0);
    }

    #[test]
    fn deserialises_all_fields() {
        let json = r#"{
            "kind":"revolute",
            "body_a":"door",
            "body_b":"wall",
            "anchor_a":[0.5,1.0,0.0],
            "anchor_b":[1.0,1.0,0.0],
            "axis":[0,1,0],
            "limits_enabled":true,
            "limits":[-90,90],
            "motor_target_velocity":30.0,
            "motor_max_force":50.0
        }"#;
        let j: Joint = serde_json::from_str(json).unwrap();
        assert_eq!(j.parsed_kind(), JointKind::Revolute);
        assert!(j.body_a.is_some());
        assert!(j.body_b.is_some());
        assert!(j.limits_enabled);
    }

    #[test]
    fn aliases_resolve_to_canonical_kind() {
        assert_eq!(JointKind::from_str_norm("hinge"), Some(JointKind::Revolute));
        assert_eq!(JointKind::from_str_norm("WELD"), Some(JointKind::Fixed));
        assert_eq!(JointKind::from_str_norm("ball"), Some(JointKind::Spherical));
        assert_eq!(
            JointKind::from_str_norm("slider"),
            Some(JointKind::Prismatic)
        );
    }

    #[test]
    fn from_args_normalises_kind_string() {
        let json = r#"{"kind":"HINGE"}"#;
        let parsed: Joint = serde_json::from_str(json).unwrap();
        let normalised = Joint::from_args(parsed);
        assert_eq!(normalised.kind, "revolute");
    }

    #[test]
    fn unknown_kind_falls_back_to_fixed() {
        let j = Joint {
            kind: "frumpus".to_string(),
            ..Default::default()
        };
        assert_eq!(j.parsed_kind(), JointKind::Fixed);
    }
}
