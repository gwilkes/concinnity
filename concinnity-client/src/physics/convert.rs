// src/physics/convert.rs
//
// Conversions between the engine's plain [f32; 3] / Euler-degree representation
// and the glam-based math types Rapier uses. Keeping these here means no other
// module ever has to name a Rapier or glam type.

use rapier3d::glamx::EulerRot;
use rapier3d::math::{Rotation, Vector};

use crate::assets::{Joint, JointKind, PropBody, PropCollider};
use crate::physics::{ColliderShape, DynamicParams, JointMotor, JointSpec};

// Convert an engine `[x, y, z]` array into a Rapier vector.
pub fn to_vec(v: [f32; 3]) -> Vector {
    Vector::new(v[0], v[1], v[2])
}

// Convert a Rapier vector back into an engine `[x, y, z]` array.
pub fn from_vec(v: Vector) -> [f32; 3] {
    [v.x, v.y, v.z]
}

// Build a Rapier rotation from engine Euler degrees `[pitch, yaw, roll]`,
// applied in YXZ order to match `Prop::model_matrix`.
pub fn to_rotation(euler_deg: [f32; 3]) -> Rotation {
    let [pitch, yaw, roll] = euler_deg;
    Rotation::from_euler(
        EulerRot::YXZ,
        yaw.to_radians(),
        pitch.to_radians(),
        roll.to_radians(),
    )
}

// Decompose a Rapier rotation back into engine Euler degrees
// `[pitch, yaw, roll]` (YXZ order).
pub fn from_rotation(rot: Rotation) -> [f32; 3] {
    let (yaw, pitch, roll) = rot.to_euler(EulerRot::YXZ);
    [pitch.to_degrees(), yaw.to_degrees(), roll.to_degrees()]
}

// The `JointSpec` a `Joint` asset describes, converting authored degrees to
// the radians Rapier expects for revolute joints.
pub fn joint_spec(joint: &Joint) -> JointSpec {
    let limits = if joint.limits_enabled {
        Some(joint.limits)
    } else {
        None
    };
    let motor = if joint.motor_max_force > 0.0 {
        Some(JointMotor {
            target_velocity: joint.motor_target_velocity,
            max_force: joint.motor_max_force,
        })
    } else {
        None
    };
    match joint.parsed_kind() {
        JointKind::Fixed => JointSpec::Fixed,
        JointKind::Spherical => JointSpec::Spherical,
        JointKind::Revolute => JointSpec::Revolute {
            axis: joint.axis,
            // Convert authored degrees to the radians Rapier expects.
            limits: limits.map(|[a, b]| [a.to_radians(), b.to_radians()]),
            motor: motor.map(|m| JointMotor {
                target_velocity: m.target_velocity.to_radians(),
                max_force: m.max_force,
            }),
        },
        JointKind::Prismatic => JointSpec::Prismatic {
            axis: joint.axis,
            limits,
            motor,
        },
    }
}

// The Rapier collision shape for a `PropCollider`, baking in the prop's
// `scale` (the simulation has no separate scale concept).
pub fn collider_shape(collider: &PropCollider, scale: [f32; 3]) -> ColliderShape {
    let [sx, sy, sz] = [scale[0].abs(), scale[1].abs(), scale[2].abs()];
    match collider.shape.as_str() {
        "ball" | "sphere" => ColliderShape::Ball {
            radius: collider.radius * sx,
        },
        "capsule" => ColliderShape::Capsule {
            half_height: collider.half_height * sy,
            radius: collider.radius * sx,
        },
        // "aabb", "cuboid", and anything unrecognised fall back to a box.
        _ => ColliderShape::Cuboid {
            half_extents: [
                collider.half_extents[0] * sx,
                collider.half_extents[1] * sy,
                collider.half_extents[2] * sz,
            ],
        },
    }
}

// The Rapier dynamic-body parameters a `PropBody` describes.
pub fn dynamic_params(body: &PropBody) -> DynamicParams {
    DynamicParams {
        mass: body.mass.max(0.0),
        friction: body.friction.max(0.0),
        restitution: body.restitution.clamp(0.0, 1.0),
        gravity_scale: body.gravity_scale,
        linear_damping: body.linear_damping.max(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_round_trips() {
        let v = [1.5, -2.0, 7.25];
        assert_eq!(from_vec(to_vec(v)), v);
    }

    #[test]
    fn rotation_round_trips_away_from_gimbal_lock() {
        // Pitch kept well clear of +-90 deg so the YXZ decomposition is unique.
        for euler in [[0.0, 0.0, 0.0], [12.0, 45.0, -30.0], [-20.0, 170.0, 60.0]] {
            let back = from_rotation(to_rotation(euler));
            for axis in 0..3 {
                let diff = (back[axis] - euler[axis]).rem_euclid(360.0);
                let diff = diff.min(360.0 - diff);
                assert!(diff < 0.01, "axis {axis}: {back:?} != {euler:?}");
            }
        }
    }

    #[test]
    fn identity_rotation_is_zero_euler() {
        assert_eq!(from_rotation(Rotation::IDENTITY), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn joint_spec_converts_revolute_units_to_radians() {
        let j = Joint {
            kind: "revolute".to_string(),
            axis: [0.0, 0.0, 1.0],
            limits_enabled: true,
            limits: [-90.0, 90.0],
            motor_target_velocity: 180.0,
            motor_max_force: 5.0,
            ..Default::default()
        };
        match joint_spec(&j) {
            JointSpec::Revolute {
                axis,
                limits,
                motor,
            } => {
                assert_eq!(axis, [0.0, 0.0, 1.0]);
                let lim = limits.expect("limits set");
                assert!((lim[0] - (-std::f32::consts::FRAC_PI_2)).abs() < 1.0e-5);
                assert!((lim[1] - std::f32::consts::FRAC_PI_2).abs() < 1.0e-5);
                let m = motor.expect("motor set");
                assert!((m.target_velocity - std::f32::consts::PI).abs() < 1.0e-5);
                assert_eq!(m.max_force, 5.0);
            }
            other => panic!("expected Revolute, got {other:?}"),
        }
    }

    #[test]
    fn joint_spec_prismatic_keeps_units() {
        let j = Joint {
            kind: "prismatic".to_string(),
            axis: [1.0, 0.0, 0.0],
            limits_enabled: true,
            limits: [-0.5, 0.5],
            ..Default::default()
        };
        match joint_spec(&j) {
            JointSpec::Prismatic {
                axis,
                limits,
                motor,
            } => {
                assert_eq!(axis, [1.0, 0.0, 0.0]);
                assert_eq!(limits, Some([-0.5, 0.5]));
                assert!(motor.is_none());
            }
            other => panic!("expected Prismatic, got {other:?}"),
        }
    }

    #[test]
    fn joint_motor_inactive_when_max_force_zero() {
        let j = Joint {
            kind: "revolute".to_string(),
            motor_target_velocity: 30.0,
            motor_max_force: 0.0,
            ..Default::default()
        };
        match joint_spec(&j) {
            JointSpec::Revolute { motor, .. } => assert!(motor.is_none()),
            _ => unreachable!(),
        }
    }
}
