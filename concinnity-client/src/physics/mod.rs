// src/physics/mod.rs
//
// A thin wrapper around the Rapier rigid-body simulation. Rapier (and its
// glam-based math) is confined to this module: callers work entirely in the
// engine's `[f32; 3]` / Euler-degree representation and only ever see the
// opaque `BodyHandle`.
//
// `PhysicsWorld` owns one Rapier simulation. `PhysicsSystem` builds it once at
// init from the world's Props / bodies and steps it every frame.

mod convert;
// The internal physics system that builds and steps the simulation from the
// world's bodies, driven by an optional `PhysicsConfig`.
pub(crate) mod system;

// Asset-to-physics conversions live with the other Rapier-type conversions so
// the asset data types (Joint, Prop, PropBody) carry no dependency on the
// physics backend.
pub(crate) use convert::{collider_shape, dynamic_params, joint_spec};
use convert::{from_rotation, from_vec, to_rotation, to_vec};
use rapier3d::control::{CharacterLength, KinematicCharacterController};
use rapier3d::math::Pose;
use rapier3d::parry::query::DefaultQueryDispatcher;
use rapier3d::parry::utils::Array2;
use rapier3d::prelude::*;

// Opaque handle to a body inside a [`PhysicsWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BodyHandle(RigidBodyHandle);

// Opaque handle to a joint inside a [`PhysicsWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointHandle(ImpulseJointHandle);

// Constraint shape connecting two bodies.
#[derive(Debug, Clone, Copy)]
pub enum JointSpec {
    // All six degrees of freedom locked: bodies move and rotate as one rigid
    // assembly relative to their anchors.
    Fixed,
    // Hinge: rotation is allowed only around `axis` (expressed in each body's
    // local frame). `limits` clamps the hinge angle in radians; `motor`
    // drives it at a target angular velocity in radians/second.
    Revolute {
        axis: [f32; 3],
        limits: Option<[f32; 2]>,
        motor: Option<JointMotor>,
    },
    // Ball-and-socket: translation locked, all three rotational axes free.
    Spherical,
    // Slider: translation is allowed only along `axis` (in each body's local
    // frame). `limits` clamps the slide distance in world units; `motor`
    // drives it at a target linear velocity in units/second.
    Prismatic {
        axis: [f32; 3],
        limits: Option<[f32; 2]>,
        motor: Option<JointMotor>,
    },
}

// Velocity-driven motor parameters for a revolute / prismatic joint.
#[derive(Debug, Clone, Copy)]
pub struct JointMotor {
    // Target velocity (radians/second for revolute, units/second for prismatic).
    pub target_velocity: f32,
    // Maximum force the motor may apply to reach the target.
    pub max_force: f32,
}

// A collision shape, in the body's local space.
#[derive(Debug, Clone, Copy)]
pub enum ColliderShape {
    // Box with the given half-extents along x, y, z.
    Cuboid { half_extents: [f32; 3] },
    // Sphere of the given radius.
    Ball { radius: f32 },
    // Y-axis capsule: a cylinder of `2 * half_height` capped by hemispheres.
    Capsule { half_height: f32, radius: f32 },
}

// Physical parameters for a dynamic (freely simulated) body.
#[derive(Debug, Clone, Copy)]
pub struct DynamicParams {
    // Mass in kilograms. `0.0` lets Rapier derive mass from shape volume.
    pub mass: f32,
    // Coulomb friction coefficient.
    pub friction: f32,
    // Bounciness in `[0, 1]`.
    pub restitution: f32,
    // Multiplier on the world gravity for this body.
    pub gravity_scale: f32,
    // Linear velocity damping (air drag).
    pub linear_damping: f32,
}

// Result of moving the character capsule for one frame.
#[derive(Debug, Clone, Copy)]
pub struct CharacterMove {
    // The translation actually applied after collision resolution.
    pub translation: [f32; 3],
    // True when the capsule is resting on a surface after the move.
    pub grounded: bool,
}

// One Rapier rigid-body simulation.
pub struct PhysicsWorld {
    bodies: RigidBodySet,
    colliders: ColliderSet,
    integration_parameters: IntegrationParameters,
    pipeline: PhysicsPipeline,
    islands: IslandManager,
    broad_phase: BroadPhaseBvh,
    narrow_phase: NarrowPhase,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    gravity: Vector,
    character: KinematicCharacterController,
    // Collider of the player capsule, excluded from its own movement query.
    character_collider: Option<ColliderHandle>,
}

impl std::fmt::Debug for PhysicsWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsWorld")
            .field("bodies", &self.bodies.len())
            .field("colliders", &self.colliders.len())
            .finish()
    }
}

impl PhysicsWorld {
    // Create an empty world. `gravity` is the downward acceleration magnitude
    // in world units per second squared.
    pub fn new(gravity: f32) -> Self {
        Self {
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            pipeline: PhysicsPipeline::new(),
            islands: IslandManager::new(),
            broad_phase: BroadPhaseBvh::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            gravity: Vector::new(0.0, -gravity, 0.0),
            character: KinematicCharacterController::default(),
            character_collider: None,
        }
    }

    // Tune the character controller. `grounded` is true for a gravity-bound
    // character (auto-steps, snaps to the ground) and false for a free-flying
    // camera (no auto-step, no ground snap). A slope of `0` disables the
    // climb limit.
    pub fn configure_character(&mut self, max_slope_deg: f32, step_height: f32, grounded: bool) {
        if max_slope_deg > 0.0 {
            self.character.max_slope_climb_angle = max_slope_deg.to_radians();
            self.character.min_slope_slide_angle = max_slope_deg.to_radians();
        }
        self.character.autostep = if step_height > 0.0 && grounded {
            Some(rapier3d::control::CharacterAutostep {
                max_height: CharacterLength::Absolute(step_height),
                min_width: CharacterLength::Absolute(0.05),
                include_dynamic_bodies: true,
            })
        } else {
            None
        };
        if !grounded {
            self.character.snap_to_ground = None;
        }
    }

    fn collider_builder(shape: &ColliderShape) -> ColliderBuilder {
        match *shape {
            ColliderShape::Cuboid {
                half_extents: [x, y, z],
            } => ColliderBuilder::cuboid(x, y, z),
            ColliderShape::Ball { radius } => ColliderBuilder::ball(radius),
            ColliderShape::Capsule {
                half_height,
                radius,
            } => ColliderBuilder::capsule_y(half_height, radius),
        }
    }

    // Add an immovable body (terrain, walls, static props).
    pub fn add_fixed(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        friction: f32,
    ) -> BodyHandle {
        let body = RigidBodyBuilder::fixed()
            .pose(Pose::from_parts(to_vec(pos), to_rotation(euler_deg)))
            .build();
        let handle = self.bodies.insert(body);
        let collider = Self::collider_builder(shape).friction(friction).build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        BodyHandle(handle)
    }

    // Add a freely simulated dynamic body.
    pub fn add_dynamic(
        &mut self,
        shape: &ColliderShape,
        pos: [f32; 3],
        euler_deg: [f32; 3],
        params: DynamicParams,
    ) -> BodyHandle {
        let mut body = RigidBodyBuilder::dynamic()
            .pose(Pose::from_parts(to_vec(pos), to_rotation(euler_deg)))
            .gravity_scale(params.gravity_scale)
            .linear_damping(params.linear_damping)
            .ccd_enabled(true);
        if params.mass > 0.0 {
            body = body.additional_mass(params.mass);
        }
        let handle = self.bodies.insert(body.build());
        let collider = Self::collider_builder(shape)
            .friction(params.friction)
            .restitution(params.restitution)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        BodyHandle(handle)
    }

    // Add the player character capsule as a position-kinematic body. `center`
    // is the world-space position of the capsule centre.
    pub fn add_character(&mut self, half_height: f32, radius: f32, center: [f32; 3]) -> BodyHandle {
        let body = RigidBodyBuilder::kinematic_position_based()
            .translation(to_vec(center))
            .build();
        let handle = self.bodies.insert(body);
        let collider = ColliderBuilder::capsule_y(half_height, radius).build();
        let collider_handle = self
            .colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
        self.character_collider = Some(collider_handle);
        BodyHandle(handle)
    }

    // Add a static heightfield. `heights` is a `rows * cols` row-major grid of
    // world-space Y values; `scale` is the full extent `[width, 1, depth]`.
    pub fn add_heightfield(
        &mut self,
        rows: usize,
        cols: usize,
        heights: Vec<f32>,
        scale: [f32; 3],
        pos: [f32; 3],
    ) {
        let grid = Array2::new(rows, cols, heights);
        let body = RigidBodyBuilder::fixed().translation(to_vec(pos)).build();
        let handle = self.bodies.insert(body);
        let collider = ColliderBuilder::heightfield(grid, to_vec(scale))
            .friction(1.0)
            .build();
        self.colliders
            .insert_with_parent(collider, handle, &mut self.bodies);
    }

    // Constrain two bodies with a joint. Anchors are in each body's local
    // frame. Returns a handle for future inspection / removal.
    pub fn add_joint(
        &mut self,
        body_a: BodyHandle,
        body_b: BodyHandle,
        anchor_a: [f32; 3],
        anchor_b: [f32; 3],
        spec: JointSpec,
    ) -> JointHandle {
        let anchor1 = to_vec(anchor_a);
        let anchor2 = to_vec(anchor_b);
        let generic: GenericJoint = match spec {
            JointSpec::Fixed => {
                let mut j = FixedJoint::new();
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                j.data
            }
            JointSpec::Revolute {
                axis,
                limits,
                motor,
            } => {
                let axis = normalize_axis(axis);
                let mut j = RevoluteJoint::new(to_vec(axis));
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                if let Some([min, max]) = limits {
                    j.set_limits([min, max]);
                }
                if let Some(m) = motor {
                    j.set_motor_velocity(m.target_velocity, 1.0);
                    j.set_motor_max_force(m.max_force);
                }
                j.data
            }
            JointSpec::Spherical => {
                let mut j = SphericalJoint::new();
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                j.data
            }
            JointSpec::Prismatic {
                axis,
                limits,
                motor,
            } => {
                let axis = normalize_axis(axis);
                let mut j = PrismaticJoint::new(to_vec(axis));
                j.set_local_anchor1(anchor1);
                j.set_local_anchor2(anchor2);
                if let Some([min, max]) = limits {
                    j.set_limits([min, max]);
                }
                if let Some(m) = motor {
                    j.set_motor_velocity(m.target_velocity, 1.0);
                    j.set_motor_max_force(m.max_force);
                }
                j.data
            }
        };
        let handle = self
            .impulse_joints
            .insert(body_a.0, body_b.0, generic, true);
        JointHandle(handle)
    }

    // Advance the simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        self.integration_parameters.dt = dt;
        self.pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }

    // Resolve a desired move of the player capsule against the world without
    // mutating it. Apply the result with [`Self::set_kinematic_translation`].
    pub fn move_character(
        &self,
        half_height: f32,
        radius: f32,
        center: [f32; 3],
        desired: [f32; 3],
        dt: f32,
    ) -> CharacterMove {
        let dispatcher = DefaultQueryDispatcher;
        let mut filter = QueryFilter::default();
        if let Some(collider) = self.character_collider {
            filter = filter.exclude_collider(collider);
        }
        let query =
            self.broad_phase
                .as_query_pipeline(&dispatcher, &self.bodies, &self.colliders, filter);
        let shape = SharedShape::capsule_y(half_height, radius);
        let movement = self.character.move_shape(
            dt.max(1.0e-4),
            &query,
            &*shape,
            &Pose::from_translation(to_vec(center)),
            to_vec(desired),
            |_collision| {},
        );
        CharacterMove {
            translation: from_vec(movement.translation),
            grounded: movement.grounded,
        }
    }

    // Set the next-frame target position of a kinematic body.
    pub fn set_kinematic_translation(&mut self, handle: BodyHandle, pos: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(handle.0) {
            body.set_next_kinematic_translation(to_vec(pos));
        }
    }

    // Switch a body to position-kinematic control (used while a prop is held).
    pub fn make_kinematic(&mut self, handle: BodyHandle) {
        if let Some(body) = self.bodies.get_mut(handle.0) {
            body.set_body_type(RigidBodyType::KinematicPositionBased, true);
        }
    }

    // Switch a body back to dynamic simulation and give it a launch velocity.
    pub fn make_dynamic(&mut self, handle: BodyHandle, linear_velocity: [f32; 3]) {
        if let Some(body) = self.bodies.get_mut(handle.0) {
            body.set_body_type(RigidBodyType::Dynamic, true);
            body.set_linvel(to_vec(linear_velocity), true);
        }
    }

    // Remove a body and its colliders from the world (used when its owning
    // entity is despawned). Rapier purges any joints incident on the body as
    // part of the removal, so no separate joint cleanup is needed.
    pub fn remove_body(&mut self, handle: BodyHandle) {
        self.bodies.remove(
            handle.0,
            &mut self.islands,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
    }

    // Number of rigid bodies currently in the world (player, props, anchors).
    // Test-only observable for the body-reaping path.
    #[cfg(test)]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    // Read a body's current world-space position and Euler rotation.
    pub fn body_pose(&self, handle: BodyHandle) -> ([f32; 3], [f32; 3]) {
        match self.bodies.get(handle.0) {
            Some(body) => (
                from_vec(body.translation()),
                from_rotation(*body.rotation()),
            ),
            None => ([0.0; 3], [0.0; 3]),
        }
    }
}

// Normalize a non-zero axis; fall back to +Y for a degenerate (zero) input so
// the joint stays valid instead of crashing inside Rapier.
fn normalize_axis(axis: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = axis;
    let len = (x * x + y * y + z * z).sqrt();
    if len > 1.0e-6 {
        [x / len, y / len, z / len]
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f32 = 20.0;

    fn ground(world: &mut PhysicsWorld, top_y: f32) {
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [50.0, 5.0, 50.0],
            },
            [0.0, top_y - 5.0, 0.0],
            [0.0; 3],
            0.8,
        );
    }

    #[test]
    fn dynamic_body_falls_under_gravity() {
        let mut world = PhysicsWorld::new(G);
        let body = world.add_dynamic(
            &ColliderShape::Ball { radius: 0.5 },
            [0.0, 10.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
        );
        for _ in 0..30 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(body);
        assert!(pos[1] < 10.0, "body should have fallen, y = {}", pos[1]);
    }

    #[test]
    fn dynamic_body_rests_on_floor() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        let body = world.add_dynamic(
            &ColliderShape::Cuboid {
                half_extents: [0.5, 0.5, 0.5],
            },
            [0.0, 6.0, 0.0],
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
        );
        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(body);
        // Half-extent 0.5 means the centre rests at y ~= 0.5.
        assert!(
            (pos[1] - 0.5).abs() < 0.1,
            "box should rest on floor, y = {}",
            pos[1]
        );
    }

    #[test]
    fn character_is_blocked_by_a_wall() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        // A wall just ahead of the capsule on +X.
        world.add_fixed(
            &ColliderShape::Cuboid {
                half_extents: [0.25, 2.0, 4.0],
            },
            [1.5, 2.0, 0.0],
            [0.0; 3],
            0.5,
        );
        world.add_character(0.6, 0.3, [0.0, 1.0, 0.0]);
        // The broad-phase BVH the movement query reads is built by step().
        world.step(1.0 / 60.0);
        let moved = world.move_character(0.6, 0.3, [0.0, 1.0, 0.0], [5.0, 0.0, 0.0], 1.0 / 60.0);
        // The wall stands at x = 1.25 (1.5 - 0.25); a 0.3-radius capsule from
        // x = 0 cannot advance the full 5 units into it.
        assert!(
            moved.translation[0] < 1.0,
            "wall should block the capsule, dx = {}",
            moved.translation[0]
        );
    }

    #[test]
    fn character_lands_on_ground() {
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, 0.0);
        // Capsule centre at 1.5: with half-height 0.6 + radius 0.3 the bottom
        // sits 0.6 above the floor, so a 10-unit drop should be arrested.
        world.add_character(0.6, 0.3, [0.0, 1.5, 0.0]);
        world.step(1.0 / 60.0);
        let moved = world.move_character(0.6, 0.3, [0.0, 1.5, 0.0], [0.0, -10.0, 0.0], 1.0 / 60.0);
        assert!(
            moved.translation[1] > -1.0 && moved.grounded,
            "fall should be arrested by the floor, dy = {}, grounded = {}",
            moved.translation[1],
            moved.grounded,
        );
    }

    // Test helper: a tiny zero-volume static body acting as a world anchor.
    fn world_anchor(world: &mut PhysicsWorld, pos: [f32; 3]) -> BodyHandle {
        world.add_fixed(&ColliderShape::Ball { radius: 0.01 }, pos, [0.0; 3], 0.0)
    }

    fn free_ball(world: &mut PhysicsWorld, pos: [f32; 3]) -> BodyHandle {
        world.add_dynamic(
            &ColliderShape::Ball { radius: 0.2 },
            pos,
            [0.0; 3],
            DynamicParams {
                mass: 1.0,
                friction: 0.5,
                restitution: 0.0,
                gravity_scale: 1.0,
                linear_damping: 0.0,
            },
        )
    }

    #[test]
    fn revolute_pendulum_swings_under_gravity() {
        // A ball hanging 2 m below a fixed anchor. The revolute hinge is on the
        // +Z axis, so under -Y gravity the ball is free to swing in the X-Y
        // plane. After a few seconds it should have arced away from straight
        // below the anchor.
        let mut world = PhysicsWorld::new(G);
        let anchor = world_anchor(&mut world, [0.0, 5.0, 0.0]);
        // Spawn the ball offset so the pendulum starts above horizontal: a
        // straight-down hang is an unstable equilibrium that may not move.
        let ball = free_ball(&mut world, [2.0, 3.0, 0.0]);
        world.add_joint(
            anchor,
            ball,
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            JointSpec::Revolute {
                axis: [0.0, 0.0, 1.0],
                limits: None,
                motor: None,
            },
        );

        let mut max_speed = 0.0_f32;
        for _ in 0..240 {
            world.step(1.0 / 60.0);
            // Without a velocity getter we infer motion by re-querying position;
            // grab the max horizontal swing from the rest pose.
            let (pos, _) = world.body_pose(ball);
            let dx = pos[0];
            let speed_proxy = dx.abs();
            if speed_proxy > max_speed {
                max_speed = speed_proxy;
            }
        }
        let (pos, _) = world.body_pose(ball);
        // The bob's distance from the anchor should still be ~2 m (the
        // revolute joint preserves the 2 m offset).
        let dx = pos[0];
        let dy = pos[1] - 5.0;
        let length = (dx * dx + dy * dy).sqrt();
        assert!(
            (length - 2.0).abs() < 0.2,
            "pendulum length drifted: {length} (expected ~2.0)"
        );
        assert!(
            max_speed > 0.5,
            "pendulum should have swung; max |dx| = {max_speed}"
        );
    }

    #[test]
    fn fixed_joint_keeps_bodies_attached() {
        // Two dynamic bodies welded together by a fixed joint fall together;
        // their relative offset is preserved.
        let mut world = PhysicsWorld::new(G);
        ground(&mut world, -10.0);
        let a = free_ball(&mut world, [0.0, 5.0, 0.0]);
        let b = free_ball(&mut world, [1.0, 5.0, 0.0]);
        world.add_joint(a, b, [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], JointSpec::Fixed);
        for _ in 0..120 {
            world.step(1.0 / 60.0);
        }
        let (pa, _) = world.body_pose(a);
        let (pb, _) = world.body_pose(b);
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let dz = pb[2] - pa[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        assert!(
            (dist - 1.0).abs() < 0.05,
            "fixed joint should preserve 1 m offset, got {dist}"
        );
    }

    #[test]
    fn prismatic_joint_limits_clamp_slide() {
        // A dynamic body anchored to a fixed point with a Y-axis prismatic
        // joint clamped to [-0.5, 0.5] should fall at most 0.5 m below the
        // anchor before the limit catches it.
        let mut world = PhysicsWorld::new(G);
        let anchor = world_anchor(&mut world, [0.0, 5.0, 0.0]);
        let ball = free_ball(&mut world, [0.0, 5.0, 0.0]);
        world.add_joint(
            anchor,
            ball,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            JointSpec::Prismatic {
                axis: [0.0, 1.0, 0.0],
                limits: Some([-0.5, 0.5]),
                motor: None,
            },
        );
        for _ in 0..240 {
            world.step(1.0 / 60.0);
        }
        let (pos, _) = world.body_pose(ball);
        assert!(
            pos[1] >= 5.0 - 0.5 - 0.05,
            "prismatic limit should arrest fall; y = {}",
            pos[1]
        );
        // Bodies should only slide along Y: X/Z drift should be near zero.
        assert!(pos[0].abs() < 0.05, "ball drifted on X: {}", pos[0]);
        assert!(pos[2].abs() < 0.05, "ball drifted on Z: {}", pos[2]);
    }

    #[test]
    fn normalize_axis_handles_degenerate_input() {
        assert_eq!(normalize_axis([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0]);
        let n = normalize_axis([3.0, 0.0, 4.0]);
        assert!((n[0] - 0.6).abs() < 1.0e-6 && (n[2] - 0.8).abs() < 1.0e-6);
        // Zero input falls back to +Y.
        assert_eq!(normalize_axis([0.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
    }
}
