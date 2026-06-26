// src/gfx/camera_controller.rs
//
// First-person / fly-through camera controller. An internal system (not a
// declarable asset): `World::start` constructs one when the world has a
// `Camera3D` whose `controller` is set, reading that controller's config. It
// turns mouse/keyboard input into a `Camera3D` orientation and a movement
// intent for the player's `RigidBody`.

use crate::assets::{Camera3D, CameraController, ControlsCommand, FrameInput, Prop};
use crate::ecs::{PipelineContext, StepResult, System};
use std::time::Instant;

// Reach distance for interacting with a Prop, in world units.
const INTERACT_REACH: f32 = 3.0;
// Minimum facing dot product (~60-degree cone) for an interaction.
const INTERACT_MIN_DOT: f32 = 0.5;

// First-person / fly-through controller behavior. Constructed internally by
// `World::start` from the controlling `Camera3D`'s `CameraController`;
// never a world-declared asset.
#[derive(Debug)]
pub struct Camera3DSystem {
    free_fly: bool,
    move_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    player_radius: f32,
    bounds_min: [f32; 3],
    bounds_max: [f32; 3],
    last_step: Option<Instant>,
    // smoothed horizontal velocity; lerped toward the target each tick so
    // WASD movement accelerates and decelerates instead of snapping
    velocity: [f32; 3],
    // indices into the Prop list for props that have interactable=true,
    // collected at init() so step() can query_mut only those props
    interactable_indices: Vec<usize>,
    // Cursor into the Events<ControlsCommand> queue (live settings changes).
    controls_cursor: crate::ecs::EventCursor,
}

impl Camera3DSystem {
    // Build a controller from a `Camera3D`'s controller settings.
    pub fn new(c: CameraController) -> Self {
        Self {
            free_fly: c.free_fly,
            move_speed: c.move_speed,
            sprint_multiplier: c.sprint_multiplier,
            mouse_sensitivity: c.mouse_sensitivity,
            player_radius: c.player_radius,
            bounds_min: c.bounds_min,
            bounds_max: c.bounds_max,
            last_step: None,
            velocity: [0.0; 3],
            interactable_indices: Vec::new(),
            controls_cursor: crate::ecs::EventCursor::default(),
        }
    }

    // Zero the smoothed movement velocity. Called when an external source (the
    // cn debug `camera-set` command) teleports the camera, so free-fly velocity
    // integration does not drift the new pose on the next step. Only reached
    // from the binary-only debug drive, hence dead in a `--lib` build.
    #[allow(dead_code)]
    pub fn reset_velocity(&mut self) {
        self.velocity = [0.0; 3];
    }
}

impl System for Camera3DSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.last_step = Some(Instant::now());

        // A persisted mouse-sensitivity choice (settings menu) overrides the
        // camera's authored value. `None` keeps the authored value.
        if let Some(s) = crate::config::Settings::load().controls.mouse_sensitivity {
            self.mouse_sensitivity = s;
        }

        // GraphicsSystem runs first (it is prepended ahead of every other
        // system) and queries Props without draining them, so they are still
        // present here.
        for (i, prop) in ctx.query::<Prop>().enumerate() {
            if prop.interactable {
                self.interactable_indices.push(i);
            }
        }

        if !self.interactable_indices.is_empty() {
            tracing::debug!(
                "Camera3DSystem: registered {} interactable prop(s)",
                self.interactable_indices.len()
            );
        }
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // Apply any live controls change (settings-menu sensitivity slider)
        // sent this tick by GraphicsSystem, which runs first. The last one this
        // tick wins.
        if let Some(events) = ctx.events::<ControlsCommand>() {
            for cmd in events.read(&mut self.controls_cursor) {
                self.mouse_sensitivity = cmd.mouse_sensitivity;
            }
        }

        // Read (not drain) the input snapshot deposited by GraphicsSystem this
        // frame, so UiInputSystem can read the same snapshot (e.g. for a pause
        // menu over this camera). GraphicsSystem clears it before the next push.
        let input = match ctx.query::<FrameInput>().next().cloned() {
            Some(i) => i,
            // no input means GraphicsSystem hasn't run yet or there is no
            // graphics backend -- nothing to do this tick
            None => return StepResult::Continue,
        };

        let now = Instant::now();
        let dt = self
            .last_step
            .map(|t| now.duration_since(t).as_secs_f32().min(0.1))
            .unwrap_or(0.0);
        self.last_step = Some(now);

        // update every Camera3D in the world (normally exactly one)
        for camera in ctx.query_mut::<Camera3D>() {
            // mouse look
            camera.yaw -= input.mouse_dx * self.mouse_sensitivity;
            camera.pitch = (camera.pitch - input.mouse_dy * self.mouse_sensitivity).clamp(
                -std::f32::consts::FRAC_PI_2 + 0.01,
                std::f32::consts::FRAC_PI_2 - 0.01,
            );

            let speed = if input.sprint {
                self.move_speed * self.sprint_multiplier
            } else {
                self.move_speed
            };

            // Two movement modes share the same input/decay/view-matrix
            // outer loop; only the basis vectors and how velocity is
            // committed differ. Free-fly drives the camera position
            // directly and adds a vertical component; the FPS walker keeps
            // motion horizontal and delegates to PhysicsSystem.
            let (fwd, right) = if self.free_fly {
                let cp = camera.pitch.cos();
                (
                    [
                        -camera.yaw.sin() * cp,
                        camera.pitch.sin(),
                        -camera.yaw.cos() * cp,
                    ],
                    [camera.yaw.cos(), 0.0_f32, -camera.yaw.sin()],
                )
            } else {
                (
                    [-camera.yaw.sin(), 0.0_f32, -camera.yaw.cos()],
                    [camera.yaw.cos(), 0.0_f32, -camera.yaw.sin()],
                )
            };

            // build the target velocity from current key state
            let mut target = [0.0_f32; 3];
            if input.forward {
                target[0] += fwd[0] * speed;
                target[1] += fwd[1] * speed;
                target[2] += fwd[2] * speed;
            }
            if input.backward {
                target[0] -= fwd[0] * speed;
                target[1] -= fwd[1] * speed;
                target[2] -= fwd[2] * speed;
            }
            if input.right {
                target[0] += right[0] * speed;
                target[2] += right[2] * speed;
            }
            if input.left {
                target[0] -= right[0] * speed;
                target[2] -= right[2] * speed;
            }
            // Free-fly: jump is "rise"; no down key, descend by pitching down + W.
            if self.free_fly && input.jump {
                target[1] += speed;
            }

            // exponential decay toward target -- time-correct so frame rate does not
            // affect the feel. half_life controls how quickly speed builds/drops.
            let half_life = 0.08_f32; // seconds to reach ~50% of target speed
            let decay = 1.0 - 2.0_f32.powf(-dt / half_life);
            self.velocity[0] += (target[0] - self.velocity[0]) * decay;
            self.velocity[1] += (target[1] - self.velocity[1]) * decay;
            self.velocity[2] += (target[2] - self.velocity[2]) * decay;

            if self.free_fly {
                // Apply directly; no PhysicsSystem, no bounds, no gravity.
                camera.position[0] += self.velocity[0] * dt;
                camera.position[1] += self.velocity[1] * dt;
                camera.position[2] += self.velocity[2] * dt;
                camera.desired_move = [0.0; 3];
                camera.jump_requested = false;
            } else {
                // soft containment: pull the camera back inside the bounds box.
                // PhysicsSystem owns the position, so this is a one-frame-lagged
                // correction applied before it runs.
                let r = self.player_radius;
                camera.position[0] =
                    camera.position[0].clamp(self.bounds_min[0] + r, self.bounds_max[0] - r);
                camera.position[2] =
                    camera.position[2].clamp(self.bounds_min[2] + r, self.bounds_max[2] - r);

                // hand the movement intent to PhysicsSystem, which resolves it
                // against the world and writes the final camera position back
                camera.desired_move = self.velocity;
                camera.jump_requested = input.jump;
            }
            camera.interact_requested = input.interact;

            // write the view matrix as a fallback for worlds with no
            // PhysicsSystem; PhysicsSystem overwrites it once it has moved.
            camera.view_matrix =
                crate::gfx::camera::view_matrix(camera.position, camera.yaw, camera.pitch);
        }

        // interactable props: press the interact key while facing one to
        // rotate it 45 degrees. Pickup/drop is handled by PhysicsSystem.
        if input.interact && !self.interactable_indices.is_empty() {
            let (cam_pos, cam_yaw) = ctx
                .query::<Camera3D>()
                .next()
                .map(|c| (c.position, c.yaw))
                .unwrap_or(([0.0; 3], 0.0));
            let fwd = [-cam_yaw.sin(), 0.0_f32, -cam_yaw.cos()];

            // nearest interactable prop within reach the player is facing
            let mut best: Option<(f32, usize)> = None;
            {
                let props: Vec<&Prop> = ctx.query::<Prop>().collect();
                for &prop_idx in &self.interactable_indices {
                    if let Some(prop) = props.get(prop_idx) {
                        let dx = prop.position[0] - cam_pos[0];
                        let dz = prop.position[2] - cam_pos[2];
                        let dist = (dx * dx + dz * dz).sqrt();
                        if dist < INTERACT_REACH && dist > 0.0 {
                            let dot = (fwd[0] * dx + fwd[2] * dz) / dist;
                            if dot > INTERACT_MIN_DOT && best.is_none_or(|(d, _)| dist < d) {
                                best = Some((dist, prop_idx));
                            }
                        }
                    }
                }
            }

            if let Some((_, prop_idx)) = best {
                let mut props: Vec<&mut Prop> = ctx.query_mut::<Prop>().collect();
                if let Some(prop) = props.get_mut(prop_idx) {
                    prop.rotation_deg[1] = (prop.rotation_deg[1] + 45.0) % 360.0;
                    tracing::info!(
                        "interacted with prop {}, yaw now {:.0}\u{00b0}",
                        prop.asset_id,
                        prop.rotation_deg[1],
                    );
                }
            }
        }

        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use crate::assets::{Camera3D, CameraController};
    use crate::ecs::World;

    fn camera(controller: Option<CameraController>) -> Camera3D {
        Camera3D {
            fov_y_degrees: 75.0,
            near: 0.05,
            far: 200.0,
            view_matrix: [[0.0; 4]; 4],
            position: [0.0; 3],
            yaw: 0.0,
            pitch: 0.0,
            desired_move: [0.0; 3],
            jump_requested: false,
            interact_requested: false,
            controller,
        }
    }

    // A Camera3D whose `controller` is set spawns the internal controller.
    #[test]
    fn controlled_camera_spawns_internal_system() {
        let mut world = World::new_empty();
        world.add_component(camera(Some(CameraController::default())));
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["Camera3DSystem"]);
    }

    // `controller: null` opts out: a cutscene camera gets no controller.
    #[test]
    fn uncontrolled_camera_has_no_system() {
        let mut world = World::new_empty();
        world.add_component(camera(None));
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }

    // A ControlsCommand pushed mid-tick updates the live mouse sensitivity, so
    // the same frame's mouse-look uses the new value (not the init-time one).
    // This is the settings-menu sensitivity slider applying without a restart.
    #[test]
    fn controls_command_updates_sensitivity_live() {
        use crate::assets::{ControlsCommand, FrameInput};

        let mut world = World::new_empty();
        // Free-fly avoids the PhysicsSystem path; start from a known sensitivity.
        let ctrl = CameraController {
            free_fly: true,
            mouse_sensitivity: 0.001,
            ..CameraController::default()
        };
        world.add_component(camera(Some(ctrl)));
        world.start().unwrap();

        // GraphicsSystem would send this when the slider is dragged; the camera
        // reads it this tick. A mouse delta in the same frame must rotate by the
        // NEW sensitivity (0.005), not the controller's 0.001.
        world.events_mut::<ControlsCommand>().send(ControlsCommand {
            mouse_sensitivity: 0.005,
        });
        world.add_component(FrameInput {
            mouse_dx: 10.0,
            ..Default::default()
        });
        world.step();

        let yaw = world.query::<Camera3D>().next().map(|c| c.yaw).unwrap();
        assert!(
            (yaw - (-10.0 * 0.005)).abs() < 1.0e-6,
            "yaw {yaw} should reflect the live sensitivity 0.005"
        );
    }

    // With both a controlled camera and a UiInputSystem (View + KeyBinding),
    // both systems read the same per-frame FrameInput: Camera3DSystem runs
    // first but no longer consumes it, so UiInputSystem still receives Escape
    // and toggles the menu. (Regression: Camera3DSystem drained the input,
    // starving the menu, so Escape did nothing over a captured camera.)
    #[test]
    fn camera_and_ui_share_frame_input() {
        use crate::assets::{FrameInput, KeyBinding, View, ViewCommand};
        use crate::ecs::asset_id::AssetId;

        let mut world = World::new_empty();
        world.add_component(camera(Some(CameraController::default())));
        world.add_component(View {
            asset_id: AssetId(50),
            initial: false,
            fade_in_secs: 0.0,
        });
        world.add_component(KeyBinding {
            key: "Escape".to_string(),
            action: "view:toggle:50".to_string(),
        });
        world.start().unwrap();

        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert!(names.contains(&"Camera3DSystem"));
        assert!(names.contains(&"UiInputSystem"));

        world.add_component(FrameInput {
            escape: true,
            ..Default::default()
        });
        world.step();

        let cmd = world.query::<ViewCommand>().next().cloned();
        assert!(
            matches!(cmd, Some(ViewCommand::Toggle(AssetId(50)))),
            "UiInputSystem must still process Escape when a camera is present"
        );
    }
}
