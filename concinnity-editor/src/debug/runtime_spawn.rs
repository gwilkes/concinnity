// src/debug/runtime_spawn.rs
//
// Runtime decal / emitter / screenshot spawn queue + dispatch (`cn debug`
// only). Both halves live here so the library never compiles them:
//
//   queue     a process-wide command queue the debug WS handlers push onto
//             (`enqueue`) and the per-frame debug drive drains (`drain`).
//   dispatch  `dispatch_runtime_spawn`, run by `DebugServer::drive_hot_reload`
//             against the live backend + the init-captured texture-name table.
//
// The WS server pushes commands off the engine thread; the drive applies them
// at frame start on the main thread. Each command carries a reply channel so
// the WS handler can hand the new stable slot index back to its client
// synchronously: the wait is bounded by one frame (~16 ms at 60 Hz). `cn run`
// has no debug hook and never reaches any of this.

use std::sync::Mutex;

use crate::gfx::graphics_system::WorldReloadState;

// A runtime decal-spawn request. `texture` is the world.jsonl name of the
// Texture asset to project; `None` (or an unresolvable name) falls back to
// the renderer's white slot 0 so the tint still stamps. Geometry is the
// same TRS triple the [`crate::assets::Decal`] component carries.
#[derive(Debug, Clone)]
pub(crate) struct DecalSpawnArgs {
    pub texture: Option<String>,
    pub position: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub size: [f32; 3],
    pub tint: [f32; 4],
}

impl Default for DecalSpawnArgs {
    fn default() -> Self {
        Self {
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

// A runtime emitter-spawn request. Same field shape as the
// [`crate::assets::ParticleEmitter`] asset; the engine clamps + normalises
// via [`crate::gfx::particles::build_particle_records`].
#[derive(Debug, Clone)]
pub(crate) struct EmitterSpawnArgs {
    pub texture: Option<String>,
    pub position: [f32; 3],
    pub direction: [f32; 3],
    pub spread_deg: f32,
    pub speed_min: f32,
    pub speed_max: f32,
    pub lifetime_min: f32,
    pub lifetime_max: f32,
    pub gravity: [f32; 3],
    pub spawn_rate: f32,
    pub max_particles: u32,
    pub size_start: f32,
    pub size_end: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
}

impl Default for EmitterSpawnArgs {
    fn default() -> Self {
        Self {
            texture: None,
            position: [0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0],
            spread_deg: 15.0,
            speed_min: 1.0,
            speed_max: 2.0,
            lifetime_min: 1.0,
            lifetime_max: 2.0,
            gravity: [0.0, -9.8, 0.0],
            spawn_rate: 32.0,
            max_particles: 256,
            size_start: 0.2,
            size_end: 0.05,
            color_start: [1.0, 1.0, 1.0, 1.0],
            color_end: [1.0, 1.0, 1.0, 0.0],
        }
    }
}

// A runtime camera-set request: a new pose for the active `Camera3D`. `yaw` /
// `pitch` are radians (the controller's own convention); `fov_y_degrees` is
// `None` to leave the field untouched. Applied against the live ECS, not the
// backend, so it carries no texture/slot fields.
#[derive(Debug, Clone)]
pub(crate) struct CameraSetArgs {
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y_degrees: Option<f32>,
}

impl Default for CameraSetArgs {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y_degrees: None,
        }
    }
}

// A runtime camera-move request: a per-frame pose delta applied to the active
// `Camera3D` for a span of frames, so the renderer sees sustained motion (TAA
// ghosting, SSGI temporal noise, motion blur) that a one-shot `camera-set`
// teleport never produces. `forward` / `right` / `up` are per-frame position
// offsets (world units) along the free-fly look basis; `yaw` / `pitch` are
// per-frame radian deltas. `frames == 0` holds the motion indefinitely until a
// `camera-stop` command clears it; `frames > 0` applies it for exactly that
// many frames then auto-stops.
#[derive(Debug, Clone)]
pub(crate) struct CameraMoveArgs {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub frames: u32,
}

impl Default for CameraMoveArgs {
    fn default() -> Self {
        Self {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames: 0,
        }
    }
}

// An in-progress camera-move the per-frame debug drive applies to the active
// `Camera3D` each tick. Built from [`CameraMoveArgs`] when a `camera-move`
// command is drained; held on the `DebugServer` (main-thread only) and
// advanced once per frame until exhausted or a `camera-stop` clears it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CameraMotion {
    pub forward: f32,
    pub right: f32,
    pub up: f32,
    pub yaw: f32,
    pub pitch: f32,
    // Frames still to apply. `None` is an indefinite hold (cleared only by
    // `camera-stop`); `Some(n)` counts down to the auto-stop.
    pub frames_left: Option<u32>,
}

impl CameraMotion {
    // Build a motion from a drained `camera-move` request. `frames == 0` maps
    // to an indefinite hold.
    pub fn from_args(args: &CameraMoveArgs) -> Self {
        Self {
            forward: args.forward,
            right: args.right,
            up: args.up,
            yaw: args.yaw,
            pitch: args.pitch,
            frames_left: if args.frames == 0 {
                None
            } else {
                Some(args.frames)
            },
        }
    }

    // The motion to apply next frame, after one step has just been applied: a
    // finite countdown decremented by one (returning `None` once exhausted),
    // an indefinite hold returning itself unchanged.
    pub fn advanced(self) -> Option<Self> {
        match self.frames_left {
            None => Some(self),
            Some(n) if n > 1 => Some(Self {
                frames_left: Some(n - 1),
                ..self
            }),
            Some(_) => None,
        }
    }
}

// Compute the pose after applying one camera-move step to a free-fly camera at
// the given pose. `forward` / `right` follow the look-direction basis (matching
// `Camera3DSystem`'s free-fly mode); `up` is world up. Pitch is clamped to the
// same near-vertical limit the controller uses so a sustained pitch delta can
// not flip the camera over.
pub(crate) fn advance_pose(
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    motion: &CameraMotion,
) -> ([f32; 3], f32, f32) {
    let cp = pitch.cos();
    let fwd = [-yaw.sin() * cp, pitch.sin(), -yaw.cos() * cp];
    let right = [yaw.cos(), 0.0, -yaw.sin()];
    let new_pos = [
        position[0] + fwd[0] * motion.forward + right[0] * motion.right,
        position[1] + fwd[1] * motion.forward + motion.up,
        position[2] + fwd[2] * motion.forward + right[2] * motion.right,
    ];
    let new_yaw = yaw + motion.yaw;
    let new_pitch = (pitch + motion.pitch).clamp(
        -std::f32::consts::FRAC_PI_2 + 0.01,
        std::f32::consts::FRAC_PI_2 - 0.01,
    );
    (new_pos, new_yaw, new_pitch)
}

// One runtime spawn / despawn command pushed onto [`enqueue`] by the debug
// WS server and drained by the per-frame debug drive. Each variant carries a
// `std::sync::mpsc::SyncSender` reply channel so the WS handler can block
// (with timeout) on the result and hand a JSON reply back to its client.
pub(crate) enum RuntimeCommand {
    DecalAdd {
        args: DecalSpawnArgs,
        reply: std::sync::mpsc::SyncSender<Result<usize, String>>,
    },
    DecalRemove {
        id: usize,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    EmitterAdd {
        args: EmitterSpawnArgs,
        reply: std::sync::mpsc::SyncSender<Result<usize, String>>,
    },
    EmitterRemove {
        id: usize,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Capture the last presented frame to a PNG at `path`; the reply carries the
    // saved path. Routed to `RenderBackend::screenshot` on the render thread.
    Screenshot {
        path: String,
        reply: std::sync::mpsc::SyncSender<Result<String, String>>,
    },
    // Teleport the active `Camera3D` to a new pose. Applied against the ECS by
    // `apply_camera_set`, not the backend, so the per-frame drive routes it to
    // `dispatch_camera_set` (which holds the `World`) rather than
    // `dispatch_runtime_spawn`.
    CameraSet {
        args: CameraSetArgs,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Install a sustained camera-move motion on the active `Camera3D`. Like
    // `CameraSet` it mutates the ECS, so the per-frame drive partitions it out
    // and installs it on the `DebugServer` rather than touching the backend.
    // The reply fires as soon as the motion is accepted (a Camera3D exists),
    // not when it finishes, so a long move never outlasts the WS timeout.
    CameraMove {
        args: CameraMoveArgs,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Clear any in-progress camera-move motion. Also ECS-side (it clears the
    // `DebugServer`'s motion slot), so the per-frame drive routes it like
    // `CameraSet` / `CameraMove`.
    CameraStop {
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Toggle a Quality-group graphics setting (taa / ssao / ssr / ssgi /
    // auto_exposure) live by pushing the same `SettingCommand` the settings
    // menu emits. Like `CameraSet` it mutates the ECS (not the backend
    // directly), so the per-frame drive partitions it out and applies it once
    // the `systems_mut` borrow ends, via `dispatch_quality_set`.
    QualitySet {
        setting: String,
        op: crate::assets::SettingOp,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
    // Bind a movement action (`key_forward` / ... ) to a key, live, by pushing
    // the same `Rebind` `SettingCommand` the settings menu emits. Like
    // `QualitySet` it mutates the ECS, so the per-frame drive routes it to
    // `dispatch_rebind` once the `systems_mut` borrow ends.
    Rebind {
        setting: String,
        key: crate::assets::Key,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
}

static QUEUE: Mutex<Vec<RuntimeCommand>> = Mutex::new(Vec::new());

// Push a command onto the runtime-spawn queue. Returns immediately; the
// caller blocks on its own reply receiver to get the result. A poisoned
// mutex is recovered and used regardless (an unrelated panic in another
// thread must not silently drop spawn commands).
pub(crate) fn enqueue(cmd: RuntimeCommand) {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    q.push(cmd);
}

// Take every queued command. Called by the `cn debug` drive
// (`DebugHook::tick`) at frame start. The returned `Vec` is the live list:
// the queue is reset to empty.
pub(crate) fn drain() -> Vec<RuntimeCommand> {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    std::mem::take(&mut *q)
}

// Process one runtime-spawn command (drained from the debug WS queue)
// against the live backend. Resolves texture-name strings via the init-time
// interner snapshot + `world_reload.texture_name_to_slot` before building
// the backend record, and sends the result back via the command's reply
// channel. Reply-channel send failures are silently dropped: the WS thread
// may have already given up waiting (e.g. its client disconnected), and
// that is not a renderer error.
pub(crate) fn dispatch_runtime_spawn(
    cmd: RuntimeCommand,
    world_reload: Option<&WorldReloadState>,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) {
    match cmd {
        RuntimeCommand::DecalAdd { args, reply } => {
            let result = resolve_texture_slot(args.texture.as_deref(), world_reload)
                .and_then(|slot| {
                    let model = crate::gfx::decal::decal_model_matrix(
                        args.position,
                        args.rotation_deg,
                        args.size,
                    );
                    let inv_model = crate::gfx::decal::invert_decal_model(model)
                        .ok_or_else(|| "decal-add: degenerate size".to_string())?;
                    Ok(crate::gfx::decal::DecalRecord {
                        model,
                        inv_model,
                        texture_slot: slot,
                        tint: args.tint,
                    })
                })
                .and_then(|rec| backend.add_decal(rec));
            let _ = reply.send(result);
        }
        RuntimeCommand::DecalRemove { id, reply } => {
            let _ = reply.send(backend.remove_decal(id));
        }
        RuntimeCommand::EmitterAdd { args, reply } => {
            let result = resolve_texture_slot(args.texture.as_deref(), world_reload)
                .map(|slot| {
                    // Mirror the clamp / normalise rules used by
                    // `build_particle_records` so a WS-spawned emitter and
                    // an authored one behave identically. We do not call
                    // that helper directly because it takes a
                    // `&[&ParticleEmitter]` and a texture-id-keyed map;
                    // here we already have a resolved slot.
                    let dir = {
                        let d = args.direction;
                        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if !len.is_finite() || len < 1e-6 {
                            [0.0, 1.0, 0.0]
                        } else {
                            [d[0] / len, d[1] / len, d[2] / len]
                        }
                    };
                    let spread_cos = args.spread_deg.clamp(0.0, 180.0).to_radians().cos();
                    let lifetime_min = args.lifetime_min.max(0.001);
                    let lifetime_max = args.lifetime_max.max(lifetime_min);
                    let speed_min = args.speed_min.max(0.0);
                    let speed_max = args.speed_max.max(speed_min);
                    let max_particles = args
                        .max_particles
                        .clamp(1, crate::gfx::particles::MAX_PARTICLES_PER_EMITTER);
                    crate::gfx::particles::ParticleEmitterRecord {
                        texture_slot: slot,
                        position: args.position,
                        direction: dir,
                        spread_cos,
                        speed_min,
                        speed_max,
                        lifetime_min,
                        lifetime_max,
                        gravity: args.gravity,
                        spawn_rate: args.spawn_rate.max(0.0),
                        max_particles,
                        size_start: args.size_start.max(0.0),
                        size_end: args.size_end.max(0.0),
                        color_start: args.color_start,
                        color_end: args.color_end,
                    }
                })
                .and_then(|rec| backend.add_emitter(rec));
            let _ = reply.send(result);
        }
        RuntimeCommand::EmitterRemove { id, reply } => {
            let _ = reply.send(backend.remove_emitter(id));
        }
        RuntimeCommand::Screenshot { path, reply } => {
            let _ = reply.send(backend.screenshot(&path));
        }
        RuntimeCommand::CameraSet { reply, .. } => {
            // CameraSet mutates the ECS, not the backend; the per-frame drive
            // partitions it out and routes it to `dispatch_camera_set`. Reaching
            // here means a future caller misrouted it.
            let _ = reply.send(Err("camera-set: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::CameraMove { reply, .. } => {
            // ECS-side like CameraSet; the per-frame drive installs it on the
            // DebugServer. Reaching the backend dispatch means a misroute.
            let _ = reply.send(Err("camera-move: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::CameraStop { reply } => {
            let _ = reply.send(Err("camera-stop: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::QualitySet { reply, .. } => {
            // ECS-side like CameraSet; the per-frame drive routes it to
            // `dispatch_quality_set`. Reaching the backend dispatch is a misroute.
            let _ = reply.send(Err("quality-set: misrouted to backend dispatch".to_string()));
        }
        RuntimeCommand::Rebind { reply, .. } => {
            // ECS-side like QualitySet; routed to `dispatch_rebind`.
            let _ = reply.send(Err("rebind: misrouted to backend dispatch".to_string()));
        }
    }
}

// Apply a drained `CameraSet` command against the live ECS and reply. Routed
// here (instead of `dispatch_runtime_spawn`) by the per-frame debug drive
// because it needs the `World`, not the backend. Non-`CameraSet` variants are
// ignored: the caller only routes `CameraSet` here.
pub(crate) fn dispatch_camera_set(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::CameraSet { args, reply } = cmd else {
        return;
    };
    let _ = reply.send(apply_camera_set(&args, world));
}

// Apply a drained `QualitySet` command by sending a `SettingCommand` into the
// ECS, exactly as `UiInputSystem` does for a settings-menu toggle. The
// `GraphicsSystem` reads it on its next step and applies the change live
// (`apply_quality_settings`), so this exercises the real toggle path rather
// than a duplicate. Routed here (like `CameraSet`) because it mutates the ECS,
// not the backend. `cn debug` only.
pub(crate) fn dispatch_quality_set(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::QualitySet { setting, op, reply } = cmd else {
        return;
    };
    world
        .events_mut::<crate::assets::SettingCommand>()
        .send(crate::assets::SettingCommand {
            setting,
            op,
            value_label: None,
            persist: true,
        });
    let _ = reply.send(Ok(()));
}

// Apply a drained `Rebind` command by sending a `Rebind` `SettingCommand` into
// the ECS, exactly as `UiInputSystem` does after a capture. `GraphicsSystem`
// reads it on its next step and applies the rebind live (swap + `set_keymap` +
// persist + label refresh via its registry, which is why `value_label` is left
// `None` here). Routed here (like `QualitySet`) because it mutates the ECS.
pub(crate) fn dispatch_rebind(cmd: RuntimeCommand, world: &mut crate::ecs::World) {
    let RuntimeCommand::Rebind {
        setting,
        key,
        reply,
    } = cmd
    else {
        return;
    };
    world
        .events_mut::<crate::assets::SettingCommand>()
        .send(crate::assets::SettingCommand {
            setting,
            op: crate::assets::SettingOp::Rebind(key),
            value_label: None,
            persist: true,
        });
    let _ = reply.send(Ok(()));
}

// Write a new pose onto the active `Camera3D` and zero the controller velocity.
// The free-fly controller integrates a smoothed velocity onto the camera
// position every step, so a leftover velocity (or held key) would drift the
// teleport away on the next step; zeroing it makes the new pose hold. The
// debug tick runs before the world step, so the controller sees the new pose
// the same frame, and with velocity zeroed (and no input in an unfocused
// window) it leaves it untouched.
pub(crate) fn apply_camera_set(
    args: &CameraSetArgs,
    world: &mut crate::ecs::World,
) -> Result<(), String> {
    use crate::assets::Camera3D;
    let Some(camera) = world.query_mut::<Camera3D>().next() else {
        return Err("camera-set: no Camera3D in world".to_string());
    };
    camera.position = args.position;
    camera.yaw = args.yaw;
    camera.pitch = args.pitch;
    if let Some(fov) = args.fov_y_degrees {
        camera.fov_y_degrees = fov;
    }
    camera.view_matrix = crate::gfx::camera::view_matrix(camera.position, camera.yaw, camera.pitch);

    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::Camera3DSystem(c) = system {
            c.reset_velocity();
        }
    }
    Ok(())
}

// Apply one camera-move step against the live ECS: advance the active
// `Camera3D`'s pose by the motion deltas, refresh its view matrix, and zero the
// controller velocity (same reason as `apply_camera_set`: keep free-fly from
// fighting the externally driven pose). Returns `false` when the world has no
// `Camera3D`, so the caller drops the motion instead of spinning forever.
pub(crate) fn apply_camera_move_step(motion: &CameraMotion, world: &mut crate::ecs::World) -> bool {
    use crate::assets::Camera3D;
    let Some(camera) = world.query_mut::<Camera3D>().next() else {
        return false;
    };
    let (pos, yaw, pitch) = advance_pose(camera.position, camera.yaw, camera.pitch, motion);
    camera.position = pos;
    camera.yaw = yaw;
    camera.pitch = pitch;
    camera.view_matrix = crate::gfx::camera::view_matrix(pos, yaw, pitch);

    for system in world.systems_mut() {
        if let crate::ecs::SystemAsset::Camera3DSystem(c) = system {
            c.reset_velocity();
        }
    }
    true
}

// Resolve an optional Texture asset name to its pool slot index. `None` (no
// texture authored on the spawn request) maps to slot 0 (the renderer's
// white fallback) so the tint / colour gradient still stamps. An unknown
// name returns `Err`; the WS client gets a clear error rather than a
// silent fallback. Texture-name resolution leans on the init-time
// `world_reload.texture_name_to_slot` snapshot, so it only succeeds under
// `cn debug` worlds: that matches the current runtime-spawn use case
// (debug WS, headless tests).
fn resolve_texture_slot(
    texture: Option<&str>,
    world_reload: Option<&WorldReloadState>,
) -> Result<usize, String> {
    let Some(name) = texture else {
        return Ok(0);
    };
    let table = crate::ecs::asset_id::name_table();
    let id = table
        .iter()
        .position(|n| n == name)
        .map(|i| crate::ecs::asset_id::AssetId(i as u32))
        .ok_or_else(|| format!("texture '{}' not found in interner", name))?;
    let reload = world_reload.ok_or_else(|| {
        "texture-name resolution requires cn debug (world_reload missing)".to_string()
    })?;
    reload
        .texture_name_to_slot
        .get(&id)
        .copied()
        .ok_or_else(|| format!("texture '{}' is not in the live texture pool", name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_drain_round_trip() {
        // Drain any leftovers from earlier tests in this process.
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::DecalRemove { id: 7, reply: tx });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        match cmds.into_iter().next().unwrap() {
            RuntimeCommand::DecalRemove { id, .. } => assert_eq!(id, 7),
            _ => panic!("wrong variant"),
        }
        // Second drain is empty.
        assert!(drain().is_empty());
    }

    #[test]
    fn decal_spawn_args_defaults() {
        let a = DecalSpawnArgs::default();
        assert_eq!(a.size, [1.0, 1.0, 1.0]);
        assert_eq!(a.tint, [1.0, 1.0, 1.0, 1.0]);
        assert!(a.texture.is_none());
    }

    #[test]
    fn emitter_spawn_args_defaults() {
        let a = EmitterSpawnArgs::default();
        assert_eq!(a.direction, [0.0, 1.0, 0.0]);
        assert_eq!(a.max_particles, 256);
        assert!((a.spread_deg - 15.0).abs() < 1e-6);
    }

    #[test]
    fn camera_set_args_defaults() {
        let a = CameraSetArgs::default();
        assert_eq!(a.position, [0.0, 0.0, 0.0]);
        assert_eq!(a.yaw, 0.0);
        assert_eq!(a.pitch, 0.0);
        assert!(a.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_enqueue_drain_round_trip() {
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::CameraSet {
            args: CameraSetArgs {
                position: [1.0, 2.0, 3.0],
                yaw: 0.5,
                pitch: -0.25,
                fov_y_degrees: Some(60.0),
            },
            reply: tx,
        });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        match cmds.into_iter().next().unwrap() {
            RuntimeCommand::CameraSet { args, .. } => {
                assert_eq!(args.position, [1.0, 2.0, 3.0]);
                assert_eq!(args.fov_y_degrees, Some(60.0));
            }
            _ => panic!("wrong variant"),
        }
        assert!(drain().is_empty());
    }

    // A world with a controlled Camera3D builds a Camera3DSystem at `start`;
    // `apply_camera_set` must write the pose, refresh the view matrix, and
    // succeed (the velocity reset runs over the constructed system).
    #[test]
    fn apply_camera_set_writes_active_camera() {
        use crate::assets::{Camera3D, CameraController};
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(Camera3D {
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
            controller: Some(CameraController::default()),
        });
        world.start().unwrap();

        let args = CameraSetArgs {
            position: [10.0, 20.0, 30.0],
            yaw: 1.0,
            pitch: -0.5,
            fov_y_degrees: Some(50.0),
        };
        assert!(apply_camera_set(&args, &mut world).is_ok());

        let cam = world.query::<Camera3D>().next().expect("camera present");
        assert_eq!(cam.position, [10.0, 20.0, 30.0]);
        assert_eq!(cam.yaw, 1.0);
        assert_eq!(cam.pitch, -0.5);
        assert_eq!(cam.fov_y_degrees, 50.0);
        // view_matrix was refreshed from the new pose, no longer all-zero.
        assert_ne!(cam.view_matrix, [[0.0; 4]; 4]);
    }

    // `fov_y_degrees: None` leaves the existing field untouched.
    #[test]
    fn apply_camera_set_keeps_fov_when_none() {
        use crate::assets::{Camera3D, CameraController};
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(Camera3D {
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
            controller: Some(CameraController::default()),
        });
        world.start().unwrap();

        let args = CameraSetArgs {
            position: [1.0, 1.0, 1.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y_degrees: None,
        };
        assert!(apply_camera_set(&args, &mut world).is_ok());
        let cam = world.query::<Camera3D>().next().expect("camera present");
        assert_eq!(cam.fov_y_degrees, 75.0);
    }

    // No Camera3D in the world is a clean error, not a panic.
    #[test]
    fn apply_camera_set_errors_without_camera() {
        let mut world = crate::ecs::World::new_empty();
        let args = CameraSetArgs::default();
        assert!(apply_camera_set(&args, &mut world).is_err());
    }

    #[test]
    fn camera_move_args_defaults_are_zero_hold() {
        let a = CameraMoveArgs::default();
        assert_eq!(a.forward, 0.0);
        assert_eq!(a.right, 0.0);
        assert_eq!(a.up, 0.0);
        assert_eq!(a.yaw, 0.0);
        assert_eq!(a.pitch, 0.0);
        // frames == 0 is the indefinite-hold sentinel.
        assert_eq!(a.frames, 0);
    }

    #[test]
    fn camera_motion_from_args_maps_zero_frames_to_indefinite_hold() {
        let hold = CameraMotion::from_args(&CameraMoveArgs {
            forward: 1.0,
            frames: 0,
            ..CameraMoveArgs::default()
        });
        assert_eq!(hold.frames_left, None);
        let finite = CameraMotion::from_args(&CameraMoveArgs {
            forward: 1.0,
            frames: 5,
            ..CameraMoveArgs::default()
        });
        assert_eq!(finite.frames_left, Some(5));
    }

    #[test]
    fn camera_motion_advanced_counts_down_then_stops() {
        let m = CameraMotion {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: Some(3),
        };
        // Three frames remain: countdown 3 -> 2 -> 1 -> exhausted.
        let m = m.advanced().expect("2 frames remain");
        assert_eq!(m.frames_left, Some(2));
        let m = m.advanced().expect("1 frame remains");
        assert_eq!(m.frames_left, Some(1));
        assert!(m.advanced().is_none(), "last frame exhausts the motion");
    }

    #[test]
    fn camera_motion_advanced_holds_indefinitely() {
        let m = CameraMotion {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.1,
            pitch: 0.0,
            frames_left: None,
        };
        // An indefinite hold returns itself unchanged forever.
        let next = m.clone().advanced().expect("hold never exhausts");
        assert_eq!(next, m);
    }

    #[test]
    fn advance_pose_moves_along_look_basis() {
        // yaw = 0, pitch = 0 looks down -Z; forward delta moves -Z, right moves
        // +X, up moves +Y.
        let motion = CameraMotion {
            forward: 2.0,
            right: 3.0,
            up: 4.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: Some(1),
        };
        let (pos, yaw, pitch) = advance_pose([0.0, 0.0, 0.0], 0.0, 0.0, &motion);
        assert!((pos[0] - 3.0).abs() < 1e-5, "right -> +X");
        assert!((pos[1] - 4.0).abs() < 1e-5, "up -> +Y");
        assert!((pos[2] + 2.0).abs() < 1e-5, "forward -> -Z");
        assert_eq!(yaw, 0.0);
        assert_eq!(pitch, 0.0);
    }

    #[test]
    fn advance_pose_accumulates_yaw_and_clamps_pitch() {
        let motion = CameraMotion {
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.5,
            // A huge pitch delta must clamp, not flip the camera over.
            pitch: 100.0,
            frames_left: Some(1),
        };
        let (_, yaw, pitch) = advance_pose([0.0, 0.0, 0.0], 1.0, 0.0, &motion);
        assert!((yaw - 1.5).abs() < 1e-6);
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        assert!(
            (pitch - limit).abs() < 1e-5,
            "pitch clamps to near-vertical"
        );
    }

    #[test]
    fn camera_move_enqueue_drain_round_trip() {
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::CameraMove {
            args: CameraMoveArgs {
                forward: 1.5,
                frames: 30,
                ..CameraMoveArgs::default()
            },
            reply: tx,
        });
        let (stx, _srx) = std::sync::mpsc::sync_channel(1);
        enqueue(RuntimeCommand::CameraStop { reply: stx });
        let cmds = drain();
        assert_eq!(cmds.len(), 2);
        let mut it = cmds.into_iter();
        match it.next().unwrap() {
            RuntimeCommand::CameraMove { args, .. } => {
                assert_eq!(args.forward, 1.5);
                assert_eq!(args.frames, 30);
            }
            _ => panic!("wrong variant"),
        }
        assert!(matches!(
            it.next().unwrap(),
            RuntimeCommand::CameraStop { .. }
        ));
        assert!(drain().is_empty());
    }

    // apply_camera_move_step advances the active camera and refreshes its view
    // matrix; a sequence of steps accumulates displacement (sustained motion).
    #[test]
    fn apply_camera_move_step_advances_active_camera() {
        use crate::assets::{Camera3D, CameraController};
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(Camera3D {
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
            controller: Some(CameraController::default()),
        });
        world.start().unwrap();

        let motion = CameraMotion {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: Some(2),
        };
        assert!(apply_camera_move_step(&motion, &mut world));
        assert!(apply_camera_move_step(&motion, &mut world));

        let cam = world.query::<Camera3D>().next().expect("camera present");
        // Two forward steps of 1.0 along -Z accumulate to -2.0.
        assert!((cam.position[2] + 2.0).abs() < 1e-5);
        assert_ne!(cam.view_matrix, [[0.0; 4]; 4]);
    }

    // No Camera3D: the step is a clean `false`, not a panic, so the drive drops
    // the motion.
    #[test]
    fn apply_camera_move_step_false_without_camera() {
        let mut world = crate::ecs::World::new_empty();
        let motion = CameraMotion {
            forward: 1.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames_left: None,
        };
        assert!(!apply_camera_move_step(&motion, &mut world));
    }
}
