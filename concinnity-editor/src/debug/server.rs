// src/debug/server.rs
//
// The localhost WebSocket debug server: `DebugServer` (the `DebugHook` the run
// loop ticks), the shared `DebugState` snapshot, the accept / per-connection
// threads, and the query-command dispatcher `handle_request`. Spawn/crossfade
// command handlers live in `super::commands`.

use crate::app::DebugHook;
use crate::ecs::{SystemAsset, World};
use crate::gfx::graphics_system::StreamingStats;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use tokio_util::sync::CancellationToken;

use tokio_tungstenite::tungstenite::{Message, accept};

use super::commands::{
    error_reply, handle_anim_crossfade, handle_camera_move, handle_camera_set, handle_camera_stop,
    handle_decal_add, handle_decal_remove, handle_despawn, handle_emitter_add,
    handle_emitter_remove, handle_quality_set, handle_rebind, handle_reparent, handle_screenshot,
    handle_spawn,
};
use super::{hot_reload, runtime_spawn};
// The world snapshot rebuilt by `tick`. The asset/system lists are not cheap
// to rebuild, so they refresh on an interval while `frame` advances every tick.
#[derive(Default)]
struct DebugState {
    frame: u64,
    system_count: usize,
    component_count: usize,
    systems: Vec<String>,
    assets: Vec<AssetEntry>,
    // `AssetId` -> declared name, indexed by id. Captured once after build.
    names: Vec<String>,
    // Asset-streaming `(resident, pending, unloaded)` counts, refreshed every
    // tick (cheap) so the `streaming` command reflects live progress.
    streaming: StreamingStats,
    // Per-system CPU step times (micros) from the last completed frame,
    // refreshed every tick for the `profile` command.
    profile_systems: Vec<(String, u32)>,
    // Render-backend stats from the most recent frame, for `profile`.
    profile_render: crate::gfx::profile::RenderStats,
    // App shutdown token, set once via `DebugHook::attach_shutdown`. The
    // `shutdown` command cancels it to exit the engine cleanly.
    shutdown_token: Option<CancellationToken>,
    // Shared shader-reload flag captured from the active graphics backend.
    // `Some` once `tick` has seen a `GraphicsSystem` whose backend opted into
    // hot-reload (Metal under `cn debug`); `None` otherwise. The
    // `reload-shaders` command flips this to `true` and the backend polls it
    // at the top of `draw_frame`.
    shader_reload: Option<Arc<std::sync::atomic::AtomicBool>>,
    // Shared asset-reload flag captured from `GraphicsSystem`. `Some` once
    // `tick` has seen a `cn debug` world with at least one file-backed
    // `Texture`; `None` otherwise. The `reload-assets` command flips it; the
    // engine consumes it at the top of `GraphicsSystem::step`.
    asset_reload: Option<Arc<std::sync::atomic::AtomicBool>>,
    // Active-camera pose, refreshed every tick for the `camera-get` query.
    // `None` until the first tick that finds a `Camera3D` (a world with no
    // camera never sets it).
    camera: Option<CameraSnapshot>,
}

// A read-only snapshot of the active `Camera3D`, served by `camera-get`. The
// matching `camera-set` mutation lives in `super::commands` /
// `super::runtime_spawn`.
#[derive(Clone, serde::Serialize)]
struct CameraSnapshot {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov_y_degrees: f32,
    near: f32,
    far: f32,
}

// A structural census entry. Per-asset names are not retained at runtime
// (`BlobAssetDef::to_def` drops them), so this carries only kind + type
// discriminant; the `names` table handles the id -> name remap separately.
#[derive(Clone, serde::Serialize)]
struct AssetEntry {
    kind: String,
    discriminant: u8,
}

// How often `tick` rebuilds the asset/system snapshot (in frames). The frame
// counter still advances every tick; only the heavier lists are throttled.
const SNAPSHOT_INTERVAL: u64 = 30;

// A running debug server. Implements `DebugHook`, so the run loop owns it as
// `Box<dyn DebugHook>` and ticks it each frame.
pub struct DebugServer {
    shared: Arc<Mutex<DebugState>>,
    frame: u64,
    // Asset / shader / world.jsonl reload state, built lazily on the first
    // tick that sees a `GraphicsSystem` carrying init-captured sources (i.e.
    // `cn debug` with a file-backed asset / world.jsonl). Owns the filesystem
    // watcher + in-flight decode handles. `None` otherwise: `cn run` never
    // reaches `tick`, and a world with no file-backed asset never builds it.
    hot_reload: Option<hot_reload::AssetHotReloadState>,
    // Active camera-move motion installed by a `camera-move` command, advanced
    // once per frame by `drive_hot_reload` until exhausted or cleared by a
    // `camera-stop`. `None` when no motion is in progress. Main-thread only.
    camera_motion: Option<runtime_spawn::CameraMotion>,
}

impl DebugServer {
    // Bind a localhost WebSocket server on `port` and spawn its accept thread.
    // Binds `127.0.0.1` only: the debug surface is never exposed off-box.
    pub fn start(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let shared = Arc::new(Mutex::new(DebugState::default()));

        let shared_for_thread = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("debug-server".to_string())
            .spawn(move || serve(listener, shared_for_thread))?;

        tracing::info!("debug server listening on ws://127.0.0.1:{port}");
        Ok(Self {
            shared,
            frame: 0,
            hot_reload: None,
            camera_motion: None,
        })
    }
}

impl DebugServer {
    // Run the asset / shader / world.jsonl hot-reload passes once per frame and
    // apply their ECS side-effects. `cn debug` only: the matching drive used
    // to sit at the top of `GraphicsSystem::run_step` / `AnimationSystem::step`.
    // The reload state is built lazily from the `GraphicsSystem`'s init-captured
    // sources on the first tick that finds them, then driven against the
    // backend + Prop-tracking handle (`hot_reload_apply_parts`). The passes
    // return ECS edits (skeleton-shape changes + added Props) applied here, once
    // the system borrow is released. A world with no captured sources never
    // builds `self.hot_reload` and this stays a cheap no-op.
    fn drive_hot_reload(&mut self, world: &mut World) {
        let mut effects = None;
        // ECS-side commands (camera-set / camera-move / camera-stop, plus
        // quality-set) mutate the ECS or this server's motion slot, not the
        // backend, so they cannot be applied inside the `systems_mut` borrow
        // below. Collect them here and apply them once that borrow ends.
        let mut deferred_ecs_cmds: Vec<runtime_spawn::RuntimeCommand> = Vec::new();
        for system in world.systems_mut() {
            match system {
                SystemAsset::GraphicsSystem(gs) => {
                    // Lazily build the reload state from the init-captured
                    // sources (must precede the apply-parts borrow of `gs`).
                    if self.hot_reload.is_none()
                        && let Some(sources) = gs.take_hot_reload_sources()
                    {
                        self.hot_reload =
                            Some(hot_reload::AssetHotReloadState::from_sources(sources));
                    }
                    if let Some(mut apply) = gs.hot_reload_apply_parts() {
                        // Runtime decal / emitter spawn: independent of the
                        // hot-reload state, available in any `cn debug` world.
                        // CameraSet is deferred; everything else hits the
                        // backend now.
                        for cmd in runtime_spawn::drain() {
                            if matches!(
                                cmd,
                                runtime_spawn::RuntimeCommand::CameraSet { .. }
                                    | runtime_spawn::RuntimeCommand::CameraMove { .. }
                                    | runtime_spawn::RuntimeCommand::CameraStop { .. }
                                    | runtime_spawn::RuntimeCommand::QualitySet { .. }
                                    | runtime_spawn::RuntimeCommand::Rebind { .. }
                                    | runtime_spawn::RuntimeCommand::Despawn { .. }
                                    | runtime_spawn::RuntimeCommand::Reparent { .. }
                                    | runtime_spawn::RuntimeCommand::Spawn { .. }
                            ) {
                                deferred_ecs_cmds.push(cmd);
                            } else {
                                runtime_spawn::dispatch_runtime_spawn(
                                    cmd,
                                    apply.world_reload.as_ref(),
                                    apply.backend,
                                );
                            }
                        }
                        // Asset / shader / world.jsonl reload passes, only when
                        // the reload state was armed at init.
                        if let Some(state) = self.hot_reload.as_mut() {
                            effects = Some(hot_reload::run_frame(state, &mut apply));
                        }
                    }
                }
                SystemAsset::AnimationSystem(anim) => {
                    crate::anim_reload::reload_clips_if_pending(anim);
                    anim.apply_crossfade_commands();
                }
                _ => {}
            }
        }

        // Apply deferred ECS commands now the `systems_mut` borrow is
        // released. tick() runs before the world step, so the Camera3DSystem
        // step this frame sees the new pose; the velocity reset inside keeps
        // free-fly from drifting it. camera-move / camera-stop install or clear
        // the motion slot; the actual per-frame advance happens just below so a
        // freshly installed motion also steps this same frame. quality-set
        // sends a `SettingCommand` the GraphicsSystem reads on its next step.
        for cmd in deferred_ecs_cmds {
            match cmd {
                runtime_spawn::RuntimeCommand::CameraSet { .. } => {
                    runtime_spawn::dispatch_camera_set(cmd, world);
                }
                runtime_spawn::RuntimeCommand::QualitySet { .. } => {
                    runtime_spawn::dispatch_quality_set(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Rebind { .. } => {
                    runtime_spawn::dispatch_rebind(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Despawn { .. } => {
                    runtime_spawn::dispatch_despawn(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Reparent { .. } => {
                    runtime_spawn::dispatch_reparent(cmd, world);
                }
                runtime_spawn::RuntimeCommand::Spawn { .. } => {
                    runtime_spawn::dispatch_spawn(cmd, world);
                }
                runtime_spawn::RuntimeCommand::CameraMove { args, reply } => {
                    // Accept the motion only when a camera exists, so the client
                    // gets a clean error in a camera-less world. The reply fires
                    // on acceptance, not completion.
                    if world.query::<crate::assets::Camera3D>().next().is_some() {
                        self.camera_motion = Some(runtime_spawn::CameraMotion::from_args(&args));
                        let _ = reply.send(Ok(()));
                    } else {
                        let _ = reply.send(Err("camera-move: no Camera3D in world".to_string()));
                    }
                }
                runtime_spawn::RuntimeCommand::CameraStop { reply } => {
                    self.camera_motion = None;
                    let _ = reply.send(Ok(()));
                }
                // Only ECS-side variants are routed into `deferred_ecs_cmds`.
                _ => {}
            }
        }

        // Advance an in-progress camera-move one step. Runs every frame (before
        // the world step) so the renderer sees sustained motion across temporal
        // passes. A finite motion counts itself down to None; a vanished camera
        // (world swap) drops the motion rather than spinning.
        if let Some(motion) = self.camera_motion.take()
            && runtime_spawn::apply_camera_move_step(&motion, world)
        {
            self.camera_motion = motion.advanced();
        }

        let Some(effects) = effects else {
            return;
        };

        // Splice any skeleton-shape changes into the ECS-owned `SkeletonPose`
        // components so `AnimationSystem` produces right-sized output going
        // forward.
        if !effects.skeleton_updates.is_empty() {
            let index_to_new: std::collections::HashMap<usize, crate::gfx::skinning::Skeleton> =
                effects
                    .skeleton_updates
                    .into_iter()
                    .map(|u| (u.skinned_index, u.new_skeleton))
                    .collect();
            let mut applied = 0usize;
            for pose in world.query_mut::<crate::assets::SkeletonPose>() {
                if let Some(new_skel) = index_to_new.get(&pose.skinned_index) {
                    pose.skeleton = new_skel.clone();
                    pose.joint_matrices = pose.skeleton.bind_skinning_matrices();
                    applied += 1;
                }
            }
            tracing::info!(
                "asset hot-reload: applied skeleton-shape change to {} SkeletonPose component(s)",
                applied
            );
        }
    }
}

impl DebugHook for DebugServer {
    fn tick(&mut self, world: &mut World) {
        self.frame += 1;

        // Drive the asset / shader / world.jsonl hot-reload passes. This is the
        // `cn debug`-only half of the reload machinery that used to run inside
        // `GraphicsSystem::run_step` / `AnimationSystem::step`; it lives here so
        // a `cn run` (no debug hook) never touches it.
        self.drive_hot_reload(world);

        let mut state = match self.shared.lock() {
            Ok(s) => s,
            // A panicked client thread should never take down the engine.
            Err(poisoned) => poisoned.into_inner(),
        };
        state.frame = self.frame;

        // Streaming counts change every frame in the early load-in, so refresh
        // them every tick -- `streaming_stats` is just two small count loops.
        // The same scan opportunistically picks up the shader-reload flag the
        // backend exposes (Some only under `cn debug` on hot-reload backends);
        // once captured, the `reload-shaders` command can fire the flag.
        let mut found_shader_flag = None;
        state.streaming = world
            .systems()
            .iter()
            .find_map(|s| match s {
                SystemAsset::GraphicsSystem(gs) => {
                    if state.shader_reload.is_none() {
                        found_shader_flag = gs.shader_reload_flag();
                    }
                    Some(gs.streaming_stats())
                }
                _ => None,
            })
            .unwrap_or_default();
        if state.shader_reload.is_none()
            && let Some(flag) = found_shader_flag
        {
            state.shader_reload = Some(flag);
        }
        // The asset-reload flag lives on the debug-owned `AssetHotReloadState`
        // (built lazily by `drive_hot_reload` above), not on `GraphicsSystem`.
        // Capture its `pending` Arc so the `reload-assets` command thread can
        // flip it.
        if state.asset_reload.is_none()
            && let Some(h) = self.hot_reload.as_ref()
        {
            state.asset_reload = Some(std::sync::Arc::clone(&h.pending));
        }

        // The profiler snapshot is small (one entry per system + a handful of
        // render counters), so refresh it every tick like the streaming stats.
        let profile = world.profile();
        state.profile_systems = profile
            .system_timings()
            .iter()
            .map(|&(name, micros)| (name.to_string(), micros))
            .collect();
        state.profile_render = profile.render;

        // Active-camera pose for `camera-get`. One component read, so refresh
        // every tick like the streaming / profiler snapshots above.
        state.camera = world
            .query::<crate::assets::Camera3D>()
            .next()
            .map(|c| CameraSnapshot {
                position: c.position,
                yaw: c.yaw,
                pitch: c.pitch,
                fov_y_degrees: c.fov_y_degrees,
                near: c.near,
                far: c.far,
            });

        if self.frame % SNAPSHOT_INTERVAL == 1 {
            state.system_count = world.system_count();
            state.component_count = world.component_count();
            state.systems = world
                .systems()
                .iter()
                .map(|s| s.name().to_string())
                .collect();
            state.assets = world
                .all_defs()
                .iter()
                .map(|def| AssetEntry {
                    kind: format!("{:?}", def.kind),
                    discriminant: def.discriminant,
                })
                .collect();
            // The AssetId -> name table is the build interner snapshot; it is
            // stable once the world is built, so capture it just once.
            if state.names.is_empty() {
                state.names = crate::ecs::asset_id::name_table();
            }
        }
    }

    fn attach_shutdown(&mut self, shutdown: CancellationToken) {
        let mut state = match self.shared.lock() {
            Ok(s) => s,
            Err(poisoned) => poisoned.into_inner(),
        };
        state.shutdown_token = Some(shutdown);
    }
}

// Accept loop. Each client is handled on its own thread so a slow or stuck
// client never blocks another. Errors are logged and dropped: a debug client
// disconnecting is routine, not a fault.
fn serve(listener: TcpListener, shared: Arc<Mutex<DebugState>>) {
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("debug server accept error: {e}");
                continue;
            }
        };
        let shared = Arc::clone(&shared);
        std::thread::spawn(move || {
            if let Err(e) = handle_conn(stream, shared) {
                tracing::debug!("debug client closed: {e}");
            }
        });
    }
}

fn handle_conn(stream: TcpStream, shared: Arc<Mutex<DebugState>>) -> Result<(), String> {
    let mut ws = accept(stream).map_err(|e| e.to_string())?;
    loop {
        match ws.read().map_err(|e| e.to_string())? {
            Message::Text(text) => {
                let reply = handle_request(&text, &shared);
                ws.send(Message::Text(reply)).map_err(|e| e.to_string())?;
            }
            Message::Ping(payload) => {
                ws.send(Message::Pong(payload)).map_err(|e| e.to_string())?;
            }
            Message::Close(_) => return Ok(()),
            _ => {}
        }
    }
}

#[derive(serde::Deserialize)]
struct Request {
    cmd: String,
}

// Dispatch one request against the shared snapshot and return a JSON reply.
fn handle_request(text: &str, shared: &Arc<Mutex<DebugState>>) -> String {
    let cmd = match serde_json::from_str::<Request>(text) {
        Ok(r) => r.cmd,
        Err(e) => return error_reply(&format!("malformed request: {e}")),
    };

    let state = match shared.lock() {
        Ok(s) => s,
        Err(poisoned) => poisoned.into_inner(),
    };

    let body = match cmd.as_str() {
        "ping" => serde_json::json!({ "ok": true, "pong": true }),
        "state" => serde_json::json!({
            "ok": true,
            "frame": state.frame,
            "system_count": state.system_count,
            "component_count": state.component_count,
            "systems": state.systems,
        }),
        "assets" => serde_json::json!({
            "ok": true,
            "frame": state.frame,
            "assets": state.assets,
        }),
        "names" => serde_json::json!({
            "ok": true,
            "names": state.names,
        }),
        "streaming" => {
            // Each pool is null when it is not streaming, else its
            // (resident, pending, unloaded) counts.
            let pool = |s: &Option<(usize, usize, usize)>| match s {
                Some((resident, pending, unloaded)) => serde_json::json!({
                    "resident": resident,
                    "pending": pending,
                    "unloaded": unloaded,
                }),
                None => serde_json::Value::Null,
            };
            // The chunk pool has no `unloaded` count -- an infinite world has
            // no bounded set of not-yet-loaded chunks.
            let chunk_pool = |s: &Option<(usize, usize)>| match s {
                Some((resident, pending)) => serde_json::json!({
                    "resident": resident,
                    "pending": pending,
                }),
                None => serde_json::Value::Null,
            };
            serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "texture": pool(&state.streaming.texture),
                "normal_map": pool(&state.streaming.normal_map),
                "mesh": pool(&state.streaming.mesh),
                "chunk": chunk_pool(&state.streaming.chunk),
            })
        }
        "profile" => {
            let r = &state.profile_render;
            let systems: Vec<_> = state
                .profile_systems
                .iter()
                .map(|(name, micros)| serde_json::json!({ "name": name, "micros": micros }))
                .collect();
            // Skip empty-name slots so the JSON reflects only the passes the
            // active backend actually populated. Per-pass GPU timing lands on
            // Metal + DirectX + Vulkan; backends that don't time a given pass
            // leave its slot at ("", 0), which the name filter drops.
            let passes: Vec<_> = r
                .pass_times_us
                .iter()
                .filter(|(name, _)| !name.is_empty())
                .map(|(name, micros)| serde_json::json!({ "name": name, "micros": micros }))
                .collect();
            serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "systems": systems,
                "render": {
                    "draw_calls": r.draw_calls,
                    "objects": r.objects,
                    "gpu_frame_us": r.gpu_frame_us,
                    "vram_bytes": r.vram_bytes,
                    "auto_exposure_ev": r.auto_exposure_ev,
                    "max_edr": r.max_edr,
                    "passes": passes,
                },
            })
        }
        "camera-get" => match &state.camera {
            Some(c) => serde_json::json!({
                "ok": true,
                "frame": state.frame,
                "position": c.position,
                "yaw": c.yaw,
                "pitch": c.pitch,
                "fov_y_degrees": c.fov_y_degrees,
                "near": c.near,
                "far": c.far,
            }),
            // No camera snapshot yet: either the world has no Camera3D or
            // `tick` has not run since startup.
            None => serde_json::json!({
                "ok": false,
                "error": "no Camera3D snapshot (world has no camera, or tick has not run yet)",
            }),
        },
        "shutdown" => {
            match &state.shutdown_token {
                Some(token) => {
                    token.cancel();
                    serde_json::json!({ "ok": true, "shutdown": true })
                }
                // attach_shutdown runs before the loop starts, so a None here
                // means the run loop has not been entered yet.
                None => serde_json::json!({
                    "ok": false,
                    "error": "shutdown token not attached yet",
                }),
            }
        }
        "reload-shaders" => {
            match &state.shader_reload {
                Some(flag) => {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    serde_json::json!({ "ok": true, "reload_queued": true })
                }
                // `tick` captures the flag once the backend exposes it, so a
                // `None` here means either hot-reload is off (`cn run` /
                // unsupported backend) or `tick` has not run yet.
                None => serde_json::json!({
                    "ok": false,
                    "error": "shader hot-reload not available (cn debug only, Metal-only today)",
                }),
            }
        }
        "reload-assets" => {
            match &state.asset_reload {
                Some(flag) => {
                    flag.store(true, std::sync::atomic::Ordering::SeqCst);
                    // AnimationSystem, the GraphicsSystem world-reload pass,
                    // and the world-loaded ShaderStage reload pass each
                    // listen on their own sibling flags. Fire all four here
                    // so a single WS command reloads every hot-reloadable
                    // surface in one shot.
                    crate::app::dev_flags::set_pending_animations();
                    hot_reload::set_pending_world();
                    hot_reload::set_pending_shader_stages();
                    serde_json::json!({ "ok": true, "reload_queued": true })
                }
                // `tick` captures the flag once `GraphicsSystem` exposes it,
                // so a `None` here means `cn run`, an all-procedural world,
                // or `tick` has not yet seen GraphicsSystem.
                None => serde_json::json!({
                    "ok": false,
                    "error": "asset hot-reload not available (cn debug only; no file-backed textures captured yet)",
                }),
            }
        }
        "decal-add" => {
            // Drop the snapshot lock before blocking on the engine reply:
            // the main thread will need to acquire it for the next tick.
            drop(state);
            return handle_decal_add(text);
        }
        "decal-remove" => {
            drop(state);
            return handle_decal_remove(text);
        }
        "emitter-add" => {
            drop(state);
            return handle_emitter_add(text);
        }
        "emitter-remove" => {
            drop(state);
            return handle_emitter_remove(text);
        }
        "anim-crossfade" => {
            // The handler needs the names table to resolve the target
            // SkinnedMesh, so capture it before dropping the snapshot lock.
            let names = state.names.clone();
            drop(state);
            return handle_anim_crossfade(text, &names);
        }
        "screenshot" => {
            // Drop the snapshot lock before blocking on the engine reply: the
            // render thread needs it for the next tick (which performs the
            // capture).
            drop(state);
            return handle_screenshot(text);
        }
        "camera-set" => {
            // Runtime mutation: drop the snapshot lock before blocking on the
            // engine reply, like the spawn commands above.
            drop(state);
            return handle_camera_set(text);
        }
        "quality-set" => {
            // Runtime mutation (live quality toggle): drop the snapshot lock
            // before blocking on the engine reply, like the spawn commands above.
            drop(state);
            return handle_quality_set(text);
        }
        "rebind" => {
            // Runtime mutation (live key rebind): drop the snapshot lock before
            // blocking on the engine reply, like the spawn commands above.
            drop(state);
            return handle_rebind(text);
        }
        "camera-move" => {
            // Sustained-motion mutation: same drop-then-block shape as
            // camera-set. The reply fires when the motion is accepted, not when
            // it finishes, so even a long move stays inside the WS timeout.
            drop(state);
            return handle_camera_move(text);
        }
        "camera-stop" => {
            drop(state);
            return handle_camera_stop();
        }
        "despawn" => {
            // Runtime mutation (remove an authored placement): drop the snapshot
            // lock before blocking on the engine reply, like the camera / quality
            // commands above.
            drop(state);
            return handle_despawn(text);
        }
        "reparent" => {
            // Runtime mutation (move an authored placement under a new parent):
            // drop the snapshot lock before blocking, like `despawn` above.
            drop(state);
            return handle_reparent(text);
        }
        "spawn" => {
            // Runtime mutation (instantiate a copy of an authored placement):
            // drop the snapshot lock before blocking, like `despawn` above.
            drop(state);
            return handle_spawn(text);
        }
        other => return error_reply(&format!("unknown cmd '{other}'")),
    };

    body.to_string()
}
