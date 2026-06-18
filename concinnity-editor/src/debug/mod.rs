// src/debug/mod.rs
// Binary-only runtime debug server.
//
// Declared by `main.rs` only, never by `lib.rs`, so it is compiled into the
// `cn-client` binary and excluded from `libconcinnity`: `cargo build --lib`
// never sees it.
//
// `cn-client debug` starts a localhost WebSocket server. The engine
// stays debug-agnostic: the only coupling is the `DebugHook` trait, which the
// run loop invokes once per frame on the main thread (see
// `crate::app::debug_hook`). `DebugServer::tick` snapshots the live world into
// shared state; the server thread answers client queries from that snapshot.
//
// Protocol, each WS text frame is one JSON request:
//   client → server:  {"cmd":"state"} {"cmd":"assets"} {"cmd":"names"}
//                      {"cmd":"streaming"} {"cmd":"profile"} {"cmd":"ping"}
//                      {"cmd":"reload-shaders"} {"cmd":"reload-assets"}
//                      {"cmd":"shutdown"} {"cmd":"camera-get"}
//                      {"cmd":"camera-set","position":[x,y,z],"yaw":Y,"pitch":P,
//                         "fov_y_degrees":F}
//                      {"cmd":"camera-move","forward":F,"right":R,"up":U,
//                         "yaw":Y,"pitch":P,"frames":N} {"cmd":"camera-stop"}
//                      {"cmd":"decal-add", ...} {"cmd":"decal-remove","id":N}
//                      {"cmd":"emitter-add", ...} {"cmd":"emitter-remove","id":N}
//                      {"cmd":"anim-crossfade","target":"hero",
//                         "weights":[0,1,0],"duration_secs":0.5}
//   server → client:  {"ok":true,...}  |  {"ok":false,"error":"..."}
//
// `anim-crossfade` re-weights the clip bucket for one SkinnedMesh (looked
// up by asset name) and ramps the live blend toward the new weights over
// `duration_secs`. The weight vector must match the target's clip count.
// The handler queues the command on [`crate::app::anim_runtime`] and
// blocks on a one-shot reply channel that `AnimationSystem::step` fulfils
// on the next frame. `duration_secs == 0` snaps immediately.
//
// `decal-add` / `decal-remove` / `emitter-add` / `emitter-remove` are the
// runtime spawn/despawn entry points. The handler queues the command on
// [`self::runtime_spawn`] and blocks on a one-shot reply channel that the
// per-frame debug drive fulfils on the next frame (~16 ms at 60 Hz). On
// success `decal-add` and `emitter-add` return `{"ok":true,"id":N}` where
// `N` is the stable slot index to feed back into the matching `remove`. The
// field shapes mirror the `Decal` / `ParticleEmitter` asset args.
//
// `camera-get` is a read-only snapshot (like `state` / `profile`): it reports
// the active `Camera3D`'s `position`, `yaw`, `pitch`, `fov_y_degrees`, `near`,
// and `far` from the per-tick snapshot. `camera-set` is a runtime mutation
// (like `decal-add` / `screenshot`): it queues a new pose on
// [`self::runtime_spawn`], blocks on a one-shot reply, and the per-frame debug
// drive writes `position` / `yaw` / `pitch` (and `fov_y_degrees` when present)
// onto the active camera and zeroes the controller velocity so a free-fly
// camera does not drift the teleport away. Used to benchmark a fixed, repeatable
// viewpoint instead of the world's authored spawn camera.
//
// `camera-move` is the in-motion counterpart to `camera-set`: it installs a
// per-frame pose delta (`forward` / `right` / `up` world-unit offsets along the
// free-fly look basis, `yaw` / `pitch` radian deltas) the per-frame debug drive
// re-applies to the active camera for `frames` frames, so the renderer sees
// sustained motion and temporal passes (TAA, SSGI, motion blur) accumulate.
// `frames == 0` holds the motion until a `camera-stop` clears it. The reply
// fires when the motion is accepted (a `Camera3D` exists), not when it
// finishes, so a long move never outlasts the WS timeout. Take a `screenshot`
// while a move is in flight to capture a mid-motion frame. `camera-stop` clears
// any in-progress motion.
//
// `reload-shaders` flips the shared atomic flag the Metal backend polls at
// frame start, so the next frame rebuilds every built-in renderer pipeline
// from disk-resident `.metal` source. Returns `{"ok":false}` when the
// backend is not yet initialised or did not opt into hot-reload (production
// `cn run` paths never expose the flag).
//
// `reload-assets` flips the analogous flag `GraphicsSystem::step` polls. It
// re-decodes every file-backed `Texture` source captured at init and pushes
// the fresh pixels through the existing `update_texture_slot` /
// `update_normal_map_slot` swap paths. Returns `{"ok":false}` when no
// file-backed textures were captured (`cn run`, all-procedural worlds, or
// `cn debug` before init has finished).
//
// `streaming` reports the asset-streaming subsystem's `(resident, pending,
// unloaded)` counts for the albedo-texture, normal-map, and mesh pools (each
// null when that pool is not streaming), so a headless run can confirm
// streaming progress without scraping the `tracing` log.
//
// `profile` reports the engine profiler: each system's last-frame CPU step
// time (micros), the render backend's draw-call / object counts plus GPU
// frame time, and (Metal only today) per-pass GPU timings under the
// `render.passes` array: `[{"name":"main","micros":1234}, ...]`. Empty-
// name slots are dropped, so a backend with no per-pass support reports
// an empty array.
//
// `names` returns the build interner's `AssetId` -> name table (index = id),
// so a client can remap any runtime `AssetId` back to its world.jsonl name.
// `shutdown` cancels the app shutdown token, so the engine exits cleanly on
// its next loop iteration: used by `scripts/debug_probe.py` to stop the
// client after a headless smoke test instead of leaving the window open.

// Submodules (all binary-only):
//   server    DebugServer + DebugState + the WS accept / connection loop
//   commands  spawn / crossfade command handlers + request bodies
//   hot_reload  asset / shader / world.jsonl reload machinery
//   runtime_spawn  decal / emitter / screenshot spawn queue + dispatch
mod commands;
mod hot_reload;
mod runtime_spawn;
mod server;

pub use server::DebugServer;
