// src/debug/commands.rs
//
// Runtime spawn / crossfade WebSocket command handlers (`decal-add`,
// `emitter-add`, `anim-crossfade`, …) plus their request-body structs and the
// shared `error_reply` helper. Each parses its JSON body, enqueues onto the
// matching process-wide queue (`super::runtime_spawn` /
// `crate::app::anim_runtime`), and blocks on a one-shot reply channel the
// per-frame debug drive fulfils. The query commands + dispatch live in
// `super::server::handle_request`.

// Maximum wait for `GraphicsSystem::step` to drain a runtime-spawn command
// and reply. The drain runs once per frame, so a healthy 60 Hz engine
// replies inside ~16 ms; 1 s gives plenty of headroom even on a slow boot
// frame (4K HDR bake) without leaving a WS client hanging forever if the
// engine has stalled.
const SPAWN_REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(1);

#[derive(serde::Deserialize)]
#[serde(default)]
struct DecalAddRequest {
    #[serde(skip)]
    _cmd: String,
    texture: Option<String>,
    position: [f32; 3],
    rotation_deg: [f32; 3],
    size: [f32; 3],
    tint: [f32; 4],
}

impl Default for DecalAddRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            texture: None,
            position: [0.0, 0.0, 0.0],
            rotation_deg: [0.0, 0.0, 0.0],
            size: [1.0, 1.0, 1.0],
            tint: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

pub(super) fn handle_decal_add(text: &str) -> String {
    let req: DecalAddRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("decal-add: {e}")),
    };
    let args = super::runtime_spawn::DecalSpawnArgs {
        texture: req.texture,
        position: req.position,
        rotation_deg: req.rotation_deg,
        size: req.size,
        tint: req.tint,
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::DecalAdd {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(id)) => serde_json::json!({ "ok": true, "id": id }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("decal-add: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
struct IdRequest {
    id: usize,
}

pub(super) fn handle_decal_remove(text: &str) -> String {
    let req: IdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("decal-remove: {e}")),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::DecalRemove {
        id: req.id,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "removed": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("decal-remove: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
#[serde(default)]
struct EmitterAddRequest {
    #[serde(skip)]
    _cmd: String,
    texture: Option<String>,
    position: [f32; 3],
    direction: [f32; 3],
    spread_deg: f32,
    speed_min: f32,
    speed_max: f32,
    lifetime_min: f32,
    lifetime_max: f32,
    gravity: [f32; 3],
    spawn_rate: f32,
    max_particles: u32,
    size_start: f32,
    size_end: f32,
    color_start: [f32; 4],
    color_end: [f32; 4],
}

impl Default for EmitterAddRequest {
    fn default() -> Self {
        let d = super::runtime_spawn::EmitterSpawnArgs::default();
        Self {
            _cmd: String::new(),
            texture: d.texture,
            position: d.position,
            direction: d.direction,
            spread_deg: d.spread_deg,
            speed_min: d.speed_min,
            speed_max: d.speed_max,
            lifetime_min: d.lifetime_min,
            lifetime_max: d.lifetime_max,
            gravity: d.gravity,
            spawn_rate: d.spawn_rate,
            max_particles: d.max_particles,
            size_start: d.size_start,
            size_end: d.size_end,
            color_start: d.color_start,
            color_end: d.color_end,
        }
    }
}

pub(super) fn handle_emitter_add(text: &str) -> String {
    let req: EmitterAddRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("emitter-add: {e}")),
    };
    let args = super::runtime_spawn::EmitterSpawnArgs {
        texture: req.texture,
        position: req.position,
        direction: req.direction,
        spread_deg: req.spread_deg,
        speed_min: req.speed_min,
        speed_max: req.speed_max,
        lifetime_min: req.lifetime_min,
        lifetime_max: req.lifetime_max,
        gravity: req.gravity,
        spawn_rate: req.spawn_rate,
        max_particles: req.max_particles,
        size_start: req.size_start,
        size_end: req.size_end,
        color_start: req.color_start,
        color_end: req.color_end,
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::EmitterAdd {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(id)) => serde_json::json!({ "ok": true, "id": id }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("emitter-add: timed out waiting for engine"),
    }
}

pub(super) fn handle_emitter_remove(text: &str) -> String {
    let req: IdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("emitter-remove: {e}")),
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::EmitterRemove {
        id: req.id,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "removed": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("emitter-remove: timed out waiting for engine"),
    }
}

#[derive(serde::Deserialize)]
struct AnimCrossfadeRequest {
    #[serde(default)]
    target: String,
    #[serde(default)]
    weights: Vec<f32>,
    #[serde(default)]
    duration_secs: f32,
}

pub(super) fn handle_anim_crossfade(text: &str, names: &[String]) -> String {
    let req: AnimCrossfadeRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("anim-crossfade: {e}")),
    };
    if req.target.is_empty() {
        return error_reply("anim-crossfade: missing 'target'");
    }
    // The names table is `Vec<String>` indexed by `AssetId`; a small linear
    // scan is fine for a debug command that fires at most a few times per
    // second.
    let Some(asset_idx) = names.iter().position(|n| n == &req.target) else {
        return error_reply(&format!(
            "anim-crossfade: unknown asset name '{}'",
            req.target
        ));
    };
    let target = crate::ecs::asset_id::AssetId(asset_idx as u32);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    crate::app::anim_runtime::enqueue(crate::app::anim_runtime::AnimCommand::Crossfade {
        req: crate::app::anim_runtime::CrossfadeRequest {
            target,
            weights: req.weights,
            duration_secs: req.duration_secs,
        },
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("anim-crossfade: timed out waiting for engine"),
    }
}

// Longer than the spawn timeout: the capture idles the GPU, copies the
// swapchain image back, and PNG-encodes + writes it on the render thread, which
// can take noticeably longer than a simple state mutation.
const SCREENSHOT_REPLY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(serde::Deserialize)]
struct ScreenshotRequest {
    #[serde(default)]
    path: String,
}

pub(super) fn handle_screenshot(text: &str) -> String {
    let req: ScreenshotRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("screenshot: {e}")),
    };
    if req.path.trim().is_empty() {
        return error_reply("screenshot: missing 'path'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Screenshot {
        path: req.path,
        reply: tx,
    });
    match rx.recv_timeout(SCREENSHOT_REPLY_TIMEOUT) {
        Ok(Ok(path)) => serde_json::json!({ "ok": true, "path": path }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("screenshot: timed out waiting for engine"),
    }
}

// Teleport the active camera. `position` / `yaw` / `pitch` are required in
// practice (the probe always sends them); missing fields fall back to the
// defaults below, matching the decal / emitter request shape. `yaw` / `pitch`
// are radians; `fov_y_degrees` is omitted to leave the camera's FOV untouched.
#[derive(serde::Deserialize)]
#[serde(default)]
struct CameraSetRequest {
    #[serde(skip)]
    _cmd: String,
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    fov_y_degrees: Option<f32>,
}

impl Default for CameraSetRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            position: [0.0, 0.0, 0.0],
            yaw: 0.0,
            pitch: 0.0,
            fov_y_degrees: None,
        }
    }
}

pub(super) fn handle_camera_set(text: &str) -> String {
    let req: CameraSetRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("camera-set: {e}")),
    };
    let args = super::runtime_spawn::CameraSetArgs {
        position: req.position,
        yaw: req.yaw,
        pitch: req.pitch,
        fov_y_degrees: req.fov_y_degrees,
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::CameraSet {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "set": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("camera-set: timed out waiting for engine"),
    }
}

// Toggle a Quality-group graphics setting. `setting` is the engine key
// (taa / ssao / ssr / ssgi / auto_exposure); `op` cycles it (next | prev,
// both flip a binary toggle). Defaults match the decal / camera request shape.
#[derive(serde::Deserialize)]
#[serde(default)]
struct QualitySetRequest {
    #[serde(skip)]
    _cmd: String,
    setting: String,
    op: String,
}

impl Default for QualitySetRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            setting: String::new(),
            op: "next".to_string(),
        }
    }
}

// Toggle a Quality-group setting live by injecting the same `SettingCommand`
// the settings menu emits, so the engine runs its real `apply_quality_settings`
// rebuild. `cn debug` only; lets a headless harness exercise the live toggle
// path and screenshot the result. The reply fires once the command is queued
// (the GraphicsSystem applies it on its next step, before the next present).
pub(super) fn handle_quality_set(text: &str) -> String {
    let req: QualitySetRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("quality-set: {e}")),
    };
    if req.setting.is_empty() {
        return error_reply("quality-set: missing 'setting'");
    }
    let op = match req.op.as_str() {
        "next" | "" => crate::assets::SettingOp::Next,
        "prev" => crate::assets::SettingOp::Prev,
        other => {
            return error_reply(&format!(
                "quality-set: unknown op '{other}' (use next | prev)"
            ));
        }
    };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::QualitySet {
        setting: req.setting,
        op,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("quality-set: timed out waiting for engine"),
    }
}

// Rebind a movement action to a key. `setting` is the engine key
// (`key_forward` / `key_backward` / `key_left` / `key_right` / `key_sprint` /
// `key_jump` / `key_interact`); `key` is a canonical `Key` variant name
// (`W`, `Space`, `Shift`, `Num1`, `Up`, ...).
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct RebindRequest {
    #[serde(skip)]
    _cmd: String,
    setting: String,
    key: String,
}

// Rebind a movement key live by injecting the same `Rebind` `SettingCommand` the
// settings menu emits on a capture, so the engine runs its real swap + persist +
// `set_keymap` path. `cn debug` only; lets a headless harness exercise the live
// rebind and screenshot the row label flipping. The reply fires once queued.
pub(super) fn handle_rebind(text: &str) -> String {
    let req: RebindRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("rebind: {e}")),
    };
    if req.setting.is_empty() {
        return error_reply("rebind: missing 'setting' (e.g. key_forward)");
    }
    // The canonical `Key` serializes to its variant name, so a JSON string
    // deserializes straight to it (W, Space, Shift, Num1, Up, ...).
    let key: crate::assets::Key =
        match serde_json::from_value(serde_json::Value::String(req.key.clone())) {
            Ok(k) => k,
            Err(_) => {
                return error_reply(&format!(
                    "rebind: unknown key '{}' (use a Key variant like W / Space / Shift)",
                    req.key
                ));
            }
        };
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Rebind {
        setting: req.setting,
        key,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("rebind: timed out waiting for engine"),
    }
}

// Despawn an authored placement by its declared name.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct DespawnCmdRequest {
    #[serde(skip)]
    _cmd: String,
    name: String,
}

// Remove an authored placement (and its descendants) live by name: enqueue a
// Despawn command the per-frame drive forwards to `dispatch_despawn`, which
// sends a `DespawnRequest` the GraphicsSystem applies on its next step (hide the
// draw slots + despawn the entity, cascading to children). `cn debug` only; lets
// a headless harness remove an entity and screenshot it gone. The reply fires
// once the command is queued.
pub(super) fn handle_despawn(text: &str) -> String {
    let req: DespawnCmdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("despawn: {e}")),
    };
    if req.name.trim().is_empty() {
        return error_reply("despawn: missing 'name'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Despawn {
        name: req.name,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("despawn: timed out waiting for engine"),
    }
}

// Re-parent an authored placement. `child` is moved under `parent`; a null or
// omitted `parent` detaches the child to a root.
#[derive(serde::Deserialize, Default)]
#[serde(default)]
struct ReparentCmdRequest {
    #[serde(skip)]
    _cmd: String,
    child: String,
    parent: Option<String>,
}

// Re-parent an authored placement live by name: enqueue a Reparent command the
// per-frame drive forwards to `dispatch_reparent`, which sends a
// `ReparentRequest` the GraphicsSystem applies on its next step (re-point the
// Parent edge + recompose world matrices). `cn debug` only; lets a headless
// harness move an entity under a new parent and screenshot the result. The reply
// fires once queued.
pub(super) fn handle_reparent(text: &str) -> String {
    let req: ReparentCmdRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("reparent: {e}")),
    };
    if req.child.trim().is_empty() {
        return error_reply("reparent: missing 'child'");
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::Reparent {
        child: req.child,
        // An empty / whitespace parent name detaches the child to a root.
        parent: req.parent.filter(|p| !p.trim().is_empty()),
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "queued": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("reparent: timed out waiting for engine"),
    }
}

// Move the active camera by a per-frame delta over a span of frames. All delta
// fields default to 0 and `frames` to 0 (an indefinite hold cleared by
// `camera-stop`); a profiling harness can then sustain motion mid-screenshot to
// surface temporal effects. `yaw` / `pitch` are radians.
#[derive(serde::Deserialize)]
#[serde(default)]
struct CameraMoveRequest {
    #[serde(skip)]
    _cmd: String,
    forward: f32,
    right: f32,
    up: f32,
    yaw: f32,
    pitch: f32,
    frames: u32,
}

impl Default for CameraMoveRequest {
    fn default() -> Self {
        Self {
            _cmd: String::new(),
            forward: 0.0,
            right: 0.0,
            up: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            frames: 0,
        }
    }
}

pub(super) fn handle_camera_move(text: &str) -> String {
    let req: CameraMoveRequest = match serde_json::from_str(text) {
        Ok(r) => r,
        Err(e) => return error_reply(&format!("camera-move: {e}")),
    };
    let args = super::runtime_spawn::CameraMoveArgs {
        forward: req.forward,
        right: req.right,
        up: req.up,
        yaw: req.yaw,
        pitch: req.pitch,
        frames: req.frames,
    };
    let frames = args.frames;
    let holding = frames == 0;
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::CameraMove {
        args,
        reply: tx,
    });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => {
            serde_json::json!({ "ok": true, "frames": frames, "holding": holding }).to_string()
        }
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("camera-move: timed out waiting for engine"),
    }
}

pub(super) fn handle_camera_stop() -> String {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    super::runtime_spawn::enqueue(super::runtime_spawn::RuntimeCommand::CameraStop { reply: tx });
    match rx.recv_timeout(SPAWN_REPLY_TIMEOUT) {
        Ok(Ok(())) => serde_json::json!({ "ok": true, "stopped": true }).to_string(),
        Ok(Err(e)) => error_reply(&e),
        Err(_) => error_reply("camera-stop: timed out waiting for engine"),
    }
}

pub(super) fn error_reply(msg: &str) -> String {
    serde_json::json!({ "ok": false, "error": msg }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_set_request_parses_full_payload() {
        let req: CameraSetRequest = serde_json::from_str(
            r#"{"cmd":"camera-set","position":[1.0,2.0,3.0],"yaw":0.5,"pitch":-0.25,"fov_y_degrees":60.0}"#,
        )
        .expect("valid payload parses");
        assert_eq!(req.position, [1.0, 2.0, 3.0]);
        assert_eq!(req.yaw, 0.5);
        assert_eq!(req.pitch, -0.25);
        assert_eq!(req.fov_y_degrees, Some(60.0));
    }

    #[test]
    fn camera_set_request_fov_optional() {
        let req: CameraSetRequest = serde_json::from_str(
            r#"{"cmd":"camera-set","position":[0.0,1.0,0.0],"yaw":0.0,"pitch":0.0}"#,
        )
        .expect("payload without fov parses");
        assert_eq!(req.position, [0.0, 1.0, 0.0]);
        assert!(req.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_request_defaults_for_missing_fields() {
        let req: CameraSetRequest =
            serde_json::from_str(r#"{"cmd":"camera-set"}"#).expect("bare command parses");
        assert_eq!(req.position, [0.0, 0.0, 0.0]);
        assert_eq!(req.yaw, 0.0);
        assert_eq!(req.pitch, 0.0);
        assert!(req.fov_y_degrees.is_none());
    }

    #[test]
    fn camera_set_request_rejects_malformed() {
        // position must be three numbers; a string is a hard parse error.
        assert!(
            serde_json::from_str::<CameraSetRequest>(r#"{"cmd":"camera-set","position":"nope"}"#)
                .is_err()
        );
    }

    #[test]
    fn camera_move_request_parses_full_payload() {
        let req: CameraMoveRequest = serde_json::from_str(
            r#"{"cmd":"camera-move","forward":2.0,"right":-1.0,"up":0.5,"yaw":0.1,"pitch":-0.2,"frames":30}"#,
        )
        .expect("valid payload parses");
        assert_eq!(req.forward, 2.0);
        assert_eq!(req.right, -1.0);
        assert_eq!(req.up, 0.5);
        assert_eq!(req.yaw, 0.1);
        assert_eq!(req.pitch, -0.2);
        assert_eq!(req.frames, 30);
    }

    #[test]
    fn camera_move_request_defaults_to_zero_hold() {
        // A bare command leaves every delta at 0 and frames at 0 (indefinite
        // hold), matching the spawn-request default convention.
        let req: CameraMoveRequest =
            serde_json::from_str(r#"{"cmd":"camera-move"}"#).expect("bare command parses");
        assert_eq!(req.forward, 0.0);
        assert_eq!(req.frames, 0);
    }

    #[test]
    fn camera_move_request_rejects_malformed() {
        // frames must be an unsigned integer; a string is a hard parse error.
        assert!(
            serde_json::from_str::<CameraMoveRequest>(r#"{"cmd":"camera-move","frames":"lots"}"#)
                .is_err()
        );
    }

    #[test]
    fn despawn_request_parses_name() {
        let req: DespawnCmdRequest =
            serde_json::from_str(r#"{"cmd":"despawn","name":"crate_a"}"#).expect("valid parses");
        assert_eq!(req.name, "crate_a");
    }

    #[test]
    fn despawn_request_defaults_to_empty_name() {
        let req: DespawnCmdRequest =
            serde_json::from_str(r#"{"cmd":"despawn"}"#).expect("bare command parses");
        assert!(req.name.is_empty());
    }

    #[test]
    fn reparent_request_parses_child_and_parent() {
        let req: ReparentCmdRequest =
            serde_json::from_str(r#"{"cmd":"reparent","child":"box_a","parent":"frame"}"#)
                .expect("valid parses");
        assert_eq!(req.child, "box_a");
        assert_eq!(req.parent.as_deref(), Some("frame"));
    }

    #[test]
    fn reparent_request_parent_optional() {
        let req: ReparentCmdRequest = serde_json::from_str(r#"{"cmd":"reparent","child":"box_a"}"#)
            .expect("bare parent parses");
        assert_eq!(req.child, "box_a");
        assert!(req.parent.is_none());
    }
}
