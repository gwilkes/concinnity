// src/app/anim_runtime.rs
//
// Process-wide command queue for runtime animation crossfades. Mirrors the
// shape of `crate::debug::runtime_spawn`, but separate so the AnimationSystem
// can drain its own commands without contending with GraphicsSystem's
// decal / particle queue.
//
// The debug WebSocket server (binary-only, off the engine thread) pushes
// commands here; `AnimationSystem::step` drains them at frame start and
// applies a new target weight vector + transition duration to the matching
// per-target clip bucket. Each command carries a reply channel so the WS
// handler can hand a synchronous result back to its client.

use std::sync::Mutex;

use crate::ecs::asset_id::AssetId;

// One queued crossfade request. `target` is the `SkinnedMesh` asset id the
// command applies to; `weights` must match the clip count registered for
// that target. `duration_secs == 0` snaps to the new weights on the next
// frame.
#[derive(Debug)]
#[allow(dead_code)]
pub struct CrossfadeRequest {
    pub target: AssetId,
    pub weights: Vec<f32>,
    pub duration_secs: f32,
}

// One runtime command pushed onto [`enqueue`] by the debug WS server and
// consumed by [`AnimationSystem::step`](crate::gfx::animation::AnimationSystem::step).
//
// `dead_code` allow: the only producer is the binary-only `crate::debug`
// module (declared by main.rs, not lib.rs), so `cargo check --lib` sees the
// variant as unconstructed.
#[allow(dead_code)]
pub enum AnimCommand {
    Crossfade {
        req: CrossfadeRequest,
        reply: std::sync::mpsc::SyncSender<Result<(), String>>,
    },
}

static QUEUE: Mutex<Vec<AnimCommand>> = Mutex::new(Vec::new());

// Push a command onto the animation runtime queue. The caller blocks on its
// own reply receiver to get the result. A poisoned mutex is recovered and
// used regardless (an unrelated panic must not silently drop commands).
//
// `dead_code` allow: the only producer is the binary-only debug module.
#[allow(dead_code)]
pub fn enqueue(cmd: AnimCommand) {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    q.push(cmd);
}

// Take every queued command. Drained by `AnimationSystem::apply_crossfade_commands`,
// which the `cn debug` drive (`DebugHook::tick`) calls each frame. The
// returned `Vec` is the live list: the queue is reset to empty.
pub(crate) fn drain() -> Vec<AnimCommand> {
    let mut q = match QUEUE.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    std::mem::take(&mut *q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_drain_round_trip() {
        let _ = drain();
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        enqueue(AnimCommand::Crossfade {
            req: CrossfadeRequest {
                target: AssetId::default(),
                weights: vec![1.0, 0.0],
                duration_secs: 0.5,
            },
            reply: tx,
        });
        let cmds = drain();
        assert_eq!(cmds.len(), 1);
        assert!(drain().is_empty());
    }
}
