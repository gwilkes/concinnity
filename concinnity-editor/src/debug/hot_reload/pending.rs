// src/debug/hot_reload/pending.rs
//
// Process-wide "world.jsonl changed" / "world-loaded ShaderStage changed"
// signals (`cn debug` only). Set by the asset hot-reload watcher and the WS
// `reload-assets` command; consumed by the per-frame reload poll in
// `super::state::run_frame`. They live in the binary-only debug tree because
// nothing in the library references them: the reload passes that read them
// (`super::passes`) are driven entirely from `DebugHook::tick`.
//
// The sibling "Animation source changed" flag stays in `crate::app::dev_flags`
// instead, because `AnimationSystem` (library) names it directly.

use std::sync::atomic::{AtomicBool, Ordering};

// "world.jsonl changed" signal. Consumed by the world.jsonl reload poll to
// re-apply `Prop` transform edits in place via `backend.update_model`. V1
// covers transform-only edits (position / rotation / scale); add / remove and
// non-transform arg changes are detected and logged but not applied.
static PENDING_WORLD: AtomicBool = AtomicBool::new(false);

// "world-loaded ShaderStage source changed" signal. Consumed by the
// shader-stage reload poll to re-compile each captured `ShaderStage` source
// and rebuild the affected backend pipelines (main / instanced / shadow) into
// temporaries before swapping them in. Kept separate from `PENDING_WORLD` so a
// shader save does not also kick the Prop-diff and procedural-mesh passes.
static PENDING_SHADER_STAGES: AtomicBool = AtomicBool::new(false);

// Raise the "world.jsonl changed" flag. Called by the asset hot-reload watcher
// when a `.jsonl` save fires and by the debug WS `reload-assets` handler.
pub(crate) fn set_pending_world() {
    PENDING_WORLD.store(true, Ordering::SeqCst);
}

// Swap the "world.jsonl changed" flag to `false`, returning whether it was
// set. The reload poll calls this at frame start; a `true` result kicks the
// Prop-transform re-apply pass.
pub(crate) fn take_pending_world() -> bool {
    PENDING_WORLD.swap(false, Ordering::SeqCst)
}

// Raise the "world-loaded ShaderStage source changed" flag. Called by the
// asset hot-reload watcher when a captured `.metal` / `.hlsl` / `.glsl` source
// is saved and by the debug WS `reload-assets` handler.
pub(crate) fn set_pending_shader_stages() {
    PENDING_SHADER_STAGES.store(true, Ordering::SeqCst);
}

// Swap the "world-loaded ShaderStage source changed" flag to `false`,
// returning whether it was set. The reload poll calls this at frame start; a
// `true` result kicks the per-stage recompile + pipeline rebuild pass.
pub(crate) fn take_pending_shader_stages() -> bool {
    PENDING_SHADER_STAGES.swap(false, Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_flag_round_trips() {
        // Capture and restore so this test does not leak state into others.
        let prior = take_pending_world();
        assert!(!take_pending_world());
        set_pending_world();
        assert!(take_pending_world());
        assert!(!take_pending_world());
        if prior {
            set_pending_world();
        }
    }

    #[test]
    fn shader_stages_flag_round_trips() {
        let prior = take_pending_shader_stages();
        assert!(!take_pending_shader_stages());
        set_pending_shader_stages();
        assert!(take_pending_shader_stages());
        assert!(!take_pending_shader_stages());
        if prior {
            set_pending_shader_stages();
        }
    }
}
