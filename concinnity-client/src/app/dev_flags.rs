// src/app/dev_flags.rs
//
// Process-wide flags shared between the engine loop (library) and the
// binary-only `cn debug` subsystem. Only the two flags the library itself
// names live here; the world.jsonl / ShaderStage "changed" flags and the
// decal / emitter spawn queue moved fully into the binary-only debug tree
// (`crate::debug`), since nothing in the library references them.
//
//   ENABLED              "are we running under cn debug?" Set once by main.rs's
//                        `Commands::Debug` arm before world build; read by
//                        `GraphicsSystem::init` / `AnimationSystem` / the draw
//                        list builder to enable disk-first shader loading + the
//                        hot-reload source capture. `cn run` leaves it false so
//                        production keeps the static `include_str!`-baked path
//                        with no filesystem dependency.
//   PENDING_ANIMATIONS   "an Animation source changed." Set by the cn debug
//                        watcher / WS `reload-assets` handler; consumed by the
//                        editor crate's `anim_reload::reload_clips_if_pending`,
//                        which the debug drive calls each frame to re-import
//                        file-backed clips. The flag lives here (in the runtime
//                        crate) because it bridges the runtime AnimationSystem,
//                        which reads ENABLED, and the editor-driven hot-reload.
//   VALIDATION           "did the launch request graphics validation?" Set by
//                        the CLI `--validation` flag (`cn run` / `cn debug`);
//                        read by `GraphicsSystem::init` to enable the DirectX /
//                        Vulkan debug layers. Tri-state: unset falls back to
//                        the build profile (on for debug, off for release).
//                        Metal's validation layer cannot be toggled from a
//                        running process, so the CLI re-execs with the env var
//                        instead; this flag does not drive Metal.
//
// A static is the pragmatic shape here: the flags are process-wide because the
// rendering backend is too (a single context per process owns the GPU), and
// plumbing them through the public `App` / `run_interpreted` signatures would
// touch far more code for the same observable behaviour.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);
static PENDING_ANIMATIONS: AtomicBool = AtomicBool::new(false);

// Tri-state validation request: 0 = unset (use the build-profile default),
// 1 = explicitly off, 2 = explicitly on.
static VALIDATION: AtomicU8 = AtomicU8::new(0);

// Mark this process as running under `cn debug` (or another dev-loop entry
// point that opts in). Call once before world build.
//
// `dead_code` allow: only the binary's `Commands::Debug` arm calls this; the
// library never sets the flag (it only reads it), so `cargo check --lib`
// reports it unused.
#[allow(dead_code)]
pub fn set_enabled(v: bool) {
    ENABLED.store(v, Ordering::SeqCst);
}

// True when the process is running under a dev-loop entry point that wants
// shader hot-reload. False for `cn run` and any embedded preview.
pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

// Raise the "Animation source changed" flag. Called by the cn debug asset
// hot-reload watcher and the WS `reload-assets` handler.
//
// `dead_code` allow: only the binary-only debug subsystem sets this; the
// library never does, so `cargo check --lib` reports it unused.
#[allow(dead_code)]
pub fn set_pending_animations() {
    PENDING_ANIMATIONS.store(true, Ordering::SeqCst);
}

// Swap the "Animation source changed" flag to `false`, returning whether it
// was set. The editor crate's `anim_reload::reload_clips_if_pending` calls
// this; a `true` result kicks the per-clip re-import pass.
#[allow(dead_code)]
pub fn take_pending_animations() -> bool {
    PENDING_ANIMATIONS.swap(false, Ordering::SeqCst)
}

// Record the CLI `--validation` request. `None` leaves the build-profile
// default in effect; `Some` forces validation on or off.
//
// `dead_code` allow: only the binary's `Commands::Run` / `Commands::Debug` arms
// call this; the library only reads it, so `cargo check --lib` reports it unused.
#[allow(dead_code)]
pub fn set_validation(v: Option<bool>) {
    let encoded = match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    VALIDATION.store(encoded, Ordering::SeqCst);
}

// The CLI validation request, or `None` when the launch did not specify one
// (the caller then falls back to `cfg!(debug_assertions)`).
pub(crate) fn validation() -> Option<bool> {
    match VALIDATION.load(Ordering::SeqCst) {
        1 => Some(false),
        2 => Some(true),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_off_and_round_trips() {
        // Capture and restore so this test does not leak state into others.
        let prior = enabled();
        set_enabled(false);
        assert!(!enabled());
        set_enabled(true);
        assert!(enabled());
        set_enabled(prior);
    }

    #[test]
    fn validation_tristate_round_trips() {
        // Capture and restore so this test does not leak state into others.
        let prior = validation();
        set_validation(None);
        assert_eq!(validation(), None);
        set_validation(Some(true));
        assert_eq!(validation(), Some(true));
        set_validation(Some(false));
        assert_eq!(validation(), Some(false));
        set_validation(prior);
    }
}
