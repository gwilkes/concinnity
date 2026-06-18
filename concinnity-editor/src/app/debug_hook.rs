// src/app/debug_hook.rs
// Runtime injection point for the binary-only debug subsystem.
//
// The library owns the world loop but knows nothing about debugging. A
// `DebugHook` is an optional per-frame callback the loop invokes on the main
// thread; the only implementation lives in `crate::debug` (CLI binary only),
// so the lib never carries the debug server. The trait is `pub(crate)` to
// keep it out of the library's public API: both the lib and the binary
// compile this source, so the binary's `debug` module can still implement it.

use crate::ecs::World;
use tokio_util::sync::CancellationToken;

// Implemented only by the binary's debug subsystem; unreferenced in the FFI
// lib build.
#[allow(dead_code)]
pub(crate) trait DebugHook: Send {
    // Called once per frame on the main thread, just before the world step.
    // Receives the live world so the hook can inspect (and later mutate) it.
    fn tick(&mut self, world: &mut World);

    // Called once before the run loop starts, handing the hook the app's
    // shutdown token. A hook can cancel it to ask the engine to exit cleanly
    // (the run loop checks the token every iteration), e.g. a debug client
    // issuing a `shutdown` command. Default: ignore the token.
    fn attach_shutdown(&mut self, _shutdown: CancellationToken) {}
}
