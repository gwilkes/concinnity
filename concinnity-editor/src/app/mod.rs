// src/app/mod.rs
//
// Editor-side app helpers moved out of the runtime crate: the build / add / rm /
// check commands, the WebSocket command channel, the interpreted (`cn debug`)
// run path, and the per-frame debug hook. Runtime app items (App, dev_flags,
// anim_runtime) are re-exported from the runtime crate so the moved code's
// `crate::app::<item>` paths keep resolving.

pub(crate) mod add;
pub(crate) mod build;
pub(crate) mod check;
pub(crate) mod commands;
pub(crate) mod debug_hook;
pub(crate) mod pending;
pub(crate) mod rm;
pub(crate) mod run;
pub(crate) mod sources;
pub(crate) mod ws_client;

// Per-frame runtime callback implemented by the debug subsystem.
// Used by the binary's debug subsystem; unreferenced in the lib build.
#[allow(unused_imports)]
pub(crate) use debug_hook::DebugHook;

// Run the app directly from world.jsonl (the `cn debug` path).
// Used by the binary's debug command; unreferenced in the lib build.
#[allow(unused_imports)]
pub(crate) use run::run_interpreted;

// Runtime app items the editor drives, re-exported so moved code's
// `crate::app::<item>` paths keep resolving to the runtime crate.
#[allow(unused_imports)]
pub(crate) use concinnity_client::app::{anim_runtime, dev_flags, state};
