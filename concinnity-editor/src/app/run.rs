// src/app/run.rs
//
// The interpreted (`cn debug`) run path: compiles world.jsonl fully in memory
// and drives the system loop with the WebSocket command channel and the
// per-frame debug hook. When a Save command arrives over the WebSocket, the
// world is rebuilt in place from the updated file. The production `cn run` path
// (compiled-blob playback, no command channel, no rebuild) lives in the runtime
// crate's `app::run`.

// The interpreted run path is driven only by the binary's `cn debug` command;
// unreferenced in the FFI lib build.
#![allow(dead_code)]

use crate::app::DebugHook;
use crate::app::state::App;
use crate::app::ws_client;
use crate::ecs::World;
use crate::world::find_world_jsonl;
use tokio_util::sync::CancellationToken;

// Interpreted entry point (`cn debug`). Compiles world.jsonl fully in memory
// -- shaders, meshes, textures, and all -- then runs the app without reading
// or writing any binary blob files. Always paired with the localhost debug
// server, and the only path that accepts a WebSocket command channel.
//
// ws_url and ws_user must both be present to enable the command channel;
// providing one without the other is an error.
//
// If no world.jsonl is found and WebSocket args are provided, starts with an
// empty world and waits for commands to populate it. Without WebSocket args,
// a missing world.jsonl is still an error.
pub(crate) fn run_interpreted(
    json_path: Option<&str>,
    ws_url: Option<String>,
    ws_user: Option<String>,
    debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    concinnity_client::app::run::init_logging();

    let resolved;
    let json_path = match json_path {
        Some(p) if std::path::Path::new(p).exists() => p,
        _ => {
            resolved = find_world_jsonl(None)?;
            resolved.as_str()
        }
    };

    let mut app = App::new();

    let content = std::fs::read_to_string(json_path).map_err(|e| {
        tracing::error!("Could not read {}: {}", json_path, e);
        e
    })?;

    let loaded = super::build::prepare(&content)?;

    *app.world_mut() = super::build::world_from_loaded(loaded)?;

    let ws_rx = match (ws_url, ws_user) {
        (Some(url), Some(user)) => {
            let rx = ws_client::connect(&url, &user)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::ConnectionRefused, e))?;
            Some(rx)
        }
        (Some(_), None) | (None, Some(_)) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--websocket and --ws-user must both be provided",
            ));
        }
        (None, None) => None,
    };

    start_app(app, ws_rx, debug)
}

// Build a World by running the full in-memory pipeline on world_path.
fn build_world_from_path(world_path: &str) -> std::io::Result<World> {
    let content = std::fs::read_to_string(world_path)?;
    let loaded = super::build::prepare(&content)?;
    super::build::world_from_loaded(loaded)
}

// Build a fresh App from the current world, reusing the given shutdown
// token so that CTRL+C and other cancellations remain wired up across rebuilds.
fn rebuild_app(shutdown: CancellationToken) -> std::io::Result<App> {
    let world_path = find_world_jsonl(None)?;

    tracing::info!("rebuilding world from {}", world_path);
    let world = build_world_from_path(&world_path)?;

    let mut app = App::new_with_token(shutdown);
    *app.world_mut() = world;
    Ok(app)
}

// Shared startup and loop entry once the App's world is populated.
//
// When a Save command is processed over the WebSocket, the world loop signals
// a rebuild. The current App is dropped (closing any open window), a new App
// is built from the saved world.jsonl, and the loop restarts with the same
// WebSocket receiver so the server stays connected.
fn start_app(
    initial_app: App,
    ws_rx: Option<ws_client::CmdReceiver>,
    mut debug: Option<Box<dyn DebugHook>>,
) -> std::io::Result<()> {
    // Capture the shutdown token once. It is shared across all rebuilds so
    // CTRL+C (registered once below) cancels the current world regardless of
    // how many rebuilds have occurred.
    let shutdown = initial_app.shutdown_token();

    let token = shutdown.clone();
    ctrlc::set_handler(move || {
        tracing::info!("CTRL+C received, cancelling all subsystems");
        token.cancel();
    })
    .expect("failed to set CTRL+C handler");

    // Hand the shutdown token to the debug hook so a debug client can request
    // a clean exit (the `shutdown` WS command). No-op when no hook is present.
    if let Some(hook) = debug.as_mut() {
        hook.attach_shutdown(shutdown.clone());
    }

    // On macOS, NSApplication is a per-process singleton. Activate it once
    // before the first NSWindow is created. Subsequent rebuilds reuse the
    // same NSApplication instance and do not need re-activation.
    #[cfg(target_os = "macos")]
    let mut ns_app_activated = false;

    let mut app = initial_app;

    loop {
        // Resolved before `start()` (while the GraphicsConfig is still present)
        // and reused after, so the post-start render-loop choice doesn't depend
        // on the config component, which `start()` drains. Only the macOS path
        // branches on it.
        #[cfg(target_os = "macos")]
        let renders = app.world().renders();

        #[cfg(target_os = "macos")]
        {
            if renders && !ns_app_activated {
                activate_app_macos();
                ns_app_activated = true;
            }
        }

        if let Err(e) = app.start() {
            eprintln!("Failed to start app: {}", e);
            std::process::exit(1);
        }

        #[cfg(target_os = "macos")]
        let rebuild = if renders {
            run_loop_macos(&mut app, ws_rx.as_ref(), debug.as_mut())
        } else {
            run_loop_default(&mut app, ws_rx.as_ref(), debug.as_mut())
        };

        #[cfg(not(target_os = "macos"))]
        let rebuild = run_loop_default(&mut app, ws_rx.as_ref(), debug.as_mut());

        if !rebuild || shutdown.is_cancelled() {
            break;
        }

        // Drop the current App (and its GPU context / window) before rebuilding.
        drop(app);

        match rebuild_app(shutdown.clone()) {
            Ok(new_app) => app = new_app,
            Err(e) => {
                tracing::error!("rebuild failed: {e}");
                break;
            }
        }
    }

    Ok(())
}

// Non-macOS (and headless macOS) world loop.
// Returns true if the loop exited due to a rebuild request.
fn run_loop_default(
    app: &mut App,
    ws_rx: Option<&ws_client::CmdReceiver>,
    mut debug: Option<&mut Box<dyn DebugHook>>,
) -> bool {
    use crate::ecs::StepResult;

    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            return false;
        }

        if let Some(rx) = ws_rx
            && ws_client::drain_commands(rx)
        {
            return true;
        }

        if let Some(hook) = debug.as_deref_mut() {
            hook.tick(app.world_mut());
        }

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop => return false,
            StepResult::Done => {
                if ws_rx.is_none() {
                    return false;
                }
                // No systems running; sleep briefly to avoid spinning while
                // waiting for commands from the server.
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
        }
    }
}

// Activate NSApplication so AppKit windows can be displayed. Must be called
// before any NSWindow is created (i.e. before GraphicsSystem::init()).
#[cfg(target_os = "macos")]
fn activate_app_macos() {
    use objc2_app_kit::{NSApplication, NSApplicationActivationPolicy};
    let mtm = objc2::MainThreadMarker::new()
        .expect("activate_app_macos must be called from the main thread");
    let ns_app = NSApplication::sharedApplication(mtm);
    ns_app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
    ns_app.activate();
}

// On macOS the Cocoa run loop must be pumped on the main thread each tick so
// AppKit can process window events, draw callbacks, and Metal presentation.
//
// CFRunLoopRunInMode is called with returnAfterSourceHandled=true so it returns
// as soon as one event is processed rather than blocking for the full timeout.
// The outer loop re-enters immediately, draining the event queue before handing
// control to the world step. This keeps the window responsive and allows Metal
// drawable callbacks to fire without delay.
//
// The shutdown token is checked each iteration so CTRL+C exits cleanly.
// Returns true if the loop exited due to a rebuild request.
#[cfg(target_os = "macos")]
fn run_loop_macos(
    app: &mut App,
    ws_rx: Option<&ws_client::CmdReceiver>,
    mut debug: Option<&mut Box<dyn DebugHook>>,
) -> bool {
    use crate::ecs::StepResult;
    use core_foundation::runloop::{CFRunLoopRunInMode, kCFRunLoopDefaultMode};

    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            tracing::info!("Shutdown token cancelled, exiting loop");
            return false;
        }

        if let Some(rx) = ws_rx
            && ws_client::drain_commands(rx)
        {
            return true;
        }

        // Drain all pending AppKit/CoreFoundation events before the world step.
        // returnAfterSourceHandled=true means the call returns after each event,
        // so we loop until the run loop reports nothing left to handle (result != 4).
        loop {
            // kCFRunLoopRunHandledSource = 4; any other value means the queue is empty
            let result = unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, true as u8) };
            if result != 4 {
                break;
            }
        }

        if let Some(hook) = debug.as_deref_mut() {
            hook.tick(app.world_mut());
        }

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop => return false,
            StepResult::Done => {
                if ws_rx.is_none() {
                    return false;
                }
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
        }
    }
}
