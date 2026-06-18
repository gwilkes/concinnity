// src/app/run.rs
//
// The runtime player path. Loads compiled blob data and drives the system
// loop. Fully synchronous -- no Tokio runtime here. Systems that need async
// (HttpServerSystem, LlmSystem, etc.) spin up their own runtimes internally.
//
// On macOS the world loop is driven by CFRunLoopRunInMode so that AppKit
// (GLFW window creation, Metal pipeline compilation, event dispatch) can
// process its callbacks on the main thread each tick. On all other platforms
// a tight Rust loop is used, which is what VulkanRenderer expects.
//
// This is the `cn run` path only: no debug server, no WebSocket command
// channel, no in-memory rebuild. A shipped run is neither remotely inspectable
// nor remotely driven. The interpreted (`cn debug`) path with hot-reload and
// the command channel lives in the editor crate.

use crate::app::state::App;
use crate::ecs::StepResult;
use tracing_subscriber::EnvFilter;

// Default tracing filter applied when RUST_LOG is unset: info for debug
// builds, warn for release builds. A RUST_LOG value always takes precedence.
fn default_log_directive() -> &'static str {
    if cfg!(debug_assertions) {
        "info"
    } else {
        "warn"
    }
}

// Build the tracing filter from RUST_LOG, falling back to the build-profile
// default when the variable is unset.
fn log_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_log_directive()))
}

// Install the global tracing subscriber. The single place the log level is
// configured: the CLI entry points call it directly, and the FFI entry point
// (cn_init) calls it for the macOS app. Safe to call once per process.
pub fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(log_filter())
        .try_init();
}

// Production entry point (`cn run`). Reads the compiled binary blobs from
// data/, written by a prior `cn build`. No debug server, no WebSocket command
// channel: a shipped run is neither remotely inspectable nor remotely driven.
pub fn run() -> std::io::Result<()> {
    init_logging();

    let mut app = App::new();

    if let Err(e) = app.load_blob() {
        tracing::info!(
            "No blob found (data/0): {:?} -- run `concinnity build` first",
            e
        );

        return Ok(());
    }

    start_runtime(app)
}

// Startup and loop entry once the App's world is populated from a compiled
// blob. Registers the CTRL+C handler, activates AppKit on macOS, starts the
// app, then runs the world loop until the window closes, a system stops the
// world, or CTRL+C is received.
pub fn start_runtime(mut app: App) -> std::io::Result<()> {
    let shutdown = app.shutdown_token();
    let token = shutdown.clone();
    ctrlc::set_handler(move || {
        tracing::info!("CTRL+C received, cancelling all subsystems");
        token.cancel();
    })
    .expect("failed to set CTRL+C handler");

    // Resolved before `start()` (while the GraphicsConfig is still present) and
    // reused after, so the post-start render-loop choice doesn't depend on the
    // config component, which `start()` drains. Only the macOS path branches.
    #[cfg(target_os = "macos")]
    let renders = app.world().renders();

    #[cfg(target_os = "macos")]
    {
        if renders {
            activate_app_macos();
        }
    }

    if let Err(e) = app.start() {
        eprintln!("Failed to start app: {}", e);
        std::process::exit(1);
    }

    #[cfg(target_os = "macos")]
    {
        if renders {
            run_loop_macos(&mut app);
        } else {
            run_loop_default(&mut app);
        }
    }

    #[cfg(not(target_os = "macos"))]
    run_loop_default(&mut app);

    Ok(())
}

// Non-macOS (and headless macOS) world loop. Exits on CTRL+C, a stopped
// world, or no remaining work.
fn run_loop_default(app: &mut App) {
    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            return;
        }

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => return,
        }
    }
}

// On macOS the Cocoa run loop must be pumped on the main thread each tick so
// AppKit can process window events, draw callbacks, and Metal presentation.
//
// CFRunLoopRunInMode is called with returnAfterSourceHandled=true so it returns
// as soon as one event is processed rather than blocking for the full timeout.
// The outer loop re-enters immediately, draining the event queue before handing
// control to the world step. The shutdown token is checked each iteration so
// CTRL+C exits cleanly.
#[cfg(target_os = "macos")]
fn run_loop_macos(app: &mut App) {
    use core_foundation::runloop::{CFRunLoopRunInMode, kCFRunLoopDefaultMode};

    let shutdown = app.shutdown_token();

    loop {
        if shutdown.is_cancelled() {
            tracing::info!("Shutdown token cancelled, exiting loop");
            return;
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

        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => return,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_directive_matches_build_profile() {
        let expected = if cfg!(debug_assertions) {
            "info"
        } else {
            "warn"
        };
        assert_eq!(default_log_directive(), expected);
    }

    #[test]
    fn default_directive_is_a_valid_filter() {
        // The fallback string must parse as an EnvFilter, otherwise log_filter
        // would panic when RUST_LOG is unset.
        EnvFilter::new(default_log_directive());
    }
}
