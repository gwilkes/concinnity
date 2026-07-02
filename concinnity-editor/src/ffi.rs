// src/ffi.rs
//
// C-ABI boundary for external clients (today the Swift macOS app; tomorrow
// any C-compatible caller). cbindgen turns this file into
// `include/concinnity.h`, so the function list below is the public
// surface.
//
// Lifecycle / runtime: cn_init, cn_connect, cn_disconnect,
// cn_is_connected, cn_set_server_config.
//
// Build / run: cn_build_world, cn_run_world_blocking,
// cn_run_world_blocking_in_view, cn_stop_world_blocking.
//
// World editing: cn_add, cn_rm, cn_check_world.
//
// Preview (in-tab Metal render): cn_preview_start, cn_preview_stop,
// cn_preview_step.
//
// All cn_* functions must be called from the same thread (the macOS main
// thread). Metal GPU objects inside App are not thread-safe; the Mutex here
// prevents re-entrancy but does not replace the main-thread requirement.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::sync::{Mutex, OnceLock};

use crate::app::state::App;
use crate::app::ws_client::{self, CmdReceiver};
use crate::ecs::StepResult;

struct ClientState {
    app: App,
    ws_rx: Option<CmdReceiver>,
    started: bool,
    // Server URL and account_id set by cn_set_server_config; used by
    // cn_build_world and cn_preview_start to fetch missing asset files.
    server_config: Option<(String, String)>,
    // Separate App instance used for the in-tab preview.
    preview_app: Option<App>,
}

unsafe impl Send for ClientState {}

static STATE: OnceLock<Mutex<ClientState>> = OnceLock::new();

// Set to true by cn_stop_world_blocking(); checked each iteration of run_world_loop_*.
// Cleared at the start of each new run so a stale Stop click can't abort the next session.
static BLOCKING_WORLD_STOP: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// Rendering functions

// Initialize the client state and logging. Call once from the main thread
// before any other cn_* function. Returns 1 on success, 0 on failure.
//
// The log level is not a parameter: it follows the same default as the CLI
// (info for debug builds, warn for release builds) and is overridden by
// RUST_LOG. See crate::app::run::init_logging.
#[unsafe(no_mangle)]
pub extern "C" fn cn_init() -> c_int {
    concinnity_client::app::run::init_logging();

    STATE.get_or_init(|| {
        Mutex::new(ClientState {
            app: App::new(),
            ws_rx: None,
            started: false,
            server_config: None,
            preview_app: None,
        })
    });

    1
}

// Connect the world-command WebSocket to the server. `ws_url` is the full
// WebSocket URL and `account_id` is the authenticated user's account id.
// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_connect(ws_url: *const c_char, account_id: *const c_char) -> c_int {
    if ws_url.is_null() || account_id.is_null() {
        return 0;
    }

    let url = match unsafe { CStr::from_ptr(ws_url) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let aid = match unsafe { CStr::from_ptr(account_id) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };

    let state_mutex = match STATE.get() {
        Some(m) => m,
        None => return 0,
    };
    let mut state = match state_mutex.lock() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    match ws_client::connect(&url, &aid) {
        Ok(rx) => {
            state.ws_rx = Some(rx);
            if !state.started {
                if let Err(e) = state.app.start() {
                    tracing::warn!("app.start() failed (empty world is OK): {:?}", e);
                }
                state.started = true;
            }
            1
        }
        Err(e) => {
            tracing::error!("cn_connect failed: {e}");
            0
        }
    }
}

// Drop the WebSocket connection. Safe to call even when not connected.
#[unsafe(no_mangle)]
pub extern "C" fn cn_disconnect() {
    if let Some(state_mutex) = STATE.get()
        && let Ok(mut state) = state_mutex.lock()
    {
        state.ws_rx = None;
    }
}

// Returns 1 if a WebSocket connection is active, 0 otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn cn_is_connected() -> c_int {
    STATE
        .get()
        .and_then(|m| m.lock().ok())
        .map(|s| if s.ws_rx.is_some() { 1 } else { 0 })
        .unwrap_or(0)
}

// Store the infra server URL and account_id so that cn_build_world and
// cn_preview_start can fetch missing asset files before building. Call
// before cn_build_world or cn_preview_start. Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_set_server_config(server: *const c_char, account_id: *const c_char) -> c_int {
    let server = match ptr_to_str(server) {
        Some(s) => s,
        None => return 0,
    };
    let account_id = match ptr_to_str(account_id) {
        Some(s) => s,
        None => return 0,
    };
    if let Some(state_mutex) = STATE.get()
        && let Ok(mut state) = state_mutex.lock()
    {
        state.server_config = Some((server, account_id));
        return 1;
    }
    0
}

//
// Build and play functions
//

// Validate and compile the world JSONL at `world_jsonl_path`, then write blob
// files to the data/ directory adjacent to assets_path.
// `assets_path` must be an absolute path to the local assets directory
// (e.g. .../Concinnity/assets). Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_build_world(
    assets_path: *const c_char,
    world_jsonl_path: *const c_char,
) -> c_int {
    let assets_path_str = match ptr_to_str(assets_path) {
        Some(s) => s,
        None => return 0,
    };
    let jsonl_path_str = match ptr_to_str(world_jsonl_path) {
        Some(s) => s,
        None => return 0,
    };

    let jsonl = match std::fs::read_to_string(&jsonl_path_str) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cn_build_world: read JSONL failed: {e}");
            return 0;
        }
    };

    let _cwd_guard = match enter_work_dir(&assets_path_str) {
        Some(g) => g,
        None => {
            tracing::error!("cn_build_world: cannot enter work dir");
            return 0;
        }
    };

    let loaded = match concinnity_cook::prepare_world(&jsonl) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("cn_build_world: world validation failed: {}", e.join("; "));
            return 0;
        }
    };

    let result = match concinnity_cook::build_compiled(loaded.assets, None) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("cn_build_world: pipeline failed: {e}");
            return 0;
        }
    };

    if let Err(e) = concinnity_cook::write_build_outputs(&result, &loaded.injected) {
        tracing::error!("cn_build_world: writing blobs/lock failed: {e}");
        return 0;
    }

    tracing::info!("cn_build_world: built world from {jsonl_path_str}");
    1
}

//
// World editing functions
//
// Each editing entry that triggers a rebuild (`cn_add`, `cn_rm`) sets cwd to
// `assets_path.parent()` for the duration of the call so blob writes land in
// the editor's project layout, matching the cn_build_world contract.
// `cn_check_world` is read-only and needs no cwd manipulation.

// Add an asset to `world_jsonl_path` and rebuild. `target` is one of:
//   - a file path (e.g. "/.../models/scene.glb", "/.../shaders/pbr.vert"),
//   - a known asset type name (e.g. "Logger", "Window"),
//   - an inline JSON object string ('{"type": "Window", ...}').
// `name_override` may be NULL; when non-NULL it sets the entry's `name`.
// When the world file does not exist and `target` is a `.glb`, a fresh
// 3D-rendering world is scaffolded at `world_jsonl_path`.
// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_add(
    assets_path: *const c_char,
    world_jsonl_path: *const c_char,
    target: *const c_char,
    name_override: *const c_char,
) -> c_int {
    cn_add_with_template(
        assets_path,
        world_jsonl_path,
        target,
        name_override,
        std::ptr::null(),
    )
}

// Variant of `cn_add` that selects a named scaffold preset (e.g. "showcase")
// when bootstrapping a fresh GLB world. `template` may be NULL to use the
// default scaffold. Kept as a separate symbol so existing `cn_add` callers
// don't need to change.
// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_add_with_template(
    assets_path: *const c_char,
    world_jsonl_path: *const c_char,
    target: *const c_char,
    name_override: *const c_char,
    template: *const c_char,
) -> c_int {
    let assets_path_str = match ptr_to_str(assets_path) {
        Some(s) => s,
        None => return 0,
    };
    let jsonl_path_str = match ptr_to_str(world_jsonl_path) {
        Some(s) => s,
        None => return 0,
    };
    let target_str = match ptr_to_str(target) {
        Some(s) => s,
        None => return 0,
    };
    let name_str = ptr_to_str(name_override);
    let template_str = ptr_to_str(template);

    let _cwd_guard = match enter_work_dir(&assets_path_str) {
        Some(g) => g,
        None => return 0,
    };

    match crate::app::add::add_to_path(
        &jsonl_path_str,
        name_str.as_deref(),
        &target_str,
        template_str.as_deref(),
    ) {
        Ok(()) => {
            tracing::info!("cn_add: added '{target_str}' to {jsonl_path_str}");
            1
        }
        Err(e) => {
            tracing::error!("cn_add: {e}");
            0
        }
    }
}

// Remove the asset named `name` from `world_jsonl_path` and rebuild.
// Returns 1 on success, 0 on failure (including "name not found").
#[unsafe(no_mangle)]
pub extern "C" fn cn_rm(
    assets_path: *const c_char,
    world_jsonl_path: *const c_char,
    name: *const c_char,
) -> c_int {
    let assets_path_str = match ptr_to_str(assets_path) {
        Some(s) => s,
        None => return 0,
    };
    let jsonl_path_str = match ptr_to_str(world_jsonl_path) {
        Some(s) => s,
        None => return 0,
    };
    let name_str = match ptr_to_str(name) {
        Some(s) => s,
        None => return 0,
    };

    let _cwd_guard = match enter_work_dir(&assets_path_str) {
        Some(g) => g,
        None => return 0,
    };

    match crate::app::rm::rm_at_path(&jsonl_path_str, &name_str) {
        Ok(()) => {
            tracing::info!("cn_rm: removed '{name_str}' from {jsonl_path_str}");
            1
        }
        Err(e) => {
            tracing::error!("cn_rm: {e}");
            0
        }
    }
}

// Validate the world JSONL at `world_jsonl_path` without producing blobs.
// Runs the same checks as `cn test`. Returns 1 if every asset passes, 0 on
// any failure (read error, parse error, or validation error).
#[unsafe(no_mangle)]
pub extern "C" fn cn_check_world(world_jsonl_path: *const c_char) -> c_int {
    let jsonl_path_str = match ptr_to_str(world_jsonl_path) {
        Some(s) => s,
        None => return 0,
    };

    match crate::app::check::check_at_path(&jsonl_path_str) {
        Ok(()) => 1,
        Err(e) => {
            tracing::error!("cn_check_world: {e}");
            0
        }
    }
}

//
// Preview functions (embedded in-tab Metal rendering)
//

// Build the world from `world_jsonl_path` and start rendering it embedded
// inside the provided NSView. ns_view must be a valid NSView* on the main thread.
// assets_path is the same absolute path passed to cn_build_world.
// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_preview_start(
    ns_view: *mut std::ffi::c_void,
    assets_path: *const c_char,
    world_jsonl_path: *const c_char,
) -> c_int {
    if ns_view.is_null() {
        return 0;
    }

    let assets_path_str = match ptr_to_str(assets_path) {
        Some(s) => s,
        None => return 0,
    };
    let jsonl_path_str = match ptr_to_str(world_jsonl_path) {
        Some(s) => s,
        None => return 0,
    };

    let jsonl = match std::fs::read_to_string(&jsonl_path_str) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cn_preview_start: read JSONL failed: {e}");
            return 0;
        }
    };

    let state_mutex = match STATE.get() {
        Some(m) => m,
        None => return 0,
    };
    let mut state = match state_mutex.lock() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    state.preview_app = None;

    let _cwd_guard = match enter_work_dir(&assets_path_str) {
        Some(g) => g,
        None => return 0,
    };

    let loaded = match concinnity_cook::prepare_world(&jsonl) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(
                "cn_preview_start: world validation failed: {}",
                e.join("; ")
            );
            return 0;
        }
    };

    let world = match crate::app::build::world_from_loaded(loaded) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!("cn_preview_start: build failed: {e}");
            return 0;
        }
    };

    #[cfg(backend_metal)]
    crate::metal::set_preview_view(ns_view);
    #[cfg(not(backend_metal))]
    let _ = ns_view;

    let mut preview_app = App::new();
    preview_app.load_world(world);
    let result = if let Err(e) = preview_app.start() {
        tracing::error!("cn_preview_start: app.start failed: {:?}", e);
        0
    } else {
        state.preview_app = Some(preview_app);
        tracing::info!("cn_preview_start: preview started from {jsonl_path_str}");
        1
    };

    #[cfg(backend_metal)]
    crate::metal::set_preview_view(std::ptr::null_mut());

    result
}

// Stop the current preview and remove its embedded MTKView from the parent NSView.
#[unsafe(no_mangle)]
pub extern "C" fn cn_preview_stop() {
    if let Some(state_mutex) = STATE.get()
        && let Ok(mut state) = state_mutex.lock()
    {
        state.preview_app = None;
    }
}

// Drive one frame of the preview render loop.
// Returns: 0 = continue, 1 = done, 2 = stop, -1 = no preview active.
#[unsafe(no_mangle)]
pub extern "C" fn cn_preview_step() -> c_int {
    let state_mutex = match STATE.get() {
        Some(m) => m,
        None => return -1,
    };
    let mut state = match state_mutex.lock() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    match &mut state.preview_app {
        None => -1,
        Some(app) => match app.world_step() {
            crate::ecs::StepResult::Continue => 0,
            crate::ecs::StepResult::Done => 1,
            crate::ecs::StepResult::Stop => 2,
        },
    }
}

// Load compiled blobs from data/ and run the world event loop synchronously,
// blocking the calling thread until the world stops (window closed or Escape).
// Uses the same CFRunLoopRunInMode-based loop as the CLI so keyboard events,
// window close, and Metal callbacks are processed correctly on macOS.
// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_run_world_blocking(assets_path: *const c_char) -> c_int {
    let assets_path_str = match ptr_to_str(assets_path) {
        Some(s) => s,
        None => return 0,
    };

    let _cwd_guard = match enter_work_dir(&assets_path_str) {
        Some(g) => g,
        None => return 0,
    };

    let mut app = App::new();
    if let Err(e) = app.load_blob() {
        tracing::error!("cn_run_world_blocking: load_blob failed: {:?}", e);
        return 0;
    }
    if let Err(e) = app.start() {
        tracing::error!("cn_run_world_blocking: app.start failed: {:?}", e);
        return 0;
    }

    tracing::info!("cn_run_world_blocking: running world");

    #[cfg(target_os = "macos")]
    {
        if app.world().renders() {
            run_world_loop_macos(&mut app);
        } else {
            run_world_loop_default(&mut app);
        }
    }

    #[cfg(not(target_os = "macos"))]
    run_world_loop_default(&mut app);

    1
}

// Load compiled blobs and run the world event loop embedded in `ns_view`,
// blocking the calling thread until the world stops or `cn_stop_world_blocking`
// is called. Mirrors `cn_run_world_blocking`'s CFRunLoopRunInMode loop so the
// host process's run-loop observers (e.g. SwiftUI's display refresh) keep
// firing while this is running on the main thread. Renders into the provided
// NSView instead of creating a new NSWindow, so the host app owns window
// lifecycle / focus / chrome.
//
// ns_view must be a valid NSView* on the main thread.
// assets_path is the same absolute path passed to cn_build_world.
// Returns 1 on success, 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn cn_run_world_blocking_in_view(
    ns_view: *mut std::ffi::c_void,
    assets_path: *const c_char,
) -> c_int {
    if ns_view.is_null() {
        return 0;
    }

    let assets_path_str = match ptr_to_str(assets_path) {
        Some(s) => s,
        None => return 0,
    };

    let _cwd_guard = match enter_work_dir(&assets_path_str) {
        Some(g) => g,
        None => return 0,
    };

    #[cfg(backend_metal)]
    {
        crate::metal::set_preview_view(ns_view);
        crate::metal::set_embedded_pump_events(true);
    }

    let mut app = App::new();
    if let Err(e) = app.load_blob() {
        tracing::error!("cn_run_world_blocking_in_view: load_blob failed: {:?}", e);
        return 0;
    }
    if let Err(e) = app.start() {
        tracing::error!("cn_run_world_blocking_in_view: app.start failed: {:?}", e);
        return 0;
    }

    tracing::info!("cn_run_world_blocking_in_view: running world");

    #[cfg(target_os = "macos")]
    {
        if app.world().renders() {
            run_world_loop_macos(&mut app);
        } else {
            run_world_loop_default(&mut app);
        }
    }

    #[cfg(not(target_os = "macos"))]
    run_world_loop_default(&mut app);

    1
}

// Request the active cn_run_world_blocking[_in_view] loop to exit cleanly on
// the next frame.
#[unsafe(no_mangle)]
pub extern "C" fn cn_stop_world_blocking() {
    BLOCKING_WORLD_STOP.store(true, std::sync::atomic::Ordering::Relaxed);
}

//
// Internal helpers
//

fn ptr_to_str(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .ok()
        .map(str::to_string)
}

// Set cwd to `assets_path.parent()` so build artifacts land in the editor's
// project layout. The returned guard restores the previous cwd on drop.
// Mirrors the inline CwdGuard pattern used by cn_build_world / cn_run_world_blocking.
fn enter_work_dir(assets_path: &str) -> Option<CwdGuard> {
    let work_dir = std::path::Path::new(assets_path).parent()?.to_path_buf();
    let prev_dir = std::env::current_dir().ok()?;
    if !work_dir.exists() {
        std::fs::create_dir_all(&work_dir).ok()?;
    }
    std::env::set_current_dir(&work_dir).ok()?;
    Some(CwdGuard(prev_dir))
}

struct CwdGuard(std::path::PathBuf);

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[cfg(target_os = "macos")]
fn run_world_loop_macos(app: &mut App) {
    use core_foundation::runloop::{CFRunLoopRunInMode, kCFRunLoopDefaultMode};
    use std::sync::atomic::Ordering;

    BLOCKING_WORLD_STOP.store(false, Ordering::Relaxed);
    loop {
        loop {
            let result = unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.0, true as u8) };
            if result != 4 {
                break;
            }
        }
        if BLOCKING_WORLD_STOP.load(Ordering::Relaxed) {
            break;
        }
        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => break,
        }
    }
    BLOCKING_WORLD_STOP.store(false, Ordering::Relaxed);
}

fn run_world_loop_default(app: &mut App) {
    use std::sync::atomic::Ordering;

    BLOCKING_WORLD_STOP.store(false, Ordering::Relaxed);
    loop {
        if BLOCKING_WORLD_STOP.load(Ordering::Relaxed) {
            break;
        }
        match app.world_step() {
            StepResult::Continue => {}
            StepResult::Stop | StepResult::Done => break,
        }
    }
    BLOCKING_WORLD_STOP.store(false, Ordering::Relaxed);
}
