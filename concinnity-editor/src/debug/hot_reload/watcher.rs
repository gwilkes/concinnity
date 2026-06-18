// src/debug/hot_reload/watcher.rs
//
// Filesystem watcher: subscribes to the parent directories of every captured
// source path and flips the shared atomic on a relevant change. Mirrors the
// per-backend shader watcher.

use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::gfx::graphics_system::hot_reload_sources::*;

// Spawn the watcher. Mirrors the shader-watcher pattern in
// `concinnity_client::metal::hot_reload`: 150 ms debounce, only
// modify/create/remove events fire the flag, only relevant extensions count.
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn_watcher(
    map: &TextureSourceMap,
    color_lut: Option<&ColorLutSource>,
    environment_map: Option<&EnvironmentMapSource>,
    meshes: &MeshSourceMap,
    skinned_meshes: &SkinnedMeshSourceMap,
    shader_stages: &ShaderStageSourceMap,
    world_jsonl_path: Option<&str>,
    flag: Arc<AtomicBool>,
) -> Option<notify::RecommendedWatcher> {
    let debounce = Duration::from_millis(150);
    let last_fire = Mutex::new(Instant::now() - debounce);
    let mut watcher = match notify::recommended_watcher(move |res: notify::Result<Event>| {
        let event = match res {
            Ok(e) => e,
            Err(e) => {
                tracing::debug!("asset hot-reload watcher error: {e}");
                return;
            }
        };
        if !is_asset_event(&event) {
            return;
        }
        let mut last = match last_fire.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = Instant::now();
        if now.duration_since(*last) < debounce {
            return;
        }
        *last = now;
        tracing::info!(
            "asset hot-reload: detected change to {:?}, scheduling asset reload",
            event.paths
        );
        // `.jsonl` events kick the world-reload poll and skip the asset
        // reload: the backend asset payloads (textures, IBL, meshes,
        // skinned, animations) don't live in the JSONL, only Prop
        // transforms and the asset-graph topology do. `.metal` / `.hlsl`
        // / `.glsl` events kick the world-loaded ShaderStage reload pass
        // alone (recompile + pipeline rebuild, no texture / mesh decode
        // is needed). Everything else (`.glb` / `.png` / `.hdr` / `.cube`)
        // kicks the asset reload + the animation reload but not the world
        // or shader passes.
        let is_shader_event = event.paths.iter().any(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(is_shader_extension)
                .unwrap_or(false)
        });
        let is_jsonl_event = event.paths.iter().any(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("jsonl"))
                .unwrap_or(false)
        });
        if is_shader_event {
            super::set_pending_shader_stages();
        } else if is_jsonl_event {
            super::set_pending_world();
        } else {
            flag.store(true, Ordering::SeqCst);
            // AnimationSystem subscribes via a sibling static flag in
            // crate::app::dev_flags; the asset map lives on
            // GraphicsSystem so a separate signal is the simplest way to
            // notify the animation graph of the same `.glb` save without
            // plumbing a shared Arc.
            crate::app::dev_flags::set_pending_animations();
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("asset hot-reload: failed to create notify watcher: {e}");
            return None;
        }
    };

    // Build the unique-dir set across textures + the optional LUT + the
    // optional EnvironmentMap so the watcher subscribes to each path at most
    // once.
    let mut dirs: BTreeSet<PathBuf> = map.watch_dirs().into_iter().collect();
    if let Some(lut) = color_lut
        && let Some(parent) = Path::new(&lut.resolved_path).parent()
        && !parent.as_os_str().is_empty()
    {
        dirs.insert(parent.to_path_buf());
    }
    if let Some(env_map) = environment_map
        && let Some(parent) = Path::new(&env_map.resolved_path).parent()
        && !parent.as_os_str().is_empty()
    {
        dirs.insert(parent.to_path_buf());
    }
    for dir in meshes.watch_dirs() {
        dirs.insert(dir);
    }
    for dir in skinned_meshes.watch_dirs() {
        dirs.insert(dir);
    }
    for dir in shader_stages.watch_dirs() {
        dirs.insert(dir);
    }
    if let Some(path) = world_jsonl_path
        && let Some(parent) = Path::new(path).parent()
    {
        // An empty parent means the path was already a bare filename in
        // CWD; subscribe to "." in that case so the same `notify` events
        // fire as for any directoried path.
        let dir = if parent.as_os_str().is_empty() {
            PathBuf::from(".")
        } else {
            parent.to_path_buf()
        };
        dirs.insert(dir);
    }
    let mut any_watched = false;
    for dir in dirs {
        match watcher.watch(&dir, RecursiveMode::NonRecursive) {
            Ok(()) => {
                tracing::info!(
                    "asset hot-reload: watching {} for asset source changes",
                    dir.display()
                );
                any_watched = true;
            }
            Err(e) => {
                tracing::warn!(
                    "asset hot-reload: failed to watch {} ({}); assets sourced from \
                     that directory will need a manual `reload-assets` to refresh",
                    dir.display(),
                    e
                );
            }
        }
    }
    if any_watched {
        Some(watcher)
    } else {
        // None of the directories could be watched (likely a packaged binary
        // run from outside its checkout). The debug command path still works.
        None
    }
}

// Filter notify events down to those that should kick a reload. We don't
// peek at the exact path against the map here because notify often reports
// paths via temp files / rename sidecars; a coarse extension match is good
// enough at V1 scale and gets debounced anyway. `.cube` covers ColorLut
// sources, `.hdr` covers EnvironmentMap sources, `.jsonl` covers the world
// file, alongside the texture / mesh extensions.
pub(super) fn is_asset_event(event: &Event) -> bool {
    if !matches!(
        event.kind,
        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
    ) {
        return false;
    }
    event.paths.iter().any(|p| {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(
            ext.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "glb" | "cube" | "hdr" | "jsonl"
        ) || is_shader_extension(ext)
    })
}

// True for the shader-source extensions recognised by world-loaded
// `ShaderStage` hot-reload. Case-insensitive so a `.METAL` save still
// triggers the rebuild. `.metal` files in the engine's bundled shader
// directory are handled by a separate watcher in
// [`crate::metal::hot_reload`]; the asset watcher here only subscribes
// to the parent directories of *captured* `ShaderStage` sources, so the
// two watchers never observe the same file even though they share an
// extension list.
pub(super) fn is_shader_extension(ext: &str) -> bool {
    matches!(ext.to_ascii_lowercase().as_str(), "metal" | "hlsl" | "glsl")
}
