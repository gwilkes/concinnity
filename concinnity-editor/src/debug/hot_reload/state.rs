// src/debug/hot_reload/state.rs
//
// `AssetHotReloadState` (the debug-owned reload catalogue + in-flight decode
// handles + live watcher) plus the off-thread decode result types, the ECS
// side-effect bundle, and `run_frame`, the per-frame entry the debug drive
// calls. Built from `HotReloadSources` (captured in the lib at init).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::gfx::graphics_system::HotReloadApplyParts;
use crate::gfx::graphics_system::hot_reload_sources::*;

use super::decode::{poll_pending_assets, poll_pending_envmap, reload_assets};
use super::passes::{
    reload_procedural_meshes, reload_shader_stages, reload_volumetric_fog, reload_world,
};
use super::watcher::spawn_watcher;

// A worker result still in flight: the receiving end of the channel a
// background reload thread sends its decoded payload on, `None` when idle.
type Inflight<T> = Mutex<Option<std::sync::mpsc::Receiver<T>>>;

// One decoded `Texture` ready for the render thread to push through the
// matching `update_texture_slot` / `update_normal_map_slot` backend call.
// Computed off-thread by the `cn-asset-reload` worker so the (sometimes
// seconds-long) decode never blocks the render loop.
#[derive(Debug)]
pub struct DecodedTexture {
    pub slot: usize,
    pub kind: TextureKind,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    // Source path retained for error logging when the GPU upload fails.
    pub source: String,
}

// One decoded `ColorLut` ready for the render thread to push through
// `update_color_lut`.
#[derive(Debug)]
pub struct DecodedColorLut {
    pub size: u32,
    pub data: Vec<u8>,
    pub source: String,
}

// One decoded static `Mesh`, ready for the render thread to dispatch into
// either an in-place [`crate::gfx::backend::RenderBackend::update_mesh_geometry`]
// per draw slot or a queued
// [`crate::gfx::backend::RenderBackend::rebuild_static_geometry`] when any
// LOD slice's size changed. `entry_idx` indexes into
// `state.meshes.entries` so the applier can look up `draw_indices` for
// fan-out to every `Prop` that shares the mesh.
#[derive(Debug)]
pub struct DecodedMesh {
    pub entry_idx: usize,
    pub vertices: Vec<crate::gfx::mesh_payload::Vertex>,
    pub indices: Vec<u16>,
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
}

// One decoded `SkinnedMesh`, ready for the render thread to dispatch into
// `update_skinned_mesh_geometry`. `entry_idx` indexes into
// `state.skinned_meshes.entries` so the applier can look up the slot's
// init-time `vertex_base` / `vertex_count` / `index_count` / `joint_count`
// and reject shape changes that need a pipeline rebuild.
#[derive(Debug)]
pub struct DecodedSkinnedMesh {
    pub entry_idx: usize,
    pub vertices: Vec<crate::gfx::mesh_payload::SkinnedVertex>,
    pub indices: Vec<u16>,
    pub skeleton: Vec<crate::assets::JointDef>,
}

// Output of one off-thread decode pass: every captured source the worker
// successfully re-read from disk plus a count of the ones that failed (logged
// individually as they happened). The render thread drains this in
// [`poll_pending_assets`] and dispatches each item through the matching
// backend call.
#[derive(Debug, Default)]
pub struct DecodedAssetBatch {
    pub textures: Vec<DecodedTexture>,
    pub color_lut: Option<DecodedColorLut>,
    pub meshes: Vec<DecodedMesh>,
    pub skinned_meshes: Vec<DecodedSkinnedMesh>,
    // Number of source entries that failed to decode (logged per-entry in
    // the worker; tallied here so the apply pass can include them in the
    // summary line).
    pub decode_failures: usize,
}

// Shared `cn debug`-only state: the source map, the optional LUT entry, the
// atomic the engine polls at frame start, and the live watcher handle. The
// watcher pushes events straight into the atomic; `GraphicsSystem` reads +
// clears it each step.
pub struct AssetHotReloadState {
    pub map: TextureSourceMap,
    // Singleton `ColorLut`, when the world declared one with a `source` path.
    // Reloaded alongside the texture map by [`reload_assets`].
    pub color_lut: Option<ColorLutSource>,
    // Singleton `EnvironmentMap`, when the world declared one with a
    // `source` path (procedural `generator` declarations are skipped).
    // Reloaded alongside the texture map by [`reload_assets`].
    pub environment_map: Option<EnvironmentMapSource>,
    // File-backed `Mesh` assets, one entry per Mesh asset (an entry may map
    // to many draw slots when several `Prop`s share the same Mesh).
    pub meshes: MeshSourceMap,
    // File-backed `SkinnedMesh` assets, one entry per skinned draw slot.
    // Each entry carries the slot's vertex region + joint count so the
    // reload helper can reject size-changing reloads and skeleton-shape
    // changes before pushing to the backend.
    pub skinned_meshes: SkinnedMeshSourceMap,
    // `ProceduralMesh` assets whose generator args can be re-applied from a
    // live `world.jsonl`. No file watcher: the trigger is the same
    // `PENDING_WORLD` flag the Prop-diff path consumes.
    pub procedural_meshes: ProceduralMeshSourceMap,
    // World-loaded `ShaderStage` assets whose source files can be re-
    // compiled from disk. The asset watcher recognises `.metal` / `.hlsl`
    // / `.glsl` events against the parent directories of these entries and
    // sets [`super::pending::set_pending_shader_stages`] (separate
    // from the texture / mesh / LUT batch path so a shader save does not
    // also kick a 43-texture re-decode).
    pub shader_stages: ShaderStageSourceMap,
    // Path to the world.jsonl the renderer was initialised from, when
    // known. Used both as a watch-dir source (its parent directory joins
    // the texture / model / HDRI / LUT dirs) and as the file the
    // `world.jsonl` reload pass re-reads from disk. `None` when init came
    // from a stream / blob and no on-disk path exists to watch.
    pub world_jsonl_path: Option<String>,
    // Flipped to `true` by either the `notify` watcher or the debug WS
    // `reload-assets` command. The next `GraphicsSystem::step` consumes
    // it and runs [`reload_assets`].
    pub pending: Arc<AtomicBool>,
    // In-flight EnvironmentMap convolution. `Some(receiver)` while a
    // worker thread is computing the irradiance + prefilter payload;
    // [`poll_pending_envmap`] picks the result up on a later frame and
    // pushes it through `backend.update_environment_map`. The convolution
    // is CPU-bound and takes seconds at default sizes: running it on the
    // render thread previously stalled the frame loop, so the reload now
    // spawns a worker and the render thread keeps drawing while it runs.
    // `Mutex` for interior mutability behind the `&AssetHotReloadState`
    // access pattern; contention is none in practice (single-threaded
    // reads from the render thread).
    pub env_map_inflight: Inflight<Result<Vec<u8>, String>>,
    // In-flight texture + ColorLut + Mesh + SkinnedMesh decode batch.
    // `Some(receiver)` while a `cn-asset-reload` worker is re-decoding
    // every captured source from disk; [`poll_pending_assets`] picks the
    // batch up on a later frame and pushes each entry through the matching
    // backend `update_*` call. Decoding 43 textures + 14 meshes from a
    // shared 43 MB `.glb` used to take ~3 s on the render thread; with the
    // worker the render loop keeps drawing while the decode runs and only
    // stalls for the (fast) GPU upload pass.
    pub asset_batch_inflight: Inflight<DecodedAssetBatch>,
    // SkinnedMesh skeleton-shape changes queued by [`poll_pending_assets`]
    // for the next step to apply to ECS-owned `SkeletonPose` components.
    // The reload helper has no ECS access, so it stashes the (skinned_index,
    // new skeleton) pair here and `GraphicsSystem::step` drains it
    // post-`poll_pending_assets` (via [`drain_pending_skeleton_updates`])
    // and rebuilds the matching pose component. Empty in steady state;
    // non-empty for at most one frame after a skeleton-shape reload.
    pub pending_skeleton_updates: Vec<PendingSkeletonUpdate>,
    // Watcher kept alive purely for its drop semantics; events are delivered
    // via the closure registered at construction.
    #[allow(dead_code)]
    watcher: Option<notify::RecommendedWatcher>,
}

// One skeleton-shape update produced by a skinned hot-reload pass that
// detected a joint-count change. [`poll_pending_assets`] queues these on
// `AssetHotReloadState.pending_skeleton_updates`; the next
// `GraphicsSystem::step` drains them and rebuilds the matching
// `SkeletonPose` component in the ECS so `AnimationSystem` produces the
// right-sized output going forward.
#[derive(Debug)]
pub struct PendingSkeletonUpdate {
    // Backend slot the new skeleton belongs to. Used to find the matching
    // `SkeletonPose` (which carries the same `skinned_index`).
    pub skinned_index: usize,
    // Fresh skeleton built from the re-imported `.glb`'s joint defs.
    pub new_skeleton: crate::gfx::skinning::Skeleton,
}

impl std::fmt::Debug for AssetHotReloadState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let env_inflight = self
            .env_map_inflight
            .try_lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        let batch_inflight = self
            .asset_batch_inflight
            .try_lock()
            .map(|g| g.is_some())
            .unwrap_or(false);
        f.debug_struct("AssetHotReloadState")
            .field("entries", &self.map.entries.len())
            .field("color_lut", &self.color_lut.is_some())
            .field("environment_map", &self.environment_map.is_some())
            .field("meshes", &self.meshes.entries.len())
            .field("skinned_meshes", &self.skinned_meshes.entries.len())
            .field("procedural_meshes", &self.procedural_meshes.entries.len())
            .field("shader_stages", &self.shader_stages.entries.len())
            .field("world_jsonl_path", &self.world_jsonl_path)
            .field("pending", &self.pending.load(Ordering::Relaxed))
            .field("env_map_inflight", &env_inflight)
            .field("asset_batch_inflight", &batch_inflight)
            .field("watcher", &self.watcher.is_some())
            .finish()
    }
}

impl AssetHotReloadState {
    // Build from the init-captured [`HotReloadSources`] bundle. The `cn debug`
    // drive calls this on its first tick after taking the sources off the
    // `GraphicsSystem`.
    pub(crate) fn from_sources(s: HotReloadSources) -> Self {
        Self::new(
            s.map,
            s.color_lut,
            s.environment_map,
            s.meshes,
            s.skinned_meshes,
            s.procedural_meshes,
            s.shader_stages,
            s.world_jsonl_path,
        )
    }

    // Build the state and (best-effort) spawn the `notify` watcher over every
    // unique parent directory of the captured source paths. Watcher creation
    // is best-effort: a missing path or notify error logs and continues;
    // the debug-WS `reload-assets` command still works on the same flag.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        map: TextureSourceMap,
        color_lut: Option<ColorLutSource>,
        environment_map: Option<EnvironmentMapSource>,
        meshes: MeshSourceMap,
        skinned_meshes: SkinnedMeshSourceMap,
        procedural_meshes: ProceduralMeshSourceMap,
        shader_stages: ShaderStageSourceMap,
        world_jsonl_path: Option<String>,
    ) -> Self {
        let pending = Arc::new(AtomicBool::new(false));
        let nothing_to_watch = map.is_empty()
            && color_lut.is_none()
            && environment_map.is_none()
            && meshes.is_empty()
            && skinned_meshes.is_empty()
            && procedural_meshes.is_empty()
            && shader_stages.is_empty()
            && world_jsonl_path.is_none();
        let watcher = if nothing_to_watch {
            None
        } else {
            spawn_watcher(
                &map,
                color_lut.as_ref(),
                environment_map.as_ref(),
                &meshes,
                &skinned_meshes,
                &shader_stages,
                world_jsonl_path.as_deref(),
                Arc::clone(&pending),
            )
        };
        Self {
            map,
            color_lut,
            environment_map,
            meshes,
            skinned_meshes,
            procedural_meshes,
            shader_stages,
            world_jsonl_path,
            pending,
            env_map_inflight: Mutex::new(None),
            asset_batch_inflight: Mutex::new(None),
            pending_skeleton_updates: Vec::new(),
            watcher,
        }
    }

    // Take ownership of any skeleton-shape updates queued by the most
    // recent [`poll_pending_assets`] pass. Drained by
    // `GraphicsSystem::step` so the ECS-owned `SkeletonPose` components
    // can be rebuilt with the new joint hierarchy.
    pub fn drain_pending_skeleton_updates(&mut self) -> Vec<PendingSkeletonUpdate> {
        std::mem::take(&mut self.pending_skeleton_updates)
    }

    // Cheap atomic load; called at the top of `GraphicsSystem::step`.
    pub fn reload_requested(&self) -> bool {
        self.pending.load(Ordering::SeqCst)
    }

    // Clear the flag after a reload pass (success or failure) so a bad source
    // file does not loop forever.
    pub fn clear_flag(&self) {
        self.pending.store(false, Ordering::SeqCst);
    }
}

// ECS-side effects a single hot-reload pass produced that the caller must
// apply to the `World` after dropping its `&mut GraphicsSystem` borrow:
// skeleton-shape changes to splice into `SkeletonPose` components, and Props
// a world.jsonl reload added that must enter the ECS so subsequent systems
// see them. Returned by [`run_frame`] instead of applied in place because the
// reload passes hold the backend + Prop-tracking borrow and have no `World`
// access: the binary-only `DebugHook::tick` drive applies these once the
// system borrow is released.
pub(crate) struct FrameHotReloadEffects {
    pub skeleton_updates: Vec<PendingSkeletonUpdate>,
    pub added_props: Vec<crate::assets::Prop>,
}

// Run every asset / shader / world.jsonl reload pass for one frame and return
// the ECS side-effects. `state` is the debug-owned reload catalogue +
// in-flight handles; `apply` is the per-frame backend + Prop-tracking handle
// from [`GraphicsSystem::hot_reload_apply_parts`](crate::gfx::graphics_system::GraphicsSystem).
// This is the per-frame entry point the `DebugHook::tick` drive calls; it
// holds the logic that previously sat at the top of `GraphicsSystem::run_step`,
// minus the ECS mutation: the caller applies that from the returned
// `FrameHotReloadEffects` once the system borrow is released.
pub(crate) fn run_frame(
    state: &mut AssetHotReloadState,
    apply: &mut HotReloadApplyParts,
) -> FrameHotReloadEffects {
    let mut effects = FrameHotReloadEffects {
        skeleton_updates: Vec::new(),
        added_props: Vec::new(),
    };

    // Asset-payload poll. Pick up any completed off-thread work first so a
    // fresh `reload_assets` below finds the in-flight slots empty. Cheap when
    // nothing is in flight (a Mutex lock + `None` check).
    poll_pending_envmap(state, apply.backend);
    poll_pending_assets(state, apply.backend);
    // Skeleton-shape changes queued by `poll_pending_assets` are applied to the
    // ECS-owned `SkeletonPose` components by the caller.
    effects.skeleton_updates = state.drain_pending_skeleton_updates();
    if state.reload_requested() {
        state.clear_flag();
        // Spawns up to two worker threads (asset decode + envmap convolution);
        // results land on a later frame via the poll calls above.
        reload_assets(state);
    }

    // World-loaded ShaderStage reload poll. The backend builds every
    // replacement into a temporary first and only swaps on success, so a typo
    // in one shader leaves the live pipelines untouched.
    if super::pending::take_pending_shader_stages() {
        let ss_result = reload_shader_stages(&state.shader_stages, apply.backend);
        if ss_result.recompiled > 0 || ss_result.failed > 0 {
            tracing::info!(
                "ShaderStage hot-reload: recompiled={} failed={} pipelines_rebuilt={}",
                ss_result.recompiled,
                ss_result.failed,
                ss_result.pipelines_rebuilt,
            );
        }
    }

    // world.jsonl reload poll: regenerate changed ProceduralMeshes first, then
    // diff Props (transforms / adds / removes / material / cull / parent /
    // scene), then re-apply VolumetricFog. Cheap when the flag is unset.
    if super::pending::take_pending_world() {
        let path = state.world_jsonl_path.clone();
        if let Some(path) = path {
            let pm_result =
                reload_procedural_meshes(&path, &mut state.procedural_meshes, apply.backend);
            if pm_result.regenerated > 0 || pm_result.failed > 0 {
                tracing::info!(
                    "ProceduralMesh hot-reload: regenerated={} unchanged={} failed={}",
                    pm_result.regenerated,
                    pm_result.unchanged,
                    pm_result.failed,
                );
            }
            if apply.init_props.is_some() && apply.world_reload.is_some() {
                let world_reload = apply.world_reload.as_ref().unwrap();
                let tracked_props = apply.init_props.as_mut().unwrap();
                let result = reload_world(
                    &path,
                    tracked_props,
                    apply.prop_parents,
                    apply.prop_draw_indices,
                    apply.prop_scene,
                    world_reload,
                    apply.backend,
                );
                tracing::info!(
                    "world.jsonl hot-reload: transforms={} added={} removed={} \
                     modified={} restart_required={}",
                    result.transforms_applied,
                    result.added,
                    result.removed,
                    result.modified,
                    result.restart_required,
                );
                effects.added_props = result.added_props;
            }
            let fog_result = reload_volumetric_fog(&path, apply.last_fog_settings, apply.backend);
            if fog_result.updated {
                tracing::info!(
                    "VolumetricFog hot-reload: applied ({})",
                    match apply.last_fog_settings {
                        Some(s) => format!(
                            "density={:.3} falloff={:.2} dist={:.0} g={:.2}",
                            s.density, s.height_falloff, s.max_distance, s.phase_g,
                        ),
                        None => "disabled".to_string(),
                    }
                );
            }
        }
    }

    effects
}
