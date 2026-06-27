// src/gfx/graphics_system/mod.rs
//
// GraphicsSystem: the 3D renderer driver. An internal system (not a declarable
// asset); `World::start` constructs one when the world declares a
// `GraphicsConfig`. Deliberately a directory rather than a single file; the
// system is large enough that splitting it by responsibility is worth it:
//   mod.rs       struct + System/Debug trait impls (init/step delegate out)
//   init.rs      run_init: one-time backend + draw-list setup
//   frame.rs     run_step: the per-frame encode + streaming drive
//   streaming.rs texture / normal-map / mesh / voxel-world streaming setup
//   scene.rs     scene-reel wiring + scene visibility
//   helpers.rs   shared free functions

use crate::assets::WindowArgs;
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::backend::RenderBackend;
use crate::gfx::{scene_reel, text};
use std::time::Instant;

const IDENTITY4: [[f32; 4]; 4] = crate::gfx::draw_list::IDENTITY4;

// Initialises the GPU backend and draws frame data.
//
// Components drained during init():
//   Window          -- window title, size, and mode
//   GraphicsConfig  -- frames-in-flight, clear color, max frames
//   Mesh            -- raw inline geometry payloads (keyed by asset name)
//   ProceduralMesh  -- generator-built geometry payloads (keyed by asset name)
//   Model           -- multi-mesh model definitions (keyed by asset name)
//   Prop            -- scene objects referencing a Mesh/ProceduralMesh or Model
//   ShaderStage     -- compiled shader payloads (vertex, fragment, shadow)
//   Texture         -- one or more compiled RGBA texture payloads (keyed by asset name)
//
// Components queried (not drained) each step():
//   Camera3D       -- current view matrix and projection parameters
//
// Build process:
//   Each Mesh is deserialized and kept in a name-keyed map. For each Prop the
//   corresponding mesh is looked up and appended to the shared vertex/index
//   buffers; a DrawObject records its slice offsets, model matrix, and texture
//   slot. One implicit DrawObject is also created for any Mesh that has no Prop
//   referencing it (e.g. the room itself), placed at the world origin.
//
// GraphicsSystem deposits a FrameInput component each step after polling the
// backend window. Camera3DSystem drains it to update Camera3D, then writes
// the new view matrix back in time for the next frame.
// Camera3DSystem must run after GraphicsSystem so FrameInput is present.
pub struct GraphicsSystem {
    window_args: WindowArgs,
    clear_color: [f32; 4],
    frames_in_flight: usize,
    vsync: bool,
    max_frames: Option<u64>,
    shadow_map_size: u32,
    shadow_update: crate::assets::ShadowUpdate,
    failed: bool,
    start_time: Option<Instant>,
    frame_count: u64,
    // Latest mouse position (window pixels, top-left origin), captured from the
    // backend each frame. Drives `follow_cursor` sprites, which are positioned
    // a frame after the input that moved them (the draw list is built before
    // the frame's input is polled).
    cursor_pos: (f32, f32),
    // A togglable menu (a View) coexists with a controlled Camera3D. When set,
    // cursor capture is driven each frame by whether a menu view is active
    // (release while open, capture otherwise) rather than fixed at startup.
    menu_mode: bool,
    // Current render-scale (upscaling) quality, seeded at init from the world's
    // PostProcessConfig overridden by any persisted choice. The settings row
    // cycles + persists it; it is restart-required, so this is display/persist
    // state only (the upscaler is sized once at init).
    render_scale: crate::assets::UpscaleQuality,
    // The active render backend, constructed during init() and driven each
    // step. Boxed `dyn RenderBackend` so the per-frame logic in init.rs /
    // frame.rs / streaming.rs / scene.rs runs as one cfg-free path across
    // Metal, DirectX, and Vulkan.
    backend: Option<Box<dyn RenderBackend>>,
    // maps the i-th Prop (in world order) to its DrawObject index/indices in the
    // backend, used to push updated model matrices when props are mutated at runtime.
    // A model-backed prop has multiple entries (one per sub-mesh); a mesh-backed
    // prop has exactly one.
    prop_draw_indices: Vec<Vec<usize>>,
    // parent index (into the same prop list) for each prop; None = world-space root.
    prop_parents: Vec<Option<usize>>,
    // scene each prop belongs to (resolved at build time), or None = always visible.
    prop_scene: Vec<Option<AssetId>>,
    // Temporary toggle (read at init from the DecomposedRender resource): when
    // on, the per-frame model-matrix push is driven from per-entity
    // GlobalTransform + RenderHandle components instead of the positional prop
    // side-tables. Both paths must render identically; this field and the prop
    // side-tables are removed once the decomposed path becomes the only one.
    decomposed_render: bool,
    // active SceneReel bookkeeping; None when no SceneReel was declared.
    reel: Option<scene_reel::ReelState>,
    // Cursor into the Events<SceneCommand> queue, tracking which scene jumps
    // this system has already applied.
    scene_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<SettingCommand> queue (settings-menu changes:
    // graphics toggles, sliders, key rebinds, volume).
    setting_cmd_cursor: crate::ecs::EventCursor,
    // Font atlas data, keyed by asset id, built during init().
    loaded_fonts: std::collections::HashMap<AssetId, text::LoadedFont>,
    // Asset-streaming subsystem for the albedo texture pool. Some only when a
    // StreamingConfig was declared and the backend supports it (Metal); None
    // means every texture was uploaded eagerly at init, as before. Read only
    // by the Metal step path -- dead on backends without streaming yet.
    #[allow(dead_code)]
    texture_streamer: Option<crate::app::texture_stream::TextureStreamer>,
    // Asset-streaming subsystem for the normal-map texture pool. A second
    // TextureStreamer instance: streamed item `i` is normal-map pool slot
    // `i + 1` (slot 0 is the never-streamed flat-normal fallback). Some only
    // when a StreamingConfig was declared and the backend supports it (Metal).
    #[allow(dead_code)]
    normal_map_streamer: Option<crate::app::texture_stream::TextureStreamer>,
    // Mesh-geometry streaming subsystem. Some only when a StreamingConfig was
    // declared and the backend supports it (Metal); None means every mesh was
    // uploaded eagerly at init, as before. Read only by the Metal step path.
    #[allow(dead_code)]
    mesh_streamer: Option<crate::app::mesh_stream::MeshStreamer>,
    // Maps a streamed mesh's id to its DrawObject index, so completed loads
    // and evictions are applied to the right draw. Empty when not streaming.
    #[allow(dead_code)]
    mesh_stream_draw_indices: Vec<usize>,
    // Infinite voxel-world chunk streaming. Some only when a VoxelWorld was
    // declared and the backend supports it (Metal). Read only by the Metal
    // step path.
    #[allow(dead_code)]
    chunk_stream: Option<ChunkStreamState>,
    // Source catalogues captured at init for asset hot-reload, handed off to
    // the `cn debug` binary's reload machinery (which owns the watcher + the
    // live `AssetHotReloadState`). `Some` only under `cn debug` with at least
    // one file-backed asset / world.jsonl; taken once by the debug drive via
    // `take_hot_reload_sources`. `cn run` never captures these; production
    // reads asset payloads from the compiled blob and never re-touches disk.
    pending_hot_reload_sources: Option<hot_reload_sources::HotReloadSources>,
    // Init-time owned clones of every `Prop`, in the order they were queried
    // by the draw-list builder. `Some` only under `cn debug`, captured so a
    // world.jsonl edit can rebuild a same-order `Vec<Prop>` with the new
    // transforms and feed it back into `compute_world_matrices`. `None`
    // means hot-reload was off at init.
    //
    // The vec grows with `ctx.push(Prop)` when a hot-reloaded world.jsonl
    // adds a Prop. `prop_parents` / `prop_draw_indices` / `prop_scene` are
    // grown in lockstep so the per-frame transform loop indexes correctly.
    init_props: Option<Vec<crate::assets::Prop>>,
    // Auxiliary maps captured at init for the world.jsonl hot-reload pass to
    // resolve a new Prop's material / texture / mesh / model references
    // without re-running the build pipeline. `Some` only under `cn debug`.
    // Read-only after init; new assets in the world.jsonl reload are rejected
    // (those need a process restart).
    world_reload: Option<WorldReloadState>,
    // Last `VolumetricFog` settings pushed to the backend, used by the
    // world.jsonl reload pass to dedupe: if the resolved value matches what's
    // already live, the reload skips the trait call and the log entry. Tracks
    // both `None` (no fog / disabled) and `Some(settings)`. Initialised by
    // `run_init` to whatever was passed into the backend constructor.
    last_fog_settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    // Live post-process parameters (bloom / exposure / vignette / LUT blend),
    // the source of truth for slider settings. Seeded at init from the world's
    // resolved PostProcessConfig (with any persisted overrides applied); a
    // slider drag mutates a field here and pushes the whole struct to the
    // backend via `update_post_process`.
    post_process: crate::gfx::render_types::PostProcessParams,
    // Live ambient (IBL) light scale, the source of truth for the Ambient
    // slider. Lives in the backend's `LightUniforms` (not `PostProcessParams`),
    // so it is held + pushed separately via `set_ambient_intensity`. Seeded at
    // init from the world's `PostProcessConfig.ambient_intensity` (with any
    // persisted override applied) and pushed to the backend once after it is
    // built.
    ambient_intensity: f32,
    // The world's resolved PostProcessConfig with the user's persisted
    // quality-toggle overrides applied (defaulted when the world declares none).
    // The source of truth for the Quality-group toggles: a toggle flips the
    // matching field here, re-derives the per-feature settings, and pushes them
    // to the backend's live rebuild. The non-toggle fields (exposure, bloom,
    // ambient) keep their authored values here; the sliders own those via
    // `post_process` / `ambient_intensity` instead.
    post_config: crate::assets::PostProcessConfig,
    // Slider rows in the world, captured at init from their drag HitRegions +
    // handle Sprites. Drives the handle position + value-label update when a
    // slider changes, and the one-time sync of both to the live value at init.
    sliders: Vec<SliderViz>,
    // Per-element clip bands (reference space) captured at init from the world's
    // ScrollPanels: each scroll-content element id maps to its panel's content
    // band, so the draw path scissors it and off-band rows do not bleed over the
    // panel chrome. Empty when no ScrollPanel was declared.
    clip_rects: std::collections::HashMap<AssetId, [f32; 4]>,
    // Live gameplay movement key map (the source of truth for the Controls-tab
    // rebind rows). Seeded at init from the persisted `ControlsSettings.keymap`
    // or the engine default, pushed to the backend once after it is built, and
    // updated (with a swap) + re-pushed + persisted on each rebind.
    keymap: crate::gfx::keymap::KeyMap,
    // Rebind rows in the world, captured at init from their `setting:key_*:rebind`
    // HitRegions. Maps each rebindable action to its value `TextLabel`, so a
    // rebind (and the swap it may trigger) can refresh both affected row labels.
    rebind_rows: Vec<RebindViz>,
    // Device capability flags, queried from the backend once it is built. Drives
    // the capability gating at init: a settings row whose feature the device
    // cannot provide (e.g. ray-traced reflections without hardware ray tracing)
    // is grayed out and made inert. Held in memory only, never persisted.
    caps: crate::gfx::backend::DeviceCapabilities,
}

// One key-rebind row's runtime bookkeeping: the action it rebinds and the value
// `TextLabel` showing its bound key. Built at init from the row's
// `setting:key_*:rebind` HitRegion (`action` -> `Bindable`, `label`).
struct RebindViz {
    action: crate::gfx::keymap::Bindable,
    value_id: AssetId,
}

// One slider row's runtime bookkeeping: the engine setting it controls, the
// track geometry it maps a fraction onto, and the handle Sprite + value
// TextLabel it drives. Built at init from the row's `setting:<key>:drag`
// HitRegion (track `x`/`width`, `label`, `drag_handle`) and the handle Sprite's
// width.
struct SliderViz {
    key: String,
    track_x: f32,
    track_w: f32,
    handle_w: f32,
    handle_id: AssetId,
    value_id: AssetId,
}

// Init-time asset-resolution tables consulted by the world.jsonl hot-reload
// pass when applying adds and non-transform edits. Captured at init and
// never mutated afterwards: the reload path cannot introduce new
// Materials / Textures / Meshes / Models on the fly (those need a process
// restart), but every authored Prop that points at an asset already in the
// init world resolves through these maps without re-running build.
// Built by init, read only by the `cn debug` binary's world.jsonl reload pass,
// so its fields read as dead under `cargo check --lib`.
#[allow(dead_code)]
pub struct WorldReloadState {
    pub material_map: std::collections::HashMap<
        AssetId,
        (usize, usize, crate::gfx::render_types::MaterialUniforms),
    >,
    pub texture_name_to_slot: std::collections::HashMap<AssetId, usize>,
    pub model_map: std::collections::HashMap<AssetId, Vec<crate::assets::SubMeshRef>>,
    // One example draw slot per mesh AssetId; the clone-static-draw-object
    // path copies geometry (vertex/index regions, base_vertex, LOD slices)
    // from this draw to seed a new prop's draw, so any draw that came from
    // the same mesh works as a template.
    pub mesh_id_to_draw: std::collections::HashMap<AssetId, usize>,
    // Scene names declared at init (raw strings, in declaration order).
    // Used by world.jsonl hot-reload to apply the same `<scene>_*` prefix
    // resolution that [`crate::build::pipeline::resolve_scene_refs`] does at
    // build time, so a hot-reload-added Prop ends up in the right scene.
    pub scene_names: Vec<String>,
}

// Disjoint mutable view of the `GraphicsSystem` fields the hot-reload passes
// edit in one tick: the active backend, plus the Prop-tracking + fog
// bookkeeping the world.jsonl reload pass mutates in place. Returned by
// [`GraphicsSystem::hot_reload_apply_parts`] so the binary-only
// `DebugHook::tick` drive can apply the reload passes from outside the
// per-system step without the library depending on it. The reload catalogue +
// in-flight state live on the debug side (`crate::debug::hot_reload`), built
// from [`HotReloadSources`]. The library never constructs this; hence the
// `dead_code` allowance (the fields are read only from the `cn debug` binary's
// drive, never under `cargo check --lib`).
#[allow(dead_code)]
pub struct HotReloadApplyParts<'a> {
    pub backend: &'a mut dyn RenderBackend,
    // Init-order Prop snapshot the world.jsonl diff rebuilds against. Grows
    // with the diff's adds; `prop_parents` / `prop_draw_indices` /
    // `prop_scene` grow in lockstep.
    pub init_props: &'a mut Option<Vec<crate::assets::Prop>>,
    pub prop_parents: &'a mut Vec<Option<usize>>,
    pub prop_draw_indices: &'a mut Vec<Vec<usize>>,
    pub prop_scene: &'a mut Vec<Option<AssetId>>,
    pub world_reload: &'a Option<WorldReloadState>,
    pub last_fog_settings: &'a mut Option<crate::gfx::volumetric_fog::FogSettings>,
}

// Runtime state for streaming an infinite `VoxelWorld`: the chunk streamer,
// the resident chunk-to-draw-index map, and the per-chunk render parameters
// (chunk size for the camera-to-chunk mapping and model placement, plus the
// shared material every chunk draws with).
// Most fields are read only by the Metal chunk-streaming arms of `step` /
// `setup_*`; on non-macOS builds the struct is still constructed but
// those fields go unread.
#[cfg_attr(not(backend_metal), allow(dead_code))]
struct ChunkStreamState {
    streamer: crate::app::chunk_stream::ChunkStreamer,
    // Maps a resident chunk's coordinate to its `DrawObject` index.
    draws: std::collections::BTreeMap<crate::gfx::chunk_coord::ChunkCoord, usize>,
    chunk_w: f32,
    chunk_d: f32,
    // Render origin for camera-relative rendering: the chunk every resident
    // chunk's model matrix is currently placed relative to. It follows the
    // camera's chunk; when it changes the resident chunks are rebased onto the
    // new origin.
    origin_chunk: crate::gfx::chunk_coord::ChunkCoord,
    texture_slot: usize,
    normal_map_slot: usize,
    material: crate::gfx::render_types::MaterialUniforms,
}

// `(resident, pending, unloaded)` counts for each streaming pool, or `None`
// when that pool is not streaming. Captured by the debug server's `streaming`
// command for headless verification. Only the `cn debug` binary's `debug`
// module consumes it, so it reads as dead code in a plain library build.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    pub texture: Option<(usize, usize, usize)>,
    pub normal_map: Option<(usize, usize, usize)>,
    pub mesh: Option<(usize, usize, usize)>,
    // `(resident, pending)` chunk counts when a `VoxelWorld` is streaming.
    pub chunk: Option<(usize, usize)>,
}

impl std::fmt::Debug for GraphicsSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphicsSystem")
            .field("frame_count", &self.frame_count)
            .field("failed", &self.failed)
            .finish()
    }
}

impl GraphicsSystem {
    // Fresh renderer driver with no backend yet. Config (frames-in-flight,
    // clear color, `max_frames`, shadow-map size) is read from the world's
    // `GraphicsConfig` in [`System::init`]; the validation request comes from
    // the CLI via `dev_flags`.
    pub fn new() -> Self {
        Self {
            window_args: Default::default(),
            clear_color: [0.01, 0.01, 0.02, 1.0],
            frames_in_flight: 2,
            vsync: false,
            max_frames: None,
            shadow_map_size: 2048,
            shadow_update: crate::assets::ShadowUpdate::default(),
            failed: false,
            start_time: None,
            frame_count: 0,
            cursor_pos: (0.0, 0.0),
            menu_mode: false,
            render_scale: crate::assets::UpscaleQuality::default(),
            backend: None,
            prop_draw_indices: Vec::new(),
            prop_parents: Vec::new(),
            prop_scene: Vec::new(),
            decomposed_render: false,
            reel: None,
            scene_cmd_cursor: crate::ecs::EventCursor::default(),
            setting_cmd_cursor: crate::ecs::EventCursor::default(),
            loaded_fonts: std::collections::HashMap::new(),
            texture_streamer: None,
            normal_map_streamer: None,
            mesh_streamer: None,
            mesh_stream_draw_indices: Vec::new(),
            chunk_stream: None,
            pending_hot_reload_sources: None,
            init_props: None,
            world_reload: None,
            last_fog_settings: None,
            post_process: crate::gfx::render_types::PostProcessParams::DEFAULT,
            // Matches PostProcessConfig's ambient_intensity default; overwritten
            // at init from the world / persisted store.
            ambient_intensity: 1.0,
            // Default until init resolves the world's config + persisted toggles.
            post_config: crate::assets::PostProcessConfig::default(),
            sliders: Vec::new(),
            clip_rects: std::collections::HashMap::new(),
            keymap: crate::gfx::keymap::KeyMap::default(),
            rebind_rows: Vec::new(),
            // All-capable until the backend reports otherwise at init.
            caps: crate::gfx::backend::DeviceCapabilities::ALL,
        }
    }
}

impl Default for GraphicsSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl System for GraphicsSystem {
    fn init(&mut self, ctx: &mut PipelineContext) {
        self.decomposed_render = crate::ecs::decompose::decomposed_render_enabled(ctx);
        self.run_init(ctx);
    }

    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        self.run_step(ctx)
    }
}

impl GraphicsSystem {
    // Return the current logical viewport size for text centering. Returns (0,0)
    // when the backend is not yet initialised; non-Metal backends default to
    // (0,0) since they don't implement logical_size.
    fn viewport_size(&self) -> (f32, f32) {
        match &self.backend {
            Some(b) => b.logical_size(),
            None => (0.0, 0.0),
        }
    }

    fn wait_idle(&mut self) {
        if let Some(b) = &self.backend {
            b.wait_idle();
        }
    }

    // Disjoint mutable view of the backend + Prop-tracking fields the
    // binary-only `DebugHook::tick` reload drive applies changes through.
    // `None` until the backend is constructed. The library never calls this
    // (the asset hot-reload drive lives in the `cn debug` binary), so it reads
    // as dead code under `cargo check --lib`.
    #[allow(dead_code)]
    pub fn hot_reload_apply_parts(&mut self) -> Option<HotReloadApplyParts<'_>> {
        let backend = self.backend.as_deref_mut()?;
        Some(HotReloadApplyParts {
            backend,
            init_props: &mut self.init_props,
            prop_parents: &mut self.prop_parents,
            prop_draw_indices: &mut self.prop_draw_indices,
            prop_scene: &mut self.prop_scene,
            world_reload: &self.world_reload,
            last_fog_settings: &mut self.last_fog_settings,
        })
    }

    // Take the init-captured hot-reload source catalogues, leaving `None`
    // behind. The `cn debug` drive calls this once on its first tick to build
    // the filesystem watcher + `AssetHotReloadState`. `None` under `cn run`,
    // or when no file-backed asset / world.jsonl was declared. Library-dead
    // (only the binary calls it).
    #[allow(dead_code)]
    pub fn take_hot_reload_sources(&mut self) -> Option<hot_reload_sources::HotReloadSources> {
        self.pending_hot_reload_sources.take()
    }

    // Stand up the albedo-texture streaming subsystem when a StreamingConfig
    // was declared. Every streamable slot is evicted to a placeholder now; the
    // streamer brings them back resident over the next frames, nearest first.
    //
    // The payload source depends on where the world came from: a disk-backed
    // `cn run` world re-reads each payload from its blob file (no RAM copy), an
    // in-memory `cn debug` world keeps the payloads RAM-resident.
}

// Quality-toggle plumbing shared by init (value-label sync + initial overlay)
// and the per-frame drain. Centralising the key -> `PostProcessConfig` field
// mapping here keeps the three call sites (read state, flip state, derive the
// backend settings) from drifting apart.

// The current on/off state of quality toggle `key` in `cfg`, or `None` for a
// key that is not a quality toggle.
pub(super) fn quality_toggle_on(cfg: &crate::assets::PostProcessConfig, key: &str) -> Option<bool> {
    match key {
        "taa" => Some(cfg.taa),
        "ssao" => Some(cfg.ssao),
        "ssr" => Some(cfg.ssr),
        "ray_traced_reflections" => Some(cfg.ray_traced_reflections),
        "ssgi" => Some(cfg.indirect_lighting == crate::assets::IndirectLighting::Ssgi),
        "auto_exposure" => Some(cfg.auto_exposure),
        _ => None,
    }
}

// Flip quality toggle `key` to `on` in `cfg`. Unknown keys are ignored.
pub(super) fn set_quality_toggle(cfg: &mut crate::assets::PostProcessConfig, key: &str, on: bool) {
    match key {
        "taa" => cfg.taa = on,
        "ssao" => cfg.ssao = on,
        "ssr" => cfg.ssr = on,
        "ray_traced_reflections" => cfg.ray_traced_reflections = on,
        "ssgi" => {
            cfg.indirect_lighting = if on {
                crate::assets::IndirectLighting::Ssgi
            } else {
                crate::assets::IndirectLighting::Ibl
            }
        }
        "auto_exposure" => cfg.auto_exposure = on,
        _ => {}
    }
}

// Derive the backend's per-feature `QualitySettings` from a resolved config.
// Mirrors the init-time derivation (the same `*_settings()` methods), so a
// live rebuild reproduces exactly what a launch with this config would build.
pub(super) fn derive_quality_settings(
    cfg: &crate::assets::PostProcessConfig,
) -> crate::gfx::backend::QualitySettings {
    crate::gfx::backend::QualitySettings {
        taa: cfg.taa,
        ssao: cfg.ssao_settings(),
        ssr: cfg.ssr_settings(),
        rt_reflections: cfg.rt_reflection_settings(),
        ssgi: cfg.ssgi_settings(),
        reflection_blur_scale: cfg.reflection_blur_divisor(),
        auto_exposure: cfg.auto_exposure_settings(),
        auto_exposure_bias_ev: cfg.exposure_ev,
    }
}

mod frame;
mod helpers;
pub mod hot_reload_sources;
mod init;
mod scene;
mod streaming;
