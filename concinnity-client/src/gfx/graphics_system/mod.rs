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
    // Frame-rate cap in FPS (GraphicsConfig.fps_cap; 0 = unlimited). Applied live
    // by a CPU frame pacer at the top of each render step, so it is backend-
    // agnostic and needs no trait method. Independent of the quality preset (a
    // user/hardware preference, like vsync). `next_frame_deadline` is the pacer's
    // running target for the next frame's start.
    fps_cap: u32,
    next_frame_deadline: Option<Instant>,
    // Whether a menu view was open last frame. The pacer runs before this
    // frame's menu state is known, so it reads the previous frame's value to
    // clamp the frame rate down to `MENU_FPS_CAP` while a menu is up (no need
    // to render a paused menu at full speed). One frame of lag is invisible.
    menu_active_prev: bool,
    max_frames: Option<u64>,
    shadow_map_size: u32,
    shadow_update: crate::assets::ShadowUpdate,
    // Shadow distance in world units (GraphicsConfig.shadow_distance). Applied
    // live via set_shadow_distance (the per-frame cascade-split math reads it);
    // preset-governed (a manual change flips the master preset to Custom).
    shadow_distance: u32,
    // Active shadow cascade count, 1..=4 (GraphicsConfig.shadow_cascades). Applied
    // live via set_shadow_cascades (the per-frame split + schedule read it);
    // preset-governed (a manual change flips the master preset to Custom).
    shadow_cascades: u32,
    // Scene-sampler max anisotropy. Restart-required (the sampler is built once at
    // backend init from this), so this is display/persist state; the value reaches
    // the backend through the ctor. Preset-governed (a manual change flips the
    // master preset to Custom).
    anisotropy: u32,
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
    // Current upscaler backend (Auto / FSR3 / DLSS / XeSS), seeded at init from
    // the world's PostProcessConfig overridden by any persisted choice. Like
    // render_scale this is restart-required display/persist state (the upscaler
    // is selected + built once at init); DirectX / Vulkan only.
    upscale_backend: crate::assets::UpscalerBackend,
    // The active render backend, constructed during init() and driven each
    // step. Boxed `dyn RenderBackend` so the per-frame logic in init.rs /
    // frame.rs / streaming.rs / scene.rs runs as one cfg-free path across
    // Metal, DirectX, and Vulkan.
    backend: Option<Box<dyn RenderBackend>>,
    // active SceneReel bookkeeping; None when no SceneReel was declared.
    reel: Option<scene_reel::ReelState>,
    // Cursor into the Events<SceneCommand> queue, tracking which scene jumps
    // this system has already applied.
    scene_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<SettingCommand> queue (settings-menu changes:
    // graphics toggles, sliders, key rebinds, volume).
    setting_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<DespawnRequest> queue (runtime entity despawn:
    // cn debug `despawn`, and gameplay-driven removal once that path exists).
    despawn_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<ReparentRequest> queue (runtime re-parenting:
    // cn debug `reparent`, and gameplay-driven moves once that path exists).
    reparent_cmd_cursor: crate::ecs::EventCursor,
    // Cursor into the Events<SpawnRequest> queue (runtime entity spawn: cn debug
    // `spawn`, and gameplay-driven spawning once that path exists).
    spawn_cmd_cursor: crate::ecs::EventCursor,
    // Cumulative elapsed seconds at the previous step, so the step can derive a
    // per-frame dt for the Lifetime countdown.
    prev_elapsed: f32,
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
    // Texture-name map captured at init for runtime decal / emitter spawn to
    // resolve an authored Texture name to its live pool slot. `Some` only under
    // `cn debug`; read-only after init.
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
    // Cycle rows' setting key -> value-label id, captured at init from their
    // `setting:<key>:next` HitRegions (drained by UiInputSystem afterwards). Lets
    // a runtime change relabel a row other than the one clicked: the master
    // "Graphics Quality" preset relabels the quality toggles + render scale it
    // re-derives, and an individual quality-row change relabels the master row.
    cycle_value_labels: std::collections::HashMap<String, AssetId>,
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
    // Coarse GPU performance profile, probed before the backend is built so the
    // auto-config quality ceiling can influence the render targets / effect
    // pipelines sized at backend init. Held in memory only, never persisted.
    gpu_profile: crate::gfx::backend::GpuProfile,
    // The live master "Graphics Quality" preset the settings-menu row cycles.
    // Seeded at init from the persisted choice (or `Auto` on first launch);
    // changing a preset re-derives the quality toggles + render scale under its
    // ceiling, and changing any individual quality row flips this to `Custom`.
    quality_preset: crate::gfx::quality_preset::QualityPreset,
    // The world's authored PostProcessConfig before the user overrides + preset
    // ceiling are applied (defaulted when the world declares none). The pristine
    // baseline a live preset change re-clamps from, so up-shifting a preset
    // restores the world's features and down-shifting clamps them off.
    authored_post_config: crate::assets::PostProcessConfig,
    // Display-output / upscaling preferences (the Display settings rows). Resolved
    // at init from the world's `PostProcessConfig` overridden by any persisted
    // choice, passed to the backend ctor, and held here so the rows display +
    // cycle them. Restart-required (swapchain format / render targets are sized
    // once at init), so a runtime change only persists + relabels; independent of
    // the quality preset.
    temporal_upscaling: bool,
    hdr_display: bool,
    hdr_pq: bool,
    // The world's authored shadow knobs before the user overrides + preset ceiling
    // (defaulted when the world declares no GraphicsConfig). The pristine baseline
    // a live preset change re-clamps from, like `authored_post_config`. The live
    // values are `shadow_map_size` / `shadow_update` above.
    authored_shadow_map_size: u32,
    authored_shadow_update: crate::assets::ShadowUpdate,
    // The world's authored shadow distance, the baseline a live preset change
    // re-clamps from. The live value is `shadow_distance` above.
    authored_shadow_distance: u32,
    // The world's authored shadow cascade count, the baseline a live preset
    // change re-clamps from. The live value is `shadow_cascades` above.
    authored_shadow_cascades: u32,
    // The world's authored anisotropy degree before the user override + preset
    // ceiling, the baseline a live preset change re-clamps from (like
    // `authored_shadow_map_size`). The live value is `anisotropy` above.
    authored_anisotropy: u32,
    // System / streaming restart preferences (the Advanced "Frame Buffering",
    // "Occlusion Culling", and "Texture Quality" rows). Resolved at init from the
    // world's config overridden by any persisted choice, passed to the backend
    // ctor / streamer, and held here so the rows display + cycle them. Restart-
    // required, independent of the quality preset. `frames_in_flight` lives above.
    occlusion_two_pass: bool,
    texture_cap: u32,
    texture_budget: u32,
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
    // Texture asset name -> live pool slot, so runtime decal / emitter spawn
    // (`cn debug`) can resolve an authored Texture name to its slot.
    pub texture_name_to_slot: std::collections::HashMap<AssetId, usize>,
}

// Disjoint mutable view of the `GraphicsSystem` fields the hot-reload passes
// edit in one tick: the active backend, the texture-name map for runtime
// decal / emitter spawn, and the fog bookkeeping the world.jsonl reload pass
// dedupes against. Returned by [`GraphicsSystem::hot_reload_apply_parts`] so the
// binary-only `DebugHook::tick` drive can apply the reload passes from outside
// the per-system step without the library depending on it. The reload catalogue
// + in-flight state live on the debug side (`crate::debug::hot_reload`), built
// from [`HotReloadSources`]. The library never constructs this; hence the
// `dead_code` allowance (the fields are read only from the `cn debug` binary's
// drive, never under `cargo check --lib`).
#[allow(dead_code)]
pub struct HotReloadApplyParts<'a> {
    pub backend: &'a mut dyn RenderBackend,
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
            fps_cap: 0,
            next_frame_deadline: None,
            menu_active_prev: false,
            max_frames: None,
            shadow_map_size: 2048,
            shadow_update: crate::assets::ShadowUpdate::default(),
            shadow_distance: 80,
            shadow_cascades: 4,
            anisotropy: 8,
            failed: false,
            start_time: None,
            frame_count: 0,
            cursor_pos: (0.0, 0.0),
            menu_mode: false,
            render_scale: crate::assets::UpscaleQuality::default(),
            upscale_backend: crate::assets::UpscalerBackend::default(),
            backend: None,
            reel: None,
            scene_cmd_cursor: crate::ecs::EventCursor::default(),
            setting_cmd_cursor: crate::ecs::EventCursor::default(),
            despawn_cmd_cursor: crate::ecs::EventCursor::default(),
            reparent_cmd_cursor: crate::ecs::EventCursor::default(),
            spawn_cmd_cursor: crate::ecs::EventCursor::default(),
            prev_elapsed: 0.0,
            loaded_fonts: std::collections::HashMap::new(),
            texture_streamer: None,
            normal_map_streamer: None,
            mesh_streamer: None,
            mesh_stream_draw_indices: Vec::new(),
            chunk_stream: None,
            pending_hot_reload_sources: None,
            world_reload: None,
            last_fog_settings: None,
            post_process: crate::gfx::render_types::PostProcessParams::DEFAULT,
            // Matches PostProcessConfig's ambient_intensity default; overwritten
            // at init from the world / persisted store.
            ambient_intensity: 1.0,
            // Default until init resolves the world's config + persisted toggles.
            post_config: crate::assets::PostProcessConfig::default(),
            sliders: Vec::new(),
            cycle_value_labels: std::collections::HashMap::new(),
            clip_rects: std::collections::HashMap::new(),
            keymap: crate::gfx::keymap::KeyMap::default(),
            rebind_rows: Vec::new(),
            // All-capable until the backend reports otherwise at init.
            caps: crate::gfx::backend::DeviceCapabilities::ALL,
            // Conservative until probed at init.
            gpu_profile: crate::gfx::backend::GpuProfile::UNKNOWN,
            // Seeded at init from the persisted preset (Auto on first launch).
            quality_preset: crate::gfx::quality_preset::QualityPreset::Auto,
            // Defaulted until init captures the world's authored config.
            authored_post_config: crate::assets::PostProcessConfig::default(),
            // Resolved at init from the world's config + persisted overrides.
            temporal_upscaling: false,
            hdr_display: false,
            hdr_pq: false,
            authored_shadow_map_size: 2048,
            authored_shadow_update: crate::assets::ShadowUpdate::default(),
            authored_shadow_distance: 80,
            authored_shadow_cascades: 4,
            authored_anisotropy: 8,
            occlusion_two_pass: false,
            texture_cap: 96,
            texture_budget: 4,
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

    // Disjoint mutable view of the backend + hot-reload bookkeeping the
    // binary-only `DebugHook::tick` reload drive applies changes through.
    // `None` until the backend is constructed. The library never calls this
    // (the asset hot-reload drive lives in the `cn debug` binary), so it reads
    // as dead code under `cargo check --lib`.
    #[allow(dead_code)]
    pub fn hot_reload_apply_parts(&mut self) -> Option<HotReloadApplyParts<'_>> {
        let backend = self.backend.as_deref_mut()?;
        Some(HotReloadApplyParts {
            backend,
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

// Whether `key` is one of the cycle (dropdown) quality knobs governed by the
// preset ceiling like the boolean toggles (a manual change flips the preset to
// Custom). The set lives in `settings::QUALITY_CYCLE_KEYS`.
pub(super) fn is_quality_cycle(key: &str) -> bool {
    crate::gfx::settings::QUALITY_CYCLE_KEYS.contains(&key)
}

// The current menu option index of cycle quality knob `key` in `cfg`, or `None`
// for a key that is not a cycle quality knob.
pub(super) fn quality_cycle_index(
    cfg: &crate::assets::PostProcessConfig,
    key: &str,
) -> Option<usize> {
    use crate::gfx::settings;
    match key {
        "aa_mode" => Some(settings::aa_mode_index(cfg.aa_mode)),
        "ssgi_resolution" => Some(settings::ssgi_resolution_index(cfg.ssgi_resolution)),
        "ssgi_rays" => Some(settings::ssgi_rays_index(cfg.ssgi_rays)),
        "ssgi_steps" => Some(settings::ssgi_steps_index(cfg.ssgi_steps)),
        "reflection_blur_resolution" => Some(settings::reflection_blur_index(
            cfg.reflection_blur_resolution,
        )),
        _ => None,
    }
}

// Set cycle quality knob `key` in `cfg` from a menu option index. Unknown keys
// are ignored.
pub(super) fn set_quality_cycle(
    cfg: &mut crate::assets::PostProcessConfig,
    key: &str,
    index: usize,
) {
    use crate::gfx::settings;
    match key {
        "aa_mode" => cfg.aa_mode = settings::aa_mode_at(index),
        "ssgi_resolution" => cfg.ssgi_resolution = settings::ssgi_resolution_at(index),
        "ssgi_rays" => cfg.ssgi_rays = settings::ssgi_rays_at(index),
        "ssgi_steps" => cfg.ssgi_steps = settings::ssgi_steps_at(index),
        "reflection_blur_resolution" => {
            cfg.reflection_blur_resolution = settings::reflection_blur_at(index)
        }
        _ => {}
    }
}

// Clamp cycle quality knob `key` in `cfg` DOWN under the ceiling (coarser
// resolution / smaller count; never raises), a no-op when the user explicitly
// overrode it. Shared by the init clamp and the live preset re-derive so both
// produce the same result.
pub(super) fn clamp_quality_cycle(
    cfg: &mut crate::assets::PostProcessConfig,
    key: &str,
    ceiling: &crate::gfx::quality_preset::QualityCeiling,
    overridden: bool,
) {
    if overridden {
        return;
    }
    use crate::gfx::quality_preset::{
        clamp_aa_mode, coarser_reflection_blur, coarser_ssgi_resolution,
    };
    match key {
        "aa_mode" => cfg.aa_mode = clamp_aa_mode(cfg.aa_mode, ceiling.aa_mode),
        "ssgi_resolution" => {
            cfg.ssgi_resolution =
                coarser_ssgi_resolution(cfg.ssgi_resolution, ceiling.ssgi_resolution)
        }
        "ssgi_rays" => cfg.ssgi_rays = cfg.ssgi_rays.min(ceiling.ssgi_rays),
        "ssgi_steps" => cfg.ssgi_steps = cfg.ssgi_steps.min(ceiling.ssgi_steps),
        "reflection_blur_resolution" => {
            cfg.reflection_blur_resolution = coarser_reflection_blur(
                cfg.reflection_blur_resolution,
                ceiling.reflection_blur_resolution,
            )
        }
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
        taa: cfg.aa_mode.taa_enabled(),
        ssao: cfg.ssao_settings(),
        ssr: cfg.ssr_settings(),
        rt_reflections: cfg.rt_reflection_settings(),
        ssgi: cfg.ssgi_settings(),
        reflection_blur_scale: cfg.reflection_blur_divisor(),
        auto_exposure: cfg.auto_exposure_settings(),
        auto_exposure_bias_ev: cfg.exposure_ev,
    }
}

mod despawn;
mod frame;
mod helpers;
pub mod hot_reload_sources;
mod init;
mod scene;
mod spawn;
mod streaming;
