// src/gfx/backend.rs
//
// RenderBackend trait: the union of methods every graphics backend
// implements, dispatched dynamically by GraphicsSystem so the per-frame
// step + setup logic lives in one cfg-free copy instead of three.
//
// Each concrete backend (MtlContext / DxContext / VkContext) supplies a
// thin forwarder impl that delegates to the existing inherent methods
// (see metal/backend.rs, directx/backend.rs, vulkan/backend.rs).
//
// Two cross-backend signature variances are handled here:
//   - `upload_skinned`: Metal uses three shader payloads (vert + frag +
//     shadow); DX/VK use one (frag). The trait method takes all three;
//     DX/VK ignore the unused bytes.
//   - `setup_chunk_streaming`: Metal binds chunk textures per draw and
//     ignores the (texture_slot, normal_map_slot) args; DX/VK bake them
//     into a shared descriptor at setup time.
//
// `logical_size` and `render_stats` are Metal-only today and have default
// no-op impls so DX/VK don't need to override them.

use crate::gfx::auto_exposure::AutoExposureSettings;
use crate::gfx::input::RenderInput;
use crate::gfx::keymap::KeyMap;
use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::profile::RenderStats;
use crate::gfx::render_types::{
    MaterialUniforms, PostProcessParams, SkinnedDrawObject, TextDrawCall,
};
use crate::gfx::rt_reflections::RtReflectionSettings;
use crate::gfx::scene_reel::SceneControl;
use crate::gfx::ssao::SsaoSettings;
use crate::gfx::ssgi::SsgiSettings;
use crate::gfx::ssr::SsrSettings;
use crate::gfx::volumetric_fog::FogSettings;

// One draw slot's fresh geometry, supplied to
// [`RenderBackend::rebuild_static_geometry`] when an asset hot-reload
// changed its vertex / index count and the slot can no longer hold the new
// data in place. The backend rebuilds the entire shared vertex / index
// buffer; draws not named here keep their current geometry, copied byte-for-
// byte from the live buffers. `indices` are mesh-relative (0-based); the
// backend rebases them onto whatever new vertex region the draw lands in.
#[allow(dead_code)] // consumed by Metal's rebuild_static_geometry; no-op on DirectX / Vulkan.
pub struct DrawGeometryUpdate {
    pub draw_idx: usize,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    // One slice per additional LOD, ordered mip 0 → mip N-1. Each is
    // `(switch_distance, mesh-relative indices)`. Empty for meshes
    // declared `lod_levels <= 1`.
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
}

// One skinned draw slot's fresh geometry, supplied to
// [`RenderBackend::rebuild_skinned_geometry`] when an asset hot-reload
// changed its vertex / index count and the slot can no longer hold the new
// data in its existing region of the shared skinned vertex / index buffers.
// The backend rebuilds both shared buffers; slots not named here keep their
// current geometry, copied byte-for-byte from the live buffers and re-based
// onto whatever new vertex region they land in. `indices` are mesh-relative
// (0-based); the backend rebases them onto the new vertex region.
#[allow(dead_code)] // consumed by Metal's rebuild_skinned_geometry; no-op on DirectX / Vulkan.
pub struct SkinnedDrawGeometryUpdate {
    pub skinned_index: usize,
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u16>,
}

// The post-rebuild layout for one skinned slot, returned by
// [`RenderBackend::rebuild_skinned_geometry`] so the asset hot-reload
// helper can refresh its `SkinnedMeshSourceEntry`s'
// `vertex_base` / `vertex_count` / `index_count` to point at the new
// regions. Returned for every slot (both the ones whose geometry was
// replaced and the ones whose geometry was carried over) because the
// rebuild may have shifted every slot's `vertex_base`.
// Constructed only by the `cn debug` binary's skinned-rebuild reload pass;
// reads as dead under `cargo check --lib`.
#[allow(dead_code)]
pub struct SkinnedSlotLayout {
    pub skinned_index: usize,
    pub vertex_base: u16,
    pub vertex_count: usize,
    pub index_count: usize,
}

// The resolved per-feature quality settings for [`RenderBackend::apply_quality_settings`].
// `GraphicsSystem` derives these from its stored `PostProcessConfig` (with the
// user's persisted toggle overrides applied) whenever a Quality-group toggle
// changes, so the backend receives ready-to-use settings rather than re-deriving
// from the asset. Each `Option` mirrors the init-time gate: `None` means the
// feature is off and its passes / resources should be torn down; `Some` means it
// is on and its resources should exist. A backend without a live-rebuild path
// ignores this (the choice still persists and applies at the next launch).
#[allow(dead_code)] // fields read only by Metal's apply_quality_settings.
pub struct QualitySettings {
    // Temporal anti-aliasing on/off. The backend additionally suppresses TAA
    // while temporal upscaling is active (the scaler does its own accumulation).
    pub taa: bool,
    pub ssao: Option<SsaoSettings>,
    pub ssr: Option<SsrSettings>,
    // Hardware ray-traced reflections. The backend further gates this on GPU
    // ray-tracing support, falling back to leaving it off when unsupported.
    pub rt_reflections: Option<RtReflectionSettings>,
    pub ssgi: Option<SsgiSettings>,
    // Per-axis divisor for the roughness-aware reflection blur target (the
    // reduced-resolution first pass of the SSR / RT reflection composite),
    // resolved from `PostProcessConfig.reflection_blur_resolution`. Every backend
    // sizes its blur target at render / this on a live reflection rebuild.
    pub reflection_blur_scale: u32,
    pub auto_exposure: Option<AutoExposureSettings>,
    // The authored exposure bias (stops) auto-exposure applies on top of its
    // adapted value; carried so a live auto-exposure enable matches init.
    pub auto_exposure_bias_ev: f32,
}

// GPU/device capability flags, queried from the backend once it is built.
// Surfaced so the settings menu can gray out (and make inert) toggles the
// device cannot honor -- e.g. ray-traced reflections on a GPU without hardware
// ray tracing. Mirrors an RHI-style capability set: a handful of bools held in
// memory and re-queried each launch, never persisted, so it is always correct
// for the current device + driver.
#[derive(Clone, Copy, Debug)]
pub struct DeviceCapabilities {
    // Hardware ray tracing for the RT-reflections pass: DXR 1.1 on DirectX, the
    // ray-query device extensions on Vulkan (and not under XeSS), and
    // `MTLDevice::supportsRaytracing` on Metal.
    pub ray_tracing: bool,
}

impl DeviceCapabilities {
    // Every capability present. The trait default, so a backend that does not
    // report capabilities never wrongly disables a toggle (it keeps the prior
    // behavior: the feature no-ops with a warning on an incapable device).
    pub const ALL: Self = Self { ray_tracing: true };
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self::ALL
    }
}

// The set of operations GraphicsSystem performs on a graphics backend.
// Implementations are thin forwarders to the inherent methods on
// MtlContext / DxContext / VkContext.
//
// The asset / world.jsonl hot-reload mutators below (`update_color_lut`,
// `rebuild_*_geometry`, `clone_static_draw_object`, etc.) are provided no-op
// methods driven only by the `cn debug` binary's reload passes, so they have
// no call site under `cargo check --lib`. Allow dead code at the trait level
// rather than annotating each; the required interface methods are never
// subject to the lint, so this only covers the binary-driven provided methods.
#[allow(dead_code)]
pub trait RenderBackend: SceneControl + Send {
    // Window / input lifecycle.
    fn window_closed(&mut self) -> bool;
    fn capture_cursor(&mut self);
    fn take_input(&mut self) -> RenderInput;
    fn wait_idle(&self);

    // Per-frame drive.
    fn draw_frame(
        &mut self,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
    ) -> Result<(), String>;
    fn update_view(&mut self, matrix: [[f32; 4]; 4]);
    fn update_model(&mut self, index: usize, model: [[f32; 4]; 4]);

    // Skinning. `vert_bytes` and `shadow_bytes` are Metal-only payloads;
    // DX/VK ignore them.
    fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u16],
        draw_objects: Vec<SkinnedDrawObject>,
        vert_bytes: &[u8],
        frag_bytes: &[u8],
        shadow_bytes: &[u8],
    ) -> Result<(), String>;
    fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]);

    // Texture streaming.
    fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String>;
    fn update_texture_slot(&mut self, slot: usize, w: u32, h: u32, px: &[u8])
    -> Result<(), String>;
    fn evict_normal_map_slot(&mut self, slot: usize) -> Result<(), String>;
    fn update_normal_map_slot(
        &mut self,
        slot: usize,
        w: u32,
        h: u32,
        px: &[u8],
    ) -> Result<(), String>;

    // Mesh streaming.
    fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
    fn upload_mesh(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        frame: u64,
    ) -> Result<(), String>;

    // Seed the streamed-mesh sub-allocators with one reserved headroom block
    // (byte ranges in the shared vertex / index buffers) instead of the
    // per-mesh build-time regions. Used by the shrinkable-seed path: the
    // streamed geometry is no longer baked into the buffers at build time, so
    // the renderer hands the allocators one contiguous block sized to the
    // cap-many resident meshes rather than the whole streamed set. Implemented
    // on Metal + DirectX + Vulkan. Default no-op: a backend without the
    // shrinkable seed keeps freeing each mesh's build-time region in
    // `setup_mesh_streaming`.
    fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        let _ = (vtx_offset, vtx_bytes, idx_offset, idx_bytes);
    }

    // Voxel-world chunk streaming. `texture_slot` and `normal_map_slot`
    // are ignored by Metal (it binds chunk textures per draw).
    fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
        texture_slot: usize,
        normal_map_slot: usize,
    ) -> Result<(), String>;
    #[allow(clippy::too_many_arguments)]
    fn add_chunk_mesh(
        &mut self,
        verts: &[Vertex],
        idxs: &[u16],
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: MaterialUniforms,
        frame: u64,
    ) -> Result<usize, String>;
    fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String>;
    fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String>;

    // Device capability flags, queried from the GPU once the backend is built.
    // Read by GraphicsSystem to gray out + disable settings rows the device
    // cannot honor. Default: all capable, so a backend that does not report
    // capabilities keeps every toggle live (the feature then no-ops with a
    // warning on an incapable device, as before).
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::ALL
    }

    // Metal-only diagnostics; default no-op for parity.
    fn logical_size(&self) -> (f32, f32) {
        (0.0, 0.0)
    }
    fn render_stats(&self) -> RenderStats {
        RenderStats::default()
    }

    // Show or hide the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // independent of camera capture. Edge-triggered by the backend, so calling
    // it every frame with the same value is cheap. Default no-op: a backend
    // without a free-mode cursor hide leaves the system cursor visible (DX /
    // Vulkan today).
    fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        let _ = hidden;
    }

    // Tell the backend a togglable menu (a View toggled by an Escape KeyBinding)
    // coexists with a captured camera. In this mode Escape routes to the ECS
    // (so the menu shows/hides) instead of releasing the cursor inline, and a
    // click never recaptures the cursor (it fires a UI action). Set once at
    // setup. Default no-op: backends without dynamic capture (DX / Vulkan today)
    // keep the static behavior.
    fn set_menu_mode(&mut self, on: bool) {
        let _ = on;
    }

    // Drive cursor capture from the menu state each frame: capture for camera
    // control, release while a menu is open. Edge-triggered by the backend.
    // Default no-op (DX / Vulkan): they keep their startup capture decision.
    fn set_camera_capture(&mut self, capture: bool) {
        let _ = capture;
    }

    // Supply the reflection-probe placements (from declared `ReflectionProbe`
    // assets, or empty to auto-seed from the scene bounds). The backend bakes a
    // cube per placement and samples the nearest for the specular reflection.
    // Pushed once after construction. Default no-op: backends without probe
    // support (DX / Vulkan today) keep the sky reflection.
    fn set_reflection_probes(&mut self, probes: &[crate::gfx::reflection_probe::ProbePlacement]) {
        let _ = probes;
    }

    // Turn display sync (vsync) on or off at runtime, applied to presentation.
    // Edge-triggered by the backend, so calling it with the unchanged value is
    // cheap. Default no-op: a backend that only honors vsync at init ignores
    // runtime changes.
    fn set_vsync(&mut self, on: bool) {
        let _ = on;
    }

    // Switch the window between windowed / borderless / fullscreen at runtime.
    // The change flows through the backend's normal resize path (no GPU rebuild
    // beyond the resize it triggers). Default no-op for backends without a
    // window (embedded / preview) or that don't yet implement it.
    fn set_window_mode(&mut self, mode: crate::assets::WindowMode) {
        let _ = mode;
    }

    // Resize the window's content area at runtime (meaningful in windowed mode).
    // Drives the same resize path as a user-dragged resize. Default no-op for
    // backends without a window or that don't yet implement it.
    fn set_window_size(&mut self, width: u32, height: u32) {
        let _ = (width, height);
    }

    // Replace the live post-process parameters (bloom / exposure / vignette /
    // LUT blend). These are pushed to the bloom + composite shaders each frame,
    // so a change takes effect on the next draw with no allocation or pipeline
    // rebuild. Default no-op: a backend that only reads the params at init
    // ignores runtime changes (DirectX / Vulkan today).
    fn update_post_process(&mut self, params: PostProcessParams) {
        let _ = params;
    }

    // Set the live ambient (IBL) light scale. Unlike the post-process params
    // above, `ambient_intensity` lives in the shared `LightUniforms` (uploaded
    // each frame by the main lighting pass), so it takes its own setter rather
    // than `update_post_process`. Default no-op: only Metal mutates it live
    // today; DirectX / Vulkan keep the init-time value (they read it at init).
    fn set_ambient_intensity(&mut self, value: f32) {
        let _ = value;
    }

    // Push the gameplay movement key map. The backend resolves each canonical
    // `Key` to its native key code and decodes physical key events through the
    // map (instead of hardcoded keys), so a settings-menu rebind takes effect on
    // the next key event. Pushed once after the backend is built and again on
    // each rebind. Default no-op: a backend without keymap decode keeps its
    // built-in defaults.
    fn set_keymap(&mut self, keymap: &KeyMap) {
        let _ = keymap;
    }

    // Apply a change to the quality-feature toggles (TAA / SSAO / SSR / RT
    // reflections / SSGI / auto-exposure) live. Unlike the post-process params,
    // these gate render passes whose GPU resources (pipelines, render targets,
    // ray-tracing acceleration structures) are built once at init, so applying a
    // change rebuilds the affected resources in place rather than flipping a
    // uniform. Default no-op: a backend that only reads these at init ignores
    // runtime changes (DirectX / Vulkan today), so the choice persists and takes
    // effect at the next launch there.
    fn apply_quality_settings(&mut self, settings: QualitySettings) {
        let _ = settings;
    }

    // Shared atomic flag the backend polls at frame start to trigger a
    // shader rebuild. `Some` only under `cn debug` on backends that ship
    // hot-reload (Metal today); `None` on production runs and on backends
    // that have not implemented hot-reload yet. The debug server reads this
    // to forward `reload-shaders` commands; the filesystem watcher writes
    // it directly. Default: `None`.
    fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        None
    }

    // Replace the live colour-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only). Default no-op: backends
    // that have not implemented the swap leave the LUT bound at whatever
    // payload was uploaded at init.
    fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        let _ = (size, data);
        Ok(())
    }

    // `(vertex_count, index_count)` for the static draw at `draw_idx`, or
    // `None` when the index is out of range / the backend does not expose
    // the field. Used by asset hot-reload to detect size-changing
    // reloads before attempting [`Self::update_mesh_geometry`], which
    // rejects size mismatches. Default returns `None`; backends that
    // implement the rebuild path also override this.
    fn draw_geometry_size(&self, draw_idx: usize) -> Option<(usize, usize)> {
        let _ = draw_idx;
        None
    }

    // Per-LOD-alternate index counts for the static draw at `draw_idx`,
    // ordered from LOD1 upward (LOD0 is reported by
    // [`Self::draw_geometry_size`]). Returns `None` when the index is out of
    // range or the backend does not expose its LOD layout. Used by asset
    // hot-reload alongside [`Self::draw_geometry_size`] to detect
    // size-changing reloads: a `.glb` that re-exports with a different LOD
    // breakdown queues the entry for [`Self::rebuild_static_geometry`]
    // instead of [`Self::update_mesh_geometry`]'s in-place write.
    fn draw_lod_index_counts(&self, draw_idx: usize) -> Option<Vec<usize>> {
        let _ = draw_idx;
        None
    }

    // Rebuild the shared static-mesh vertex + index buffers, replacing the
    // geometry of each `DrawGeometryUpdate.draw_idx` with the new
    // vertices / indices / LOD alternates. Draws not named in `changes`
    // keep their current geometry, copied byte-for-byte from the live
    // buffers. The slot's `vertex_count`, `index_count`, and
    // `lod_alternates` index offsets are rewritten as the new buffers are
    // laid out. Driven by asset hot-reload (`cn debug` only) when a
    // size-changing `.glb` re-export means the existing
    // [`Self::update_mesh_geometry`] in-place write no longer fits.
    // `wait_idle` first; the rebuild swaps the GPU buffers wholesale.
    // Default no-op: backends that have not implemented the rebuild
    // return `Ok(())` and the size-changing reload is logged + skipped at
    // the caller (the existing in-place path already errored on size
    // mismatch).
    fn rebuild_static_geometry(&mut self, changes: Vec<DrawGeometryUpdate>) -> Result<(), String> {
        let _ = changes;
        Ok(())
    }

    // Replace a `SkinnedMesh` draw slot's vertex + index data in place.
    // Driven by asset hot-reload (`cn debug` only). Reuses the slot's
    // existing vertex region + index region in the shared skinned vertex /
    // index buffers (created once by [`Self::upload_skinned`]), so the new
    // geometry must match the slot's init-time vertex count + index count
    // and the new skeleton must keep the same joint count; pipelines stay
    // untouched, only the bytes change. `vertex_base` is the init-time
    // vertex offset (in vertex units) into the shared buffer; indices are
    // rebased onto it before writing. Default no-op.
    fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u16,
        verts: &[SkinnedVertex],
        idxs: &[u16],
    ) -> Result<(), String> {
        let _ = (skinned_index, vertex_base, verts, idxs);
        Ok(())
    }

    // Rebuild the shared skinned-mesh vertex + index buffers, replacing the
    // geometry of each `SkinnedDrawGeometryUpdate.skinned_index` with the
    // new vertices / indices. Slots not named in `changes` keep their
    // current geometry, copied byte-for-byte from the live buffers and
    // re-based onto the new vertex region they land in. Returns the
    // post-rebuild layout (one [`SkinnedSlotLayout`] per slot, in
    // `skinned_index` order) so the caller can refresh its source-map
    // `vertex_base` / `vertex_count` / `index_count` to point at the new
    // regions. Driven by asset hot-reload (`cn debug` only) when a
    // size-changing `.glb` re-export means the existing
    // [`Self::update_skinned_mesh_geometry`] in-place write no longer fits.
    // The backend `wait_idle`s first; the rebuild swaps the GPU buffers
    // wholesale. The skinned pipelines, shadow + velocity + SSAO + SSR
    // variants, and `skinned_draw_objects` slot metadata
    // (`texture_slot` / `normal_map_slot` / `material` / `joint_count`)
    // all stay untouched; only the `index_offset` / `index_count` on each
    // `SkinnedDrawObject` (and the buffers themselves) move. Default no-op
    // (returns an empty layout vec): backends that have not implemented
    // the rebuild leave the size-changing reload as logged + skipped at
    // the caller, the same behaviour as before, since the in-place path
    // already errored on size mismatch.
    fn rebuild_skinned_geometry(
        &mut self,
        changes: Vec<SkinnedDrawGeometryUpdate>,
    ) -> Result<Vec<SkinnedSlotLayout>, String> {
        let _ = changes;
        Ok(Vec::new())
    }

    // Update a skinned slot's joint count and resize the backend's per-slot
    // joint-matrix buffers to match. Driven by asset hot-reload (`cn debug`
    // only) when a re-imported `.glb`'s skeleton has a different joint
    // count than the slot was initialised with. Shrinking truncates the
    // per-slot Vec; growing seeds the new entries to identity so the slot
    // renders undeformed on the next `update_skinned_pose`. The skinned
    // shaders consume the joints buffer through a pointer (not a fixed-
    // size array) and use vertex-attribute-encoded joint indices, so no
    // pipeline or shader rebuild is required for a joint-count change;
    // only the CPU-side per-slot buffer and `SkinnedDrawObject.joint_count`
    // change. Default no-op: backends that have not implemented the resize
    // leave the skeleton-shape change logged + skipped at the caller.
    fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let _ = (skinned_index, new_joint_count);
        Ok(())
    }

    // Replace a `Mesh` draw slot's vertex + index data in place. Driven by
    // asset hot-reload (`cn debug` only). Reuses the slot's existing offset
    // in the shared vertex / index buffers, so the new geometry must match
    // the slot's init-time vertex count + index count; a size-changing
    // reload returns an error so the caller can queue
    // [`Self::rebuild_static_geometry`] instead, which repacks the shared
    // buffers. Each entry in
    // `lod_alternates` (`(switch_distance, mesh-relative indices)`) is
    // written to the matching slot's pre-allocated LOD index region; the
    // number of LODs and each LOD's index count must match the slot's
    // init-time layout, otherwise the call returns an error so the caller
    // can queue [`Self::rebuild_static_geometry`]. `switch_distance` is
    // re-stored per LOD so a JSON-side tweak to `lod_distances` propagates
    // without a process restart. Default no-op.
    fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        verts: &[Vertex],
        idxs: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let _ = (draw_idx, verts, idxs, lod_alternates);
        Ok(())
    }

    // Replace the live IBL environment map with a freshly precomputed payload.
    // `payload` is the serialised byte format emitted by
    // [`crate::build::environment_map::compile_environment_map_payload`]
    // (header + irradiance cube + prefilter mip chain), so init and hot-reload
    // share a single byte format. Driven by asset hot-reload (`cn debug`
    // only). Default no-op: backends that have not implemented the swap leave
    // the IBL cubes bound at whatever payload was uploaded at init.
    fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let _ = payload;
        Ok(())
    }

    // Replace the live volumetric-fog settings, or disable the fog pass when
    // `None`. Driven by world.jsonl hot-reload (`cn debug` only). Default
    // no-op: backends that have not implemented the swap leave the fog pass
    // at whatever settings were resolved at init.
    //
    // A backend that built its fog pipeline lazily based on the world's
    // init-time `VolumetricFog` cannot enable the pass via this call when
    // the world started with no fog declared; re-enabling fog on a world
    // that did not declare it at startup requires a relaunch.
    fn update_fog_settings(&mut self, settings: Option<FogSettings>) {
        let _ = settings;
    }

    // Capture the last presented frame to a PNG at `path` and return the saved
    // path. Driven by the `cn debug` WS `screenshot` command for headless
    // on-GPU render verification. Default `Err`: a backend without a capture
    // path reports it unsupported (all current backends override this).
    fn screenshot(&mut self, path: &str) -> Result<String, String> {
        let _ = path;
        Err("screenshot capture not supported on this backend".to_string())
    }

    // Append a new draw object that re-uses an existing draw slot's geometry
    // region (`vertex_offset` / `vertex_count` / `index_offset` / `index_count`
    // / `base_vertex` / `lod_alternates`) with a fresh model matrix, texture
    // slots, material, and cull distance. Returned index is the new
    // `DrawObject` slot. Driven by `world.jsonl` hot-reload (`cn debug` only)
    // when an authored Prop is added to a mesh / model that already has at
    // least one draw in the world. The clone is non-cullable (sentinel AABB)
    // since the init-time BVH cannot refit; the dynamically added prop is
    // drawn every frame regardless of camera position, like a streamed chunk.
    // Default no-op (returns `Err`): backends without an implementation leave
    // the add path logged + skipped at the caller.
    fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: MaterialUniforms,
        cull_distance: f32,
    ) -> Result<usize, String> {
        let _ = (
            src_draw_idx,
            model,
            texture_slot,
            normal_map_slot,
            material,
            cull_distance,
        );
        Err("clone_static_draw_object: not implemented on this backend".to_string())
    }

    // Rewrite a draw slot's material parameters + texture/normal-map pool
    // indices in place. Driven by `world.jsonl` hot-reload (`cn debug` only)
    // when a Prop edits its `material` / `texture` arg. Default no-op: the
    // caller logs the change as skipped on backends without an implementation.
    fn set_draw_material(
        &mut self,
        draw_idx: usize,
        material: MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        let _ = (draw_idx, material, texture_slot, normal_map_slot);
    }

    // Rewrite a draw slot's `cull_distance` in place. Driven by `world.jsonl`
    // hot-reload (`cn debug` only) when a Prop edits its `cull_distance` arg.
    // Default no-op.
    fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32) {
        let _ = (draw_idx, cull_distance);
    }

    // Append a projected-decal record at runtime, returning a stable slot
    // index the caller hands to [`Self::remove_decal`] later. Lets a
    // gameplay system stamp bullet holes, footprints, or other ad-hoc
    // decals after the world has built. Backends that have not implemented
    // the runtime path return `Err`; the caller logs and drops the request.
    fn add_decal(&mut self, record: crate::gfx::decal::DecalRecord) -> Result<usize, String> {
        let _ = record;
        Err("add_decal: not implemented on this backend".to_string())
    }

    // Tombstone a runtime decal slot. The id returned by
    // [`Self::add_decal`] becomes invalid; the next add may reuse it.
    // Default no-op-with-Err: backends without a runtime path leave the
    // remove logged + skipped at the caller.
    fn remove_decal(&mut self, decal_id: usize) -> Result<(), String> {
        let _ = decal_id;
        Err("remove_decal: not implemented on this backend".to_string())
    }

    // Append a particle-emitter record at runtime, returning a stable slot
    // index. The backend allocates the per-emitter GPU pool + atomic
    // spawn counter (matching the init-time path) so the compute kernel
    // can begin ticking on the next frame. Default no-op-with-Err.
    fn add_emitter(
        &mut self,
        record: crate::gfx::particles::ParticleEmitterRecord,
    ) -> Result<usize, String> {
        let _ = record;
        Err("add_emitter: not implemented on this backend".to_string())
    }

    // Tombstone a runtime emitter slot and release its GPU pool +
    // counter buffers (the GPU keeps them alive via its own refcount
    // until any in-flight command buffer that referenced them completes).
    // Default no-op-with-Err.
    fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String> {
        let _ = emitter_id;
        Err("remove_emitter: not implemented on this backend".to_string())
    }

    // Rebuild the live main / instanced / shadow render pipelines from
    // freshly compiled world-loaded shader stage bytes. Driven by asset
    // hot-reload (`cn debug` only) when one of the captured `ShaderStage`
    // source files is saved or a debug-WS `reload-assets` command fires.
    // Each `Some(bytes)` replaces the matching live pipeline (and any
    // dependent state: bindless-texture argument encoder, cull pipeline,
    // instanced variant, shadow variant); `None` leaves the pipeline
    // untouched (e.g. a world without an instanced shader passes `None`
    // for the instanced slot). The backend should build every replacement
    // into a temporary first and only swap when every build succeeds;
    // mirrors the safety pattern in [`crate::metal::hot_reload`] so a
    // compile error never overwrites a live pipeline with a half-built
    // replacement. Default no-op (returns `Err`): backends without an
    // implementation leave the world-loaded shader reload logged + skipped
    // at the caller.
    //
    // Skinned-mesh variants are out of scope here: their pipelines depend
    // on the world's `SkinnedMesh`-injected library bytes that
    // [`Self::upload_skinned`] consumes and drops.
    fn update_world_shader_pipelines(
        &mut self,
        vert_bytes: Option<&[u8]>,
        frag_bytes: Option<&[u8]>,
        shadow_bytes: Option<&[u8]>,
        vert_instanced_bytes: Option<&[u8]>,
    ) -> Result<(), String> {
        let _ = (vert_bytes, frag_bytes, shadow_bytes, vert_instanced_bytes);
        Err("update_world_shader_pipelines: not implemented on this backend".to_string())
    }
}
