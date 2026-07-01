// src/gfx.rs
//
// Backend-agnostic render-prep helpers. These modules produce GPU-ready data
// from asset components but do not own or borrow a backend handle.
//
// The pure GPU data layouts and CPU-side math (mesh payloads, LOD, skinning,
// camera, frustum, chunk-streaming helpers, post-process settings) now live in
// concinnity-core and are re-exported below so the historical
// crate::gfx::<module> paths keep resolving. The render graph, draw lists,
// scene reel, and per-backend executors stay here.
// `pub` so the editor crate can reach these core GPU-layout modules through
// `concinnity_client::gfx::*` (e.g. shader-layout reflection).
pub use concinnity_core::gfx::{
    auto_exposure, camera, frustum, lod, mesh_payload, mesh_seed, profile, render_types,
    rt_reflections, skinning, ssao, ssgi, ssr,
};
// Chunk-streaming layout helpers: driven only by the Metal backend today, so
// the re-exports are unused on other backends (mirrors the chunk_window /
// streaming gating below).
#[cfg_attr(not(target_os = "macos"), allow(unused_imports))]
pub(crate) use concinnity_core::gfx::{chunk_coord, range_alloc};

// Skeletal animation playback. Internal system, constructed by `World::start`
// when the world declares any `Animation`; produces per-frame skinning matrices.
// `pub` so the editor crate can drive the clip hot-reload through the
// `AnimationSystem` setter API.
pub mod animation;
pub mod backend;
pub(crate) mod bvh;
// Generic Send/Sync shim shared by the three backends' parallel-encode fan-outs.
pub(crate) mod parallel_ctx;
// Backend-agnostic fullscreen-pass encoder seam (HAL pilot). Adopted by DirectX
// + Vulkan; unused (dead code) on a Metal build, which keeps its own
// `fullscreen_pass` helper and ends its composite encoder with a `ScopedEncoder`
// RAII guard (the seam's split begin/draw/text/end would leave a Metal render
// encoder open at commit if a text draw errored).
#[cfg_attr(backend_metal, allow(dead_code))]
pub(crate) mod fullscreen;
// First-person / fly-through camera controller. Internal system, constructed by
// `World::start` from a `Camera3D`'s controller settings.
pub(crate) mod camera_controller;
pub(crate) mod csm;
pub(crate) mod cursor;
pub mod decal;
pub mod draw_list;
pub(crate) mod draw_slot;
// Free pool for pre-reserved skinned instance slots, consumed by the runtime
// skinned-spawn path.
pub(crate) mod skinned_pool;
// The renderer driver. An internal system (not a declarable asset), constructed
// by `World::start` when the world declares a `GraphicsConfig`.
pub mod graphics_system;
pub(crate) mod hdr_output;
pub mod input;
// The runtime, rebindable key map (canonical action -> physical key) for the
// gameplay movement keys. Persisted in `ControlsSettings` and pushed to the
// active backend, which resolves it to native key codes.
pub(crate) mod keymap;
pub(crate) mod lights;
// Backend-agnostic mip-chain generation for streamed RGBA8 textures; each
// backend's texture upload builds and uploads the full chain.
pub(crate) mod mipmap;
pub mod particles;
// Backend-agnostic planar-reflection math (mirror matrices, oblique near-plane
// clip, plane-set assignment). Consumed by the Metal backend today;
// compiled-but-unreferenced on the other backends until their planar ports land.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod planar_reflection;
// Backend-agnostic reflection-probe bake queue, async state machine, and
// auto-seed. Consumed by the Metal backend today; compiled-but-unreferenced on
// the other backends until their probe ports land.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod reflection_probe;
// Backend-agnostic render graph: types + builder + compile pass with
// unit tests. Per-backend executors live alongside each backend.
pub(crate) mod quality_preset;
// Cross-backend render-graph types + builder + compile/alias passes (the
// graphs-and-dags effort). Each backend's barrier + resource-aliasing path
// consumes only a subset today, so a portion stays unused on any given build
// while the graph is still being wired in. Allow dead code module-wide so the
// build stays clean on all three backends; drop this once every backend's
// barrier + aliasing path consumes the full set.
#[allow(dead_code)]
pub(crate) mod render_graph;
// Backend-agnostic planner for the incremental RT acceleration-structure
// topology refresh (reuse-unchanged / build-new / retire-orphan). Consumed by
// the DirectX + Vulkan backends; the Metal backend keeps its own equivalent
// copy, so this module is unreferenced on a macOS build.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) mod rt_topology;
pub mod scene_reel;
pub(crate) mod settings;
// Cross-backend cascade re-render scheduling for the cascaded shadow map
// (hybrid / every_frame). Used by all three backends' shadow passes.
pub(crate) mod shadow_schedule;
pub(crate) mod sprite;
pub(crate) mod text;
// CPU-side ordering policy for the transparent pass. Consumed by the Metal
// backend today; compiled-but-unreferenced on the other backends until their
// transparent ports land.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod transparent;
pub mod volumetric_fog;

// First-person camera controller. Currently unreferenced scaffolding
// (Camera3DSystem drives the camera); kept for a future stateful controller.
#[cfg(not(target_os = "macos"))]
pub(crate) mod fps_controller;

// Streaming / chunk-world support. Currently driven only by the Metal
// backend, so on non-macOS builds these modules are compiled but
// unreferenced.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod chunk_window;
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) mod streaming;
