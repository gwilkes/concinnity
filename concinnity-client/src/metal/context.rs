#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::NSWindow;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLCommandBuffer as _, MTLCommandQueue, MTLDepthStencilState,
    MTLDevice as _, MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLPixelFormat, MTLRenderPipelineState, MTLResourceOptions,
    MTLSamplerState, MTLTexture,
};
use objc2_metal_kit::MTKView;

use crate::gfx::render_types::{
    DrawObject, InstancedCluster, LightUniforms, NUM_SHADOW_CASCADES, ShadowUniforms,
};

use super::auto_exposure::AutoExposureGpu;
use super::cull::CullState;
use super::decal::DecalState;
use super::fog::FogState;
use super::input::KeyState;
use super::particle::ParticleState;
use super::post::{
    BloomPipelines, BloomTargets, GBufferState, SsaoState, SsgiState, SsrState, TaaState,
    UpscaleState,
};
use super::raytrace::RtState;
use super::resources::skinning::SkinnedState;
use super::texture::{EnvironmentMapTextures, HdrTargets};
use super::transient_pool::TransientTexturePool;

// MSAA sample count for the off-screen HDR target. Matches the sample
// count used pre-post-process (4×). Kept explicit here so all the
// pipelines that target the HDR buffer can reference the same constant.
pub(super) const HDR_SAMPLE_COUNT: u32 = 4;

// Size of the bindless texture pool the static main pass samples. The pool
// holds every albedo texture followed by every normal map; `GpuObjectData`
// carries pool indices into it. Must match `BINDLESS_TEXTURE_COUNT` in
// `default.metal`. Worlds with more than this many textures fall back to
// clamped indices (logged once at init).
pub(super) const BINDLESS_TEXTURE_COUNT: usize = 1024;

// Fragment buffer index the bindless static pass binds its `BindlessTextures`
// argument buffer at. Discrete `[[texture(n)]]` bindings make a fragment
// shader unusable from an indirect command buffer on Apple GPUs, so the
// texture pool + shadow/IBL maps travel in an argument buffer instead. Must
// match the `[[buffer(7)]]` on `fragment_main_bindless` in `default.metal`.
pub(super) const BINDLESS_TEXTURE_ARG_BUFFER_INDEX: usize = 7;

// Stores the NSView* pointer set by cn_preview_start before world.start() is called.
// MtlContext::new() atomically takes it: non-null → embedded mode, null → windowed mode.
static EMBEDDED_VIEW_PTR: std::sync::atomic::AtomicPtr<std::ffi::c_void> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

// Whether the next MtlContext should pump NSEvents in draw_frame even when in
// embedded mode. Preview leaves this false (SwiftUI owns input dispatch); the
// blocking-in-view play path sets it true so the world receives keyboard/mouse.
static EMBEDDED_PUMP_EVENTS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// Called by cn_preview_start to register the NSView that MtlContext should embed into.
pub fn set_preview_view(ptr: *mut std::ffi::c_void) {
    EMBEDDED_VIEW_PTR.store(ptr, std::sync::atomic::Ordering::SeqCst);
}

// Called by cn_run_world_blocking_in_view to opt the next embedded MtlContext
// into pumping NSEvents (so the world receives input). The flag is consumed
// in MtlContext::new and reset to false, so subsequent previews stay quiet.
pub fn set_embedded_pump_events(v: bool) {
    EMBEDDED_PUMP_EVENTS.store(v, std::sync::atomic::Ordering::SeqCst);
}

pub(super) fn take_embedded_view() -> *mut std::ffi::c_void {
    EMBEDDED_VIEW_PTR.swap(std::ptr::null_mut(), std::sync::atomic::Ordering::SeqCst)
}

pub(super) fn take_embedded_pump_events() -> bool {
    EMBEDDED_PUMP_EVENTS.swap(false, std::sync::atomic::Ordering::SeqCst)
}

// Metal rendering context. Owns all GPU resources and the window.
// Only ever accessed from the main thread.
pub struct MtlContext {
    pub(super) device: Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
    pub(super) command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    // Pixel format the MTKView's CAMetalLayer is currently presenting at:
    // `BGRA8Unorm` for SDR, `RGBA16Float` for HDR EDR. The post + text
    // pipelines bake this format into their colour attachment descriptors,
    // so it has to be stable between `MtlContext::new` and any subsequent
    // hot-reload rebuild. `swap_pixel_format == RGBA16Float` is the runtime
    // equivalent of `HdrOutputMode::is_hdr()`, and the per-frame
    // `PostProcessParams.hdr_output` flag carries the EDR signal into the
    // shader.
    pub(super) swap_pixel_format: MTLPixelFormat,
    // Maximum extended-range colour-component multiplier reported by the
    // active panel when the renderer is on the HDR path. `Some(2.0)` on
    // HDR400, `Some(8.0+)` on HDR1000-class panels; `None` on SDR (whether
    // the world disabled HDR or the platform fell back). Surfaced via
    // `RenderStats.max_edr` so the `StatHud` overlay can render an `EDR`
    // chip showing the available headroom.
    pub(super) max_edr: Option<f32>,
    // Resolved HDR encoding of the swapchain (scRGB-linear vs PQ), or `None`
    // on the SDR path. Read only by the headless `screenshot` path to decode
    // the captured `RGBA16Float` EDR drawable. Mirrors DX `hdr_encoding`.
    pub(super) hdr_encoding: Option<crate::gfx::hdr_output::HdrEncoding>,
    // Colour texture of the most recently presented drawable, retained so the
    // `cn debug` `screenshot` command can blit it back to a host buffer and
    // PNG-encode it (see metal/screenshot.rs). Set each frame only under
    // `hot_reload` (the path that runs the WS server able to request a
    // capture); `None` in production and before the first present, so a
    // capture then returns a clean error. The MTKView has `framebufferOnly`
    // switched off under the same gate so the drawable is blit-readable.
    // Mirrors the DX/VK `last_present_index`.
    pub(super) last_present_texture: Option<Retained<ProtocolObject<dyn MTLTexture>>>,
    pub(super) pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // True when the static main-pass pipeline runs the bindless fragment
    // shader (`fragment_main_bindless`). The static draw loop then reads each
    // object from the per-frame `GpuObjectData` buffer and a bindless texture
    // pool instead of rebinding model/material/textures per draw. False for
    // shaders without that entry point (custom shaders), which
    // use the legacy per-draw binding path.
    pub(super) bindless: bool,
    // GPU-driven cull feature state: the phase-1/phase-2 cull pipelines,
    // their indirect command buffers + argument encoders/buffers, the
    // per-object status buffer, the two-pass-occlusion toggle, and the Hi-Z
    // pyramid + view-projection snapshots the occlusion test reprojects
    // through. All `Some`/active only on the bindless path (non-bindless
    // shaders keep the legacy per-draw CPU loop). See [`CullState`].
    pub(super) cull: CullState,
    // Encoder that packs the bindless pass's textures into a per-frame
    // argument buffer (`BindlessTextures` in `default.metal`). `Some` only
    // when `bindless`; the argument buffer itself is rebuilt every frame so
    // streamed texture swaps are picked up and the GPU never reads a buffer
    // the CPU is mid-rewrite. (A main-pass resource, not part of `cull`.)
    pub(super) bindless_tex_arg_encoder: Option<Retained<ProtocolObject<dyn MTLArgumentEncoder>>>,
    pub(super) depth_state: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    // Read-only depth state: `LessEqual` test, no write. Used by translucent
    // draws that must be occluded by nearer opaque geometry but must not
    // update the depth buffer (volumetric raymarch volumes). Metal forbids
    // `setDepthStencilState(nil)` under the validation layer, so translucent
    // passes bind this instead of clearing the state.
    pub(super) depth_state_read_only: Retained<ProtocolObject<dyn MTLDepthStencilState>>,
    // Single shared vertex buffer containing geometry for all draw objects.
    pub(super) vertex_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Single shared index buffer containing indices for all draw objects.
    pub(super) index_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    // One entry per renderable object. Replaces the former single index_count.
    pub(super) draw_objects: Vec<DrawObject>,
    // Spatial index over the cullable subset of `draw_objects`, built once
    // at init. The main pass queries it per frame to skip off-screen draws.
    pub(super) cull_bvh: crate::gfx::bvh::Bvh,
    // Indices into `draw_objects` for non-cullable items (skybox, rooms,
    // dynamic props). Drawn unconditionally after the BVH-visible set.
    pub(super) always_draw: Vec<u32>,
    // Per-frame scratch for the legacy CPU draw path's visible set
    // (BVH-culled cullables + always_draw fallback). `mem::take`d at the
    // top of draw_frame and returned at the bottom so the heap allocation
    // is reused across frames instead of `Vec::with_capacity`'d each tick.
    pub(super) visible_scratch: Vec<u32>,
    // One entry per InstancedProp cluster. Each cluster issues one
    // drawIndexedInstanced call with all of its per-instance transforms
    // uploaded to a transient GPU buffer per frame. Empty when there are
    // no clusters in the scene.
    pub(super) instanced_clusters: Vec<InstancedCluster>,
    // Total instances across every cluster. When the bindless static pass is
    // active (`bindless && !draw_objects.is_empty()`), each instance is folded
    // into the GPU-driven cull buffers as an extra `GpuObjectData` record after
    // the static objects, so the cull dispatch + indirect draw cover
    // `cull_count() == draw_objects.len() + n_instances`. 0 leaves the static
    // path identical and routes instances through the legacy instanced draw.
    pub(super) n_instances: usize,
    // The per-instance `GpuObjectData` / `GpuDrawArgs` records, built once at
    // init (instances are placed at world load and never move). `build_object_buffer`
    // / `build_draw_args_buffer` append these after their per-frame static fill
    // each frame, so the transient object / draw-args rings carry both. Per-instance
    // LOD is deferred (every instance draws the cluster base LOD), so these stay
    // static; supporting it would move the build per-frame.
    pub(super) instance_records: Vec<crate::gfx::render_types::GpuObjectData>,
    pub(super) instance_draw_args: Vec<crate::gfx::render_types::GpuDrawArgs>,
    // Number of skinned draw objects folded into the GPU-driven cull.
    // Set by `upload_skinned` ONLY when bindless + static geometry is present;
    // 0 otherwise (a pure-skinned or non-bindless world keeps the legacy skinned
    // VS draw). When > 0, each skinned object is one extra `GpuObjectData` record
    // after the static + instance records, so `cull_count()` extends to
    // `draw_objects.len() + n_instances + n_skinned` and the skinned tail draws
    // the compute-deformed geometry through the skinned u16 index buffer.
    pub(super) n_skinned: usize,
    // PSO that pairs `vertex_main_instanced` with `fragment_main`. None when
    // no clusters were provided or no instanced shader was compiled.
    pub(super) instanced_pipeline_state:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(super) clear_color: [f32; 4],
    // True when the world has no 3D geometry (e.g. a text-only world). The
    // off-screen HDR / bloom / effect targets are then allocated at 1x1 since
    // nothing is rendered into them; the composite pass still runs at the full
    // drawable size, so text stays crisp.
    pub(super) geometry_less: bool,
    pub(super) view_matrix: [[f32; 4]; 4],
    // Textures bound per draw call (slot == vec index).
    // A 1x1 opaque-white fallback is always present at slot 0 so shaders that
    // sample texture(0) produce correct output even when no Texture asset was
    // declared in the blob.
    pub(super) textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    // Normal-map textures bound per draw call at texture(1).
    // Slot 0 is always the 1x1 flat-normal fallback (RGBA 128,128,255,255).
    // DrawObject.normal_map_slot indexes into this vec.
    pub(super) normal_map_textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    // All scene lights packed and pushed to the fragment shader at buffer(4).
    pub(super) light_uniforms: LightUniforms,
    pub(super) sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    // Shadow map resources.
    // shadow_pipeline_state is None when no ShadowStage was declared or
    // shadow_map_size == 0, in which case the shadow pass is skipped.
    // shadow_map and shadow_sampler are always Some (1x1 fallback when disabled)
    // so fragment shaders can always safely sample texture(2) / sampler(1).
    pub(super) shadow_pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Depth32Float texture array, one slice per cascade. When the shadow pass
    // is disabled this is a 1x1 fallback that always reads 1.0 (fully lit).
    pub(super) shadow_map: Retained<ProtocolObject<dyn MTLTexture>>,
    // Per-cascade resolution stored so the shadow pass can size the viewport
    // to match the texture array's per-slice dimensions.
    pub(super) shadow_map_size: u32,
    // Cascade re-render policy from GraphicsConfig.shadow_update. Hybrid
    // refreshes the near cascade every frame and the far cascades round-robin.
    pub(super) shadow_update: crate::assets::ShadowUpdate,
    // Round-robin clock + primed-set for the cascade schedule; advanced once per
    // frame by `next_shadow_cascade_mask`.
    pub(super) shadow_scheduler: crate::gfx::shadow_schedule::ShadowCascadeScheduler,
    // Cascades re-rendered this frame (bit `i` = cascade `i`). Computed in
    // draw_frame and read by encode_shadow_pass so the two agree on which
    // slices to refresh and which to leave intact.
    pub(super) shadow_render_mask: u32,
    pub(super) shadow_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    // Cascaded light VPs + split depths. The near cascade's VP refreshes every
    // frame; far cascades' VPs persist between their round-robin refreshes
    // (Hybrid mode) so each slice is sampled with the VP it was rendered with.
    pub(super) shadow_uniforms: ShadowUniforms,
    // World-space unit vector pointing TOWARD the first directional light.
    // Captured at init from the light_uniforms; used by per-frame CSM updates.
    pub(super) shadow_light_dir: [f32; 3],
    // IBL cubemaps + mip count. Always Some: the runtime synthesizes a 1x1
    // grey fallback for both cubes when no EnvironmentMap was supplied, so
    // the fragment shader's texture(3) / texture(4) bindings are always
    // valid. `prefilter_mip_count == 0` is the "IBL disabled" signal the
    // shader uses to fall back to the legacy ambient/skybox path.
    pub(super) env_map: EnvironmentMapTextures,
    // Local reflection probes: the scene captured into one cube per placement
    // (metal/probe.rs). Distinct from `env_map` (which stays the sky -- it drives
    // the skybox + diffuse irradiance) so the bake never corrupts the visible
    // sky. Each surface's specular reflection samples the nearest probe whose box
    // contains it; the skybox + diffuse keep the sky.
    //
    // `probe_placements` is the where/box list (declared `ReflectionProbe` assets
    // or `auto_seed_probes`); `probe_maps` is the baked cube per placement
    // (parallel index); `probe_set` is the box uniform pushed to the shader.
    pub(super) probe_placements: Vec<crate::gfx::reflection_probe::ProbePlacement>,
    pub(super) probe_maps: Vec<EnvironmentMapTextures>,
    // Staggered bake cursor. Reset to the placement count when placements are set;
    // each eligible frame bakes a bounded budget and advances it, so the load cost
    // spreads over several frames instead of one. See metal/probe.rs.
    pub(super) probe_bake_queue: crate::gfx::reflection_probe::ProbeBakeQueue,
    // Per-probe influence boxes + count, pushed to the fragment shader at
    // buffer(6). `EMPTY` until a bake. See metal/probe.rs.
    pub(super) probe_set: super::uniforms::ProbeSet,
    // The probe currently rendering its six cube faces on the GPU (one at a time;
    // owns the reserved-ring-slot buffers + capture targets). The render thread never
    // blocks: the faces are submitted without `waitUntilCompleted` and a completion
    // handler flags GPU completion. `None` when nothing is rendering. See metal/probe.rs.
    pub(super) probe_rendering: Option<super::probe::RenderingBake>,
    // The probe whose read-back faces are convolving on a worker thread (one at a time).
    // Holds only the worker's payload slot (plain data), so it overlaps the next probe's
    // render -- pipelining the convolution shortens the bake warm-up. `None` when idle.
    pub(super) probe_converting: Option<super::probe::ConvertingBake>,
    // Deferred-free pool for an in-flight bake's GPU resources when a re-placement
    // (`set_reflection_probes`) interrupts it: the capture command buffers may
    // still be reading those buffers/textures, so they are parked here and freed
    // once the frames-in-flight fence guarantees the bake has retired.
    pub(super) probe_retire_pool: super::transient::RetirePool<super::probe::BakeGpu>,
    // Linear-clamp sampler bound at sampler(2) for cubemap sampling.
    pub(super) cube_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    // Text rendering resources. text_pipeline_state is None when no Font assets
    // were declared; text_atlas_textures is empty in the same case. The text
    // pipeline now targets the single-sample drawable in the composite pass
    // (after HDR tonemap), so text is rendered in display-referred LDR.
    pub(super) text_pipeline_state: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(super) text_atlas_textures: Vec<Retained<ProtocolObject<dyn MTLTexture>>>,
    // Linear-clamp sampler for glyph atlas lookups.
    pub(super) text_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    // Off-screen HDR render targets (MSAA RGBA16Float + resolve + MSAA
    // Depth32Float). Re-created lazily in `draw_frame` whenever the
    // drawable size changes. The main + instanced pipelines render into
    // these, and the post-process pass samples `hdr_resolve` to write
    // the tonemapped + FXAA-filtered output into the drawable.
    pub(super) hdr_targets: HdrTargets,
    // Pipeline that performs ACES tonemap + gamma 2.2 + FXAA from the
    // resolved HDR target into the drawable.
    pub(super) post_pipeline_state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    // Linear-clamp sampler bound at sampler(0) when sampling the HDR
    // resolve target during the post pass. Also reused by the bloom passes.
    pub(super) post_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    // Bloom mip chain (prefilter/downsample/upsample targets). Re-created
    // alongside `hdr_targets` whenever the drawable size changes.
    pub(super) bloom_targets: BloomTargets,
    // Prefilter / downsample / upsample pipelines for the bloom chain.
    pub(super) bloom_pipelines: BloomPipelines,
    // Pool backing the render graph's transient textures
    // (`gfx::render_graph::alias`). Owns `ao_output` today (relocated off SSAO);
    // a later stage aliases it with `bloom_top` on one `MTLHeap` slot. Rebuilt on
    // resize. See [`TransientTexturePool`].
    pub(super) transient_pool: TransientTexturePool,
    // Post-process tunables (bloom intensity / threshold / knee). Pushed to
    // the bloom prefilter and composite fragment shaders. `bloom_intensity`
    // of 0 skips the bloom passes entirely.
    pub(super) post_process: crate::gfx::render_types::PostProcessParams,
    // 3D colour-grading LUT sampled in the composite pass. Holds the declared
    // `ColorLut` payload, or a 2x2x2 identity LUT when the world declares
    // none, so the composite pass binds a valid 3D texture either way.
    pub(super) color_lut: Retained<ProtocolObject<dyn MTLTexture>>,
    // Temporal-anti-aliasing feature state: the toggle, resolve pipeline,
    // ping-pong history buffers, and per-frame bookkeeping. See [`TaaState`].
    pub(super) taa: TaaState,
    // Previous frame's un-jittered view-projection, fed to the velocity
    // pre-pass to reproject motion. Identity until the first frame completes.
    // (Shared by both TAA and the upscaler when either drives the velocity
    // pre-pass, so it is kept flat rather than under `taa`.)
    pub(super) prev_view_proj: [[f32; 4]; 4],
    // MetalFX-temporal-upscaling feature state: the scaler, the input/output
    // scale ratio, the per-frame projection jitter, and the history-reset
    // flag. When the scaler is `Some`, the 3D scene renders at
    // `(output * upscale.scale)` and the scaler reconstructs a
    // drawable-resolution image the bloom + composite stack reads as
    // `scene_color`; the TAA pass is bypassed (the scaler accumulates
    // temporally), though the velocity pre-pass + projection jitter still
    // run. See [`UpscaleState`].
    pub(super) upscale: UpscaleState,
    // SSAO (GTAO) feature state: resolved tunables, occlusion targets, the
    // kernel + blur pipelines, and the 1×1 white fallback. See [`SsaoState`].
    pub(super) ssao: SsaoState,
    // Screen-space-reflection feature state: resolved tunables, the resolve
    // output target (shared with SSGI/RT), and the resolve pipeline. See
    // [`SsrState`].
    pub(super) ssr: SsrState,
    // Unified G-buffer pre-pass feature state: the shared normal+depth /
    // roughness / velocity / sampleable-depth targets plus the static /
    // instanced / skinned pipelines. See [`GBufferState`].
    pub(super) gbuffer: GBufferState,
    // Screen-space-GI feature state: resolved tunables, the `gi` gather
    // target, and the gather + composite pipelines. See [`SsgiState`].
    pub(super) ssgi: SsgiState,
    // Resolved + clamped ray-traced-reflection tunables. `Some` only when the
    // Hardware-ray-traced-reflection feature state: resolved tunables, the
    // scene acceleration structure, the dynamic-update mode + failure flag, and
    // the resolve / textured-resolve / skinning pipelines. See [`RtState`].
    pub(super) rt: RtState,
    // Projected-decal feature state: the decal records (+ tombstone
    // free-list), the pipeline, the shared unit-cube geometry, and the
    // sampler. See [`DecalState`]. The pipeline / cube buffers / sampler are
    // built lazily at init (≥1 declared decal) or on the first runtime
    // [`MtlContext::add_decal`].
    pub(super) decal: DecalState,
    // Volumetric-fog feature state: resolved tunables, the ray-march
    // pipeline, and the froxel-volume compute pipeline + 3D output volume.
    // See [`FogState`].
    pub(super) fog: FogState,
    // Particle-system feature state: the per-emitter records (+ tombstone
    // free-list), the parallel per-emitter GPU pools, the shared compute +
    // render pipelines, and the per-frame timing bookkeeping. See
    // [`ParticleState`].
    pub(super) particle: ParticleState,
    // Auto-exposure feature state: resolved tunables, the EMA-tracked adapted
    // EV + authored bias, the histogram/average compute pipelines + buffers,
    // and the per-frame timing bookkeeping. See [`AutoExposureGpu`].
    pub(super) auto_exposure: AutoExposureGpu,
    // True only under `cn debug`. Switches the built-in `.metal` source loader
    // to a disk-first read with embedded fallback, so a saved shader edit is
    // picked up by [`MtlContext::reload_shaders`] (triggered by either the
    // `reload-shaders` debug command or the filesystem watcher). False under
    // `cn run`: production keeps the static include_str!-baked path.
    pub(super) hot_reload: bool,
    // Atomic flag set by the `notify` filesystem watcher or the debug WS
    // `reload-shaders` command. Polled at the top of `draw_frame`; when set,
    // `MtlContext::reload_shaders` rebuilds every built-in pipeline before
    // the next frame's passes run. `Some` only when `hot_reload` is on; the
    // debug server reads its `Arc` clone via `GraphicsSystem`.
    pub(super) shader_reload_pending: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    // Live `notify` watcher handle; dropping it stops the watcher. `Some`
    // only when `hot_reload` is on. Held purely for lifetime: the watcher
    // pushes events into `shader_reload_pending` directly.
    #[allow(dead_code)]
    pub(super) shader_watcher: Option<crate::metal::hot_reload::WatcherHandle>,
    // Previous frame's model matrix for every `draw_objects` entry, parallel
    // to it. The velocity pre-pass diffs current against previous so props
    // moved via `update_model` produce a correct motion vector.
    pub(super) prev_draw_models: Vec<[[f32; 4]; 4]>,
    // Skinned-mesh rendering feature state: the main + shadow pipelines, the
    // shared skinned vertex / index buffers, the per-mesh draw objects, and
    // the current + previous joint-palette matrices. See [`SkinnedState`].
    pub(super) skinned: SkinnedState,
    // Sub-allocators for the streamed-mesh regions of `vertex_buffer` and
    // `index_buffer`. Seeded at init by evicting every streamed mesh; from
    // then on `upload_mesh` / `evict_mesh` allocate and free byte ranges so a
    // streamed mesh can be placed wherever there is free space.
    pub(super) mesh_vtx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub(super) mesh_idx_alloc: crate::gfx::range_alloc::RangeAllocator,
    // Sub-allocators for the chunk-geometry headroom region appended to
    // `vertex_buffer` / `index_buffer` by `setup_chunk_streaming`. They manage
    // a byte range disjoint from the build-time geometry and the
    // mesh-streaming allocators, so a streamed `VoxelWorld` chunk never
    // collides with static geometry. Empty until `setup_chunk_streaming` runs.
    pub(super) chunk_vtx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub(super) chunk_idx_alloc: crate::gfx::range_alloc::RangeAllocator,
    // `draw_objects` slots vacated by removed chunks, reused by the next
    // `add_chunk_mesh` so the draw list does not grow without bound as the
    // camera roams an infinite world.
    pub(super) chunk_free_slots: Vec<usize>,
    // None in embedded mode (no separate NSWindow is created).
    pub(super) window: Option<Retained<NSWindow>>,
    // MTKView with isPaused=true and enableSetNeedsDisplay=false so its internal
    // display link never fires. draw() is called manually from draw_frame().
    pub(super) mtk_view: Retained<MTKView>,
    pub(super) window_closed: bool,
    // Whether draw_frame should pump NSEvents and honour cursor capture.
    // True for windowed mode and for the blocking-in-view play path; false
    // for the preview (which lets SwiftUI own input dispatch).
    pub(super) pump_events: bool,
    pub(super) was_visible: bool,
    pub(super) cursor_captured: bool,
    // Set when the cursor is released via Escape so a subsequent left-click
    // recaptures it rather than firing a UI click event.
    pub(super) recapture_on_click: bool,
    // Whether the OS cursor is currently hidden for an in-engine UI cursor
    // (e.g. a MainMenu). Tracked so `set_ui_cursor_hidden` only calls the
    // ref-counted NSCursor hide/unhide on a transition, not every frame.
    pub(super) ui_cursor_hidden: bool,
    // A togglable menu coexists with a captured camera (a MainMenu over a
    // Camera3D world). When set, Escape routes to the ECS and clicks never
    // recapture; GraphicsSystem drives capture from the active menu instead.
    pub(super) menu_mode: bool,
    // Authoritative native-fullscreen state, kept in sync by `window_delegate`
    // (the NSWindow `FullScreen` style-mask bit lags the animated transition).
    // Read by `set_window_mode` / `set_window_size`; an `AtomicBool` because the
    // delegate stores into it from AppKit's notification callbacks. Always
    // false in embedded mode (no NSWindow to go fullscreen).
    pub(super) fullscreen: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // NSWindowDelegate that tracks the fullscreen transition. None in embedded
    // mode. Retained here because NSWindow holds its delegate as a zeroing weak
    // reference, so dropping this would detach the delegate; the field is never
    // read directly (the delegate communicates through `fullscreen`).
    #[allow(dead_code)]
    pub(super) window_delegate: Option<Retained<super::window_delegate::WindowDelegate>>,
    pub(super) keys: KeyState,
    // The runtime movement key map (canonical action -> key). `handle_key`
    // decodes physical events through this instead of hardcoded keys, so a
    // settings-menu rebind takes effect immediately. Defaults to W/S/A/D/Shift/
    // Space/E; GraphicsSystem pushes any persisted override via `set_keymap`.
    pub(super) keymap: crate::gfx::keymap::KeyMap,
    // Render statistics for the most recent frame: draw-call and object
    // counts (filled by `draw_frame`) plus the GPU frame time pulled from
    // `gpu_time_us`. Surfaced to the profiler overlay via `render_stats()`.
    pub(super) frame_stats: crate::gfx::profile::RenderStats,
    // GPU execution time of the last completed frame, in microseconds.
    // Written by each command buffer's completion handler, which runs on a
    // GPU callback thread, so it is shared behind an atomic.
    pub(super) gpu_time_us: std::sync::Arc<std::sync::atomic::AtomicU32>,
    // Set once the frame render command buffer is observed to have faulted on
    // the GPU (in its completion handler, on a GPU callback thread). A render
    // fault is the usual *origin* of a `SubmissionsIgnored` cascade that then
    // shows up downstream on the next acceleration-structure build; logging the
    // render buffer's own error names the real culprit. Logged once (this flag
    // throttles it) so a per-frame fault streak does not spam at frame rate.
    pub(super) render_fault_logged: std::sync::Arc<std::sync::atomic::AtomicBool>,
    // Count of render-graph per-pass command-buffer faults logged so far.
    // Each graph pass commits its own command buffer; this throttle logs the
    // first handful (with the pass name + error) so the *original* fault in a
    // `SubmissionsIgnored`/`InnocentVictim` cascade (which pass actually broke)
    // is identifiable, while later victims do not spam at frame rate.
    pub(super) pass_fault_count: std::sync::Arc<std::sync::atomic::AtomicU32>,
    // Per-pass GPU sample buffers, when the active device supports the
    // `MTLCommonCounterSetTimestamp` counter set. Each `draw_frame` rotates
    // to the next slot in the ring; the completion handler resolves that
    // slot into [`pass_times_us`] for the profiler overlay.
    pub(super) pass_timing: Option<super::pass_timing::PassTimingResources>,
    // Per-pass GPU microseconds from the most recently resolved frame. One
    // atomic per pass slot, shared between the GPU completion handler that
    // writes it and `render_stats()` that reads it.
    pub(super) pass_times_us:
        std::sync::Arc<[std::sync::atomic::AtomicU32; super::pass_timing::PASS_COUNT]>,
    // Atomic accumulator the parallel-dispatched workers fetch_add their
    // draw counts into. Drained into `frame_stats.draw_calls` at the end
    // of `execute_graph`. AtomicU32 because workers may run concurrently.
    pub(super) draw_calls_accum: std::sync::atomic::AtomicU32,
    // Frames-in-flight pacing. `draw_frame` acquires a slot before encoding
    // and the frame command buffer's completion handler releases it once the
    // GPU retires the frame, bounding how far the CPU may queue ahead of the
    // GPU (and thus how many sets of per-frame transient buffers can pile up).
    pub(super) frame_pacing: super::frame_pacing::FrameInFlight,
    // Ring depth for the per-frame transient buffers below: equal to the
    // frames-in-flight count (≥1). The fence guarantees frame `R − depth` has
    // retired before frame `R` reuses ring slot `R % depth`, so overwriting
    // that slot's buffer never races an in-flight GPU read.
    pub(super) frames_in_flight: usize,
    // Monotonic counter over frames that build per-frame buffers; `% depth`
    // selects this frame's ring slot. Advanced once per such frame.
    pub(super) frame_ring_index: u64,
    // Ring of per-frame `GpuObjectData` buffers for the bindless static pass,
    // replacing a fresh allocation each frame. Written by `build_object_buffer`.
    pub(super) object_ring: super::transient::TransientRing,
    // Ring of per-frame `GpuDrawArgs` buffers for the GPU-cull pass. Written by
    // `build_draw_args_buffer`.
    pub(super) draw_args_ring: super::transient::TransientRing,
    // Ring of per-frame `prev_model` buffers for the GPU-driven G-buffer/velocity
    // pre-pass: one column-major `float4x4` per cull record, indexed
    // identically to the object buffer. Written by `build_gbuffer_prev_models`.
    pub(super) prev_model_ring: super::transient::TransientRing,
    // Ring of per-frame `BindlessTextures` argument buffers. The argument
    // encoder fills the slot in place each frame; see `build_bindless_texture_args`.
    pub(super) bindless_tex_ring: super::transient::TransientRing,
    // Ring of per-skinned-object joint-palette buffers, one inner buffer per
    // object, for the current and previous (velocity) poses. Written by
    // `build_joint_buffers`.
    pub(super) joint_ring: super::transient::JointRing,
    pub(super) prev_joint_ring: super::transient::JointRing,
    // Ring of per-cluster instance-matrix buffers. `prepare_instanced_draws`
    // fills this frame's slot once; the main / SSR / SSAO / velocity passes
    // share the result instead of each re-uploading the instance matrices.
    pub(super) instance_ring: super::transient::InstanceRing,
    // Reused scratch for the `GpuObjectData` / `GpuDrawArgs` builds so the
    // per-frame `collect` reuses one heap allocation instead of allocating a
    // fresh `Vec` each frame. `mem::take`n during the build and returned after.
    pub(super) object_scratch: Vec<crate::gfx::render_types::GpuObjectData>,
    pub(super) draw_args_scratch: Vec<crate::gfx::render_types::GpuDrawArgs>,
    // Reused scratch for the per-frame `prev_model` build, same
    // pattern as `object_scratch`.
    pub(super) prev_model_scratch: Vec<[[f32; 4]; 4]>,
    // Transparent water-surface pipeline. `Some` only when the world
    // declared ≥1 `WaterSurface`; the transparent pass executor short-
    // circuits otherwise.
    pub(super) water_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Ray-traced water pipeline: traces a sharp reflection ray against the scene
    // BVH instead of sampling the probe cube. `Some` only when the world has
    // ≥1 `WaterSurface` AND the device supports ray tracing; selected per-frame
    // only while `rt.accel` is live (RT on), the probe pipeline otherwise. This
    // is the FLAT variant (per-object material tint as albedo); the non-bindless
    // RT fallback.
    pub(super) water_pipeline_rt: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Ray-traced water pipeline, TEXTURED variant: samples the reflected hit's
    // albedo / normal / emissive maps from the bindless pool (bound at buffer 10
    // for water, since the main pass's index 7 is the ProbeSet here). Built under
    // the same RT gate; selected over the flat variant only in a bindless world
    // (where the pool exists), while `rt.accel` is live.
    pub(super) water_pipeline_rt_textured:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // One GPU record per `WaterSurface` asset: tessellated VB+IB plus the
    // per-surface fragment / vertex uniforms (rebuilt at init from the
    // asset; `prefilter_mip_count` is patched per-frame).
    pub(super) water_surfaces: Vec<super::water::WaterSurfaceRecord>,
    // Planar reflection targets, one set per distinct reflector plane (water
    // surfaces + glass panes, grouped by `assign_planar_slots`). `Some` only when
    // the world declared >=1 such reflector; the scene is re-rendered mirrored
    // across each plane into these each frame (RT off) and the reflective shader
    // samples the resolve of its slot. Rebuilt on resize alongside `hdr_targets`.
    pub(super) planar_reflection: Option<super::planar::PlanarReflectionSet>,
    // Shared pipeline for the `GlassPanel` transparent producer. `Some` only
    // when the world declared ≥1 `GlassPanel`.
    pub(super) glass_pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Ray-traced glass pipeline: traces a sharp reflection ray against the scene
    // BVH instead of sampling the probe cube. `Some` only when the world has
    // ≥1 `GlassPanel` AND the device supports ray tracing; selected per-frame
    // only while `rt.accel` is live (RT on), the probe pipeline otherwise. This
    // is the FLAT variant (per-object material tint as albedo); the non-bindless
    // RT fallback.
    pub(super) glass_pipeline_rt: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Ray-traced glass pipeline, TEXTURED variant: samples the reflected hit's
    // albedo / normal / emissive maps from the bindless pool (bound at buffer 10
    // for glass, since the main pass's index 7 is the ProbeSet here). Built under
    // the same RT gate; selected over the flat variant only in a bindless world
    // (where the pool exists), while `rt.accel` is live.
    pub(super) glass_pipeline_rt_textured:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Ray-traced transparent glass MESH pipelines (`glass_mesh_rt.metal`): an
    // imported `Material` with `transparent: true` routed through the transparent
    // pass with a per-pixel RT trace off the interpolated mesh normal, instead of
    // the Layer-1 opaque-reflective fallback. Built only on RT-capable devices
    // (regardless of whether the world declares a transparent material -- a live
    // RT toggle then has the pipeline ready). `glass_mesh_pipeline_rt.is_some()`
    // gates the whole transparent-mesh path: when live (RT on) transparent meshes
    // are skipped in the opaque pass + the RT BLAS and drawn here; otherwise they
    // render opaque (Layer 1). The flat variant uses the reflected hit's material
    // tint; the textured variant samples the bindless pool (selected in a bindless
    // world).
    pub(super) glass_mesh_pipeline_rt: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(super) glass_mesh_pipeline_rt_textured:
        Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Indices into `draw_objects` of every see-through glass mesh (its material
    // has both `transparent` and `see_through` set), precomputed at init so the
    // per-frame Layer 2 producer does not rescan all objects. Empty on non-RT
    // devices or when no material opts into see-through (those transparent
    // meshes then render opaque, Layer 1). The objects stay IN `draw_objects` (a
    // DrawObject's position is a key into the cull / prev-model / RT parallel
    // arrays); this list only marks which to reroute. A non-empty list (plus a
    // built mesh pipeline) is what enables the Layer 2 path -- see
    // `seethrough_meshes_enabled`. See-through is opt-in per `Material` because
    // it only looks right when the space behind the glass is modelled; without
    // it, Layer 1's tinted reflective glass hides the interior.
    pub(super) seethrough_mesh_indices: Vec<usize>,
    // One GPU record per `GlassPanel` asset: the static world-space quad VB+IB
    // plus the per-panel uniforms. Contributes to the transparent pass.
    pub(super) glass_panels: Vec<super::glass::GlassPanelRecord>,
    // One GPU record per `SdfVolume` asset: the per-volume render
    // pipeline (compiled lazily at init from the user's fragment
    // shader source + the engine-shipped helpers/template) plus the
    // static per-volume uniforms (centre, extent, params, …). Drives
    // the raymarch pass at `PassId::Raymarch`.
    pub(super) raymarch_volumes: Vec<super::raymarch::RaymarchVolumeRecord>,
    // Shared unit-cube proxy geometry the raymarch pass rasterises (8
    // vertices, 36 indices). `Some` whenever any `SdfVolume` exists in
    // the world; the encoder reads them per-frame and the asset cost
    // is fixed (96 + 72 bytes).
    pub(super) raymarch_cube_vertex_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    pub(super) raymarch_cube_index_buffer: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

// MtlContext is only ever accessed from the main thread (as documented on the
// struct). The Retained<ProtocolObject<...>> Metal handles aren't Send by
// default, but `RenderBackend: Send` requires it so GraphicsSystem can box
// the backend behind a trait object. Mirrors DxContext / VkContext.
unsafe impl Send for MtlContext {}

// Debug-only guard that the caller is on the main thread.
//
// The `unsafe impl Send for MtlContext` above is sound only because the
// context is touched from the main thread alone: AppKit and Metal command
// submission are both main-thread-affine. `draw_frame` proves this with a
// `MainThreadMarker`, but the `RenderBackend` mutation entry points (reached
// through the boxed trait object) did not, so scheduling `GraphicsSystem`
// off the main thread would silently race AppKit/Metal instead of failing.
// This makes that mistake panic loudly in debug builds and compiles to
// nothing in release. `entry` is the offending method name, for the message.
#[inline]
#[track_caller]
pub(super) fn debug_assert_main_thread(entry: &str) {
    debug_assert!(
        objc2::MainThreadMarker::new().is_some(),
        "{entry} must be called from the main thread: MtlContext is main-thread-only \
         (see `unsafe impl Send for MtlContext`); driving GraphicsSystem off the main \
         thread races AppKit/Metal",
    );
}

impl MtlContext {
    // Number of records the GPU-driven cull processes this frame: the static
    // draw objects, every folded instance, then every folded skinned object.
    // Metal has no separate `n_objects` field; `draw_objects.len()` is the
    // static count. Drives the cull dispatch width + `object_count` uniform, the
    // shared ICB capacity, and the indirect-draw `NSRange`. Equals
    // `draw_objects.len()` for static-only worlds, so those paths are untouched.
    pub(super) fn cull_count(&self) -> usize {
        self.draw_objects.len() + self.n_instances + self.n_skinned
    }

    // The prefiltered radiance cube for probe array slot `i`: the baked probe
    // when present, else the sky `env_map` prefilter (a valid fallback for unused
    // slots and for slots past the baked count). The skybox + diffuse always use
    // `env_map` directly, so they keep the sky regardless.
    pub(super) fn probe_cube_or_sky(&self, i: usize) -> &ProtocolObject<dyn MTLTexture> {
        match self.probe_maps.get(i) {
            Some(p) => p.prefilter.as_ref(),
            None => self.env_map.prefilter.as_ref(),
        }
    }

    // Index in the unified cull list where the folded skinned records begin
    // (static objects + instances precede them). The cull kernel draws records
    // at or past this through the skinned u16 index buffer (the skinned tail),
    // and the main pass binds the deformed vertex buffer for that range. Equals
    // `cull_count()` when no skinned mesh is folded.
    pub(super) fn skinned_record_base(&self) -> usize {
        self.draw_objects.len() + self.n_instances
    }

    // The buffer to bind at the cull kernel's skinned-index slot (buffer 6):
    // the skinned u16 index buffer when a SkinnedMesh has uploaded, else the
    // static index buffer as a harmless placeholder. The kernel only reads it
    // for records at/after `skinned_base`, which equals `cull_count()` (so the
    // skinned branch never fires) whenever the skinned buffer is absent; Metal
    // still requires a referenced buffer to be bound, hence the placeholder.
    pub(super) fn skinned_index_or_placeholder(
        &self,
    ) -> &ProtocolObject<dyn objc2_metal::MTLBuffer> {
        match self.skinned.index_buffer.as_ref() {
            Some(b) => b.as_ref(),
            None => self.index_buffer.as_ref(),
        }
    }

    // Ensure the GPU-driven cull indirect command buffer has a command slot
    // for every one of `count` draw objects, rebuilding it (and re-encoding
    // its argument buffer) when the draw list has outgrown it. A no-op for
    // non-bindless contexts, which have no cull pipeline. New capacity is
    // rounded up to the next power of two so streamed chunks growing
    // `draw_objects` do not rebuild the ICB every frame.
    //
    // The per-object `cull_status_buffer` (always, for the phase-1 kernel's
    // buffer(5) binding) and (under two-pass occlusion) the phase-2 ICB
    // `cull_icb_2` + its argument buffer are grown in lockstep on the same
    // trigger, so all three stay sized to the live draw-object count.
    pub(super) fn ensure_icb_capacity(&mut self, count: usize) -> Result<(), String> {
        // Retained is reference-counted; cloning the handle lets the rest of
        // the method mutate `self` without holding a borrow on the encoder.
        let arg_encoder = match &self.cull.icb_arg_encoder {
            Some(e) => e.clone(),
            None => return Ok(()),
        };
        if self.cull.icb.is_some() && count <= self.cull.icb_capacity {
            return Ok(());
        }
        let new_cap = count.next_power_of_two().max(64);

        let icb = self.build_cull_icb(new_cap)?;

        // The argument buffer is a fixed-size handle to the ICB; create it
        // once, then re-encode it to point at each freshly built ICB.
        if self.cull.icb_arg_buffer.is_none() {
            let len = arg_encoder.encodedLength().max(16);
            let buf = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to create ICB argument buffer")?;
            self.cull.icb_arg_buffer = Some(buf);
        }
        let arg_buf = self
            .cull
            .icb_arg_buffer
            .as_ref()
            .expect("ICB argument buffer was just ensured");
        // SAFETY: the argument buffer was sized to `encodedLength()`, and the
        // ICB is encoded at slot 0: the single `[[id(0)]]` member of the
        // kernel's `ICBContainer` argument-buffer struct.
        unsafe {
            arg_encoder.setArgumentBuffer_offset(Some(arg_buf), 0);
            arg_encoder.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
        }

        // Per-object status buffer: one u32 per command slot, private storage
        // (GPU-written by phase-1 cull, GPU-read by phase-2 cull). Always
        // allocated so the phase-1 kernel's buffer(5) binding always resolves.
        let status = self
            .device
            .newBufferWithLength_options(
                new_cap * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or("failed to create cull status buffer")?;
        self.cull.status_buffer = Some(status);

        // Second-pass ICB + argument buffer, only when two-pass occlusion is on.
        if self.cull.two_pass_occlusion {
            let arg_encoder2 = match &self.cull.icb_2_arg_encoder {
                Some(e) => e.clone(),
                None => {
                    return Err(
                        "two-pass occlusion on but phase-2 ICB argument encoder missing".into(),
                    );
                }
            };
            let icb2 = self.build_cull_icb(new_cap)?;
            if self.cull.icb_2_arg_buffer.is_none() {
                let len = arg_encoder2.encodedLength().max(16);
                let buf = self
                    .device
                    .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                    .ok_or("failed to create phase-2 ICB argument buffer")?;
                self.cull.icb_2_arg_buffer = Some(buf);
            }
            let arg_buf2 = self
                .cull
                .icb_2_arg_buffer
                .as_ref()
                .expect("phase-2 ICB argument buffer was just ensured");
            // SAFETY: same layout as the phase-1 encoder: the phase-2 kernel's
            // ICBContainer argument buffer is a single `[[id(0)]]` member.
            unsafe {
                arg_encoder2.setArgumentBuffer_offset(Some(arg_buf2), 0);
                arg_encoder2.setIndirectCommandBuffer_atIndex(Some(&icb2), 0);
            }
            self.cull.icb_2 = Some(icb2);
        }

        self.cull.icb = Some(icb);
        self.cull.icb_capacity = new_cap;
        Ok(())
    }

    // Ensure the GPU-driven cascaded-shadow ICB has a command slot
    // for every cascade of every record: `NUM_SHADOW_CASCADES * count` total
    // (cascade `c`'s commands live at `[c*count, (c+1)*count)`, the same stride
    // `encode_shadow_culls` writes at and the shadow render pass executes). A
    // no-op for non-bindless / no-shadow contexts (no shadow cull arg encoder).
    // Rounded to the next power of two so a streamed chunk growing `cull_count()`
    // does not rebuild the ICB every frame. Called from `draw_frame` (where
    // `&mut self` is available) right after `ensure_icb_capacity`, so the encode
    // pass only ever reads the sized ICB.
    pub(super) fn ensure_shadow_icb_capacity(&mut self, count: usize) -> Result<(), String> {
        let arg_encoder = match &self.cull.shadow_icb_arg_encoder {
            Some(e) => e.clone(),
            None => return Ok(()),
        };
        let needed = count.saturating_mul(NUM_SHADOW_CASCADES);
        if self.cull.shadow_icb.is_some() && needed <= self.cull.shadow_icb_capacity {
            return Ok(());
        }
        let new_cap = needed.next_power_of_two().max(64);
        let icb = self.build_cull_icb(new_cap)?;
        if self.cull.shadow_icb_arg_buffer.is_none() {
            let len = arg_encoder.encodedLength().max(16);
            let buf = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to create shadow ICB argument buffer")?;
            self.cull.shadow_icb_arg_buffer = Some(buf);
        }
        let arg_buf = self
            .cull
            .shadow_icb_arg_buffer
            .as_ref()
            .expect("shadow ICB argument buffer was just ensured");
        // SAFETY: the argument buffer was sized to `encodedLength()`, and the ICB
        // is encoded at slot 0 (the single `[[id(0)]]` member of the kernel's
        // `ICBContainer` argument-buffer struct), exactly like the main ICB.
        unsafe {
            arg_encoder.setArgumentBuffer_offset(Some(arg_buf), 0);
            arg_encoder.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
        }
        self.cull.shadow_icb = Some(icb);
        self.cull.shadow_icb_capacity = new_cap;
        Ok(())
    }

    // Ensure `slot_count` per-planar-slot mirror cull ICBs exist, each with a
    // command slot for every one of `count` draw objects (static + folded
    // instances + skinned, exactly like the main ICB). The slots' ICBs + argument
    // buffers are rebuilt when the slot count changes or the draw list outgrows
    // the capacity; the shared single-pass status scratch grows in lockstep. A
    // no-op for non-bindless contexts (no main ICB argument encoder to reuse) and
    // when `slot_count` is 0 (no planar set -> the slots are cleared). Called from
    // `draw_frame` right after `ensure_icb_capacity`, so the planar pass only ever
    // reads a sized mirror ICB.
    pub(super) fn ensure_mirror_icb_capacity(
        &mut self,
        slot_count: usize,
        count: usize,
    ) -> Result<(), String> {
        if slot_count == 0 {
            self.cull.mirror_slots.clear();
            self.cull.mirror_status = None;
            self.cull.mirror_icb_capacity = 0;
            return Ok(());
        }
        // Reuse the main phase-1 ICB argument encoder: a mirror ICB has the
        // identical `ICBContainer` layout (one `[[id(0)]]` member). Absent on
        // non-bindless contexts, where there is no GPU cull to mirror.
        let arg_encoder = match &self.cull.icb_arg_encoder {
            Some(e) => e.clone(),
            None => return Ok(()),
        };
        if self.cull.mirror_slots.len() == slot_count && count <= self.cull.mirror_icb_capacity {
            return Ok(());
        }
        let new_cap = count.next_power_of_two().max(64);
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            let icb = self.build_cull_icb(new_cap)?;
            let len = arg_encoder.encodedLength().max(16);
            let arg_buffer = self
                .device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                .ok_or("failed to create mirror ICB argument buffer")?;
            // SAFETY: the argument buffer is sized to `encodedLength()` and the
            // ICB is encoded at slot 0, exactly like the main + shadow ICBs. Each
            // slot re-points the shared encoder at its own (arg buffer, ICB) pair;
            // the encoding is fully written before the next slot re-points it.
            unsafe {
                arg_encoder.setArgumentBuffer_offset(Some(&arg_buffer), 0);
                arg_encoder.setIndirectCommandBuffer_atIndex(Some(&icb), 0);
            }
            slots.push(super::cull::MirrorCullSlot { icb, arg_buffer });
        }
        // Shared single-pass status scratch: the mirror cull writes per-object
        // status the same as phase 1, but nothing reads it (no phase-2 over the
        // mirror), so one buffer serves every slot. One u32 per command slot.
        let status = self
            .device
            .newBufferWithLength_options(
                new_cap * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or("failed to create mirror cull status buffer")?;
        self.cull.mirror_status = Some(status);
        self.cull.mirror_slots = slots;
        self.cull.mirror_icb_capacity = new_cap;
        Ok(())
    }

    // Create one `DrawIndexed` indirect command buffer with `cap` command
    // slots. Both the phase-1 (`cull_icb`) and phase-2 (`cull_icb_2`) ICBs
    // share this shape: each command inherits the render encoder's buffer
    // bindings + pipeline state, so the cull kernels only encode the
    // indexed-draw arguments. Private storage keeps it GPU-resident
    // (kernel-written, render-pass-consumed, never CPU-touched).
    fn build_cull_icb(
        &self,
        cap: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>, String> {
        let desc = MTLIndirectCommandBufferDescriptor::new();
        desc.setCommandTypes(MTLIndirectCommandType::DrawIndexed);
        desc.setInheritBuffers(true);
        desc.setInheritPipelineState(true);
        desc.setMaxVertexBufferBindCount(0);
        desc.setMaxFragmentBufferBindCount(0);
        // SAFETY: `cap` is a valid command count; private storage as documented.
        unsafe {
            self.device
                .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                    &desc,
                    cap,
                    MTLResourceOptions::StorageModePrivate,
                )
        }
        .ok_or_else(|| "failed to create indirect command buffer".to_string())
    }

    // Logical (point-space) size of the drawable area. Used by systems to pass
    // viewport dimensions to text layout (e.g. for centred labels).
    pub fn logical_size(&self) -> (f32, f32) {
        let s = self.mtk_view.bounds().size;
        (s.width as f32, s.height as f32)
    }

    // Device capability flags for the settings menu. Ray tracing is queried
    // from the live MTLDevice (cheap; the same check the RT pass gates on).
    pub fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities {
        crate::gfx::backend::DeviceCapabilities {
            ray_tracing: super::raytrace::raytracing_supported(&self.device),
        }
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. The GPU frame time is the last value reported by a completed
    // command buffer, so it may lag the draw counts by a frame or two.
    pub fn render_stats(&self) -> crate::gfx::profile::RenderStats {
        let mut stats = self.frame_stats;
        stats.gpu_frame_us = self.gpu_time_us.load(std::sync::atomic::Ordering::Relaxed);
        // Per-pass timings are filled in pass-index order with their stable
        // names, leaving slots past PASS_COUNT at the default ("", 0).
        // Reports zero for any pass that did not write its sample slot this
        // frame (e.g. SSR when disabled, or any pass not yet wired up).
        for (i, name) in super::pass_timing::PASS_NAMES.iter().enumerate() {
            let micros = self.pass_times_us[i].load(std::sync::atomic::Ordering::Relaxed);
            stats.pass_times_us[i] = (*name, micros);
        }
        // Surface the auto-exposure EMA state. `None` when the world did not
        // opt in to auto-exposure (the static-exposure path leaves the field
        // empty so the StatHud chip stays blank).
        stats.auto_exposure_ev = self.auto_exposure.state.as_ref().map(|s| s.current_ev);
        // Surface the active panel's EDR headroom. `None` on SDR: both the
        // world-opt-out case and the request-on-an-SDR-display fallback case
        // map to the same blank chip.
        stats.max_edr = self.max_edr;
        stats
    }

    // Push a new view matrix; takes effect on the next draw_frame call.
    pub fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.view_matrix = matrix;
    }

    // Update the model matrix for a single draw object by index.
    // Has no effect if the index is out of range.
    pub fn update_model(&mut self, index: usize, model: [[f32; 4]; 4]) {
        if let Some(obj) = self.draw_objects.get_mut(index) {
            obj.model = model;
        }
    }

    // Show or hide a single draw object. Hidden objects are skipped in both
    // the shadow and main passes. Has no effect if the index is out of range.
    pub fn update_visibility(&mut self, index: usize, visible: bool) {
        if let Some(obj) = self.draw_objects.get_mut(index) {
            obj.visible = visible;
        }
    }

    // Replace the framebuffer clear colour for the next draw_frame call.
    // Used by SceneReel to lerp toward black during FadeBlack transitions.
    pub fn update_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    // Append a new draw object that re-uses an existing draw slot's geometry
    // region (vertex/index offsets, base_vertex, LOD alternates) with a fresh
    // model matrix, texture slots, material, and cull distance. Driven by
    // `world.jsonl` hot-reload (`cn debug` only) when a newly authored Prop
    // references a mesh/model already present in the world. The clone is
    // marked non-cullable (sentinel AABB) and added to `always_draw` since
    // the init-time BVH cannot refit; the dynamically added prop is drawn
    // every frame like a streamed chunk. Returns the new draw_idx.
    pub fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: crate::gfx::render_types::MaterialUniforms,
        cull_distance: f32,
    ) -> Result<usize, String> {
        let src = self.draw_objects.get(src_draw_idx).ok_or_else(|| {
            format!(
                "clone_static_draw_object: src draw {} out of range",
                src_draw_idx
            )
        })?;
        let obj = DrawObject {
            vertex_offset: src.vertex_offset,
            vertex_count: src.vertex_count,
            index_offset: src.index_offset,
            index_count: src.index_count,
            base_vertex: src.base_vertex,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance,
            lod_alternates: src.lod_alternates.clone(),
        };
        self.draw_objects.push(obj);
        self.prev_draw_models.push(model);
        let idx = self.draw_objects.len() - 1;
        self.always_draw.push(idx as u32);
        // The cloned prop joins the RT-relevant draw set; the next RT update
        // folds it into the BVH (it reuses the source mesh's geometry slice, so
        // only this clone's BLAS is built).
        self.rt.topology_dirty = true;
        Ok(idx)
    }

    // Rewrite a draw slot's material parameters + texture/normal-map pool
    // indices in place. Driven by `world.jsonl` hot-reload (`cn debug` only).
    // Has no effect if the index is out of range.
    pub fn set_draw_material(
        &mut self,
        draw_idx: usize,
        material: crate::gfx::render_types::MaterialUniforms,
        texture_slot: usize,
        normal_map_slot: usize,
    ) {
        if let Some(obj) = self.draw_objects.get_mut(draw_idx) {
            obj.material = material;
            obj.texture_slot = texture_slot;
            obj.normal_map_slot = normal_map_slot;
            // A material edit can flip RT participation (`see_through`) and always
            // changes the geometry-table entry; flag a topology refresh so the
            // next RT update rebuilds the table (BLAS are reused -- geometry is
            // unchanged -- so this is cheap).
            self.rt.topology_dirty = true;
        }
    }

    // Rewrite a draw slot's `cull_distance` in place. Driven by
    // `world.jsonl` hot-reload (`cn debug` only). Has no effect if the index
    // is out of range.
    pub fn set_draw_cull_distance(&mut self, draw_idx: usize, cull_distance: f32) {
        if let Some(obj) = self.draw_objects.get_mut(draw_idx) {
            obj.cull_distance = cull_distance.max(0.0);
        }
    }

    // Append a projected-decal record at runtime, returning a stable slot
    // index the caller hands to [`Self::remove_decal`] later. Builds the
    // decal pipeline + unit-cube buffers on first use so a world that never
    // declared a decal still pays zero pipeline cost until the first add.
    // A vacated slot from [`Self::remove_decal`] is reused before growing
    // the vec so a steady-state spawn/despawn pattern (bullet holes,
    // footprints) does not grow `decals` without bound.
    pub fn add_decal(&mut self, record: crate::gfx::decal::DecalRecord) -> Result<usize, String> {
        if self.decal.pipeline.is_none() {
            let (ps, vbuf, ibuf, samp) = super::init::effects::build_decal_resources_for_runtime(
                &self.device,
                self.hot_reload,
            )?;
            self.decal.pipeline = Some(ps);
            self.decal.cube_vertex_buffer = Some(vbuf);
            self.decal.cube_index_buffer = Some(ibuf);
            self.decal.sampler = Some(samp);
        }
        let idx = if let Some(slot) = self.decal.free_slots.pop() {
            self.decal.records[slot] = Some(record);
            slot
        } else {
            self.decal.records.push(Some(record));
            self.decal.records.len() - 1
        };
        Ok(idx)
    }

    // Tombstone a runtime decal slot. The slot index returned by
    // [`Self::add_decal`] becomes invalid; the next add may reuse it.
    // Returns an error when the index is out of range or already tombstoned.
    // The decal pipeline + unit-cube buffers are kept around so a later add
    // does not pay the rebuild cost.
    pub fn remove_decal(&mut self, decal_id: usize) -> Result<(), String> {
        let slot = self
            .decal
            .records
            .get_mut(decal_id)
            .ok_or_else(|| format!("remove_decal: id {} out of range", decal_id))?;
        if slot.is_none() {
            return Err(format!("remove_decal: id {} already removed", decal_id));
        }
        *slot = None;
        self.decal.free_slots.push(decal_id);
        Ok(())
    }

    // Append a particle-emitter record at runtime, returning a stable slot
    // index. Allocates the per-emitter GPU pool + atomic counter buffer
    // (matching the init-time path) and builds the compute + render
    // pipelines on first use so a world that never declared an emitter
    // pays zero pipeline cost until the first add. Tombstoned slots from
    // [`Self::remove_emitter`] are reused before growing the vec.
    pub fn add_emitter(
        &mut self,
        record: crate::gfx::particles::ParticleEmitterRecord,
    ) -> Result<usize, String> {
        if self.particle.pipelines.is_none() {
            let pipelines =
                super::particle::build_particle_pipelines(&self.device, self.hot_reload)?;
            self.particle.pipelines = Some(pipelines);
        }
        let gpu_state = super::particle::build_emitter_gpu_state(&self.device, &record)?;
        let idx = if let Some(slot) = self.particle.free_slots.pop() {
            self.particle.records[slot] = Some(record);
            self.particle.emitter_state[slot] = Some(gpu_state);
            slot
        } else {
            self.particle.records.push(Some(record));
            self.particle.emitter_state.push(Some(gpu_state));
            self.particle.records.len() - 1
        };
        Ok(idx)
    }

    // Tombstone a runtime emitter slot. Drops the `ParticleEmitterGpuState`:
    // Metal keeps the underlying pool + counter buffers alive until any
    // in-flight command buffer referencing them completes, so this is safe
    // to call mid-frame between encode passes (the debug-WS path runs in
    // the `DebugHook::tick` window before the world step). Returns an error
    // when the index is out of range or already tombstoned.
    pub fn remove_emitter(&mut self, emitter_id: usize) -> Result<(), String> {
        let rec_slot = self
            .particle
            .records
            .get_mut(emitter_id)
            .ok_or_else(|| format!("remove_emitter: id {} out of range", emitter_id))?;
        if rec_slot.is_none() {
            return Err(format!("remove_emitter: id {} already removed", emitter_id));
        }
        *rec_slot = None;
        if let Some(gpu_slot) = self.particle.emitter_state.get_mut(emitter_id) {
            *gpu_slot = None;
        }
        self.particle.free_slots.push(emitter_id);
        Ok(())
    }

    // Returns true if the window has been closed by the user.
    pub fn window_closed(&self) -> bool {
        if self.window_closed {
            return true;
        }
        // Detect close via the red-X button: NSWindow.close() hides the window
        // without posting an ApplicationDefined event, so window_closed never
        // becomes true through the event pump alone. Guard with was_visible so
        // we don't misfire before the first frame appears.
        self.was_visible && self.window.as_ref().is_some_and(|w| !w.isVisible())
    }

    // Block until the GPU has finished all in-flight work.
    pub fn wait_idle(&self) {
        if let Some(cmd_buf) = self.command_queue.commandBuffer() {
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
    }
}

impl crate::gfx::scene_reel::SceneControl for MtlContext {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        self.update_visibility(draw_idx, visible);
    }

    fn update_clear_color(&mut self, color: [f32; 4]) {
        self.update_clear_color(color);
    }
}

impl Drop for MtlContext {
    fn drop(&mut self) {
        // Wait for all in-flight GPU work to finish before releasing Metal
        // objects. Without this, releasing the command queue while the GPU is
        // still executing a committed command buffer can corrupt ObjC retain
        // counts and produce EXC_BAD_ACCESS in objc_release.
        self.wait_idle();
        // Always release the cursor on teardown so the OS mouse association and
        // cursor visibility are restored even if the caller didn't do it.
        self.release_cursor();
        if let Some(ref window) = self.window {
            // Close the game window so it doesn't linger after the run loop exits.
            window.close();
        } else {
            // In embedded mode (no NSWindow), the MTKView was added as a subview.
            // Explicitly remove it so it doesn't outlive the preview session.
            self.mtk_view.removeFromSuperview();
        }
    }
}

// Reinterpret a POD slice as raw bytes for a GPU buffer copy.
// Copy the first `len` bytes of `src` into `dst`. Both must be
// shared-storage buffers at least `len` bytes long; used by
// `setup_chunk_streaming` to carry build-time geometry into a grown buffer.
pub(super) fn copy_buffer_prefix(
    src: &ProtocolObject<dyn MTLBuffer>,
    dst: &ProtocolObject<dyn MTLBuffer>,
    len: usize,
) {
    if len == 0 {
        return;
    }
    unsafe {
        let s = src.contents().as_ptr() as *const u8;
        let d = dst.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(s, d, len);
    }
}

pub(super) fn bytes_of_slice<T>(slice: &[T]) -> &[u8] {
    // SAFETY: reinterprets the slice as raw bytes over the same length. Sound
    // only for plain-old-data `T` with no padding/uninitialised bytes; every
    // caller passes a `#[repr(C)]` GPU upload type that satisfies this.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice)) }
}

// Copy `src` into a shared-storage buffer at `offset` bytes, bounds-checked
// against the buffer length.
pub(super) fn write_buffer_region(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    src: &[u8],
) -> Result<(), String> {
    let len = buffer.length();
    if offset.checked_add(src.len()).is_none_or(|end| end > len) {
        return Err(format!(
            "buffer write [{}, {}) exceeds buffer length {}",
            offset,
            offset.saturating_add(src.len()),
            len
        ));
    }
    if src.is_empty() {
        return Ok(());
    }
    let dst = buffer.contents().as_ptr() as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst.add(offset), src.len());
    }
    Ok(())
}

// Zero a `len`-byte region of a shared-storage buffer at `offset` bytes,
// bounds-checked against the buffer length.
pub(super) fn zero_buffer_region(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    offset: usize,
    len: usize,
) -> Result<(), String> {
    let buf_len = buffer.length();
    if offset.checked_add(len).is_none_or(|end| end > buf_len) {
        return Err(format!(
            "buffer zero [{}, {}) exceeds buffer length {}",
            offset,
            offset.saturating_add(len),
            buf_len
        ));
    }
    if len == 0 {
        return Ok(());
    }
    let dst = buffer.contents().as_ptr() as *mut u8;
    unsafe {
        std::ptr::write_bytes(dst.add(offset), 0, len);
    }
    Ok(())
}
