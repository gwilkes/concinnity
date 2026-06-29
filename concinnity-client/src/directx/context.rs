// D3D12 rendering context. Owns all GPU resources, the Win32 window, and input
// state. Mirrors the public API of VkContext / MtlContext so GraphicsSystem can
// drive all three backends identically.

use std::cell::RefCell;
use std::sync::OnceLock;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{GetCurrentThreadId, WaitForSingleObject};
use windows::core::Interface;

use crate::gfx::render_types::*;

use super::auto_exposure::AutoExposureResources;
use super::decal::*;
use super::fog::*;
use super::input::*;
use super::particle::{ParticleEmitterGpuState, ParticleResources};
use super::post::gbuffer::GbufferResources;
use super::post::ssao::*;
use super::post::ssr::*;
use super::post::taa::*;
use super::texture::*;
use super::window::*;

// Constants
pub(super) const FRAMES: usize = 3; // triple-buffered
// Constant buffer alignment required by D3D12.
pub(super) const CB_ALIGN: u64 = 256;

// Upper bound on the number of `SkinnedMesh` draws. The SRV heap is sized once
// at init, so a fixed block of `MAX_SKINNED_OBJECTS * 2` descriptors is
// reserved for skinned (albedo, normal) pairs even when no skinned mesh is
// declared (the reservation costs only descriptor-heap slots, mirroring the
// chunk-streaming SRV reservation). `upload_skinned` rejects worlds exceeding
// this count.
pub(super) const MAX_SKINNED_OBJECTS: usize = 64;

// Upper bound on the number of runtime-cloned static draw objects produced by
// `world.jsonl` hot-reload (`cn debug` only). Each clone reserves its own
// (albedo, normal) SRV pair in the descriptor heap, so the heap is sized for
// `MAX_CLONE_DRAWS * 2` extra slots at init. Editor-only; exhausting the
// pool means a long editing session has added more new Props than the cap;
// `clone_static_draw_object` errors out at that boundary.
pub(super) const MAX_CLONE_DRAWS: usize = 128;

pub(super) fn align256(n: u64) -> u64 {
    (n + CB_ALIGN - 1) & !(CB_ALIGN - 1)
}

// One LOD bucket of an instanced cluster for the current frame. Filled by
// `build_instance_upload` once at the top of every frame; consumed by
// every instanced draw site so all passes agree on the per-instance LOD
// pick + the bucket-ordered byte layout in the cluster's upload buffer.
#[derive(Clone, Debug)]
pub(super) struct InstanceBucketLayout {
    // Byte offset of this bucket's matrices within the cluster's per-frame
    // upload buffer. Pair with the cluster's upload-buffer GPU virtual
    // address to point a root SRV at the bucket's slice.
    pub instance_byte_offset: u64,
    // Number of instance matrices in this bucket.
    pub instance_count: u32,
    // LOD slice's index-buffer offset (in u32 indices). Drives the
    // `StartIndexLocation` arg of `DrawIndexedInstanced`.
    pub index_offset: usize,
    // LOD slice's index count.
    pub index_count: usize,
    // Bucket-ordered model matrices, sourced from `InstancedCluster::lod_buckets(cam_pos)`.
    // Cached here so the shadow pass's per-instance iteration can read the
    // same data without re-bucketing.
    pub instances: Vec<[[f32; 4]; 4]>,
}

// Build the timestamp-query heap + readback buffer for the per-frame GPU
// time chip. Returns `(None, None, null, 0)` when the queue does not
// support timestamps or any resource allocation fails; `draw_frame` then
// leaves `gpu_frame_us` at zero. The readback buffer is persistently
// mapped (READBACK-heap pointers stay valid for the resource's lifetime,
// matching the auto-exposure readback pattern).
pub(super) fn build_timestamp_resources(
    device: &ID3D12Device,
    command_queue: &ID3D12CommandQueue,
) -> (
    Option<ID3D12QueryHeap>,
    Option<ID3D12Resource>,
    *const u64,
    u64,
) {
    let frequency = unsafe { command_queue.GetTimestampFrequency() }.unwrap_or(0);
    if frequency == 0 {
        return (None, None, std::ptr::null(), 0);
    }
    // Heap holds one block of [whole_frame_start, whole_frame_end, then
    // PASS_COUNT (start, end) pairs] per in-flight frame. See
    // `directx/pass_timing.rs` for the slot layout. Whole-frame stays at
    // the front of each block so legacy `gpu_frame_us` indexing keeps
    // working with only a stride adjustment.
    let heap_desc = D3D12_QUERY_HEAP_DESC {
        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        Count: (super::pass_timing::SLOTS_PER_FRAME * FRAMES) as u32,
        NodeMask: 0,
    };
    let mut heap: Option<ID3D12QueryHeap> = None;
    if let Err(e) = unsafe { device.CreateQueryHeap(&heap_desc, &mut heap) } {
        tracing::warn!("timestamp query heap create failed: {e}");
        return (None, None, std::ptr::null(), 0);
    }
    let readback = match super::texture::create_buffer(
        device,
        super::pass_timing::FRAME_BLOCK_BYTES * FRAMES as u64,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST,
    ) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("timestamp readback buffer create failed: {e}");
            return (None, None, std::ptr::null(), 0);
        }
    };
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    if let Err(e) = unsafe { readback.Map(0, None, Some(&mut ptr)) } {
        tracing::warn!("timestamp readback map failed: {e}");
        return (None, None, std::ptr::null(), 0);
    }
    (heap, Some(readback), ptr as *const u64, frequency)
}

// Per-pass GPU timestamp query state. `query_heap` + `readback` are `None`
// when the command queue does not expose a non-zero timestamp frequency; the
// per-pass chip then reports 0 us. See [`super::pass_timing`] for the slot
// helpers and [`build_timestamp_resources`] for construction.
pub(super) struct TimestampState {
    // Timestamp query heap with `SLOTS_PER_FRAME * FRAMES` slots: one block
    // per in-flight frame, each laid out as a whole-frame start/end pair
    // followed by one (start, end) pair per `PassId`.
    pub query_heap: Option<ID3D12QueryHeap>,
    // Persistently-mapped READBACK buffer paired with `query_heap`, holding
    // `FRAMES` blocks of `SLOTS_PER_FRAME` `u64` ticks each.
    pub readback: Option<ID3D12Resource>,
    pub readback_ptr: *const u64,
    // Ticks per second from `ID3D12CommandQueue::GetTimestampFrequency`; zero
    // when timestamps are unsupported.
    pub frequency: u64,
}

// Bloom mip chain + pipelines. `mips[0]` is half-res; each subsequent mip
// halves again. The prefilter + downsample + upsample passes accumulate a soft
// glow into `mips[0]`, which the composite samples. Skipped entirely when
// `post_process.bloom_intensity` is 0.
pub(super) struct BloomState {
    pub mips: Vec<ID3D12Resource>,
    pub mip_rtvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub mip_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
    pub mip_extents: Vec<(u32, u32)>,
    pub root_sig: ID3D12RootSignature,
    pub pso_prefilter: ID3D12PipelineState,
    pub pso_downsample: ID3D12PipelineState,
    pub pso_upsample: ID3D12PipelineState,
}

// Skinned (skeletally animated) mesh rendering. All `None` / empty until
// `upload_skinned` runs; with no `SkinnedMesh` in the world every skinned pass
// is skipped. The skinned main pass reuses the instanced root signature (its
// root SRV at t3 carries the per-object joint matrices); the shadow pass uses a
// dedicated skinned shadow root signature.
pub(super) struct SkinnedState {
    pub pso: Option<ID3D12PipelineState>,
    pub root_sig: Option<ID3D12RootSignature>,
    pub shadow_pso: Option<ID3D12PipelineState>,
    pub shadow_root_sig: Option<ID3D12RootSignature>,
    // Shared skinned vertex/index buffers. Kept alive for the GPU; referenced
    // through `vertex_buffer_view` / `index_buffer_view`.
    #[allow(dead_code)]
    pub vertex_buffer: Option<ID3D12Resource>,
    #[allow(dead_code)]
    pub index_buffer: Option<ID3D12Resource>,
    pub vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    pub index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
    pub draw_objects: Vec<SkinnedDrawObject>,
    // Per-frame, per-object joint-matrix upload buffers, indexed
    // [frame_idx][skinned_idx]. Each holds MAX_JOINTS float4x4 matrices,
    // persistently mapped; rewritten each frame from `joint_matrices`.
    pub joint_buffers: Vec<Vec<ID3D12Resource>>,
    pub joint_ptrs: Vec<Vec<*mut u8>>,
    // Current skinning matrices per skinned object, parallel to `draw_objects`.
    pub joint_matrices: Vec<Vec<[[f32; 4]; 4]>>,
    // SRV-heap base slot of the skinned (albedo, normal) descriptor block.
    pub srv_base_slot: usize,
    // GPU-driven main-pass skinning. `skin_pipeline` is the `rt_skin` compute
    // kernel reused to deform the bind-pose verts into a per-frame buffer for the
    // bindless main pass (independent of RT, which keeps its own skin dispatch);
    // built in `upload_skinned`. `deformed_buffers` is one UAV-writable buffer per
    // frame-in-flight holding this frame's posed 56-byte `Vertex`s (global skinned
    // indexing, so the draw uses `base_vertex = 0`); rests in
    // VERTEX_AND_CONSTANT_BUFFER, flipped to UNORDERED_ACCESS for the skin
    // dispatch each frame. `deformed_vbvs` is the parallel vertex-buffer view the
    // main pass's 2nd `ExecuteIndirect` binds. All empty / `None` until
    // `upload_skinned` runs.
    pub skin_pipeline: Option<super::raytrace::SkinPipeline>,
    pub deformed_buffers: Vec<ID3D12Resource>,
    pub deformed_vbvs: Vec<D3D12_VERTEX_BUFFER_VIEW>,
    // `false` until the deformed-vertex ring has been posed at least one full
    // frame; `true` once a prior frame's `encode_skin` has filled the slot the
    // next frame reads as its velocity history. While false the GPU-driven
    // G-buffer velocity binds the current deformed buffer as the previous one
    // (prev_pos == cur_pos), so an unposed ring slot never feeds a garbage
    // skinned motion vector on the first frame (or after a runtime ring rebuild).
    // Mirrors the legacy joint buffers' identity seeding + Metal's prev-palette
    // priming. Reset by `upload_skinned`. Atomic, not `Cell`: the G-buffer pass
    // encodes on a `jobs::pool()` rayon worker thread (the parallel per-pass
    // encoder shares `&self` across workers), so any interior mutation reachable
    // from `encode_pass_into` must be atomic, like `draw_calls_accum`.
    pub deformed_primed: std::sync::atomic::AtomicBool,
}

// GPU-driven cull + bindless static main pass. A compute kernel
// frustum/distance-tests the build-time static objects and writes one
// `ExecuteIndirect` command per object; the bindless main pass issues the whole
// buffer with one `ExecuteIndirect`. All `Some`/non-empty only when the world
// uses the built-in bindless shader with build-time geometry; non-bindless
// shaders keep the legacy per-draw CPU loop.
pub(super) struct CullState {
    // Bindless static main pass. `Some` only on the built-in shader; `None`
    // keeps the legacy per-draw main pass. Streamed `VoxelWorld` chunks always
    // keep the legacy pipeline.
    pub main_bindless_root_sig: Option<ID3D12RootSignature>,
    pub main_bindless_pso: Option<ID3D12PipelineState>,
    // Per-frame `StructuredBuffer<GpuObjectData>` upload buffers, one per
    // frame-in-flight, persistently mapped. Rebuilt each frame.
    pub object_buffer_resources: Vec<ID3D12Resource>,
    pub object_buffer_ptrs: Vec<*mut u8>,
    // Per-object SRV region base, bound to bindless root param [5] as the
    // texture pool (pool index `2*i` / `2*i+1` = object `i`'s albedo/normal).
    pub bindless_pool_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Cull compute pipeline; `cull_pso_phase2` is the two-pass-occlusion PSO
    // (same root signature as `cull_pso`).
    pub cull_root_sig: Option<ID3D12RootSignature>,
    pub cull_pso: Option<ID3D12PipelineState>,
    pub cull_pso_phase2: Option<ID3D12PipelineState>,
    pub cull_command_signature: Option<ID3D12CommandSignature>,
    // Per-frame `StructuredBuffer<GpuDrawArgs>` upload buffers (indexed-draw
    // args + per-frame cull-decision bits the kernel reads).
    pub draw_args_buffer_resources: Vec<ID3D12Resource>,
    pub draw_args_buffer_ptrs: Vec<*mut u8>,
    // Per-frame indirect-command buffers the cull kernel writes (UAV) and the
    // main pass consumes (`ExecuteIndirect`). Resting `INDIRECT_ARGUMENT`.
    pub indirect_cmd_buffers: Vec<ID3D12Resource>,
    // Per-frame per-object cull-status buffers (one u32 each): phase-1 writes,
    // phase-2 reads. Resting `UNORDERED_ACCESS`.
    pub cull_status_buffers: Vec<ID3D12Resource>,
    // Per-frame second indirect-command buffers for two-pass occlusion.
    pub indirect_cmd_buffers_2: Vec<ID3D12Resource>,
    // GPU-driven shadow pass. A depth-only bindless pipeline whose VS
    // reads `model` from `GpuObjectData[object_id]` (root SRV) and projects
    // through `light_vps[cascade_idx]`; `shadow_bindless_cmd_sig` is the shared
    // cull command signature rebuilt against its root sig. `shadow_indirect_buffers`
    // is one buffer per frame-in-flight sized `NUM_SHADOW_CASCADES * cull_count()`
    // commands -- cascade `c`'s region is `[c*cull_count, (c+1)*cull_count)`,
    // written by binding the cull output UAV at a per-cascade GPU-address offset.
    // `shadow_cull_status_buffers` is a per-frame scratch the shadow cull writes
    // but never reads (kept separate from `cull_status_buffers`, which the
    // phase-2 main cull consumes AFTER the shadow pass). All `Some`/non-empty
    // only when the bindless cull path is active AND shadows are enabled.
    pub shadow_bindless_root_sig: Option<ID3D12RootSignature>,
    pub shadow_bindless_pso: Option<ID3D12PipelineState>,
    pub shadow_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    // Frustum-only shadow cull kernel (`main_shadow`), shares the cull root sig.
    pub cull_pso_shadow: Option<ID3D12PipelineState>,
    pub shadow_indirect_buffers: Vec<ID3D12Resource>,
    pub shadow_cull_status_buffers: Vec<ID3D12Resource>,
    // GPU-driven G-buffer pre-pass. A 3-MRT bindless pipeline whose VS
    // reads `model` + `roughness` from `GpuObjectData[object_id]` (root SRV) and
    // the previous-frame model from `prev_model_buffers` (a parallel per-frame
    // buffer, one column-major `float4x4` per cull record); `gbuffer_bindless_cmd_sig`
    // is the shared cull command signature rebuilt against its root sig. The pass
    // reuses the main pass's `indirect_cmd_buffers` (camera frustum, no extra cull
    // dispatch). All `Some`/non-empty only when the bindless cull path is active
    // AND the G-buffer is enabled.
    pub gbuffer_bindless_root_sig: Option<ID3D12RootSignature>,
    pub gbuffer_bindless_pso: Option<ID3D12PipelineState>,
    pub gbuffer_bindless_cmd_sig: Option<ID3D12CommandSignature>,
    // Per-frame previous-frame model upload buffers (one column-major `float4x4`
    // per cull record), persistently mapped. The static + skinned regions are
    // rewritten each frame in `build_gbuffer_prev_models`; the instance region is
    // init-written + immutable (camera-only motion).
    pub prev_model_buffers: Vec<ID3D12Resource>,
    pub prev_model_buffer_ptrs: Vec<*mut u8>,
    // `PostProcessConfig.occlusion_two_pass`, as requested by the world.
    pub occlusion_two_pass: bool,
    // Hi-Z (depth-mip pyramid) used by the cull kernel for occlusion culling.
    // Built each frame after `execute_graph`; consumed by the *next* frame's
    // cull dispatch through `prev_view_proj`.
    pub hiz: Option<super::hiz::HiZResources>,
    // Snapshot of the previous frame's un-jittered view-projection matrix the
    // next frame's cull kernel reprojects AABBs through.
    pub prev_view_proj: std::cell::Cell<[[f32; 4]; 4]>,
    // `false` on the first frame and after a resize; while false the cull
    // kernel skips the Hi-Z test.
    pub hiz_valid: std::cell::Cell<bool>,
}

// Shader hot-reload state. `enabled` is true only under `cn debug`: it routes
// every built-in HLSL source resolve through `pipeline::shader_source`'s
// disk-first path and gates the `directx/shaders/` filesystem watcher (false
// under `cn run`, where the `include_str!`-baked HLSL is the only source).
// `reload_pending` is the atomic flag set by the `notify` watcher or the debug
// `reload-shaders` command, polled at the top of `draw_frame` to trigger a PSO
// rebuild; `Some` only when `enabled`, and the debug server reads its `Arc`
// clone via `GraphicsSystem`. `watcher` is the live `notify` handle held purely
// for lifetime (dropping it stops the watcher); `Some` only when `enabled`.
pub(super) struct HotReloadState {
    pub enabled: bool,
    pub reload_pending: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    #[allow(dead_code)]
    pub watcher: Option<crate::directx::hot_reload::WatcherHandle>,
}

// Byte-range sub-allocators for the streamed-mesh regions of the shared
// vertex/index buffers. Empty until mesh streaming is active; seeded at init by
// one of two paths: the shrinkable-seed path hands them the single compacted
// headroom block via `seed_mesh_streaming`, while the full-set path frees each
// streamed draw's build-time region via `evict_mesh`. From then on `upload_mesh`
// / `evict_mesh` allocate and free byte ranges so a streamed mesh lands wherever
// there is room.
pub(super) struct MeshStreamState {
    pub vtx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub idx_alloc: crate::gfx::range_alloc::RangeAllocator,
}

// Byte-range sub-allocators for the headroom region appended to the shared
// vertex/index buffers by `setup_chunk_streaming` for streamed `VoxelWorld`
// chunks, disjoint from the build-time geometry and the mesh-streaming
// allocators. `draw_objects` slots vacated by removed chunks are recycled
// through the shared `DrawSlotAllocator` (`draw_slots`), so the draw list does
// not grow without bound as the camera roams an infinite world. `srv_base_slot`
// is the shared chunk (albedo, normal) pair's SRV-heap base, written by
// `setup_chunk_streaming`. Empty until that runs.
pub(super) struct ChunkStreamState {
    pub vtx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub idx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub srv_base_slot: usize,
}

// Off-screen HDR scene target. The main + instanced passes render linear-light
// HDR into `color`; the composite pass tonemaps it down onto the swapchain
// backbuffer. With MSAA on (`msaa_samples > 1`), `color` is the multisampled
// target and the per-frame loop resolves it into the single-sample `resolve`;
// with MSAA off `color` is single-sample and `resolve` is `None`.
// `resolve_rtv` is `Some` only when MSAA is on (the projected-decal pass renders
// into the resolved scene target; the MSAA-off path uses `color_rtv`).
// `srv_gpu` points at whichever target the composite pass samples.
pub(super) struct HdrState {
    pub color: ID3D12Resource,
    pub color_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub resolve: Option<ID3D12Resource>,
    pub resolve_rtv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub msaa_samples: u32,
}

// SSAO (GTAO). `resources` is `Some` only when `PostProcessConfig.ssao` is set;
// otherwise the pre-pass / kernel / blur are skipped and the main pass samples
// the 1x1 `white` fallback (always present, so the main-pass root signature's AO
// SRV slot always points at a valid descriptor) through `white_srv_gpu` for a
// pass-through ambient term. SSAO always runs its own depth + normal pre-pass on
// DirectX even when SSR is on (no shared-G-buffer shortcut here).
pub(super) struct SsaoState {
    pub resources: Option<SsaoResources>,
    #[allow(dead_code)]
    pub white: ID3D12Resource,
    pub white_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

// GPU-compute particle system. `resources` (compute + render PSOs + per-frame
// uniform rings + spawn-budget upload ring) is built lazily either at init (when
// the world declared >= 1 emitter) or on the first runtime `add_emitter`; it
// stays `None` when no emitter has ever existed. `records` and `emitter_state`
// are parallel `Vec<Option<...>>`s walked in lockstep by the per-frame dispatch,
// skipping `None` pairs; `free_slots` recycles vacated slots. `srv_base_slot` is
// the SRV-heap slot where emitter `i`'s albedo SRV lives (written by
// `add_emitter`). `last_elapsed` is the previous frame's `elapsed` (the diff is
// the frame `dt`); `frame_index` is mixed into the compute kernel's per-thread
// RNG seed. Both are interior-mutable because `record_frame` is `&self` and they
// are only touched on the render thread.
pub(super) struct ParticleState {
    pub resources: Option<ParticleResources>,
    pub records: Vec<Option<crate::gfx::particles::ParticleEmitterRecord>>,
    pub emitter_state: Vec<Option<ParticleEmitterGpuState>>,
    pub free_slots: Vec<usize>,
    pub srv_base_slot: usize,
    pub last_elapsed: std::cell::Cell<f32>,
    pub frame_index: std::cell::Cell<u32>,
}

// Projected decals. `state` (pipeline + unit-cube buffers + per-frame uniform
// rings) is always built so runtime `add_decal` works from a world that started
// empty; the encoder skips the pass when every slot is `None` or every live
// decal culls. `records` and `free_slots` mirror Metal's freelist pattern so id
// reuse stays bounded.
pub(super) struct DecalState {
    pub state: Option<DecalResources>,
    pub records: Vec<Option<crate::gfx::decal::DecalRecord>>,
    pub free_slots: Vec<usize>,
}

// Runtime-clone (albedo, normal) descriptor pool for `clone_static_draw_object`
// (runtime entity spawn + `world.jsonl` hot-reload). `MAX_CLONE_DRAWS` SRV pairs
// are reserved at init starting at `srv_base_slot`; `count` is the high-water
// mark of distinct pool offsets ever handed out (capped by `MAX_CLONE_DRAWS`).
// A clone writes its pair at `srv_base_slot + offset * 2`. `slot_by_draw_idx`
// maps a live clone's `draw_idx` to its pool offset so the legacy draw loop and
// `rewrite_albedo_slot` / `rewrite_normal_slot` can find its descriptors. When a
// clone is retired its offset returns to `free_offsets` for the next clone to
// reuse (the clone re-points the offset's SRV pair before drawing), so steady
// spawn/despawn churn does not exhaust the pool. Empty until the first clone
// fires.
pub(super) struct CloneState {
    pub srv_base_slot: usize,
    pub count: usize,
    pub slot_by_draw_idx: std::collections::HashMap<usize, usize>,
    pub free_offsets: Vec<usize>,
}

// Temporal upscaling (AMD FidelityFX FSR3 / DLSS / XeSS). `backend` is `Some`
// only when the world's `PostProcessConfig.temporal_upscaling` is on AND the
// backend DLL loaded + its context created successfully. The dispatch passes
// `render_size == upscale_size == (render_width, render_height)`, so it runs as
// a temporal-AA replacement rather than an actual upscaler. `requested` is the
// backend the world asked for (FSR3 / DLSS / XeSS / auto), kept so a window
// resize rebuilds the same one. `jitter` is the current frame's sub-pixel
// projection offset (each axis roughly `[-0.5, 0.5]`); `prev_elapsed` is the
// previous frame's elapsed time feeding FSR's `frameTimeDelta`.
pub(super) struct UpscaleState {
    pub backend: Option<Box<dyn super::post::upscale::UpscaleBackend>>,
    pub requested: crate::assets::UpscalerBackend,
    pub jitter: std::cell::Cell<[f32; 2]>,
    pub prev_elapsed: std::cell::Cell<f32>,
}

// Auto-exposure (EV adaptation) state. `resources` is `Some` only when the
// world's `PostProcessConfig` opts in; it holds the histogram + average compute
// PSOs, the histogram UAV, the output UAV, and the per-frame readback buffers.
// `state` carries the EMA target; `settings` carries the clamped tunables;
// `bias_ev` is the authored EV bias added to the target; `last_elapsed` is the
// previous frame's elapsed time used to derive `dt` for the EMA. Mirrors the
// Metal pattern.
pub(super) struct AutoExposureState {
    pub resources: Option<AutoExposureResources>,
    pub settings: Option<crate::gfx::auto_exposure::AutoExposureSettings>,
    pub state: Option<crate::gfx::auto_exposure::AutoExposureState>,
    pub bias_ev: f32,
    pub last_elapsed: f32,
}

// Shadow map resources. `resource` / `dsvs` are `None` / empty when the shadow
// pass is disabled (a 1x1 array fallback SRV is still bound at `srv_gpu`).
// `dsvs` is one DSV per cascade slice. `light_dir` is the world-space unit
// vector pointing toward the first directional light, captured at init from
// `light_uniforms` and used by per-frame CSM updates.
pub(super) struct ShadowState {
    pub resource: Option<GpuResource>,
    pub dsvs: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub map_size: u32,
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub light_dir: [f32; 3],
    // Cascade re-render policy from GraphicsConfig.shadow_update. Hybrid
    // refreshes the near cascade every frame and the far cascades round-robin.
    pub update: crate::assets::ShadowUpdate,
    // Shadow distance in world units (GraphicsConfig.shadow_distance), read by the
    // per-frame cascade-split computation and capped at the camera far plane.
    pub distance: u32,
    // Round-robin clock + primed-set for the cascade schedule; advanced once per
    // frame in record_frame.
    pub scheduler: crate::gfx::shadow_schedule::ShadowCascadeScheduler,
    // Cascades re-rendered this frame (bit `i` = cascade `i`). Set in
    // record_frame and read by encode_shadow_pass so the two agree on which
    // slices to refresh and which to leave intact.
    pub render_mask: u32,
    // Carried CSM uniforms: skipped cascades keep the VP their slice was last
    // rendered with, so the Main pass samples each slice consistently. Splits
    // refresh every frame; per-cascade light VPs only when the mask includes
    // that cascade. Uploaded to the per-frame shadow UBO each frame.
    pub uniforms: ShadowUniforms,
}

// Volumetric fog. All fields `None`/default until the world declares a
// `VolumetricFog`; the fog pass is skipped while `resources` is `None`. The
// settings are cached so the per-frame encoder can build its `FogParams`
// without re-resolving the asset. `sun_dir` / `sun_color` mirror the first
// directional light captured at init (LightUniforms is uploaded once on
// DirectX, so the sun is fixed).
pub(super) struct FogState {
    pub resources: Option<FogResources>,
    pub settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    pub sun_dir: [f32; 3],
    pub sun_color: [f32; 3],
}

// The main-pass constant buffers, grouped off the flat `DxContext`. Mirrors
// Vulkan's `VkUniforms` (which carries view + light; DirectX keeps the shadow
// CBV alongside them, matching how all three are triple-buffered together). The
// view + shadow buffers are per-frame, persistently mapped; the light buffer is
// uploaded once at init. The COM resources auto-release on drop; only the
// persistent mappings need an explicit `unmap`.
pub(super) struct DxUniforms {
    pub view_ubo_resources: Vec<ID3D12Resource>,
    pub view_ubo_ptrs: Vec<*mut u8>,
    pub light_ubo: ID3D12Resource,
    // CPU-side copy of the values in `light_ubo`, kept so a live Ambient-slider
    // change can mutate `ambient_intensity` and re-upload. The light buffer is a
    // single (not per-frame) UPLOAD resource, so `set_ambient_intensity`
    // `wait_idle`s before the rewrite to avoid racing an in-flight read; ambient
    // changes only on a slider drag, so the stall is rare.
    pub light_uniforms: crate::gfx::render_types::LightUniforms,
    pub shadow_ubo_resources: Vec<ID3D12Resource>,
    pub shadow_ubo_ptrs: Vec<*mut u8>,
}

impl DxUniforms {
    // Unmap the persistent view + shadow CBV mappings. Called from
    // `DxContext::drop`; the COM resources themselves auto-release.
    pub(super) fn unmap(&self) {
        for res in self
            .view_ubo_resources
            .iter()
            .chain(self.shadow_ubo_resources.iter())
        {
            unsafe { res.Unmap(0, None) };
        }
    }
}

// Per-frame command infrastructure (allocators + lists), grouped off the flat
// `DxContext`. Mirrors Vulkan's `VkCommands`. The frame is split across two
// outer cmd lists + a per-pass pool so non-composite passes can record in
// parallel:
//
//   * `command_allocators` / `command_lists`: "start" outer cmd list. Holds the
//     timestamp pre-init at the top of every frame (D3D12 debug layer flags an
//     unwritten slot in the `ResolveQueryData` range, so every pass's pair is
//     pre-initialised here). FRAMES-sized.
//
//   * `pass_allocators` / `pass_cmd_lists`: per-pass pool. Sized
//     `FRAMES * PASS_COUNT` so each pass owns its own allocator + cmd list per
//     in-flight slot. Workers reset their own allocator + cmd list before
//     recording, so multiple workers can encode in parallel without contending.
//     Indexed as `frame_idx * PASS_COUNT + (PassId as usize)`. Passes the graph
//     never activates simply leave their slot untouched; it never enters the
//     submission list.
//
//   * `end_command_allocators` / `end_command_lists`: "end" outer cmd list.
//     Holds the composite pass + the final timestamp `EndQuery` +
//     `ResolveQueryData`. Submitted after every per-pass cmd list so the resolve
//     sees every prior pass's `EndQuery` writes. FRAMES-sized.
pub(super) struct DxCommands {
    pub command_allocators: Vec<ID3D12CommandAllocator>,
    pub command_lists: Vec<ID3D12GraphicsCommandList>,
    pub pass_allocators: Vec<ID3D12CommandAllocator>,
    pub pass_cmd_lists: Vec<ID3D12GraphicsCommandList>,
    pub end_command_allocators: Vec<ID3D12CommandAllocator>,
    pub end_command_lists: Vec<ID3D12GraphicsCommandList>,
}

// CPU/GPU frame synchronization, grouped off the flat `DxContext`. Mirrors
// Vulkan's `VkFrameSync` (D3D12 uses one monotonic fence + per-slot signalled
// values instead of per-frame semaphores). The ring cursor (`current_frame`)
// and the per-frame draw-call accumulator stay flat on `DxContext`. All COM /
// plain fields auto-release on drop; only `fence_event` needs an explicit
// `CloseHandle` (done in `DxContext::drop`).
pub(super) struct DxFrameSync {
    pub fence: ID3D12Fence,
    // Per-slot fence value last signalled for that slot's submission. Compared
    // against `fence.GetCompletedValue()` to gate the slot's allocator reset.
    pub fence_values: Vec<u64>,
    // Global monotonic counter feeding every Signal. Must be unique per
    // submission across all slots, otherwise the wait-before-reuse check can
    // be satisfied by another slot's signal of the same value. `Cell` because
    // `wait_idle` advances it through `&self` (trait-imposed signature).
    pub next_fence_value: std::cell::Cell<u64>,
    pub fence_event: windows::Win32::Foundation::HANDLE,
}

// Shared static-mesh geometry buffers + their views, grouped off the flat
// `DxContext`. Mirrors Vulkan's `VkGeometry`. The streamed-mesh + chunk
// sub-allocators stay in their own `MeshStreamState` / `ChunkStreamState`.
pub(super) struct DxGeometry {
    pub vertex_buffer: ID3D12Resource,
    pub index_buffer: ID3D12Resource,
    pub vertex_buffer_view: D3D12_VERTEX_BUFFER_VIEW,
    pub index_buffer_view: D3D12_INDEX_BUFFER_VIEW,
}

// Instanced-prop rendering (drawIndexedInstanced over `InstancedProp` clusters),
// grouped off the flat `DxContext`. Mirrors Vulkan's `VkInstanced`. `root_sig` /
// `pso` are `None` when no cluster was declared; the prefixes
// (`main_instanced_` / `instanced_` / `instance_`) are dropped inside the struct.
pub(super) struct DxInstanced {
    pub root_sig: Option<ID3D12RootSignature>,
    pub pso: Option<ID3D12PipelineState>,
    pub clusters: Vec<InstancedCluster>,
    // Per-frame upload buffers holding the per-instance matrices for each
    // cluster, indexed [frame_idx][cluster_idx]. Each row owns one buffer
    // per cluster, sized to hold its instance count. Persistently mapped.
    pub upload_buffers: Vec<Vec<ID3D12Resource>>,
    pub upload_ptrs: Vec<Vec<*mut u8>>,
    // Per-cluster LOD-bucket layout for the current frame. Filled at the top of
    // `record_frame` by `build_instance_upload`. `RwLock` (not `RefCell`)
    // because the parallel-encoding executor fans the shadow / main / SSAO /
    // SSR / TAA-velocity passes onto rayon workers that all `read()` this slice
    // while encoding; the single writer runs on the main thread before fan-out.
    pub bucket_layouts: std::sync::RwLock<Vec<Vec<InstanceBucketLayout>>>,
}

// Shader-visible descriptor heaps + samplers + the scene texture pools, grouped
// off the flat `DxContext`. DirectX's binding model couples these: the SRV heap
// holds the per-object/per-cluster texture SRVs whose layout the `textures` /
// `normal_map_textures` pools feed, and the sampler heap holds the static
// samplers. No direct Vulkan single-struct equivalent (VK keeps its textures +
// samplers flat), so this is DX progress toward Metal's struct-of-structs. The
// `n_objects` / `n_clusters` heap-layout counts stay flat on `DxContext` (read
// widely; `n_objects` also gates the cull path).
pub(super) struct DxDescriptors {
    // CBV/SRV/UAV descriptor heap (shader-visible).
    // Layout: [0]=shadow_map_array, [1]=irradiance_cube, [2]=prefilter_cube,
    //         [3..3+2N]=per-object (albedo,normal) pairs,
    //         [3+2N..3+2N+2C]=per-cluster pairs, then text atlases, the HDR
    //         scene SRV, the bloom mip SRVs, and the 3D colour-grading LUT SRV.
    pub srv_heap: ID3D12DescriptorHeap,
    pub srv_descriptor_size: usize,
    // Base slot of the flat deduplicated bindless pool: `[albedo SRVs..] ++
    // [normal SRVs..]`. The bindless main pass and the RT hit shader address it
    // by a flat index; the streaming-residency rewrite re-points the one SRV per
    // swapped pool slot here (in addition to the legacy per-object pairs).
    pub flat_pool_base_slot: usize,
    // Base slot of the contiguous MAX_PROBES reflection-probe cube SRV block (the
    // bindless main shader's `probe_cubes` table). Filled with the sky prefilter
    // cube at init; a baked probe overwrites its slot. See [`super::probe`].
    pub probe_cube_base_slot: usize,
    // Sampler heap (shader-visible). Slots:
    //   [0] shadow comparison (s0)   [1] linear repeat (s1)
    //   [2] cube linear-clamp + mip linear (s2)   [3] text linear-clamp
    pub sampler_heap: ID3D12DescriptorHeap,
    pub shadow_sampler_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub linear_sampler_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pub text_sampler_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Scene texture pools (kept alive). Layout matches Metal/Vulkan so that
    // DrawObject::texture_slot/normal_map_slot index directly into these vecs.
    // Slot 0 of `normal_map_textures` is the flat-normal fallback; real maps
    // start at slot 1, matching the convention used by graphics_system.rs when
    // it builds the material map.
    pub textures: Vec<ID3D12Resource>,
    pub normal_map_textures: Vec<ID3D12Resource>,
    // Held only to keep the text-atlas textures resident; the SRV handles
    // below are what the text pass actually binds.
    #[allow(dead_code)]
    pub text_atlas_textures: Vec<GpuResource>,
    pub text_atlas_srv_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
}

pub struct DxContext {
    // Win32 window
    pub(super) win_state: Box<WindowState>,

    // D3D12 core
    pub(super) device: ID3D12Device,
    pub(super) command_queue: ID3D12CommandQueue,

    // Swapchain
    pub(super) swapchain: IDXGISwapChain3,
    pub(super) back_buffers: Vec<ID3D12Resource>,
    pub(super) rtv_heap: ID3D12DescriptorHeap,
    pub(super) rtv_descriptor_size: usize,

    // Off-screen HDR scene target. See [`HdrState`].
    pub(super) hdr: HdrState,

    // Off-screen scene render resolution. The HDR + depth + velocity + SSAO
    // + SSR + Hi-Z + raymarch targets are sized to this. When temporal
    // upscaling is active this is `output_* * upscale_quality.scale()`,
    // strictly smaller than the drawable; FSR reconstructs the drawable
    // resolution into the upscaler's output texture, which bloom +
    // composite then sample. When upscaling is off it equals the output
    // dims and the renderer behaves as a single-resolution pipeline.
    pub(super) render_width: u32,
    pub(super) render_height: u32,
    // Drawable (swapchain) resolution. The back buffers, the bloom mip
    // chain, the upscaler's output texture, and the composite/text pass run
    // at this size. Equals `render_*` whenever temporal upscaling is off.
    pub(super) output_width: u32,
    pub(super) output_height: u32,

    // Temporal upscaling (AMD FidelityFX FSR3). See [`UpscaleState`].
    pub(super) upscale: UpscaleState,

    // Depth buffer
    pub(super) depth_dsv: D3D12_CPU_DESCRIPTOR_HANDLE,
    // Held only to keep the depth buffer and its DSV heap resident; the
    // DSV handles above index into them; the resources themselves are
    // never read back.
    #[allow(dead_code)]
    pub(super) depth_resource: ID3D12Resource,
    #[allow(dead_code)]
    pub(super) dsv_heap: ID3D12DescriptorHeap,

    // Shadow map resources. See [`ShadowState`].
    pub(super) shadow: ShadowState,

    // IBL resources. The fragment shader always samples these; when no
    // EnvironmentMap was supplied, both are 1×1 grey fallback cubes and
    // ViewUniforms::prefilter_mip_count is 0 (the shader takes the legacy
    // ambient/skybox path).
    pub(super) env_map: EnvironmentMapTextures,

    // 3D colour-grading LUT sampled in the composite pass. Holds the declared
    // `ColorLut` payload baked into a Texture3D, or a 2×2×2 identity LUT when
    // the world declares none (the grade is then a no-op at any `lut_strength`).
    // Resolution-independent, so it is never rebuilt.
    pub(super) color_lut: GpuResource,

    // Shader-visible descriptor heaps + samplers + scene texture pools. See
    // `DxDescriptors`.
    pub(super) descriptors: DxDescriptors,
    // SRV-heap layout counts (kept flat; `n_objects` also gates the cull path,
    // matching how Vulkan keeps its `n_objects` out of the grouped sub-structs).
    pub(super) n_objects: usize,
    // Total instanced-cluster instances folded into the GPU-driven bindless pass:
    // each occupies a `GpuObjectData` / `GpuDrawArgs` record at buffer index
    // `n_objects + k`, so the cull dispatch + `ExecuteIndirect` count is
    // `cull_count()`. 0 when the world has no instanced props (or the bindless
    // pass is inactive), leaving the static path's counts unchanged.
    pub(super) n_instances: usize,
    // Streamed-chunk record reserve folded into the GPU-driven bindless pass
    // BETWEEN the instances and the skinned tail: the cull buffers reserve
    // `[n_objects + n_instances, +n_chunk)` at init (capacity = the worst-case
    // resident chunk window). Resident chunks are packed into this region each
    // frame (`build_object_buffer` / `build_draw_args_buffer`) and drawn by the
    // static+instance prefix `ExecuteIndirect` (chunk geometry already lives in the
    // shared VB/IB); the unused tail is disabled. 0 for a non-voxel world. Fixed at
    // init, unlike `n_skinned` (the skinned tail base sits past this reserve).
    pub(super) n_chunk: usize,
    // Skinned draw objects folded into the GPU-driven bindless pass: each occupies
    // a `GpuObjectData` / `GpuDrawArgs` record at buffer index `n_objects +
    // n_instances + k`, drawn (as rigid deformed geometry) by the main pass's 2nd
    // `ExecuteIndirect`. The cull / object / draw-args / indirect buffers reserve
    // these slots at init (capacity threaded through `new`); this count is set in
    // `upload_skinned` once the skinned geometry is resident, so it stays 0 (and
    // `cull_count()` excludes the reserved tail) when no skinned mesh loads.
    pub(super) n_skinned: usize,
    #[allow(dead_code)]
    pub(super) n_clusters: usize,

    // Shared static-mesh geometry buffers + views. See `DxGeometry`.
    pub(super) geometry: DxGeometry,

    // Streamed-mesh byte-range sub-allocators. See [`MeshStreamState`].
    pub(super) mesh_stream: MeshStreamState,

    // Chunk-streaming byte-range sub-allocators + slot recycling. See
    // [`ChunkStreamState`].
    pub(super) chunk_stream: ChunkStreamState,

    // Skinned (skeletally animated) mesh rendering. All `None` / empty until
    // `upload_skinned` runs.
    pub(super) skinned: SkinnedState,
    // Free pool for the pre-reserved skinned instance slots a runtime skinned
    // spawn claims. Seeded once from `seed_skinned_instance_pool` with the hidden
    // bind-pose copies `upload_skinned` uploaded; empty for a world with no
    // skinned mesh opting into runtime spawning.
    pub(super) skinned_pool: crate::gfx::skinned_pool::SkinnedInstancePool,

    // Constant buffers (view + shadow per-frame persistent-mapped, light once).
    // See `DxUniforms`.
    pub(super) uniforms: DxUniforms,

    // Root signatures + PSOs
    pub(super) main_root_sig: ID3D12RootSignature,
    pub(super) main_pso: ID3D12PipelineState,
    // GPU-driven cull + bindless static main pass. All `Some`/non-empty only
    // on the bindless path with build-time geometry. See [`CullState`].
    pub(super) cull: CullState,
    pub(super) shadow_root_sig: Option<ID3D12RootSignature>,
    pub(super) shadow_pso: Option<ID3D12PipelineState>,
    pub(super) text_root_sig: ID3D12RootSignature,
    pub(super) text_pso: Option<ID3D12PipelineState>,
    // Composite (post-process) pass: fullscreen-triangle tonemap of the HDR
    // scene target onto the swapchain backbuffer.
    pub(super) composite_root_sig: ID3D12RootSignature,
    pub(super) composite_pso: ID3D12PipelineState,

    // Bloom mip chain + pipelines. The mips are shared across frame slots; the
    // command queue runs frames serially, so a frame's bloom writes never race
    // a prior frame's composite read.
    pub(super) bloom: BloomState,
    // Post-process tunables (bloom / exposure / vignette). Drives whether the
    // bloom chain runs and feeds the bloom-prefilter + composite root constants.
    pub(super) post_process: crate::gfx::render_types::PostProcessParams,

    // Unified geometry G-buffer pre-pass. `Some` whenever any screen-space
    // consumer (SSR, SSGI, SSAO, TAA, or temporal upscaling) is enabled: one
    // jittered traversal writes view normal+depth, roughness, and motion into
    // one MRT that all those consumers read, replacing the separate SSR / SSAO
    // / velocity geometry pre-passes. See [`GbufferResources`].
    pub(super) gbuffer: Option<GbufferResources>,

    // Temporal anti-aliasing. `Some` only when `PostProcessConfig.taa` is set;
    // when `None` the history resolve and the projection jitter are skipped and
    // the composite samples the HDR scene target directly.
    pub(super) taa: Option<TaaResources>,

    // SSAO (GTAO). See [`SsaoState`].
    pub(super) ssao: SsaoState,

    // Backing store for the render graph's transient render targets (the
    // resources the aliasing planner manages). Owns each managed transient as a
    // placed resource on an `ID3D12Heap`; features read them back by label and
    // the executor's barrier registry resolves them the same way. Rebuilt on
    // swapchain resize. Today it manages `ao_output`.
    pub(super) transient_pool: super::transient_pool::TransientResourcePool,

    // SSR. `Some` when `PostProcessConfig.ssr` is set, or when SSGI is on
    // (SSGI reuses the depth + normal pre-pass G-buffer). The resolve half
    // (`ssr.resolve`) is `Some` only when SSR itself is authored on; with it
    // off the TAA / bloom / composite passes sample `hdr_srv_gpu` directly as
    // the scene colour. When the resolve is on it writes into
    // `ssr.resolve.output`, whose SRV becomes the scene the post stack consumes
    // (see `scene_srv_for_post`).
    pub(super) ssr: Option<SsrResources>,

    // SSGI. `Some` only when `PostProcessConfig.indirect_lighting` is `ssgi`.
    // A hemisphere-gather + depth-aware-blur composite that bleeds nearby lit
    // surfaces' colour onto one another, additively on top of the IBL ambient.
    // Reuses the SSR pre-pass G-buffer (so `ssr` is also `Some` whenever this
    // is); the render-graph `PassId::Ssgi` node is gated on `ssgi.is_some()`.
    pub(super) ssgi: Option<super::post::ssgi::SsgiResources>,

    // Roughness-aware reflection composite: the SSR / RT resolve writes reflected
    // radiance + weight, then this blurs it by surface roughness (a reduced-res blur
    // pass) and composites it over the scene into its own output -- the scene the
    // post stack consumes via `scene_srv_for_post`. `Some` when SSR resolve or RT is
    // authored (both feed it).
    pub(super) reflection_composite:
        Option<super::post::reflection_composite::ReflectionCompositeResources>,

    // Hardware ray-traced reflections (DXR). `rt_reflections` (output target +
    // RtParams UBO + root sig + flat/textured PSOs) and `rt_accel` (BLAS/TLAS +
    // geometry table) are both `Some` only when the world enables
    // `ray_traced_reflections`, the GPU supports the DXR tier, and the DXC
    // compile + acceleration-structure build succeeded; otherwise the graph
    // falls back to `SsrResolve`. RT occupies the `SsrResolve` slot, reuses the
    // SSR pre-pass G-buffer (forced on), and its output becomes the scene the
    // post stack consumes via `scene_srv_for_post`. `FrameGraphInputs::
    // rt_reflections_enabled` is gated on both being `Some`.
    pub(super) rt_reflections: Option<super::post::rt_reflections::RtReflectionsResources>,
    pub(super) rt_accel: Option<super::raytrace::RtAccelData>,
    // How the acceleration structure is kept current as props move (read once
    // from `CN_RT_DYNAMIC` at init; `Auto` by default).
    pub(super) rt_dynamic_mode: super::raytrace::RtDynamicMode,

    // Projected decals. See [`DecalState`].
    pub(super) decal: DecalState,

    // Raymarched SDF volumes. `Some` when at least one `SdfVolume` whose
    // `fragment_shader` resolves to `.hlsl` survived the init filter, i.e.
    // the world declares SDF volumes authored for DirectX. Metal-first
    // (`.metal`) volumes degrade with a logged warning at init and do not
    // contribute to this field. Render-graph `PassId::Raymarch` is gated
    // on `Self::raymarch_enabled()` so worlds with no DX-targeted SDF skip
    // the slot entirely.
    pub(super) raymarch: Option<super::raymarch::RaymarchResources>,

    // Translucent glass panels. `Some` only when the world declared any
    // `GlassPanel`; with none the field stays `None` and the transparent pass
    // is skipped. The generic producer for the shared `PassId::Transparent`
    // slot; render-graph inclusion is gated on `Self::transparent_enabled()`.
    // Water is a separate (Metal-only) producer not ported here. Mirrors
    // src/metal glass handling.
    pub(super) glass: Option<super::glass::GlassResources>,

    // Planar reflections for flat glass panes: a per-frame mirror render of the
    // scene reflected across each distinct pane plane, sampled projectively by the
    // glass shader (sharper + scene-correct vs the box-projected probe cube).
    // `Some` only when the world has glass panes assigned to a planar slot. Driven
    // inline at the head of the transparent pass (`encode_planar_reflections`).
    pub(super) planar_reflection: Option<super::planar::PlanarReflectionSet>,

    // Volumetric fog. See [`FogState`].
    pub(super) fog: FogState,

    // GPU-compute particle system. See [`ParticleState`].
    pub(super) particle: ParticleState,

    // Per-frame command allocators + lists (start / per-pass / end). See
    // `DxCommands`.
    pub(super) commands: DxCommands,
    // CPU-side accumulator for this frame's draw calls. The Metal +
    // Vulkan executors use the same pattern: encoders (which may run in
    // parallel) bump this atomic via `inc_draw_calls`; the main thread
    // drains it into `frame_stats.draw_calls` at the end of every frame.
    // `AtomicU32` because `Cell<RenderStats>` is single-threaded and
    // would race when workers encode in parallel.
    pub(super) draw_calls_accum: std::sync::atomic::AtomicU32,
    // CPU/GPU frame synchronization (one monotonic fence + per-slot values).
    // See `DxFrameSync`.
    pub(super) frame_sync: DxFrameSync,
    pub(super) current_frame: usize,

    // Draw state
    pub(super) draw_objects: Vec<DrawObject>,
    pub(super) cull_bvh: crate::gfx::bvh::Bvh,
    pub(super) always_draw: Vec<u32>,
    // Parallel to `draw_objects`: true where that slot is a member of
    // `always_draw`, so `ensure_always_draw` adds a recycled slot at most once.
    // A slot vacated by a culled static prop is not yet in `always_draw`; one
    // recycled from a chunk / clone already is.
    pub(super) always_draw_member: Vec<bool>,
    // Free-list allocator over `draw_objects` slots. `retire_draw_object` /
    // `remove_chunk_mesh` push a vacated slot; `clone_static_draw_object` /
    // `add_chunk_mesh` pop one before growing the vec, so runtime spawn/despawn
    // and chunk streaming reuse slots instead of leaking them. Indices stay
    // stable (RenderHandle stores raw indices into draw_objects), so this is a
    // free-list, never a compaction.
    pub(super) draw_slots: crate::gfx::draw_slot::DrawSlotAllocator,
    // Per-frame scratch for the legacy CPU draw path's visible set
    // (BVH-culled cullables + always_draw fallback). Wrapped in a RefCell
    // because `record_frame` is &self (matches the existing per-frame
    // interior-mutability pattern). Swapped out via `replace(Vec::new())` at the top
    // of record_frame and put back at the bottom so the heap allocation is
    // reused across frames instead of `Vec::with_capacity`'d each tick.
    pub(super) visible_scratch: RefCell<Vec<u32>>,
    // Instanced-prop pipeline + per-frame upload buffers + LOD buckets. See
    // `DxInstanced`.
    pub(super) instanced: DxInstanced,
    pub(super) clear_color: [f32; 4],
    pub(super) view_matrix: [[f32; 4]; 4],

    // Per-frame-slot persistent upload buffers for transient HUD text geometry.
    // Each slot's cursor resets and its buffer (re)maps inside the ring's
    // `reserve`, which the composite pass calls once the frame fence confirms
    // the GPU is done with slot i. See [`TextUploadRing`].
    pub(super) text_upload: super::draw::TextUploadRing,

    // D3D12 validation message sink (Some only when validation=true).
    pub(super) info_queue: Option<ID3D12InfoQueue>,

    // IDXGIAdapter3 captured at init for `QueryVideoMemoryInfo`, the VRAM
    // chip's source. `None` when the adapter does not expose the v3
    // interface; the chip then reads `0 MB` and the rest of the overlay
    // still works. See `init/window.rs`.
    pub(super) adapter: Option<IDXGIAdapter3>,

    // Auto-exposure (EV adaptation) state. See [`AutoExposureState`].
    pub(super) auto_exposure: AutoExposureState,

    // Reported maximum extended-range colour-component multiplier captured
    // from the resolved [`HdrOutputMode`] at init. `Some` only when the
    // renderer is on the HDR path (the swapchain was created in
    // `RGBA16Float` + scRGB-linear colour space). Surfaced through
    // `RenderStats.max_edr` so the `StatHud` overlay can render an `EDR
    // ×X.X` chip. Mirrors `MtlContext.max_edr`.
    pub(super) max_edr: Option<f32>,

    // Render statistics for the most recent frame: draw-call and object
    // counts (filled by `draw_frame`) plus VRAM bytes pulled from the
    // adapter. Lives in a `Cell` because the per-pass increments happen
    // through the `&self` `record_frame` path. Surfaced to the profiler
    // overlay via [`Self::render_stats`]; mirrors `MtlContext::frame_stats`.
    pub(super) frame_stats: std::cell::Cell<crate::gfx::profile::RenderStats>,

    // Runtime-clone (albedo, normal) descriptor pool. See [`CloneState`].
    pub(super) clone: CloneState,

    // Per-pass GPU timestamp queries. Read at the top of `draw_frame` after
    // the matching fence wait so the CPU sees a fully committed block.
    pub(super) timestamps: TimestampState,

    // Swapchain RTV format captured at init. Stored so the composite + text
    // PSO rebuilds during shader hot-reload can target the same format the
    // PSOs were originally created against.
    pub(super) swap_format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,

    // Presentation pacing. `present_sync_interval` is 1 (vsync, lock to refresh)
    // or 0 (uncapped). `allow_tearing` is true when the swapchain was created
    // with DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING (vsync off + tearing supported);
    // it selects the tearing present flag and must be mirrored in ResizeBuffers.
    pub(super) present_sync_interval: u32,
    pub(super) allow_tearing: bool,

    // Resolved HDR encoding (scRGB-linear vs PQ) captured from the init-time
    // `HdrOutputMode`. `None` on the SDR path. Only the headless `screenshot`
    // path reads it, to decode the float swapchain for display. Mirrors the
    // `encoding` the Vulkan screenshot path pulls from `VkContext::hdr_mode`.
    pub(super) hdr_encoding: Option<crate::gfx::hdr_output::HdrEncoding>,

    // Back-buffer index of the most recently presented frame. `None` until the
    // first `draw_frame` presents. The headless `screenshot` path copies this
    // buffer (the one currently on screen); `GetCurrentBackBufferIndex` after a
    // present already points at the next buffer to render into, so the captured
    // index must be recorded at present time. Mirrors `VkContext::
    // last_present_index`.
    pub(super) last_present_index: Option<usize>,

    // Shader hot-reload state. See [`HotReloadState`].
    pub(super) hot_reload: HotReloadState,

    // Fixed descriptor-heap slots for the live-toggleable Quality effects (TAA /
    // SSAO / SSR / SSGI / RT-reflection output). Minted once at init (the slots
    // are reserved unconditionally) and stashed so `apply_quality_settings` can
    // build a launched-off feature into its slot without re-deriving the heap
    // layout.
    pub(super) quality_slots: super::quality::QualitySlotHandles,

    // Whether the GPU reports the DXR 1.1 tier (queried once at init). Gates the
    // live RT-reflections toggle: an enable on a non-DXR GPU no-ops with a
    // warning, mirroring the init fallback to SSR.
    pub(super) rt_capable: bool,
    // Total static vertices uploaded at init (the shared VB element count); the
    // acceleration-structure build needs it to size the hit-shader vertex SSBO.
    // There is no separate count field, so it is captured here for a live RT
    // build. Static-geometry rebuilds are not reflected (a pre-existing RT
    // topology limitation).
    pub(super) rt_static_vertex_count: usize,

    // Reflection-probe placements (declared `ReflectionProbe` assets or an
    // auto-seeded grid). Indexed in order by the staggered capture pass; one cube
    // is baked per placement. See [`super::probe`].
    pub(super) probe_placements: Vec<crate::gfx::reflection_probe::ProbePlacement>,
    // Staggered bake cursor over `probe_placements`: a not-yet-baked probe falls
    // back to the sky until its turn, so no single frame pays the whole capture.
    pub(super) probe_bake_queue: crate::gfx::reflection_probe::ProbeBakeQueue,
    // Per-frame probe set (parallax boxes + live count) bound to the forward / SSR
    // / RT shaders. `EMPTY` until a bake installs a cube; distinct from `env_map`
    // so the skybox + diffuse irradiance keep the sky.
    #[allow(dead_code)] // bound to the forward shader (next slice)
    pub(super) probe_set: super::probe_uniforms::ProbeSet,
    // The probe whose six cube faces are currently rendering on the GPU (one at a
    // time, spread one face per frame). Owns the reserved-ring-slot capture
    // resources until its faces are read back. See [`super::probe`].
    pub(super) probe_rendering: Option<super::probe::RenderingBake>,
    // The prior probe whose read-back faces are convolving on a worker thread.
    // Holds only the worker's payload slot (plain data), so it drops freely.
    pub(super) probe_converting: Option<super::probe::ConvertingBake>,
    // One baked prefilter cube per installed probe, aligned with `probe_set`
    // (index `i` is placement `i`). Distinct from `env_map`; sampled only by the
    // specular reflection term.
    pub(super) probe_maps: Vec<super::probe::ProbeCube>,
    // Per-frame CBVs holding `probe_set` (the parallax boxes + live count) bound at
    // root param [11] by the main pass. A `FRAMES` ring so a frame writes its own
    // slot without racing a prior frame's in-flight GPU read.
    pub(super) probe_set_cbvs: Vec<ID3D12Resource>,
    pub(super) probe_set_cbv_ptrs: Vec<*mut u8>,
    // A static count-0 ProbeSet CBV the asynchronous capture binds at [11], so a
    // probe face render samples the sky (no probe feedback) and never reads the live
    // ring while `record_frame` rewrites it.
    pub(super) probe_set_empty_cbv: ID3D12Resource,
}

// uniforms.view_ubo_ptrs are host-mapped and only touched on the render thread.
// text_upload's RefCells + map pointers are single-threaded.
unsafe impl Send for DxContext {}

// Win32 thread id of the thread that built the context. `DxContext::new` runs
// on the main thread and records it here; `debug_assert_main_thread` checks
// every mutation entry point against it.
static MAIN_THREAD_ID: OnceLock<u32> = OnceLock::new();

// Record the calling thread as the main (render) thread. Called once from
// `DxContext::new`, which always runs on the main thread.
pub(super) fn record_main_thread() {
    let _ = MAIN_THREAD_ID.set(unsafe { GetCurrentThreadId() });
}

// Debug-only guard that the caller is on the main thread.
//
// The `unsafe impl Send for DxContext` above is sound only because the context
// is touched from the main thread alone: the Win32 window message pump and
// D3D12 command-queue submission are both thread-affine, and the parallel
// encoder fan-out only ever shares `&self` read-only. The `RenderBackend`
// mutation entry points (reached through the boxed trait object) had nothing
// proving this, so scheduling `GraphicsSystem` off the main thread would
// silently race the window/queue instead of failing. This makes that mistake
// panic loudly in debug builds and compiles to nothing in release. `entry` is
// the offending method name, for the message. Mirrors
// `metal/context.rs::debug_assert_main_thread`.
#[inline]
#[track_caller]
pub(super) fn debug_assert_main_thread(entry: &str) {
    debug_assert!(
        MAIN_THREAD_ID
            .get()
            .is_none_or(|&main| unsafe { GetCurrentThreadId() } == main),
        "{entry} must be called from the main thread: DxContext is main-thread-only \
         (see `unsafe impl Send for DxContext`); driving GraphicsSystem off the main \
         thread races the Win32 window + D3D12 command submission",
    );
}

impl DxContext {
    pub fn draw_frame(
        &mut self,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
    ) -> Result<(), String> {
        // Shader hot-reload: if either the filesystem watcher or the debug
        // `reload-shaders` command set the flag, rebuild every built-in PSO
        // from disk-resident source before the frame's passes start using
        // them. The flag is cleared regardless of outcome so a failed rebuild
        // (typo in a shader edit) doesn't loop, and the previous pipelines
        // stay live so the session keeps rendering. Wait for the GPU to
        // drain first so swapping PSOs out from under in-flight command
        // lists is safe.
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
            self.wait_idle();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }

        // Window resize: rebuild the swapchain back-buffers, the HDR / depth
        // scene targets, the bloom mip chain, and the TAA / SSAO / SSR
        // resource sets at the new size. A no-op when the size hasn't
        // changed; skips the rebuild (and the frame) when the window is
        // minimised so we never present 0×0. Failures are logged but not
        // fatal; the client keeps trying on subsequent frames.
        if let Err(e) = self.maybe_handle_resize() {
            tracing::error!("D3D12 resize failed: {e}");
        }

        let frame = self.current_frame;

        // Wait for this frame slot's previous work to finish before reusing it.
        let completed = unsafe { self.frame_sync.fence.GetCompletedValue() };
        if self.frame_sync.fence_values[frame] > completed {
            unsafe {
                self.frame_sync.fence.SetEventOnCompletion(
                    self.frame_sync.fence_values[frame],
                    self.frame_sync.fence_event,
                )
            }
            .map_err(|e| format!("SetEventOnCompletion: {e}"))?;
            unsafe { WaitForSingleObject(self.frame_sync.fence_event, u32::MAX) };
        }

        // Advance the staggered reflection-probe bake. Called after the frame-slot
        // fence wait (so any in-flight capture resources are safe to recycle) and
        // before the frame's passes record. Non-fatal: a failure is logged and the
        // frame proceeds with whatever probes have baked.
        if let Err(e) = self.bake_pending_probes(elapsed, near, far) {
            tracing::warn!("reflection probe bake step failed: {e}");
        }

        // Auto-exposure EMA step. The fence wait above ensured the GPU work
        // that wrote this slot's readback buffer has completed, so the read
        // is race-free. Must happen *before* the bloom prefilter / composite
        // consume `self.post_process.exposure`. No-op when auto-exposure is
        // disabled.
        self.update_auto_exposure(elapsed, frame);

        // Reset this frame's render stats. `record_frame` accumulates
        // `draw_calls` through `inc_draw_calls` (an interior-mutability path
        // because the encoders run through `&self`); `objects` and
        // `vram_bytes` are filled here from `&mut self` state. Mirrors the
        // Metal `frame_stats` reset at the top of `draw_frame`.
        let instanced_total: usize = self
            .instanced
            .clusters
            .iter()
            .map(|c| c.instances.len())
            .sum();
        let objects =
            (self.draw_objects.len() + instanced_total + self.skinned.draw_objects.len()) as u32;
        // Live skinned count: authored meshes plus runtime-spawned instances,
        // excluding the hidden pre-reserved pool slots. `objects` above counts the
        // whole pool and so stays flat across skinned spawn/despawn; this tracks
        // the visible count, so a spawn bumps it and a despawn drops it.
        let skinned_visible = self
            .skinned
            .draw_objects
            .iter()
            .filter(|o| o.visible)
            .count() as u32;
        let skinned_pool_free = self.skinned_pool.total_free() as u32;
        // Current GPU memory residency, in bytes. `Local` is the dedicated VRAM
        // budget on a discrete GPU and the local-process system-memory budget on
        // an integrated GPU; either way, `CurrentUsage` is what the HUD's
        // "VRAM N MB" chip reports. Zero when the adapter does not expose the
        // v3 interface (pre-WDDM 2.0).
        let vram_bytes = self
            .adapter
            .as_ref()
            .and_then(|a| {
                let mut info =
                    windows::Win32::Graphics::Dxgi::DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                unsafe {
                    a.QueryVideoMemoryInfo(
                        0,
                        windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                        &mut info,
                    )
                }
                .ok()
                .map(|_| info.CurrentUsage)
            })
            .unwrap_or(0);
        // Pull the most recently completed GPU time for this slot. The fence
        // wait at the top of this `draw_frame` already ensured the GPU work
        // that wrote this slot's readback bytes has retired, so the
        // persistently-mapped pointer reflects a fully committed pair (`FRAMES`
        // frames stale by construction: the same slot's writes from the
        // previous trip through the ring). Zero before the slot has been
        // visited a second time (the readback buffer starts zero-initialised).
        //
        // The frame's block in the readback buffer is laid out as
        // [whole_frame_start, whole_frame_end, then PASS_COUNT (start, end)
        // pairs]; see directx/pass_timing.rs. The whole-frame pair sits
        // at the front, exactly where the legacy layout placed it.
        let timestamps_live =
            !self.timestamps.readback_ptr.is_null() && self.timestamps.frequency > 0;
        let ticks_to_micros = |ticks: u64| -> u32 {
            (ticks.saturating_mul(1_000_000) / self.timestamps.frequency).min(u32::MAX as u64)
                as u32
        };
        let block_base = if timestamps_live {
            // SAFETY: `timestamp_readback_ptr` is the persistently-mapped
            // base of a READBACK buffer sized for FRAMES blocks of
            // SLOTS_PER_FRAME u64s each (see build_timestamp_resources).
            // The fence wait above ensures this block's writes have
            // retired.
            unsafe {
                self.timestamps
                    .readback_ptr
                    .add(frame * super::pass_timing::SLOTS_PER_FRAME)
            }
        } else {
            std::ptr::null()
        };
        let gpu_frame_us = if timestamps_live {
            unsafe {
                let ts_start = block_base.read();
                let ts_end = block_base.add(1).read();
                if ts_end > ts_start {
                    ticks_to_micros(ts_end - ts_start)
                } else {
                    0
                }
            }
        } else {
            0
        };
        let mut pass_times_us = [("", 0u32); crate::gfx::profile::MAX_PASS_TIMINGS];
        if timestamps_live {
            // Walk the PASS_COUNT pairs that follow the whole-frame pair
            // and surface (pass-name, micros) tuples for the StatHud chip.
            // Inactive passes (not seeded into the graph this frame) keep
            // the frame-start timestamp in both their start and end slots
            // (see the pre-init loop in `record_frame`) so `ts_end > ts_start`
            // evaluates false and they report 0 µs; the shared
            // `passes_text` filter naturally hides them from the chip.
            for (i, name) in crate::gfx::render_graph::PASS_NAMES.iter().enumerate() {
                if i >= crate::gfx::profile::MAX_PASS_TIMINGS {
                    break;
                }
                // Slot 2 + 2*i = start, slot 3 + 2*i = end (skip the
                // whole-frame pair at the front of the block).
                let off = 2 + 2 * i;
                let ts_start = unsafe { block_base.add(off).read() };
                let ts_end = unsafe { block_base.add(off + 1).read() };
                let micros = if ts_end > ts_start {
                    ticks_to_micros(ts_end - ts_start)
                } else {
                    0
                };
                pass_times_us[i] = (*name, micros);
            }
        }

        // Reset the parallel-encoder draw-call accumulator so this frame's
        // encoders bump from zero. Drained back into `frame_stats.draw_calls`
        // after `record_frame` returns (the actual encoding fan-out happens
        // inside it).
        self.draw_calls_accum
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(crate::gfx::profile::RenderStats {
            draw_calls: 0,
            objects,
            skinned_visible,
            skinned_pool_free,
            gpu_frame_us,
            vram_bytes,
            pass_times_us,
            // EMA-adapted exposure value, surfaced to the StatHud `EV ±X.XX`
            // chip. `None` when the world stayed on the authored static
            // exposure; the chip blanks itself in that case. Mirrors
            // `MtlContext::render_stats`.
            auto_exposure_ev: self.auto_exposure.state.as_ref().map(|s| s.current_ev),
            // Captured from the resolved `HdrOutputMode` at init. `None` on
            // the SDR path (chip blanks). Mirrors `MtlContext::render_stats`.
            max_edr: self.max_edr,
        });

        // Flush any D3D12 validation messages from the previous frame.
        self.flush_validation();

        // Frame command-list pipeline (parallel encoding):
        //
        //   start_cmd  : pre-init timestamps only (closed up-front)
        //   per-pass   : one cmd list per non-composite pass, recorded
        //                in parallel by rayon workers inside execute_graph
        //   end_cmd    : Composite + final timestamp + ResolveQueryData
        //                 + per-frame restore barriers (composed by
        //                 execute_graph's main-thread composite arm + the
        //                 tail of record_frame)
        //
        // ExecuteCommandLists is called once with the whole topological
        // sequence so the GPU sees them in submission order.

        // 1. Reset + record the START cmd list (timestamp pre-init only),
        //    then close it immediately. The fence wait at the top of the
        //    function gated this slot's previous submission, so it's
        //    safe to reset.
        unsafe { self.commands.command_allocators[frame].Reset() }
            .map_err(|e| format!("start allocator reset: {e}"))?;
        // Owned clone (COM refcount bump) so the per-frame RT acceleration-
        // structure update below can take `&mut self` without holding a borrow
        // of `self.commands.command_lists`.
        let start_cmd = self.commands.command_lists[frame].clone();
        unsafe { start_cmd.Reset(&self.commands.command_allocators[frame], None) }
            .map_err(|e| format!("start cmd reset: {e}"))?;

        // Timestamp the start of this frame's GPU work + pre-initialise
        // every per-pass slot in this frame's block. The end-of-frame
        // `ResolveQueryData` covers the whole block, and the D3D12 debug
        // layer flags any slot in the resolved range that never had
        // `EndQuery` called on it; without the pre-init the graph
        // would spam those errors for every feature the world opted out
        // of. The executor's per-pass start/end calls overwrite the
        // slots of active passes with real timestamps; inactive slots
        // keep this frame-start timestamp for both start and end, so
        // `ts_end > ts_start` evaluates false on readback and they
        // cleanly report 0 µs.
        if let Some(heap) = self.timestamps.query_heap.as_ref() {
            let (start_slot, _) = super::pass_timing::whole_frame_pair(frame);
            let block_base = (frame * super::pass_timing::SLOTS_PER_FRAME) as u32;
            unsafe {
                start_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot);
                // Pre-init each per-pass (start, end) pair as **end then
                // start** so inactive passes wind up with `ts_start >
                // ts_end` and the readback's `ts_end > ts_start` check
                // returns false (clean 0 µs reading).
                let pass_count = super::pass_timing::SLOTS_PER_FRAME / 2 - 1;
                for pass_idx in 0..pass_count as u32 {
                    let pair_start = block_base + 2 + 2 * pass_idx;
                    let pair_end = pair_start + 1;
                    start_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, pair_end);
                    start_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, pair_start);
                }
            }
        }

        // Per-frame hardware-RT acceleration-structure update: when a
        // participating prop moved, rebuild the TLAS + geometry table onto the
        // start cmd list (submitted before every per-pass trace on the serial
        // DIRECT queue, so the rebuild is ordered before this frame's reflection
        // trace reads it). A no-op when RT reflections are off or the BVH is
        // static this frame.
        self.rt_dynamic_update(&start_cmd, frame);

        unsafe { start_cmd.Close() }.map_err(|e| format!("start cmd close: {e}"))?;

        // 2. Open the END cmd list (Composite + final timestamp +
        //    ResolveQueryData + per-frame restore barriers). The
        //    executor's main-thread Composite arm encodes onto this
        //    cmd list; this function appends the final timestamp +
        //    resolve after `record_frame` returns.
        unsafe { self.commands.end_command_allocators[frame].Reset() }
            .map_err(|e| format!("end allocator reset: {e}"))?;
        let end_cmd = &self.commands.end_command_lists[frame];
        unsafe { end_cmd.Reset(&self.commands.end_command_allocators[frame], None) }
            .map_err(|e| format!("end cmd reset: {e}"))?;

        let back_idx = unsafe { self.swapchain.GetCurrentBackBufferIndex() } as usize;
        let back_buffer = self.back_buffers[back_idx].clone();
        let rtv_base = unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() };
        let back_buffer_rtv = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: rtv_base.ptr + back_idx * self.rtv_descriptor_size,
        };

        // Cascaded-shadow update policy. Advance the round-robin schedule, then
        // refresh only this frame's cascades' light VPs (splits always refresh).
        // Skipped cascades keep the VP + depth their slice was last rendered
        // with, so the Main pass samples each cascade consistently. record_frame
        // uploads the merged `self.shadow.uniforms` to this frame's shadow UBO,
        // and encode_shadow_pass re-rasterizes only the masked slices. Mirrors
        // Metal; no-op (mask stays 0, uniforms stay empty) when shadows are off.
        if !self.shadow.dsvs.is_empty() {
            let aspect = self.render_width.max(1) as f32 / self.render_height.max(1) as f32;
            let fresh = crate::gfx::csm::compute_shadow_uniforms(
                self.view_matrix,
                cam_pos,
                fov_y_radians,
                aspect,
                near,
                (self.shadow.distance as f32).min(far),
                self.shadow.light_dir,
                self.shadow.map_size,
            );
            let update = self.shadow.update;
            let mask = self.shadow.scheduler.next_mask(update);
            self.shadow.render_mask = mask;
            self.shadow.uniforms.cascade_splits = fresh.cascade_splits;
            for i in 0..crate::gfx::render_types::NUM_SHADOW_CASCADES {
                if mask & (1u32 << i) != 0 {
                    self.shadow.uniforms.light_vps[i] = fresh.light_vps[i];
                }
            }
        }

        // 3. record_frame fans non-composite passes onto rayon workers
        //    (each records into its own cmd list from the per-pass pool)
        //    and dispatches Composite + per-frame restore barriers onto
        //    `end_cmd`. Returns the per-pass cmd lists in topological
        //    pass order.
        let pass_cmd_lists = self.record_frame(
            end_cmd,
            &back_buffer,
            back_buffer_rtv,
            elapsed,
            fov_y_radians,
            near,
            far,
            cam_pos,
            text_calls,
            frame,
            self.render_width.max(1),
            self.render_height.max(1),
            self.output_width.max(1),
            self.output_height.max(1),
        )?;

        // Drain the parallel-encoder draw-call accumulator into this
        // frame's `frame_stats.draw_calls`. The accumulator was reset
        // to 0 above and bumped by each `inc_draw_calls` call site
        // (potentially from worker threads) during `record_frame`.
        let mut s = self.frame_stats.get();
        s.draw_calls = self
            .draw_calls_accum
            .load(std::sync::atomic::Ordering::Relaxed);
        self.frame_stats.set(s);

        // 4. Timestamp the end of GPU work and resolve this frame's
        //    entire block (whole-frame pair + every per-pass pair the
        //    workers wrote) into the matching slice of the readback
        //    buffer. Resolves are cmd-list ops so they precede `Close`.
        if let (Some(heap), Some(readback)) = (
            self.timestamps.query_heap.as_ref(),
            self.timestamps.readback.as_ref(),
        ) {
            let (_, end_slot) = super::pass_timing::whole_frame_pair(frame);
            unsafe {
                end_cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot);
                end_cmd.ResolveQueryData(
                    heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    super::pass_timing::frame_resolve_start(frame),
                    super::pass_timing::SLOTS_PER_FRAME as u32,
                    readback,
                    super::pass_timing::frame_readback_byte_offset(frame),
                );
            }
        }

        unsafe { end_cmd.Close() }.map_err(|e| format!("end cmd close: {e}"))?;

        // 5. Submit everything in topological order: [start, per-pass...,
        //    end]. Single ExecuteCommandLists call → the GPU executes
        //    them in submission order; the queue is serial on a single
        //    DIRECT command queue, so this guarantees pass ordering.
        let mut submission: Vec<Option<ID3D12CommandList>> =
            Vec::with_capacity(2 + pass_cmd_lists.len());
        let start_handle: ID3D12CommandList = start_cmd
            .cast()
            .map_err(|e| format!("start cmd cast: {e}"))?;
        submission.push(Some(start_handle));
        for cl in &pass_cmd_lists {
            let h: ID3D12CommandList = cl.cast().map_err(|e| format!("per-pass cmd cast: {e}"))?;
            submission.push(Some(h));
        }
        let end_handle: ID3D12CommandList =
            end_cmd.cast().map_err(|e| format!("end cmd cast: {e}"))?;
        submission.push(Some(end_handle));
        unsafe { self.command_queue.ExecuteCommandLists(&submission) };

        // Present. Sync interval 1 locks to the display refresh (vsync); 0 runs
        // uncapped. The tearing present flag is required (and only valid) at
        // sync interval 0 on a swapchain created with ALLOW_TEARING, so gate it
        // on the current interval too -- `set_vsync` flips the interval at
        // runtime, and ALLOW_TEARING with interval >= 1 is an invalid Present.
        let present_flags = if self.present_sync_interval == 0 && self.allow_tearing {
            DXGI_PRESENT_ALLOW_TEARING
        } else {
            DXGI_PRESENT(0)
        };
        let present_result = unsafe {
            self.swapchain
                .Present(self.present_sync_interval, present_flags)
        };
        if let Err(e) = present_result.ok() {
            self.flush_validation();
            let reason = unsafe { self.device.GetDeviceRemovedReason() };
            return Err(format!("Present: {e}; device removed reason: {reason:?}"));
        }
        // Record the buffer just shown so a headless `screenshot` captures the
        // on-screen image (the next `GetCurrentBackBufferIndex` already advanced
        // past it).
        self.last_present_index = Some(back_idx);

        // Advance fence. The signalled value must be globally unique across
        // slots so each slot's wait-before-reuse only observes completion of
        // its own prior submission.
        let next_val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(next_val + 1);
        self.frame_sync.fence_values[frame] = next_val;
        unsafe { self.command_queue.Signal(&self.frame_sync.fence, next_val) }
            .map_err(|e| format!("Signal: {e}"))?;

        self.current_frame = (self.current_frame + 1) % FRAMES;
        Ok(())
    }

    // Drain any queued D3D12 validation messages and emit them via tracing.
    fn flush_validation(&self) {
        if let Some(ref iq) = self.info_queue {
            drain_info_queue(iq);
        }
    }

    pub fn update_view(&mut self, matrix: [[f32; 4]; 4]) {
        self.view_matrix = matrix;
    }

    pub fn update_model(&mut self, index: usize, model: [[f32; 4]; 4]) {
        if let Some(obj) = self.draw_objects.get_mut(index) {
            obj.model = model;
        }
    }

    pub fn update_visibility(&mut self, index: usize, visible: bool) {
        if let Some(obj) = self.draw_objects.get_mut(index) {
            obj.visible = visible;
        }
    }

    // Retire a draw object for a despawned entity: clear `visible` (drops it
    // from the main / shadow / velocity passes) and `resident` (drops it from
    // the ray-tracing BLAS / geometry-table rebuild), so it leaves no ghost in
    // any pass, then return its slot to the free list so the next runtime clone
    // (or streamed chunk) recycles it. The geometry buffers stay allocated. If
    // the slot held a runtime clone, its descriptor-pool offset is freed too so
    // a steady spawn/despawn cadence does not exhaust the clone pool. No-op if
    // the index is out of range.
    pub fn retire_draw_object(&mut self, index: usize) {
        if let Some(obj) = self.draw_objects.get_mut(index) {
            obj.visible = false;
            obj.resident = false;
            if let Some(offset) = self.clone.slot_by_draw_idx.remove(&index) {
                self.clone.free_offsets.push(offset);
            }
            // Only the runtime-append region (streamed chunks + spawned clones,
            // `index >= n_objects`) recycles its draw slots. A build-time slot
            // stays allocated when hidden: the init-time cull BVH and the RT
            // acceleration structure's `object_indices` are keyed to fixed
            // build-time slot indices and cannot refit, so reusing one would
            // mis-key them. (Metal recycles build-time slots too because its
            // per-frame RT topology refresh re-admits them; DX has no such
            // refresh -- tracked as the RT incremental topology parity item.)
            if index >= self.n_objects {
                self.draw_slots.free(index);
            }
        }
    }

    // Add a draw slot to `always_draw` if it is not already a member. Runtime
    // draws (chunks, spawned clones) are drawn unconditionally because the
    // init-time BVH cannot refit to admit them; a slot recycled from a culled
    // static prop is not yet in `always_draw` and must be added, while one
    // recycled from another chunk / clone already is.
    pub(super) fn ensure_always_draw(&mut self, slot: usize) {
        if !self.always_draw_member[slot] {
            self.always_draw.push(slot as u32);
            self.always_draw_member[slot] = true;
        }
    }

    pub fn update_clear_color(&mut self, color: [f32; 4]) {
        self.clear_color = color;
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. `gpu_frame_us` is filled at the top of each `draw_frame` from
    // the timestamp pair this slot resolved on its previous trip through the
    // ring (so a `FRAMES`-stale window, matching Metal's "frame or two
    // stale" reading).
    pub fn render_stats(&self) -> crate::gfx::profile::RenderStats {
        self.frame_stats.get()
    }

    // Shared atomic clone of the shader-reload flag, or `None` when the
    // context was built without hot-reload. Surfaced through the
    // `RenderBackend` trait so the debug WebSocket server's
    // `reload-shaders` command can flip it from a non-render thread.
    pub fn shader_reload_pending(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.hot_reload
            .reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
    }

    // True when the render graph should include `PassId::Raymarch`. Wraps
    // the `raymarch.any_visible()` check so callers don't have to know
    // the resource is `Option`. Drives `FrameGraphInputs::raymarch_enabled`
    // in `record_frame::seed_inputs`.
    pub(super) fn raymarch_enabled(&self) -> bool {
        self.raymarch
            .as_ref()
            .map(|r| r.any_visible())
            .unwrap_or(false)
    }

    // True when the render graph should include `PassId::RtReflections` (and
    // omit `SsrResolve`). Both the resolve resources and the acceleration
    // structure must be live; either being absent (DXR unsupported, DXC missing,
    // accel build failed, or an empty scene) falls the graph back to SSR. Drives
    // `FrameGraphInputs::rt_reflections_enabled` in `record_frame::seed_inputs`
    // and the `scene_srv_for_post` precedence.
    pub(super) fn rt_reflections_active(&self) -> bool {
        self.rt_reflections.is_some() && self.rt_accel.is_some()
    }

    // True when a reflection resolve (SSR resolve or RT reflections) runs this
    // frame. RT takes precedence at the graph level, so at most one resolve
    // runs; either feeds the same composite. Single-sources the predicate that
    // `scene_srv_for_post` and the glass pass use to pick the scene-with-
    // reflections target, and that the forward shader reads (via
    // `ViewUniforms::reflections_enabled`) to hand glossy dielectric specular to
    // that resolve instead of double-counting the forward probe reflection.
    pub(super) fn reflection_resolve_active(&self) -> bool {
        self.rt_reflections_active() || self.ssr.as_ref().and_then(|s| s.resolve.as_ref()).is_some()
    }

    // True when glass panes trace a per-pixel RT reflection this frame: RT is live
    // AND the glass RT pipelines built (DXR + DXC). Single-sources the glass-RT
    // decision so the two consumers agree: `encode_transparent` selects the RT
    // trace, and `graph_exec` skips the planar mirror re-render (RT supersedes
    // planar). They MUST gate on the same predicate -- if RT is live but the glass
    // RT pipelines failed to build, the glass pass falls back to the probe/planar
    // path, so the planar resolve must still be rendered for it to sample (gating
    // the skip on `rt_reflections_active()` alone would leave it sampling a stale
    // resolve).
    pub(super) fn rt_glass_active(&self) -> bool {
        self.rt_reflections_active() && self.glass.as_ref().is_some_and(|g| g.rt_pipelines_ready())
    }

    // True when the render graph should include `PassId::Transparent`. Wraps
    // the `glass.any_visible()` check; drives
    // `FrameGraphInputs::transparent_enabled` in `record_frame::seed_inputs`.
    pub(super) fn transparent_enabled(&self) -> bool {
        self.glass
            .as_ref()
            .map(|g| g.any_visible())
            .unwrap_or(false)
    }

    // Bump this frame's CPU-issued draw-call counter. Called from each draw
    // site in the shadow, main, decal, composite, and text passes. Mirrors
    // `MtlContext::frame_stats.draw_calls += 1`; fullscreen post-process
    // passes (SSAO, SSR, TAA, bloom) are not counted per the `RenderStats`
    // doc comment.
    pub(super) fn inc_draw_calls(&self, n: u32) {
        // Bump the atomic accumulator so worker threads encoding in
        // parallel don't race. Drained into `frame_stats.draw_calls`
        // by `draw_frame` at the end of every frame.
        self.draw_calls_accum
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }
}

impl DxContext {
    pub fn window_closed(&mut self) -> bool {
        pump_messages();
        self.win_state.closed
    }

    pub fn wait_idle(&self) {
        // Signal a new fence value and wait until the GPU reaches it.
        let val = self.frame_sync.next_fence_value.get();
        self.frame_sync.next_fence_value.set(val + 1);
        if unsafe { self.command_queue.Signal(&self.frame_sync.fence, val) }.is_ok()
            && unsafe { self.frame_sync.fence.GetCompletedValue() } < val
            && let Ok(()) = unsafe {
                self.frame_sync
                    .fence
                    .SetEventOnCompletion(val, self.frame_sync.fence_event)
            }
        {
            unsafe { WaitForSingleObject(self.frame_sync.fence_event, u32::MAX) };
        }
    }

    pub fn capture_cursor(&mut self) {
        // Don't grab the cursor immediately. A freshly spawned window may not be
        // focused yet, and clipping + hiding the system cursor before the user
        // has interacted with the window is jarring; it also diverges from the
        // Vulkan/GLFW backend, where a disabled cursor only engages once the
        // window gains focus. Instead arm the click-to-capture path: the first
        // left-click in the content area grabs the cursor, the same as
        // recapturing after Escape or a focus loss (see `wnd_proc`'s
        // `WM_LBUTTONDOWN` arm).
        self.win_state.recapture_on_click = true;
    }

    #[allow(dead_code)]
    pub fn release_cursor(&mut self) {
        do_release_cursor(&mut self.win_state);
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the helper, so calling
    // it every frame with the same value is cheap.
    pub fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        do_set_ui_cursor_hidden(&mut self.win_state, hidden);
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`. The wnd_proc reads this flag to route
    // Escape to the ECS and suppress click-to-recapture.
    pub fn set_menu_mode(&mut self, on: bool) {
        self.win_state.menu_mode = on;
    }

    // Edge-triggered capture: capture for camera control, release while a menu
    // is open. GraphicsSystem calls this each frame in menu mode. Unlike the
    // startup `capture_cursor` (which arms click-to-capture), closing the menu
    // recaptures immediately so the camera resumes without an extra click.
    pub fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.win_state.cursor_captured {
            return;
        }
        if capture {
            let hwnd = self.win_state.hwnd;
            do_capture_cursor(hwnd, &mut self.win_state);
        } else {
            do_release_cursor(&mut self.win_state);
        }
    }

    // Turn display sync (vsync) on or off at runtime. Only the present sync
    // interval changes (1 = lock to refresh, 0 = uncapped); the present flags
    // gate ALLOW_TEARING on this interval. The swapchain's ALLOW_TEARING flag
    // is fixed at creation, so true tearing is available only when the
    // swapchain was created with vsync off; turning vsync off later still
    // presents uncapped (interval 0) but without the tearing flag if the
    // swapchain lacks it.
    pub fn set_vsync(&mut self, on: bool) {
        self.present_sync_interval = if on { 1 } else { 0 };
    }

    // Switch window mode / resize at runtime (windowed / borderless / fullscreen
    // and content-size presets). The Win32 work lives in `window.rs`; the resize
    // path picks up the resulting WM_SIZE. Code-only on macOS; verify on Windows.
    pub fn set_window_mode(&mut self, mode: crate::assets::WindowMode) {
        do_set_window_mode(&mut self.win_state, mode);
    }

    pub fn set_window_size(&mut self, width: u32, height: u32) {
        do_set_window_size(&mut self.win_state, width, height);
    }

    // Replace the live post-process parameters, pushed to the bloom + composite
    // shaders each frame. Code-only on macOS; verify on Windows.
    pub fn update_post_process(&mut self, params: crate::gfx::render_types::PostProcessParams) {
        self.post_process = params;
    }

    // Set the live ambient (IBL) light scale (the Ambient slider). It lives in
    // `LightUniforms`, uploaded to a single (not per-frame) constant buffer, so
    // unlike `update_post_process` (root constants) it cannot just stash a value:
    // it mutates the CPU-side copy and re-uploads the buffer. Because the buffer
    // is shared across frames-in-flight, the GPU is drained first so the rewrite
    // never races an in-flight read; ambient changes only on a slider drag, so
    // the stall is rare. Edge-triggered: a no-op when the value is unchanged
    // (e.g. an init push with no persisted override), so a steady scene never
    // stalls.
    pub fn set_ambient_intensity(&mut self, value: f32) {
        if self.uniforms.light_uniforms.ambient_intensity == value {
            return;
        }
        self.uniforms.light_uniforms.ambient_intensity = value;
        self.wait_idle();
        if let Err(e) = super::draw::upload_light_uniforms(
            &self.uniforms.light_ubo,
            &self.uniforms.light_uniforms,
        ) {
            tracing::warn!("set_ambient_intensity: re-upload light uniforms failed: {e}");
        }
    }

    // Replace the runtime movement key map. The window message loop decodes
    // key events through it, so a settings-menu rebind takes effect immediately.
    pub fn set_keymap(&mut self, keymap: &crate::gfx::keymap::KeyMap) {
        self.win_state.key.set_keymap(keymap);
    }

    pub fn take_input(&mut self) -> InputState {
        let dx = self.win_state.mouse_dx;
        let dy = self.win_state.mouse_dy;
        let mx = self.win_state.mouse_x;
        let my = self.win_state.mouse_y;
        let lc = self.win_state.left_click_pending;
        // Held-button (UI drag) persists across the drain until WM_LBUTTONUP;
        // the accumulated scroll delta is one-shot and reset like the mouse delta.
        let lbd = self.win_state.left_button_down;
        let scroll = self.win_state.scroll_delta;
        self.win_state.mouse_dx = 0.0;
        self.win_state.mouse_dy = 0.0;
        self.win_state.left_click_pending = false;
        self.win_state.scroll_delta = 0.0;
        self.win_state.key.take(dx, dy, mx, my, lc, lbd, scroll)
    }

    // Live window size for overlay (view-owned UI) scaling and cursor
    // hit-testing. Returns the drawable (swapchain) pixel size, which is the
    // attachment the composite + text pass writes and the space the UI shader
    // divides vertices by; WM_MOUSEMOVE reports the cursor in the same client
    // pixels, so the overlay forward / inverse transforms stay consistent.
    pub fn logical_size(&self) -> (f32, f32) {
        (self.output_width as f32, self.output_height as f32)
    }

    // Device capability flags for the settings menu. RT reflects the DXR-tier
    // query made at init (`rt_capable`).
    pub fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities {
        crate::gfx::backend::DeviceCapabilities {
            ray_tracing: self.rt_capable,
        }
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the adapter description (vendor id + dedicated VRAM). `UNKNOWN` when
    // the adapter does not expose the v3 interface or the desc query fails.
    pub fn gpu_profile(&self) -> crate::gfx::backend::GpuProfile {
        use crate::gfx::backend::{GpuClassInput, GpuProfile, GpuVendor, classify_tier};
        let Some(adapter) = self.adapter.as_ref() else {
            return GpuProfile::UNKNOWN;
        };
        let desc = match unsafe { adapter.GetDesc1() } {
            Ok(d) => d,
            Err(_) => return GpuProfile::UNKNOWN,
        };
        let vendor = match desc.VendorId {
            0x10DE => GpuVendor::Nvidia,
            0x1002 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            _ => GpuVendor::Other,
        };
        let dedicated = desc.DedicatedVideoMemory as u64;
        // A discrete GPU has dedicated VRAM; an integrated part reports little or
        // none (and large shared system memory). A small floor keeps a few MB of
        // carve-out from reading as discrete.
        let discrete = dedicated >= (256u64 << 20);
        let tier = classify_tier(&GpuClassInput {
            vendor,
            memory_budget_bytes: dedicated,
            discrete,
            apple_family: 0,
        });
        GpuProfile {
            vendor,
            tier,
            memory_budget_bytes: dedicated,
            unified_memory: !discrete,
            discrete,
        }
    }

    // GPU descriptor handle for the per-object (albedo, normal) SRV pair.
    pub(super) fn object_srv_gpu(&self, obj_idx: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        // albedo slot = 1 + obj_idx*2; descriptor table covers 2 SRVs from there.
        let slot = 3 + obj_idx * 2;
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + (slot * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // GPU descriptor handle for the per-cluster (albedo, normal) SRV pair.
    pub(super) fn cluster_srv_gpu(&self, cluster_idx: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        let slot = 3 + self.n_objects * 2 + cluster_idx * 2;
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + (slot * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // GPU descriptor handle for skinned object `i`'s (albedo, normal) SRV pair.
    pub(super) fn skinned_srv_gpu(&self, i: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let srv_gpu_base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        let slot = self.skinned.srv_base_slot + i * 2;
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: srv_gpu_base.ptr + (slot * self.descriptors.srv_descriptor_size) as u64,
        }
    }
}

impl crate::gfx::scene_reel::SceneControl for DxContext {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        self.update_visibility(draw_idx, visible);
    }
    fn update_clear_color(&mut self, color: [f32; 4]) {
        self.update_clear_color(color);
    }
}

impl Drop for DxContext {
    fn drop(&mut self) {
        self.wait_idle();
        // Restore cursor clip + visibility so the OS isn't left in a bad state
        // if the caller didn't release explicitly.
        do_release_cursor(&mut self.win_state);
        // Unmap persistent CBV mappings (view + shadow).
        self.uniforms.unmap();
        unsafe { CloseHandle(self.frame_sync.fence_event) }.ok();
        // The remaining COM objects (device, swapchain, heaps, etc.) are reference-
        // counted and released automatically when the struct fields are dropped.
    }
}

//  D3D12 debug-layer message draining
//
// Standalone version of the per-frame flush_validation so init paths can dump
// validation messages before bailing; without this, PSO/root-sig failures
// surface only as the bare `E_INVALIDARG` HRESULT from CreateGraphicsPipelineState.

pub(super) fn drain_info_queue(iq: &ID3D12InfoQueue) {
    let count = unsafe { iq.GetNumStoredMessages() };
    for i in 0..count {
        let mut len = 0usize;
        if unsafe { iq.GetMessage(i, None, &mut len) }.is_err() {
            continue;
        }
        let mut buf = vec![0u8; len];
        let msg_ptr = buf.as_mut_ptr() as *mut D3D12_MESSAGE;
        if unsafe { iq.GetMessage(i, Some(msg_ptr), &mut len) }.is_err() {
            continue;
        }
        let msg = unsafe { &*msg_ptr };
        let text = if msg.pDescription.is_null() {
            "(no description)".to_owned()
        } else {
            unsafe { std::ffi::CStr::from_ptr(msg.pDescription as *const i8) }
                .to_string_lossy()
                .into_owned()
        };
        match msg.Severity {
            D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR => {
                tracing::error!(target: "d3d12", "{text}");
            }
            D3D12_MESSAGE_SEVERITY_WARNING => {
                tracing::warn!(target: "d3d12", "{text}");
            }
            _ => {
                tracing::debug!(target: "d3d12", "{text}");
            }
        }
    }
    unsafe { iq.ClearStoredMessages() };
}

// Wrap an init-path Result so that any D3D12 validation messages queued
// during the failing op are dumped to tracing before the error bubbles up.
pub(super) fn dump_on_err<T>(
    info_queue: Option<&ID3D12InfoQueue>,
    r: Result<T, String>,
) -> Result<T, String> {
    if r.is_err()
        && let Some(iq) = info_queue
    {
        drain_info_queue(iq);
    }
    r
}
