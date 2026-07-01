// Vulkan rendering context. Owns all GPU resources, the GLFW window, and input state.
// Mirrors the public API of metal::MtlContext so GraphicsSystem can drive both
// backends identically.

use std::cell::RefCell;

use ash::{Device, vk};

use crate::gfx::render_types::*;

use super::draw::*;
use super::input::*;
use super::post::*;
use super::texture::*;

// Off-screen HDR render-target format. The main pass renders linear-light
// radiance into this; the composite pass tonemaps it down to the swapchain's
// 8-bit format. `R16G16B16A16_SFLOAT` is universally supported as a colour
// attachment + sampled image on desktop GPUs.
pub(super) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

// Cap on runtime-cloned static draws (`clone_static_draw_object`). The
// clone descriptor pool is sized for this many (albedo, normal) sets at
// init. Editor-only: exhausting the pool only happens under
// `world.jsonl` hot-reload churn that adds 129+ new Props referencing
// existing meshes; the call returns an error past that. Mirrors
// `directx::context::MAX_CLONE_DRAWS`. Used by `clone_static_draw_object`,
// which is reached only through the bin's `cn debug` runtime-mutation path
// (dead in the FFI lib, live in the bin) -- hence the allow, matching DirectX.
#[allow(dead_code)]
pub(super) const MAX_CLONE_DRAWS: usize = 128;

// MAX_BLOOM_MIPS now lives in `crate::vulkan::post::bloom` (re-exported as
// `crate::vulkan::post::MAX_BLOOM_MIPS`).

// Cascaded-shadow-map resources, grouped off the flat `VkContext` field soup
// (mirrors the DirectX backend's `self.shadow`). The barrier executor resolves
// `shadow_map` through `build_barrier_registry`, so moving these fields behind
// one field left the parallel emit path untouched.
pub(super) struct VkShadow {
    pub(super) render_pass: vk::RenderPass,
    pub(super) map: GpuImage,
    pub(super) map_size: u32,
    // One framebuffer per cascade slice. Empty when the shadow pass is disabled.
    pub(super) framebuffers: Vec<vk::Framebuffer>,
    pub(super) pipeline: Option<vk::Pipeline>,
    pub(super) pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) global_set_layout: Option<vk::DescriptorSetLayout>,
    pub(super) global_sets: Vec<vk::DescriptorSet>,
    pub(super) sampler: vk::Sampler,
    pub(super) skinned_pipeline: Option<vk::Pipeline>,
    pub(super) skinned_pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) ubo: vk::Buffer,
    pub(super) ubo_memory: vk::DeviceMemory,
    // Carried CSM uniforms: skipped cascades keep the VP their slice was last
    // rendered with. Splits refresh every frame; per-cascade light VPs only when
    // `render_mask` includes that cascade. Uploaded to `ubo` each frame.
    pub(super) uniforms: ShadowUniforms,
    // World-space direction toward the first directional light, cached at init.
    // Per-frame CSM updates use this; refresh it when lights change for a moving
    // sun.
    pub(super) light_dir: [f32; 3],
    // Cascade re-render policy from GraphicsConfig.shadow_update. Hybrid
    // refreshes the near cascade every frame and the far cascades round-robin.
    pub(super) update: crate::assets::ShadowUpdate,
    // Shadow distance in world units (GraphicsConfig.shadow_distance), read by the
    // per-frame cascade-split computation and capped at the camera far plane.
    pub(super) distance: u32,
    // Active shadow cascade count, 1..=4 (GraphicsConfig.shadow_cascades). The
    // per-frame split + schedule read it; only the first `cascades` of the four
    // slots are rendered + sampled. Stored at init (applies at the next launch).
    pub(super) cascades: u32,
    // Round-robin clock + primed-set for the cascade schedule; advanced once per
    // frame in draw_frame.
    pub(super) scheduler: crate::gfx::shadow_schedule::ShadowCascadeScheduler,
    // Cascades re-rendered this frame (bit `i` = cascade `i`). Set in draw_frame
    // and read by encode_shadow_pass so the two agree on which slices to refresh
    // and which to leave intact.
    pub(super) render_mask: u32,
}

impl VkShadow {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The per-frame `global_sets` are freed with the shared
    // descriptor pool, so they are not destroyed here.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(p) = self.skinned_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.skinned_pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(sl) = self.global_set_layout {
                device.destroy_descriptor_set_layout(sl, None);
            }
        }
        self.map.destroy(device);
        unsafe {
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_buffer(self.ubo, None);
            device.free_memory(self.ubo_memory, None);
            device.destroy_sampler(self.sampler, None);
        }
    }
}

// Skinned (skeletally animated) mesh resources, grouped off the flat `VkContext`
// field soup. All `None` / empty until `upload_skinned` runs; with no
// `SkinnedMesh` in the world every skinned pass is skipped. The joint matrices
// live in per-(frame, object) storage buffers bound through `joint_sets`: set 2
// for the main pass, set 1 for the shadow pass; the descriptor set layout is
// identical so one set serves both.
pub(super) struct VkSkinned {
    pub(super) pipeline: Option<vk::Pipeline>,
    pub(super) pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) joint_set_layout: Option<vk::DescriptorSetLayout>,
    pub(super) descriptor_pool: Option<vk::DescriptorPool>,
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) vertex_buffer_memory: vk::DeviceMemory,
    pub(super) index_buffer: vk::Buffer,
    pub(super) index_buffer_memory: vk::DeviceMemory,
    // Current byte sizes of the skinned VB / IB. Used by
    // `update_skinned_mesh_geometry` to bound-check the slot region the asset
    // hot-reload write lands in. Zero until `upload_skinned` runs.
    pub(super) vertex_buffer_bytes: u64,
    pub(super) index_buffer_bytes: u64,
    pub(super) draw_objects: Vec<SkinnedDrawObject>,
    // Per-object (albedo, normal) descriptor sets (set 1 for the main pass).
    pub(super) object_sets: Vec<vk::DescriptorSet>,
    // Per-(frame, object) joint storage buffers (host-mapped) + their
    // descriptor sets. Indexed [frame_idx][skinned_idx].
    pub(super) joint_buffers: Vec<Vec<vk::Buffer>>,
    pub(super) joint_memories: Vec<Vec<vk::DeviceMemory>>,
    pub(super) joint_ptrs: Vec<Vec<*mut u8>>,
    pub(super) joint_sets: Vec<Vec<vk::DescriptorSet>>,
    // Current skinning matrices per skinned object, parallel to `draw_objects`.
    // Rewritten each frame by `update_skinned_pose`.
    pub(super) joint_matrices: Vec<Vec<[[f32; 4]; 4]>>,
    // GPU-driven main-pass skinning fold. `skin` is the `rt_skin` compute pipeline
    // (reused independently of RT) + its per-(frame, object) descriptor sets,
    // written once in `build_main_skin`. `deformed` is one storage+vertex buffer
    // per frame-in-flight holding this frame's posed 56-byte `Vertex`s (global
    // skinned indexing, so the draw uses `base_vertex = 0`); `encode_skin` writes
    // it each frame and the bindless main pass's 2nd indirect draw reads it. Both
    // `None`/empty until `upload_skinned` runs with the bindless cull path active.
    pub(super) skin: Option<super::raytrace::SkinPipeline>,
    pub(super) deformed: Vec<super::raytrace::DeviceBuffer>,
    // `false` until the deformed-vertex ring has been posed at least one full
    // frame. While false the GPU-driven G-buffer velocity binds the current
    // deformed buffer as the previous one (prev_pos == cur_pos), so an unposed
    // ring slot never feeds a garbage skinned motion vector on the first frame
    // (or after a runtime ring rebuild). Mirrors the legacy joint priming. Reset
    // by `build_main_skin` / `upload_skinned`. Atomic, not `Cell`: the G-buffer
    // pass encodes on a `jobs::pool()` rayon worker thread (the parallel per-pass
    // encoder shares `&self` across workers), so any interior mutation reachable
    // from `encode_pass_into` must be atomic, like `draw_calls_accum`.
    pub(super) deformed_primed: std::sync::atomic::AtomicBool,
}

impl VkSkinned {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The per-object `object_sets` and per-frame `joint_sets` are
    // freed with `descriptor_pool`, so they are not destroyed here.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(l) = self.joint_set_layout {
                device.destroy_descriptor_set_layout(l, None);
            }
            if let Some(pool) = self.descriptor_pool {
                device.destroy_descriptor_pool(pool, None);
            }
            if self.vertex_buffer != vk::Buffer::null() {
                device.destroy_buffer(self.vertex_buffer, None);
                device.free_memory(self.vertex_buffer_memory, None);
                device.destroy_buffer(self.index_buffer, None);
                device.free_memory(self.index_buffer_memory, None);
            }
            for frame_bufs in &self.joint_buffers {
                for &buf in frame_bufs {
                    device.destroy_buffer(buf, None);
                }
            }
            for frame_mems in &self.joint_memories {
                for &mem in frame_mems {
                    device.unmap_memory(mem);
                    device.free_memory(mem, None);
                }
            }
        }
        // GPU-driven main-pass skinning resources.
        if let Some(skin) = &self.skin {
            skin.destroy(device);
        }
        for buf in &self.deformed {
            buf.destroy(device);
        }
    }
}

// Shared static vertex/index buffers plus the byte-range sub-allocators that
// carve streamed-mesh regions out of them, grouped off the flat `VkContext`
// field soup. Created at init and live for the context's lifetime; the
// streaming and geometry-rebuild paths swap the buffers in place.
pub(super) struct VkGeometry {
    pub(super) vertex_buffer: vk::Buffer,
    pub(super) vertex_buffer_memory: vk::DeviceMemory,
    pub(super) index_buffer: vk::Buffer,
    pub(super) index_buffer_memory: vk::DeviceMemory,
    // Byte-range sub-allocators for the streamed-mesh regions of the shared
    // vertex/index buffers. Empty until mesh streaming is active; `evict_mesh`
    // seeds them with each streamed draw's build-time region at init, then
    // `upload_mesh` / `evict_mesh` allocate and free byte ranges so a streamed
    // mesh lands wherever there is room.
    pub(super) mesh_vtx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub(super) mesh_idx_alloc: crate::gfx::range_alloc::RangeAllocator,
    // Current byte sizes of the shared vertex/index buffers. Tracked so
    // `setup_chunk_streaming` knows how much build-time geometry to copy when
    // it grows them.
    pub(super) vertex_buffer_bytes: u64,
    pub(super) index_buffer_bytes: u64,
}

impl VkGeometry {
    // Destroy the shared vertex/index buffers and their memory. Called from
    // `VkContext::drop` after `wait_idle`. The range allocators and byte counts
    // are plain CPU state with nothing to free.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_buffer_memory, None);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_buffer_memory, None);
        }
    }
}

// The main geometry-path descriptor set layouts plus the shared pool the
// per-frame sets are allocated from, grouped off the flat `VkContext` field
// soup. Global set 0 (camera / lights / shadow / IBL / SSAO), object set 1
// (per-draw albedo + normal), and the text-overlay set; the `*_sets` are
// allocated from `descriptor_pool` at init and freed with it. Post, instanced,
// chunk, clone, and skinned descriptors live in their own pools, not here.
pub(super) struct VkDescriptors {
    pub(super) global_set_layout: vk::DescriptorSetLayout,
    pub(super) object_set_layout: vk::DescriptorSetLayout,
    pub(super) text_set_layout: vk::DescriptorSetLayout,
    pub(super) descriptor_pool: vk::DescriptorPool,
    pub(super) global_sets: Vec<vk::DescriptorSet>,
    pub(super) object_sets: Vec<vk::DescriptorSet>,
    pub(super) text_atlas_sets: Vec<vk::DescriptorSet>,
}

impl VkDescriptors {
    // Destroy the shared descriptor pool (which frees every set allocated from
    // it: global_sets / object_sets / text_atlas_sets) and the three set
    // layouts. Called from `VkContext::drop` after `wait_idle`.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.global_set_layout, None);
            device.destroy_descriptor_set_layout(self.object_set_layout, None);
            device.destroy_descriptor_set_layout(self.text_set_layout, None);
        }
    }
}

// Instanced-prop rendering: the pipeline + per-cluster material sets + the
// per-(frame, cluster) instance storage buffers and their descriptor sets,
// grouped off the flat `VkContext` field soup. All `None` / empty when the
// world declares no `InstancedProp` clusters. `clusters` holds the declared
// clusters (each with its per-instance transforms); `lod_buckets` is the
// per-frame LOD partition every instanced draw site shares.
pub(super) struct VkInstanced {
    // Instanced pipeline; None when no InstancedProp clusters were declared.
    pub(super) pipeline: Option<vk::Pipeline>,
    pub(super) pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) set_layout: Option<vk::DescriptorSetLayout>,
    pub(super) clusters: Vec<InstancedCluster>,
    // Per-cluster (albedo, normal) sets used by the instanced pipeline.
    // Indexed by cluster index. Empty when no clusters are declared.
    pub(super) object_sets: Vec<vk::DescriptorSet>,
    // Per-frame, per-cluster instance buffer descriptor sets bound to set=2.
    // Indexed [frame_idx][cluster_idx].
    pub(super) sets: Vec<Vec<vk::DescriptorSet>>,
    // Per-frame, per-cluster instance storage buffers (host-mapped).
    pub(super) buffers: Vec<Vec<vk::Buffer>>,
    pub(super) memories: Vec<Vec<vk::DeviceMemory>>,
    pub(super) ptrs: Vec<Vec<*mut u8>>,
    // Per-cluster LOD-bucket partition for the current frame, indexed by
    // cluster index. Recomputed once per frame by `prepare_instanced_clusters`
    // (on `&mut self`, before the parallel pass fan-out) and consumed read-only
    // by every instanced draw site (main, shadow, SSR / SSAO / velocity
    // pre-passes) so all passes agree on the per-instance LOD pick and the
    // bucket-ordered byte layout uploaded into each cluster's instance SSBO.
    // Empty until the first frame / when no clusters are declared.
    pub(super) lod_buckets: Vec<Vec<InstancedLodBucket>>,
}

impl VkInstanced {
    // Destroy the pipeline, pipeline layout, instance set layout, and the
    // per-frame instance storage buffers + their mapped memory. Called from
    // `VkContext::drop` after `wait_idle`. The descriptor sets (`object_sets`,
    // `sets`) are freed with the shared descriptor pool, so they are not
    // destroyed here; `clusters` / `lod_buckets` are plain CPU state.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(l) = self.set_layout {
                device.destroy_descriptor_set_layout(l, None);
            }
            for frame_bufs in &self.buffers {
                for &buf in frame_bufs {
                    device.destroy_buffer(buf, None);
                }
            }
            for frame_mems in &self.memories {
                for &mem in frame_mems {
                    device.unmap_memory(mem);
                    device.free_memory(mem, None);
                }
            }
        }
    }
}

// Streamed VoxelWorld chunk rendering resources, grouped off the flat
// `VkContext` field soup (mirrors the DirectX backend's `chunk_stream:
// ChunkStreamState`, though Vulkan needs the extra descriptor pool + set and
// the reload-tracking material slots where DX reuses stable SRV-heap slots).
// All `None` / empty until `setup_chunk_streaming` runs; with no streamed
// chunks every field stays inert.
pub(super) struct VkChunkStream {
    // Byte-range sub-allocators for the headroom region appended to the shared
    // vertex/index buffers, disjoint from the build-time geometry and the
    // mesh-streaming allocators.
    pub(super) vtx_alloc: crate::gfx::range_alloc::RangeAllocator,
    pub(super) idx_alloc: crate::gfx::range_alloc::RangeAllocator,
    // Dedicated pool + one shared (albedo, normal) descriptor set for streamed
    // chunks.
    pub(super) descriptor_pool: Option<vk::DescriptorPool>,
    pub(super) object_set: Option<vk::DescriptorSet>,
    // Albedo / normal-map pool slots the shared chunk material samples, stored
    // (already clamped) so a streamed swap of either slot re-points `object_set`.
    pub(super) texture_slot: Option<usize>,
    pub(super) normal_map_slot: Option<usize>,
}

impl VkChunkStream {
    // Destroy the chunk descriptor pool (which frees `object_set`). Called from
    // `VkContext::drop` after `wait_idle`. The allocators, free-slot list, and
    // material slots are plain CPU state with nothing to free.
    pub(super) fn destroy(&self, device: &Device) {
        if let Some(pool) = self.descriptor_pool {
            unsafe { device.destroy_descriptor_pool(pool, None) };
        }
    }
}

// GPU-driven cull + bindless static main pass (+ optional two-pass Hi-Z
// occlusion), grouped off the flat `VkContext` field soup. Mirrors the DirectX
// backend's `cull: CullState`. A compute kernel frustum/distance-tests the
// build-time static objects and writes one indirect draw per survivor; the
// bindless main pass issues the whole buffer with one indirect draw. All
// `Some` / non-empty only when the world uses the built-in bindless shader with
// build-time geometry; non-bindless shaders keep the legacy per-draw loop. Field
// names are kept verbatim (heterogeneous prefixes, no single cluster prefix to
// drop). The two-pass Hi-Z pyramid + its temporal state live here too. The
// legacy `main_pipeline` and the CPU `cull_bvh` are NOT part of this.
pub(super) struct VkCull {
    // Bindless static main pass. `Some` only on the built-in shader; `None`
    // keeps the legacy per-draw main pass. The bindless descriptor sets are
    // freed with the shared descriptor pool.
    pub(super) bindless_pipeline: Option<vk::Pipeline>,
    pub(super) bindless_pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) bindless_set_layout: Option<vk::DescriptorSetLayout>,
    // One bindless descriptor set per frame-in-flight: binding 0 is that frame's
    // GpuObjectData storage buffer, binding 1 the shared texture pool.
    pub(super) bindless_sets: Vec<vk::DescriptorSet>,
    // Per-frame GpuObjectData storage buffers, persistently mapped; rebuilt each
    // frame from `draw_objects[..n_objects]`.
    pub(super) object_buffers: Vec<vk::Buffer>,
    pub(super) object_buffer_memories: Vec<vk::DeviceMemory>,
    pub(super) object_buffer_ptrs: Vec<*mut u8>,
    // Compute cull pipeline + its per-frame sets (bindings 0/1/2 = that frame's
    // object SSBO, draw-args SSBO, indirect-command SSBO). Sets are pool-freed.
    pub(super) cull_pipeline: Option<vk::Pipeline>,
    pub(super) cull_pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) cull_set_layout: Option<vk::DescriptorSetLayout>,
    pub(super) cull_sets: Vec<vk::DescriptorSet>,
    // Per-frame `GpuDrawArgs` storage buffers, persistently mapped.
    pub(super) draw_args_buffers: Vec<vk::Buffer>,
    pub(super) draw_args_buffer_memories: Vec<vk::DeviceMemory>,
    pub(super) draw_args_buffer_ptrs: Vec<*mut u8>,
    // Per-frame indirect draw-command buffers the cull kernel writes and the
    // main pass consumes (`INDIRECT_BUFFER`). Device-local.
    pub(super) indirect_buffers: Vec<vk::Buffer>,
    pub(super) indirect_buffer_memories: Vec<vk::DeviceMemory>,
    // Per-frame per-object cull-status buffers (one u32 each): phase-1 writes,
    // phase-2 reads. Device-local storage.
    pub(super) cull_status_buffers: Vec<vk::Buffer>,
    pub(super) cull_status_buffer_memories: Vec<vk::DeviceMemory>,
    // Two-pass Hi-Z occlusion (HizBuild -> Cull2 -> Main2). `occlusion_two_pass`
    // records the world's request; the live resources below are `Some` /
    // non-empty only when it AND the bindless cull path are active.
    pub(super) occlusion_two_pass: bool,
    // Phase-2 cull pipeline (same layout as `cull_pipeline`) + its per-frame
    // sets, allocated from `two_pass_pool`.
    pub(super) cull_pipeline_phase2: Option<vk::Pipeline>,
    pub(super) cull_sets2: Vec<vk::DescriptorSet>,
    pub(super) two_pass_pool: Option<vk::DescriptorPool>,
    // Per-frame second indirect draw-command buffers `Cull2` writes and `Main2`
    // consumes. Device-local.
    pub(super) indirect_buffers2: Vec<vk::Buffer>,
    pub(super) indirect_buffer2_memories: Vec<vk::DeviceMemory>,
    // Phase-1 / phase-2 main render passes (render-pass-compatible with the
    // main-pass `framebuffers`).
    pub(super) main_render_pass_phase1: Option<vk::RenderPass>,
    pub(super) main_render_pass_phase2: Option<vk::RenderPass>,
    // Hi-Z occlusion culling. The depth-mip pyramid (built at end of frame
    // from this frame's main depth) + its build pipelines + the cull pipeline's
    // set 1 (`sampler2D` Hi-Z + per-frame `CullHizParams` UBO). `Some` exactly
    // when the GPU-cull pipeline is active (same gating as `cull_pipeline`):
    // the next frame's `Cull` kernel projects each AABB through the previous
    // frame's un-jittered VP and discards objects fully behind the pyramid.
    pub(super) hiz: Option<crate::vulkan::hiz::HiZResources>,
    // False on the first frame and immediately after a swapchain resize (no
    // valid pyramid yet); drives the cull UBO's `hiz_enabled` so the cull
    // kernel falls back to frustum + distance only until a pyramid at the
    // current resolution exists. Set true at the end of `record_frame` once a
    // build has run.
    pub(super) hiz_valid: bool,
    // Previous frame's un-jittered camera view-projection, fed to the Hi-Z cull
    // test. Updated every frame (independent of TAA, which keeps its own
    // `prev_view_proj`). The pyramid is reduced from depth rendered with the
    // jittered VP; the sub-pixel discrepancy is conservative, matching DirectX
    // / Metal which also project through the previous un-jittered VP.
    pub(super) hiz_prev_view_proj: [[f32; 4]; 4],
    // GPU-driven shadow pass. `shadow_cull_pipeline` is a frustum +
    // distance only cull kernel (`SHADOW_CULL`, no Hi-Z / status) over a lean
    // 3-SSBO set (objects + draw-args + this cascade's indirect-command buffer);
    // one dispatch per re-rendered cascade writes that cascade's indirect buffer.
    // `shadow_bindless_pipeline` is a depth-only graphics pipeline whose VS reads
    // `model` from the GpuObjectData SSBO (gl_InstanceIndex) and projects through
    // `light_vps[cascade_idx]` (a push constant); each cascade is then issued with
    // one `cmd_draw_indexed_indirect` (static+instance prefix) + one for the
    // skinned tail. `shadow_indirect_buffers` / `shadow_cull_sets` are indexed
    // [frame][cascade]. All `Some`/non-empty only when the bindless cull path is
    // active AND shadows are enabled.
    pub(super) shadow_cull_pipeline: Option<vk::Pipeline>,
    pub(super) shadow_cull_pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) shadow_cull_set_layout: Option<vk::DescriptorSetLayout>,
    pub(super) shadow_cull_sets: Vec<Vec<vk::DescriptorSet>>,
    pub(super) shadow_bindless_pipeline: Option<vk::Pipeline>,
    pub(super) shadow_bindless_pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) shadow_indirect_buffers: Vec<Vec<vk::Buffer>>,
    pub(super) shadow_indirect_buffer_memories: Vec<Vec<vk::DeviceMemory>>,
    // GPU-driven G-buffer pre-pass. A 3-MRT bindless pipeline whose VS
    // reads `model` + `roughness` from the GpuObjectData SSBO (gl_InstanceIndex)
    // and the previous-frame model from `prev_model_buffers`; the velocity history
    // for the skinned tail rides the previous-frame deformed buffer. The pass
    // reuses the main pass's `indirect_buffers` (camera frustum, no extra cull).
    // Set 0 (`gbuffer_set_layout`) = GbView UBO + prev_model SSBO; set 1 = the
    // shared bindless set. `gbuffer_sets` is one set 0 per frame; the per-frame
    // `prev_model_*` buffers are host-mapped (instance region init-written, static
    // + skinned regions rewritten each frame). All `Some`/non-empty only when the
    // bindless cull path is active AND the G-buffer is enabled.
    pub(super) gbuffer_bindless_pipeline: Option<vk::Pipeline>,
    pub(super) gbuffer_bindless_pipeline_layout: Option<vk::PipelineLayout>,
    pub(super) gbuffer_set_layout: Option<vk::DescriptorSetLayout>,
    pub(super) gbuffer_sets: Vec<vk::DescriptorSet>,
    pub(super) prev_model_buffers: Vec<vk::Buffer>,
    pub(super) prev_model_memories: Vec<vk::DeviceMemory>,
    pub(super) prev_model_ptrs: Vec<*mut u8>,
}

impl VkCull {
    // Destroy every owned GPU object. Called from `VkContext::drop` after
    // `wait_idle`. The bindless / cull / phase-2 descriptor sets are freed with
    // the shared descriptor pool + `two_pass_pool`, so they are not destroyed
    // here. `occlusion_two_pass` is plain CPU state. Takes `&mut self` because
    // `HiZResources::destroy` nulls out its handles as it frees them.
    pub(super) fn destroy(&mut self, device: &Device) {
        // Hi-Z occlusion resources (image + build pipelines + cull-read sets +
        // per-frame cull UBOs).
        if let Some(hiz) = &mut self.hiz {
            hiz.destroy(device);
        }
        unsafe {
            if let Some(p) = self.bindless_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.bindless_pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(sl) = self.bindless_set_layout {
                device.destroy_descriptor_set_layout(sl, None);
            }
            for &buf in &self.object_buffers {
                device.destroy_buffer(buf, None);
            }
            for &mem in &self.object_buffer_memories {
                device.free_memory(mem, None);
            }
            if let Some(p) = self.cull_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.cull_pipeline_phase2 {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.cull_pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(sl) = self.cull_set_layout {
                device.destroy_descriptor_set_layout(sl, None);
            }
            if let Some(pool) = self.two_pass_pool {
                device.destroy_descriptor_pool(pool, None);
            }
            if let Some(rp) = self.main_render_pass_phase1 {
                device.destroy_render_pass(rp, None);
            }
            if let Some(rp) = self.main_render_pass_phase2 {
                device.destroy_render_pass(rp, None);
            }
            // GPU-driven shadow pass. The per-(frame, cascade) `shadow_cull_sets`
            // are freed with the shared descriptor pool, so only the pipelines,
            // the set layout, and the per-cascade indirect buffers are destroyed.
            if let Some(p) = self.shadow_cull_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.shadow_cull_pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(sl) = self.shadow_cull_set_layout {
                device.destroy_descriptor_set_layout(sl, None);
            }
            if let Some(p) = self.shadow_bindless_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.shadow_bindless_pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            for &buf in self.shadow_indirect_buffers.iter().flatten() {
                device.destroy_buffer(buf, None);
            }
            for &mem in self.shadow_indirect_buffer_memories.iter().flatten() {
                device.free_memory(mem, None);
            }
            for &buf in self
                .draw_args_buffers
                .iter()
                .chain(self.indirect_buffers.iter())
                .chain(self.cull_status_buffers.iter())
                .chain(self.indirect_buffers2.iter())
            {
                device.destroy_buffer(buf, None);
            }
            for &mem in self
                .draw_args_buffer_memories
                .iter()
                .chain(self.indirect_buffer_memories.iter())
                .chain(self.cull_status_buffer_memories.iter())
                .chain(self.indirect_buffer2_memories.iter())
            {
                device.free_memory(mem, None);
            }
            // GPU-driven G-buffer pre-pass. The per-frame `gbuffer_sets` are freed
            // with the shared descriptor pool, so only the pipeline, layout, set
            // layout, and the per-frame prev_model buffers are destroyed here.
            if let Some(p) = self.gbuffer_bindless_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(pl) = self.gbuffer_bindless_pipeline_layout {
                device.destroy_pipeline_layout(pl, None);
            }
            if let Some(sl) = self.gbuffer_set_layout {
                device.destroy_descriptor_set_layout(sl, None);
            }
            for &buf in &self.prev_model_buffers {
                device.destroy_buffer(buf, None);
            }
            for &mem in &self.prev_model_memories {
                device.free_memory(mem, None);
            }
        }
    }
}

// Per-frame-in-flight CPU/GPU synchronization primitives, grouped off the flat
// `VkContext` field soup. `image_available` + `in_flight` are one-per-frame-in-
// flight (`frames_in_flight` deep); `render_finished` is one-per-swapchain-image
// (its length tracks the swapchain, so a resize rebuilds it). The ring cursor
// (`current_frame`) and depth (`frames_in_flight`) stay flat on `VkContext`:
// they are read pervasively and are frame-pacing counters, not sync handles.
pub(super) struct VkFrameSync {
    // Signalled by `acquire_next_image`, waited on by that frame's submit.
    pub(super) image_available: Vec<vk::Semaphore>,
    // Signalled by the frame's submit, waited on by its present. Indexed by
    // swapchain image, so one per swapchain image (not per frame-in-flight).
    pub(super) render_finished: Vec<vk::Semaphore>,
    // Per-frame-in-flight submission fence; gates reuse of that slot's
    // resources (command buffers, mapped UBOs, per-pass pools).
    pub(super) in_flight: Vec<vk::Fence>,
}

impl VkFrameSync {
    // Destroy every owned semaphore + fence. Called from `VkContext::drop`
    // after `wait_idle`, so none are still in flight.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            for &s in &self.image_available {
                device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished {
                device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                device.destroy_fence(f, None);
            }
        }
    }
}

// Per-frame command pools + buffers, grouped off the flat `VkContext` field
// soup. Each frame's submission splits into three tiers: a "start" outer buffer
// (leading timestamp), one buffer per render-graph pass recorded in parallel
// (each from its own externally-synchronized pool), and an "end" outer buffer
// (Composite + post-graph work). `command_pool` also doubles as the shared
// one-shot pool for upload / layout-transition submits during resource
// creation. DX keeps the analogous allocators / lists flat on `DxContext`, so
// there is no DX sub-struct to mirror here.
pub(super) struct VkCommands {
    // Shared pool: allocates the per-frame "end" buffers below AND backs every
    // one-shot upload / layout-transition submit during resource creation.
    pub(super) command_pool: vk::CommandPool,
    // Per-frame outer "end" command buffer (one per frame-in-flight). Carries
    // the Composite pass + the inline end-of-frame Hi-Z build + the
    // shadow-cascade reset + the trailing timestamp. Submitted last in the
    // per-frame batch.
    pub(super) command_buffers: Vec<vk::CommandBuffer>,
    // Per-frame outer "start" command buffer (one per frame-in-flight): just
    // the leading timestamp-pool reset + TOP_OF_PIPE write. Submitted first so
    // the timestamp brackets the whole frame. From its own pool (timestamp
    // reset must precede every pass).
    pub(super) start_command_pools: Vec<vk::CommandPool>,
    pub(super) start_command_buffers: Vec<vk::CommandBuffer>,
    // Per-(frame, pass) command pools + primary command buffers for parallel
    // command-buffer recording: each non-composite render-graph pass records
    // into its own buffer on a `jobs::pool()` worker, then the whole frame is
    // submitted in graph order as one `vkQueueSubmit`. Vulkan command pools are
    // externally synchronized, so each (frame, pass) slot owns its own pool;
    // no two workers ever touch the same pool. Length `frames_in_flight *
    // PASS_COUNT`, indexed `frame_idx * PASS_COUNT + pass_id as usize`. Mirrors
    // the DirectX `pass_allocators` / `pass_cmd_lists` pool.
    pub(super) pass_command_pools: Vec<vk::CommandPool>,
    pub(super) pass_command_buffers: Vec<vk::CommandBuffer>,
}

impl VkCommands {
    // Destroy every command pool, which frees the buffers allocated from it.
    // Called from `VkContext::drop` after `wait_idle`.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            // The shared pool (also frees the per-frame "end" buffers).
            device.destroy_command_pool(self.command_pool, None);
            // The parallel-recording pools (each frees its own buffer).
            for &pool in self
                .start_command_pools
                .iter()
                .chain(self.pass_command_pools.iter())
            {
                device.destroy_command_pool(pool, None);
            }
        }
    }
}

// The main-pass view + light uniform buffers, grouped off the flat `VkContext`
// field soup. `view_ubo_*` is one host-mapped buffer per frame-in-flight (the
// per-frame `ViewUniforms` write in `record_frame`); `light_ubo` is a single
// buffer uploaded once at init and bound into the object descriptor sets. NOTE
// the field names collide with the per-pass resource structs (decal / glass /
// raymarch / particle / gbuffer each own their own `view_ubo_*`), so accesses
// are always anchored on the `self.<field>` form, never a bare leading-dot.
pub(super) struct VkUniforms {
    // Per-frame-in-flight `ViewUniforms` UBO (camera + IBL params), persistently
    // mapped. `record_frame` memcpys this frame's view into `view_ubo_ptrs`.
    pub(super) view_ubo_buffers: Vec<vk::Buffer>,
    pub(super) view_ubo_memories: Vec<vk::DeviceMemory>,
    pub(super) view_ubo_ptrs: Vec<*mut u8>,
    // Per-frame-in-flight `ProbeSet` UBO (reflection-probe count + per-probe
    // parallax boxes), bound at global set 0 binding 7, persistently mapped.
    // `record_frame` memcpys `self.probe_set` into `probe_set_ubo_ptrs` each
    // frame; it stays `EMPTY` (count 0 = sky reflection) until a probe bakes.
    pub(super) probe_set_ubo_buffers: Vec<vk::Buffer>,
    pub(super) probe_set_ubo_memories: Vec<vk::DeviceMemory>,
    pub(super) probe_set_ubo_ptrs: Vec<*mut u8>,
    // Single `LightUniforms` UBO, uploaded once at init and bound into every
    // object descriptor set.
    pub(super) light_ubo: vk::Buffer,
    pub(super) light_ubo_memory: vk::DeviceMemory,
    // CPU-side copy of the values in `light_ubo`, kept so a live Ambient-slider
    // change can mutate `ambient_intensity` and re-upload. The light UBO is a
    // single (not per-frame) buffer, so `set_ambient_intensity` `wait_idle`s
    // before the rewrite to avoid racing an in-flight read; ambient changes only
    // on a slider drag, so the stall is rare.
    pub(super) light_uniforms: crate::gfx::render_types::LightUniforms,
}

impl VkUniforms {
    // Destroy the per-frame view UBOs (unmapping first) + the light UBO. Called
    // from `VkContext::drop` after `wait_idle`.
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            for (&buf, &mem) in self
                .view_ubo_buffers
                .iter()
                .zip(self.view_ubo_memories.iter())
            {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            for (&buf, &mem) in self
                .probe_set_ubo_buffers
                .iter()
                .zip(self.probe_set_ubo_memories.iter())
            {
                device.unmap_memory(mem);
                device.destroy_buffer(buf, None);
                device.free_memory(mem, None);
            }
            device.destroy_buffer(self.light_ubo, None);
            device.free_memory(self.light_ubo_memory, None);
        }
    }
}

//  Public struct

pub struct VkContext {
    // Vulkan core
    pub(super) instance: ash::Instance,
    pub(super) device: Device,
    pub(super) physical_device: vk::PhysicalDevice,
    pub(super) surface: vk::SurfaceKHR,
    pub(super) surface_loader: ash::khr::surface::Instance,
    pub(super) graphics_queue: vk::Queue,
    pub(super) present_queue: vk::Queue,
    pub(super) graphics_family: u32,

    // Swapchain
    pub(super) swapchain_loader: ash::khr::swapchain::Device,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) swapchain_images: Vec<vk::Image>,
    pub(super) swapchain_image_views: Vec<vk::ImageView>,
    pub(super) swapchain_format: vk::Format,
    pub(super) swapchain_extent: vk::Extent2D,
    // Resolution the 3D scene is rendered at. Equals `swapchain_extent` unless
    // temporal upscaling is active, in which case it is
    // `round(swapchain_extent * upscale_scale)` and an FSR pass reconstructs the
    // swapchain-resolution image. Every off-screen scene pass (main, velocity,
    // SSR, SSAO, decals, fog, raymarch, glass, particles, auto-exposure, Hi-Z)
    // sizes its targets + viewports to this; bloom / composite / swapchain stay
    // at `swapchain_extent` (display resolution).
    pub(super) render_extent: vk::Extent2D,
    // Index of the most recently presented swapchain image, or `None` before
    // the first present / right after a swapchain rebuild. The `screenshot`
    // debug command reads this image back; `None` makes a too-early capture a
    // clean error instead of reading an unrendered image.
    pub(super) last_present_index: Option<u32>,

    // Render passes
    pub(super) main_render_pass: vk::RenderPass,
    // Composite (post-process) pass: tonemaps the HDR resolve image onto the
    // swapchain. The text overlay also draws here, post-tonemap.
    pub(super) composite_render_pass: vk::RenderPass,

    // Multisampling
    pub(super) msaa_samples: vk::SampleCountFlags,

    // Off-screen HDR attachments, one set per frame-in-flight slot (indexed by
    // `current_frame`). The main pass renders into these; the composite pass
    // samples `hdr_resolve_images`.
    pub(super) color_images: Vec<GpuImage>, // MSAA HDR colour; empty when msaa == 1
    pub(super) depth_images: Vec<GpuImage>, // MSAA depth
    pub(super) hdr_resolve_images: Vec<GpuImage>, // single-sample HDR resolve target

    // Main-pass framebuffers (one per frame-in-flight slot): HDR colour +
    // depth (+ resolve when multisampled).
    pub(super) framebuffers: Vec<vk::Framebuffer>,
    // Composite-pass framebuffers (one per swapchain image): the swapchain
    // backbuffer the composite pass writes to.
    pub(super) composite_framebuffers: Vec<vk::Framebuffer>,

    // Cascaded shadow map + its pipelines, framebuffers, UBO, and sampler.
    pub(super) shadow: VkShadow,

    // Scene textures
    pub(super) textures: Vec<GpuImage>,
    pub(super) normal_map_textures: Vec<GpuImage>,
    pub(super) text_atlas_textures: Vec<GpuImage>,

    // Samplers
    pub(super) linear_sampler: vk::Sampler,
    pub(super) text_sampler: vk::Sampler,

    // Pipelines
    pub(super) main_pipeline: vk::Pipeline,
    pub(super) main_pipeline_layout: vk::PipelineLayout,
    // GPU-driven cull + bindless static main pass + two-pass Hi-Z occlusion
    // (pyramid + temporal state). See `VkCull`.
    pub(super) cull: VkCull,

    pub(super) text_pipeline: Option<vk::Pipeline>,
    pub(super) text_pipeline_layout: vk::PipelineLayout,

    // Composite (post-process) pipeline.
    pub(super) composite_pipeline: vk::Pipeline,
    pub(super) composite_pipeline_layout: vk::PipelineLayout,
    pub(super) composite_set_layout: vk::DescriptorSetLayout,
    // Per-frame-slot descriptor sets binding the matching HDR resolve image
    // (binding 0), bloom mip 0 (binding 1), and the 3D colour LUT (binding 2).
    pub(super) composite_sets: Vec<vk::DescriptorSet>,
    // Linear-clamp sampler the composite + bloom shaders read HDR images with.
    // The 3D colour LUT is sampled through the same sampler.
    pub(super) composite_sampler: vk::Sampler,
    // 3D colour-grading LUT sampled in the composite pass. Holds the declared
    // `ColorLut` payload, or a 2x2x2 identity LUT when the world declares none.
    // Resolution-independent, so it is never rebuilt on swapchain resize.
    pub(super) color_lut: GpuImage,

    // Bloom chain. The mips, framebuffers, and input descriptor sets are all
    // per-frame-in-flight slot (outer Vec): concurrent slots must not share a
    // bloom target. Render passes / pipelines / layouts are slot-agnostic.
    pub(super) bloom_write_pass: vk::RenderPass,
    pub(super) bloom_blend_pass: vk::RenderPass,
    pub(super) bloom_pipeline_prefilter: vk::Pipeline,
    pub(super) bloom_pipeline_downsample: vk::Pipeline,
    pub(super) bloom_pipeline_upsample: vk::Pipeline,
    pub(super) bloom_pipeline_layout: vk::PipelineLayout,
    pub(super) bloom_set_layout: vk::DescriptorSetLayout,
    pub(super) bloom_descriptor_pool: vk::DescriptorPool,
    // `[frame][mip]`, largest mip first; `mip 0` is half the HDR resolution.
    pub(super) bloom_mips: Vec<Vec<GpuImage>>,
    // Resolution of each mip level (shared across frame slots).
    pub(super) bloom_mip_extents: Vec<vk::Extent2D>,
    // `[frame][mip]` framebuffers for the DONT_CARE-load write pass.
    pub(super) bloom_write_framebuffers: Vec<Vec<vk::Framebuffer>>,
    // `[frame][mip]` framebuffers for the LOAD additive-blend pass; one fewer
    // entry than `bloom_write_framebuffers` (the smallest mip is never
    // upsampled into).
    pub(super) bloom_blend_framebuffers: Vec<Vec<vk::Framebuffer>>,
    // `[frame][input]` sets: input 0 binds the HDR resolve image, input
    // `1 + m` binds bloom mip `m`.
    pub(super) bloom_input_sets: Vec<Vec<vk::DescriptorSet>>,
    // Post-process tunables (bloom intensity / threshold / knee, exposure,
    // vignette). Drives whether the bloom passes run and feeds the composite
    // + bloom-prefilter push constants.
    pub(super) post_process: crate::gfx::render_types::PostProcessParams,

    // Temporal anti-aliasing resources. `Some` only when the world's
    // `PostProcessConfig` set `taa: true`; `None` skips the velocity pre-pass
    // and history resolve entirely (and the projection jitter with them).
    // Also forced `Some` when temporal upscaling is on (FSR consumes the
    // velocity pre-pass's motion + depth), in which case the TAA *resolve* is
    // dropped from the graph and `Upscale` runs in its slot.
    pub(super) taa: Option<TaaResources>,

    // Temporal upscaling (FSR / DLSS / XeSS, behind `VkUpscaleBackend`). `Some`
    // only when the world's `PostProcessConfig` set `temporal_upscaling: true`
    // AND a backend resolved + built; `None` renders at native resolution
    // (`render_extent == swapchain_extent`). When `Some`, the scene renders at
    // the reduced `render_extent` and this pass reconstructs the swapchain
    // resolution; bloom + composite sample its output.
    pub(super) upscale: Option<Box<dyn VkUpscaleBackend>>,

    // The upscaler backend the world requested (`PostProcessConfig.upscale_backend`).
    // Kept so a swapchain resize rebuilds the same backend via `build_upscaler`
    // (the DLSS / XeSS device extensions are fixed at device creation, so the
    // resize must re-resolve to the same first choice; it does, deterministically).
    pub(super) upscale_requested: crate::assets::UpscalerBackend,

    // Screen-space ambient occlusion (GTAO) resources. `Some` only when the
    // world's `PostProcessConfig` set `ssao: true`; `None` binds the
    // `ssao_white` 1×1 fallback at set 0 binding 6 so the main pass's SSAO
    // multiplier is a constant 1.0.
    pub(super) ssao: Option<SsaoResources>,
    // 1×1 white fallback bound at set 0 binding 6 when SSAO is off.
    pub(super) ssao_white: GpuImage,

    // Backing store for the render graph's transient images (the resources the
    // aliasing planner manages). Owns each managed transient's image + memory;
    // features read them back by label and the executor's barrier registry
    // resolves them the same way. Rebuilt on swapchain resize. Today it manages
    // `ao_output`; the set grows as more transients migrate off their feature
    // structs.
    pub(super) transient_pool: super::transient_pool::TransientImagePool,

    // Screen-space reflections. `Some` when the world's `PostProcessConfig`
    // set `ssr: true` *or* selected `indirect_lighting: ssgi` (SSGI reuses the
    // SSR depth + normal pre-pass G-buffer, so the pre-pass half is built
    // whenever either is on). When on, the bloom prefilter / composite / TAA
    // scene input descriptors are re-pointed at `SsrResources::output` only when
    // `ssr_resolve_active` is true (a SSGI-only build leaves the resolve off).
    pub(super) ssr: Option<SsrResources>,
    // True when the SSR *resolve* (the reflection compositing half) should run
    // and own the post-stack scene image. False for a SSGI-only build, where
    // `ssr` exists for the G-buffer but the post stack samples `hdr_resolve`
    // directly (SSGI has already composited into it). Mirrors DirectX's
    // `scene_srv_for_post` gating on `s.resolve.as_ref()`.
    pub(super) ssr_resolve_active: bool,

    // Roughness-aware reflection composite. `Some` whenever a reflection path owns
    // the post-stack scene image (the SSR resolve is active OR RT reflections are
    // active). Both resolves write reflected radiance + weight into their output
    // target, then this blurs by roughness and composites over the scene into
    // `reflection_composite.output` -- the scene image the post stack consumes in
    // place of the raw resolve output. Mirrors `DxContext::reflection_composite`.
    pub(super) reflection_composite: Option<ReflectionCompositeResources>,

    // Screen-space global illumination. `Some` only when the world's
    // `PostProcessConfig` selected `indirect_lighting: ssgi`. The gather +
    // composite run on the hdr_resolve RMW chain after the main pass, reusing
    // `ssr`'s pre-pass G-buffer.
    pub(super) ssgi: Option<SsgiResources>,

    // Unified geometry G-buffer pre-pass. `Some` whenever any screen-space
    // consumer of the merged buffer is on (SSR resolve OR SSGI OR RT OR SSAO OR
    // velocity for TAA / upscale): one jittered traversal rasterises the
    // normal+depth / roughness / velocity MRT every reader samples, replacing
    // the separate SSR / SSAO / velocity pre-passes (the `PassId::GBufferPrepass`
    // node). Mirrors `DxContext::gbuffer`.
    pub(super) gbuffer: Option<GbufferResources>,

    // Hardware ray-traced reflections (`VK_KHR_ray_query`). `rt_reflections` (the
    // fullscreen inline-`rayQueryEXT` pass + its output target) and `rt_accel`
    // (the scene BLAS / TLAS + geometry table) are both `Some` only when the
    // world set `ray_traced_reflections: true`, the GPU exposed the ray-query
    // extensions, and the acceleration-structure build succeeded; otherwise both
    // stay `None` and the graph falls back to `SsrResolve`. Like SSGI, RT reuses
    // the SSR depth + normal + roughness pre-pass G-buffer (so `ssr` is built
    // whenever RT is on), and it replaces the SSR *resolve* in the frame graph:
    // when `rt_reflections_active()` the post stack samples `rt_reflections.output`
    // (RT takes precedence over SSR, which stays the non-RT-GPU fallback).
    pub(super) rt_reflections: Option<RtReflectionsResources>,
    pub(super) rt_accel: Option<crate::vulkan::raytrace::RtAccelData>,
    // How the TLAS is kept current when props move (`CN_RT_DYNAMIC`); read by the
    // per-frame `rt_dynamic_update`. Inert when `rt_accel` is `None`.
    pub(super) rt_dynamic_mode: crate::vulkan::raytrace::RtDynamicMode,
    // Whether the device is RT-capable (the ray-query extensions + features were
    // enabled at device creation, and XeSS is not active). Enabled whenever
    // capable -- independent of whether RT is on at launch -- so a live
    // `apply_quality_settings` toggle can bring RT up at runtime (a device
    // extension cannot be enabled after `create_device`). Read by `upload_skinned`
    // to add the AS-build / storage / device-address flags to the skinned VB/IB
    // whenever capable (mirroring how the static VB/IB gate their RT flags at
    // init), and by the RT toggle to reject an enable on an incapable device.
    pub(super) rt_capable: bool,
    // Total static vertices uploaded at init (the shared VB element count). The
    // acceleration-structure build needs it to size the hit-shader vertex SSBO;
    // there is no separate count field, so it is captured here for a live RT
    // build. Static-geometry rebuilds are not reflected (a pre-existing RT
    // topology limitation).
    pub(super) rt_static_vertex_count: usize,

    // Projected decals. `decals_state` (pipeline + unit-cube buffers +
    // per-frame uniforms + per-decal albedo sets) is always built so
    // runtime `add_decal` works from a world that started empty; the
    // encoder simply skips when every slot is `None` or every live
    // decal culls. `decals` and `decal_free_slots` mirror Metal /
    // DirectX's freelist pattern so id reuse stays bounded.
    pub(super) decals_state: Option<crate::vulkan::decal::DecalResources>,
    pub(super) decals: Vec<Option<crate::gfx::decal::DecalRecord>>,
    pub(super) decal_free_slots: Vec<usize>,

    // Volumetric fog. `Some` only when the world declared a `VolumetricFog`
    // asset; with none, both fields stay `None` and the fog pass is skipped
    // entirely. The settings are cached so the per-frame encoder can build
    // its `FogParams` without re-resolving the asset. `fog_sun_dir` /
    // `fog_sun_color` mirror the first directional light captured at init:
    // the Vulkan backend uploads `LightUniforms` once, so the sun is fixed.
    pub(super) fog_resources: Option<crate::vulkan::fog::FogResources>,
    pub(super) fog_settings: Option<crate::gfx::volumetric_fog::FogSettings>,
    pub(super) fog_sun_dir: [f32; 3],
    pub(super) fog_sun_color: [f32; 3],

    // Raymarched SDF volumes. `Some` only when the world declared at least one
    // `SdfVolume` whose `fragment_shader` is a `.glsl` payload; the `Raymarch`
    // pass is omitted from the frame graph otherwise. Built at init; the encoder
    // composites each visible volume into the scene between `AutoExposure` and
    // `Decals`. While present, the main pass switches to a STORE-colour render
    // pass (MSAA) so this pass can load + re-resolve the multisampled colour.
    pub(super) raymarch: Option<crate::vulkan::raymarch::RaymarchResources>,

    // Translucent glass panels: the generic producer for the shared
    // `PassId::Transparent` slot. `Some` only when the world declared any
    // `GlassPanel`; with none the field stays `None` and the transparent pass
    // is omitted from the frame graph (gated on `glass.any_visible()`). Built
    // at init; the encoder draws the panels back-to-front over the post-SSR
    // scene between `SsrResolve` and `TaaResolve`. Water is a separate
    // (Metal-only) producer not ported here. Mirrors `src/directx/glass.rs`.
    pub(super) glass: Option<crate::vulkan::glass::GlassResources>,

    // Planar reflections for flat glass panes: one render-resolution mirror render
    // per distinct reflector plane, sampled projectively by the glass pass. `Some`
    // only when the world declared glass panes assigned to a planar slot. Mirrors
    // `src/directx/planar.rs`.
    pub(super) planar_reflection: Option<crate::vulkan::planar::PlanarReflectionSet>,

    // Resolved swapchain colour-output mode, selected when the world's
    // `PostProcessConfig.hdr_display` was on AND the surface advertised a
    // matching HDR colour space via the `VK_EXT_swapchain_colorspace` instance
    // extension. Two HDR flavours: `HdrEncoding::ExtendedLinear` runs the
    // swapchain in `R16G16B16A16_SFLOAT` + `EXTENDED_SRGB_LINEAR_EXT` (scRGB
    // linear) and the composite emits linear extended-range values;
    // `HdrEncoding::Pq` (requested via `hdr_pq`, only when an `HDR10_ST2084_EXT`
    // pair is advertised) runs the swapchain in that colour space and the
    // composite PQ-encodes (SMPTE ST 2084) in-shader. On SDR the swapchain runs
    // in `BGRA8_UNORM` + sRGB-nonlinear and the ACES + gamma + FXAA + LUT path
    // runs unchanged. Mirrors `DxContext::hdr_mode`. Stored so the swapchain
    // rebuild path preserves the format + colour space on resize.
    pub(super) hdr_mode: crate::gfx::hdr_output::HdrOutputMode,

    // GPU-compute particle system. `particle_resources` (pipelines +
    // per-frame view UBO + descriptor pool + framebuffers) is built only
    // when the world declared at least one `ParticleEmitter` (or when
    // runtime `add_particle_emitter` fires); the encoder is a no-op
    // otherwise. `particles` and `particle_emitter_state` mirror Metal /
    // DirectX's parallel-vec freelist pattern so id reuse stays bounded.
    // `particle_last_elapsed` + `particle_frame_index` live in `Cell`s
    // because `encode_particles` is reached through `&self` from the
    // graph executor (per-frame mutable state has to be interior-mut).
    pub(super) particle_resources: Option<crate::vulkan::particle::ParticleResources>,
    pub(super) particles: Vec<Option<crate::gfx::particles::ParticleEmitterRecord>>,
    pub(super) particle_emitter_state:
        Vec<Option<crate::vulkan::particle::ParticleEmitterGpuState>>,
    pub(super) particle_free_slots: Vec<usize>,
    pub(super) particle_last_elapsed: std::cell::Cell<f32>,
    pub(super) particle_frame_index: std::cell::Cell<u32>,

    // Auto-exposure (EV adaptation) resources. `Some` only when
    // `PostProcessConfig.auto_exposure` is enabled. Holds the build +
    // average compute pipelines, histogram + output buffers, and the
    // per-frame readback buffers. `auto_exposure_state` carries the EMA
    // target; `auto_exposure_settings` carries the clamped tunables;
    // `auto_exposure_bias_ev` is the authored EV bias added to the target.
    // `auto_exposure_last_elapsed` is the previous frame's elapsed time
    // used to derive `dt` for the EMA.
    pub(super) auto_exposure: Option<crate::vulkan::auto_exposure::AutoExposureResources>,
    pub(super) auto_exposure_settings: Option<crate::gfx::auto_exposure::AutoExposureSettings>,
    pub(super) auto_exposure_state: Option<crate::gfx::auto_exposure::AutoExposureState>,
    pub(super) auto_exposure_bias_ev: f32,
    pub(super) auto_exposure_last_elapsed: f32,

    // True only under `cn debug`. Routes every built-in GLSL source resolve
    // through `pipeline::shader_source`'s disk-first path and gates the
    // `vulkan/shaders/` filesystem watcher. False under `cn run`: the
    // `include_str!`-baked GLSL is the only source the binary ever sees.
    // Mirrors `DxContext::hot_reload` / `MtlContext::hot_reload`.
    pub(super) hot_reload: bool,
    // Atomic flag set by the `notify` filesystem watcher or the debug WS
    // `reload-shaders` command. Polled at the top of `draw_frame`; when set,
    // [`VkContext::reload_shaders`] rebuilds every built-in pipeline before
    // the next frame's passes run. `Some` only when `hot_reload` is on.
    pub(super) shader_reload_pending: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    // Live `notify` watcher handle; dropping it stops the watcher. `Some`
    // only when `hot_reload` is on. Held purely for lifetime: the watcher
    // pushes events into `shader_reload_pending` directly.
    #[allow(dead_code)]
    pub(super) shader_watcher: Option<crate::vulkan::hot_reload::WatcherHandle>,

    // Per-frame draw-call / VRAM / GPU-time counters surfaced to the
    // profiler overlay via [`Self::render_stats`]. Lives in a `Cell` because
    // the `objects` / `gpu_frame_us` / `vram_bytes` fields are filled from
    // `&mut self` in `draw_frame`. Mirrors `DxContext::frame_stats`.
    pub(super) frame_stats: std::cell::Cell<crate::gfx::profile::RenderStats>,
    // Draw-call accumulator the pass encoders bump via `inc_draw_calls`. An
    // `AtomicU32` (not the `frame_stats` Cell) because the parallel
    // command-buffer recording fans the encoders onto rayon workers that bump
    // it concurrently; a `Cell` would be a data race. Reset to 0 at the top of
    // `draw_frame` and drained into `frame_stats.draw_calls` at the end of
    // `record_frame`. Mirrors `DxContext::draw_calls_accum`.
    pub(super) draw_calls_accum: std::sync::atomic::AtomicU32,

    // Timestamp query pool with `2 * frames_in_flight` slots (one start +
    // end pair per in-flight frame). `record_frame` issues
    // `cmd_write_timestamp` at the top and bottom of recording; the CPU
    // reads the previous trip's pair at the top of `draw_frame` after the
    // matching fence wait. `None` when the queue does not expose
    // timestamps; `gpu_frame_us` then stays 0.
    pub(super) timestamp_query_pool: Option<vk::QueryPool>,
    // `timestamp_period` from the physical device, in nanoseconds per tick.
    // Combined with the resolved tick delta to derive microseconds.
    pub(super) timestamp_period_ns: f32,

    // `VK_EXT_memory_budget` device-local heap indices summed for the
    // VRAM-residency chip. Empty when the extension is unavailable; the
    // chip then reports 0.
    pub(super) device_local_heaps: Vec<u32>,
    // `true` when [`Self::device_local_heaps`] should be queried via
    // `VK_EXT_memory_budget`.
    pub(super) memory_budget_supported: bool,

    // Main geometry-path descriptor layouts, shared pool, and per-frame sets.
    // See `VkDescriptors`.
    pub(super) descriptors: VkDescriptors,
    // Instanced-prop pipeline, per-cluster material sets, per-frame instance
    // buffers + sets, and the cluster list. See `VkInstanced`.
    pub(super) instanced: VkInstanced,

    // Shared static vertex/index buffers plus their byte-range sub-allocators.
    // See `VkGeometry`.
    pub(super) geometry: VkGeometry,

    // Streamed VoxelWorld chunk rendering: headroom sub-allocators, the chunk
    // draw-slot freelist, the shared chunk descriptor pool + set, and the
    // material slots it samples. See `VkChunkStream`.
    pub(super) chunk_stream: VkChunkStream,
    // Build-time `draw_objects` count. Streamed chunks are appended past this,
    // so a draw index >= `n_objects` identifies a chunk -- which binds the
    // shared `chunk_object_set` rather than a per-object descriptor set.
    pub(super) n_objects: usize,

    // Instanced-cluster instances folded into the GPU-driven bindless cull
    // buffers as per-object `GpuObjectData` records after the `n_objects`
    // static records (so the cull kernel tests each instance independently).
    // 0 when the world has no instanced props or the bindless pass is inactive.
    // `cull_count() == n_objects + n_instances`. See `gfx::render_types`.
    pub(super) n_instances: usize,

    // Streamed-chunk record reserve folded into the GPU-driven bindless cull
    // buffers BETWEEN the instances and the skinned tail: the buffers reserve
    // `[n_objects + n_instances, +n_chunk)` at init (capacity = the worst-case
    // resident chunk window). Resident chunks pack into this region each frame and
    // are drawn by the static+instance prefix indirect draw (chunk geometry already
    // lives in the shared VB/IB); the unused tail is disabled. Fixed at init, 0 for
    // a non-voxel world. Mirrors `DxContext.n_chunk`.
    pub(super) n_chunk: usize,

    // Skinned draw objects folded into the GPU-driven bindless cull buffers as
    // `GpuObjectData` / `GpuDrawArgs` records after the instance records (at
    // `n_objects + n_instances + k`), drawn as rigid deformed geometry by the main
    // pass's 2nd indirect draw against the per-frame deformed-vertex buffer. The
    // cull buffers reserve these slots at init (capacity threaded through `new`);
    // this count is set in `upload_skinned` once the skin fold is built, so it
    // stays 0 (and `cull_count()` excludes the reserved tail) when no skinned mesh
    // loads or the bindless pass is inactive.
    pub(super) n_skinned: usize,

    // Pre-allocated descriptor pool for `clone_static_draw_object`. Holds up
    // to `MAX_CLONE_DRAWS` per-object (albedo, normal) sets so an asset
    // hot-reload that adds a new authored Prop referencing an existing
    // mesh / model can wire its descriptors without growing any other
    // pool. Built at init (along with `clone_object_sets`), regardless of
    // whether any clone exists yet. Mirrors DirectX's `clone_srv_base_slot`
    // reservation.
    pub(super) clone_descriptor_pool: Option<vk::DescriptorPool>,
    // Per-clone (albedo, normal) descriptor sets, indexed by clone offset.
    // Empty until the first `clone_static_draw_object` runs; bounded by
    // `MAX_CLONE_DRAWS` from `gfx::clone::MAX_CLONE_DRAWS`. A set whose offset is
    // in `clone_free_offsets` is allocated but unreferenced, ready for the next
    // clone to reuse (re-pointed only if its textures differ).
    pub(super) clone_object_sets: Vec<vk::DescriptorSet>,
    // Clone offsets vacated by a retired clone, popped by the next
    // `clone_static_draw_object` so steady spawn/despawn churn reuses descriptor
    // sets instead of exhausting the `MAX_CLONE_DRAWS` pool.
    pub(super) clone_free_offsets: Vec<usize>,
    // `draw_idx → clone_offset` lookup the legacy main pass uses to pick
    // the right per-clone descriptor set when drawing an entry past
    // `n_objects` (chunks fall through to `chunk_object_set` instead).
    pub(super) clone_slot_by_draw_idx: std::collections::HashMap<usize, usize>,
    // Texture-pool slot each clone samples, parallel to
    // `clone_object_sets`. Read by `rewrite_albedo_slot` so a streamed
    // albedo swap repoints the matching clone sets.
    pub(super) clone_texture_slots: Vec<usize>,
    // Normal-map pool slot each clone samples, parallel to
    // `clone_object_sets`. Read by `rewrite_normal_slot` for the same
    // reason as `clone_texture_slots`.
    pub(super) clone_normal_map_slots: Vec<usize>,

    // Skinned (skeletally animated) mesh rendering. See `VkSkinned`.
    pub(super) skinned: VkSkinned,
    // Free pool for the pre-reserved skinned instance slots a runtime skinned
    // spawn claims. Seeded once from `seed_skinned_instance_pool` with the hidden
    // bind-pose copies `upload_skinned` uploaded; empty for a world with no
    // skinned mesh opting into runtime spawning.
    pub(super) skinned_pool: crate::gfx::skinned_pool::SkinnedInstancePool,

    // Main-pass view (per-frame) + light (shared) uniform buffers. See
    // `VkUniforms`.
    pub(super) uniforms: VkUniforms,

    // Per-frame-in-flight synchronization primitives. See `VkFrameSync`.
    pub(super) frame_sync: VkFrameSync,
    pub(super) current_frame: usize,
    pub(super) frames_in_flight: usize,
    // Lock presentation to the display refresh. Captured so `rebuild_swapchain`
    // re-selects the same present mode (FIFO vsync vs MAILBOX uncapped) on resize.
    pub(super) vsync: bool,

    // Per-frame command pools + buffers (start / per-pass / end tiers + the
    // shared one-shot pool). See `VkCommands`.
    pub(super) commands: VkCommands,

    // Draw state
    pub(super) draw_objects: Vec<DrawObject>,
    pub(super) cull_bvh: crate::gfx::bvh::Bvh,
    pub(super) always_draw: Vec<u32>,
    // Parallel to `draw_objects`: true where that slot is a member of
    // `always_draw`, so `ensure_always_draw` adds a recycled slot at most once.
    pub(super) always_draw_member: Vec<bool>,
    // Free-list allocator over `draw_objects` slots. `retire_draw_object` /
    // `remove_chunk_mesh` push a vacated slot; `clone_static_draw_object` /
    // `add_chunk_mesh` pop one before growing the vec, so runtime spawn/despawn
    // and chunk streaming reuse slots instead of leaking them. Indices stay
    // stable (RenderHandle stores raw indices into draw_objects), so this is a
    // free-list, never a compaction.
    pub(super) draw_slots: crate::gfx::draw_slot::DrawSlotAllocator,
    // Per-frame scratch for the legacy CPU draw path's visible set
    // (BVH-culled cullables + always_draw fallback). `mem::take`d at the
    // top of record_frame and returned at the bottom so the heap allocation
    // is reused across frames instead of `Vec::with_capacity`'d each tick.
    pub(super) visible_scratch: Vec<u32>,
    pub(super) clear_color: [f32; 4],
    pub(super) view_matrix: [[f32; 4]; 4],
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = no
    // EnvironmentMap declared; the fragment shader uses this as the IBL
    // on/off signal and falls back to the legacy ambient path.
    pub(super) prefilter_mip_count: u32,
    // Cube sampler shared by the IBL irradiance + prefilter cube bindings.
    // Held here so Drop can destroy it after the device idles.
    pub(super) cube_sampler: vk::Sampler,
    // Owned IBL cube textures. Live for the lifetime of the context.
    pub(super) env_map: EnvironmentMapTextures,

    // Reflection-probe placements (declared `ReflectionProbe`s or an auto-seeded
    // grid), supplied once after construction via `set_reflection_probes`. The
    // cube capture that bakes one prefiltered cube per placement runs across
    // later frames (next slice); held here so that capture can walk them.
    #[allow(dead_code)] // consumed by the probe capture pass (next slice).
    pub(super) probe_placements: Vec<crate::gfx::reflection_probe::ProbePlacement>,
    // The probe set (count + per-probe parallax boxes) bound to the forward /
    // SSR / RT shaders. `EMPTY` (count 0 = sky reflection) until the staggered
    // capture bakes cubes and installs them; each install bumps the count.
    pub(super) probe_set: super::probe_uniforms::ProbeSet,
    // Baked reflection-probe prefilter cubes, one per installed probe, parallel
    // to `probe_set.probes[..probe_set.count]`. Distinct from `env_map`; sampled
    // only by the specular reflection term once the capture installs them. Grows as
    // the staggered bake installs each probe. Destroyed in `Drop`.
    pub(super) probe_maps: Vec<GpuImage>,

    // Staggered asynchronous probe-bake state, driven each frame by
    // `bake_pending_probes` (the shared `reflection_probe::next_bake_action`
    // transition table). `probe_bake_queue` hands out placements in order; at most one
    // probe is `probe_rendering` (six faces submitting one per frame, on per-face
    // fences) and one `probe_converting` (its faces read back, the prefilter
    // convolution running off the render thread). Mirrors DirectX / Metal.
    pub(super) probe_bake_queue: crate::gfx::reflection_probe::ProbeBakeQueue,
    pub(super) probe_rendering: Option<super::probe::RenderingBake>,
    pub(super) probe_converting: Option<super::probe::ConvertingBake>,

    // Deferred buffer destruction (text transient buffers)
    pub(super) deferred_destroy: RefCell<Vec<DeferredBuffer>>,

    // Window + input
    pub(super) window: crate::vulkan::window::GlfwWindow,

    // Optional validation debug messenger
    pub(super) debug_utils: Option<ash::ext::debug_utils::Instance>,
    pub(super) debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    // Budget of benign DLSS first-frame layout errors the messenger callback
    // drops (the messenger holds a raw pointer to this; it must outlive the
    // messenger, which `Drop` destroys before fields). `None` without validation.
    pub(super) debug_filter: Option<Box<std::sync::atomic::AtomicU32>>,

    // Keep Entry alive for the lifetime of the instance
    pub(super) _entry: ash::Entry,
}

// GpuImage raw pointers (uniforms.view_ubo_ptrs) are host-mapped and used only
// on this thread; the RefCell<Vec<DeferredBuffer>> is also single-threaded.
unsafe impl Send for VkContext {}

// Thread id of the thread that built the context. `VkContext::new` runs on the
// main thread and records it here; `debug_assert_main_thread` checks every
// mutation entry point against it. Portable across platforms (unlike the Win32
// `GetCurrentThreadId` the DirectX backend uses) since Vulkan also targets Linux.
static MAIN_THREAD_ID: std::sync::OnceLock<std::thread::ThreadId> = std::sync::OnceLock::new();

// Record the calling thread as the main (render) thread. Called once from
// `VkContext::new`, which always runs on the main thread.
pub(super) fn record_main_thread() {
    let _ = MAIN_THREAD_ID.set(std::thread::current().id());
}

// Debug-only guard that the caller is on the main thread.
//
// The `unsafe impl Send for VkContext` above is sound only because the context
// is touched from one thread alone: the GLFW window/event pump is thread-affine,
// the host-mapped view UBO pointers and the `deferred_destroy` RefCell are
// single-threaded, and the parallel-encoder fan-out only ever shares `&self`
// read-only. The `RenderBackend` mutation entry points (reached through the
// boxed trait object) had nothing proving this, so scheduling `GraphicsSystem`
// off the main thread would silently race the window + queue submission instead
// of failing. This makes that mistake panic loudly in debug builds and compiles
// to nothing in release. `entry` is the offending method name, for the message.
// Mirrors `directx/context.rs::debug_assert_main_thread`.
#[inline]
#[track_caller]
pub(super) fn debug_assert_main_thread(entry: &str) {
    debug_assert!(
        MAIN_THREAD_ID
            .get()
            .is_none_or(|main| *main == std::thread::current().id()),
        "{entry} must be called from the main thread: VkContext is main-thread-only \
         (see `unsafe impl Send for VkContext`); driving GraphicsSystem off the main \
         thread races the GLFW window + Vulkan queue submission",
    );
}

//  Public API

impl VkContext {
    #[allow(clippy::too_many_arguments)]
    pub fn draw_frame(
        &mut self,
        elapsed: f32,
        fov_y_radians: f32,
        near: f32,
        far: f32,
        cam_pos: [f32; 3],
        text_calls: &[TextDrawCall],
        world_hidden: bool,
    ) -> Result<(), String> {
        // Shader hot-reload: if either the filesystem watcher or the debug
        // `reload-shaders` command set the flag, rebuild every built-in
        // pipeline from disk-resident source before this frame's passes
        // start using them. The flag is cleared regardless of outcome so a
        // failed rebuild (typo in a shader edit) doesn't loop, and the
        // previous pipelines stay live so the session keeps rendering.
        // Wait for the GPU to drain first so swapping pipelines out from
        // under in-flight command buffers is safe. Mirrors the DirectX
        // path at the top of its `draw_frame`.
        if self.shader_reload_requested() {
            self.clear_shader_reload_flag();
            self.wait_idle();
            match self.reload_shaders() {
                Ok(()) => tracing::info!("hot-reload: shader pipelines rebuilt"),
                Err(e) => tracing::error!("hot-reload: shader rebuild failed: {}", e),
            }
        }

        let frame = self.current_frame;
        // Cheap-cloneable handle (ash::Device is Arc-like). Holding a local
        // copy avoids tying the rest of the function to `&self.device` while
        // record_frame takes `&mut self`.
        let device = self.device.clone();
        let device = &device;

        // Wait for this frame's slot to finish.
        unsafe {
            device
                .wait_for_fences(
                    std::slice::from_ref(&self.frame_sync.in_flight[frame]),
                    true,
                    u64::MAX,
                )
                .map_err(|e| format!("wait fences: {e}"))?;
        }

        // Advance the staggered reflection-probe bake one step. Runs here -- after
        // this frame's slot fence wait, before `record_frame` -- so any cube it
        // installs (a binding-8 rewrite + `probe_set.count` bump) is picked up by this
        // frame's `record_frame` ProbeSet upload + rendering. Non-fatal.
        if let Err(e) = self.bake_pending_probes() {
            tracing::warn!("reflection probe bake step failed: {e}");
        }

        // Reset this frame's render stats. `record_frame` accumulates
        // `draw_calls` through `inc_draw_calls` (interior-mutability since
        // the encoders run through `&self`); `objects`, `gpu_frame_us`,
        // and `vram_bytes` are filled here from `&mut self` state. Mirrors
        // the DirectX `frame_stats` reset at the top of `draw_frame`.
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
        // GPU timing for the most-recently completed block on this frame slot:
        // the whole-frame pair plus one (start, end) pair per render pass. The
        // fence wait above guarantees the previous trip's writes have retired, so
        // the available query results are committed. The block is read with
        // `WITH_AVAILABILITY` so a pass that did not run this trip (its slots were
        // reset but never written) reads back unavailable -> 0, without stalling
        // the host (no `WAIT`). Zero before a slot has been visited a second time.
        let empty_pass_times = [("", 0u32); crate::gfx::profile::MAX_PASS_TIMINGS];
        let (gpu_frame_us, pass_times_us) = if let Some(pool) = self.timestamp_query_pool {
            // One [value, availability] pair per query slot (TYPE_64 +
            // WITH_AVAILABILITY -> two u64 per query; ash uses the element size as
            // the stride and the slice length as the query count).
            let mut results = vec![[0u64; 2]; super::pass_timing::SLOTS_PER_FRAME];
            let res = unsafe {
                device.get_query_pool_results(
                    pool,
                    super::pass_timing::frame_block_base(frame),
                    &mut results,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            // WITH_AVAILABILITY fills the buffer + per-query availability bits and
            // returns SUCCESS; tolerate NOT_READY defensively (the buffer is still
            // written, and the availability bits gate every read).
            if matches!(res, Ok(()) | Err(vk::Result::NOT_READY)) {
                let period = self.timestamp_period_ns;
                let pair_micros = |start_slot: usize, end_slot: usize| -> u32 {
                    let [s_val, s_avail] = results[start_slot];
                    let [e_val, e_avail] = results[end_slot];
                    if s_avail != 0 && e_avail != 0 && e_val > s_val && period > 0.0 {
                        let nanos = (e_val - s_val) as f64 * period as f64;
                        ((nanos / 1000.0) as u64).min(u32::MAX as u64) as u32
                    } else {
                        0
                    }
                };
                let frame_us = pair_micros(0, 1);
                let mut times = empty_pass_times;
                for (i, name) in crate::gfx::render_graph::PASS_NAMES.iter().enumerate() {
                    if i >= crate::gfx::profile::MAX_PASS_TIMINGS {
                        break;
                    }
                    times[i] = (*name, pair_micros(2 + 2 * i, 3 + 2 * i));
                }
                (frame_us, times)
            } else {
                (0, empty_pass_times)
            }
        } else {
            (0, empty_pass_times)
        };
        let vram_bytes = self.query_vram_bytes();
        // Reset the parallel-safe draw-call accumulator for this frame; the
        // encoders fetch_add into it during recording and `record_frame`
        // drains it back into `frame_stats.draw_calls` once recording is done.
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
            // Adapted auto-exposure EV for the StatHud `EV` chip. `Some` only
            // when the world opted into auto-exposure (the EMA state is then
            // live); the static-exposure path leaves it `None` so the chip
            // stays blank. The value is the EV the most recent
            // `update_auto_exposure` EMA step settled on (the multiplier the
            // post stack pushes is `2^ev`). Mirrors `DxContext` / `MtlContext`.
            auto_exposure_ev: self.auto_exposure_state.as_ref().map(|s| s.current_ev),
            // EDR headroom for the StatHud `EDR x.X` chip, taken from the
            // `HdrOutputMode` resolved at init. `Some` only on the HDR path
            // (Vulkan has no portable max-EDR query, so the value is the
            // synthesised placeholder set in `init`); `None` on SDR blanks the
            // chip. Mirrors `DxContext` / `MtlContext::render_stats`.
            max_edr: match self.hdr_mode {
                crate::gfx::hdr_output::HdrOutputMode::Hdr { max_edr, .. } => Some(max_edr),
                crate::gfx::hdr_output::HdrOutputMode::Sdr => None,
            },
        });

        // Destroy transient text buffers from this slot's previous submission.
        // The fence wait above guarantees that command buffer has completed,
        // so its referenced buffers are no longer in use. Buffers belonging to
        // the *other* in-flight slot may still be executing, so leave them.
        {
            let mut deferred = self.deferred_destroy.borrow_mut();
            let mut i = 0;
            while i < deferred.len() {
                if deferred[i].frame == frame {
                    deferred.swap_remove(i).destroy(device);
                } else {
                    i += 1;
                }
            }
        }

        // Acquire swapchain image.
        let acquire = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.frame_sync.image_available[frame],
                vk::Fence::null(),
            )
        };
        let image_index = match acquire {
            Ok((idx, suboptimal)) => {
                if suboptimal {
                    self.rebuild_swapchain()?;
                    return Ok(());
                }
                idx
            }
            Err(_) => {
                self.rebuild_swapchain()?;
                return Ok(());
            }
        };

        unsafe { device.reset_fences(std::slice::from_ref(&self.frame_sync.in_flight[frame])) }
            .map_err(|e| format!("reset fences: {e}"))?;

        // Record the frame. `record_frame` records the leading timestamp into
        // the `start` buffer, fans each non-composite pass onto its own
        // per-pass command buffer, and records Composite + the post-graph work
        // into `cmd` (the outer "end" buffer begun here). It returns the
        // ordered `[start, ...pass buffers]` to submit before `end`.
        let cmd = self.commands.command_buffers[frame];
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| format!("reset cmd buf: {e}"))?;
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|e| format!("begin cmd buf: {e}"))?;
        }

        let mut submit_bufs = self.record_frame(
            cmd,
            image_index,
            elapsed,
            fov_y_radians,
            near,
            far,
            cam_pos,
            text_calls,
            frame,
            world_hidden,
        )?;

        unsafe { device.end_command_buffer(cmd) }.map_err(|e| format!("end cmd buf: {e}"))?;
        // The outer "end" buffer (Composite + post-graph work + trailing
        // timestamp) submits last, after every per-pass buffer.
        submit_bufs.push(cmd);

        // Submit the whole batch in one call: submission order = GPU order on
        // the single graphics queue. The render-finished semaphore is indexed
        // by swapchain image (not frame slot) so present never reuses one still
        // in flight.
        let wait_sems = [self.frame_sync.image_available[frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_sems = [self.frame_sync.render_finished[image_index as usize]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_sems)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&submit_bufs)
            .signal_semaphores(&signal_sems);
        unsafe {
            device
                .queue_submit(
                    self.graphics_queue,
                    std::slice::from_ref(&submit_info),
                    self.frame_sync.in_flight[frame],
                )
                .map_err(|e| format!("queue submit: {e}"))?;
        }

        // Present.
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_sems)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };
        if present_result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR) || present_result == Ok(true) {
            self.rebuild_swapchain()?;
        } else {
            present_result.map_err(|e| format!("present: {e}"))?;
            // Record which swapchain image now holds a complete, presented frame
            // so the `screenshot` debug command can read it back.
            self.last_present_index = Some(image_index);
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        Ok(())
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
            if let Some(offset) = self.clone_slot_by_draw_idx.remove(&index) {
                self.clone_free_offsets.push(offset);
            }
            // Only the runtime-append region (streamed chunks + spawned clones,
            // `index >= n_objects`) recycles its draw slots. A build-time slot
            // stays allocated when hidden: the init-time cull BVH and the RT
            // acceleration structure's `object_indices` are keyed to fixed
            // build-time slot indices and cannot refit, so reusing one would
            // mis-key them. (Metal recycles build-time slots too because its
            // per-frame RT topology refresh re-admits them; Vulkan has no such
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

    // Re-point the combined-image-sampler at `binding` of `set` to `view`.
    // Shared by the texture-streaming descriptor rewrites below.
    pub fn window_closed(&mut self) -> bool {
        self.window.poll()
    }

    pub fn wait_idle(&self) {
        let _ = unsafe { self.device.device_wait_idle() };
    }

    // Render statistics for the most recent `draw_frame`, for the profiler
    // overlay. `gpu_frame_us` is filled at the top of each `draw_frame`
    // from the timestamp pair this slot resolved on its previous trip
    // through the ring (so the reading is `frames_in_flight`-stale by
    // construction, matching DirectX / Metal). Per-pass GPU timing is
    // still a follow-up.
    pub fn render_stats(&self) -> crate::gfx::profile::RenderStats {
        self.frame_stats.get()
    }

    // Current device-local memory residency in bytes, via
    // `VK_EXT_memory_budget`. Sums `heap_usage` on every DEVICE_LOCAL heap;
    // returns 0 when the extension is unavailable (so the chip degrades
    // gracefully on adapters that don't expose budgets, matching DirectX's
    // behaviour on pre-WDDM-2.0 adapters).
    pub(super) fn query_vram_bytes(&self) -> u64 {
        if !self.memory_budget_supported || self.device_local_heaps.is_empty() {
            return 0;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            self.instance
                .get_physical_device_memory_properties2(self.physical_device, &mut props2);
        }
        self.device_local_heaps
            .iter()
            .map(|&i| budget.heap_usage[i as usize])
            .sum()
    }

    // Bump this frame's CPU-issued draw-call counter. Called from each
    // draw site in the shadow, main, decal, and composite + text passes.
    // Mirrors `DxContext::inc_draw_calls`; fullscreen post-process passes
    // (SSAO, SSR, TAA, bloom, fog) are not counted per the `RenderStats`
    // doc comment.
    pub(super) fn inc_draw_calls(&self, n: u32) {
        // Bump the atomic accumulator (not the `frame_stats` Cell) so the
        // parallel-recording workers don't race. Drained into
        // `frame_stats.draw_calls` at the end of `record_frame`.
        self.draw_calls_accum
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn capture_cursor(&mut self) {
        self.window.capture_cursor();
    }

    // Symmetric with `capture_cursor`; reached only through `set_camera_capture`
    // today, kept public so the cursor API stays complete.
    #[allow(dead_code)]
    pub fn release_cursor(&mut self) {
        self.window.release_cursor();
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered in the window helper.
    pub fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        self.window.set_ui_cursor_hidden(hidden);
    }

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // `poll` (in `window_closed`); false while captured or in fullscreen (which
    // confines the cursor instead).
    pub fn cursor_outside_window(&self) -> bool {
        self.window.cursor_outside_window()
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`.
    pub fn set_menu_mode(&mut self, on: bool) {
        self.window.set_menu_mode(on);
    }

    // Drive cursor capture from the menu state each frame: capture for camera
    // control, release while a menu is open. Edge-triggered in the window.
    pub fn set_camera_capture(&mut self, capture: bool) {
        self.window.set_camera_capture(capture);
    }

    // Turn display sync (vsync) on or off at runtime. The present mode is fixed
    // at swapchain creation (FIFO for vsync, MAILBOX/IMMEDIATE for uncapped), so
    // a change recreates the swapchain, which re-selects the mode from
    // `self.vsync`. Edge-triggered: a redundant call (a swapchain rebuild is
    // expensive) is skipped.
    pub fn set_vsync(&mut self, on: bool) {
        if on == self.vsync {
            return;
        }
        self.vsync = on;
        if let Err(e) = self.rebuild_swapchain() {
            tracing::warn!("set_vsync: rebuild_swapchain failed: {}", e);
        }
    }

    // Switch window mode / resize at runtime. The GLFW work lives in window.rs;
    // the framebuffer-size change drives a swapchain rebuild via the present
    // path's OUT_OF_DATE handling. Code-only on macOS; verify on Linux/Windows.
    pub fn set_window_mode(&mut self, mode: crate::assets::WindowMode) {
        self.window.set_window_mode(mode);
    }

    pub fn set_window_size(&mut self, width: u32, height: u32) {
        self.window.set_window_size(width, height);
    }

    // Replace the live post-process parameters, pushed to the bloom + composite
    // shaders each frame. Code-only on macOS; verify on Linux/Windows.
    pub fn update_post_process(&mut self, params: crate::gfx::render_types::PostProcessParams) {
        self.post_process = params;
    }

    // Set the live ambient (IBL) light scale (the Ambient slider). It lives in
    // `LightUniforms`, uploaded to a single (not per-frame) UBO, so unlike
    // `update_post_process` (push constants) it mutates the CPU-side copy and
    // re-uploads the buffer. Because the buffer is shared across frames-in-flight,
    // the device is drained first so the rewrite never races an in-flight read;
    // ambient changes only on a slider drag, so the stall is rare. Edge-triggered:
    // a no-op when the value is unchanged (e.g. an init push with no persisted
    // override).
    pub fn set_ambient_intensity(&mut self, value: f32) {
        if self.uniforms.light_uniforms.ambient_intensity == value {
            return;
        }
        self.uniforms.light_uniforms.ambient_intensity = value;
        self.wait_idle();
        if let Err(e) = super::draw::upload_light_uniforms(
            &self.device,
            self.uniforms.light_ubo_memory,
            &self.uniforms.light_uniforms,
        ) {
            tracing::warn!("set_ambient_intensity: re-upload light uniforms failed: {e}");
        }
    }

    // Set the live shadow cascade re-render cadence. The per-frame cascade split
    // reads `shadow.update` at the start of each draw (see draw.rs), so a change
    // takes effect on the next frame with no rebuild or allocation.
    pub fn set_shadow_update(&mut self, update: crate::assets::ShadowUpdate) {
        self.shadow.update = update;
    }

    // Set the live shadow distance (world units). The per-frame cascade-split
    // computation reads `shadow.distance` each draw (capped at the camera far
    // plane), so a change takes effect on the next frame with no allocation (it
    // sizes no GPU resource).
    pub fn set_shadow_distance(&mut self, distance: u32) {
        self.shadow.distance = distance;
    }

    // Set the live shadow cascade count (1..=4). The per-frame split + schedule
    // read `shadow.cascades` each draw; only the first `count` of the four slots
    // are rendered + sampled, so a change takes effect on the next frame with no
    // resize (the shadow-map array stays sized for the 4-cascade capacity).
    pub fn set_shadow_cascades(&mut self, count: u32) {
        self.shadow.cascades = count;
    }

    // Update the live scalar sub-tunables of the SSAO / SSR / SSGI / auto-exposure
    // passes without rebuilding anything. Each pass rebuilds its per-frame uniform
    // from these stored `*Settings` every draw (`settings.params(...)`), so
    // mutating the stored struct here is picked up on the next frame. Only a
    // feature whose resources are currently live has settings to mutate; the rest
    // are skipped (the value still persists for the next launch). SSAO / SSR /
    // auto-exposure are fully scalar, so they are replaced wholesale; SSGI keeps
    // its gather resolution / ray / step counts (those size the gather target or
    // ride `apply_quality_settings`), so only its scalar intensity / distance are
    // updated. Auto-exposure settings live flat on the context here
    // (`auto_exposure_settings`), not inside a resources struct as on Metal.
    pub fn update_quality_params(&mut self, q: crate::gfx::backend::QualitySettings) {
        if let (Some(live), Some(cur)) = (q.ssao, self.ssao.as_mut().map(|s| &mut s.settings)) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (q.ssr, self.ssr.as_mut().map(|s| &mut s.settings)) {
            *cur = live;
        }
        if let (Some(live), Some(cur)) = (q.ssgi, self.ssgi.as_mut().map(|s| &mut s.settings)) {
            cur.intensity = live.intensity;
            cur.max_distance = live.max_distance;
        }
        if let (Some(live), Some(cur)) = (q.auto_exposure, self.auto_exposure_settings.as_mut()) {
            *cur = live;
        }
    }

    // Public accessor for the shared shader-reload flag. Cloning the `Arc`
    // lets the debug WebSocket server flip it from a non-render thread.
    // `None` outside `cn debug`. Mirrors `DxContext::shader_reload_pending`.
    pub fn shader_reload_pending(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.shader_reload_pending
            .as_ref()
            .map(std::sync::Arc::clone)
    }

    pub fn take_input(&mut self) -> InputState {
        let raw = self.window.take_input();
        InputState {
            forward: raw.forward,
            backward: raw.backward,
            left: raw.left,
            right: raw.right,
            sprint: raw.sprint,
            interact: raw.interact,
            jump: raw.jump,
            mouse_dx: raw.mouse_dx,
            mouse_dy: raw.mouse_dy,
            scroll_delta: raw.scroll_delta,
            mouse_x: raw.mouse_x,
            mouse_y: raw.mouse_y,
            left_click: raw.left_click,
            left_button_down: raw.left_button_down,
            hud_toggle: raw.hud_toggle,
            escape: raw.escape,
            captured_key: raw.captured_key,
        }
    }

    // Replace the runtime movement key map. The GLFW key callback decodes events
    // through it, so a settings-menu rebind takes effect immediately.
    pub fn set_keymap(&mut self, keymap: &crate::gfx::keymap::KeyMap) {
        self.window.set_keymap(keymap);
    }

    // Live window size for overlay (view-owned UI) scaling and cursor
    // hit-testing. Returns the swapchain pixel extent, which is the attachment
    // the composite + text pass writes and the space the UI shader divides
    // vertices by. The cursor reported by `poll()` is mapped from GLFW window
    // coordinates into this framebuffer-pixel space at the source (see
    // `scale_cursor_to_framebuffer` in window.rs), so the overlay forward /
    // inverse transforms stay consistent both where the two are equal (Windows,
    // unscaled X11) and on a scaled surface (hi-DPI Wayland, framebuffer larger
    // than the window).
    pub fn logical_size(&self) -> (f32, f32) {
        (
            self.swapchain_extent.width as f32,
            self.swapchain_extent.height as f32,
        )
    }

    // Device capability flags for the settings menu. RT reflects whether the
    // ray-query device extensions were enabled at device creation
    // (`rt_capable`).
    pub fn capabilities(&self) -> crate::gfx::backend::DeviceCapabilities {
        crate::gfx::backend::DeviceCapabilities {
            ray_tracing: self.rt_capable,
        }
    }

    // Coarse GPU performance profile for default-quality selection, read live
    // from the physical device: vendor id, discrete / integrated device type,
    // and the summed DEVICE_LOCAL heap size as the VRAM budget (the true heap
    // size, unlike the residency chip which sums live usage).
    pub fn gpu_profile(&self) -> crate::gfx::backend::GpuProfile {
        use crate::gfx::backend::{GpuClassInput, GpuProfile, GpuVendor, classify_tier};
        let props = unsafe {
            self.instance
                .get_physical_device_properties(self.physical_device)
        };
        let vendor = match props.vendor_id {
            0x10DE => GpuVendor::Nvidia,
            0x1002 => GpuVendor::Amd,
            0x8086 => GpuVendor::Intel,
            0x106B => GpuVendor::Apple, // Apple / MoltenVK
            _ => GpuVendor::Other,
        };
        let discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        let unified = props.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU;
        let mem = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        let budget: u64 = (0..mem.memory_heap_count as usize)
            .filter(|&i| {
                mem.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            })
            .map(|i| mem.memory_heaps[i].size)
            .sum();
        let tier = classify_tier(&GpuClassInput {
            vendor,
            memory_budget_bytes: budget,
            discrete,
            apple_family: 0,
        });
        GpuProfile {
            vendor,
            tier,
            memory_budget_bytes: budget,
            unified_memory: unified,
            discrete,
        }
    }
}

impl crate::gfx::scene_reel::SceneControl for VkContext {
    fn update_visibility(&mut self, draw_idx: usize, visible: bool) {
        self.update_visibility(draw_idx, visible);
    }
    fn update_clear_color(&mut self, color: [f32; 4]) {
        self.update_clear_color(color);
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        self.wait_idle();
        let device = self.device.clone();
        let device = &device;

        // Abandon any in-flight staggered probe bake: free its per-face command
        // buffers (before `self.commands` is destroyed below) + fences + bake target.
        // `wait_idle` above retired its GPU work. The converting slot holds only CPU
        // data (drops freely; its worker thread, if still running, touches no vk
        // handle, only the shared payload `OnceLock`).
        if let Some(rendering) = self.probe_rendering.take() {
            rendering.destroy(device, self.commands.command_pool);
        }

        // Deferred text buffers.
        for db in self.deferred_destroy.borrow().iter() {
            db.destroy(device);
        }

        // Sync (per-frame-in-flight semaphores + fences).
        self.frame_sync.destroy(device);

        // Command pools (each frees the buffers allocated from it).
        self.commands.destroy(device);

        // Framebuffers + attachments.
        self.destroy_swapchain_resources();

        // Shadow (framebuffers, pipelines, layouts, map, render pass, UBO,
        // sampler).
        self.shadow.destroy(device);

        // IBL cubes + cube sampler.
        self.env_map.irradiance.destroy(device);
        self.env_map.prefilter.destroy(device);
        unsafe { device.destroy_sampler(self.cube_sampler, None) };

        // Pipelines.
        unsafe { device.destroy_pipeline(self.main_pipeline, None) };
        if let Some(p) = self.text_pipeline {
            unsafe { device.destroy_pipeline(p, None) };
        }
        // Instanced-prop pipeline + per-frame instance buffers (see
        // `VkInstanced::destroy`).
        self.instanced.destroy(device);
        unsafe { device.destroy_pipeline_layout(self.main_pipeline_layout, None) };
        unsafe { device.destroy_pipeline_layout(self.text_pipeline_layout, None) };

        // GPU-driven cull + bindless static pass resources, including the Hi-Z
        // pyramid (see `VkCull::destroy`). The bindless / cull / phase-2
        // descriptor sets are freed with the shared descriptor pool +
        // `two_pass_pool`.
        self.cull.destroy(device);

        // Composite pass resources.
        self.color_lut.destroy(device);
        unsafe {
            device.destroy_pipeline(self.composite_pipeline, None);
            device.destroy_pipeline_layout(self.composite_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.composite_set_layout, None);
            device.destroy_sampler(self.composite_sampler, None);
        }

        // Bloom resources (mips + framebuffers freed by
        // destroy_swapchain_resources above).
        unsafe {
            device.destroy_pipeline(self.bloom_pipeline_prefilter, None);
            device.destroy_pipeline(self.bloom_pipeline_downsample, None);
            device.destroy_pipeline(self.bloom_pipeline_upsample, None);
            device.destroy_pipeline_layout(self.bloom_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.bloom_set_layout, None);
            device.destroy_descriptor_pool(self.bloom_descriptor_pool, None);
            device.destroy_render_pass(self.bloom_write_pass, None);
            device.destroy_render_pass(self.bloom_blend_pass, None);
        }

        // TAA resources (velocity + history passes, pipelines, targets, UBOs).
        if let Some(taa) = &mut self.taa {
            taa.destroy(device);
        }

        // SSAO resources (pre-pass + kernel + blur). The blur framebuffer
        // references the pool's `ao_output` view, so SSAO is torn down before
        // the transient pool below (framebuffers before their views).
        if let Some(ssao) = &mut self.ssao {
            ssao.destroy(device);
        }
        self.ssao_white.destroy(device);

        // Transient image pool (the graph-owned transients, e.g. `ao_output`).
        self.transient_pool.destroy(device);

        // SSR resources (pre-pass + resolve).
        if let Some(ssr) = &mut self.ssr {
            ssr.destroy(device);
        }

        // Reflection composite (roughness blur + composite of the SSR/RT output).
        if let Some(rc) = &mut self.reflection_composite {
            rc.destroy(device);
        }

        // SSGI resources (gather + composite).
        if let Some(ssgi) = &mut self.ssgi {
            ssgi.destroy(device);
        }

        // Unified G-buffer pre-pass resources (per-frame MRT + pipelines + UBOs).
        if let Some(gb) = &mut self.gbuffer {
            gb.destroy(device);
        }

        // Hardware ray-traced reflection resources (the pass + the acceleration
        // structures). The pass is destroyed first (its output + pipelines), then
        // the BLAS / TLAS / scratch / geometry table.
        if let Some(rt) = &mut self.rt_reflections {
            rt.destroy(device);
        }
        if let Some(accel) = &mut self.rt_accel {
            accel.destroy(device);
        }

        // Temporal upscaling (FSR / DLSS / XeSS): the vendor context + the
        // output texture, via the backend trait.
        if let Some(up) = &mut self.upscale {
            up.destroy(device);
        }

        // Decal resources (pipeline + per-frame uniforms + per-decal sets).
        if let Some(decals) = &mut self.decals_state {
            decals.destroy(device);
        }

        // Volumetric-fog resources (pipeline + per-frame uniforms).
        if let Some(fog) = &mut self.fog_resources {
            fog.destroy(device);
        }

        // Raymarched SDF volume resources (per-volume pipelines + UBOs, view
        // ring, descriptor pool, render passes, snapshot image).
        if let Some(rm) = &mut self.raymarch {
            rm.destroy(device);
        }

        // Planar reflection resources (mirror targets + framebuffers + per-(plane,
        // frame) view ring + global sets + descriptor pool). Destroyed before glass,
        // whose per-pane sets reference the planar target views.
        if let Some(planar) = &mut self.planar_reflection {
            planar.destroy(device);
        }

        // Glass / transparent-pass resources (pipeline, per-panel buffers +
        // UBOs, per-frame view ring, descriptor pool, render pass, framebuffers,
        // snapshot image).
        if let Some(glass) = &mut self.glass {
            glass.destroy(device);
        }

        // Auto-exposure resources (pipelines + histogram + per-frame readbacks).
        if let Some(ae) = &mut self.auto_exposure {
            ae.destroy(device);
        }

        // Particle resources (compute + render pipelines, view UBO ring,
        // per-emitter descriptor pool, framebuffers). Per-emitter pool /
        // counter buffers are destroyed via the dedicated helper before
        // the shared pipeline state: Vulkan needs the per-emitter buffers
        // gone first so the upcoming pipeline destroys can't trip a
        // validation error on a still-referenced descriptor.
        self.destroy_particle_emitter_states(device);
        if let Some(p) = &mut self.particle_resources {
            p.destroy(device);
        }

        // Profiler-overlay timestamp pool.
        if let Some(pool) = self.timestamp_query_pool.take() {
            unsafe { device.destroy_query_pool(pool, None) };
        }

        // Render passes.
        unsafe {
            device.destroy_render_pass(self.main_render_pass, None);
            device.destroy_render_pass(self.composite_render_pass, None);
        }

        // Chunk-stream descriptor pool (frees `chunk_stream.object_set`); the
        // shared main pool + its set layouts are freed by
        // `self.descriptors.destroy` below.
        self.chunk_stream.destroy(device);
        if let Some(pool) = self.clone_descriptor_pool {
            unsafe { device.destroy_descriptor_pool(pool, None) };
        }

        // Skinned-mesh resources.
        self.skinned.destroy(device);

        // Main descriptor pool + the three set layouts (the pool frees the
        // global / object / text_atlas sets).
        self.descriptors.destroy(device);

        // Geometry.
        self.geometry.destroy(device);

        // UBOs (per-frame view + shared light).
        self.uniforms.destroy(device);

        // Samplers.
        unsafe {
            device.destroy_sampler(self.linear_sampler, None);
            device.destroy_sampler(self.text_sampler, None);
        }

        // Scene textures.
        for t in &self.textures {
            t.destroy(device);
        }
        for t in &self.normal_map_textures {
            t.destroy(device);
        }
        for t in &self.text_atlas_textures {
            t.destroy(device);
        }
        // Baked reflection-probe cubes (empty until the capture pass bakes).
        for t in &self.probe_maps {
            t.destroy(device);
        }

        // Surface.
        unsafe { self.surface_loader.destroy_surface(self.surface, None) };

        // Debug.
        if let (Some(du), Some(dm)) = (&self.debug_utils, self.debug_messenger) {
            unsafe { du.destroy_debug_utils_messenger(dm, None) };
        }

        unsafe { self.device.destroy_device(None) };
        unsafe { self.instance.destroy_instance(None) };
    }
}
