// src/vulkan/graph_exec.rs
//
// Vulkan-side executor for the render graph. `VkContext::execute_graph`
// walks the `CompiledGraph` produced by the shared
// [`gfx::render_graph::build_frame_graph`](../gfx/render_graph/frame.rs)
// and dispatches each pass to its `encode_*` method. Mirrors the Metal
// + DirectX executors: every backend now drives the same builder.
//
// Decals landed 2026-05-24, Fog followed, AutoExposure landed
// 2026-05-25, and Particles landed 2026-05-25. The catch-all arm at the
// bottom returns a clear error if any not-yet-ported `PassId` slips into
// the compiled graph.
//
// Per-pass `barriers_before` is consumed for the resources the executor's
// barrier registry resolves (`ao_output`, `shadow_map`, `fog_froxel_volume`):
// `emit_graph_barriers` translates their graph state transitions into explicit
// `vkCmdPipelineBarrier` image transitions at the start of each pass's command
// buffer. Every other resource still owns its transitions inline or
// render-pass-baked (attachment layout pinning). Migrating the rest off those
// paths is the open follow-up shared with DirectX. All three migrated resources
// now rest sampled between frames, so BOTH their transitions are graph-driven:
// `shadow_map`'s producer is the SHADER_READ_ONLY → DEPTH_STENCIL_ATTACHMENT
// cross-frame reset (folded off the old inline `record_frame` restore) and its
// consumer the DEPTH_STENCIL → SHADER_READ_ONLY sample; `fog_froxel_volume`'s
// producer is the SHADER_READ_ONLY → GENERAL open and its consumer the
// GENERAL → SHADER_READ_ONLY close. Only the shadow-map sync the froxel kernel's
// CSM tap needs stays inline in `encode_fog_froxel`, since shadow_map isn't a
// graph read of that pass.
//
// Bundled passes:
//   * `PassId::SsaoBlur` dispatches the bundled `encode_ssao` (GTAO
//     kernel + depth-aware blur over the unified pre-pass normal+depth).
//     `PassId::SsaoPrepass` / `PassId::SsaoKernel` stay timing-only and
//     the executor rejects them as graph nodes.

use ash::vk;

use crate::gfx::frustum::Frustum;
use crate::gfx::render_graph::{CompiledGraph, CompiledPass, GraphResourceClass, PassId};
use crate::gfx::render_types::TextDrawCall;

use super::barrier_translate::vk_transition;
use super::context::VkContext;
use super::parallel_encoder::ParallelCtxRef;

// One resolved barrier target: the image a graph resource backs, its class, and
// its array-layer count (1 for a plain target, the cascade count for the CSM
// `shadow_map`). Built once per frame by `build_barrier_registry`. `vk::Image`
// is a plain `Send + Sync` handle, so the registry shares into the parallel pass
// workers with no wrapper (unlike the DirectX side's COM handles).
struct VkBarrierTarget {
    image: vk::Image,
    class: GraphResourceClass,
    layer_count: u32,
}

// `ResourceId`-indexed table of barrier targets for the migrated graph resources
// (`None` for every resource the executor doesn't graph-drive). A resource is
// graph-driven iff it has a `Some` entry, so this table is the single source of
// truth that replaced the old label allowlist + per-label resolver. Built on the
// main thread by `build_barrier_registry`, where the only field-naming of the
// migrated resources lives; the parallel emit path stays field-agnostic.
struct VkBarrierRegistry(Vec<Option<VkBarrierTarget>>);

// Emit the explicit image-layout transitions for the migrated graph resources
// from a pass's `barriers_before`, resolved through the registry. Called at the
// start of each pass's own command buffer, before the pass encodes, so the
// transition lands ahead of the pass's render pass in the same submission. A
// resource with no registry entry is skipped and keeps its render-pass-driven
// transition; a transition whose layout does not change (e.g. the depth
// producer's no-op Undefined -> Write) is skipped too. Takes `&ash::Device`, not
// `&VkContext`: the field-to-image mapping was already resolved into the
// registry, so this parallel path is field-agnostic.
fn emit_graph_barriers(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    registry: &VkBarrierRegistry,
    pass: &CompiledPass,
) {
    for op in &pass.barriers_before {
        let Some(Some(target)) = registry.0.get(op.resource_index()) else {
            continue;
        };
        let Some((old_layout, new_layout, src_access, dst_access, src_stage, dst_stage)) =
            vk_transition(
                target.class,
                op.from_state(),
                op.to_state(),
                op.read_stages(),
            )
        else {
            continue;
        };
        let aspect = match target.class {
            GraphResourceClass::DepthTarget => vk::ImageAspectFlags::DEPTH,
            GraphResourceClass::ColorTarget | GraphResourceClass::StorageImage => {
                vk::ImageAspectFlags::COLOR
            }
        };
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: target.layer_count,
            })
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }
}

// Emit the aliasing barriers for a pass: for each pooled transient this pass
// first-writes whose memory is reused from an earlier transient in the same slot
// (`images`), order that earlier resource's prior use before this write. The
// members are colour targets, so the dependency is the colour/fragment domain:
// the predecessor's last use is either a fragment-shader sample (e.g.
// `ao_output` read by Main) or a colour write, and this member's first use is a
// colour write (e.g. the bloom prefilter). `UNDEFINED -> COLOR_ATTACHMENT`
// discards the predecessor's contents in the shared memory (the member is fully
// rewritten before it is read). Per-resource stage derivation can refine this
// when a non-colour member is aliased.
fn emit_alias_barriers(device: &ash::Device, cmd: vk::CommandBuffer, images: &[vk::Image]) {
    for &image in images {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }
}

// Per-frame params the executor threads into each pass's `encode_*`
// method. The set grows as more passes migrate; Composite needs the
// swapchain image index + text calls, Shadow needs neither (it reads
// per-frame state straight off `&self`), Main needs the BVH-culled
// visible set + frustum + camera position. New fields land here when
// a pass that needs them migrates.
pub(in crate::vulkan) struct GraphFrameParams<'a> {
    pub cmd: vk::CommandBuffer,
    pub image_index: u32,
    pub frame_idx: usize,
    pub text_calls: &'a [TextDrawCall],
    // An opaque menu backdrop hides the scene: the Main pass clears its target
    // and skips every draw (the masked graph drops all other world passes), so
    // nothing of the world renders behind the menu.
    pub world_hidden: bool,
    // CPU visibility list (BVH-culled cullables + always_draw fallback).
    // Consumed by Main's legacy + instanced fallback passes and the
    // unified G-buffer pre-pass.
    pub visible: &'a [u32],
    // Camera frustum used to cull instanced clusters during Main's
    // per-cluster draw loop, and by the G-buffer pre-pass for the same
    // reason.
    pub frustum: &'a Frustum,
    // Camera world-space position used for per-cluster distance-cull
    // during Main's instanced sub-pass and the G-buffer pre-pass.
    pub cam_pos: [f32; 3],
    // Jittered view-projection matrix (with TAA Halton jitter when
    // TAA is on). Consumed by the G-buffer pre-pass to rasterise the
    // normal+depth / roughness / velocity MRT.
    pub vp_mat: [[f32; 4]; 4],
    // Un-jittered current-frame view-projection matrix. The G-buffer
    // pre-pass uses it (alongside `vp_mat` for the jittered VP and the
    // prior frame's `prev_view_proj` stored on `GbufferResources`) so the
    // stored motion vector is free of sub-pixel jitter. `Default::default()`
    // for frames where the velocity channel isn't dispatched.
    pub cur_vp: [[f32; 4]; 4],
    // Vertical FOV in radians: SSAO needs it for the projection
    // reconstruction used by the GTAO horizon search; SSR resolve
    // uses it for the same.
    pub fov_y_radians: f32,
    // Camera aspect ratio (width / height); same SSAO + SSR use
    // as `fov_y_radians`.
    pub aspect: f32,
    // Frame-global elapsed seconds. The particle encoder needs it to
    // derive `dt` from the last-frame snapshot it stashed in a `Cell`;
    // the compute kernel multiplies dt against `spawn_rate` and the
    // integration step.
    pub elapsed: f32,
    // Camera near-plane in view units. The FogFroxel kernel needs it to map
    // each Z slab onto the linear-Z `[near, max_distance]` volume range.
    pub near: f32,
    // Camera far-plane in view units. The temporal-upscale dispatch (FSR;
    // DLSS / XeSS ignore it) needs the near + far + FOV to linearise depth for
    // its reprojection.
    pub far: f32,
}

impl VkContext {
    // Walk a compiled render graph and record each pass. Every non-composite
    // pass is recorded into its own per-`(frame, pass)` primary command buffer
    // (parallel command-buffer recording); Composite is recorded into the
    // frame's outer "end" buffer (`params.cmd`) on the main thread because it
    // writes the swapchain image + allocates transient text buffers. Returns
    // the per-pass buffers in graph (toposort) order; the caller submits
    // `[start, ...returned, end]` in one `vkQueueSubmit`, so submission order =
    // GPU order and every encoder's inline barrier still synchronises against
    // the prior pass across the command-buffer boundary. Any not-yet-migrated
    // `PassId` returns a clear error.
    pub(in crate::vulkan) fn execute_graph(
        &mut self,
        graph: &CompiledGraph,
        params: &GraphFrameParams<'_>,
    ) -> Result<Vec<vk::CommandBuffer>, String> {
        // Particle per-frame state (dt / frame index / per-emitter spawn
        // budgets) is advanced here on `&mut self` before any pass encodes, so
        // the `&self` `encode_particles` (which may run on a parallel-recording
        // worker) never mutates the particle `Cell`s. `None` when the pass is
        // inert. Mirrors Metal's `prepare_particle_pass` hoist.
        let particle_frame = self.prepare_particle_pass(params.elapsed);

        // Instanced clusters: recompute the per-cluster LOD-bucket partition
        // and upload the bucket-ordered instance matrices on `&mut self` before
        // the fan-out, so every instanced pass (Main + the unified G-buffer
        // pre-pass + Shadow) reads a consistent partition while recording on
        // worker threads. Inert when no clusters are declared.
        self.prepare_instanced_clusters(params.frame_idx, params.cam_pos);

        // Composite stays on the main thread (it writes the swapchain image +
        // allocates transient text buffers + touches `deferred_destroy`); every
        // other pass fans onto a `jobs::pool()` worker that records into its own
        // `(frame, pass)` command buffer.
        let composite_idx = graph.passes.iter().position(|p| p.id == PassId::Composite);
        let frame_idx = params.frame_idx;
        let device = self.device.clone();

        // One output slot per graph pass index; each worker stores its finished
        // command buffer at its own index. Disjoint indices, but a `Mutex`
        // keeps the store sound + simple (it's once per pass). `first_error`
        // captures the first worker failure.
        let worker_slots: std::sync::Mutex<Vec<Option<vk::CommandBuffer>>> =
            std::sync::Mutex::new(vec![None; graph.passes.len()]);
        let first_error: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

        // Resolve every migrated resource's barrier target once, on the main
        // thread, then share the table read-only into the parallel pass workers.
        let registry = self.build_barrier_registry(graph, frame_idx);
        // Per-pass aliasing barriers for the pooled transients that share memory
        // this frame (e.g. `bloom_top` reusing `ao_output`'s slot). Empty when no
        // slot is shared.
        let alias_barriers = self.build_alias_barriers(graph, frame_idx);
        let ctx_ref = ParallelCtxRef::new(self);
        let particle_ref = particle_frame.as_ref();
        let device_ref = &device;
        let worker_slots_ref = &worker_slots;
        let first_error_ref = &first_error;
        let registry_ref = &registry;
        let alias_barriers_ref = &alias_barriers;

        crate::jobs::pool().install(|| {
            rayon::scope(|scope| {
                for (idx, pass) in graph.passes.iter().enumerate() {
                    if Some(idx) == composite_idx {
                        continue;
                    }
                    let pass_id = pass.id;
                    scope.spawn(move |_| {
                        let ctx = ctx_ref.as_ctx();
                        let pool_idx =
                            frame_idx * crate::gfx::render_graph::PASS_COUNT + pass_id as usize;
                        let buf = ctx.commands.pass_command_buffers[pool_idx];
                        let set_err = |msg: String| {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(msg);
                            }
                        };
                        // Reset + begin this pass's own buffer (its own pool, so
                        // no cross-worker pool contention), encode, end.
                        let begin = unsafe {
                            device_ref
                                .reset_command_buffer(buf, vk::CommandBufferResetFlags::empty())
                                .and_then(|()| {
                                    device_ref.begin_command_buffer(
                                        buf,
                                        &vk::CommandBufferBeginInfo::default()
                                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                                    )
                                })
                        };
                        if let Err(e) = begin {
                            set_err(format!("begin pass cmd buf ({}): {e}", pass_id.name()));
                            return;
                        }
                        // Per-pass GPU timing: bracket this pass's encode with a
                        // (start, end) timestamp pair in its own buffer. The block
                        // was reset in the start buffer (submitted first), so these
                        // writes are valid. A pass absent from a later frame's graph
                        // leaves its slots unwritten; the readback's
                        // `WITH_AVAILABILITY` reports those as 0.
                        if let Some(pool) = ctx.timestamp_query_pool {
                            let (ts_start, _) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            unsafe {
                                device_ref.cmd_write_timestamp(
                                    buf,
                                    vk::PipelineStageFlags::TOP_OF_PIPE,
                                    pool,
                                    ts_start,
                                );
                            }
                        }
                        // Graph-driven transitions for the migrated resources
                        // (replacing the encoder's stripped/render-pass-baked
                        // transitions), then any aliasing barriers for pooled
                        // transients that reuse a slot this frame, then the pass
                        // body. Resolved through the registry / alias table, so
                        // this path names no `VkContext` fields.
                        emit_graph_barriers(device_ref, buf, registry_ref, pass);
                        emit_alias_barriers(device_ref, buf, &alias_barriers_ref[idx]);
                        if let Err(e) = ctx.encode_pass_into(pass_id, buf, params, particle_ref) {
                            set_err(e);
                            return;
                        }
                        if let Some(pool) = ctx.timestamp_query_pool {
                            let (_, ts_end) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            unsafe {
                                device_ref.cmd_write_timestamp(
                                    buf,
                                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                                    pool,
                                    ts_end,
                                );
                            }
                        }
                        if let Err(e) = unsafe { device_ref.end_command_buffer(buf) } {
                            set_err(format!("end pass cmd buf ({}): {e}", pass_id.name()));
                            return;
                        }
                        worker_slots_ref.lock().unwrap()[idx] = Some(buf);
                    });
                }
            });
        });

        if let Some(e) = first_error.into_inner().unwrap() {
            return Err(e);
        }

        // Composite on the main thread, into the outer "end" buffer. Bracket it
        // with its own per-pass timestamp pair (in the end buffer, which also
        // carries the whole-frame end timestamp written later in `record_frame`).
        if composite_idx.is_some() {
            if let Some(pool) = self.timestamp_query_pool {
                let (ts_start, _) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                unsafe {
                    self.device.cmd_write_timestamp(
                        params.cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        pool,
                        ts_start,
                    );
                }
            }
            self.encode_pass_into(
                PassId::Composite,
                params.cmd,
                params,
                particle_frame.as_ref(),
            )?;
            if let Some(pool) = self.timestamp_query_pool {
                let (_, ts_end) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                unsafe {
                    self.device.cmd_write_timestamp(
                        params.cmd,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        pool,
                        ts_end,
                    );
                }
            }
        }

        // Collect the per-pass buffers in ascending graph index = toposort
        // order (the `None` Composite slot is skipped). Never sort: the submit
        // array order must equal toposort order for the inline barriers to
        // synchronise correctly across buffer boundaries.
        let ordered: Vec<vk::CommandBuffer> = worker_slots
            .into_inner()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        Ok(ordered)
    }

    // Resolve every migrated graph resource to its barrier target, indexed by
    // `ResourceId` (its position in `graph.resources`), so the parallel emit path
    // can look a target up by `BarrierOp::resource_index()`. This is the single
    // place that names the migrated resources' backing `VkContext` fields;
    // field-grouping re-cuts here, not in the executor. A resource the owning
    // feature disabled (or one never migrated) gets `None`, and the graph carries
    // no barrier for it either.
    fn build_barrier_registry(&self, graph: &CompiledGraph, frame_idx: usize) -> VkBarrierRegistry {
        VkBarrierRegistry(
            graph
                .resources
                .iter()
                .map(|res| self.barrier_target_for_label(res.label, frame_idx))
                .collect(),
        )
    }

    // Build the per-pass aliasing-barrier table for this frame: `table[i]` holds
    // the pooled images to alias-barrier at the start of graph pass `i`. A pooled
    // transient that reuses an earlier transient's memory (a slot predecessor in
    // the pool) needs the predecessor's prior use ordered before its first write;
    // the barrier lands before the pass that first writes it (`lifetime.first`).
    // Empty for every resource the pool does not alias (no predecessor), so the
    // table is empty whenever no slot is shared this frame.
    fn build_alias_barriers(&self, graph: &CompiledGraph, frame_idx: usize) -> Vec<Vec<vk::Image>> {
        let mut table = vec![Vec::new(); graph.passes.len()];
        for res in &graph.resources {
            if self.transient_pool.alias_predecessor(res.label).is_none() {
                continue;
            }
            if let Some(image) = self.transient_pool.image_for(res.label, frame_idx) {
                let first = res.lifetime.first;
                if first < table.len() {
                    table[first].push(image);
                }
            }
        }
        table
    }

    // Map one graph resource label to its backing `vk::Image`, class, and
    // array-layer count (1 for a plain target, the cascade count for the CSM
    // `shadow_map`). `frame_idx` selects the frame-in-flight copy for the
    // per-frame pooled transients (`ao_output`).
    fn barrier_target_for_label(&self, label: &str, frame_idx: usize) -> Option<VkBarrierTarget> {
        match label {
            "ao_output" => self
                .transient_pool
                .image_for("ao_output", frame_idx)
                .map(|image| VkBarrierTarget {
                    image,
                    class: GraphResourceClass::ColorTarget,
                    layer_count: 1,
                }),
            // shadow_map rests sampled (SHADER_READ_ONLY) between frames, so the
            // Shadow producer barrier is the real SHADER_READ_ONLY ->
            // DEPTH_STENCIL_ATTACHMENT cross-frame reset and the Main consumer
            // (Write -> Read) returns it to sampled, both over all cascade layers.
            // The map is created sampled, so frame 0's producer starts from the
            // image's real layout; there is no inline reset.
            "shadow_map" if !self.shadow.framebuffers.is_empty() => Some(VkBarrierTarget {
                image: self.shadow.map.image,
                class: GraphResourceClass::DepthTarget,
                layer_count: self.shadow.framebuffers.len() as u32,
            }),
            // fog_froxel_volume: the froxel kernel writes it (GENERAL) and the
            // fog fragment samples it (SHADER_READ_ONLY). Both transitions are
            // graph-driven: the producer (Undefined -> Write) is the real
            // SHADER_READ_ONLY -> GENERAL open, the consumer (Write -> Read) the
            // GENERAL -> SHADER_READ_ONLY close. One array layer (the 3D volume).
            "fog_froxel_volume" => self.fog_resources.as_ref().map(|f| VkBarrierTarget {
                image: f.volume_image,
                class: GraphResourceClass::StorageImage,
                layer_count: 1,
            }),
            _ => None,
        }
    }

    // Record a single render-graph pass into `cmd`. Shared by the (current)
    // serial driver and the parallel fan-out: takes `&self` so it can run on a
    // worker thread. `particle_frame` is the precomputed per-frame particle
    // state from `prepare_particle_pass` (the only pass needing pre-advanced
    // state).
    pub(in crate::vulkan) fn encode_pass_into(
        &self,
        pass_id: PassId,
        cmd: vk::CommandBuffer,
        params: &GraphFrameParams<'_>,
        particle_frame: Option<&(f32, u32, Vec<u32>)>,
    ) -> Result<(), String> {
        match pass_id {
            PassId::Cull => {
                self.encode_cull(cmd, params.frame_idx, params.frustum, params.cam_pos);
                // Pose the skinned objects' deformed-vertex buffer for this frame
                // (a no-op when no skinned mesh is folded in). Independent of the
                // cull; both feed Main, which the toposort orders after Cull.
                self.encode_skin(cmd, params.frame_idx);
            }
            PassId::SsaoBlur => {
                // The single graph node for the bundled `encode_ssao`
                // dispatch: encodes the GTAO kernel + depth-aware blur over the
                // unified pre-pass's normal+depth. The SsaoPrepass / SsaoKernel
                // PassIds stay timing-only (rejected as graph nodes below) like
                // Metal's same pattern.
                self.encode_ssao(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::SsaoPrepass | PassId::SsaoKernel => {
                return Err(format!(
                    "graph executor (vulkan): pass {} is bundled inside SsaoBlur \
                     (encode_ssao encodes the SSAO kernel + blur sub-passes); it \
                     should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::ReflectionComposite => {
                // Metal-only inline pass; never scheduled on Vulkan. Handled here
                // only to keep the dispatch match exhaustive.
                return Err(format!(
                    "graph executor (vulkan): pass {} is a Metal-only inline \
                     reflection composite and should not appear as a graph node",
                    pass_id.name()
                ));
            }
            PassId::SsrPrepass => {
                // Merged into GBufferPrepass on Vulkan: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (vulkan): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::SsrResolve => {
                self.encode_ssr_resolve(
                    cmd,
                    params.frame_idx,
                    params.fov_y_radians,
                    params.aspect,
                    params.cam_pos,
                );
            }
            PassId::Ssgi => {
                self.encode_ssgi(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::RtReflections => {
                // Hardware ray-traced reflections (inline `rayQueryEXT`). Traces a
                // reflection ray per glossy pixel against the scene TLAS and
                // composites into the RT output target, which then feeds the post
                // stack. Occupies the `SsrResolve` slot; gated by
                // `FrameGraphInputs::rt_reflections_enabled`
                // (`VkContext::rt_reflections_active`). The per-frame TLAS update +
                // descriptor re-point already ran on the outer "start" buffer.
                self.encode_rt_reflections(
                    cmd,
                    params.frame_idx,
                    params.fov_y_radians,
                    params.aspect,
                    params.cam_pos,
                );
            }
            PassId::Velocity => {
                // Merged into GBufferPrepass on Vulkan: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (vulkan): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::TaaResolve => {
                self.encode_taa(cmd, params.frame_idx);
            }
            PassId::Upscale => {
                self.encode_upscale(cmd, params)?;
            }
            PassId::Bloom => {
                self.encode_bloom(cmd, params.frame_idx);
            }
            PassId::Shadow => {
                self.encode_shadow_pass(cmd, params.frame_idx, params.cam_pos, params.elapsed);
            }
            PassId::Main => {
                self.encode_main_pass(
                    cmd,
                    params.frame_idx,
                    params.visible,
                    params.frustum,
                    params.cam_pos,
                    params.world_hidden,
                );
            }
            PassId::Composite => {
                self.encode_composite_and_text(
                    cmd,
                    params.image_index,
                    params.frame_idx,
                    params.text_calls,
                )?;
            }
            PassId::Decals => {
                self.encode_decals(cmd, params.frame_idx, params.vp_mat, params.frustum);
            }
            PassId::FogFroxel => {
                // Populate the screen-aligned 3D scatter/transmittance volume
                // the `Fog` render pass samples. The shared graph seeds this
                // before `Fog` (RAW edge on the froxel volume handle).
                self.encode_fog_froxel(
                    cmd,
                    params.frame_idx,
                    params.near,
                    params.vp_mat,
                    params.cam_pos,
                );
            }
            PassId::Fog => {
                self.encode_fog(cmd, params.frame_idx, params.vp_mat, params.cam_pos);
            }
            PassId::AutoExposure => {
                self.encode_auto_exposure(cmd, params.frame_idx);
            }
            PassId::ParticlesDraw => {
                if let Some(frame) = particle_frame {
                    self.encode_particles(
                        cmd,
                        params.frame_idx,
                        frame,
                        params.vp_mat,
                        params.frustum,
                    );
                }
            }
            PassId::Raymarch => {
                // Composite each visible SDF volume into the scene. Uses the
                // jittered VP (the matrix the main pass rasterised depth with)
                // so the reprojected hit depth shares the scene's depth space.
                let view = self.build_raymarch_view(params.vp_mat, params.cam_pos, params.elapsed);
                self.encode_raymarch(cmd, params.frame_idx, &view)?;
            }
            PassId::Transparent => {
                // Generic translucent pass: draws the world's glass panels
                // back-to-front over the post-SSR scene. Gated by
                // `FrameGraphInputs::transparent_enabled` (set from
                // `glass.any_visible()`), so it only appears when the world
                // declared visible `GlassPanel`s. Uses the jittered VP (the
                // matrix the main pass rasterised depth with) so the glass
                // quad's clip-space depth matches the stored main-depth the
                // fragment shader tests against. Water is a separate
                // (Metal-only) producer not ported here.
                // Planar reflections run inline at the head of the pass (same cmd
                // buffer -> each plane's mirror target is ready before the glass
                // draws sample it). A no-op when the world has no planar set.
                // Skipped when the per-pixel RT glass trace is live: it supersedes
                // planar (sharp + off-screen-correct), so the mirror re-render would
                // be wasted. Gating on `rt_glass_active` (not `rt_reflections_active`)
                // keeps planar alive when RT is live but the glass RT pipelines
                // failed to build, so the glass probe / planar fallback samples a
                // freshly rendered resolve. Mirrors DirectX.
                if !self.rt_glass_active() {
                    self.encode_planar_reflections(
                        cmd,
                        params.frame_idx,
                        params.vp_mat,
                        params.cam_pos,
                        params.elapsed,
                    )?;
                }
                let view =
                    self.build_transparent_view(params.vp_mat, params.cam_pos, params.elapsed);
                self.encode_transparent(
                    cmd,
                    params.frame_idx,
                    &view,
                    params.fov_y_radians,
                    params.aspect,
                )?;
            }
            PassId::HizBuild => {
                // Two-pass occlusion: rebuild the Hi-Z pyramid mid-frame from
                // this frame's phase-1 depth so `Cull2` re-tests the phase-1
                // occluded objects against up-to-date depth. Same
                // `encode_hiz_build` the end-of-frame (next-frame) build uses.
                self.encode_hiz_build(cmd, params.frame_idx);
            }
            PassId::Cull2 => {
                self.encode_cull_phase2(
                    cmd,
                    params.frame_idx,
                    params.frustum,
                    params.cam_pos,
                    params.cur_vp,
                );
            }
            PassId::Main2 => {
                self.encode_main_pass_phase2(cmd, params.frame_idx);
            }
            PassId::GBufferPrepass => {
                // Unified geometry pre-pass: one jittered traversal writes
                // normal+depth, roughness, and motion for every screen-space
                // consumer (SSR / SSAO / SSGI / TAA / FSR), replacing the
                // separate SSR / SSAO / velocity pre-passes. `params.vp_mat` is
                // the jittered VP (rasterisation, matching the main pass);
                // `params.cur_vp` is the un-jittered VP the shader uses with the
                // previous VP for the motion vector. The velocity channel carries
                // real motion only when a consumer reads it (TAA or FSR active);
                // otherwise cur == prev and it stays a harmless zero. The merged
                // buffer is built whenever any of these consumers is on, so a
                // missing `self.gbuffer` here means the builder emitted this node
                // with no merged buffer present, a programming error.
                let gb = self.gbuffer.as_ref().ok_or(
                    "graph executor (vulkan): GBufferPrepass emitted but self.gbuffer is None",
                )?;
                let velocity_active = self.taa.is_some() || self.upscale.is_some();
                self.encode_gbuffer_prepass(
                    gb,
                    cmd,
                    params.frame_idx,
                    params.vp_mat,
                    params.cur_vp,
                    params.visible,
                    params.frustum,
                    params.cam_pos,
                    velocity_active,
                );
            }
            other => {
                return Err(format!(
                    "graph executor (vulkan): pass {} is not handled by this \
                     executor; it should not appear in the frame graph",
                    other.name()
                ));
            }
        }
        Ok(())
    }
}
