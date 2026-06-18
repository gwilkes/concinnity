// src/directx/graph_exec.rs
//
// DirectX-side executor for the render graph. `DxContext::execute_graph`
// walks the `CompiledGraph` produced by the shared
// [`gfx::render_graph::build_frame_graph`](../gfx/render_graph/frame.rs)
// and dispatches each pass to its `encode_*` method. Mirrors the Metal
// + Vulkan executors; every backend now drives the same builder.
//
// **Per-pass command lists.** Each non-composite pass records into its
// own `ID3D12GraphicsCommandList` (drawn from the `pass_cmd_lists` pool
// on `DxContext`). The fan-out runs on `jobs::pool()` via `rayon::scope`
// so workers encode in parallel; each worker resets its assigned
// allocator + cmd list, brackets the encode with start/end TIMESTAMP
// queries, encodes the pass, and closes the cmd list. The main thread
// then submits every closed cmd list in topological pass order via
// `ExecuteCommandLists`. The Composite pass keeps using the outer
// "end" cmd list that `draw_frame` owns (so the final timestamp +
// `ResolveQueryData` ride the same submission). Mirrors
// `metal/graph_exec.rs`.
//
// Per-pass `barriers_before` is dropped on DirectX; each
// migrated encoder owns its inline `D3D12_RESOURCE_BARRIER` calls.
//
// Bundled passes:
//   * `PassId::SsaoBlur` dispatches the bundled `encode_ssao` (which
//     internally encodes the SSAO pre-pass + GTAO kernel + depth-aware
//     blur). `PassId::SsaoPrepass` / `PassId::SsaoKernel` stay
//     timing-only and the executor rejects them as graph nodes.
//   * `PassId::ParticlesDraw` dispatches the bundled `encode_particles`
//     (compute sim + render draw). `PassId::ParticlesSim` stays
//     timing-only and the executor rejects it as a graph node.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::render_graph::{CompiledGraph, CompiledPass, GraphResourceClass, PassId};
use crate::gfx::render_types::TextDrawCall;

use super::barrier_translate::d3d12_transition;
use super::context::DxContext;
use super::parallel_encoder::{ParallelCtxRef, SendableCmdList, pool_index};
use super::texture::{aliasing_barrier, transition_barrier};

// One resolved barrier target: the D3D12 resource a graph resource backs, its
// class, and its resting state (created / cross-frame-restored). Built once per
// frame by `build_barrier_registry`; the resource is a refcount clone, read only
// to record transitions into a worker's command list.
struct DxBarrierTarget {
    resource: ID3D12Resource,
    class: GraphResourceClass,
    resting: D3D12_RESOURCE_STATES,
}

// `ResourceId`-indexed table of barrier targets for the migrated graph resources
// (`None` for every resource the executor doesn't graph-drive). A resource is
// graph-driven iff it has a `Some` entry, so this table is the single source of
// truth that replaced the old label allowlist + per-label resolver. Built on the
// main thread by `build_barrier_registry`, where the only field-naming of the
// migrated resources lives (so it is what re-cuts when those fields move into
// sub-structs); the parallel emit path stays field-agnostic.
struct DxBarrierRegistry(Vec<Option<DxBarrierTarget>>);

// SAFETY: same read-only contract as `ParallelCtxRef` / `SendableCmdList` (see
// `parallel_encoder.rs`). The registry holds refcount clones of D3D12 resource
// handles that workers only read, to record `ResourceBarrier` calls into their
// own command lists; every worker joins before the borrow that built the
// registry ends. D3D12 device-derived objects are thread-safe for shared read
// per Microsoft's free-threading rules.
unsafe impl Sync for DxBarrierRegistry {}

// Per-pass aliasing barriers, indexed by topological pass position: the pooled
// transients this pass first-writes that reclaim a shared heap region from an
// earlier transient. Built once per frame on the main thread; the resources are
// refcount clones workers only read.
struct DxAliasBarriers(Vec<Vec<ID3D12Resource>>);

// SAFETY: same read-only contract as `DxBarrierRegistry` above.
unsafe impl Sync for DxAliasBarriers {}

// Emit the aliasing barriers for a pass: for each pooled transient that reclaims
// a shared heap region here, announce the reuse, then re-initialize the resource
// so its first write is legal. The aliasing barrier leaves the memory's contents
// undefined and D3D12 rejects a placed render target's use until a
// Clear/Discard/Copy initializes it, so Discard each (in RENDER_TARGET, then
// back to its resting PIXEL_SHADER_RESOURCE state) before the producing pass's
// own resting -> RENDER_TARGET transition runs. The pass then fully overwrites
// it. Both managed transients rest sampled; a future non-sampled aliased member
// would need its resting state threaded through here.
fn emit_alias_barriers(cmd: &ID3D12GraphicsCommandList, resources: &[ID3D12Resource]) {
    const RESTING: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    for res in resources {
        unsafe {
            cmd.ResourceBarrier(&[aliasing_barrier(res)]);
            cmd.ResourceBarrier(&[transition_barrier(
                res,
                RESTING,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            )]);
            cmd.DiscardResource(res, None);
            cmd.ResourceBarrier(&[transition_barrier(
                res,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                RESTING,
            )]);
        }
    }
}

// Emit the native transitions for the migrated graph resources from a pass's
// `barriers_before`, resolved through the registry. Called at the start of each
// pass's own command list, before the pass encodes, so the transition lands in
// the same submission slot the prior inline barrier used to. A resource with no
// registry entry is skipped and keeps its inline barriers. Takes no `DxContext`:
// the field-to-resource mapping was already resolved into the registry, so this
// parallel path is field-agnostic.
fn emit_graph_barriers(
    cmd: &ID3D12GraphicsCommandList,
    registry: &DxBarrierRegistry,
    pass: &CompiledPass,
) {
    for op in &pass.barriers_before {
        let Some(Some(target)) = registry.0.get(op.resource_index()) else {
            continue;
        };
        if let Some((before, after)) = d3d12_transition(
            target.class,
            target.resting,
            op.from_state(),
            op.to_state(),
            op.read_stages(),
        ) {
            unsafe {
                cmd.ResourceBarrier(&[transition_barrier(&target.resource, before, after)]);
            }
        }
    }
}

// Per-frame params the executor threads into each pass's `encode_*`
// method. Mirrors the Vulkan executor's `GraphFrameParams`.
pub(in crate::directx) struct GraphFrameParams<'a> {
    pub cmd: &'a ID3D12GraphicsCommandList,
    pub frame_idx: usize,
    pub back_buffer: &'a ID3D12Resource,
    pub back_buffer_rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub text_calls: &'a [TextDrawCall],
    // Scene SRV the composite shader samples: TAA history when TAA is
    // on, SSR output when SSR is on and TAA is off, raw HDR scene SRV
    // otherwise. Computed in `record_frame` once before the dispatch;
    // stable across the whole graph because `taa.frame` only ticks
    // after Composite, so reading `taa.output_index()` upfront points
    // at the same TAA history slot the TaaResolve encoder will write
    // into and the Composite encoder samples.
    pub scene_srv: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Off-screen scene render resolution. Every scene pass (Shadow, Main,
    // SSAO, SSR, Velocity, Fog, Raymarch, Decals, Particles) rasterises at
    // this size, and the sub-pixel jitter is converted to NDC against it.
    // Equals `output_*` when temporal upscaling is off.
    pub width: u32,
    pub height: u32,
    // Drawable (swapchain) resolution. Only the Composite + text pass uses
    // these; it renders the fullscreen tonemap triangle into the
    // output-sized back buffer and sets the text overlay's window-dim
    // uniform from them. Bloom samples its own output-sized mip extents, so
    // it needs no dim param. Equals `width`/`height` when upscaling is off.
    pub output_width: u32,
    pub output_height: u32,
    // Camera world-space position. Shadow uses it for CSM cascade
    // distance bookkeeping inside `encode_shadow_pass`; Main uses it
    // for per-cluster distance culling and the SSAO bundle's pre-pass;
    // future migrating passes (SSAO standalone, SSR pre-pass,
    // Velocity) will share it.
    pub cam_pos: [f32; 3],
    // GPU virtual address of this frame's `ShadowUniforms` constant
    // buffer (the cached cascade VPs + light direction). Consumed by
    // Shadow and Main.
    pub shadow_ubo_gva: u64,
    // GPU virtual address of this frame's `ViewUniforms` constant
    // buffer. Consumed by Main.
    pub view_gva: u64,
    // GPU virtual address of the shared `LightUniforms` constant
    // buffer. Consumed by Main (and any future pass that lights the
    // scene).
    pub light_gva: u64,
    // Jittered camera view-projection matrix (sub-pixel Halton jitter
    // applied when TAA is on). Consumed by Main and Velocity (the
    // jittered VP path); when SSR-prepass migrates it shares this.
    pub vp_mat: [[f32; 4]; 4],
    // Un-jittered camera view-projection matrix. Velocity uses it
    // alongside `vp_mat` (jittered) and the prior frame's `prev_vp`
    // stored on the context, so the stored motion vector is free of
    // sub-pixel jitter.
    pub cur_vp: [[f32; 4]; 4],
    // Camera frustum derived from `vp_mat`. Consumed by Main's
    // per-cluster culling and the bundled SSAO pre-pass.
    pub frustum: &'a crate::gfx::frustum::Frustum,
    // Vertical FOV in radians. Consumed by Main's SSAO pre-pass
    // (depth-reconstruction geometry) and SsrResolve's ray-march
    // projection.
    pub fov_y_radians: f32,
    // Camera aspect ratio (width / height). Consumed by SsrResolve
    // for the ray-march projection.
    pub aspect: f32,
    // Seconds since the engine started. Consumed by ParticlesDraw's
    // bundled compute sim (delta-time computed against the last
    // per-emitter elapsed snapshot stored on `DxContext`).
    pub elapsed: f32,
    // Camera near-plane in view units. Consumed by `FogFroxel` to map the
    // front edge of the froxel volume onto view-space depth, and by
    // `Upscale` (FSR3 dispatch's `cameraNear`).
    pub near: f32,
    // Camera far-plane in view units. Consumed by `Upscale` (FSR3
    // dispatch's `cameraFar`).
    pub far: f32,
    // BVH-culled visible-object indices (sorted, with `always_draw`
    // appended). Consumed by Main's bindless + legacy + instanced
    // sub-passes.
    pub visible: &'a [u32],
}

impl DxContext {
    // Walk a compiled render graph and dispatch each non-composite pass
    // to its own freshly-reset per-pass `ID3D12GraphicsCommandList`,
    // fanning the encode work across rayon workers. Composite stays on
    // the outer "end" cmd list (`params.cmd`) the caller provides; the
    // final timestamp + `ResolveQueryData` ride the same submission.
    // Returns the closed per-pass cmd lists in topological pass order
    // (excluding composite); the caller submits them via
    // `ExecuteCommandLists` between the "start" outer cmd list (which
    // holds the timestamp pre-init) and the "end" outer cmd list (which
    // holds composite + post).
    //
    // `&self` mirrors every DirectX `encode_*` method; per-frame mutable
    // state lives behind `RwLock` / `Cell` / `AtomicU32` so the encoders
    // stay sound under the parallel fan-out (see
    // [`super::parallel_encoder`] for the Send/Sync contract).
    pub(in crate::directx) fn execute_graph(
        &self,
        graph: &CompiledGraph,
        params: &GraphFrameParams<'_>,
    ) -> Result<Vec<ID3D12GraphicsCommandList>, String> {
        // Find Composite's slot (if any) so we can skip it in the
        // worker fan-out and run it inline on the main thread instead.
        let composite_idx = graph.passes.iter().position(|p| p.id == PassId::Composite);

        // Slot per graph pass: each worker stashes its closed cmd list
        // here on success, indexed by topological position. Main thread
        // collects them into the return Vec in order after the join.
        let worker_slots: Mutex<Vec<Option<SendableCmdList>>> =
            Mutex::new((0..graph.passes.len()).map(|_| None).collect());
        let first_error: Mutex<Option<String>> = Mutex::new(None);

        let ctx_ref = ParallelCtxRef::new(self);
        // Resolve every migrated resource's barrier target once, on the main
        // thread, then share the table read-only into the parallel pass workers.
        let registry = self.build_barrier_registry(graph);
        let registry_ref = &registry;
        // Likewise resolve the per-pass aliasing barriers (which pooled transients
        // reclaim a shared heap region) once, shared read-only into the workers.
        let alias_barriers = self.build_alias_barriers(graph);
        let alias_barriers_ref = &alias_barriers;
        let frame_idx = params.frame_idx;

        crate::jobs::pool().install(|| {
            rayon::scope(|scope| {
                for (idx, pass) in graph.passes.iter().enumerate() {
                    if Some(idx) == composite_idx {
                        continue;
                    }
                    let pass_id = pass.id;
                    let first_error_ref = &first_error;
                    let worker_slots_ref = &worker_slots;
                    scope.spawn(move |_| {
                        let ctx = ctx_ref.as_ctx();
                        let pool_idx = pool_index(frame_idx, pass_id);
                        let alloc = &ctx.commands.pass_allocators[pool_idx];
                        let cmd = &ctx.commands.pass_cmd_lists[pool_idx];

                        // Reset this pass's allocator + cmd list so we
                        // can record fresh into it. The previous frame's
                        // submission for this same (frame, pass) slot
                        // has already retired by the time we get here
                        // (the FRAMES-deep fence wait at the top of
                        // `draw_frame` gates the entire slot).
                        if let Err(e) = unsafe { alloc.Reset() } {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(format!(
                                    "per-pass allocator reset ({}): {e}",
                                    pass_id.name()
                                ));
                            }
                            return;
                        }
                        if let Err(e) = unsafe { cmd.Reset(alloc, None) } {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(format!(
                                    "per-pass cmd list reset ({}): {e}",
                                    pass_id.name()
                                ));
                            }
                            return;
                        }

                        // Per-pass GPU timing: bracket the encoder with
                        // start + end TIMESTAMP `EndQuery` calls into
                        // pre-allocated heap slots. The frame's whole
                        // block is resolved by the "end" outer cmd list
                        // at the end of the frame and read back at the
                        // top of the next frame. See
                        // [`super::pass_timing`] for the slot layout.
                        if let Some(heap) = ctx.timestamps.query_heap.as_ref() {
                            let (start_slot, _) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            unsafe {
                                cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot);
                            }
                        }

                        // Aliasing barriers first: claim + re-initialize any
                        // pooled transient this pass first-writes that reuses a
                        // shared heap region, before its resting -> RENDER_TARGET
                        // transition below.
                        emit_alias_barriers(cmd, &alias_barriers_ref.0[idx]);

                        // Graph-driven transitions for the migrated resources
                        // (replacing the encoder's stripped inline barriers), then
                        // the pass body. Resolved through the registry, so this
                        // path names no `DxContext` fields.
                        emit_graph_barriers(cmd, registry_ref, pass);

                        let encode_result = ctx.encode_pass_into(pass_id, cmd, params);

                        if let Some(heap) = ctx.timestamps.query_heap.as_ref() {
                            let (_, end_slot) = super::pass_timing::pass_pair(frame_idx, pass_id);
                            unsafe {
                                cmd.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot);
                            }
                        }

                        if let Err(e) = unsafe { cmd.Close() } {
                            let mut lock = first_error_ref.lock().unwrap();
                            if lock.is_none() {
                                *lock = Some(format!(
                                    "per-pass cmd list close ({}): {e}",
                                    pass_id.name()
                                ));
                            }
                            return;
                        }

                        match encode_result {
                            Ok(()) => {
                                let mut lock = worker_slots_ref.lock().unwrap();
                                lock[idx] = Some(SendableCmdList(cmd.clone()));
                            }
                            Err(e) => {
                                let mut lock = first_error_ref.lock().unwrap();
                                if lock.is_none() {
                                    *lock = Some(e);
                                }
                            }
                        }
                    });
                }
            });
        });

        if let Some(err) = first_error.into_inner().unwrap_or(None) {
            return Err(err);
        }

        // Composite stays on the outer "end" cmd list (`params.cmd`) the
        // caller supplied. The final timestamp `EndQuery` +
        // `ResolveQueryData` are appended onto the same cmd list by
        // `draw_frame` after this returns, so composite + post-resolve
        // ride one submission.
        if composite_idx.is_some() {
            if let Some(heap) = self.timestamps.query_heap.as_ref() {
                let (start_slot, _) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                unsafe {
                    params
                        .cmd
                        .EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, start_slot);
                }
            }
            self.encode_composite_and_text(
                params.cmd,
                params.frame_idx,
                params.back_buffer,
                params.back_buffer_rtv,
                params.text_calls,
                params.scene_srv,
                // Composite runs at drawable resolution; it samples the
                // (output-sized) upscaler result / scene SRV through a
                // fullscreen triangle and writes the output-sized back
                // buffer. Under upscaling this differs from the scene
                // render dims in `params.width`/`height`.
                params.output_width,
                params.output_height,
            )?;
            if let Some(heap) = self.timestamps.query_heap.as_ref() {
                let (_, end_slot) = super::pass_timing::pass_pair(frame_idx, PassId::Composite);
                unsafe {
                    params
                        .cmd
                        .EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, end_slot);
                }
            }
        }

        // Collect every worker-encoded cmd list in topological pass
        // order. The empty slots (composite, plus any skipped no-op
        // pass that returned without stashing) drop out; workers only
        // stash on success.
        let slots = worker_slots
            .into_inner()
            .map_err(|_| "graph executor (directx): worker slot mutex poisoned".to_string())?;
        let mut ordered = Vec::with_capacity(graph.passes.len());
        for cb in slots.into_iter().flatten() {
            ordered.push(cb.0);
        }
        Ok(ordered)
    }

    // Resolve every migrated graph resource to its barrier target, indexed by
    // `ResourceId` (its position in `graph.resources`), so the parallel emit path
    // can look a target up by `BarrierOp::resource_index()`. This is the single
    // place that names the migrated resources' backing `DxContext` fields;
    // field-grouping re-cuts here, not in the executor. A resource the owning
    // feature disabled (or one never migrated) gets `None`, and the graph carries
    // no barrier for it either.
    fn build_barrier_registry(&self, graph: &CompiledGraph) -> DxBarrierRegistry {
        DxBarrierRegistry(
            graph
                .resources
                .iter()
                .map(|res| self.barrier_target_for_label(res.label))
                .collect(),
        )
    }

    // Resolve, per pass, the pooled transients that reclaim a shared heap region
    // when this pass first-writes them. A resource is aliased iff the pool gives
    // it a slot predecessor; its aliasing barrier lands before the pass at its
    // `lifetime.first`. Empty for every resource the pool does not alias, so the
    // table is empty whenever no slot is shared this frame (e.g. bloom off leaves
    // `ao_output` aliased but `bloom_top` absent from the graph; ssao off leaves
    // `bloom_top` un-aliased). Mirrors the Vulkan executor's `build_alias_barriers`.
    fn build_alias_barriers(&self, graph: &CompiledGraph) -> DxAliasBarriers {
        let mut table: Vec<Vec<ID3D12Resource>> = vec![Vec::new(); graph.passes.len()];
        for res in &graph.resources {
            if self.transient_pool.alias_predecessor(res.label).is_none() {
                continue;
            }
            if let Some(r) = self.transient_pool.resource_for(res.label) {
                let first = res.lifetime.first;
                if first < table.len() {
                    table[first].push(r.clone());
                }
            }
        }
        DxAliasBarriers(table)
    }

    // Map one graph resource label to its backing D3D12 resource, class, and
    // resting state (the state it was created in and returns to at the end of
    // every frame, via the encoder's kept restore barrier). The resting state
    // lets the executor translate a first-use `Undefined` transition into one
    // whose `from` matches the resource's real state.
    fn barrier_target_for_label(&self, label: &str) -> Option<DxBarrierTarget> {
        match label {
            "ao_output" => self
                .transient_pool
                .resource_for("ao_output")
                .map(|r| DxBarrierTarget {
                    resource: r.clone(),
                    class: GraphResourceClass::ColorTarget,
                    resting: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                }),
            // shadow_map rests sampled (PIXEL_SHADER_RESOURCE): the Shadow
            // producer barrier (Undefined -> Write) is the real
            // PIXEL_SHADER_RESOURCE -> DEPTH_WRITE reset for this frame's shadow
            // loop, and the Main consumer (Write -> Read) returns it to sampled.
            // Both transitions are graph-driven; there is no inline cross-frame
            // reset (the map is created sampled, so frame 0's producer barrier
            // starts from the resource's real state).
            "shadow_map" => self
                .shadow
                .resource
                .as_ref()
                .filter(|_| !self.shadow.dsvs.is_empty())
                .map(|s| DxBarrierTarget {
                    resource: s.resource.clone(),
                    class: GraphResourceClass::DepthTarget,
                    resting: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                }),
            // fog_froxel_volume rests sampled (PIXEL_SHADER_RESOURCE): the
            // FogFroxel producer (Undefined -> Write) is the real
            // PIXEL_SHADER_RESOURCE -> UNORDERED_ACCESS open for the compute
            // write, and the Fog consumer (Write -> Read) the matching UAV ->
            // PIXEL_SHADER_RESOURCE close for the trilinear sample. Both are
            // graph-driven; there is no inline reset (the volume is created
            // sampled, matching Vulkan, which already graph-drives both).
            "fog_froxel_volume" => self.fog.resources.as_ref().map(|f| DxBarrierTarget {
                resource: f.volume_resource.clone(),
                class: GraphResourceClass::StorageImage,
                resting: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            }),
            _ => None,
        }
    }

    // Build the per-frame `RaymarchView` cbuffer payload from the
    // graph executor's frame params. The matrix inputs match what the
    // Main pass rasterises with (the un-jittered VP), so raymarched
    // surfaces share their NDC depth space with rasterised geometry.
    fn build_raymarch_view(&self, params: &GraphFrameParams<'_>) -> super::raymarch::RaymarchView {
        let inv_vp = super::math::mat4_inverse(params.cur_vp);
        super::raymarch::RaymarchView {
            vp: params.cur_vp,
            inv_vp,
            cam_pos: params.cam_pos,
            _pad0: 0.0,
            viewport: [params.width as f32, params.height as f32],
            time: params.elapsed,
            prefilter_mip_count: self.env_map.prefilter_mip_count as f32,
        }
    }

    // Build the per-frame `TransparentView` cbuffer payload for the glass pass.
    // Uses the jittered VP (`vp_mat`) the Main pass rasterised with, so the
    // glass quad's clip-space depth matches the stored main-depth the fragment
    // shader tests against. Mirrors `encode_decals`' use of `vp_mat`.
    fn build_transparent_view(
        &self,
        params: &GraphFrameParams<'_>,
    ) -> super::glass::TransparentViewGpu {
        let inv_vp = super::math::mat4_inverse(params.vp_mat);
        super::glass::TransparentViewGpu {
            vp: params.vp_mat,
            inv_vp,
            camera_pos: [params.cam_pos[0], params.cam_pos[1], params.cam_pos[2], 0.0],
            viewport: [params.width as f32, params.height as f32],
            time: params.elapsed,
            _pad: 0.0,
        }
    }

    // Per-pass dispatch, called from both the worker fan-out and the
    // main-thread composite arm. Each arm encodes onto the `cmd` it's
    // given (the worker's per-pass cmd list, or the outer "end" cmd
    // list for composite). Composite is **not** routed through this
    // method; the caller calls `encode_composite_and_text` directly
    // so the trailing timestamp + resolve land on the same cmd list.
    fn encode_pass_into(
        &self,
        pass_id: PassId,
        cmd: &ID3D12GraphicsCommandList,
        params: &GraphFrameParams<'_>,
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
                self.encode_ssao(cmd, params.fov_y_radians, params.aspect);
            }
            PassId::SsaoPrepass | PassId::SsaoKernel => {
                return Err(format!(
                    "graph executor (directx): pass {} is bundled inside SsaoBlur \
                     (encode_ssao encodes all three SSAO sub-passes); it \
                     should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::SsrPrepass => {
                // Merged into GBufferPrepass on DX: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (directx): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::Shadow => {
                // Build the raymarch view only when at least one volume
                // opted in to shadow casting; otherwise pass `None` so
                // the shadow encoder skips the SDF caster sub-pass with
                // zero overhead. The view stays consistent with the
                // matching `PassId::Raymarch` build later this frame:
                // same `cur_vp`, `cam_pos`, `elapsed`, viewport, and
                // prefilter mip count.
                let raymarch_view = self
                    .raymarch
                    .as_ref()
                    .filter(|rm| rm.any_shadow_casters())
                    .map(|_| self.build_raymarch_view(params));
                self.encode_shadow_pass(
                    cmd,
                    params.frame_idx,
                    params.shadow_ubo_gva,
                    params.cam_pos,
                    raymarch_view.as_ref(),
                );
            }
            PassId::AutoExposure => {
                self.encode_auto_exposure(cmd, params.frame_idx);
            }
            PassId::Main => {
                self.encode_main_pass(
                    cmd,
                    params.frame_idx,
                    params.width,
                    params.height,
                    params.view_gva,
                    params.light_gva,
                    params.shadow_ubo_gva,
                    params.frustum,
                    params.cam_pos,
                    params.visible,
                );
            }
            PassId::Decals => {
                self.encode_decals(cmd, params.frame_idx, params.vp_mat, params.frustum);
            }
            PassId::Fog => {
                self.encode_fog(cmd, params.frame_idx, params.vp_mat, params.cam_pos);
            }
            PassId::ParticlesDraw => {
                self.encode_particles(
                    cmd,
                    params.frame_idx,
                    params.elapsed,
                    params.vp_mat,
                    params.frustum,
                );
            }
            PassId::ParticlesSim => {
                return Err(format!(
                    "graph executor (directx): pass {} is bundled inside ParticlesDraw \
                     (encode_particles runs both compute sim and render); it \
                     should not appear as its own graph node",
                    pass_id.name()
                ));
            }
            PassId::SsrResolve => {
                self.encode_ssr_resolve(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::Velocity => {
                // Merged into GBufferPrepass on DX: the builder emits the
                // unified node (unified_gbuffer_prepass = true) and never this.
                return Err(format!(
                    "graph executor (directx): pass {} is merged into GBufferPrepass \
                     and should not appear in the frame graph",
                    pass_id.name()
                ));
            }
            PassId::TaaResolve => {
                self.encode_taa(cmd);
            }
            PassId::Bloom => {
                self.encode_bloom(cmd, params.scene_srv);
            }
            PassId::Composite => {
                // Composite is run inline on the outer "end" cmd list
                // by `execute_graph` itself so it shares a submission
                // with the trailing timestamp + resolve. This arm is
                // unreachable through the worker fan-out; see the
                // method docstring.
                return Err(
                    "graph executor (directx): Composite must run on the outer cmd \
                     list: encode_pass_into is not the right entry point"
                        .into(),
                );
            }
            PassId::Raymarch => {
                let view = self.build_raymarch_view(params);
                self.encode_raymarch(cmd, params.frame_idx, &view)?;
            }
            PassId::FogFroxel => {
                self.encode_fog_froxel(
                    cmd,
                    params.frame_idx,
                    params.near,
                    params.vp_mat,
                    params.cam_pos,
                    params.shadow_ubo_gva,
                );
            }
            PassId::Upscale => {
                // FSR3 temporal upscaler. Driven by the shared graph
                // when `FrameGraphInputs::upscale_enabled` is on (see
                // `record_frame::seed_inputs`). The encoder dispatches
                // FFX against this pass's per-pass cmd list, reading
                // the post-SSR scene + velocity + main depth and
                // writing into the upscaler's output texture (which
                // bloom + composite then sample via `scene_srv_for_post`).
                self.encode_upscale(cmd, params)?;
            }
            PassId::Transparent => {
                // Generic translucent pass: draws the world's glass panels
                // back-to-front over the post-SSR scene. Gated by
                // `FrameGraphInputs::transparent_enabled`
                // (`DxContext::transparent_enabled`), so it only appears when
                // the world declared visible `GlassPanel`s. Water is a separate
                // (Metal-only) producer not ported here.
                let view = self.build_transparent_view(params);
                self.encode_transparent(cmd, params.frame_idx, &view)?;
            }
            PassId::HizBuild => {
                // Two-pass occlusion: rebuild the Hi-Z pyramid mid-frame from
                // this frame's phase-1 depth so Cull2 re-tests the phase-1
                // occluded objects against up-to-date depth. The same
                // `encode_hiz_build` the end-of-frame (next-frame) build uses,
                // just dispatched here as a graph node ordered after Main.
                self.encode_hiz_build(cmd);
            }
            PassId::Cull2 => {
                self.encode_cull_phase2(cmd, params.frame_idx, params.frustum, params.cur_vp);
            }
            PassId::Main2 => {
                self.encode_main_pass_phase2(
                    cmd,
                    params.frame_idx,
                    params.width,
                    params.height,
                    params.view_gva,
                    params.light_gva,
                    params.shadow_ubo_gva,
                );
            }
            PassId::Ssgi => {
                self.encode_ssgi(cmd, params.frame_idx, params.fov_y_radians, params.aspect);
            }
            PassId::RtReflections => {
                // Hardware ray-traced reflections (DXR inline `RayQuery`). Traces
                // a reflection ray per glossy pixel against the scene TLAS and
                // composites into the RT output target, which `scene_srv_for_post`
                // then feeds the post stack. Occupies the SsrResolve slot; gated
                // by `FrameGraphInputs::rt_reflections_enabled`
                // (`DxContext::rt_reflections_active`). The per-frame TLAS update
                // already ran on the outer "start" cmd list before this trace.
                self.encode_rt_reflections(
                    cmd,
                    params.frame_idx,
                    params.fov_y_radians,
                    params.aspect,
                    params.cam_pos,
                );
            }
            PassId::GBufferPrepass => {
                // Unified geometry pre-pass: one jittered traversal writes
                // normal+depth, roughness, and motion for every screen-space
                // consumer (SSR / SSAO / SSGI / TAA / FSR). `params.vp_mat` is
                // the jittered VP (rasterisation, matching the main pass);
                // `params.cur_vp` is the un-jittered VP the shader uses with the
                // previous VP for the motion vector. The velocity channel
                // carries real motion only when a consumer reads it (TAA or
                // FSR active, i.e. `self.taa.is_some()`); otherwise cur == prev
                // and it stays a harmless zero.
                self.encode_gbuffer_prepass(
                    cmd,
                    params.frame_idx,
                    params.vp_mat,
                    params.cur_vp,
                    params.visible,
                    params.frustum,
                    params.cam_pos,
                    self.taa.is_some(),
                );
            }
        }
        Ok(())
    }
}
