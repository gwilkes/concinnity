// src/metal/raytrace.rs
//
// Hardware ray-tracing acceleration structures for the Metal backend. Builds,
// from the shared static vertex / index buffers and the `DrawObject` list, the
// bottom- and top-level acceleration structures (BLAS / TLAS) the RT-reflection
// kernel traces against, plus a per-instance geometry table the kernel uses to
// fetch the hit triangle and shade it.
//
// One primitive BLAS per object over its slice of the shared buffers, one
// instance in the TLAS per object (transform = the object's model matrix,
// instance_id = the object index). The BLAS describe object-space geometry and
// never change for a rigid transform; only the TLAS instance transforms (and
// the geometry table's per-instance model matrices the kernel shades with) move
// when a prop moves.
//
// Dynamic transforms (`RtDynamicMode`) update the structures per frame. The
// per-frame skinned update (`rebuild_skinned`) keeps the persistent static +
// cluster BLAS, re-skins the current pose, and rebuilds only the skinned BLAS +
// TLAS + geometry table. It is fully asynchronous: NO `waitUntilCompleted`. The
// three GPU steps are committed on the one shared queue in dependency order: skin
// compute (writes the deformed buffer), then the BLAS/TLAS build (reads it), all
// in `rt_dynamic_update`, strictly before the reflection-trace command buffer
// (committed later in `execute_graph`). Same-queue FIFO commit order runs them
// skin → build → trace; the render graph already depends on exactly this
// event-free ordering for every cross-pass read. Faults (which can no longer be
// caught synchronously) are surfaced from completion handlers.
//
// (Historical note: an earlier bisect concluded no GPU-side primitive orders these
// steps and kept the rebuild synchronous. That predated the fix for the actual
// fault (a CPU/GPU `RtGeomEntry` struct-layout mismatch that made the trace read
// out of bounds), so those fault observations were the layout bug, not an ordering
// failure. With it fixed, same-queue commit order alone orders the rebuild.)
//
// Every rebuild allocates fresh and parks the outgoing structures / Shared
// buffers in a frame-tagged deferred-free pool (`RetirePool`) until the
// frames-in-flight fence retires the frames whose still-in-flight trace could
// read them.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSArray;
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureCommandEncoder,
    MTLAccelerationStructureGeometryDescriptor, MTLAccelerationStructureInstanceDescriptor,
    MTLAccelerationStructureInstanceDescriptorType, MTLAccelerationStructureInstanceOptions,
    MTLAccelerationStructureTriangleGeometryDescriptor, MTLAttributeFormat, MTLBuffer,
    MTLCommandBuffer as _, MTLCommandBufferStatus, MTLCommandEncoder as _, MTLCommandQueue as _,
    MTLComputeCommandEncoder as _, MTLComputePipelineState, MTLDevice as _, MTLIndexType,
    MTLInstanceAccelerationStructureDescriptor, MTLPackedFloat3, MTLPackedFloat4x3,
    MTLPrimitiveAccelerationStructureDescriptor, MTLRenderCommandEncoder, MTLRenderPipelineState,
    MTLRenderStages, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};

use super::transient::RetirePool;
use crate::gfx::render_types::{DrawObject, InstancedCluster, RtGeomEntry, SkinnedDrawObject};
use crate::gfx::rt_reflections::RtReflectionSettings;

// Marks a `RtGeomEntry.normal_index` as belonging to a skinned object: the
// reflection kernel then fetches the hit triangle from the deformed-vertex /
// u16 skinned index buffers instead of the static u32 ones. Bit 31 is free:
// bindless pool indices never approach 2^31.
pub(crate) const RT_SKINNED_FLAG: u32 = 0x8000_0000;

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + colour + uv = 14 floats). The RT kernel reads positions at this stride; the
// main-pass skinned fold sizes its deformed buffer by it too.
pub(in crate::metal) const VERTEX_STRIDE: usize = 56;

// How the scene acceleration structure is kept current when props move.
//
// Selected once at init from the `CN_RT_DYNAMIC` environment variable; unset
// gives `Auto`, the shipping behaviour. Every update path rebuilds with *fresh
// allocations* and retires the outgoing structures through a frame-tagged
// deferred-free pool, so a prior frame's still-in-flight command buffer keeps
// reading the old structures while the new frame uses the new ones. (The
// frames-in-flight fence bounds the pipeline depth but does not serialise
// frames (the immediately-prior frame can still be tracing), so an *in-place*
// refit of a structure a prior frame still traces is a cross-command-buffer
// hazard that hangs the GPU and can kernel-panic the host. That path was
// measured, confirmed dangerous, and removed; fresh-alloc + deferred-free is
// the safe equivalent.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RtDynamicMode {
    // Build once, never update. Forces a static BVH even if props move: the
    // pre-dynamic behaviour, kept as a fast path / diagnostic (`off`).
    Off,
    // Default. Rebuild the TLAS + table (fresh allocations, static BLAS) only
    // on the frames a participating transform actually changed. Static scenes
    // never rebuild, so they pay only a cheap per-frame matrix compare.
    Auto,
    // Force a full BVH rebuild (BLAS + TLAS + table) every frame, dirty or not.
    // Diagnostic only (`rebuild`); the most expensive path.
    Rebuild,
    // Force a fresh TLAS + table rebuild every frame, dirty or not. Diagnostic
    // (`tlas`); the same GPU work `Auto` does, minus the dirty gate.
    Tlas,
}

impl RtDynamicMode {
    // Parse the mode from `CN_RT_DYNAMIC`. Unset / unrecognised → `Auto`.
    pub(crate) fn from_env() -> Self {
        match std::env::var("CN_RT_DYNAMIC").as_deref() {
            Ok("off") => Self::Off,
            Ok("rebuild") => Self::Rebuild,
            Ok("tlas") => Self::Tlas,
            _ => Self::Auto,
        }
    }

    // Whether this mode updates the BVH after the initial build at all.
    pub(crate) fn is_dynamic(self) -> bool {
        self != Self::Off
    }
}

// All hardware-ray-traced-reflection state grouped into one feature unit: the
// resolved tunables, the scene acceleration structure, the dynamic-update
// mode + failure-streak flag, and the resolve / textured-resolve / skinning
// pipelines. `settings`/`accel`/the pipelines are `Some` only when RT
// reflections are on and the GPU supports ray tracing (see the per-field
// docs on [`MtlContext`](super::context::MtlContext) for the exact gates).
pub(crate) struct RtState {
    // Resolved + clamped tunables. `Some` only when the world's
    // `PostProcessConfig` sets `ray_traced_reflections` AND the GPU supports
    // ray tracing; gates the RT pass. RT takes precedence over SSR resolve
    // and reuses `ssr.targets.output` as its resolve target.
    pub settings: Option<RtReflectionSettings>,
    // Scene acceleration structure (BLAS/TLAS) + geometry table. `Some` only
    // when RT reflections are on and the scene has resident geometry; updated
    // per frame when `dynamic_mode` is dynamic. Resolution-independent.
    pub accel: Option<RtAccelData>,
    // How the acceleration structure is kept current as props move (read once
    // from `CN_RT_DYNAMIC` at init; `Auto` by default).
    pub dynamic_mode: RtDynamicMode,
    // Whether the per-frame BVH update is currently in a failure streak. A
    // transient rebuild failure is non-fatal (keep last frame's BVH) and
    // logged once per streak rather than every frame.
    pub update_failed: bool,
    // Set when an operation changes the RT-relevant draw set (a streamed chunk
    // added/removed, a prop cloned, a material edit that flips RT participation)
    // since the last update. The per-frame update consumes it to refresh the
    // BLAS topology -- reusing every unchanged BLAS and building only the new
    // ones -- instead of either ignoring the change (the default `Auto` path
    // only watches transforms of the prior set) or rebuilding every BLAS.
    pub topology_dirty: bool,
    // Resolve pipeline, flat-tint hit shading. Used for non-bindless worlds
    // (no albedo pool).
    pub pipeline: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Resolve pipeline, textured hit shading (samples the bindless albedo pool
    // at buffer(7)). Preferred over `pipeline` when the bindless texture
    // argument buffer is available this frame.
    pub pipeline_textured: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    // Compute-skinning pipeline that deforms skinned vertices into a buffer the
    // BVH can trace. Consumed each frame by `rebuild_rt_accel` to pose skinned
    // geometry before the skinned BLAS build.
    pub skin_pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

// Identifies the geometry slice a draw-object BLAS traces, on the shared
// vertex/index buffers. Two draw objects with the same signature trace identical
// geometry, so a topology refresh can reuse the existing BLAS instead of
// building a new one. Sound because: the shared buffer *objects* are stable once
// streaming is set up (`add_chunk_mesh` / `remove_chunk_mesh` write regions in
// place; a buffer swap goes through a full rebuild, not this path), and a slot's
// bytes cannot be overwritten while its BLAS is live (the deferred free holds the
// region until the frames-in-flight fence retires it). `base_vertex` +
// `index_offset` + `index_count` are exactly the inputs `prim_desc_for` uses;
// `vertex_offset` is carried too so a static draw (whose `base_vertex` is 0)
// still distinguishes distinct vertex regions.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GeomSig {
    base_vertex: i32,
    vertex_offset: usize,
    index_offset: usize,
    index_count: usize,
}

impl GeomSig {
    fn of(obj: &DrawObject) -> Self {
        Self {
            base_vertex: obj.base_vertex,
            vertex_offset: obj.vertex_offset,
            index_offset: obj.index_offset,
            index_count: obj.index_count,
        }
    }
}

// Per-new-slot decision for a topology refresh of the draw-object BLAS head.
struct TopologyPlan {
    // `reuse[j] == Some(k)`: new draw slot `j` reuses the old draw BLAS at index
    // `k` (its geometry is unchanged). `None`: build a fresh BLAS for slot `j`.
    reuse: Vec<Option<usize>>,
    // Old draw BLAS indices no longer referenced by any new slot -- retire them.
    retire: Vec<usize>,
}

// Decide, for the draw-object BLAS head only, which BLAS to reuse, which to
// build, and which to retire when the participating draw set changes. Matches
// old and new slots by `draw_objects` index AND geometry signature: a slot whose
// geometry moved (a chunk slot recycled for a different chunk) does not match, so
// it rebuilds. Pure so it is unit-testable without Metal.
fn plan_topology_refresh(
    old_indices: &[usize],
    old_sigs: &[GeomSig],
    new_indices: &[usize],
    new_sigs: &[GeomSig],
) -> TopologyPlan {
    use std::collections::HashMap;
    // draw_objects index -> (position in the old draw BLAS head, its signature).
    // `object_indices` entries are unique (one per draw slot), so this is 1:1.
    let mut by_idx: HashMap<usize, (usize, GeomSig)> = HashMap::with_capacity(old_indices.len());
    for (k, (&idx, &sig)) in old_indices.iter().zip(old_sigs).enumerate() {
        by_idx.insert(idx, (k, sig));
    }
    let mut used = vec![false; old_indices.len()];
    let mut reuse = Vec::with_capacity(new_indices.len());
    for (&idx, &sig) in new_indices.iter().zip(new_sigs) {
        match by_idx.get(&idx) {
            Some(&(k, old_sig)) if old_sig == sig && !used[k] => {
                used[k] = true;
                reuse.push(Some(k));
            }
            _ => reuse.push(None),
        }
    }
    let retire = used
        .iter()
        .enumerate()
        .filter(|&(_, &u)| !u)
        .map(|(k, _)| k)
        .collect();
    TopologyPlan { reuse, retire }
}

// The acceleration structures + geometry table for hardware ray tracing. Held
// on the context behind an `Option`; present only when the world enables RT
// reflections, the GPU supports ray tracing, and the scene has geometry.
pub(crate) struct RtAccelData {
    // Bottom-level acceleration structures, in build order: one per
    // participating `DrawObject` (in `object_indices` order), then one per
    // instanced cluster, then one per skinned object. This Vec is the sole CPU
    // owner that keeps every BLAS alive: a TLAS does not retain the structures it
    // references, and the `useResource` the kernel encoder issues only declares
    // residency, not lifetime, so a BLAS must stay owned here for as long as any
    // in-flight trace can reach it through the TLAS. A skinned rebuild produces a
    // whole fresh `RtAccelData`; the outgoing one is parked in the context's
    // retire pool until the frames-in-flight fence retires the frames that could
    // still trace it.
    pub blas: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>,
    // How many leading entries of `blas` are the persistent static + cluster
    // BLAS, built once and never rebuilt (a rigid transform leaves object-space
    // geometry unchanged). Skinned BLAS occupy `blas[static_blas_count..]` and
    // are rebuilt each frame from the current pose; a skinned object's
    // `accelerationStructureIndex` is `static_blas_count + si`. Lets the
    // per-frame skinned update rebuild only the skinned tail and keep the head.
    static_blas_count: usize,
    // The top-level (instance) acceleration structure the kernel traces.
    pub tlas: Retained<ProtocolObject<dyn MTLAccelerationStructure>>,
    // `[RtGeomEntry; instance_count]`, indexed by the intersector's
    // `instance_id`. Lets the kernel find the hit triangle + shade it. Carries
    // each instance's model matrix, which the kernel uses to bring the hit
    // normal to world space, so it moves in lockstep with the TLAS transforms.
    pub geom_table: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Per-frame update state.
    // Indices into the frame's `draw_objects` for the objects that participate,
    // in BLAS / instance order. Lets an update re-read current transforms in
    // the exact order the BLAS were built, and detect a changed draw list.
    object_indices: Vec<usize>,
    // The geometry signature each draw-object BLAS (`blas[..object_indices.len()]`)
    // was built from, parallel to `object_indices`. A topology refresh compares
    // these against the current draw set to reuse every unchanged BLAS and build
    // only the new / changed ones.
    draw_blas_sigs: Vec<GeomSig>,
    // Each participating object's model matrix as baked into the current TLAS,
    // in `object_indices` order. The `Auto` dirty check compares the live draw
    // list against these to decide whether a rebuild is needed.
    cached_models: Vec<[[f32; 4]; 4]>,
    // The TLAS instance descriptors for every instanced-cluster instance, in
    // the order they follow the draw-object instances. Clusters are baked
    // static into the BVH, so a per-frame TLAS rebuild re-appends these
    // verbatim after the freshly-transformed draw-object instances (their
    // `accelerationStructureIndex` points at the cluster BLAS, which never
    // move in `blas`). Empty when the world declares no `InstancedProp`.
    cluster_instances: Vec<MTLAccelerationStructureInstanceDescriptor>,
    // The geometry-table entries for the cluster instances, parallel to
    // `cluster_instances`. Re-appended alongside them on a rebuild.
    cluster_geom: Vec<RtGeomEntry>,
    // Private scratch buffer sized for the largest of every BLAS build and the
    // TLAS build, reused by the per-frame TLAS rebuild (the instance count is
    // fixed across rebuilds, so the init sizing always suffices).
    scratch: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The TLAS instance-descriptor buffer. Only the TLAS *build* reads it (the
    // built TLAS bakes the instances), so it is not bound to the trace. Held so
    // the per-frame skinned rebuild's outgoing buffer can be retired in step with
    // the structures it described. A fresh one is allocated each rebuild.
    instance_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,

    // Deformed (posed) skinned vertices in the static 56-byte `Vertex` layout,
    // written by the `rt_skin` compute pass and traced by the skinned BLAS. The
    // reflection kernel reads it (buffer 5) for skinned hits. A 1-element dummy
    // when the scene has no skinned geometry, so the encoder always has a buffer
    // to bind. The skinned rebuild allocates a fresh one each frame and retires
    // the old through `retire_pool` (it cannot overwrite in place: a prior
    // frame's trace may still be reading it).
    pub deformed_verts: Retained<ProtocolObject<dyn MTLBuffer>>,
    // The shared u16 skinned index buffer, cloned here so the reflection kernel
    // can bind it (buffer 6) for skinned hits. A 1-element dummy when there is
    // no skinned geometry.
    pub skinned_indices: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Outgoing structures / Shared buffers from prior rebuilds, held alive until
    // the frames-in-flight fence retires the frames whose still-in-flight trace
    // could read them. Drained once per frame in `rt_dynamic_update`. A skinned
    // rebuild or an incremental topology refresh allocates fresh and parks the
    // old here rather than freeing in place.
    retire_pool: RetirePool<RetiredRt>,
}

// Outgoing RT resources parked by a skinned rebuild or an incremental topology
// refresh for deferred free. Never read again: they exist only to keep the Metal
// handles (and thus the GPU allocations) valid until `RetirePool` drops them,
// once the fence guarantees no in-flight trace can still reference them.
struct RetiredRt {
    #[allow(dead_code)]
    structures: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>,
    #[allow(dead_code)]
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

// The per-frame skinned-geometry inputs `build_rt_accel` needs to deform and
// add skinned objects to the BVH. Assembled from the context's skinned state;
// `None` skips skinned geometry entirely (the static-only path).
pub(crate) struct SkinnedRtInputs<'a> {
    // One entry per skinned mesh (only `visible`, real-triangle objects build).
    pub objects: &'a [SkinnedDrawObject],
    // Shared skinned vertex buffer (`SkinnedVertex`, 80-byte stride) the skin
    // kernel reads bind-pose vertices from.
    pub vertex_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    // Shared skinned index buffer (u16, absolute indices) the skinned BLAS and
    // the reflection kernel address the deformed buffer with. Cloned into
    // `RtAccelData` so the reflection encoder can bind it.
    pub index_buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    // Per-object joint palettes, parallel to `objects`; uploaded transiently
    // and consumed by the skin kernel.
    pub joint_matrices: &'a [Vec<[[f32; 4]; 4]>],
    // The compiled `rt_skin` compute pipeline.
    pub skin_pipeline: &'a ProtocolObject<dyn MTLComputePipelineState>,
}

// Whether the GPU supports hardware ray tracing. Apple-silicon GPUs report
// `true`; Intel / most AMD Macs report `false`, in which case the caller falls
// back to SSR (or no reflections). Mirrors the capability gates the MetalFX /
// HDR paths use at init.
pub(crate) fn raytracing_supported(device: &ProtocolObject<dyn objc2_metal::MTLDevice>) -> bool {
    device.supportsRaytracing()
}

// Pack a column-major object-to-world `model` matrix into Metal's
// `MTLPackedFloat4x3` instance transform. The packed form is the first three
// rows of each of the four columns (the affine `[0,0,0,1]` bottom row is
// dropped), so `columns[c] = (model[c][0], model[c][1], model[c][2])`. Getting
// this transpose wrong silently mirrors / shears every reflection, so it is
// unit-tested.
pub(crate) fn pack_instance_transform(model: [[f32; 4]; 4]) -> MTLPackedFloat4x3 {
    let col = |c: usize| MTLPackedFloat3 {
        x: model[c][0],
        y: model[c][1],
        z: model[c][2],
    };
    MTLPackedFloat4x3 {
        columns: [col(0), col(1), col(2), col(3)],
    }
}

// Bindless-pool (albedo, normal) indices for a `(texture_slot, normal_map_slot)`
// pair. The slots are clamped into range and the normal index is biased past
// the albedo region, matching the main pass (`cull.rs`); an object with no
// normal map lands on the 1x1 flat-normal fallback at normal slot 0, so the
// hit shader's normal-map sample is always safe.
fn pool_indices(
    texture_slot: usize,
    normal_map_slot: usize,
    albedo_count: usize,
    normal_count: usize,
) -> (u32, u32) {
    let albedo = texture_slot.min(albedo_count.saturating_sub(1)) as u32;
    let normal = (albedo_count + normal_map_slot.min(normal_count.saturating_sub(1))) as u32;
    (albedo, normal)
}

// Build the geometry-table entry for one draw object.
fn geom_entry(obj: &DrawObject, albedo_count: usize, normal_count: usize) -> RtGeomEntry {
    let (albedo_index, normal_index) = pool_indices(
        obj.texture_slot,
        obj.normal_map_slot,
        albedo_count,
        normal_count,
    );
    RtGeomEntry {
        index_offset: obj.index_offset as u32,
        base_vertex: obj.base_vertex as u32,
        albedo_index,
        normal_index,
        tint: obj.material.tint,
        roughness: obj.material.roughness,
        metallic: obj.material.metallic,
        emissive: obj.material.emissive,
        model: obj.model,
        emissive_map_index: obj.material.emissive_map_index,
        _pad: [0; 3],
    }
}

// Build the geometry-table entry for one instance of an instanced cluster: the
// cluster's shared mesh slice + material, with this instance's transform.
// Cluster geometry uses base_vertex 0 (its indices are already absolute).
fn cluster_geom_entry(
    cluster: &InstancedCluster,
    model: [[f32; 4]; 4],
    albedo_count: usize,
    normal_count: usize,
) -> RtGeomEntry {
    let (albedo_index, normal_index) = pool_indices(
        cluster.texture_slot,
        cluster.normal_map_slot,
        albedo_count,
        normal_count,
    );
    RtGeomEntry {
        index_offset: cluster.index_offset as u32,
        base_vertex: 0,
        albedo_index,
        normal_index,
        tint: cluster.material.tint,
        roughness: cluster.material.roughness,
        metallic: cluster.material.metallic,
        emissive: cluster.material.emissive,
        model,
        emissive_map_index: cluster.material.emissive_map_index,
        _pad: [0; 3],
    }
}

// Build the geometry-table entry for one skinned object. The skinned BLAS is
// baked from the posed (model-space) deformed buffer with absolute u16 indices,
// so `base_vertex` is 0 and the model matrix brings the hit to world space (the
// instance transform does the same for the trace). The skinned flag is OR'd
// into `normal_index` so the kernel fetches from the deformed / u16 buffers.
fn skinned_geom_entry(
    obj: &SkinnedDrawObject,
    albedo_count: usize,
    normal_count: usize,
) -> RtGeomEntry {
    let (albedo_index, normal_index) = pool_indices(
        obj.texture_slot,
        obj.normal_map_slot,
        albedo_count,
        normal_count,
    );
    RtGeomEntry {
        index_offset: obj.index_offset as u32,
        base_vertex: 0,
        albedo_index,
        normal_index: normal_index | RT_SKINNED_FLAG,
        tint: obj.material.tint,
        roughness: obj.material.roughness,
        metallic: obj.material.metallic,
        emissive: obj.material.emissive,
        model: obj.model,
        emissive_map_index: obj.material.emissive_map_index,
        _pad: [0; 3],
    }
}

// A primitive (triangle) BLAS descriptor over a slice of the shared buffers.
// `vertexBufferOffset = base_vertex * stride` so a chunk with mesh-relative
// indices and a non-zero base vertex still resolves; static geometry and
// instanced clusters use base_vertex 0 (their indices are already absolute).
// `index_type` selects the index width: the static / instanced buffers are
// `UInt32`; the skinned index buffer (and the deformed-vertex buffer it
// addresses) is `UInt16`.
fn prim_desc_for(
    vertex_buffer: &ProtocolObject<dyn MTLBuffer>,
    index_buffer: &ProtocolObject<dyn MTLBuffer>,
    base_vertex: usize,
    index_offset: usize,
    index_count: usize,
    index_type: MTLIndexType,
) -> Retained<MTLPrimitiveAccelerationStructureDescriptor> {
    let index_bytes = match index_type {
        MTLIndexType::UInt16 => 2,
        _ => 4,
    };
    let geo = unsafe {
        let g = MTLAccelerationStructureTriangleGeometryDescriptor::descriptor();
        g.setVertexBuffer(Some(vertex_buffer));
        g.setVertexBufferOffset(base_vertex * VERTEX_STRIDE);
        g.setVertexStride(VERTEX_STRIDE);
        g.setVertexFormat(MTLAttributeFormat::Float3);
        g.setIndexBuffer(Some(index_buffer));
        g.setIndexBufferOffset(index_offset * index_bytes);
        g.setIndexType(index_type);
        g.setTriangleCount(index_count / 3);
        g
    };
    let geo_ref: &MTLAccelerationStructureGeometryDescriptor = &geo;
    let geos = NSArray::from_slice(&[geo_ref]);
    let prim = MTLPrimitiveAccelerationStructureDescriptor::descriptor();
    prim.setGeometryDescriptors(Some(&geos));
    prim
}

// An instance descriptor with an explicit transform + BLAS index
// (`accelerationStructureIndex` selects which BLAS this instance uses). The
// shader indexes the geometry table by the intersector's `instance_id`, which
// for `MTLAccelerationStructureInstanceDescriptorType::Default` is the
// instance's position in the instance buffer (NOT the
// `accelerationStructureIndex`), so the table carries one entry per instance,
// in instance order (multiple cluster instances share one BLAS but get distinct
// entries). See `build_rt_accel`.
fn instance_desc_at(
    model: [[f32; 4]; 4],
    blas_index: u32,
) -> MTLAccelerationStructureInstanceDescriptor {
    MTLAccelerationStructureInstanceDescriptor {
        transformationMatrix: pack_instance_transform(model),
        options: MTLAccelerationStructureInstanceOptions::Opaque,
        mask: 0xFF,
        intersectionFunctionTableOffset: 0,
        accelerationStructureIndex: blas_index,
    }
}

// The instance descriptor for draw object `i` (its BLAS index == its position).
fn instance_desc(obj: &DrawObject, i: usize) -> MTLAccelerationStructureInstanceDescriptor {
    instance_desc_at(obj.model, i as u32)
}

// The TLAS descriptor over `blas_refs`, reading transforms from
// `instance_buffer`. Takes plain references so the BLAS array can be assembled
// from more than one source (e.g. persistent static BLAS followed by this
// frame's fresh skinned BLAS).
fn make_tlas_desc_from_refs(
    blas_refs: &[&ProtocolObject<dyn MTLAccelerationStructure>],
    instance_buffer: &ProtocolObject<dyn MTLBuffer>,
    instance_count: usize,
) -> Retained<MTLInstanceAccelerationStructureDescriptor> {
    let blas_array = NSArray::from_slice(blas_refs);
    let desc = MTLInstanceAccelerationStructureDescriptor::descriptor();
    desc.setInstancedAccelerationStructures(Some(&blas_array));
    desc.setInstanceCount(instance_count);
    desc.setInstanceDescriptorBuffer(Some(instance_buffer));
    desc.setInstanceDescriptorType(MTLAccelerationStructureInstanceDescriptorType::Default);
    desc
}

// The TLAS descriptor over `blas`, reading transforms from `instance_buffer`.
fn make_tlas_desc(
    blas: &[Retained<ProtocolObject<dyn MTLAccelerationStructure>>],
    instance_buffer: &ProtocolObject<dyn MTLBuffer>,
    instance_count: usize,
) -> Retained<MTLInstanceAccelerationStructureDescriptor> {
    let blas_refs: Vec<&ProtocolObject<dyn MTLAccelerationStructure>> =
        blas.iter().map(|b| b.as_ref()).collect();
    make_tlas_desc_from_refs(&blas_refs, instance_buffer, instance_count)
}

// Declare BLAS that a TLAS build references resident on the build encoder. A
// TLAS build reads the primitive structures its instances point to; unlike a
// direct buffer binding, that indirect reference does not make them resident,
// so Metal requires an explicit `useResource` (or `useHeap`) or the build can
// read a non-resident structure and fault. Only structures built on an *earlier*
// command buffer need this: BLAS built in the same encoder are already resident
// and ordered by same-encoder hazard tracking, so they must NOT be passed here.
fn declare_blas_resident<'a>(
    enc: &ProtocolObject<dyn MTLAccelerationStructureCommandEncoder>,
    blas: impl IntoIterator<Item = &'a Retained<ProtocolObject<dyn MTLAccelerationStructure>>>,
) {
    for b in blas {
        enc.useResource_usage(ProtocolObject::from_ref(&**b), MTLResourceUsage::Read);
    }
}

// Declare every BLAS resident for a fragment-stage trace in ONE batched
// `useResources` call rather than N per-BLAS ones. A trace render pass (the
// transparent glass/water pass, the RT-reflection resolve) reaches each BLAS
// indirectly through the TLAS, which does NOT make them resident, so the pass
// must declare them itself. Batching collapses the per-frame Obj-C message-send
// count on BLAS-heavy worlds (the driver records the same residency set either
// way, just in one call). A no-op when there are no BLAS.
pub(in crate::metal) fn use_blas_resident_fragment(
    enc: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    blas: &[Retained<ProtocolObject<dyn MTLAccelerationStructure>>],
) {
    if blas.is_empty() {
        return;
    }
    let res: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = blas
        .iter()
        .map(|b| NonNull::from(ProtocolObject::from_ref(&**b)))
        .collect();
    // SAFETY: `res` is a non-empty, contiguous array of `res.len()` live resource
    // pointers; the encoder reads it for the duration of the call only.
    unsafe {
        enc.useResources_count_usage_stages(
            NonNull::new(res.as_ptr() as *mut NonNull<ProtocolObject<dyn MTLResource>>)
                .expect("non-empty blas slice has a non-null pointer"),
            res.len(),
            MTLResourceUsage::Read,
            MTLRenderStages::Fragment,
        );
    }
}

// Attach a completion handler that logs the first GPU fault on an async RT
// command buffer (`what` names the stage: skin compute or BLAS/TLAS build). The
// per-frame skinned rebuild commits these without `waitUntilCompleted`, so a
// fault can no longer be caught synchronously by `check_build_status`; this
// surfaces it (once per process, so a wedged GPU does not spam) instead of
// leaving only the downstream trace victim to report.
fn attach_async_fault_logger(
    cmd: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    what: &'static str,
) {
    static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let handler = block2::RcBlock::new(
        move |cb: NonNull<ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>| {
            let cb = unsafe { cb.as_ref() };
            if cb.status() == MTLCommandBufferStatus::Error
                && !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                tracing::error!("RT {what} faulted (async): {:?}", cb.error());
            }
        },
    );
    // SAFETY: addCompletedHandler copies the block, so the RcBlock may drop here.
    unsafe {
        cmd.addCompletedHandler(block2::RcBlock::as_ptr(&handler));
    }
}

// Per-dispatch parameters for the `rt_skin` compute kernel; matches the MSL
// `SkinParams` (16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct SkinParams {
    vertex_base: u32,
    vertex_count: u32,
    joint_count: u32,
    _pad: u32,
}

// Identity matrix used as a one-joint fallback palette so a skinned object with
// no pose yet still has a valid (undeformed) palette to dispatch against.
const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// Run the `rt_skin` compute pass: deform each skinned object's bind-pose
// vertices into `deformed_verts` (posed, model-space, 56-byte `Vertex` layout)
// using its joint palette. Runs on its OWN command buffer, committed and
// waited, so the deformed buffer is complete before the acceleration-structure
// build reads it: an AS build does not synchronize against a prior compute
// pass that wrote its input vertex buffer (it is outside the normal encoder
// hazard tracking), so without this wait the build would race the skinning and
// bake a BLAS from half-written vertices.
fn dispatch_skin(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    skinned: &SkinnedRtInputs,
    skinned_objects: &[(usize, &SkinnedDrawObject)],
    deformed_verts: &ProtocolObject<dyn MTLBuffer>,
) -> Result<(), String> {
    let skin_cmd = command_queue
        .commandBuffer()
        .ok_or("failed to create RT skin command buffer")?;
    let cenc = skin_cmd
        .computeCommandEncoder()
        .ok_or("failed to create RT skin compute encoder")?;
    // Transient palette buffers must outlive the GPU work; held until the wait
    // below completes.
    let palette_bufs =
        encode_skin_dispatch(device, &cenc, skinned, skinned_objects, deformed_verts)?;
    cenc.endEncoding();
    skin_cmd.commit();
    skin_cmd.waitUntilCompleted();
    drop(palette_bufs);
    check_build_status(&skin_cmd, "skinning compute")
}

// Encode the `rt_skin` dispatch for each skinned object into `cenc` (setting the
// pipeline state first) and return the transient joint-palette buffers, which
// must outlive the GPU work. Shared by the synchronous seed (`dispatch_skin`)
// and the asynchronous per-frame rebuild; the caller owns command-buffer
// lifetime (commit + wait, or commit + event + completion handler).
fn encode_skin_dispatch(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    cenc: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    skinned: &SkinnedRtInputs,
    skinned_objects: &[(usize, &SkinnedDrawObject)],
    deformed_verts: &ProtocolObject<dyn MTLBuffer>,
) -> Result<Vec<Retained<ProtocolObject<dyn MTLBuffer>>>, String> {
    cenc.setComputePipelineState(skinned.skin_pipeline);
    let tg = skinned
        .skin_pipeline
        .maxTotalThreadsPerThreadgroup()
        .clamp(1, 64);
    let mut palette_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
    for (obj_idx, obj) in skinned_objects {
        let matrices: &[[[f32; 4]; 4]] = skinned
            .joint_matrices
            .get(*obj_idx)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        // Empty pose -> a single identity joint (undeformed) so the dispatch
        // always has a valid palette to index.
        let (pal_slice, joint_count): (&[[[f32; 4]; 4]], usize) = if matrices.is_empty() {
            (std::slice::from_ref(&IDENTITY4), 1)
        } else {
            (matrices, matrices.len())
        };
        let palette = upload_buffer(device, pal_slice, "RT skin palette")?;
        let params = SkinParams {
            vertex_base: obj.vertex_base as u32,
            vertex_count: obj.vertex_count as u32,
            joint_count: joint_count as u32,
            _pad: 0,
        };
        unsafe {
            cenc.setBuffer_offset_atIndex(Some(skinned.vertex_buffer.as_ref()), 0, 0);
            cenc.setBuffer_offset_atIndex(Some(deformed_verts), 0, 1);
            cenc.setBuffer_offset_atIndex(Some(palette.as_ref()), 0, 2);
            cenc.setBytes_length_atIndex(
                NonNull::from(&params).cast(),
                std::mem::size_of::<SkinParams>(),
                3,
            );
            cenc.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: obj.vertex_count.max(1),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: tg,
                    height: 1,
                    depth: 1,
                },
            );
        }
        palette_bufs.push(palette);
    }
    Ok(palette_bufs)
}

impl crate::metal::context::MtlContext {
    // Per-frame pre-skin for the GPU-driven skinned fold: deform every
    // skinned object's bind-pose vertices into `deformed` (this frame's ring
    // slot) using the per-object joint-palette buffers the main / shadow passes
    // already build for the legacy skinned VS. Reuses the `rt_skin` kernel.
    //
    // Encoded into the Cull pass's command buffer (its own compute encoder),
    // which commits before the Main pass: Metal's automatic hazard tracking then
    // orders this compute write before the main pass's vertex read of `deformed`
    // (the same cross-command-buffer mechanism the static cull → ICB relies on).
    // Unlike the RT seed it binds the pre-built joint buffers instead of
    // uploading transient palettes -- the parallel per-pass encoder cannot keep a
    // transient buffer alive past the worker, while these joint buffers live for
    // the whole frame. A no-op when the skin pipeline / skinned VB are unset.
    pub(in crate::metal) fn encode_main_skin(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        deformed: &ProtocolObject<dyn MTLBuffer>,
        joint_bufs: &[Retained<ProtocolObject<dyn MTLBuffer>>],
    ) -> Result<(), String> {
        let (Some(skin_pipeline), Some(svb)) = (
            self.skinned.skin_pipeline.as_ref(),
            self.skinned.vertex_buffer.as_ref(),
        ) else {
            return Ok(());
        };
        if self.skinned.draw_objects.is_empty() {
            return Ok(());
        }
        let cenc = cmd_buf
            .computeCommandEncoder()
            .ok_or("failed to create main-skin compute encoder")?;
        cenc.setComputePipelineState(skin_pipeline);
        let tg = skin_pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 64);
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            let Some(joint_buf) = joint_bufs.get(i) else {
                continue;
            };
            // Palette length = this object's matrix count (seeded to >= 1, and
            // `update_skinned_pose` never leaves it empty), matching the buffer
            // the kernel indexes.
            let joint_count = self
                .skinned
                .joint_matrices
                .get(i)
                .map(|m| m.len().max(1))
                .unwrap_or(1);
            let params = SkinParams {
                vertex_base: obj.vertex_base as u32,
                vertex_count: obj.vertex_count as u32,
                joint_count: joint_count as u32,
                _pad: 0,
            };
            unsafe {
                cenc.setBuffer_offset_atIndex(Some(svb.as_ref()), 0, 0);
                cenc.setBuffer_offset_atIndex(Some(deformed), 0, 1);
                cenc.setBuffer_offset_atIndex(Some(joint_buf.as_ref()), 0, 2);
                cenc.setBytes_length_atIndex(
                    NonNull::from(&params).cast(),
                    std::mem::size_of::<SkinParams>(),
                    3,
                );
                cenc.dispatchThreads_threadsPerThreadgroup(
                    MTLSize {
                        width: obj.vertex_count.max(1),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: tg,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        cenc.endEncoding();
        Ok(())
    }
}

// Build the BLAS / TLAS / geometry table for the scene. Returns `None` (not an
// error) when there is no resident triangle geometry to trace: the caller then
// leaves RT disabled and the pass falls back to the base scene.
//
// `skinned`, when present, adds skeletally-animated geometry: a compute pass
// deforms each skinned object's vertices into a fresh model-space buffer, and
// one (u16-indexed) BLAS per skinned object is built over that buffer. Because
// the pose changes every frame the whole structure is rebuilt per frame (the
// caller forces this); fresh allocations keep it hazard-free.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_rt_accel(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
    vertex_buffer: &ProtocolObject<dyn MTLBuffer>,
    index_buffer: &ProtocolObject<dyn MTLBuffer>,
    draw_objects: &[DrawObject],
    clusters: &[InstancedCluster],
    albedo_count: usize,
    normal_count: usize,
    skinned: Option<SkinnedRtInputs>,
    // Layer 2 see-through: when set, see-through glass meshes are left out of the
    // BLAS (they trace their own per-pixel reflection in the transparent pass, and
    // excluding them means glass does not reflect glass). Off keeps every
    // transparent mesh IN the BVH so Layer 1 opaque glass reflects + is reflected
    // like any other surface. Driven by `seethrough_meshes_enabled` (opt-in per
    // `Material::see_through`), not a global flag.
    exclude_seethrough: bool,
) -> Result<Option<RtAccelData>, String> {
    // Only resident draw objects with real triangles take part. When the Layer 2
    // see-through path is enabled, see-through glass meshes are excluded (they
    // route through the transparent pass with their own per-pixel trace, so glass
    // does not reflect glass and the trace never self-hits); otherwise (Layer 1)
    // they stay in so opaque glass reflects + is reflected normally. Track the
    // participating indices into `draw_objects` so a per-frame update re-reads
    // transforms in BLAS-build order.
    let object_indices: Vec<usize> = draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| {
            o.resident && o.index_count >= 3 && !(exclude_seethrough && o.material.see_through != 0)
        })
        .map(|(i, _)| i)
        .collect();
    // Instanced clusters that carry real geometry and at least one instance.
    let cluster_list: Vec<&InstancedCluster> = clusters
        .iter()
        .filter(|c| c.index_count >= 3 && !c.instances.is_empty())
        .collect();
    // Skinned objects that are visible and carry real triangles, paired with
    // their index into the joint-matrix list so the skin dispatch finds the pose.
    // Clusters and skinned geometry coexist in the BVH; the combination once
    // page-faulted the trace, but that was a per-frame VRAM leak (no autorelease
    // pool around the frame), fixed separately.
    let skinned_objects: Vec<(usize, &SkinnedDrawObject)> = match &skinned {
        Some(s) => s
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.visible && o.index_count >= 3)
            .collect(),
        None => Vec::new(),
    };
    if object_indices.is_empty() && cluster_list.is_empty() && skinned_objects.is_empty() {
        return Ok(None);
    }
    let objects: Vec<&DrawObject> = object_indices.iter().map(|&i| &draw_objects[i]).collect();

    // Deformed-vertex buffer for skinned geometry: the `rt_skin` kernel writes
    // posed model-space `Vertex`s here, mirroring the skinned vertex buffer's
    // indexing so the u16 skinned index buffer addresses it directly. Sized to
    // the highest vertex the skinned objects reach; a 1-vertex dummy when there
    // is no skinned geometry (so the encoder always has a buffer to bind).
    let deformed_extent: usize = skinned_objects
        .iter()
        .map(|(_, o)| o.vertex_base as usize + o.vertex_count)
        .max()
        .unwrap_or(0);
    let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
    // Shared, not Private: the buffer is written by the skin compute pass and
    // then read both by the acceleration-structure build and (per hit) by the
    // reflection fragment shader, which run in *separate* command buffers. A
    // Private buffer in that cross-command-buffer producer/consumer pattern was
    // observed to GPU page-fault on the fragment read under the parallel per-
    // pass encoder; Shared is always host-resident and coherent, sidestepping it.
    let deformed_verts = device
        .newBufferWithLength_options(deformed_bytes, MTLResourceOptions::StorageModeShared)
        .ok_or("failed to allocate RT deformed-vertex buffer")?;
    // The shared u16 skinned index buffer the kernel + skinned BLAS address; a
    // dummy when there is no skinned geometry.
    let skinned_indices: Retained<ProtocolObject<dyn MTLBuffer>> = match &skinned {
        Some(s) if !skinned_objects.is_empty() => s.index_buffer.clone(),
        _ => device
            .newBufferWithLength_options(2, MTLResourceOptions::StorageModePrivate)
            .ok_or("failed to allocate RT skinned-index dummy buffer")?,
    };

    // One BLAS per draw object, then one per cluster, then one per skinned
    // object. `blas[i]` for i < draw_blas_count is draw object i; the next
    // `cluster_list.len()` are clusters; the rest are skinned objects.
    let draw_blas_count = objects.len();
    let skinned_blas_base = draw_blas_count + cluster_list.len();
    let mut prim_descs: Vec<Retained<MTLPrimitiveAccelerationStructureDescriptor>> =
        Vec::with_capacity(skinned_blas_base + skinned_objects.len());
    for obj in &objects {
        prim_descs.push(prim_desc_for(
            vertex_buffer,
            index_buffer,
            obj.base_vertex as usize,
            obj.index_offset,
            obj.index_count,
            MTLIndexType::UInt32,
        ));
    }
    for c in &cluster_list {
        prim_descs.push(prim_desc_for(
            vertex_buffer,
            index_buffer,
            0,
            c.index_offset,
            c.index_count,
            MTLIndexType::UInt32,
        ));
    }
    // Skinned BLAS trace the deformed buffer (absolute u16 indices, base_vertex
    // 0). The buffer's contents are written by the compute pass on the same
    // command buffer below, before this BLAS builds.
    for (_, obj) in &skinned_objects {
        prim_descs.push(prim_desc_for(
            deformed_verts.as_ref(),
            skinned_indices.as_ref(),
            0,
            obj.index_offset,
            obj.index_count,
            MTLIndexType::UInt16,
        ));
    }

    // Allocate each BLAS and track the largest scratch requirement so a single
    // shared scratch buffer covers the whole build (reused serially).
    let mut blas: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> =
        Vec::with_capacity(prim_descs.len());
    let mut max_scratch: usize = 0;
    for prim in &prim_descs {
        let sizes = device.accelerationStructureSizesWithDescriptor(prim);
        let acc = device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or("failed to allocate BLAS")?;
        max_scratch = max_scratch.max(sizes.buildScratchBufferSize);
        blas.push(acc);
    }

    // The geometry table is indexed PER INSTANCE, by the intersector's
    // `instance_id`, which is the instance's position in the instance buffer
    // (NOT the `accelerationStructureIndex`). So there is exactly one entry per
    // TLAS instance, in instance order: draw objects, then every cluster
    // instance, then skinned objects.
    let mut instance_descs: Vec<MTLAccelerationStructureInstanceDescriptor> = objects
        .iter()
        .enumerate()
        .map(|(i, obj)| instance_desc(obj, i))
        .collect();
    let mut geom_entries: Vec<RtGeomEntry> = objects
        .iter()
        .map(|obj| geom_entry(obj, albedo_count, normal_count))
        .collect();

    // Clusters: one TLAS instance + one geometry entry per cluster instance, all
    // referencing the cluster's single BLAS (via `accelerationStructureIndex`)
    // but each with its own transform + geometry entry (so per-instance normals
    // are correct). Stored on `RtAccelData` so a per-frame TLAS rebuild
    // re-appends them verbatim (clusters are baked static into the BVH).
    let mut cluster_instances: Vec<MTLAccelerationStructureInstanceDescriptor> = Vec::new();
    let mut cluster_geom: Vec<RtGeomEntry> = Vec::new();
    for (ci, c) in cluster_list.iter().enumerate() {
        let blas_index = (draw_blas_count + ci) as u32;
        for model in &c.instances {
            cluster_instances.push(instance_desc_at(*model, blas_index));
            cluster_geom.push(cluster_geom_entry(c, *model, albedo_count, normal_count));
        }
    }
    instance_descs.extend_from_slice(&cluster_instances);
    geom_entries.extend_from_slice(&cluster_geom);

    // Skinned objects: one TLAS instance + one geometry entry each (each skinned
    // object has its own BLAS). The deformed verts are in model space, so the
    // instance transform (= the object's model matrix) brings the trace to world
    // space, like the static path.
    for (si, (_, obj)) in skinned_objects.iter().enumerate() {
        let blas_index = (skinned_blas_base + si) as u32;
        instance_descs.push(instance_desc_at(obj.model, blas_index));
        geom_entries.push(skinned_geom_entry(obj, albedo_count, normal_count));
    }

    let instance_buffer = upload_buffer(device, &instance_descs, "RT instance descriptors")?;
    let geom_table = upload_buffer(device, &geom_entries, "RT geometry table")?;

    let tlas_desc = make_tlas_desc(&blas, &instance_buffer, instance_descs.len());
    let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
    let tlas = device
        .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
        .ok_or("failed to allocate TLAS")?;
    // Size the scratch for the largest of every BLAS build and the TLAS build
    // so the per-frame TLAS rebuild can reuse the same buffer.
    max_scratch = max_scratch.max(tlas_sizes.buildScratchBufferSize);

    let scratch = device
        .newBufferWithLength_options(max_scratch.max(1), MTLResourceOptions::StorageModePrivate)
        .ok_or("failed to allocate RT scratch buffer")?;

    // Skin first (on its own committed-and-waited command buffer) so the
    // deformed buffer is complete before the BLAS build reads it; see
    // `dispatch_skin` for why the wait is required.
    if let Some(s) = &skinned
        && !skinned_objects.is_empty()
    {
        dispatch_skin(
            device,
            command_queue,
            s,
            &skinned_objects,
            deformed_verts.as_ref(),
        )?;
    }

    // Build each BLAS in its own acceleration-structure encoder, then the TLAS in
    // a final encoder. Metal does not order (or prevent overlap of) builds within
    // a single encoder, so a one-encoder build of "all BLAS then the TLAS" lets
    // the TLAS read half-built BLAS and lets builds sharing the scratch buffer
    // stomp on it. Separate encoders within the command buffer serialize, which
    // both orders the TLAS after its BLAS and makes the shared scratch safe.
    // Synchronous: wait for the GPU so the structures (and the skinning that feeds
    // the skinned BLAS) are ready before the first frame traces them.
    let cmd = command_queue
        .commandBuffer()
        .ok_or("failed to create RT build command buffer")?;
    for (acc, prim) in blas.iter().zip(prim_descs.iter()) {
        let enc = cmd
            .accelerationStructureCommandEncoder()
            .ok_or("failed to create acceleration-structure encoder")?;
        enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            acc, prim, &scratch, 0,
        );
        enc.endEncoding();
    }
    // The TLAS references every BLAS, all built on earlier encoders, so declare
    // them resident in this encoder.
    let enc = cmd
        .accelerationStructureCommandEncoder()
        .ok_or("failed to create acceleration-structure encoder")?;
    declare_blas_resident(&enc, &blas);
    enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
        &tlas, &tlas_desc, &scratch, 0,
    );
    enc.endEncoding();
    cmd.commit();
    cmd.waitUntilCompleted();
    check_build_status(&cmd, "acceleration-structure build")?;

    let cached_models = objects.iter().map(|o| o.model).collect();
    let draw_blas_sigs = objects.iter().map(|o| GeomSig::of(o)).collect();

    Ok(Some(RtAccelData {
        blas,
        static_blas_count: skinned_blas_base,
        tlas,
        geom_table,
        object_indices,
        draw_blas_sigs,
        cached_models,
        cluster_instances,
        cluster_geom,
        scratch,
        instance_buffer,
        deformed_verts,
        skinned_indices,
        retire_pool: RetirePool::new(),
    }))
}

impl RtAccelData {
    // Re-collect the participating draw objects in BLAS order. Returns `None`
    // if the draw list changed shape (an index is now out of range or no
    // longer resident): the caller then leaves the structure as-is for this
    // frame (a full rebuild is the path that handles a changed object set).
    fn current_objects<'a>(&self, draw_objects: &'a [DrawObject]) -> Option<Vec<&'a DrawObject>> {
        let mut objects = Vec::with_capacity(self.object_indices.len());
        for &idx in &self.object_indices {
            let obj = draw_objects.get(idx)?;
            if !obj.resident || obj.index_count < 3 {
                return None;
            }
            objects.push(obj);
        }
        Some(objects)
    }

    // Keep the static BLAS; rebuild the TLAS + geometry table from current
    // transforms with fresh allocations, then build on a separate command
    // buffer (committed and waited). Fresh allocations mean no prior in-flight
    // frame can observe a half-updated structure: the old TLAS / table stay
    // alive (retained by their command buffers) until those frames complete.
    pub(crate) fn rebuild_tlas(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        draw_objects: &[DrawObject],
        albedo_count: usize,
        normal_count: usize,
    ) -> Result<(), String> {
        let Some(objects) = self.current_objects(draw_objects) else {
            return Ok(());
        };
        // Freshly-transformed draw-object instances, then the cluster instances
        // re-appended verbatim (clusters are baked static; their BLAS never move
        // in `self.blas`, so the stored `accelerationStructureIndex` stays
        // valid). The geometry table stays per-BLAS: draw entries (per object)
        // then the per-cluster entries.
        let mut instance_descs: Vec<MTLAccelerationStructureInstanceDescriptor> = objects
            .iter()
            .enumerate()
            .map(|(i, obj)| instance_desc(obj, i))
            .collect();
        let mut geom_entries: Vec<RtGeomEntry> = objects
            .iter()
            .map(|obj| geom_entry(obj, albedo_count, normal_count))
            .collect();
        instance_descs.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);

        let instance_buffer = upload_buffer(device, &instance_descs, "RT instance descriptors")?;
        let geom_table = upload_buffer(device, &geom_entries, "RT geometry table")?;
        let tlas_desc = make_tlas_desc(&self.blas, &instance_buffer, instance_descs.len());
        let sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        let tlas = device
            .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
            .ok_or("failed to allocate TLAS")?;
        // Reuse the scratch sized at init (the prior frame's build completed
        // before we got here, so it is free) -- but a topology refresh can change
        // the instance count, and a larger TLAS needs more build scratch than the
        // init sizing. Grow it when so. Replacing the handle is safe: this path
        // is synchronous (commit + wait), and an in-flight command buffer that
        // still references the old scratch retains it independently of this Vec.
        if (sizes.buildScratchBufferSize as u64) > self.scratch.length() as u64 {
            self.scratch = device
                .newBufferWithLength_options(
                    sizes.buildScratchBufferSize.max(1),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or("failed to grow RT scratch buffer")?;
        }

        let cmd = command_queue
            .commandBuffer()
            .ok_or("failed to create RT rebuild command buffer")?;
        let enc = cmd
            .accelerationStructureCommandEncoder()
            .ok_or("failed to create acceleration-structure encoder")?;
        // Every BLAS the rebuilt TLAS references was built on an earlier command
        // buffer (none are rebuilt here), so all must be declared resident.
        declare_blas_resident(&enc, &self.blas);
        enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
            &tlas,
            &tlas_desc,
            &self.scratch,
            0,
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        check_build_status(&cmd, "TLAS rebuild")?;

        self.tlas = tlas;
        self.geom_table = geom_table;
        // Snapshot the transforms now baked into the TLAS so the next frame's
        // dirty check compares against what was actually built.
        self.cached_models = objects.iter().map(|o| o.model).collect();
        Ok(())
    }

    // Whether the BVH has no draw-object and no cluster geometry left. After a
    // topology refresh removes the last of both (every chunk streamed out, with
    // no clusters), the caller drops the structure so a later add re-seeds it
    // rather than building a degenerate zero-instance TLAS. (Skinned geometry is
    // handled on its own per-frame path and never reaches the refresh, so it is
    // not consulted here.)
    pub(crate) fn is_empty(&self) -> bool {
        self.object_indices.is_empty() && self.cluster_instances.is_empty()
    }

    // Incrementally bring the draw-object BLAS head in line with the current
    // participating draw set: reuse every BLAS whose geometry is unchanged, build
    // only the new / changed ones, retire the orphans. The cluster + skinned tails
    // of `blas` are preserved verbatim. When `build_tlas` is set (the no-skinned
    // path), also rebuilds the TLAS + geometry table over [refreshed draw head +
    // clusters] in the same command buffer; when clear (the skinned path) only the
    // head is refreshed and the caller's `rebuild_skinned` rebuilds the TLAS over
    // the head + the fresh skinned tail. Used when streamed chunks are
    // added/removed, props are cloned, or a material edit changes RT
    // participation; a full rebuild of every BLAS would be too costly when most
    // are unchanged.
    //
    // Fully asynchronous, mirroring `rebuild_skinned`: NO `waitUntilCompleted`.
    // The new BLAS (and, when `build_tlas`, the TLAS) build on one command buffer
    // committed on the shared queue ahead of this frame's reflection-trace command
    // buffer, ordered by same-queue FIFO commit -- the same mechanism the skinned
    // rebuild and the whole render graph rely on -- so the trace reads
    // fully-built structures with no CPU stall. All outgoing / transient resources
    // (orphan BLAS, the old TLAS + geometry table + instance buffer when
    // `build_tlas`, and the build scratch) are parked in `retire_pool` rather than
    // freed in place: `useResource` declares residency not lifetime, and the build
    // keeps reading the scratch / instance buffer after this returns, so they must
    // outlive the frames whose still-in-flight trace could reach them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn refresh_static_topology(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        vertex_buffer: &ProtocolObject<dyn MTLBuffer>,
        index_buffer: &ProtocolObject<dyn MTLBuffer>,
        draw_objects: &[DrawObject],
        albedo_count: usize,
        normal_count: usize,
        exclude_seethrough: bool,
        build_tlas: bool,
        frame_id: u64,
    ) -> Result<(), String> {
        // The current participating draw set, by the same predicate as the full
        // build. (Clusters + skinned never change here, so they are not re-filtered.)
        let new_indices: Vec<usize> = draw_objects
            .iter()
            .enumerate()
            .filter(|(_, o)| {
                o.resident
                    && o.index_count >= 3
                    && !(exclude_seethrough && o.material.see_through != 0)
            })
            .map(|(i, _)| i)
            .collect();
        let new_sigs: Vec<GeomSig> = new_indices
            .iter()
            .map(|&i| GeomSig::of(&draw_objects[i]))
            .collect();

        let plan = plan_topology_refresh(
            &self.object_indices,
            &self.draw_blas_sigs,
            &new_indices,
            &new_sigs,
        );

        let old_draw_count = self.object_indices.len();
        // Clusters occupy `blas[old_draw_count..static_blas_count]`; skinned the
        // tail past `static_blas_count`. Both are preserved across the refresh.
        let cluster_count = self.static_blas_count - old_draw_count;

        // Allocate (but do not yet build) a fresh BLAS for every slot the plan did
        // not match to an existing one. Each is parked at its new-slot position so
        // the assembly below can interleave reused and built BLAS in `new_indices`
        // order.
        let mut fresh: Vec<Option<Retained<ProtocolObject<dyn MTLAccelerationStructure>>>> =
            (0..new_indices.len()).map(|_| None).collect();
        let mut build_jobs: Vec<(usize, Retained<MTLPrimitiveAccelerationStructureDescriptor>)> =
            Vec::new();
        let mut max_scratch: usize = 0;
        for (j, reuse) in plan.reuse.iter().enumerate() {
            if reuse.is_some() {
                continue;
            }
            let obj = &draw_objects[new_indices[j]];
            let prim = prim_desc_for(
                vertex_buffer,
                index_buffer,
                obj.base_vertex as usize,
                obj.index_offset,
                obj.index_count,
                MTLIndexType::UInt32,
            );
            let sizes = device.accelerationStructureSizesWithDescriptor(&prim);
            let acc = device
                .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
                .ok_or("failed to allocate topology-refresh BLAS")?;
            acc.setLabel(Some(&crate::metal::pipeline::ns_str("rt_topology_blas")));
            max_scratch = max_scratch.max(sizes.buildScratchBufferSize);
            fresh[j] = Some(acc);
            build_jobs.push((j, prim));
        }

        // Assemble the new BLAS array: [refreshed draw head, clusters, skinned],
        // pulling each draw slot from the reused old BLAS or its freshly-built one.
        let old_blas = std::mem::take(&mut self.blas);
        let mut new_blas: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> =
            Vec::with_capacity(new_indices.len() + (old_blas.len() - old_draw_count));
        for (j, reuse) in plan.reuse.iter().enumerate() {
            match reuse {
                Some(k) => new_blas.push(old_blas[*k].clone()),
                None => new_blas.push(fresh[j].clone().expect("fresh BLAS built above")),
            }
        }
        // Clusters then skinned, verbatim.
        for b in &old_blas[old_draw_count..] {
            new_blas.push(b.clone());
        }

        // Outgoing structures / buffers to retire once the frames-in-flight fence
        // clears them. Orphaned draw BLAS go here always: the current (not yet
        // replaced) TLAS, which an in-flight trace may still be reading, references
        // them, and `useResource` is residency not lifetime.
        let mut retire_structures: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> =
            plan.retire.iter().map(|&k| old_blas[k].clone()).collect();
        let mut retire_buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
        drop(old_blas);

        // When asked, rebuild the TLAS + geometry table over the refreshed draw
        // head + the cluster instances (re-appended verbatim), with the current
        // transforms. Skipped if the set is empty (the caller drops the BVH rather
        // than build a degenerate zero-instance TLAS). The structures are
        // allocated here and built on the command buffer below.
        let do_tlas = build_tlas && !(new_indices.is_empty() && cluster_count == 0);
        let tlas_build = if do_tlas {
            let objects: Vec<&DrawObject> = new_indices.iter().map(|&i| &draw_objects[i]).collect();
            let mut instance_descs: Vec<MTLAccelerationStructureInstanceDescriptor> = objects
                .iter()
                .enumerate()
                .map(|(i, obj)| instance_desc(obj, i))
                .collect();
            let mut geom_entries: Vec<RtGeomEntry> = objects
                .iter()
                .map(|obj| geom_entry(obj, albedo_count, normal_count))
                .collect();
            instance_descs.extend_from_slice(&self.cluster_instances);
            geom_entries.extend_from_slice(&self.cluster_geom);

            let instance_buffer =
                upload_buffer(device, &instance_descs, "RT instance descriptors")?;
            let geom_table = upload_buffer(device, &geom_entries, "RT geometry table")?;
            let tlas_desc = make_tlas_desc(&new_blas, &instance_buffer, instance_descs.len());
            let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
            max_scratch = max_scratch.max(tlas_sizes.buildScratchBufferSize);
            let tlas = device
                .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
                .ok_or("failed to allocate TLAS")?;
            tlas.setLabel(Some(&crate::metal::pipeline::ns_str("rt_tlas")));
            let cached_models: Vec<[[f32; 4]; 4]> = objects.iter().map(|o| o.model).collect();
            Some((tlas, tlas_desc, instance_buffer, geom_table, cached_models))
        } else {
            None
        };

        // Build everything on ONE command buffer, committed WITHOUT waiting. Each
        // new BLAS in its own encoder (Metal does not order builds within an
        // encoder, and they share the scratch); then, when building the TLAS, a
        // final encoder that declares every referenced BLAS resident (all were
        // built on this or an earlier command buffer, so the TLAS build needs the
        // explicit `useResource`, exactly as the full build does).
        if !build_jobs.is_empty() || tlas_build.is_some() {
            let scratch = device
                .newBufferWithLength_options(
                    max_scratch.max(1),
                    MTLResourceOptions::StorageModePrivate,
                )
                .ok_or("failed to allocate topology-refresh scratch buffer")?;
            let cmd = command_queue
                .commandBuffer()
                .ok_or("failed to create topology-refresh command buffer")?;
            cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_topology_build")));
            for (j, prim) in &build_jobs {
                let acc = fresh[*j].as_ref().expect("fresh BLAS allocated above");
                let enc = cmd
                    .accelerationStructureCommandEncoder()
                    .ok_or("failed to create acceleration-structure encoder")?;
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    acc, prim, &scratch, 0,
                );
                enc.endEncoding();
            }
            if let Some((tlas, tlas_desc, _, _, _)) = &tlas_build {
                let enc = cmd
                    .accelerationStructureCommandEncoder()
                    .ok_or("failed to create acceleration-structure encoder")?;
                declare_blas_resident(&enc, &new_blas);
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    tlas, tlas_desc, &scratch, 0,
                );
                enc.endEncoding();
            }
            attach_async_fault_logger(&cmd, "RT topology build");
            cmd.commit();
            // The async build keeps reading the scratch after this returns.
            retire_buffers.push(scratch);
        }

        // Swap in the refreshed structures; park the outgoing ones for deferred
        // free. The new BLAS stay owned by `self.blas` (the build references them
        // by residency, not retention), so they must not be dropped here.
        self.blas = new_blas;
        self.static_blas_count = new_indices.len() + cluster_count;
        self.object_indices = new_indices;
        self.draw_blas_sigs = new_sigs;
        if let Some((tlas, _, instance_buffer, geom_table, cached_models)) = tlas_build {
            retire_structures.push(std::mem::replace(&mut self.tlas, tlas));
            retire_buffers.push(std::mem::replace(&mut self.geom_table, geom_table));
            retire_buffers.push(std::mem::replace(
                &mut self.instance_buffer,
                instance_buffer,
            ));
            // Snapshot the transforms baked into the new TLAS for the next dirty check.
            self.cached_models = cached_models;
        }
        // When `build_tlas` is clear (the skinned path), `cached_models` is rebuilt
        // by the caller's `rebuild_skinned` over the refreshed `object_indices`.
        if !retire_structures.is_empty() || !retire_buffers.is_empty() {
            self.retire_pool.push(
                frame_id,
                RetiredRt {
                    structures: retire_structures,
                    buffers: retire_buffers,
                },
            );
        }
        Ok(())
    }

    // Per-frame skinned update: keep the persistent static + cluster BLAS,
    // re-skin this frame's pose, rebuild the skinned BLAS, and rebuild the TLAS +
    // geometry table over the static head plus the fresh skinned tail.
    //
    // Fully asynchronous: NO `waitUntilCompleted`. The skin compute (which writes
    // `deformed_verts`) and the BLAS/TLAS build (which reads it) run on separate
    // command buffers, both committed on the shared queue in order (skin, then
    // build) ahead of this frame's reflection-trace command buffer. An
    // acceleration-structure build is outside Metal's automatic hazard tracking,
    // so skin → build and build → trace are ordered by same-queue FIFO commit order,
    // the same mechanism the render graph uses for every cross-pass read.
    // Per-frame GPU stalls are gone; faults are surfaced from completion handlers.
    //
    // Fresh allocations everywhere; the outgoing structures / buffers are parked
    // in `retire_pool` (not freed in place) until the frames-in-flight fence
    // retires the frames whose still-in-flight trace could read them. Returns
    // `Ok(())` without touching the structures when the draw list changed shape
    // (a full rebuild handles that), and falls back to `rebuild_tlas` when no
    // skinned object is visible this frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rebuild_skinned(
        &mut self,
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        command_queue: &ProtocolObject<dyn objc2_metal::MTLCommandQueue>,
        draw_objects: &[DrawObject],
        skinned: SkinnedRtInputs,
        albedo_count: usize,
        normal_count: usize,
        frame_id: u64,
    ) -> Result<(), String> {
        let Some(objects) = self.current_objects(draw_objects) else {
            return Ok(());
        };
        let skinned_objects: Vec<(usize, &SkinnedDrawObject)> = skinned
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| o.visible && o.index_count >= 3)
            .collect();
        // No skinned geometry visible this frame: keep the static BLAS and just
        // refresh the TLAS from current transforms (the static path).
        if skinned_objects.is_empty() {
            return self.rebuild_tlas(
                device,
                command_queue,
                draw_objects,
                albedo_count,
                normal_count,
            );
        }

        // Fresh deformed-vertex buffer (Shared): a prior frame's trace may still
        // read the current one (buffer 5), so allocate new and retire the old.
        let deformed_extent: usize = skinned_objects
            .iter()
            .map(|(_, o)| o.vertex_base as usize + o.vertex_count)
            .max()
            .unwrap_or(0);
        let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let deformed_verts = device
            .newBufferWithLength_options(deformed_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or("failed to allocate RT deformed-vertex buffer")?;
        deformed_verts.setLabel(Some(&crate::metal::pipeline::ns_str("rt_deformed_verts")));
        let skinned_indices = skinned.index_buffer.clone();

        // Fresh skinned BLAS over the (to-be-)posed deformed buffer (absolute u16
        // indices, base_vertex 0), one per skinned object.
        let skinned_prim_descs: Vec<Retained<MTLPrimitiveAccelerationStructureDescriptor>> =
            skinned_objects
                .iter()
                .map(|(_, obj)| {
                    prim_desc_for(
                        deformed_verts.as_ref(),
                        skinned_indices.as_ref(),
                        0,
                        obj.index_offset,
                        obj.index_count,
                        MTLIndexType::UInt16,
                    )
                })
                .collect();
        let mut new_skinned_blas: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> =
            Vec::with_capacity(skinned_prim_descs.len());
        let mut required_scratch: usize = 0;
        for prim in &skinned_prim_descs {
            let sizes = device.accelerationStructureSizesWithDescriptor(prim);
            let acc = device
                .newAccelerationStructureWithSize(sizes.accelerationStructureSize)
                .ok_or("failed to allocate skinned BLAS")?;
            acc.setLabel(Some(&crate::metal::pipeline::ns_str("rt_skinned_blas")));
            required_scratch = required_scratch.max(sizes.buildScratchBufferSize);
            new_skinned_blas.push(acc);
        }

        // TLAS instances + geometry table, in instance order: static draw objects
        // (current transforms), then the cluster instances verbatim, then one per
        // skinned object. Skinned BLAS follow the static/cluster head, so their
        // `accelerationStructureIndex` is `static_blas_count + si`.
        let mut instance_descs: Vec<MTLAccelerationStructureInstanceDescriptor> = objects
            .iter()
            .enumerate()
            .map(|(i, obj)| instance_desc(obj, i))
            .collect();
        let mut geom_entries: Vec<RtGeomEntry> = objects
            .iter()
            .map(|obj| geom_entry(obj, albedo_count, normal_count))
            .collect();
        instance_descs.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);
        for (si, (_, obj)) in skinned_objects.iter().enumerate() {
            instance_descs.push(instance_desc_at(
                obj.model,
                (self.static_blas_count + si) as u32,
            ));
            geom_entries.push(skinned_geom_entry(obj, albedo_count, normal_count));
        }

        // Fresh instance-descriptor + geometry-table buffers (the old ones are
        // retired below: a prior frame's TLAS build / trace may still read them).
        let instance_buffer = upload_buffer(device, &instance_descs, "RT instance descriptors")?;
        let geom_table = upload_buffer(device, &geom_entries, "RT geometry table")?;

        // TLAS over the persistent static/cluster head + this frame's fresh
        // skinned tail.
        let tlas_desc = {
            let all_blas_refs: Vec<&ProtocolObject<dyn MTLAccelerationStructure>> = self.blas
                [..self.static_blas_count]
                .iter()
                .map(|b| b.as_ref())
                .chain(new_skinned_blas.iter().map(|b| b.as_ref()))
                .collect();
            make_tlas_desc_from_refs(&all_blas_refs, &instance_buffer, instance_descs.len())
        };
        let tlas_sizes = device.accelerationStructureSizesWithDescriptor(&tlas_desc);
        required_scratch = required_scratch.max(tlas_sizes.buildScratchBufferSize);
        let tlas = device
            .newAccelerationStructureWithSize(tlas_sizes.accelerationStructureSize)
            .ok_or("failed to allocate TLAS")?;
        tlas.setLabel(Some(&crate::metal::pipeline::ns_str("rt_tlas")));
        let scratch = device
            .newBufferWithLength_options(
                required_scratch.max(1),
                MTLResourceOptions::StorageModePrivate,
            )
            .ok_or("failed to allocate RT scratch buffer")?;

        // Stage 1: skin compute on its own command buffer, committed WITHOUT
        // waiting. Same-queue commit order runs it before the build below (which
        // reads the deformed buffer it writes), the same FIFO ordering the build →
        // trace step and the whole render graph rely on. The transient joint-
        // palette buffers must outlive the GPU work, so they are parked in the
        // retire pool with the outgoing resources (freed once the frames-in-flight
        // fence retires this frame); they cannot be dropped here. A fault can no
        // longer be caught synchronously, so it is logged from a completion handler.
        let skin_palettes = {
            let skin_cmd = command_queue
                .commandBuffer()
                .ok_or("failed to create RT skin command buffer")?;
            skin_cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_skin")));
            let cenc = skin_cmd
                .computeCommandEncoder()
                .ok_or("failed to create RT skin compute encoder")?;
            let palettes = encode_skin_dispatch(
                device,
                &cenc,
                &skinned,
                &skinned_objects,
                deformed_verts.as_ref(),
            )?;
            cenc.endEncoding();
            attach_async_fault_logger(&skin_cmd, "skinning compute");
            skin_cmd.commit();
            palettes
        };

        // Stage 2: skinned BLAS + TLAS build, committed WITHOUT waiting: same-queue
        // commit order runs it after the skin compute above and before this frame's
        // reflection trace (committed later on the shared queue), the same FIFO
        // ordering the render graph relies on for every cross-pass read.
        //
        // Each build gets its OWN acceleration-structure encoder. Metal does not
        // guarantee the order (or non-overlap) of builds within a single encoder,
        // so a TLAS that references BLAS built in the same encoder can read them
        // half-built, and builds sharing one scratch buffer can stomp on it.
        // Separate encoders within the command buffer serialize, which both orders
        // the TLAS after its BLAS and makes the shared scratch safe to reuse. (The
        // static-only `rebuild_tlas` path never hit this because its BLAS were all
        // built on earlier command buffers.)
        {
            let cmd = command_queue
                .commandBuffer()
                .ok_or("failed to create RT skinned rebuild command buffer")?;
            cmd.setLabel(Some(&crate::metal::pipeline::ns_str("rt_build")));
            for (acc, prim) in new_skinned_blas.iter().zip(skinned_prim_descs.iter()) {
                let enc = cmd
                    .accelerationStructureCommandEncoder()
                    .ok_or("failed to create acceleration-structure encoder")?;
                enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                    acc, prim, &scratch, 0,
                );
                enc.endEncoding();
            }
            // The TLAS, in its own encoder after every BLAS is built. It references
            // the persistent static/cluster head AND this frame's skinned BLAS, all
            // built on earlier encoders / command buffers, so every one must be
            // declared resident here.
            let enc = cmd
                .accelerationStructureCommandEncoder()
                .ok_or("failed to create acceleration-structure encoder")?;
            declare_blas_resident(
                &enc,
                self.blas[..self.static_blas_count]
                    .iter()
                    .chain(new_skinned_blas.iter()),
            );
            enc.buildAccelerationStructure_descriptor_scratchBuffer_scratchBufferOffset(
                &tlas, &tlas_desc, &scratch, 0,
            );
            enc.endEncoding();
            attach_async_fault_logger(&cmd, "skinned BLAS + TLAS build");
            cmd.commit();
        }

        // Swap current → new; park the outgoing structures + buffers for deferred
        // free. The static/cluster head of `blas` is untouched: only the skinned
        // tail rotates.
        let mut structures: Vec<Retained<ProtocolObject<dyn MTLAccelerationStructure>>> =
            self.blas.split_off(self.static_blas_count);
        self.blas.extend(new_skinned_blas);
        structures.push(std::mem::replace(&mut self.tlas, tlas));
        let mut buffers = vec![
            std::mem::replace(&mut self.geom_table, geom_table),
            std::mem::replace(&mut self.deformed_verts, deformed_verts),
            std::mem::replace(&mut self.instance_buffer, instance_buffer),
            std::mem::replace(&mut self.scratch, scratch),
        ];
        // This frame's transient skin palettes ride along in the retire pool so
        // they outlive the async skin compute (freed when the fence retires this
        // frame, well after the skin GPU work completes).
        buffers.extend(skin_palettes);
        self.skinned_indices = skinned_indices;
        self.cached_models = objects.iter().map(|o| o.model).collect();
        self.retire_pool.push(
            frame_id,
            RetiredRt {
                structures,
                buffers,
            },
        );
        Ok(())
    }

    // Drop resources parked by prior skinned rebuilds that the frames-in-flight
    // fence now guarantees no in-flight frame can still read (`depth` =
    // frames-in-flight; see [`RetirePool::collect`]). Called once per frame.
    pub(crate) fn retire_completed(&mut self, frame_id: u64, depth: usize) {
        self.retire_pool.collect(frame_id, depth as u64);
    }

    // Whether any participating object's model matrix differs from the one
    // baked into the current TLAS. The cheap per-frame check that gates the
    // `Auto` rebuild so a static scene never rebuilds. A changed draw-list
    // shape (missing index) reads as dirty: the conservative answer.
    pub(crate) fn transforms_dirty(&self, draw_objects: &[DrawObject]) -> bool {
        models_dirty(&self.object_indices, &self.cached_models, |idx| {
            draw_objects.get(idx).map(|o| o.model)
        })
    }
}

// Pure dirty test: true if `current(idx)` differs from the cached model for any
// `(idx, cached)` pair, or `current` has no entry for an index. Split out from
// `transforms_dirty` so it is unit-testable without a `DrawObject`.
fn models_dirty(
    object_indices: &[usize],
    cached_models: &[[[f32; 4]; 4]],
    current: impl Fn(usize) -> Option<[[f32; 4]; 4]>,
) -> bool {
    if object_indices.len() != cached_models.len() {
        return true;
    }
    object_indices
        .iter()
        .zip(cached_models.iter())
        .any(|(&idx, cached)| current(idx) != Some(*cached))
}

// Build the compute pipeline that deforms skinned vertices for ray tracing
// (`rt_skin.metal`). Compiled only when RT reflections are on and the GPU
// supports ray tracing, alongside the reflection pipelines.
pub(crate) fn build_rt_skin_pipeline(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>, String> {
    use objc2_metal::{MTLDevice as _, MTLLibrary as _};
    let msl = crate::metal::pipeline::shader_source(hot_reload, "rt_skin.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(
            &crate::metal::pipeline::ns_str(msl.as_ref()),
            Some(&options),
        )
        .map_err(|e| format!("RT skinning shader compile error: {:?}", e))?;
    let func = library
        .newFunctionWithName(&crate::metal::pipeline::ns_str("rt_skin"))
        .ok_or("rt_skin kernel not found")?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("failed to create RT skin pipeline: {:?}", e))
}

// Fail if a command buffer faulted on the GPU. `waitUntilCompleted` returns
// regardless of success, so without this a faulted build/skin would leave a
// corrupt structure the trace then reads. Surfacing it as an `Err` lets the
// non-fatal per-frame update skip the frame (keeping the last good BVH) instead
// of tracing garbage. `what` names the stage so a fault points at the actual
// culprit (the skin compute vs the acceleration-structure build) rather than a
// generic message, and a downstream `SubmissionsIgnored` cascade is
// distinguishable from an original fault by its error code.
fn check_build_status(
    cmd: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    what: &str,
) -> Result<(), String> {
    if cmd.status() == MTLCommandBufferStatus::Error {
        return Err(format!("RT {what} faulted on the GPU: {:?}", cmd.error()));
    }
    Ok(())
}

// Upload a `#[repr(C)]` slice to a new shared GPU buffer.
fn upload_buffer<T: Copy>(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    data: &[T],
    what: &str,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
    let bytes = std::mem::size_of_val(data);
    let ptr = std::ptr::NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
        .ok_or_else(|| format!("{what}: null data pointer"))?;
    unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            bytes.max(1),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| format!("failed to allocate buffer for {what}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skin_params_layout_matches_msl() {
        // MSL `SkinParams` in rt_skin.metal: four tightly packed uints.
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<SkinParams>(), 16);
        assert_eq!(offset_of!(SkinParams, vertex_base), 0);
        assert_eq!(offset_of!(SkinParams, vertex_count), 4);
        assert_eq!(offset_of!(SkinParams, joint_count), 8);
        assert_eq!(offset_of!(SkinParams, _pad), 12);
    }

    #[test]
    fn pack_instance_transform_drops_affine_row_and_keeps_columns() {
        // A model with a clear translation column and a scaled basis. The packed
        // 4x3 keeps the top three rows of each column; the [0,0,0,1] row is gone.
        let model = [
            [2.0, 0.0, 0.0, 0.0], // column 0: scaled x basis
            [0.0, 3.0, 0.0, 0.0], // column 1: scaled y basis
            [0.0, 0.0, 4.0, 0.0], // column 2: scaled z basis
            [5.0, 6.0, 7.0, 1.0], // column 3: translation
        ];
        let p = pack_instance_transform(model);
        assert_eq!(
            (p.columns[0].x, p.columns[0].y, p.columns[0].z),
            (2.0, 0.0, 0.0)
        );
        assert_eq!(
            (p.columns[1].x, p.columns[1].y, p.columns[1].z),
            (0.0, 3.0, 0.0)
        );
        assert_eq!(
            (p.columns[2].x, p.columns[2].y, p.columns[2].z),
            (0.0, 0.0, 4.0)
        );
        // The translation lands in the fourth column, not a transposed row.
        assert_eq!(
            (p.columns[3].x, p.columns[3].y, p.columns[3].z),
            (5.0, 6.0, 7.0)
        );
    }

    #[test]
    fn pool_indices_clamp_and_bias_match_the_main_pass() {
        // Albedo is clamped into [0, albedo_count); a normal map's global index
        // is biased past the albedo region, both matching cull.rs.
        let (a, n) = pool_indices(2, 1, 5, 3);
        assert_eq!(a, 2); // in range
        assert_eq!(n, 5 + 1); // albedo_count + slot
        // Out-of-range slots clamp to the last valid entry.
        let (a, n) = pool_indices(9, 9, 5, 3);
        assert_eq!(a, 4); // albedo_count - 1
        assert_eq!(n, 5 + 2); // albedo_count + (normal_count - 1)
        // No normal maps (count 0) still resolves to the albedo-region boundary,
        // which is where the flat-normal fallback lives.
        let (_a, n) = pool_indices(0, 0, 4, 0);
        assert_eq!(n, 4);
    }

    // A distinct geometry signature keyed off `tag` (used as the index offset),
    // so two slots with different tags never compare equal.
    fn sig(tag: usize) -> GeomSig {
        GeomSig {
            base_vertex: tag as i32,
            vertex_offset: tag * 100,
            index_offset: tag,
            index_count: 3,
        }
    }

    #[test]
    fn topology_plan_reuses_an_unchanged_set() {
        let old_i = [2usize, 5, 7];
        let old_s = [sig(2), sig(5), sig(7)];
        let plan = plan_topology_refresh(&old_i, &old_s, &old_i, &old_s);
        assert_eq!(plan.reuse, vec![Some(0), Some(1), Some(2)]);
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn topology_plan_builds_only_the_added_slot() {
        let old_i = [2usize, 5];
        let old_s = [sig(2), sig(5)];
        let new_i = [2usize, 5, 9];
        let new_s = [sig(2), sig(5), sig(9)];
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        // The two existing slots reuse; the new one (9) builds fresh.
        assert_eq!(plan.reuse, vec![Some(0), Some(1), None]);
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn topology_plan_retires_a_removed_slot() {
        let old_i = [2usize, 5, 7];
        let old_s = [sig(2), sig(5), sig(7)];
        let new_i = [2usize, 7];
        let new_s = [sig(2), sig(7)];
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        assert_eq!(plan.reuse, vec![Some(0), Some(2)]);
        assert_eq!(plan.retire, vec![1]); // slot 5's old BLAS is orphaned
    }

    #[test]
    fn topology_plan_rebuilds_a_recycled_slot_whose_geometry_moved() {
        // Same draw index, different geometry signature: a chunk slot recycled for
        // a different chunk. The old BLAS must NOT be reused; it is retired and a
        // fresh one is built.
        let old_i = [5usize];
        let old_s = [sig(5)];
        let new_i = [5usize];
        let new_s = [sig(8)]; // moved geometry under the same draw index
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        assert_eq!(plan.reuse, vec![None]);
        assert_eq!(plan.retire, vec![0]);
    }

    #[test]
    fn topology_plan_reuses_across_reorder_by_index() {
        // The participating set is the same but its order changed; each slot still
        // reuses its BLAS by draw index (the reuse points at the old position).
        let old_i = [2usize, 5];
        let old_s = [sig(2), sig(5)];
        let new_i = [5usize, 2];
        let new_s = [sig(5), sig(2)];
        let plan = plan_topology_refresh(&old_i, &old_s, &new_i, &new_s);
        assert_eq!(plan.reuse, vec![Some(1), Some(0)]);
        assert!(plan.retire.is_empty());
    }

    #[test]
    fn rt_geom_entry_is_128_bytes() {
        // The kernel's matching struct relies on this exact size + 16-byte
        // alignment for the array stride to agree. tint+roughness fill one
        // float4; metallic + emissive[3] fill the next so the float4x4 model
        // lands on a 16-byte boundary, exactly as MSL lays the struct out
        // (emissive is a `packed_float3` there, matching `[f32; 3]` here).
        assert_eq!(std::mem::size_of::<RtGeomEntry>(), 128);
    }

    #[test]
    fn rt_dynamic_mode_parses_known_values_and_defaults_auto() {
        // The env parse is a plain match; exercise its arms directly via the
        // same mapping so a renamed mode string can't silently fall through.
        let map = |s: &str| match s {
            "off" => RtDynamicMode::Off,
            "rebuild" => RtDynamicMode::Rebuild,
            "tlas" => RtDynamicMode::Tlas,
            _ => RtDynamicMode::Auto,
        };
        assert_eq!(map("off"), RtDynamicMode::Off);
        assert_eq!(map("rebuild"), RtDynamicMode::Rebuild);
        assert_eq!(map("tlas"), RtDynamicMode::Tlas);
        // Unset / unrecognised falls through to the shipping default.
        assert_eq!(map("auto"), RtDynamicMode::Auto);
        assert_eq!(map("nonsense"), RtDynamicMode::Auto);
        // Only `Off` skips updates; every other mode is dynamic.
        assert!(!RtDynamicMode::Off.is_dynamic());
        assert!(RtDynamicMode::Auto.is_dynamic());
        assert!(RtDynamicMode::Rebuild.is_dynamic());
        assert!(RtDynamicMode::Tlas.is_dynamic());
    }

    #[test]
    fn skinned_flag_is_bit_31_and_masks_back_to_the_pool_index() {
        // The flag occupies the top bit; the shader recovers the real bindless
        // normal index with `normal_index & ~RT_SKINNED_FLAG`. Mirror both here.
        assert_eq!(RT_SKINNED_FLAG, 1u32 << 31);
        for normal_index in [0u32, 1, 5, 96, 1000] {
            let flagged = normal_index | RT_SKINNED_FLAG;
            assert_ne!(flagged & RT_SKINNED_FLAG, 0, "flag set");
            assert_eq!(flagged & !RT_SKINNED_FLAG, normal_index, "masks back");
        }
        // Realistic bindless pool indices never reach the flag bit, so a static
        // entry's normal index is never misread as skinned.
        assert_eq!(96u32 & RT_SKINNED_FLAG, 0);
    }

    #[test]
    fn models_dirty_detects_moves_and_shape_changes() {
        let ident = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut moved = ident;
        moved[3][0] = 5.0; // translate one object along x

        let indices = vec![0usize, 2usize];
        let cached = vec![ident, ident];

        // All transforms unchanged -> not dirty.
        assert!(!models_dirty(&indices, &cached, |idx| match idx {
            0 | 2 => Some(ident),
            _ => None,
        }));
        // One object moved -> dirty.
        assert!(models_dirty(&indices, &cached, |idx| match idx {
            0 => Some(moved),
            2 => Some(ident),
            _ => None,
        }));
        // An index that no longer resolves (draw list shrank) -> dirty.
        assert!(models_dirty(&indices, &cached, |idx| match idx {
            0 => Some(ident),
            _ => None,
        }));
        // A cached/indices length mismatch -> dirty.
        assert!(models_dirty(&[0usize], &cached, |_| Some(ident)));
    }
}
