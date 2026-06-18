// src/vulkan/raytrace.rs
//
// Vulkan ray-query acceleration structures for the hardware ray-traced
// reflection pass. Builds, from the shared static vertex / index buffers and the
// `DrawObject` + `InstancedCluster` lists, the bottom- and top-level
// acceleration structures (BLAS / TLAS) the inline-`rayQueryEXT` reflection
// shader traces against, plus a per-instance geometry table the shader uses to
// fetch the hit triangle and shade it.
//
// One triangle BLAS per participating static object (over its slice of the
// shared buffers) and one per instanced cluster; one TLAS instance per object
// and one per cluster instance (transform = the object/instance model matrix,
// `instanceCustomIndex` = the geometry-table index). The BLAS describe
// object-space geometry and never change for a rigid transform; only the TLAS
// instance transforms (and the geometry table's per-instance model matrices the
// shader shades with) move when a prop moves.
//
// Mirrors `directx/raytrace.rs` (DXR inline ray tracing). Skinned geometry is
// added per frame (`rebuild_skinned`): a compute pass deforms each skinned
// object's bind-pose vertices into a fresh model-space buffer, one u16-indexed
// BLAS per skinned object is built over it, and the TLAS + geometry table are
// rebuilt over the persistent static/cluster BLAS plus the fresh skinned tail.
// The dynamic-transform update (`RtDynamicMode`) rebuilds the TLAS + geometry
// table with fresh allocations on the frames a participating transform actually
// changed, parking the outgoing structures in a frames-in-flight-deep retire
// pool so a prior frame's still-in-flight trace keeps reading the old structures
// while the new frame uses the new ones (the Vulkan renderer fences
// `frames_in_flight`-deep via the `in_flight` fences, so this is hazard-free
// without a new fence). Unlike DXR
// (which binds the TLAS as a root SRV by GPU virtual address each frame), Vulkan
// binds the TLAS + geometry table through a descriptor set, so the RT pass
// re-points the current frame's set at the live handles every frame; see
// `post::rt_reflections::VkContext::rt_update_descriptors`.
//
// TODO(rt-pipeline-vulkan): this uses `VK_KHR_ray_query` (inline tracing in the
// reflection fragment shader), the direct analog of the DXR 1.1 `RayQuery` path.
// A future `VK_KHR_ray_tracing_pipeline` path (raygen/closest-hit/miss + a shader
// binding table) would only be worth it if a feature needs recursive tracing or
// per-material hit shaders, which screen-space reflections do not.

use ash::{Device, vk};

use crate::gfx::render_types::{DrawObject, InstancedCluster, RtGeomEntry, SkinnedDrawObject};

use super::pipeline::{compile_glsl_rt, spv_module};
use super::texture::create_buffer;

// Byte stride of a `Vertex` in the shared vertex buffer (pos + normal + tangent
// + colour + uv = 14 floats). The BLAS reads positions at this stride and the
// shader fetches attributes at this stride. The deformed (posed) skinned vertex
// buffer the skin kernel writes carries the same 56-byte layout.
const VERTEX_STRIDE: u64 = 56;

// Marks a `RtGeomEntry.normal_index` as belonging to a skinned object: the
// reflection trace then fetches the hit triangle from the deformed-vertex / u16
// skinned index buffers instead of the static u32 ones. Bit 31 is free (bindless
// pool indices never approach 2^31); matches render_types / rt_reflections.frag.
const RT_SKINNED_FLAG: u32 = 0x8000_0000;

// GLSL source for the RT skinning compute kernel (compiled via shaderc to
// SPIR-V 1.4 / Vulkan 1.2, the same target the ray-query shaders use).
const RT_SKIN_COMP_GLSL: &str = include_str!("shaders/rt_skin.comp");

// Per-dispatch parameters for the `rt_skin` compute kernel; matches the GLSL
// `SkinParams` push-constant block (16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct SkinParams {
    vertex_base: u32,
    vertex_count: u32,
    joint_count: u32,
    _pad: u32,
}

// How the scene acceleration structure is kept current when props move. Selected
// once at init from `CN_RT_DYNAMIC`; unset gives `Auto`, the shipping behaviour.
// Mirrors the DirectX mode ladder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum RtDynamicMode {
    // Build once, never update. Forces a static BVH even if props move.
    Off,
    // Default. Rebuild the TLAS + table (fresh allocations, static BLAS) only on
    // the frames a participating transform actually changed.
    Auto,
    // Force a full TLAS + table rebuild every frame, dirty or not. Diagnostic.
    Rebuild,
    // Same GPU work as `Auto` minus the dirty gate. Diagnostic.
    Tlas,
}

impl RtDynamicMode {
    pub(super) fn from_env() -> Self {
        match std::env::var("CN_RT_DYNAMIC").as_deref() {
            Ok("off") => Self::Off,
            Ok("rebuild") => Self::Rebuild,
            Ok("tlas") => Self::Tlas,
            _ => Self::Auto,
        }
    }

    pub(super) fn is_dynamic(self) -> bool {
        self != Self::Off
    }
}

// Pack a column-major object-to-world `model` matrix into a Vulkan instance
// transform: a 3x4 ROW-major affine (`VkTransformMatrixKHR`, `matrix[3][4]`),
// row r = `[m_r0 m_r1 m_r2 m_r3]` where element (row r, col c) is the world-matrix
// value. The Rust `model` is column-major, so math element (r, c) lives at
// `model[c][r]`. `VkTransformMatrixKHR` and the DXR 3x4 row-major transform are
// byte-identical, so this is the same packing as `directx::raytrace`. Unit-tested.
pub(super) fn pack_instance_transform(model: [[f32; 4]; 4]) -> vk::TransformMatrixKHR {
    vk::TransformMatrixKHR {
        matrix: [
            model[0][0],
            model[1][0],
            model[2][0],
            model[3][0],
            model[0][1],
            model[1][1],
            model[2][1],
            model[3][1],
            model[0][2],
            model[1][2],
            model[2][2],
            model[3][2],
        ],
    }
}

// Bindless-pool (albedo, normal) indices for a draw whose authored albedo /
// normal-map slots are `texture_slot` / `normal_map_slot`. The Vulkan bindless
// pool is the deduplicated `[albedo..] ++ [normal..]` image set, so albedo =
// `texture_slot` (clamped) and normal = `albedo_count + normal_map_slot`
// (clamped), matching `draw.rs::build_object_buffer`. The textured RT shader
// binds that same pool and indexes it with these.
fn pool_indices(
    texture_slot: usize,
    normal_map_slot: usize,
    albedo_count: usize,
    last_tex: usize,
    last_nm: usize,
) -> (u32, u32) {
    let albedo = texture_slot.min(last_tex) as u32;
    let normal = (albedo_count + normal_map_slot.min(last_nm)) as u32;
    (albedo, normal)
}

// Build the geometry-table entry for one static draw object.
fn geom_entry(
    obj: &DrawObject,
    albedo_count: usize,
    last_tex: usize,
    last_nm: usize,
) -> RtGeomEntry {
    let (albedo_index, normal_index) = pool_indices(
        obj.texture_slot,
        obj.normal_map_slot,
        albedo_count,
        last_tex,
        last_nm,
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

// Build the geometry-table entry for one instance of an instanced cluster. The
// cluster's shared mesh slice uses base_vertex 0 (its indices are absolute);
// `model` is this instance's transform.
fn cluster_geom_entry(
    cluster: &InstancedCluster,
    model: [[f32; 4]; 4],
    albedo_count: usize,
    last_tex: usize,
    last_nm: usize,
) -> RtGeomEntry {
    let (albedo_index, normal_index) = pool_indices(
        cluster.texture_slot,
        cluster.normal_map_slot,
        albedo_count,
        last_tex,
        last_nm,
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
// so `base_vertex` is 0 and the model matrix brings the hit to world space. The
// skinned flag is OR'd into `normal_index` so the trace fetches from the
// deformed / u16 buffers. The skinned object's albedo / normal-map images bake
// into the shared bindless pool from the same `texture_slot` / `normal_map_slot`
// as the static path, so its pool indices come out of `pool_indices` exactly
// like a static draw, letting skinned hits shade textured. Mirrors
// `directx::raytrace::skinned_geom_entry`.
fn skinned_geom_entry(
    obj: &SkinnedDrawObject,
    albedo_count: usize,
    last_tex: usize,
    last_nm: usize,
) -> RtGeomEntry {
    let (albedo_index, normal_index) = pool_indices(
        obj.texture_slot,
        obj.normal_map_slot,
        albedo_count,
        last_tex,
        last_nm,
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

// True when any participating object's current model matrix differs from the one
// baked into the live TLAS. Pure (no GPU) so the dirty gate is unit-testable.
fn models_dirty(cached: &[[[f32; 4]; 4]], current: &[[[f32; 4]; 4]]) -> bool {
    cached.len() != current.len() || cached.iter().zip(current).any(|(a, b)| a != b)
}

// One TLAS instance descriptor: explicit 3x4 transform, custom index (indexes
// the geometry table), full visibility mask, no SBT offset / flags, and the BLAS
// device address. Inline tracing ignores hit groups so the SBT fields are zero.
fn tlas_instance(
    model: [[f32; 4]; 4],
    custom_index: u32,
    blas_address: u64,
) -> vk::AccelerationStructureInstanceKHR {
    vk::AccelerationStructureInstanceKHR {
        transform: pack_instance_transform(model),
        // instanceCustomIndex (low 24) + mask (high 8 = 0xFF).
        instance_custom_index_and_mask: vk::Packed24_8::new(custom_index & 0x00FF_FFFF, 0xFFu8),
        // instanceShaderBindingTableRecordOffset (24) + flags (8), both zero.
        instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, 0u8),
        acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
            device_handle: blas_address,
        },
    }
}

// Round `value` up to a multiple of `align` (a power of two). Used for the
// scratch buffer's `minAccelerationStructureScratchOffsetAlignment`.
fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        (value + align - 1) & !(align - 1)
    }
}

// A device-local buffer holding an acceleration structure plus its handle.
// `size` is the backing buffer's byte size, so a recycled `AccelBuffer` can be
// reused in place when a later build still fits.
struct AccelBuffer {
    accel: vk::AccelerationStructureKHR,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

impl AccelBuffer {
    fn destroy(&self, device: &Device, as_loader: &ash::khr::acceleration_structure::Device) {
        unsafe {
            as_loader.destroy_acceleration_structure(self.accel, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

// A host-visible buffer (the geometry table + the TLAS instance descriptors),
// filled once at creation and read by the GPU. A fresh one is allocated on each
// dynamic rebuild, so there is no need to keep it mapped past the initial copy.
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl HostBuffer {
    fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

// A plain device-local buffer (the deformed-vertex buffer the skin pass writes
// and the skinned BLAS + reflection trace read). Owns its memory + cached device
// address. `size` is the byte size, so a recycled buffer can be reused in place
// when a later rebuild still fits.
pub(super) struct DeviceBuffer {
    pub(super) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    address: u64,
    size: u64,
}

impl DeviceBuffer {
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

// The compute pipeline that deforms skinned vertices for ray tracing
// (`rt_skin.comp`): set 0 = [src skinned verts, joint palette, deformed output]
// (three storage buffers) + a 16-byte `SkinParams` push-constant block. Built in
// `build_rt_accel` (gated on RT) and held on `RtAccelData`; mirrors DirectX's
// `SkinPipeline` / Metal's `skin_pipeline`.
pub(super) struct SkinPipeline {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Per-(frame, object) compute descriptor sets, sized + allocated lazily on
    // the first `rebuild_skinned` (the skinned object count is unknown at init,
    // before `upload_skinned` runs). Indexed `[frame_idx][object]`; rewritten in
    // place each rebuild at the current frame's slot (fence-gated, so safe, like
    // the RT resolve set's per-frame re-point).
    descriptor_pool: vk::DescriptorPool,
    sets: Vec<Vec<vk::DescriptorSet>>,
}

impl SkinPipeline {
    pub(super) fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            if self.descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
            }
        }
    }
}

// The per-frame skinned-geometry inputs `rebuild_skinned` needs to deform and
// add skinned objects to the BVH. Assembled by `rt_dynamic_update` from the
// context's skinned state.
pub(super) struct SkinnedRtInputs<'a> {
    // One entry per skinned mesh (only `visible`, real-triangle objects build).
    pub objects: &'a [SkinnedDrawObject],
    // The shared bind-pose skinned vertex buffer (`SkinnedVertex`, 80-byte
    // stride) the skin kernel reads, bound as the compute set's binding 0.
    pub vertex_buffer: vk::Buffer,
    // The shared u16 skinned index buffer the skinned BLAS + reflection trace
    // address the deformed buffer with. Its device address is the BLAS index
    // input; the buffer handle is the trace's SSBO.
    pub index_buffer: vk::Buffer,
    // This frame's per-object joint-palette buffers, parallel to `objects` (each
    // is that object's `MAX_JOINTS`-matrix upload buffer for the current frame),
    // bound as the compute set's binding 1.
    pub joint_buffers: &'a [vk::Buffer],
}

// Orphaned skinned BLAS parked by a skinned -> static transition for deferred
// free. When the last skinned object turns invisible, the rebuilt static TLAS no
// longer references the skinned BLAS, but a prior frame's in-flight trace may
// still read them, so they are freed only after `free_at` frames have elapsed (by
// then the frames-in-flight fence guarantees no in-flight trace references them).
// The per-frame static / skinned buffers recycle through their rings instead, so
// this pool only ever holds this rare transition's BLAS.
struct Retired {
    free_at: u64,
    blas: Vec<AccelBuffer>,
}

impl Retired {
    fn destroy(&self, device: &Device, as_loader: &ash::khr::acceleration_structure::Device) {
        for b in &self.blas {
            b.destroy(device, as_loader);
        }
    }
}

// An outgoing scratch buffer parked by a `grow_scratch`, freed once the
// frames-in-flight fence retires the frames whose build could still read it.
struct RetiredScratch {
    free_at: u64,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

// One frame slot's recyclable skinned-rebuild buffers. The skinned rebuild swaps
// the live set (`self.*`) with the slot every frame: it recycles the buffers this
// slot last held (displaced `frames_in_flight` frames ago, so their fence has
// signalled and no in-flight trace still reads them), rebuilds into them in place,
// and parks the outgoing live set back here. Reuse is hazard-free for the same
// reason the retire pool's deferred free was; the difference is the buffers are
// reused rather than freed + reallocated, so the steady state allocates nothing.
// This replaces the per-frame allocate-fresh-every-frame + retire-pool churn that
// grew the driver's video-memory pool without bound. Ownership stays single: each
// buffer is owned by exactly the live `self.*` fields or one ring slot. Each
// buffer self-describes its byte size, so a slot is recreated only when a later
// build outgrows it.
#[derive(Default)]
struct SkinnedFrameRing {
    deformed: Option<DeviceBuffer>,
    // One BLAS per skinned object.
    blas: Vec<AccelBuffer>,
    tlas: Option<AccelBuffer>,
    instance: Option<HostBuffer>,
    geom: Option<HostBuffer>,
}

impl SkinnedFrameRing {
    fn destroy(&mut self, device: &Device, as_loader: &ash::khr::acceleration_structure::Device) {
        if let Some(d) = &self.deformed {
            d.destroy(device);
        }
        for b in &self.blas {
            b.destroy(device, as_loader);
        }
        if let Some(t) = &self.tlas {
            t.destroy(device, as_loader);
        }
        if let Some(i) = &self.instance {
            i.destroy(device);
        }
        if let Some(g) = &self.geom {
            g.destroy(device);
        }
    }
}

// One ring slot of the per-rebuild static-transform buffers (the TLAS + its
// instance descriptors + the geometry table). The dynamic-transform rebuild
// advances `static_cursor` to the next slot each rebuild and recycles that slot's
// buffers in place (re-map + copy / build-over), growing one only when a later
// rebuild outgrows it (the static instance count is fixed, so the steady state
// allocates nothing). Reuse is hazard-free: the cursor revisits a slot only after
// a full ring cycle, by which point the frames-in-flight fence has retired every
// trace that read it. This replaces the allocate-fresh-every-rebuild + retire-pool
// path, whose per-frame churn grew the driver's video-memory pool without bound
// when a prop animated continuously. Ownership stays single: each buffer lives in
// exactly the live `self.*` fields or one ring slot (swapped, never cloned).
#[derive(Default)]
struct StaticFrameRing {
    tlas: Option<AccelBuffer>,
    instance: Option<HostBuffer>,
    geom: Option<HostBuffer>,
}

impl StaticFrameRing {
    fn destroy(&self, device: &Device, as_loader: &ash::khr::acceleration_structure::Device) {
        if let Some(t) = &self.tlas {
            t.destroy(device, as_loader);
        }
        if let Some(i) = &self.instance {
            i.destroy(device);
        }
        if let Some(g) = &self.geom {
            g.destroy(device);
        }
    }
}

// Advance a ring cursor to the next slot, wrapping at `len`. Pure so the
// wrap-around is unit-testable without a device.
fn next_slot(cursor: usize, len: usize) -> usize {
    (cursor + 1) % len.max(1)
}

// The Vulkan ray-query acceleration structures + geometry table for hardware ray
// tracing. Held on the context behind an `Option`; present only when RT
// reflections are enabled, the GPU exposes the ray-query extensions, and the
// scene has resident geometry.
pub(super) struct RtAccelData {
    as_loader: ash::khr::acceleration_structure::Device,

    // BLAS in build order: one per participating static object (in
    // `object_indices` order), then one per instanced cluster, then one per
    // skinned object. The leading `static_blas_count` entries are the persistent
    // static + cluster BLAS, built once and never rebuilt (a rigid transform
    // leaves object-space geometry unchanged); the skinned tail
    // (`blas[static_blas_count..]`) is rebuilt each frame from the current pose.
    blas: Vec<AccelBuffer>,
    // How many leading `blas` entries are the persistent static + cluster BLAS. A
    // skinned object's BLAS index is `static_blas_count + si`.
    static_blas_count: usize,
    // The top-level (instance) acceleration structure the trace reads.
    tlas: AccelBuffer,
    // `[RtGeomEntry; instance_count]` (host-visible), bound as a storage buffer;
    // indexed by the trace's `instanceCustomIndex`.
    geom_table: HostBuffer,
    // The TLAS instance-descriptor buffer (host-visible). Only the TLAS *build*
    // reads it; the live set swapped out of a `static_ring` / `skinned_ring` slot.
    instance_buffer: HostBuffer,
    // Scratch sized for the largest of every BLAS build and the TLAS build;
    // reused by the per-frame TLAS rebuild (the instance count is fixed). Its
    // device address is pre-aligned to the scratch-offset alignment. The skinned
    // rebuild grows it (retiring the old) when a skinned BLAS + TLAS build needs
    // more; `scratch_capacity` is the buffer's byte size.
    scratch_buffer: vk::Buffer,
    scratch_memory: vk::DeviceMemory,
    scratch_addr: u64,
    scratch_capacity: u64,
    // Outgoing scratch buffers parked by a `grow_scratch` for deferred free.
    retired_scratch: Vec<RetiredScratch>,
    // Size the TLAS prebuild reported; the static rebuild recycles the ring slot's
    // TLAS at this size (the static instance count is fixed).
    tlas_size: u64,
    instance_count: u32,
    // Frames-in-flight depth; a retired structure is freed this many frames
    // after the rebuild that displaced it (by then its frame's fence has
    // signalled, so no in-flight trace can still read it).
    frames_in_flight: u64,

    // Per-frame update state.
    // Indices into the frame's `draw_objects` for the participating objects, in
    // BLAS / instance order. Lets a rebuild re-read current transforms in build
    // order and detect a changed draw list.
    object_indices: Vec<usize>,
    // BLAS device addresses, parallel to `blas`, cached so a rebuild re-emits the
    // instance descriptors without re-querying.
    blas_addresses: Vec<u64>,
    // Each participating object's model matrix as baked into the live TLAS. The
    // `Auto` dirty check compares the live draw list against these.
    cached_models: Vec<[[f32; 4]; 4]>,
    // The TLAS instance descriptors for every cluster instance, re-appended
    // verbatim on a rebuild (clusters are baked static into the BVH).
    cluster_instances: Vec<vk::AccelerationStructureInstanceKHR>,
    // The geometry-table entries for the cluster instances, parallel to
    // `cluster_instances`.
    cluster_geom: Vec<RtGeomEntry>,
    // Bindless-pool sizing for the geometry-table pool indices on a rebuild.
    albedo_count: usize,
    last_tex: usize,
    last_nm: usize,

    // Deferred-free pool (for the rare orphaned skinned BLAS on a skinned ->
    // static transition) + the monotonic per-update counter that drives it +
    // `retired_scratch`. The per-frame static / skinned buffers recycle through
    // their rings, so this no longer churns on the steady-state rebuild path.
    retire: Vec<Retired>,
    frame_counter: u64,

    // Per-rebuild static-transform buffers (see `StaticFrameRing`), recycled in
    // place by the static `rebuild_tlas` path. `static_cursor` advances one slot
    // per rebuild; a slot is revisited only after a full ring cycle, so its prior
    // trace has retired. The skinned path uses `skinned_ring` instead.
    static_ring: Vec<StaticFrameRing>,
    static_cursor: usize,

    // Per-frame skinned-rebuild buffers, one slot per frame in flight, recycled in
    // place (see `SkinnedFrameRing`). Indexed by `frame_idx`.
    skinned_ring: Vec<SkinnedFrameRing>,

    // Skinned geometry.
    // The compute-skinning pipeline (`rt_skin`). `Some` only when the GLSL
    // compile + pipeline creation succeeded; without it skinned geometry is
    // absent from the BVH (the RT pass still runs for static geometry).
    skin: Option<SkinPipeline>,
    // The fresh-per-rebuild deformed (posed) skinned vertex buffer the skin pass
    // writes and the skinned BLAS + reflection trace read. A 1-element dummy when
    // the scene has no skinned geometry, so the trace's binding is always valid.
    // The skinned rebuild allocates a new one each frame and retires the old (a
    // prior frame's trace may still read it). Re-pointed onto the RT descriptor
    // set each frame, like the TLAS.
    deformed_verts: DeviceBuffer,
    // The shared u16 skinned index buffer (the BLAS index input + the trace's
    // SSBO). A dummy `vk::Buffer::null()`-backed handle when there is no skinned
    // geometry; the post pass binds a dummy SSBO in that case.
    skinned_indices: vk::Buffer,
    // Whether any skinned object is currently live in the BVH (drives whether the
    // per-frame update runs `rebuild_skinned` or the static `rebuild_tlas`).
    has_skinned: bool,
    frames_in_flight_usize: usize,
}

// Raw pointers in `HostBuffer` are host-mapped and only touched on the render
// thread; the acceleration-structure loader holds plain fn pointers. The whole
// struct lives inside `VkContext`, which is already `unsafe impl Send`.
unsafe impl Send for RtAccelData {}

impl RtAccelData {
    // The live TLAS handle (bound through the RT pass's descriptor set).
    pub(super) fn tlas(&self) -> vk::AccelerationStructureKHR {
        self.tlas.accel
    }

    // The live geometry-table buffer + its byte range (bound as a storage buffer).
    pub(super) fn geom_table(&self) -> (vk::Buffer, vk::DeviceSize) {
        (self.geom_table.buffer, self.geom_table.size)
    }

    // The live deformed (posed) skinned vertex buffer (bound as the RT pass's
    // skinned-verts SSBO). Fresh per skinned rebuild, so the RT pass re-points
    // its descriptor at this every frame, like the TLAS. A 1-element dummy when
    // the scene has no skinned geometry, so the binding is always valid.
    pub(super) fn deformed_verts(&self) -> vk::Buffer {
        self.deformed_verts.buffer
    }

    // The shared u16 skinned index buffer (bound as the RT pass's skinned-index
    // SSBO). `vk::Buffer::null()` when there is no skinned geometry; the post
    // pass substitutes a dummy SSBO so the binding is always live.
    pub(super) fn skinned_indices(&self) -> vk::Buffer {
        self.skinned_indices
    }
}

// Per-build geometry parameters captured once, used both for sizing and for the
// recorded build (so the temporary `vk::*` builder structs can be reconstructed
// cheaply inside the command-buffer recording closure).
struct BlasParams {
    vertex_address: u64,
    max_vertex: u32,
    index_byte_offset: u32,
    primitive_count: u32,
}

fn blas_geometry(p: &BlasParams, index_address: u64) -> vk::AccelerationStructureGeometryKHR<'_> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: p.vertex_address,
        })
        .vertex_stride(VERTEX_STRIDE)
        .max_vertex(p.max_vertex)
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

// Same as `blas_geometry` but over a u16 index buffer + the deformed (posed)
// skinned vertex buffer. The skinned BLAS bakes absolute u16 indices into the
// deformed buffer (base vertex folded to 0), so `vertex_address` is the deformed
// buffer's base address and `index_address` is the u16 index buffer offset for
// this object. Same 56-byte vertex stride as the static path.
fn skinned_blas_geometry(
    p: &BlasParams,
    index_address: u64,
) -> vk::AccelerationStructureGeometryKHR<'_> {
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR {
            device_address: p.vertex_address,
        })
        .vertex_stride(VERTEX_STRIDE)
        .max_vertex(p.max_vertex)
        .index_type(vk::IndexType::UINT16)
        .index_data(vk::DeviceOrHostAddressConstKHR {
            device_address: index_address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

fn tlas_geometry(instance_address: u64) -> vk::AccelerationStructureGeometryKHR<'static> {
    let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default()
        .array_of_pointers(false)
        .data(vk::DeviceOrHostAddressConstKHR {
            device_address: instance_address,
        });
    vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::INSTANCES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { instances })
        .flags(vk::GeometryFlagsKHR::OPAQUE)
}

// Device address of a buffer (core in Vulkan 1.2; the device enables
// `bufferDeviceAddress` for the RT path).
fn buffer_address(device: &Device, buffer: vk::Buffer) -> u64 {
    unsafe {
        device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    }
}

// Allocate a fresh acceleration-structure backing buffer + create the AS handle.
fn create_accel(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    as_loader: &ash::khr::acceleration_structure::Device,
    size: u64,
    ty: vk::AccelerationStructureTypeKHR,
) -> Result<AccelBuffer, String> {
    let size = size.max(256);
    let (buffer, memory) = create_buffer(
        instance,
        device,
        pd,
        size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let info = vk::AccelerationStructureCreateInfoKHR::default()
        .buffer(buffer)
        .offset(0)
        .size(size)
        .ty(ty);
    let accel = unsafe { as_loader.create_acceleration_structure(&info, None) }
        .map_err(|e| format!("create acceleration structure: {e}"))?;
    Ok(AccelBuffer {
        accel,
        buffer,
        memory,
        size,
    })
}

// Allocate a host-visible, persistently-mapped buffer of `size` bytes with the
// given usage, copy `data` into it, and return the mapped handle.
fn create_host_buffer<T: Copy>(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    data: &[T],
    usage: vk::BufferUsageFlags,
    label: &str,
) -> Result<HostBuffer, String> {
    let size = (std::mem::size_of_val(data) as vk::DeviceSize).max(16);
    let (buffer, memory) = create_buffer(
        instance,
        device,
        pd,
        size,
        usage,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let ptr = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
        .map_err(|e| format!("map {label}: {e}"))? as *mut u8;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr, std::mem::size_of_val(data));
        // HOST_COHERENT + written once: unmap immediately, the GPU reads it as-is.
        device.unmap_memory(memory);
    }
    Ok(HostBuffer {
        buffer,
        memory,
        size,
    })
}

// Reuse `existing` (re-map + copy `data` into it) when it can hold the data, else
// destroy it and allocate fresh. The recycled skinned-rebuild host buffers are
// rewritten every frame, so this keeps them allocation-free in the steady state
// while still growing on demand. `existing` must have been created with `usage`
// (the ring only ever stores a buffer of the matching usage in each slot).
fn write_or_recreate_host<T: Copy>(
    existing: Option<HostBuffer>,
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    data: &[T],
    usage: vk::BufferUsageFlags,
    label: &str,
) -> Result<HostBuffer, String> {
    let needed = (std::mem::size_of_val(data) as vk::DeviceSize).max(16);
    if let Some(buf) = existing {
        if buf.size >= needed {
            let ptr =
                unsafe { device.map_memory(buf.memory, 0, buf.size, vk::MemoryMapFlags::empty()) }
                    .map_err(|e| format!("map {label}: {e}"))? as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    ptr,
                    std::mem::size_of_val(data),
                );
                device.unmap_memory(buf.memory);
            }
            return Ok(buf);
        }
        buf.destroy(device);
    }
    create_host_buffer(instance, device, pd, data, usage, label)
}

// Allocate a fresh device-local buffer usable as the deformed-vertex buffer: a
// storage buffer (skin compute writes it, the trace reads it), a BLAS vertex
// input, and device-addressable (the BLAS reads it by address). Caches the
// device address.
fn create_device_buffer(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    size: u64,
) -> Result<DeviceBuffer, String> {
    let size = size.max(VERTEX_STRIDE);
    let (buffer, memory) = create_buffer(
        instance,
        device,
        pd,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let address = buffer_address(device, buffer);
    Ok(DeviceBuffer {
        buffer,
        memory,
        address,
        size,
    })
}

// Build the `rt_skin` compute pipeline: a 3-storage-buffer descriptor set layout
// (set 0: src skinned verts, joint palette, deformed output) + a 16-byte
// `SkinParams` push constant. Returns `Err` when shaderc is unavailable or the
// kernel fails to compile; the caller then leaves the skin pipeline absent and
// skinned geometry is omitted from the BVH (the RT pass still runs for static
// geometry). Per-(frame, object) descriptor sets are allocated lazily on the
// first `rebuild_skinned`, when the skinned object count is known.
pub(super) fn build_skin_pipeline(
    device: &Device,
    hot_reload: bool,
) -> Result<SkinPipeline, String> {
    let src = super::pipeline::shader_source(hot_reload, "rt_skin.comp", RT_SKIN_COMP_GLSL);
    let spv = compile_glsl_rt(&src, shaderc::ShaderKind::Compute, "rt_skin.comp")?;
    let module = spv_module(device, &spv)?;

    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..3u32)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        })
        .collect();
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )
    }
    .map_err(|e| {
        unsafe { device.destroy_shader_module(module, None) };
        format!("rt skin descriptor set layout: {e}")
    })?;

    let pc = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<SkinParams>() as u32);
    let set_layouts = [set_layout];
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(std::slice::from_ref(&pc)),
            None,
        )
    }
    .map_err(|e| {
        unsafe {
            device.destroy_shader_module(module, None);
            device.destroy_descriptor_set_layout(set_layout, None);
        }
        format!("rt skin pipeline layout: {e}")
    })?;

    let entry = std::ffi::CString::new("main").unwrap();
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(&entry);
    let info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&info),
            None,
        )
    };
    unsafe { device.destroy_shader_module(module, None) };
    let pipeline = pipeline.map_err(|(_, e)| {
        unsafe {
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(set_layout, None);
        }
        format!("create rt skin pipeline: {e}")
    })?[0];

    Ok(SkinPipeline {
        set_layout,
        pipeline_layout,
        pipeline,
        descriptor_pool: vk::DescriptorPool::null(),
        sets: Vec::new(),
    })
}

// A global acceleration-structure-build memory barrier: orders one build's
// writes before the next build reads/writes (shared scratch reuse + TLAS reading
// the just-built BLAS). Mirrors the DXR UAV barrier between builds.
fn build_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
        .dst_access_mask(
            vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
                | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
        );
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
            vk::DependencyFlags::empty(),
            std::slice::from_ref(&barrier),
            &[],
            &[],
        );
    }
}

// Query the device's minimum scratch-offset alignment for AS builds.
fn scratch_alignment(instance: &ash::Instance, pd: vk::PhysicalDevice) -> u64 {
    let mut as_props = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut as_props);
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    (as_props.min_acceleration_structure_scratch_offset_alignment as u64).max(1)
}

// Build the BLAS / TLAS / geometry table for the scene on a one-shot command
// buffer (submitted and fence-waited so the structures are ready before the
// first frame traces them). Returns `Ok(None)` when there is no resident
// triangle geometry to trace: the caller then leaves RT disabled and falls back
// to SSR.
//
// `albedo_count` is the bindless albedo-pool length (cluster + object normal
// indices offset past it); `total_vertices` is the shared vertex buffer's vertex
// count (used to bound each geometry's `max_vertex`).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_rt_accel(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    draw_objects: &[DrawObject],
    clusters: &[InstancedCluster],
    albedo_count: usize,
    normal_count: usize,
    total_vertices: usize,
    frames_in_flight: usize,
    hot_reload: bool,
) -> Result<Option<RtAccelData>, String> {
    let as_loader = ash::khr::acceleration_structure::Device::new(instance, device);
    let last_tex = albedo_count.saturating_sub(1);
    let last_nm = normal_count.saturating_sub(1);

    // Participating static objects + clusters (real triangles, resident).
    let object_indices: Vec<usize> = draw_objects
        .iter()
        .enumerate()
        .filter(|(_, o)| o.resident && o.index_count >= 3)
        .map(|(i, _)| i)
        .collect();
    let cluster_list: Vec<(usize, &InstancedCluster)> = clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| c.index_count >= 3 && !c.instances.is_empty())
        .collect();
    if object_indices.is_empty() && cluster_list.is_empty() {
        return Ok(None);
    }

    let vbuf_addr = buffer_address(device, vertex_buffer);
    let ibuf_addr = buffer_address(device, index_buffer);

    // One BLAS-build params entry per participating object first, then clusters.
    // Each object folds its base_vertex into the vertex device address + uses its
    // mesh-relative indices (the shader adds base_vertex back via the geom table),
    // mirroring the DirectX vertex-address fold.
    let mut params: Vec<BlasParams> = Vec::with_capacity(object_indices.len() + cluster_list.len());
    for &i in &object_indices {
        let obj = &draw_objects[i];
        let base_vertex = obj.base_vertex as u64;
        params.push(BlasParams {
            vertex_address: vbuf_addr + base_vertex * VERTEX_STRIDE,
            max_vertex: (total_vertices as u64)
                .saturating_sub(base_vertex)
                .saturating_sub(1) as u32,
            index_byte_offset: obj.index_offset as u32 * 4,
            primitive_count: (obj.index_count / 3) as u32,
        });
    }
    for (_, c) in &cluster_list {
        params.push(BlasParams {
            vertex_address: vbuf_addr,
            max_vertex: (total_vertices as u64).saturating_sub(1) as u32,
            index_byte_offset: c.index_offset as u32 * 4,
            primitive_count: (c.index_count / 3) as u32,
        });
    }

    // Size + allocate each BLAS; track the largest scratch requirement.
    let mut blas: Vec<AccelBuffer> = Vec::with_capacity(params.len());
    let mut max_scratch: u64 = 0;
    for p in &params {
        let geo = blas_geometry(p, ibuf_addr);
        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&geo));
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[p.primitive_count],
                &mut sizes,
            );
        }
        blas.push(create_accel(
            instance,
            device,
            pd,
            &as_loader,
            sizes.acceleration_structure_size,
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        )?);
        max_scratch = max_scratch.max(sizes.build_scratch_size);
    }
    let blas_addresses: Vec<u64> = blas
        .iter()
        .map(|b| unsafe {
            as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(b.accel),
            )
        })
        .collect();

    // Instance descriptors + geometry table, in instance order: static objects
    // (each referencing its own BLAS), then every cluster instance (referencing
    // the cluster's single BLAS, each with its own transform + geom entry).
    let draw_blas_count = object_indices.len();
    let mut instances: Vec<vk::AccelerationStructureInstanceKHR> =
        Vec::with_capacity(object_indices.len());
    let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(object_indices.len());
    for (slot, &i) in object_indices.iter().enumerate() {
        let obj = &draw_objects[i];
        instances.push(tlas_instance(obj.model, slot as u32, blas_addresses[slot]));
        geom_entries.push(geom_entry(obj, albedo_count, last_tex, last_nm));
    }
    let mut cluster_instances: Vec<vk::AccelerationStructureInstanceKHR> = Vec::new();
    let mut cluster_geom: Vec<RtGeomEntry> = Vec::new();
    for (ci, (_, c)) in cluster_list.iter().enumerate() {
        let blas_address = blas_addresses[draw_blas_count + ci];
        for model in &c.instances {
            let id = (instances.len() + cluster_instances.len()) as u32;
            cluster_instances.push(tlas_instance(*model, id, blas_address));
            cluster_geom.push(cluster_geom_entry(
                c,
                *model,
                albedo_count,
                last_tex,
                last_nm,
            ));
        }
    }
    instances.extend_from_slice(&cluster_instances);
    geom_entries.extend_from_slice(&cluster_geom);
    let instance_count = instances.len() as u32;

    let instance_buffer = create_host_buffer(
        instance,
        device,
        pd,
        &instances,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
        "RT instance buffer",
    )?;
    let geom_table = create_host_buffer(
        instance,
        device,
        pd,
        &geom_entries,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        "RT geometry table",
    )?;

    // Size + allocate the TLAS + the shared scratch (>= the largest BLAS/TLAS).
    let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
    let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(std::slice::from_ref(&tlas_geo));
    let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        as_loader.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &tlas_build_info,
            &[instance_count],
            &mut tlas_sizes,
        );
    }
    max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
    let tlas = create_accel(
        instance,
        device,
        pd,
        &as_loader,
        tlas_sizes.acceleration_structure_size,
        vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    )?;

    // Scratch sized to the largest build + the offset alignment so the aligned
    // device address still leaves room for the largest scratch requirement.
    let align = scratch_alignment(instance, pd);
    let scratch_capacity = max_scratch + align;
    let (scratch_buffer, scratch_memory) = create_buffer(
        instance,
        device,
        pd,
        scratch_capacity,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let scratch_addr = align_up(buffer_address(device, scratch_buffer), align);

    // Record every BLAS build (build-barrier-serialised over the shared scratch),
    // then the TLAS build, on a one-shot command buffer; fence-wait so the BVH is
    // ready before the first trace.
    super::texture::one_shot_submit(device, command_pool, queue, |cmd| {
        for (slot, p) in params.iter().enumerate() {
            let geo = blas_geometry(p, ibuf_addr);
            let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geo));
            bi.dst_acceleration_structure = blas[slot].accel;
            bi.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            };
            let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(p.primitive_count)
                .primitive_offset(p.index_byte_offset)
                .first_vertex(0)
                .transform_offset(0);
            unsafe {
                as_loader.cmd_build_acceleration_structures(
                    cmd,
                    std::slice::from_ref(&bi),
                    &[std::slice::from_ref(&range)],
                );
            }
            build_barrier(device, cmd);
        }
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas.accel;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        unsafe {
            as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
        }
    })?;

    let cached_models = object_indices
        .iter()
        .map(|&i| draw_objects[i].model)
        .collect();
    let static_blas_count = blas.len();

    // Skinned geometry is seeded on the first dynamic frame (like DirectX /
    // Metal), so the init build is static-only. Allocate a 1-element dummy
    // deformed-vertex buffer so the trace's skinned-verts SSBO always binds a
    // valid resource; the first `rebuild_skinned` replaces it with the real one.
    let deformed_verts = create_device_buffer(instance, device, pd, VERTEX_STRIDE)?;

    // The compute-skinning pipeline (gated on RT, which is the only path that
    // reaches `build_rt_accel`). A build failure is non-fatal: the RT pass still
    // runs for static geometry, just without skinned hits.
    let skin = match build_skin_pipeline(device, hot_reload) {
        Ok(s) => Some(s),
        Err(e) => {
            tracing::warn!(
                "RT skin pipeline build failed (skinned meshes absent from reflections): {e}"
            );
            None
        }
    };

    Ok(Some(RtAccelData {
        as_loader,
        blas,
        static_blas_count,
        tlas,
        geom_table,
        instance_buffer,
        scratch_buffer,
        scratch_memory,
        scratch_addr,
        scratch_capacity,
        retired_scratch: Vec::new(),
        tlas_size: tlas_sizes.acceleration_structure_size,
        instance_count,
        frames_in_flight: (frames_in_flight.max(1)) as u64,
        object_indices,
        blas_addresses,
        cached_models,
        cluster_instances,
        cluster_geom,
        albedo_count,
        last_tex,
        last_nm,
        retire: Vec::new(),
        frame_counter: 0,
        static_ring: (0..frames_in_flight.max(1))
            .map(|_| StaticFrameRing::default())
            .collect(),
        static_cursor: 0,
        skinned_ring: (0..frames_in_flight.max(1))
            .map(|_| SkinnedFrameRing::default())
            .collect(),
        skin,
        deformed_verts,
        skinned_indices: vk::Buffer::null(),
        has_skinned: false,
        frames_in_flight_usize: frames_in_flight.max(1),
    }))
}

impl RtAccelData {
    // Per-frame dynamic update, recorded onto `cmd` (the frame's "start" command
    // buffer, submitted before every per-pass trace on the single graphics
    // queue). Drains the retire pool, then, when the mode + dirty gate call for
    // it, rebuilds the TLAS + geometry table from current transforms with fresh
    // allocations and parks the outgoing structures for deferred free. A
    // transient failure is non-fatal (keeps the live BVH). Returns whether a
    // rebuild ran (the caller logs failures via the warn inside).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dynamic_update(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        mode: RtDynamicMode,
        frame_idx: usize,
        skinned: Option<SkinnedRtInputs>,
    ) {
        self.frame_counter += 1;
        let now = self.frame_counter;
        // Free any retired resources whose frames-in-flight window has elapsed.
        let mut i = 0;
        while i < self.retire.len() {
            if self.retire[i].free_at <= now {
                let r = self.retire.swap_remove(i);
                r.destroy(device, &self.as_loader);
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.retired_scratch.len() {
            if self.retired_scratch[i].free_at <= now {
                let s = self.retired_scratch.swap_remove(i);
                unsafe {
                    device.destroy_buffer(s.buffer, None);
                    device.free_memory(s.memory, None);
                }
            } else {
                i += 1;
            }
        }

        if !mode.is_dynamic() {
            return;
        }

        // Re-collect current transforms in BLAS order. A changed draw-list shape
        // (an index now out of range / non-resident) is left for a full rebuild
        // elsewhere; skip this frame.
        let mut current = Vec::with_capacity(self.object_indices.len());
        for &idx in &self.object_indices {
            match draw_objects.get(idx) {
                Some(o) if o.resident && o.index_count >= 3 => current.push(o.model),
                _ => return,
            }
        }

        // Skinned objects visible this frame, paired with their index into the
        // joint-buffer list. The skin pipeline must be present (GLSL compiled);
        // with none, skinned geometry stays absent (the static path runs).
        let skinned_objects: Vec<(usize, &SkinnedDrawObject)> = match (&self.skin, &skinned) {
            (Some(_), Some(s)) => s
                .objects
                .iter()
                .enumerate()
                .filter(|(_, o)| o.visible && o.index_count >= 3)
                .collect(),
            _ => Vec::new(),
        };

        // Skinned geometry present: always re-skin + rebuild (the pose changes
        // every frame), regardless of the dirty gate.
        if !skinned_objects.is_empty() {
            let s = skinned.expect("skinned_objects non-empty implies inputs present");
            if let Err(e) = self.rebuild_skinned(
                instance,
                device,
                pd,
                cmd,
                draw_objects,
                &current,
                &s,
                &skinned_objects,
                frame_idx,
            ) {
                tracing::warn!("RT skinned rebuild failed (keeping live BVH): {e}");
            }
            return;
        }

        // No skinned geometry this frame: if the BVH still carries a skinned tail
        // (an object just turned invisible), drop it back to the static head with
        // a fresh TLAS so the trace stops reaching stale skinned BLAS. Otherwise
        // fall through to the dirty-gated static rebuild.
        let needs_rebuild = match mode {
            RtDynamicMode::Auto => self.has_skinned || models_dirty(&self.cached_models, &current),
            RtDynamicMode::Rebuild | RtDynamicMode::Tlas => true,
            RtDynamicMode::Off => false,
        };
        if !needs_rebuild {
            return;
        }

        if let Err(e) = self.rebuild_tlas(instance, device, pd, cmd, draw_objects, &current, now) {
            tracing::warn!("RT dynamic TLAS rebuild failed (keeping live BVH): {e}");
        }
    }

    // Rebuild the TLAS + geometry table from `current` transforms, recycling the
    // next `static_ring` slot's buffers in place, and record the build onto `cmd`.
    // The BLAS are kept (rigid transforms leave object-space geometry unchanged).
    #[allow(clippy::too_many_arguments)]
    fn rebuild_tlas(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        current: &[[[f32; 4]; 4]],
        now: u64,
    ) -> Result<(), String> {
        // Freshly-transformed draw-object instances, then the cluster instances
        // re-appended verbatim. The geometry table mirrors this order.
        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> =
            Vec::with_capacity(self.object_indices.len() + self.cluster_instances.len());
        let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(instances.capacity());
        for (slot, &idx) in self.object_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instances.push(tlas_instance(
                obj.model,
                slot as u32,
                self.blas_addresses[slot],
            ));
            geom_entries.push(geom_entry(
                obj,
                self.albedo_count,
                self.last_tex,
                self.last_nm,
            ));
        }
        instances.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);

        // Refresh the live instance count so the TLAS build below covers exactly
        // this rebuild's descriptors. A prior skinned rebuild may have left a
        // larger count; reusing it would read past the valid instance buffer.
        self.instance_count = instances.len() as u32;

        // Advance to the next ring slot and recycle its buffers in place. The slot
        // was last current a full ring cycle ago, so the frames-in-flight fence has
        // retired every trace that read it; the static instance count is fixed, so
        // the host buffers + TLAS are reused without growing after warm-up.
        self.static_cursor = next_slot(self.static_cursor, self.static_ring.len());
        let mut slot = std::mem::take(&mut self.static_ring[self.static_cursor]);
        let instance_buffer = write_or_recreate_host(
            slot.instance.take(),
            instance,
            device,
            pd,
            &instances,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
        )?;
        let geom_table = write_or_recreate_host(
            slot.geom.take(),
            instance,
            device,
            pd,
            &geom_entries,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
        )?;
        let tlas = match slot.tlas.take() {
            Some(b) if b.size >= self.tlas_size => b,
            Some(b) => {
                b.destroy(device, &self.as_loader);
                create_accel(
                    instance,
                    device,
                    pd,
                    &self.as_loader,
                    self.tlas_size,
                    vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                )?
            }
            None => create_accel(
                instance,
                device,
                pd,
                &self.as_loader,
                self.tlas_size,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            )?,
        };

        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas.accel;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: self.scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(self.instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        unsafe {
            self.as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
            // Order the build before this frame's trace reads the TLAS.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // Swap the just-built buffers into the live set and park the displaced live
        // set back in this ring slot, to be recycled a full ring cycle later (by
        // then its fence has signalled, so no in-flight trace still reads it).
        let displaced_tlas = std::mem::replace(&mut self.tlas, tlas);
        let displaced_geom = std::mem::replace(&mut self.geom_table, geom_table);
        let displaced_instance = std::mem::replace(&mut self.instance_buffer, instance_buffer);
        slot.tlas = Some(displaced_tlas);
        slot.geom = Some(displaced_geom);
        slot.instance = Some(displaced_instance);
        self.static_ring[self.static_cursor] = slot;

        // If a skinned tail was still owned (the last skinned object just turned
        // invisible), the rebuilt static TLAS no longer references it: drop it back
        // to the static head and retire the orphaned skinned BLAS. No ring slot
        // recycles them and a prior frame's trace may still read them, so they go
        // through the deferred-free pool.
        if self.blas.len() > self.static_blas_count {
            self.has_skinned = false;
            let orphaned_blas = self.blas.split_off(self.static_blas_count);
            self.retire.push(Retired {
                free_at: now + self.frames_in_flight,
                blas: orphaned_blas,
            });
        }
        self.cached_models = current.to_vec();
        Ok(())
    }

    // Per-frame skinned update, recorded onto `cmd` (the frame's "start" command
    // buffer, which supports compute dispatch + AS builds). Keeps the persistent
    // static + cluster BLAS, re-skins this frame's pose into a fresh deformed
    // buffer, rebuilds one u16 BLAS per skinned object over it, and rebuilds the
    // TLAS + geometry table over the static head plus the fresh skinned tail. All
    // with fresh allocations; the outgoing structures / buffers are parked in the
    // retire pool (not freed in place) until the frames-in-flight fence retires
    // the frames whose still-in-flight trace could read them.
    //
    // The three GPU steps are recorded in dependency order on the one command
    // buffer: skin dispatch (writes the deformed buffer), a pipeline barrier
    // (COMPUTE write -> AS-build + FRAGMENT read), then the BLAS/TLAS build (reads
    // it). The start buffer is submitted before every per-pass trace, so build ->
    // trace is ordered by submission too.
    #[allow(clippy::too_many_arguments)]
    fn rebuild_skinned(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        cmd: vk::CommandBuffer,
        draw_objects: &[DrawObject],
        current: &[[[f32; 4]; 4]],
        skinned: &SkinnedRtInputs,
        skinned_objects: &[(usize, &SkinnedDrawObject)],
        frame_idx: usize,
    ) -> Result<(), String> {
        let skin = self
            .skin
            .as_ref()
            .ok_or("rebuild_skinned called without a skin pipeline")?;
        let pipeline = skin.pipeline;
        let pipeline_layout = skin.pipeline_layout;

        // This frame slot's recyclable buffers, taken out for the swap (sidesteps
        // the `&mut self` borrow while reading other fields; put back at the end).
        let mut slot = std::mem::take(&mut self.skinned_ring[frame_idx]);

        // Deformed-vertex buffer: the skin pass writes posed `Vertex`s here,
        // mirroring the skinned VB's indexing so the u16 index buffer addresses it
        // directly. Sized to the highest vertex the skinned objects reach. Recycle
        // this slot's prior deformed buffer in place when it still fits, else grow.
        let deformed_extent: u64 = skinned_objects
            .iter()
            .map(|(_, o)| o.vertex_base as u64 + o.vertex_count as u64)
            .max()
            .unwrap_or(0);
        let deformed_bytes = (deformed_extent * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let deformed = match slot.deformed.take() {
            Some(buf) if buf.size >= deformed_bytes => buf,
            Some(buf) => {
                buf.destroy(device);
                create_device_buffer(instance, device, pd, deformed_bytes)?
            }
            None => create_device_buffer(instance, device, pd, deformed_bytes)?,
        };

        // Ensure per-(frame, object) compute descriptor sets exist for this
        // skinned object count, then point this frame's sets at the skinned VB
        // (binding 0), each object's current-frame joint buffer (binding 1), and
        // the fresh deformed buffer (binding 2).
        self.ensure_skin_sets(device, skinned.objects.len())?;
        let skin = self.skin.as_ref().expect("skin pipeline present");
        let frame_sets = &skin.sets[frame_idx];
        for (obj_idx, _) in skinned_objects {
            let joint_buffer = skinned
                .joint_buffers
                .get(*obj_idx)
                .copied()
                .unwrap_or(vk::Buffer::null());
            if joint_buffer == vk::Buffer::null() {
                continue;
            }
            let src_info = vk::DescriptorBufferInfo::default()
                .buffer(skinned.vertex_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let pal_info = vk::DescriptorBufferInfo::default()
                .buffer(joint_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let dst_info = vk::DescriptorBufferInfo::default()
                .buffer(deformed.buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let set = frame_sets[*obj_idx];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&src_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&pal_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&dst_info)),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        // Stage 1: skin dispatch per visible skinned object onto `cmd`.
        let skin = self.skin.as_ref().expect("skin pipeline present");
        let frame_sets = &skin.sets[frame_idx];
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        }
        for (obj_idx, obj) in skinned_objects {
            let joint_buffer = skinned
                .joint_buffers
                .get(*obj_idx)
                .copied()
                .unwrap_or(vk::Buffer::null());
            if joint_buffer == vk::Buffer::null() {
                continue;
            }
            let params = SkinParams {
                vertex_base: obj.vertex_base as u32,
                vertex_count: obj.vertex_count as u32,
                joint_count: obj.joint_count.max(1) as u32,
                _pad: 0,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &params as *const SkinParams as *const u8,
                    std::mem::size_of::<SkinParams>(),
                )
            };
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline_layout,
                    0,
                    std::slice::from_ref(&frame_sets[*obj_idx]),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                device.cmd_dispatch(cmd, (obj.vertex_count as u32).div_ceil(64), 1, 1);
            }
        }

        // Order the skin writes before the BLAS build (AS-build input geometry)
        // and the later hit-shader read (the trace samples the deformed buffer as
        // an SSBO in a fragment shader). An AS build does not auto-synchronise
        // against a prior compute write to its input vertex buffer, so this
        // cross-pass residency barrier is required (Metal / DirectX document the
        // same).
        unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR | vk::AccessFlags::SHADER_READ,
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // Stage 2: one u16 BLAS per skinned object over the deformed buffer.
        let skinned_idx_addr = buffer_address(device, skinned.index_buffer);
        let max_vertex = deformed_extent.saturating_sub(1) as u32;
        let skinned_params: Vec<(BlasParams, u64)> = skinned_objects
            .iter()
            .map(|(_, obj)| {
                (
                    BlasParams {
                        vertex_address: deformed.address,
                        max_vertex,
                        // u16 indices = 2 bytes each.
                        index_byte_offset: obj.index_offset as u32 * 2,
                        primitive_count: (obj.index_count / 3) as u32,
                    },
                    skinned_idx_addr,
                )
            })
            .collect();

        // Size each skinned BLAS, recycling this slot's prior BLAS in place when it
        // still fits (else growing); track the largest scratch. Leftover recycled
        // BLAS (the skinned count shrank) are destroyed.
        let mut recycled_blas = std::mem::take(&mut slot.blas).into_iter();
        let mut skinned_blas: Vec<AccelBuffer> = Vec::with_capacity(skinned_params.len());
        let mut max_scratch: u64 = 0;
        for (p, idx_addr) in &skinned_params {
            let geo = skinned_blas_geometry(p, *idx_addr);
            let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geo));
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                self.as_loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[p.primitive_count],
                    &mut sizes,
                );
            }
            let needed = sizes.acceleration_structure_size;
            let blas = match recycled_blas.next() {
                Some(b) if b.size >= needed => b,
                Some(b) => {
                    b.destroy(device, &self.as_loader);
                    create_accel(
                        instance,
                        device,
                        pd,
                        &self.as_loader,
                        needed,
                        vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                    )?
                }
                None => create_accel(
                    instance,
                    device,
                    pd,
                    &self.as_loader,
                    needed,
                    vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                )?,
            };
            skinned_blas.push(blas);
            max_scratch = max_scratch.max(sizes.build_scratch_size);
        }
        for leftover in recycled_blas {
            leftover.destroy(device, &self.as_loader);
        }
        let skinned_blas_addresses: Vec<u64> = skinned_blas
            .iter()
            .map(|b| unsafe {
                self.as_loader.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                        .acceleration_structure(b.accel),
                )
            })
            .collect();

        // Instance descriptors + geometry table, in instance order: static
        // objects (current transforms), then the cluster instances verbatim, then
        // one per skinned object (BLAS index `static_blas_count + si`).
        let mut instances: Vec<vk::AccelerationStructureInstanceKHR> =
            Vec::with_capacity(self.object_indices.len() + self.cluster_instances.len());
        let mut geom_entries: Vec<RtGeomEntry> = Vec::with_capacity(instances.capacity());
        for (slot, &idx) in self.object_indices.iter().enumerate() {
            let obj = &draw_objects[idx];
            instances.push(tlas_instance(
                obj.model,
                slot as u32,
                self.blas_addresses[slot],
            ));
            geom_entries.push(geom_entry(
                obj,
                self.albedo_count,
                self.last_tex,
                self.last_nm,
            ));
        }
        instances.extend_from_slice(&self.cluster_instances);
        geom_entries.extend_from_slice(&self.cluster_geom);
        for (si, (_, obj)) in skinned_objects.iter().enumerate() {
            let id = instances.len() as u32;
            instances.push(tlas_instance(obj.model, id, skinned_blas_addresses[si]));
            // The skinned object's textures bake into the shared bindless pool
            // from its own `texture_slot` / `normal_map_slot`, so the pool index
            // reads off `obj` directly (no list-position dependence).
            geom_entries.push(skinned_geom_entry(
                obj,
                self.albedo_count,
                self.last_tex,
                self.last_nm,
            ));
        }
        let instance_count = instances.len() as u32;

        // Recycle this slot's prior host buffers in place (re-map + copy) when they
        // still fit, else grow.
        let instance_buffer = write_or_recreate_host(
            slot.instance.take(),
            instance,
            device,
            pd,
            &instances,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            "RT instance buffer",
        )?;
        let geom_table = write_or_recreate_host(
            slot.geom.take(),
            instance,
            device,
            pd,
            &geom_entries,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "RT geometry table",
        )?;

        // Size the TLAS + scratch (>= the largest skinned BLAS + the TLAS). The
        // skinned instance count can change frame to frame, so size the TLAS from
        // this frame's prebuild rather than the cached size.
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
        let tlas_build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        let mut tlas_sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            self.as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &tlas_build_info,
                &[instance_count],
                &mut tlas_sizes,
            );
        }
        max_scratch = max_scratch.max(tlas_sizes.build_scratch_size);
        // Recycle this slot's prior TLAS in place when it still fits, else grow.
        let tlas = match slot.tlas.take() {
            Some(b) if b.size >= tlas_sizes.acceleration_structure_size => b,
            Some(b) => {
                b.destroy(device, &self.as_loader);
                create_accel(
                    instance,
                    device,
                    pd,
                    &self.as_loader,
                    tlas_sizes.acceleration_structure_size,
                    vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                )?
            }
            None => create_accel(
                instance,
                device,
                pd,
                &self.as_loader,
                tlas_sizes.acceleration_structure_size,
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            )?,
        };

        // The shared scratch was sized for the static build; the skinned BLAS +
        // this frame's TLAS may need more. Grow it (retire the old) if so.
        let align = scratch_alignment(instance, pd);
        if max_scratch + align > self.scratch_size() {
            self.grow_scratch(instance, device, pd, max_scratch, align, self.frame_counter)?;
        }
        let scratch_addr = self.scratch_addr;

        // Record the skinned BLAS builds (build-barrier-serialised over the shared
        // scratch), then the TLAS build, on `cmd`.
        for (si, (p, idx_addr)) in skinned_params.iter().enumerate() {
            let geo = skinned_blas_geometry(p, *idx_addr);
            let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(std::slice::from_ref(&geo));
            bi.dst_acceleration_structure = skinned_blas[si].accel;
            bi.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: scratch_addr,
            };
            let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
                .primitive_count(p.primitive_count)
                .primitive_offset(p.index_byte_offset)
                .first_vertex(0)
                .transform_offset(0);
            unsafe {
                self.as_loader.cmd_build_acceleration_structures(
                    cmd,
                    std::slice::from_ref(&bi),
                    &[std::slice::from_ref(&range)],
                );
            }
            build_barrier(device, cmd);
        }
        let tlas_geo = tlas_geometry(buffer_address(device, instance_buffer.buffer));
        let mut bi = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(std::slice::from_ref(&tlas_geo));
        bi.dst_acceleration_structure = tlas.accel;
        bi.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_addr,
        };
        let range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instance_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);
        unsafe {
            self.as_loader.cmd_build_acceleration_structures(
                cmd,
                std::slice::from_ref(&bi),
                &[std::slice::from_ref(&range)],
            );
            // Order the TLAS build before this frame's trace reads it.
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_access_mask(vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }

        // Swap live <-> slot: the just-built buffers (recycled from this slot, or
        // grown) become the live set; the outgoing live set is parked back in the
        // slot to be recycled the next time this frame slot comes around (by then
        // its fence has signalled, so no in-flight trace still reads it). No
        // allocation, no retire-pool growth. The static/cluster head of `blas` is
        // untouched; only the skinned tail rotates.
        let displaced_blas: Vec<AccelBuffer> = if self.blas.len() > self.static_blas_count {
            self.blas.split_off(self.static_blas_count)
        } else {
            Vec::new()
        };
        self.blas.extend(skinned_blas);
        let displaced_tlas = std::mem::replace(&mut self.tlas, tlas);
        let displaced_geom = std::mem::replace(&mut self.geom_table, geom_table);
        let displaced_instance = std::mem::replace(&mut self.instance_buffer, instance_buffer);
        let displaced_deformed = std::mem::replace(&mut self.deformed_verts, deformed);
        self.instance_count = instance_count;
        self.skinned_indices = skinned.index_buffer;

        slot.deformed = Some(displaced_deformed);
        slot.blas = displaced_blas;
        slot.tlas = Some(displaced_tlas);
        slot.instance = Some(displaced_instance);
        slot.geom = Some(displaced_geom);
        self.skinned_ring[frame_idx] = slot;

        self.has_skinned = true;
        self.cached_models = current.to_vec();
        Ok(())
    }

    // Current scratch buffer byte size (queried lazily; the scratch was sized
    // `max_scratch + align` at the build that allocated it).
    fn scratch_size(&self) -> u64 {
        self.scratch_capacity
    }

    // Grow the shared scratch buffer to cover `required + align` bytes and retire
    // the old (a prior frame's build may still read it). Re-aligns the cached
    // scratch device address.
    fn grow_scratch(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        pd: vk::PhysicalDevice,
        required: u64,
        align: u64,
        now: u64,
    ) -> Result<(), String> {
        let new_capacity = required + align;
        let (buffer, memory) = create_buffer(
            instance,
            device,
            pd,
            new_capacity,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let addr = align_up(buffer_address(device, buffer), align);
        let old_buffer = std::mem::replace(&mut self.scratch_buffer, buffer);
        let old_memory = std::mem::replace(&mut self.scratch_memory, memory);
        self.scratch_addr = addr;
        self.scratch_capacity = new_capacity;
        self.retired_scratch.push(RetiredScratch {
            free_at: now + self.frames_in_flight,
            buffer: old_buffer,
            memory: old_memory,
        });
        Ok(())
    }

    // Ensure the per-(frame, object) compute descriptor sets cover `object_count`
    // skinned objects. Allocated lazily on the first skinned rebuild (the count
    // is unknown at init, before `upload_skinned`). Idempotent once sized.
    fn ensure_skin_sets(&mut self, device: &Device, object_count: usize) -> Result<(), String> {
        let frames = self.frames_in_flight_usize;
        let skin = self
            .skin
            .as_mut()
            .ok_or("ensure_skin_sets called without a skin pipeline")?;
        ensure_skin_sets(device, skin, frames, object_count)
    }

    // Destroy every acceleration-structure resource. The caller has already
    // idled the device.
    pub(super) fn destroy(&mut self, device: &Device) {
        for r in self.retire.drain(..) {
            r.destroy(device, &self.as_loader);
        }
        for s in self.retired_scratch.drain(..) {
            unsafe {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }
        for slot in &mut self.skinned_ring {
            slot.destroy(device, &self.as_loader);
        }
        for slot in &self.static_ring {
            slot.destroy(device, &self.as_loader);
        }
        for b in &self.blas {
            b.destroy(device, &self.as_loader);
        }
        self.tlas.destroy(device, &self.as_loader);
        self.geom_table.destroy(device);
        self.instance_buffer.destroy(device);
        self.deformed_verts.destroy(device);
        if let Some(skin) = &self.skin {
            skin.destroy(device);
        }
        unsafe {
            device.destroy_buffer(self.scratch_buffer, None);
            device.free_memory(self.scratch_memory, None);
        }
    }
}

// Grow a `SkinPipeline`'s per-(frame, object) descriptor-set pool to hold at least
// `object_count` objects per frame, reallocating the pool from scratch when it must
// grow. A no-op when the pool already holds enough (or `object_count == 0`). Shared
// by the RT skin path (`RtAccelData::ensure_skin_sets`) and the GPU-driven main-pass
// skin fold (`VkContext::build_main_skin`).
pub(super) fn ensure_skin_sets(
    device: &Device,
    skin: &mut SkinPipeline,
    frames: usize,
    object_count: usize,
) -> Result<(), String> {
    let have = skin.sets.first().map(|s| s.len()).unwrap_or(0);
    if object_count == 0 || have >= object_count {
        return Ok(());
    }
    // Re-allocate the pool from scratch sized for the (possibly grown) count. The
    // old pool's sets are only ever bound on the frame's own command buffer, which
    // has completed (the per-frame fence gated the frame at the top of
    // `draw_frame`), so freeing the old pool here is safe.
    unsafe {
        if skin.descriptor_pool != vk::DescriptorPool::null() {
            device.destroy_descriptor_pool(skin.descriptor_pool, None);
        }
    }
    let total = (frames * object_count) as u32;
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(total * 3);
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(total),
            None,
        )
    }
    .map_err(|e| format!("skin descriptor pool: {e}"))?;
    let mut sets: Vec<Vec<vk::DescriptorSet>> = Vec::with_capacity(frames);
    for _ in 0..frames {
        let layouts: Vec<vk::DescriptorSetLayout> =
            (0..object_count).map(|_| skin.set_layout).collect();
        let alloc = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .map_err(|e| format!("alloc skin descriptor sets: {e}"))?;
        sets.push(alloc);
    }
    skin.descriptor_pool = pool;
    skin.sets = sets;
    Ok(())
}

// Allocate a device-local buffer for the GPU-driven main pass's per-frame deformed
// skinned vertices: a storage buffer the `rt_skin` compute writes + a vertex buffer
// the bindless main pass draws. Unlike the RT deformed buffer it needs no
// acceleration-structure / device-address usage (the main pass binds it as a vertex
// buffer, not by address), so this stays independent of the RT feature being enabled.
pub(super) fn create_main_deformed_buffer(
    instance: &ash::Instance,
    device: &Device,
    pd: vk::PhysicalDevice,
    size: u64,
) -> Result<DeviceBuffer, String> {
    let size = size.max(VERTEX_STRIDE);
    let (buffer, memory) = create_buffer(
        instance,
        device,
        pd,
        size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    Ok(DeviceBuffer {
        buffer,
        memory,
        address: 0,
        size,
    })
}

impl super::context::VkContext {
    // Build the GPU-driven main-pass skinning resources: the `rt_skin` compute
    // pipeline (reused independently of RT), one deformed-vertex buffer per
    // frame-in-flight (storage + vertex usage), and the per-(frame, object)
    // descriptor sets pointing at [skinned bind-pose VB, this object's joint
    // buffer, this frame's deformed buffer]. The deformed + joint buffers are
    // stable for the world's lifetime, so the sets are written once here (no
    // per-frame re-point). Sets `self.n_skinned`, which engages the fold. Called
    // from `upload_skinned` when the bindless cull path is active. Mirrors the
    // DirectX `upload_skinned` skin block.
    pub(in crate::vulkan) fn build_main_skin(&mut self, vertex_total: usize) -> Result<(), String> {
        let device = self.device.clone();
        let frames = self.frames_in_flight.max(1);
        let n = self.skinned.draw_objects.len();
        if n == 0 {
            return Ok(());
        }

        let mut skin = build_skin_pipeline(&device, self.hot_reload)?;
        ensure_skin_sets(&device, &mut skin, frames, n)?;

        let deformed_bytes = (vertex_total as u64 * VERTEX_STRIDE).max(VERTEX_STRIDE);
        let mut deformed: Vec<DeviceBuffer> = Vec::with_capacity(frames);
        for _ in 0..frames {
            deformed.push(create_main_deformed_buffer(
                &self.instance,
                &device,
                self.physical_device,
                deformed_bytes,
            )?);
        }

        // Point every set at its buffers once: binding 0 = the shared bind-pose
        // skinned VB, binding 1 = this object's joint buffer for that frame,
        // binding 2 = that frame's deformed output.
        let src_buffer = self.skinned.vertex_buffer;
        for (f, deformed_buf) in deformed.iter().enumerate() {
            for o in 0..n {
                let joint_buffer = self.skinned.joint_buffers[f][o];
                let set = skin.sets[f][o];
                let src_info = vk::DescriptorBufferInfo::default()
                    .buffer(src_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let pal_info = vk::DescriptorBufferInfo::default()
                    .buffer(joint_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let dst_info = vk::DescriptorBufferInfo::default()
                    .buffer(deformed_buf.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&src_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&pal_info)),
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&dst_info)),
                ];
                unsafe { device.update_descriptor_sets(&writes, &[]) };
            }
        }

        self.skinned.skin = Some(skin);
        self.skinned.deformed = deformed;
        // Fresh ring: no slot has been posed yet, so the G-buffer velocity must
        // treat the previous deformed buffer as the current one until a full
        // frame has primed it.
        self.skinned
            .deformed_primed
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.n_skinned = n;
        Ok(())
    }

    // Per-frame main-pass skinning compute pass: deform every skinned object's
    // bind-pose vertices into this frame's deformed buffer, which the bindless
    // main pass's 2nd indirect draw reads as a vertex buffer. A no-op when the
    // fold is inactive (no skin pipeline / deformed buffer). Run in the Cull graph
    // arm after `encode_cull`, before Main; mirrors the stage-1 skin dispatch in
    // `rebuild_skinned` but targets a per-frame vertex buffer and barriers to
    // VERTEX_ATTRIBUTE_READ instead of the RT BLAS-build read. Independent of RT.
    pub(in crate::vulkan) fn encode_skin(&self, cmd: vk::CommandBuffer, frame_idx: usize) {
        let Some(skin) = self.skinned.skin.as_ref() else {
            return;
        };
        if self.n_skinned == 0 || self.skinned.deformed.len() <= frame_idx {
            return;
        }
        let device = &self.device;
        let frame_sets = &skin.sets[frame_idx];
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, skin.pipeline);
        }
        for (o, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.n_skinned)
            .enumerate()
        {
            let params = SkinParams {
                vertex_base: obj.vertex_base as u32,
                vertex_count: obj.vertex_count as u32,
                joint_count: obj.joint_count.max(1) as u32,
                _pad: 0,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &params as *const SkinParams as *const u8,
                    std::mem::size_of::<SkinParams>(),
                )
            };
            unsafe {
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    skin.pipeline_layout,
                    0,
                    std::slice::from_ref(&frame_sets[o]),
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    skin.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                device.cmd_dispatch(cmd, (obj.vertex_count as u32).div_ceil(64), 1, 1);
            }
        }
        // Order the skin writes before the main pass's vertex fetch of the
        // deformed buffer.
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ);
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&barrier),
                &[],
                &[],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_mode_from_env_default_is_auto() {
        // The env var isn't set in the test process, so it resolves to Auto.
        assert_eq!(RtDynamicMode::from_env(), RtDynamicMode::Auto);
        assert!(RtDynamicMode::Auto.is_dynamic());
        assert!(RtDynamicMode::Rebuild.is_dynamic());
        assert!(RtDynamicMode::Tlas.is_dynamic());
        assert!(!RtDynamicMode::Off.is_dynamic());
    }

    #[test]
    fn next_slot_wraps_around_the_ring() {
        // Advancing the static-rebuild cursor cycles through every slot and wraps
        // at the end, so a slot is revisited only after a full ring cycle.
        assert_eq!(next_slot(0, 3), 1);
        assert_eq!(next_slot(1, 3), 2);
        assert_eq!(next_slot(2, 3), 0);
        // A degenerate single-slot ring always returns slot 0.
        assert_eq!(next_slot(0, 1), 0);
    }

    #[test]
    fn pack_instance_transform_transposes_column_major_to_3x4_row_major() {
        // A column-major model with a known translation column [10, 20, 30].
        let model = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ];
        let t = pack_instance_transform(model);
        // VkTransformMatrixKHR is 3x4 row-major (flat); the translation is the
        // last entry of each 4-wide row.
        assert_eq!(
            t.matrix,
            [
                1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0
            ]
        );
    }

    #[test]
    fn pack_instance_transform_preserves_a_rotation_shear() {
        // Distinct values in every cell so a row/col swap would be detectable.
        let model = [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
            [10.0, 11.0, 12.0, 1.0],
        ];
        let t = pack_instance_transform(model);
        // Flat row-major: row r is [model[0][r], model[1][r], model[2][r], model[3][r]].
        assert_eq!(
            t.matrix,
            [
                1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0
            ]
        );
    }

    #[test]
    fn pool_indices_match_the_bindless_dedup_layout() {
        // albedo = texture_slot (clamped); normal = albedo_count + normal_map_slot
        // (clamped). With 5 albedos + 3 normals: object texture 2 / normal 1.
        assert_eq!(pool_indices(2, 1, 5, 4, 2), (2, 6));
        // Clamping: out-of-range slots saturate to the last valid index.
        assert_eq!(pool_indices(9, 9, 5, 4, 2), (4, 7));
    }

    #[test]
    fn instance_packs_custom_index_and_full_mask() {
        let d = tlas_instance(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            7,
            0xDEAD_BEEF,
        );
        assert_eq!(d.instance_custom_index_and_mask.low_24(), 7);
        assert_eq!(d.instance_custom_index_and_mask.high_8(), 0xFF);
        assert_eq!(
            unsafe { d.acceleration_structure_reference.device_handle },
            0xDEAD_BEEF
        );
    }

    #[test]
    fn models_dirty_detects_a_changed_transform() {
        let a = [[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]];
        let mut b = a;
        assert!(!models_dirty(&a, &b));
        b[0][3][0] = 5.0;
        assert!(models_dirty(&a, &b));
        // A length change is dirty.
        assert!(models_dirty(&a, &[]));
    }

    #[test]
    fn align_up_rounds_to_power_of_two() {
        assert_eq!(align_up(0, 256), 0);
        assert_eq!(align_up(1, 256), 256);
        assert_eq!(align_up(256, 256), 256);
        assert_eq!(align_up(257, 256), 512);
        // align <= 1 is identity.
        assert_eq!(align_up(123, 1), 123);
    }

    #[test]
    fn rt_skin_kernel_compiles() {
        // The skin compute kernel compiles to SPIR-V (ray-query target, the same
        // Vulkan-1.2 / SPIR-V-1.4 env the trace shaders use). Guards the
        // `GL_EXT`-free GLSL + the std430 byte indexing.
        let spv = compile_glsl_rt(
            RT_SKIN_COMP_GLSL,
            shaderc::ShaderKind::Compute,
            "rt_skin.comp",
        )
        .expect("rt skin kernel compiles");
        assert!(super::super::pipeline::is_spirv(&spv));
    }

    #[test]
    fn skin_params_layout_matches_glsl() {
        // GLSL `SkinParams` push-constant block in rt_skin.comp: four tightly
        // packed uints (16 bytes). The skin kernel reads them as push constants,
        // so the byte offsets must line up with this `#[repr(C)]` struct.
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<SkinParams>(), 16);
        assert_eq!(offset_of!(SkinParams, vertex_base), 0);
        assert_eq!(offset_of!(SkinParams, vertex_count), 4);
        assert_eq!(offset_of!(SkinParams, joint_count), 8);
        assert_eq!(offset_of!(SkinParams, _pad), 12);
    }

    #[test]
    fn skinned_vertex_layout_pins_the_glsl_scalar_offsets() {
        // The skin kernel (rt_skin.comp) reads the 80-byte `SkinnedVertex` as a
        // flat float/uint array: pos@0, normal@12, tangent@24, color@36, uv@48,
        // u16 joints[4]@56 (words 14/15), f32 weights[4]@64 (words 16..19). Pin
        // those byte offsets here so a struct reshuffle is caught.
        use crate::gfx::mesh_payload::SkinnedVertex;
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<SkinnedVertex>(), 80);
        assert_eq!(offset_of!(SkinnedVertex, pos), 0);
        assert_eq!(offset_of!(SkinnedVertex, normal), 12);
        assert_eq!(offset_of!(SkinnedVertex, tangent), 24);
        assert_eq!(offset_of!(SkinnedVertex, color), 36);
        assert_eq!(offset_of!(SkinnedVertex, uv), 48);
        assert_eq!(offset_of!(SkinnedVertex, joints), 56);
        assert_eq!(offset_of!(SkinnedVertex, weights), 64);
    }

    #[test]
    fn vertex_layout_pins_the_deformed_buffer_offsets() {
        // The skin kernel writes the deformed buffer in the static 56-byte
        // `Vertex` layout (pos@0, normal@12, tangent@24, color@36, uv@48), and the
        // trace's skinned fetchers read it back at those offsets.
        use crate::gfx::mesh_payload::Vertex;
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<Vertex>(), 56);
        assert_eq!(size_of::<Vertex>() as u64, VERTEX_STRIDE);
        assert_eq!(offset_of!(Vertex, pos), 0);
        assert_eq!(offset_of!(Vertex, normal), 12);
        assert_eq!(offset_of!(Vertex, tangent), 24);
        assert_eq!(offset_of!(Vertex, color), 36);
        assert_eq!(offset_of!(Vertex, uv), 48);
    }

    #[test]
    fn skinned_flag_is_bit_31_and_masks_back_to_the_pool_index() {
        // The flag occupies the top bit; the shader recovers the real bindless
        // normal index with `normal_index & ~RT_SKINNED_FLAG`. Matches the bit-31
        // flag in rt_reflections.frag + render_types / directx / metal.
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
    fn skinned_geom_entry_flags_and_zeroes_base_vertex() {
        use crate::gfx::render_types::{MaterialUniforms, SkinnedDrawObject};
        let material = MaterialUniforms {
            tint: [0.2, 0.4, 0.6],
            roughness: 0.3,
            metallic: 0.5,
            emissive: [0.1, 0.0, 0.0],
            ..MaterialUniforms::DEFAULT
        };
        let obj = SkinnedDrawObject {
            vertex_base: 7,
            vertex_count: 100,
            index_offset: 42,
            index_count: 300,
            model: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [3.0, 4.0, 5.0, 1.0],
            ],
            texture_slot: 9,
            normal_map_slot: 3,
            material,
            visible: true,
            joint_count: 12,
            local_bb_min: [-1.0, -1.0, -1.0],
            local_bb_max: [1.0, 1.0, 1.0],
            lod_alternates: Vec::new(),
        };
        let albedo_count = 6usize;
        let last_tex = 11usize;
        let last_nm = 4usize;
        let e = skinned_geom_entry(&obj, albedo_count, last_tex, last_nm);
        // The skinned BLAS bakes absolute indices, so base_vertex is folded to 0.
        assert_eq!(e.base_vertex, 0);
        // The skinned flag is set; masking it off recovers the real bindless pool
        // index, computed the same way as a static draw (so skinned hits texture).
        let (exp_albedo, exp_normal) = pool_indices(
            obj.texture_slot,
            obj.normal_map_slot,
            albedo_count,
            last_tex,
            last_nm,
        );
        assert_ne!(e.normal_index & RT_SKINNED_FLAG, 0);
        assert_eq!(e.albedo_index, exp_albedo);
        assert_eq!(e.normal_index & !RT_SKINNED_FLAG, exp_normal);
        // Material + index offset carry through; the model lifts the hit to world.
        assert_eq!(e.index_offset, 42);
        assert_eq!(e.tint, [0.2, 0.4, 0.6]);
        assert_eq!(e.model[3], [3.0, 4.0, 5.0, 1.0]);
    }
}
