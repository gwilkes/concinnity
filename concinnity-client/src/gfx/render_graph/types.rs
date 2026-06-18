// src/gfx/render_graph/types.rs
//
// Shared, backend-agnostic types for the render graph: resource handles,
// resource descriptions, state / access enums, and the small structs the
// compile pass emits. The graph tracks *order*, *barriers*, and
// *lifetimes*: it does not allocate transient GPU resources (those stay
// backend-owned).

use std::num::NonZeroU32;

// One side of a `PassBuilder::read_*` / `write_*` declaration. The
// resource is a small dense index into the graph's resource arena; the
// `version` increments on every write so a read-after-write chain
// (`main → decals → fog` writing the same hdr_resolve) is an unambiguous
// DAG.
//
// Each `TextureHandle` / `BufferHandle` pairs the resource id with the
// version it refers to. `write_*` returns a new handle pointing at the
// post-write version; the old handle stays valid (and refers to the
// pre-write version) so a pass can still legally read the prior content
// if it wants.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TextureHandle {
    pub(super) resource: ResourceId,
    pub(super) version: u32,
}

impl TextureHandle {
    // Sentinel for "no texture", used by the per-frame graph builder
    // for conditional passes (SSR off, TAA off, ...) so the call sites
    // stay branchless. The compile pass treats reads / writes of an
    // invalid handle as no-ops.
    pub const INVALID: Self = Self {
        resource: ResourceId::INVALID,
        version: 0,
    };

    // `true` when this handle was produced by a valid `create_*` /
    // `import_*` call; `false` when it's the `INVALID` sentinel.
    pub fn is_valid(self) -> bool {
        self.resource.is_valid()
    }
}

// Buffer counterpart to [`TextureHandle`]. Same handle / version model.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct BufferHandle {
    pub(super) resource: ResourceId,
    pub(super) version: u32,
}

impl BufferHandle {
    pub const INVALID: Self = Self {
        resource: ResourceId::INVALID,
        version: 0,
    };

    pub fn is_valid(self) -> bool {
        self.resource.is_valid()
    }
}

// Dense resource identifier. `u32::MAX` reserved as the "invalid"
// sentinel; everything else is a valid index into the compiled graph's
// `resources` Vec. The executor uses `index()` to look up a resource's
// realised GPU object.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ResourceId(pub(super) u32);

impl ResourceId {
    pub const INVALID: Self = Self(u32::MAX);

    pub fn is_valid(self) -> bool {
        self.0 != u32::MAX
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

// What kind of work an executor encodes for a pass: render vs compute.
// The graph cares about this only enough to pick the right
// `MTLRenderPassDescriptor` / `MTLComputePassDescriptor` analogue per
// backend; the actual encoding stays in the per-backend `encode_*`
// methods. Blit passes are not yet in scope (today's engine has none).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PassKind {
    Render,
    Compute,
}

// Whether a resource is engine-owned (the graph references it) or
// declared inside the graph (the graph tracks its lifetime; the graph
// does not yet own its allocation).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResourceOrigin {
    // Engine owns the GPU object; the graph just references it by
    // handle. Most of today's `MtlContext` targets enter the graph via
    // `import_texture` / `import_buffer`.
    Imported,
    // Graph-tracked resource declared via `create_texture` /
    // `create_buffer`. The backend asserts it has a target of matching
    // shape; the graph does not yet allocate from a pool with aliasing.
    Transient,
}

// Coarse per-resource state used by the barrier deriver. The
// executor maps each to the backend's concrete state: for Vulkan a
// `VkImageLayout` + `VkAccessFlags` pair, for DirectX a
// `D3D12_RESOURCE_STATES`, for Metal mostly a no-op except `useResource`
// on the ICB-driven cull pass.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResourceState {
    // Initial state before any pass uses the resource. Reading from
    // `Undefined` is a compile error (a read with no producer); writing
    // to it is always legal and transitions the state to a writer
    // variant below.
    Undefined,
    // A pass reads the resource (sampled texture, uniform / SSBO,
    // indirect-args buffer, depth read, ...). Multiple consecutive
    // reads are coalesced: they don't insert barriers between
    // themselves.
    Read,
    // A pass writes the resource (render target, depth-stencil target,
    // storage write, blend write). A second write after a read inserts
    // a Write→Read→Write barrier chain; consecutive writes by the same
    // pass do not, but consecutive writes across passes do.
    Write,
}

// How a graph resource a backend drives from `barriers_before` is used, so
// the backend can translate the coarse `ResourceState` into a concrete native
// state: the same `Write` means a colour render target for one resource and a
// depth-stencil target for another, which map to different
// `D3D12_RESOURCE_STATES` / `vk::ImageLayout`s. The backend resolver assigns a
// class to each migrated resource; the backend's barrier translator maps
// `(class, state)` to its native state. Extend as resources of new kinds
// (storage / compute targets, ...) migrate.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GraphResourceClass {
    // Sampled colour render target (e.g. the SSAO occlusion `ao_output`).
    ColorTarget,
    // Sampled depth-stencil render target (e.g. the CSM `shadow_map`).
    DepthTarget,
    // Compute-written, shader-sampled storage image (e.g. the volumetric-fog
    // `fog_froxel_volume`): a compute pass writes it (DirectX UNORDERED_ACCESS /
    // Vulkan GENERAL) and a later fragment pass samples it. Unlike the two
    // target classes its `Write` happens in the compute stage, so the translated
    // state pairs a storage-write layout with the compute pipeline stage.
    StorageImage,
}

// Which shader stage(s) read a graph resource across a contiguous read-run
// (the passes that read one resource version before the next writer). Carried
// on a barrier whose Read side spans this run so a backend can satisfy it in a
// single transition: a write made visible to both a compute consumer and a
// fragment consumer needs one barrier covering both stages, not a per-consumer
// read-to-read barrier (which would not carry the producing write). Derived
// from each reading pass's `PassKind` (a render pass samples in the fragment
// stage, a compute pass in the compute stage); empty on a barrier with no Read
// side (a write-only producer transition). Add bits as passes read in stages
// the two current ones do not model.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ReadStages(u32);

impl ReadStages {
    // A render pass's sampled read (DirectX PIXEL_SHADER_RESOURCE / Vulkan
    // FRAGMENT_SHADER stage).
    pub const FRAGMENT: Self = Self(1 << 0);
    // A compute pass's read (DirectX NON_PIXEL_SHADER_RESOURCE / Vulkan
    // COMPUTE_SHADER stage).
    pub const COMPUTE: Self = Self(1 << 1);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    // The stage a pass of `kind` reads a resource in: render passes sample in
    // the fragment stage, compute passes in the compute stage. This is the one
    // place a `PassKind` becomes a read stage, so the approximation lives here:
    // a render pass that sampled in the vertex / geometry stage would be
    // labelled FRAGMENT. No graph-driven resource is read that way today; if
    // one ever is, carry an explicit per-read stage instead of deriving it.
    pub const fn for_pass_kind(kind: PassKind) -> Self {
        match kind {
            PassKind::Render => Self::FRAGMENT,
            PassKind::Compute => Self::COMPUTE,
        }
    }
}

impl std::ops::BitOr for ReadStages {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

// One barrier the executor must insert before a pass runs. Per-backend
// interpretation: Vulkan emits `vkCmdPipelineBarrier`; DirectX emits
// `D3D12_RESOURCE_BARRIER`; Metal mostly ignores them (implicit hazard
// tracking) but may translate `from: Write, to: Read` on the cull ICB
// path into an explicit `useResource(.Write)` declaration.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BarrierOp {
    pub(super) resource: ResourceId,
    pub(super) from: ResourceState,
    pub(super) to: ResourceState,
    // Stage union of this barrier's Read side (see `ReadStages`): the consuming
    // run's stages for a `* -> Read` transition, the prior run's stages for a
    // `Read -> Write` (WAR), empty when neither side is Read. The backend
    // translator targets this union so one transition covers every consuming
    // stage.
    pub(super) read_stages: ReadStages,
}

impl BarrierOp {
    // Pass-local accessors so callers don't have to import `ResourceId`.
    // Returns the resource's stable index, the same value the executor
    // uses to look the resource up in `CompiledGraph.resources`.
    pub fn resource_index(self) -> usize {
        self.resource.index()
    }
    // Accessor for the transition's source state, paired with `to_state`; not a
    // constructor despite the `from_` prefix.
    #[allow(clippy::wrong_self_convention)]
    pub fn from_state(self) -> ResourceState {
        self.from
    }
    pub fn to_state(self) -> ResourceState {
        self.to
    }
    // Stage union of this barrier's Read side: the consuming run's stages for a
    // `* -> Read` transition (the backend must make the producing write visible
    // to all of them), the prior run's stages for a `Read -> Write` (WAR).
    // Empty when neither side is Read.
    pub fn read_stages(self) -> ReadStages {
        self.read_stages
    }
}

// Inclusive `[first, last]` range over pass indices in the compiled
// graph's `passes` Vec. Used to describe a transient resource's
// lifetime so an aliaser can overlap non-overlapping lifetimes.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PassRange {
    pub first: usize,
    pub last: usize,
}

// Texture-shape description carried by both imported and transient
// resources. Imported resources just declare it for documentation +
// the aliaser's compatibility check; transients use it to drive
// allocation later.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct TextureDesc {
    pub width: TextureSize,
    pub height: TextureSize,
    pub format: PixelFormat,
    // MSAA sample count, 1 for non-multisample. The graph doesn't care
    // what value this is; the backend executor maps it to its API's
    // sample-count enum.
    pub sample_count: u32,
    // Number of array layers. 1 for plain 2D, 6 for cube, N for CSM
    // shadow-map arrays.
    pub array_layers: u32,
    pub usage: TextureUsage,
}

// Buffer-shape description. Size is optional because some
// imported buffers grow dynamically per-frame (the GPU object data
// buffer, the per-emitter spawn ring, ...): the graph then just
// tracks the dependency, not the size.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BufferDesc {
    pub size_bytes: Option<NonZeroU32>,
    pub usage: BufferUsage,
}

// How a texture is sized. Two non-absolute variants let bloom mips and
// full-resolution targets express their size without the graph needing
// to know the swapchain dimensions at declaration time.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TextureSize {
    // Fixed pixel count. CSM shadow-map slices use this
    // (`Absolute(2048)`); the rest of the engine's targets follow the
    // drawable.
    Absolute(u32),
    // Tracks the swapchain drawable's width or height.
    Drawable,
    // Scaled fraction of the drawable, floored to >= 1 by the executor.
    // Bloom mips chain through this (`DrawableScaled(0.5)^n`).
    DrawableScaled(f32),
}

// Backend-agnostic pixel format. Maps to `MTLPixelFormat` /
// `vk::Format` / `DXGI_FORMAT` per executor. Only the formats the
// engine actually uses are enumerated; extend as new passes need new
// targets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba16Float,
    Rgba8Unorm,
    Rg16Float,
    R8Unorm,
    R32Float,
    Depth32Float,
    BgraSwapchain,
}

impl PixelFormat {
    // Bytes per texel (per sample). Used by the aliasing planner to size a
    // resource's memory footprint. The engine uses only single-plane,
    // power-of-two formats, so this is one byte count per variant.
    pub const fn bytes_per_texel(self) -> u32 {
        match self {
            PixelFormat::Rgba16Float => 8,
            PixelFormat::Rgba8Unorm
            | PixelFormat::Rg16Float
            | PixelFormat::R32Float
            | PixelFormat::Depth32Float
            | PixelFormat::BgraSwapchain => 4,
            PixelFormat::R8Unorm => 1,
        }
    }

    // Whether this is a depth format. The aliasing planner keeps depth and
    // colour resources in separate memory pools because their backend memory
    // requirements (heap flags / memory type) differ; the finer per-usage
    // compatibility is the backend's concern when it realises the plan.
    pub const fn is_depth(self) -> bool {
        matches!(self, PixelFormat::Depth32Float)
    }
}

impl TextureSize {
    // Resolve to a concrete pixel count against the current drawable extent.
    // `DrawableScaled` floors to >= 1 so a mip-scaled target never degenerates
    // to zero.
    pub fn resolve(self, drawable: u32) -> u32 {
        match self {
            TextureSize::Absolute(n) => n.max(1),
            TextureSize::Drawable => drawable.max(1),
            TextureSize::DrawableScaled(f) => ((drawable as f32 * f).floor() as u32).max(1),
        }
    }
}

impl TextureDesc {
    // The resource's memory footprint in bytes at the given drawable extent:
    // width * height * bytes-per-texel * sample_count * array_layers (single
    // mip; the engine's graph targets are all single-level). The aliasing
    // planner sums and packs these.
    pub fn byte_size(&self, drawable_w: u32, drawable_h: u32) -> u64 {
        let w = self.width.resolve(drawable_w) as u64;
        let h = self.height.resolve(drawable_h) as u64;
        w * h
            * self.format.bytes_per_texel() as u64
            * self.sample_count.max(1) as u64
            * self.array_layers.max(1) as u64
    }
}

// Bitset describing how a texture can be used. The graph doesn't
// enforce these against declared reads / writes (executors do);
// the field exists so the aliaser can match transient resources to
// pool entries with the right usage flags.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextureUsage(pub u32);

impl TextureUsage {
    pub const SHADER_READ: Self = Self(1 << 0);
    pub const RENDER_TARGET: Self = Self(1 << 1);
    pub const DEPTH_STENCIL: Self = Self(1 << 2);
    pub const STORAGE: Self = Self(1 << 3);
    pub const TRANSFER_SRC: Self = Self(1 << 4);
    pub const TRANSFER_DST: Self = Self(1 << 5);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for TextureUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

// Buffer-side counterpart to [`TextureUsage`]. Same bitset shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BufferUsage(pub u32);

impl BufferUsage {
    pub const UNIFORM: Self = Self(1 << 0);
    pub const STORAGE: Self = Self(1 << 1);
    pub const INDEX: Self = Self(1 << 2);
    pub const VERTEX: Self = Self(1 << 3);
    pub const INDIRECT: Self = Self(1 << 4);
    pub const TRANSFER_SRC: Self = Self(1 << 5);
    pub const TRANSFER_DST: Self = Self(1 << 6);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_handle_is_invalid() {
        assert!(!TextureHandle::INVALID.is_valid());
        assert!(!BufferHandle::INVALID.is_valid());
    }

    #[test]
    fn texture_usage_bitset_round_trips() {
        let u = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        assert!(u.contains(TextureUsage::SHADER_READ));
        assert!(u.contains(TextureUsage::RENDER_TARGET));
        assert!(!u.contains(TextureUsage::STORAGE));
        assert_eq!(
            u.union(TextureUsage::STORAGE).0,
            TextureUsage::SHADER_READ.0 | TextureUsage::RENDER_TARGET.0 | TextureUsage::STORAGE.0
        );
    }

    #[test]
    fn pass_range_is_inclusive() {
        let r = PassRange { first: 2, last: 5 };
        assert_eq!(r.first, 2);
        assert_eq!(r.last, 5);
    }
}
