// src/gfx/render_graph/builder.rs
//
// Builder API for the render graph. A `GraphBuilder` collects resource
// declarations and pass declarations; once every pass has been added,
// the caller hands the builder to `compile()` (see [`super::compile`])
// to produce a frozen, topologically-sorted `CompiledGraph`.
//
// All state mutation happens here; the compile pass is a pure read.

use super::passes::PassId;
use super::types::{
    BufferDesc, BufferHandle, PassKind, ResourceId, ResourceOrigin, TextureDesc, TextureHandle,
};

// One entry in the builder's resource arena. Texture and buffer kinds
// live in the same Vec so a `ResourceId` is unique across both, which
// keeps the executor's lookup table flat. The `version` counter tracks
// how many `write_*` calls have produced a new version of this
// resource; reading the latest version means reading the resource as
// of `version - 1` (the last write index).
#[derive(Debug, Clone)]
pub(super) enum ResourceDecl {
    Texture {
        label: &'static str,
        desc: TextureDesc,
        origin: ResourceOrigin,
        // `0` until the first write; increments per `write_texture`.
        version: u32,
    },
    Buffer {
        label: &'static str,
        desc: BufferDesc,
        origin: ResourceOrigin,
        version: u32,
    },
}

impl ResourceDecl {
    pub(super) fn label(&self) -> &'static str {
        match self {
            ResourceDecl::Texture { label, .. } | ResourceDecl::Buffer { label, .. } => label,
        }
    }

    pub(super) fn origin(&self) -> ResourceOrigin {
        match self {
            ResourceDecl::Texture { origin, .. } | ResourceDecl::Buffer { origin, .. } => *origin,
        }
    }

    pub(super) fn current_version(&self) -> u32 {
        match self {
            ResourceDecl::Texture { version, .. } | ResourceDecl::Buffer { version, .. } => {
                *version
            }
        }
    }

    pub(super) fn bump_version(&mut self) -> u32 {
        match self {
            ResourceDecl::Texture { version, .. } | ResourceDecl::Buffer { version, .. } => {
                *version += 1;
                *version
            }
        }
    }

    // `true` for `ResourceDecl::Texture`. Used by the compile pass to
    // route declared reads / writes through the right state machine
    // (textures vs buffers differ only by label / executor lookup).
    pub(super) fn is_texture(&self) -> bool {
        matches!(self, ResourceDecl::Texture { .. })
    }

    // The texture description (format / size / sample count), or `None` for a
    // buffer. The compile pass carries it onto `CompiledResource` so the
    // aliasing planner can size each transient resource.
    pub(super) fn texture_desc(&self) -> Option<TextureDesc> {
        match self {
            ResourceDecl::Texture { desc, .. } => Some(*desc),
            ResourceDecl::Buffer { .. } => None,
        }
    }
}

// One side of a pass's read or write declaration. Bundles the resource
// id with the *version of that resource the pass touches*: for a write
// that's the post-write version (== `version` after `bump_version`);
// for a read it's the version current when the read was declared.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ResourceVersion {
    pub(super) resource: ResourceId,
    pub(super) version: u32,
}

impl ResourceVersion {
    // Stable resource index, the same value the executor uses to look up
    // the GPU object in its handle→resource map.
    pub fn resource_index(self) -> usize {
        self.resource.index()
    }
    pub fn version(self) -> u32 {
        self.version
    }
}

// One pass declaration. Reads and writes are kept in declaration order
// for stable executor dispatch. `presents` marks the terminal pass; the
// compile pass validates exactly one pass per graph has it set.
#[derive(Debug, Clone)]
pub(super) struct PassDecl {
    pub(super) id: PassId,
    pub(super) kind: PassKind,
    pub(super) reads: Vec<ResourceVersion>,
    pub(super) writes: Vec<ResourceVersion>,
    pub(super) presents: bool,
}

// Collects resource and pass declarations to feed `compile()`. Held by
// value, never shared; each frame's graph builds fresh.
pub struct GraphBuilder {
    pub(super) resources: Vec<ResourceDecl>,
    pub(super) passes: Vec<PassDecl>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            resources: Vec::with_capacity(32),
            passes: Vec::with_capacity(super::passes::PASS_COUNT),
        }
    }

    // Declare an engine-owned texture the graph references but does
    // not allocate. Returns a fresh handle at version 0.
    pub fn import_texture(&mut self, label: &'static str, desc: TextureDesc) -> TextureHandle {
        self.push_texture(label, desc, ResourceOrigin::Imported)
    }

    // Declare an engine-owned buffer the graph references but does
    // not allocate. Returns a fresh handle at version 0.
    pub fn import_buffer(&mut self, label: &'static str, desc: BufferDesc) -> BufferHandle {
        self.push_buffer(label, desc, ResourceOrigin::Imported)
    }

    // Declare a graph-tracked transient texture. The graph just tracks
    // lifetime; it does not yet allocate from a pool.
    pub fn create_texture(&mut self, label: &'static str, desc: TextureDesc) -> TextureHandle {
        self.push_texture(label, desc, ResourceOrigin::Transient)
    }

    // Buffer counterpart to `create_texture`.
    pub fn create_buffer(&mut self, label: &'static str, desc: BufferDesc) -> BufferHandle {
        self.push_buffer(label, desc, ResourceOrigin::Transient)
    }

    // Add a pass to the graph. Returns a `PassBuilder` that the caller
    // uses to declare what the pass reads / writes. Passes are
    // dispatched in the order returned by `compile`'s topological
    // sort, not in the order they were added here; the order this
    // matters for is purely "tie-breaking when toposort has flexibility".
    pub fn add_pass(&mut self, id: PassId, kind: PassKind) -> PassBuilder<'_> {
        let pass_idx = self.passes.len();
        self.passes.push(PassDecl {
            id,
            kind,
            reads: Vec::with_capacity(4),
            writes: Vec::with_capacity(2),
            presents: false,
        });
        PassBuilder {
            builder: self,
            pass_idx,
        }
    }

    fn push_texture(
        &mut self,
        label: &'static str,
        desc: TextureDesc,
        origin: ResourceOrigin,
    ) -> TextureHandle {
        let resource = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceDecl::Texture {
            label,
            desc,
            origin,
            version: 0,
        });
        TextureHandle {
            resource,
            version: 0,
        }
    }

    fn push_buffer(
        &mut self,
        label: &'static str,
        desc: BufferDesc,
        origin: ResourceOrigin,
    ) -> BufferHandle {
        let resource = ResourceId(self.resources.len() as u32);
        self.resources.push(ResourceDecl::Buffer {
            label,
            desc,
            origin,
            version: 0,
        });
        BufferHandle {
            resource,
            version: 0,
        }
    }
}

// Pass-level fluent builder returned by `GraphBuilder::add_pass`. Drops
// without doing anything; the pass declaration lives in
// `GraphBuilder.passes` from the moment `add_pass` returns; this struct
// only mediates safe `&mut` access to it.
pub struct PassBuilder<'g> {
    builder: &'g mut GraphBuilder,
    pass_idx: usize,
}

impl PassBuilder<'_> {
    // Declare that this pass reads `h`. Silently no-ops on
    // `TextureHandle::INVALID` so conditional graph builds stay
    // branch-free. Returns `&mut Self` for chaining.
    pub fn read_texture(&mut self, h: TextureHandle) -> &mut Self {
        if h.is_valid() {
            self.builder.passes[self.pass_idx]
                .reads
                .push(ResourceVersion {
                    resource: h.resource,
                    version: h.version,
                });
        }
        self
    }

    // Declare that this pass reads `h`. Same INVALID-safety as
    // `read_texture`.
    pub fn read_buffer(&mut self, h: BufferHandle) -> &mut Self {
        if h.is_valid() {
            self.builder.passes[self.pass_idx]
                .reads
                .push(ResourceVersion {
                    resource: h.resource,
                    version: h.version,
                });
        }
        self
    }

    // Declare that this pass writes `h`. Returns a new handle pointing
    // at the post-write version; the input handle stays valid as a
    // reference to the pre-write content.
    //
    // Returns `TextureHandle::INVALID` when called with the invalid
    // sentinel so optional passes can chain freely.
    pub fn write_texture(&mut self, h: TextureHandle) -> TextureHandle {
        if !h.is_valid() {
            return TextureHandle::INVALID;
        }
        let new_version = self.builder.resources[h.resource.index()].bump_version();
        self.builder.passes[self.pass_idx]
            .writes
            .push(ResourceVersion {
                resource: h.resource,
                version: new_version,
            });
        TextureHandle {
            resource: h.resource,
            version: new_version,
        }
    }

    // Buffer counterpart to `write_texture`.
    pub fn write_buffer(&mut self, h: BufferHandle) -> BufferHandle {
        if !h.is_valid() {
            return BufferHandle::INVALID;
        }
        let new_version = self.builder.resources[h.resource.index()].bump_version();
        self.builder.passes[self.pass_idx]
            .writes
            .push(ResourceVersion {
                resource: h.resource,
                version: new_version,
            });
        BufferHandle {
            resource: h.resource,
            version: new_version,
        }
    }

    // Mark this pass as the one that writes the final swapchain image.
    // The compile pass requires exactly one of these per graph; failing
    // that produces a `GraphError::MissingPresenter` or
    // `GraphError::MultiplePresenters`.
    pub fn presents(&mut self) -> &mut Self {
        self.builder.passes[self.pass_idx].presents = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{BufferUsage, PixelFormat, TextureSize, TextureUsage};
    use super::*;

    fn dummy_tex_desc() -> TextureDesc {
        TextureDesc {
            width: TextureSize::Drawable,
            height: TextureSize::Drawable,
            format: PixelFormat::Rgba16Float,
            sample_count: 1,
            array_layers: 1,
            usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        }
    }

    fn dummy_buf_desc() -> BufferDesc {
        BufferDesc {
            size_bytes: None,
            usage: BufferUsage::STORAGE,
        }
    }

    #[test]
    fn create_texture_assigns_dense_ids() {
        let mut b = GraphBuilder::new();
        let a = b.import_texture("a", dummy_tex_desc());
        let c = b.create_texture("c", dummy_tex_desc());
        assert_eq!(a.resource.index(), 0);
        assert_eq!(c.resource.index(), 1);
        assert_eq!(a.version, 0);
        assert_eq!(c.version, 0);
    }

    #[test]
    fn write_bumps_version_and_returns_new_handle() {
        let mut b = GraphBuilder::new();
        let h0 = b.create_texture("t", dummy_tex_desc());
        let h1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.write_texture(h0)
        };
        assert_eq!(h0.version, 0);
        assert_eq!(h1.version, 1);
        assert_eq!(h0.resource, h1.resource);
    }

    #[test]
    fn read_write_record_declarations() {
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", dummy_tex_desc());
        let buf = b.create_buffer("b", dummy_buf_desc());
        {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(t).read_buffer(buf);
            let _ = p.write_texture(t);
        }

        let pass = &b.passes[0];
        assert_eq!(pass.reads.len(), 2);
        assert_eq!(pass.writes.len(), 1);
        assert_eq!(pass.reads[0].resource, t.resource);
        assert_eq!(pass.reads[1].resource, buf.resource);
        assert_eq!(pass.writes[0].version, 1);
    }

    #[test]
    fn invalid_handle_skips_declaration() {
        let mut b = GraphBuilder::new();
        let out = {
            let mut p = b.add_pass(PassId::Composite, PassKind::Render);
            p.read_texture(TextureHandle::INVALID);
            p.write_texture(TextureHandle::INVALID)
        };

        assert!(!out.is_valid());
        let pass = &b.passes[0];
        assert!(pass.reads.is_empty());
        assert!(pass.writes.is_empty());
    }

    #[test]
    fn presents_marks_the_pass() {
        let mut b = GraphBuilder::new();
        b.add_pass(PassId::Composite, PassKind::Render).presents();
        assert!(b.passes[0].presents);
    }
}
