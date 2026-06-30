// src/gfx/render_graph/alias.rs
//
// Transient-resource memory aliasing planner. The compile pass already
// computes each resource's `[first, last]` lifetime over the sorted pass list
// (`CompiledResource.lifetime`); this module turns those intervals into a
// physical-memory plan: transient resources whose lifetimes do not overlap can
// share one backing allocation, since they are never live at the same time.
//
// The planner is backend-agnostic and pure. It only decides *which resources
// share a slot* and *how big each slot must be*; the per-backend executor
// realises the plan (allocates a pool, creates the aliased resources, binds
// them, and inserts aliasing barriers at the slot's reuse boundaries). This
// mirrors how the graph plans barriers (`barriers_before`) while each backend
// emits them.
//
// Only `Transient` textures are candidates: `Imported` resources are
// engine-owned and outlive the frame (cross-frame TAA history, the resting
// shadow map, the `scene_pre_taa` alias), so the graph never reuses their
// memory. Buffers are not aliased yet (none of today's graph buffers are large
// or short-lived enough to matter).
//
// The packing is a linear scan over lifetime-start order (the classic
// interval-graph greedy, optimal for the slot *count* on an interval graph):
// each resource takes the first compatible slot whose last occupant's lifetime
// ended strictly before this resource's begins, else opens a new slot. A slot
// is sized to its largest member. Compatibility is coarse for now (depth vs
// colour, which separates the two backend memory classes); the backend refines
// it against concrete usage flags when it allocates.

use super::compile::CompiledGraph;
use super::types::ResourceOrigin;

// One physical memory slot shared by one or more transient resources with
// pairwise-disjoint lifetimes. `byte_size` is the max footprint of its members
// (the allocation the backend must make); `members` are resource indices into
// `CompiledGraph.resources`, in assignment order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasSlot {
    pub byte_size: u64,
    pub members: Vec<usize>,
}

// The computed aliasing plan for one compiled graph at one drawable extent.
#[derive(Debug, Clone)]
pub struct AliasPlan {
    // Physical slots; the backend allocates one pool entry per slot.
    pub slots: Vec<AliasSlot>,
    // Per-resource slot index, indexed by `ResourceId`. `None` for every
    // resource the planner does not place (imported, buffer, or a transient
    // with no texture desc).
    pub assignment: Vec<Option<usize>>,
    // Total bytes the slots occupy (the aliased footprint).
    pub aliased_bytes: u64,
    // Total bytes the same resources would occupy with no aliasing (one
    // allocation each).
    pub unaliased_bytes: u64,
}

impl AliasPlan {
    // Bytes saved by aliasing: the unaliased footprint minus the slot
    // footprint. Zero when no two transients have disjoint lifetimes.
    pub fn saved_bytes(&self) -> u64 {
        self.unaliased_bytes.saturating_sub(self.aliased_bytes)
    }
}

// Compute the aliasing plan for `graph` at the given drawable extent. Pure: the
// same graph + extent always produces the same plan. See the module header for
// the packing strategy.
pub fn plan_aliasing(graph: &CompiledGraph, drawable_w: u32, drawable_h: u32) -> AliasPlan {
    // Gather the transient texture candidates with their lifetime + size.
    struct Cand {
        idx: usize,
        first: usize,
        last: usize,
        size: u64,
        depth: bool,
    }
    let mut cands: Vec<Cand> = Vec::new();
    for (idx, res) in graph.resources.iter().enumerate() {
        if res.origin != ResourceOrigin::Transient {
            continue;
        }
        let Some(desc) = res.tex_desc else {
            continue;
        };
        cands.push(Cand {
            idx,
            first: res.lifetime.first,
            last: res.lifetime.last,
            size: desc.byte_size(drawable_w, drawable_h),
            depth: desc.format.is_depth(),
        });
    }

    let unaliased_bytes: u64 = cands.iter().map(|c| c.size).sum();

    // Process in lifetime-start order (ties by resource index for determinism),
    // so a slot's running `free_at` (max last over its members) is enough to
    // test disjointness against the next candidate.
    cands.sort_by(|a, b| a.first.cmp(&b.first).then(a.idx.cmp(&b.idx)));

    // Slot bookkeeping kept alongside the public `AliasSlot` so we can track the
    // running free-time without recomputing it.
    struct SlotMeta {
        depth: bool,
        free_at: usize,
        byte_size: u64,
        members: Vec<usize>,
    }
    let mut slots: Vec<SlotMeta> = Vec::new();
    let mut assignment: Vec<Option<usize>> = vec![None; graph.resources.len()];

    for c in &cands {
        // First compatible slot whose last occupant ended strictly before this
        // resource begins. `free_at < c.first` (strict) because an inclusive
        // `[..=free_at]` and `[c.first..=..]` that touch at `free_at == c.first`
        // are both live on that pass and must not share memory.
        let chosen = slots
            .iter()
            .position(|s| s.depth == c.depth && s.free_at < c.first);
        let si = match chosen {
            Some(si) => {
                let s = &mut slots[si];
                s.free_at = c.last;
                s.byte_size = s.byte_size.max(c.size);
                s.members.push(c.idx);
                si
            }
            None => {
                slots.push(SlotMeta {
                    depth: c.depth,
                    free_at: c.last,
                    byte_size: c.size,
                    members: vec![c.idx],
                });
                slots.len() - 1
            }
        };
        assignment[c.idx] = Some(si);
    }

    let aliased_bytes: u64 = slots.iter().map(|s| s.byte_size).sum();
    let slots: Vec<AliasSlot> = slots
        .into_iter()
        .map(|s| AliasSlot {
            byte_size: s.byte_size,
            members: s.members,
        })
        .collect();

    AliasPlan {
        slots,
        assignment,
        aliased_bytes,
        unaliased_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx::render_graph::builder::GraphBuilder;
    use crate::gfx::render_graph::frame::{FrameGraphInputs, build_frame_graph};
    use crate::gfx::render_graph::passes::PassId;
    use crate::gfx::render_graph::types::{
        PassKind, PixelFormat, TextureDesc, TextureSize, TextureUsage,
    };

    // Mirror of `frame::tests::all_off` (that helper is private to the frame
    // test module). All gated passes off; used by the integration test below.
    const ALL_OFF: FrameGraphInputs = FrameGraphInputs {
        shadow_enabled: false,
        shadow_map_size: 2048,
        hdr_width: 1920,
        hdr_height: 1080,
        hdr_sample_count: 1,
        bindless_cull_enabled: false,
        auto_exposure_enabled: false,
        bloom_enabled: false,
        velocity_enabled: false,
        taa_enabled: false,
        ssr_enabled: false,
        particles_enabled: false,
        fog_enabled: false,
        decals_enabled: false,
        ssr_prepass_enabled: false,
        ssao_enabled: false,
        upscale_enabled: false,
        transparent_enabled: false,
        raymarch_enabled: false,
        two_pass_occlusion_enabled: false,
        ssgi_enabled: false,
        rt_reflections_enabled: false,
        unified_gbuffer_prepass: false,
        world_hidden: false,
    };

    // A transient texture desc of the given format at full drawable size.
    fn tex(format: PixelFormat) -> TextureDesc {
        TextureDesc {
            width: TextureSize::Drawable,
            height: TextureSize::Drawable,
            format,
            sample_count: 1,
            array_layers: 1,
            usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        }
    }

    // Bytes a full-drawable texture of `format` occupies at 100x100.
    fn size_at_100(format: PixelFormat) -> u64 {
        100 * 100 * format.bytes_per_texel() as u64
    }

    #[test]
    fn byte_size_resolves_drawable_and_format() {
        let d = tex(PixelFormat::Rgba16Float);
        assert_eq!(d.byte_size(64, 32), 64 * 32 * 8);
        // Half-res quarter-byte format.
        let half_r8 = TextureDesc {
            width: TextureSize::DrawableScaled(0.5),
            height: TextureSize::DrawableScaled(0.5),
            format: PixelFormat::R8Unorm,
            sample_count: 1,
            array_layers: 1,
            usage: TextureUsage::SHADER_READ,
        };
        assert_eq!(half_r8.byte_size(64, 64), 32 * 32);
        // Sample count + array layers multiply.
        let msaa = TextureDesc {
            sample_count: 4,
            array_layers: 2,
            ..tex(PixelFormat::Rgba8Unorm)
        };
        assert_eq!(msaa.byte_size(10, 10), 10 * 10 * 4 * 4 * 2);
    }

    #[test]
    fn disjoint_transients_share_one_slot_sized_to_largest() {
        // Main writes `a` (R8, small), read by SsaoBlur; Fog writes `b` (RGBA16,
        // big), read by Composite. `a` lifetime [0,1], `b` [2,3] -> disjoint, so
        // they pack into one slot sized to the larger (`b`).
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::R8Unorm));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        g.add_pass(PassId::SsaoBlur, PassKind::Render)
            .read_texture(a1);
        let b1 = g.add_pass(PassId::Fog, PassKind::Render).write_texture(b);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 1, "disjoint a + b share one slot");
        assert_eq!(
            plan.slots[0].byte_size,
            size_at_100(PixelFormat::Rgba16Float)
        );
        assert_eq!(plan.slots[0].members.len(), 2);
        // Both resources point at slot 0.
        assert_eq!(plan.assignment[a.resource.index()], Some(0));
        assert_eq!(plan.assignment[b.resource.index()], Some(0));
        // Saved = the smaller resource's footprint (it reuses b's slot).
        assert_eq!(plan.saved_bytes(), size_at_100(PixelFormat::R8Unorm));
        assert_eq!(
            plan.unaliased_bytes,
            size_at_100(PixelFormat::R8Unorm) + size_at_100(PixelFormat::Rgba16Float)
        );
    }

    #[test]
    fn overlapping_transients_get_separate_slots() {
        // Main writes both `a` and `b`; Composite reads both. Their lifetimes
        // both span [0,1], so they overlap and cannot share memory.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let (a1, b1) = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            (p.write_texture(a), p.write_texture(b))
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(a1)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 2, "overlapping a + b need two slots");
        assert_eq!(plan.saved_bytes(), 0);
    }

    #[test]
    fn touching_lifetimes_do_not_alias() {
        // `a` lifetime [0,1], `b` [1,2]: they share pass 1 (a read there, b
        // written there), so the strict `free_at < first` rule keeps them apart.
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        let b1 = {
            let mut p = g.add_pass(PassId::Decals, PassKind::Render);
            p.read_texture(a1);
            p.write_texture(b)
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(b1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 2, "touching lifetimes overlap at pass 1");
    }

    #[test]
    fn depth_and_colour_do_not_share() {
        // Two disjoint transients, one depth one colour. Even though their
        // lifetimes don't overlap, the planner keeps them in separate pools
        // (different backend memory class).
        let mut g = GraphBuilder::new();
        let depth = g.create_texture("depth", tex(PixelFormat::Depth32Float));
        let colour = g.create_texture("colour", tex(PixelFormat::Rgba16Float));
        let d1 = g
            .add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(depth);
        g.add_pass(PassId::SsaoBlur, PassKind::Render)
            .read_texture(d1);
        let c1 = g
            .add_pass(PassId::Fog, PassKind::Render)
            .write_texture(colour);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(c1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 2, "depth + colour never share a slot");
        assert_eq!(plan.saved_bytes(), 0);
    }

    #[test]
    fn imported_resources_are_not_placed() {
        // An imported texture is engine-owned and outlives the frame; the
        // planner never assigns it a slot even if its lifetime would fit one.
        let mut g = GraphBuilder::new();
        let imported = g.import_texture("imported", tex(PixelFormat::Rgba16Float));
        let transient = g.create_texture("transient", tex(PixelFormat::Rgba16Float));
        let i1 = {
            let mut p = g.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(imported);
            p.write_texture(transient)
        };
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(i1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.assignment[imported.resource.index()], None);
        assert!(plan.assignment[transient.resource.index()].is_some());
    }

    #[test]
    fn three_disjoint_chain_packs_into_one_slot() {
        // Three RGBA16 transients with gapped, non-touching lifetimes:
        // a [0,1] (Main writes, Decals reads), b [2,3] (Fog writes, Particles
        // reads), c [4,5] (SsrResolve writes, Composite reads). Each begins
        // strictly after the previous ends, so all three reuse one slot:
        // saved = two of the three footprints. (A write-then-immediately-read
        // chain a->b->c would instead TOUCH at the shared pass and not alias;
        // the gaps here are deliberate.)
        let mut g = GraphBuilder::new();
        let a = g.create_texture("a", tex(PixelFormat::Rgba16Float));
        let b = g.create_texture("b", tex(PixelFormat::Rgba16Float));
        let c = g.create_texture("c", tex(PixelFormat::Rgba16Float));
        let a1 = g.add_pass(PassId::Main, PassKind::Render).write_texture(a);
        g.add_pass(PassId::Decals, PassKind::Render)
            .read_texture(a1);
        let b1 = g.add_pass(PassId::Fog, PassKind::Render).write_texture(b);
        g.add_pass(PassId::ParticlesDraw, PassKind::Render)
            .read_texture(b1);
        let c1 = g
            .add_pass(PassId::SsrResolve, PassKind::Render)
            .write_texture(c);
        g.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(c1)
            .presents();
        let g = g.compile().expect("compiles");

        let plan = plan_aliasing(&g, 100, 100);
        assert_eq!(plan.slots.len(), 1);
        assert_eq!(plan.slots[0].members.len(), 3);
        assert_eq!(
            plan.saved_bytes(),
            2 * size_at_100(PixelFormat::Rgba16Float)
        );
    }

    #[test]
    fn real_frame_graph_aliases_some_transients() {
        // The actual per-frame graph with SSAO + bloom on: `ao_output` (early,
        // SsaoBlur -> Main) and `bloom_top` (late, Bloom -> Composite) are both
        // transient with disjoint lifetimes, so the planner saves at least
        // `ao_output`'s footprint. Guards the import->create reclassification +
        // the end-to-end planner against a representative graph.
        let mut inputs = ALL_OFF;
        inputs.ssao_enabled = true;
        inputs.bloom_enabled = true;
        let g = build_frame_graph(&inputs).expect("frame graph compiles");
        let plan = plan_aliasing(&g, 1920, 1080);
        assert!(
            plan.saved_bytes() > 0,
            "ao_output + bloom_top have disjoint lifetimes and should alias"
        );
    }
}
