// src/gfx/render_graph/compile.rs
//
// Frozen `CompiledGraph` produced by `GraphBuilder::compile`. The compile
// pass:
//
//   1. Validates exactly one pass declares `presents()`.
//   2. Validates every declared read has a producer (write) in the graph.
//   3. Topologically sorts passes by read-after-write + write-after-write
//      edges. Cycles are an error.
//   4. Derives a `barriers_before` list per pass from the per-resource
//      state machine (Undefined → Read → Write transitions).
//   5. Computes a `[first, last]` pass-index lifetime per resource so a
//      future aliaser can overlap non-overlapping lifetimes in
//      physical memory.
//
// The graph does not yet allocate transient resources or interpret
// barriers per-backend. This module only produces the data the backend
// executor consumes.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use super::builder::{GraphBuilder, ResourceVersion};
use super::passes::PassId;
use super::types::{
    BarrierOp, PassKind, PassRange, ReadStages, ResourceId, ResourceOrigin, ResourceState,
    TextureDesc,
};

// Compiler error. Returned by [`GraphBuilder::compile`]; the call site
// typically panics (these are programmer errors, not runtime conditions).
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    // No pass declared `presents()`. Every graph must terminate at the
    // swapchain write.
    MissingPresenter,
    // More than one pass declared `presents()`. The graph compiler does
    // not pick one for you.
    MultiplePresenters(usize),
    // A pass declared `read_*(handle)` but no pass writes that exact
    // `(resource, version)` pair. Most commonly this happens when a
    // caller forgets to thread the post-write handle from a prior pass
    // (used the pre-write `h` instead of the `h1` returned by
    // `write_texture`).
    MissingProducer {
        pass: PassId,
        resource_label: &'static str,
        version: u32,
    },
    // The read / write graph has a cycle: toposort visited fewer
    // passes than declared.
    Cycle,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::MissingPresenter => {
                write!(
                    f,
                    "no pass declared presents(); the graph has no terminal node"
                )
            }
            GraphError::MultiplePresenters(n) => {
                write!(f, "{} passes declared presents(); only one is allowed", n)
            }
            GraphError::MissingProducer {
                pass,
                resource_label,
                version,
            } => write!(
                f,
                "pass {:?} reads {} v{} but no pass writes that version",
                pass, resource_label, version
            ),
            GraphError::Cycle => write!(f, "cycle in render-graph read/write edges"),
        }
    }
}

impl std::error::Error for GraphError {}

// One pass in execution order. Carries the declared reads / writes, the
// barriers the executor must emit before running this pass, and the
// `PassKind` + `PassId` the backend dispatches on.
#[derive(Debug, Clone)]
pub struct CompiledPass {
    pub id: PassId,
    pub kind: PassKind,
    pub reads: Vec<ResourceVersion>,
    pub writes: Vec<ResourceVersion>,
    pub presents: bool,
    pub barriers_before: Vec<BarrierOp>,
}

// One resource in the compiled graph. Carries the lifetime interval
// (in compiled-pass-index space) and the origin distinction the
// aliaser will care about.
#[derive(Debug, Clone)]
pub struct CompiledResource {
    pub label: &'static str,
    pub origin: ResourceOrigin,
    pub lifetime: PassRange,
    // `true` for textures, `false` for buffers. The executor branches
    // on this when resolving handles to backend objects.
    pub is_texture: bool,
    // Texture shape (format / size / sample count / layers), `None` for a
    // buffer. The aliasing planner uses it to size each transient resource;
    // the backend will use it to allocate the realised resource.
    pub tex_desc: Option<TextureDesc>,
}

// Frozen graph the per-backend executor consumes. Passes are in
// execution order; barriers are pre-derived; resource lifetimes are
// ready for a future aliaser.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    pub passes: Vec<CompiledPass>,
    pub resources: Vec<CompiledResource>,
}

impl CompiledGraph {
    // Restrict a pass's `barriers_before` to the resources whose label is in
    // `allow`, pairing each kept barrier with that label. A backend executor
    // uses this to drive native transitions for the subset of resources that
    // have moved off hand-written inline barriers, while every other resource
    // keeps its existing inline / render-pass-driven path. The label is the
    // stable join key the backend resolves to its GPU object.
    pub fn pass_barriers_for(
        &self,
        pass: &CompiledPass,
        allow: &[&str],
    ) -> Vec<(&'static str, BarrierOp)> {
        pass.barriers_before
            .iter()
            .filter_map(|op| {
                let label = self.resources[op.resource_index()].label;
                allow.contains(&label).then_some((label, *op))
            })
            .collect()
    }
}

impl GraphBuilder {
    // Topologically sort, validate, and derive per-pass barriers.
    // Returns the frozen graph the executor consumes. The graph must
    // declare exactly one `presents()` pass: the terminal swapchain
    // write that ends the frame.
    pub fn compile(self) -> Result<CompiledGraph, GraphError> {
        let GraphBuilder { resources, passes } = self;
        let n_passes = passes.len();
        let n_resources = resources.len();

        // Step 1: presenter validation
        let presenters: Vec<usize> = passes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.presents)
            .map(|(i, _)| i)
            .collect();
        match presenters.len() {
            0 => return Err(GraphError::MissingPresenter),
            1 => {}
            n => return Err(GraphError::MultiplePresenters(n)),
        }

        // Step 2: build (resource, version) -> writer-pass lookup
        let mut writer_of: HashMap<(ResourceId, u32), usize> = HashMap::new();
        for (i, pass) in passes.iter().enumerate() {
            for w in &pass.writes {
                // Each (resource, version) pair has exactly one writer
                // by construction (write_texture bumps the version, so
                // two passes can't claim the same one).
                writer_of.insert((w.resource, w.version), i);
            }
        }

        // Step 3: validate every read has a producer
        // An imported resource at version 0 has an *implicit* producer
        // (the engine that owns the GPU object). Reads of (imported, v0)
        // are therefore always legal even though no graph pass writes
        // them. Reads of (transient, v0) and reads at any version > 0
        // still need a real producer in the same graph.
        for pass in passes.iter() {
            for r in &pass.reads {
                if writer_of.contains_key(&(r.resource, r.version)) {
                    continue;
                }
                let decl = &resources[r.resource.index()];
                let implicit_producer = r.version == 0 && decl.origin() == ResourceOrigin::Imported;
                if !implicit_producer {
                    return Err(GraphError::MissingProducer {
                        pass: pass.id,
                        resource_label: decl.label(),
                        version: r.version,
                    });
                }
            }
        }

        // Step 4: build dependency edges and run Kahn's toposort
        // edges[i] = list of passes that depend on i (must run after i).
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n_passes];
        let mut in_degree: Vec<usize> = vec![0; n_passes];
        let add_edge =
            |from: usize, to: usize, edges: &mut [Vec<usize>], in_degree: &mut [usize]| {
                if from != to {
                    edges[from].push(to);
                    in_degree[to] += 1;
                }
            };

        // Build a (resource, version) → [reader pass_idxs] lookup so the
        // WAR step below can find prior readers without rescanning every
        // pass per write.
        let mut readers_of: HashMap<(ResourceId, u32), Vec<usize>> = HashMap::new();
        for (i, pass) in passes.iter().enumerate() {
            for r in &pass.reads {
                readers_of
                    .entry((r.resource, r.version))
                    .or_default()
                    .push(i);
            }
        }

        for (pass_idx, pass) in passes.iter().enumerate() {
            // Read-after-write: each read of (resource, version) requires
            // the pass that wrote (resource, version) to precede us.
            for r in &pass.reads {
                if let Some(&w) = writer_of.get(&(r.resource, r.version)) {
                    add_edge(w, pass_idx, &mut edges, &mut in_degree);
                }
            }
            // Write-after-write: a write that produced version V depends
            // on the pass that produced version V-1 (the prior content
            // matters for blend-style writes; for plain overwrites it
            // doesn't but the ordering is still correct).
            for w in &pass.writes {
                if w.version > 1
                    && let Some(&prev_writer) = writer_of.get(&(w.resource, w.version - 1))
                {
                    add_edge(prev_writer, pass_idx, &mut edges, &mut in_degree);
                }
            }
            // Write-after-read: a write that produces version V requires
            // every reader of version V-1 to precede us. Without this,
            // a reader of the pre-write version could otherwise be
            // re-ordered after the writer, sampling stale-from-the-other-
            // direction data. In the Metal frame this fires on
            // AutoExposure (reads `hdr_resolve_v1` produced by Main)
            // pinning before Decals / Fog / ParticlesDraw (which bump
            // hdr_resolve to v2+). Self-edges are filtered by add_edge.
            for w in &pass.writes {
                if w.version > 0
                    && let Some(readers) = readers_of.get(&(w.resource, w.version - 1))
                {
                    for &reader in readers {
                        add_edge(reader, pass_idx, &mut edges, &mut in_degree);
                    }
                }
            }
        }

        // Stable order: when two passes are both ready, pick the one
        // declared first. A min-heap on the original pass index achieves
        // this without rescanning the ready set.
        let mut ready: BinaryHeap<Reverse<usize>> = (0..n_passes)
            .filter(|&i| in_degree[i] == 0)
            .map(Reverse)
            .collect();
        let mut order: Vec<usize> = Vec::with_capacity(n_passes);
        while let Some(Reverse(idx)) = ready.pop() {
            order.push(idx);
            for &neighbor in &edges[idx] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    ready.push(Reverse(neighbor));
                }
            }
        }
        if order.len() != n_passes {
            return Err(GraphError::Cycle);
        }

        // Step 5: realise compiled passes in execution order
        let mut compiled_passes: Vec<CompiledPass> = order
            .iter()
            .map(|&orig_idx| {
                let decl = &passes[orig_idx];
                CompiledPass {
                    id: decl.id,
                    kind: decl.kind,
                    reads: decl.reads.clone(),
                    writes: decl.writes.clone(),
                    presents: decl.presents,
                    barriers_before: Vec::new(),
                }
            })
            .collect();

        // Step 6: derive per-pass barriers
        derive_barriers(&mut compiled_passes, n_resources);

        // Step 7: compute resource lifetimes
        let mut lifetimes: Vec<Option<PassRange>> = vec![None; n_resources];
        for (sorted_idx, pass) in compiled_passes.iter().enumerate() {
            for v in pass.writes.iter().chain(pass.reads.iter()) {
                let i = v.resource.index();
                let merged = match lifetimes[i] {
                    None => PassRange {
                        first: sorted_idx,
                        last: sorted_idx,
                    },
                    Some(PassRange { first, .. }) => PassRange {
                        first,
                        last: sorted_idx,
                    },
                };
                lifetimes[i] = Some(merged);
            }
        }

        let compiled_resources: Vec<CompiledResource> = resources
            .into_iter()
            .enumerate()
            .map(|(i, decl)| {
                // A resource that's declared but never touched gets a
                // degenerate `[0, 0]` lifetime; the executor can treat
                // it as a leak warning later.
                let lifetime = lifetimes[i].unwrap_or(PassRange { first: 0, last: 0 });
                CompiledResource {
                    label: decl.label(),
                    origin: decl.origin(),
                    lifetime,
                    is_texture: decl.is_texture(),
                    tex_desc: decl.texture_desc(),
                }
            })
            .collect();

        Ok(CompiledGraph {
            passes: compiled_passes,
            resources: compiled_resources,
        })
    }
}

// Derive each pass's `barriers_before` from the per-resource state machine
// (Undefined -> Read -> Write). Emit a `BarrierOp` whenever a resource's
// effective access (Write if the pass writes it, else Read) differs from its
// prior state.
//
// Effective-access rule: a pass that writes a resource leaves it in `Write`
// state regardless of whether it also reads it (blend-style read-modify-write
// is handled by the backend's render-pass setup, not by an intra-pass
// barrier).
//
// Read-stage union: a `* -> Read` barrier carries the stage union of the WHOLE
// following read-run (every pass that reads this version before the next
// writer), and a `Read -> Write` barrier carries the prior run's union. This is
// why the work is done per resource rather than per pass: a write read by a
// compute consumer and a fragment consumer needs ONE producer barrier that
// makes the write visible to both stages, so the deriver must see the whole run
// before emitting that first barrier. A per-consumer read-to-read barrier would
// not carry the producing write and so would not synchronise the second stage.
fn derive_barriers(passes: &mut [CompiledPass], n_resources: usize) {
    // Per-pass effective access to one resource: a write (which wins over any
    // read the same pass declares) or a read in the pass's shader stage.
    #[derive(Copy, Clone)]
    enum Eff {
        Write,
        Read(ReadStages),
    }

    // Build each resource's timeline: its (pass index, effective access) pairs
    // in pass order. The outer loop is ascending pass index and each pass
    // contributes at most one entry per resource, so every timeline stays
    // sorted by pass index without an explicit sort.
    let mut timeline: Vec<Vec<(usize, Eff)>> = (0..n_resources).map(|_| Vec::new()).collect();
    for (i, pass) in passes.iter().enumerate() {
        let stage = ReadStages::for_pass_kind(pass.kind);
        let mut access: HashMap<ResourceId, Eff> = HashMap::new();
        for r in &pass.reads {
            access.entry(r.resource).or_insert(Eff::Read(stage));
        }
        for w in &pass.writes {
            access.insert(w.resource, Eff::Write);
        }
        // Push in resource-index order so each pass's `barriers_before` ends up
        // sorted by resource index (the outer resource loop below is ascending),
        // matching the deterministic order the executor expects.
        let mut touched: Vec<(ResourceId, Eff)> = access.into_iter().collect();
        touched.sort_by_key(|(r, _)| r.0);
        for (res, eff) in touched {
            timeline[res.index()].push((i, eff));
        }
    }

    // Walk each resource's timeline, tracking its running state and the current
    // read-run's stage union, emitting a barrier on each state change.
    for (r_idx, entries) in timeline.iter().enumerate() {
        let resource = ResourceId(r_idx as u32);
        let mut state = ResourceState::Undefined;
        let mut run_stages = ReadStages::empty();

        for (k, &(pass_idx, eff)) in entries.iter().enumerate() {
            match eff {
                Eff::Write => {
                    if state != ResourceState::Write {
                        // A `Read -> Write` carries the prior run's union (the
                        // write must wait on every reader); `Undefined -> Write`
                        // has no Read side.
                        let read_stages = if state == ResourceState::Read {
                            run_stages
                        } else {
                            ReadStages::empty()
                        };
                        passes[pass_idx].barriers_before.push(BarrierOp {
                            resource,
                            from: state,
                            to: ResourceState::Write,
                            read_stages,
                        });
                        state = ResourceState::Write;
                        run_stages = ReadStages::empty();
                    }
                }
                Eff::Read(_) => {
                    if state != ResourceState::Read {
                        // First read of a new run: union the contiguous run's
                        // stages so the one producer barrier covers them all.
                        let mut run = ReadStages::empty();
                        for &(_, e) in entries[k..].iter() {
                            match e {
                                Eff::Read(s) => run = run.union(s),
                                Eff::Write => break,
                            }
                        }
                        passes[pass_idx].barriers_before.push(BarrierOp {
                            resource,
                            from: state,
                            to: ResourceState::Read,
                            read_stages: run,
                        });
                        state = ResourceState::Read;
                        run_stages = run;
                    }
                    // Continuation read: the run's barrier already carries this
                    // stage (unioned above), so emit nothing.
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::{BufferUsage, PixelFormat, TextureDesc, TextureSize, TextureUsage};
    use super::*;
    use crate::gfx::render_graph::builder::GraphBuilder;
    use crate::gfx::render_graph::passes::PassId;
    use crate::gfx::render_graph::types::{BufferDesc, PassKind};

    fn tex() -> TextureDesc {
        TextureDesc {
            width: TextureSize::Drawable,
            height: TextureSize::Drawable,
            format: PixelFormat::Rgba16Float,
            sample_count: 1,
            array_layers: 1,
            usage: TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        }
    }

    fn buf() -> BufferDesc {
        BufferDesc {
            size_bytes: None,
            usage: BufferUsage::STORAGE,
        }
    }

    #[test]
    fn linear_chain_toposorts_in_declared_order() {
        // A writes T -> B reads T writes U -> C reads U presents.
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let u = b.create_texture("u", tex());

        let t1 = b
            .add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(t);
        let u1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(t1);
            p.write_texture(u)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(u1)
            .presents();

        let g = b.compile().expect("graph compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Shadow, PassId::Main, PassId::Composite]);
    }

    #[test]
    fn diamond_toposorts_with_stable_tiebreak() {
        // A writes T; B reads T writes U; C reads T writes V; D reads U+V presents.
        // B and C are both ready after A; declared first wins, so the
        // order is [A, B, C, D].
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let u = b.create_texture("u", tex());
        let v = b.create_texture("v", tex());

        let t1 = b
            .add_pass(PassId::Shadow, PassKind::Render)
            .write_texture(t);
        let u1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(t1);
            p.write_texture(u)
        };
        let v1 = {
            let mut p = b.add_pass(PassId::SsaoKernel, PassKind::Render);
            p.read_texture(t1);
            p.write_texture(v)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(u1)
            .read_texture(v1)
            .presents();

        let g = b.compile().expect("graph compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        // Shadow must precede both Main and SsaoKernel (they read its
        // output); Composite must come last (it reads both their
        // outputs). The stable tie-break picks Main over SsaoKernel
        // because Main was declared first.
        assert_eq!(
            order,
            vec![
                PassId::Shadow,
                PassId::Main,
                PassId::SsaoKernel,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn read_modify_write_chain_orders_correctly() {
        // A writes hdr v1; B reads v1 writes v2 (decals); C reads v2 writes v3
        // (fog); D reads v3 presents. The version-bump-implies-edge rule
        // forces strict serial order.
        let mut b = GraphBuilder::new();
        let hdr = b.create_texture("hdr", tex());

        let v1 = b
            .add_pass(PassId::Main, PassKind::Render)
            .write_texture(hdr);
        let v2 = {
            let mut p = b.add_pass(PassId::Decals, PassKind::Render);
            p.read_texture(v1);
            p.write_texture(v1)
        };
        let v3 = {
            let mut p = b.add_pass(PassId::Fog, PassKind::Render);
            p.read_texture(v2);
            p.write_texture(v2)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(v3)
            .presents();

        let g = b.compile().expect("graph compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(
            order,
            vec![PassId::Main, PassId::Decals, PassId::Fog, PassId::Composite,]
        );
    }

    #[test]
    fn war_exposes_cross_rmw_cycle() {
        // Two passes RMW different resources, each reading the other's
        // pre-write version: Decals reads y_v1 + writes x_v1 → x_v2;
        // SsaoBlur reads x_v1 + writes y_v1 → y_v2. With WAR enforced,
        // each write needs the *other* pass to have completed its read
        // first, giving SsaoBlur → Decals (Decals's write of x_v2
        // depends on SsaoBlur reading x_v1) AND Decals → SsaoBlur
        // (SsaoBlur's write of y_v2 depends on Decals reading y_v1).
        // That's a cycle, and it's a real one: there's no valid serial
        // order for this pattern.
        let mut b = GraphBuilder::new();
        let x = b.create_texture("x", tex());
        let y = b.create_texture("y", tex());

        let x1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(x);
        let y1 = b.add_pass(PassId::Fog, PassKind::Render).write_texture(y);

        let _x2 = {
            let mut p = b.add_pass(PassId::Decals, PassKind::Render);
            p.read_texture(y1);
            p.write_texture(x1)
        };
        let _y2 = {
            let mut p = b.add_pass(PassId::SsaoBlur, PassKind::Render);
            p.read_texture(x1);
            p.write_texture(y1)
        };
        b.add_pass(PassId::Composite, PassKind::Render).presents();

        match b.compile() {
            Err(GraphError::Cycle) => {}
            other => panic!("expected Cycle, got {:?}", other),
        }
    }

    #[test]
    fn mutual_write_cycle_errors() {
        // A writes X v1 reads Y v2; B writes Y v1 reads X v2; C writes X v2;
        // D writes Y v2. C and D depend on A and B respectively (WAW on the
        // version bump). A reads Y v2 → depends on D. B reads X v2 → depends
        // on C. So: A -> C, B -> D, A -> D, B -> C, D -> A, C -> B. Cycle.
        let mut b = GraphBuilder::new();
        let x = b.create_texture("x", tex());
        let y = b.create_texture("y", tex());

        // First writes establish v1 for both.
        let x1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(x);
        let y1 = b.add_pass(PassId::Fog, PassKind::Render).write_texture(y);

        // Second writes claim v2 for both, declaring the cross-dependency.
        // Decals reads y v2 (will exist), writes x v2.
        // SsaoBlur reads x v2 (will exist), writes y v2.
        // Each "reads v2" requires the other pass to have written v2 first.
        // Mutual.
        {
            let mut p = b.add_pass(PassId::Decals, PassKind::Render);
            // Read of v2 of y; producer is SsaoBlur (pass index 3).
            p.read_texture(super::super::types::TextureHandle {
                resource: y1.resource,
                version: 2,
            });
            p.write_texture(x1);
        }
        {
            let mut p = b.add_pass(PassId::SsaoBlur, PassKind::Render);
            p.read_texture(super::super::types::TextureHandle {
                resource: x1.resource,
                version: 2,
            });
            p.write_texture(y1);
        }
        b.add_pass(PassId::Composite, PassKind::Render).presents();

        match b.compile() {
            Err(GraphError::Cycle) => {}
            other => panic!("expected Cycle, got {:?}", other),
        }
    }

    #[test]
    fn war_edges_pin_reader_before_writer() {
        // Main writes hdr_resolve v1. AutoExposure reads v1. Decals
        // writes v1 → v2. Without WAR edges, toposort could place
        // Decals before AutoExposure (Decals → Main → AutoExposure has
        // no edge constraint either way). WAR forces AutoExposure to
        // precede Decals because Decals's write of v2 depends on all
        // readers of v1 completing first.
        //
        // Declaration order in this test deliberately puts Decals
        // BEFORE AutoExposure so the toposort can't rely on it. Only
        // the WAR edge gets us the right order.
        let mut b = GraphBuilder::new();
        let hdr = b.create_texture("hdr_resolve", tex());

        let hdr_v1 = b
            .add_pass(PassId::Main, PassKind::Render)
            .write_texture(hdr);

        // Decals declared before AutoExposure on purpose; the WAR edge
        // from AutoExposure to Decals must override declaration order.
        let _hdr_v2 = b
            .add_pass(PassId::Decals, PassKind::Render)
            .write_texture(hdr_v1);

        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(hdr_v1);

        b.add_pass(PassId::Composite, PassKind::Render).presents();

        let g = b.compile().expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        // Main first, then AutoExposure (WAR-pinned before Decals),
        // then Decals, then Composite.
        assert_eq!(
            order,
            vec![
                PassId::Main,
                PassId::AutoExposure,
                PassId::Decals,
                PassId::Composite,
            ]
        );
    }

    #[test]
    fn imported_v0_read_does_not_error() {
        // The engine owns imported resources, so reading them at version
        // 0 (their initial / engine-produced version) is always legal,
        // even when no graph pass writes them. Mirrors how the Main pass
        // reads env_irradiance / env_prefilter cubemaps the engine
        // uploaded at init.
        let mut b = GraphBuilder::new();
        let env = b.import_texture("env", tex());
        let scene = b.create_texture("scene", tex());
        {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_texture(env);
            p.write_texture(scene);
        }
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(super::super::types::TextureHandle {
                resource: scene.resource,
                version: 1,
            })
            .presents();

        let g = b.compile().expect("imported v0 read should compile");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Main, PassId::Composite]);
    }

    #[test]
    fn transient_v0_read_still_errors() {
        // The imported-v0 escape hatch only covers imported resources.
        // A transient declared via create_texture and then read without
        // a write is a real bug: the engine has no implicit producer.
        let mut b = GraphBuilder::new();
        let t = b.create_texture("scratch", tex());
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t)
            .presents();
        match b.compile() {
            Err(GraphError::MissingProducer {
                pass: PassId::Composite,
                resource_label: "scratch",
                version: 0,
            }) => {}
            other => panic!("expected MissingProducer for transient v0, got {:?}", other),
        }
    }

    #[test]
    fn missing_producer_errors() {
        // Read of a resource version that was never written.
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());

        // Skip the write; directly synthesise a handle at version 1
        // (which write_texture would have returned).
        let phantom = super::super::types::TextureHandle {
            resource: t.resource,
            version: 1,
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(phantom)
            .presents();

        match b.compile() {
            Err(GraphError::MissingProducer {
                pass: PassId::Composite,
                resource_label: "t",
                version: 1,
            }) => {}
            other => panic!("expected MissingProducer, got {:?}", other),
        }
    }

    #[test]
    fn no_presenter_errors() {
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        b.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        // No presents()!
        match b.compile() {
            Err(GraphError::MissingPresenter) => {}
            other => panic!("expected MissingPresenter, got {:?}", other),
        }
    }

    #[test]
    fn multiple_presenters_errors() {
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.presents();
            let _ = p.write_texture(t);
        }
        b.add_pass(PassId::Composite, PassKind::Render).presents();
        match b.compile() {
            Err(GraphError::MultiplePresenters(2)) => {}
            other => panic!("expected MultiplePresenters(2), got {:?}", other),
        }
    }

    #[test]
    fn barriers_emit_on_state_transitions() {
        // Main writes T -> Composite reads T. The barrier on Composite
        // should be Undefined→...→Write (Main's effective state is
        // Write because it writes) and then Write→Read (Composite reads).
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let t1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();

        let g = b.compile().expect("compiles");
        // Pass 0 (Main): T transitions Undefined -> Write, one barrier.
        assert_eq!(g.passes[0].barriers_before.len(), 1);
        assert_eq!(
            g.passes[0].barriers_before[0].from,
            ResourceState::Undefined
        );
        assert_eq!(g.passes[0].barriers_before[0].to, ResourceState::Write);
        // Pass 1 (Composite): T transitions Write -> Read, one barrier.
        assert_eq!(g.passes[1].barriers_before.len(), 1);
        assert_eq!(g.passes[1].barriers_before[0].from, ResourceState::Write);
        assert_eq!(g.passes[1].barriers_before[0].to, ResourceState::Read);
    }

    #[test]
    fn pass_barriers_for_filters_by_label() {
        // Main writes "keep" + "skip"; Composite reads both. The allowlist
        // ["keep"] must select only "keep"'s barrier on each pass, dropping
        // "skip"'s entirely.
        let mut b = GraphBuilder::new();
        let keep = b.create_texture("keep", tex());
        let skip = b.create_texture("skip", tex());
        let (keep1, skip1) = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            (p.write_texture(keep), p.write_texture(skip))
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(keep1)
            .read_texture(skip1)
            .presents();

        let g = b.compile().expect("compiles");
        // Main writes both: two barriers total, one kept by the allowlist.
        let main = &g.passes[0];
        let kept = g.pass_barriers_for(main, &["keep"]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].0, "keep");
        assert_eq!(kept[0].1.from_state(), ResourceState::Undefined);
        assert_eq!(kept[0].1.to_state(), ResourceState::Write);
        // Composite reads both: "keep" transitions Write -> Read.
        let composite = &g.passes[1];
        let kept = g.pass_barriers_for(composite, &["keep"]);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].1.from_state(), ResourceState::Write);
        assert_eq!(kept[0].1.to_state(), ResourceState::Read);
        // An empty allowlist keeps nothing; an unknown label keeps nothing.
        assert!(g.pass_barriers_for(main, &[]).is_empty());
        assert!(g.pass_barriers_for(main, &["nope"]).is_empty());
    }

    #[test]
    fn consecutive_reads_coalesce_no_barriers() {
        // A writes T. B reads T. C reads T. D reads T presents.
        // Barriers: A=W, B=W->R, C=R->R (none), D=R->R (none).
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let t1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        b.add_pass(PassId::Decals, PassKind::Render)
            .read_texture(t1);
        b.add_pass(PassId::Fog, PassKind::Render).read_texture(t1);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();

        let g = b.compile().expect("compiles");
        // pass[0] = Main: 1 barrier (Undefined -> Write)
        assert_eq!(g.passes[0].barriers_before.len(), 1);
        // pass[1] = Decals: 1 barrier (Write -> Read)
        assert_eq!(g.passes[1].barriers_before.len(), 1);
        // pass[2] = Fog: 0 barriers (already Read)
        assert_eq!(g.passes[2].barriers_before.len(), 0);
        // pass[3] = Composite: 0 barriers
        assert_eq!(g.passes[3].barriers_before.len(), 0);
    }

    #[test]
    fn lifetime_intervals_span_first_write_to_last_read() {
        // Main(0) writes T; Decals(1) reads T; Fog(2) ignores T; Composite(3) reads T.
        // T lifetime: [0, 3].
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let unrelated = b.create_texture("u", tex());

        let t1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        b.add_pass(PassId::Decals, PassKind::Render)
            .read_texture(t1);
        b.add_pass(PassId::Fog, PassKind::Render)
            .write_texture(unrelated);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();

        let g = b.compile().expect("compiles");
        let t_idx = t.resource.index();
        let u_idx = unrelated.resource.index();
        assert_eq!(g.resources[t_idx].lifetime.first, 0);
        assert_eq!(g.resources[t_idx].lifetime.last, 3);
        assert_eq!(g.resources[u_idx].lifetime.first, 2);
        assert_eq!(g.resources[u_idx].lifetime.last, 2);
    }

    #[test]
    fn buffer_dep_edges_work_too() {
        // GPU-cull-style: cull writes draw_args (buffer), main reads it.
        let mut b = GraphBuilder::new();
        let draw_args = b.create_buffer("draw_args", buf());
        let scene = b.create_texture("scene", tex());

        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(draw_args);
        let scene1 = {
            let mut p = b.add_pass(PassId::Main, PassKind::Render);
            p.read_buffer(args1);
            p.write_texture(scene)
        };
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(scene1)
            .presents();

        let g = b.compile().expect("compiles");
        let order: Vec<PassId> = g.passes.iter().map(|p| p.id).collect();
        assert_eq!(order, vec![PassId::Cull, PassId::Main, PassId::Composite]);
    }

    fn find(g: &CompiledGraph, id: PassId) -> &CompiledPass {
        g.passes.iter().find(|p| p.id == id).expect("pass present")
    }

    #[test]
    fn mixed_stage_read_run_unions_consumer_stages() {
        // Main writes hdr; AutoExposure (compute) and Composite (render) read
        // it. The single producer barrier on the first reader must carry BOTH
        // stages so the write is made visible to the compute and the fragment
        // consumer; the second reader coalesces (no barrier).
        let mut b = GraphBuilder::new();
        let hdr = b.create_texture("hdr", tex());
        let hdr_v1 = b
            .add_pass(PassId::Main, PassKind::Render)
            .write_texture(hdr);
        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(hdr_v1);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(hdr_v1)
            .presents();

        let g = b.compile().expect("compiles");
        // AutoExposure is the first reader: one Write -> Read barrier carrying
        // the whole run's stage union.
        let ae = find(&g, PassId::AutoExposure);
        assert_eq!(ae.barriers_before.len(), 1);
        assert_eq!(ae.barriers_before[0].from_state(), ResourceState::Write);
        assert_eq!(ae.barriers_before[0].to_state(), ResourceState::Read);
        let rs = ae.barriers_before[0].read_stages();
        assert!(rs.contains(ReadStages::COMPUTE));
        assert!(rs.contains(ReadStages::FRAGMENT));
        // Composite coalesces into the run: no hdr barrier.
        assert_eq!(find(&g, PassId::Composite).barriers_before.len(), 0);
    }

    #[test]
    fn fragment_read_carries_only_fragment_stage() {
        // A lone render-pass consumer carries FRAGMENT only (the common case;
        // the existing migrated resources all look like this).
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let t1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();

        let g = b.compile().expect("compiles");
        let comp = find(&g, PassId::Composite);
        assert_eq!(comp.barriers_before.len(), 1);
        let rs = comp.barriers_before[0].read_stages();
        assert!(rs.contains(ReadStages::FRAGMENT));
        assert!(!rs.contains(ReadStages::COMPUTE));
    }

    #[test]
    fn compute_read_carries_only_compute_stage() {
        // Cull (compute) writes draw_args; AutoExposure (compute) reads it.
        // The reader's barrier carries COMPUTE only. A render presenter writes
        // an unrelated target so the graph is well-formed.
        let mut b = GraphBuilder::new();
        let args = b.create_buffer("draw_args", buf());
        let scene = b.create_texture("scene", tex());
        let args1 = b
            .add_pass(PassId::Cull, PassKind::Compute)
            .write_buffer(args);
        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_buffer(args1);
        {
            let mut p = b.add_pass(PassId::Composite, PassKind::Render);
            let _ = p.write_texture(scene);
            p.presents();
        }

        let g = b.compile().expect("compiles");
        let ae = find(&g, PassId::AutoExposure);
        assert_eq!(ae.barriers_before.len(), 1);
        let rs = ae.barriers_before[0].read_stages();
        assert!(rs.contains(ReadStages::COMPUTE));
        assert!(!rs.contains(ReadStages::FRAGMENT));
    }

    #[test]
    fn war_barrier_carries_prior_read_run_stage_union() {
        // Main writes hdr v1; AutoExposure (compute) and Fog (render) read v1;
        // SsaoBlur writes v1 -> v2 (write only). SsaoBlur's WAR barrier must
        // wait on BOTH readers, so it carries the prior run's union.
        let mut b = GraphBuilder::new();
        let hdr = b.create_texture("hdr", tex());
        let v1 = b
            .add_pass(PassId::Main, PassKind::Render)
            .write_texture(hdr);
        b.add_pass(PassId::AutoExposure, PassKind::Compute)
            .read_texture(v1);
        b.add_pass(PassId::Fog, PassKind::Render).read_texture(v1);
        let v2 = b
            .add_pass(PassId::SsaoBlur, PassKind::Render)
            .write_texture(v1);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(v2)
            .presents();

        let g = b.compile().expect("compiles");
        let blur = find(&g, PassId::SsaoBlur);
        assert_eq!(blur.barriers_before.len(), 1);
        assert_eq!(blur.barriers_before[0].from_state(), ResourceState::Read);
        assert_eq!(blur.barriers_before[0].to_state(), ResourceState::Write);
        let rs = blur.barriers_before[0].read_stages();
        assert!(rs.contains(ReadStages::COMPUTE));
        assert!(rs.contains(ReadStages::FRAGMENT));
    }

    #[test]
    fn producer_write_barrier_has_empty_read_stages() {
        // A write-only producer transition (Undefined -> Write) has no Read
        // side, so its stage mask is empty.
        let mut b = GraphBuilder::new();
        let t = b.create_texture("t", tex());
        let t1 = b.add_pass(PassId::Main, PassKind::Render).write_texture(t);
        b.add_pass(PassId::Composite, PassKind::Render)
            .read_texture(t1)
            .presents();

        let g = b.compile().expect("compiles");
        let main = find(&g, PassId::Main);
        assert_eq!(main.barriers_before.len(), 1);
        assert_eq!(main.barriers_before[0].to_state(), ResourceState::Write);
        assert!(main.barriers_before[0].read_stages().is_empty());
    }
}
