// concinnity-eas/src/join_bench.rs
//
// Measures the cost of joining two component columns by entity three ways, so
// the draw-list build path can be sized before the engine commits to a
// multi-component layout. A draw list pairs each renderable entity's transform
// with its mesh descriptor; how that pairing is resolved is the question:
//
//   hashmap_crossref  iterate one column, look the partner up by id in a
//                     HashMap. This is the engine's current AssetId pattern.
//   joinindex_probe   iterate one column, test the JoinIndex mask and read the
//                     partner's row directly. This is what a query!(A, B) would
//                     generate.
//   dense_zip         iterate two entity-aligned arrays in lockstep. The
//                     archetype ideal: no lookup, pure sequential reads. An
//                     upper bound the column layout cannot beat, only approach.
//
// The partner (mesh) column is laid out in an order independent of the driver
// (transform) column, so the partner read is a real scattered access in the
// hashmap and probe paths, not a coincidental sequential one.

use std::collections::HashMap;

use crate::column::Column;
use crate::entity::{Entities, Entity};
use crate::join::JoinIndex;
use crate::mask::{ComponentId, ComponentMask};
use crate::tick::Tick;

fn transform_id() -> ComponentId {
    ComponentId::new(1)
}

fn mesh_id() -> ComponentId {
    ComponentId::new(2)
}

// The bulky per-entity datum a draw list must place: a world matrix.
#[derive(Clone, Copy)]
struct Transform {
    m: [f32; 16],
}

// The small renderable descriptor joined onto each transform.
#[derive(Clone, Copy)]
struct MeshRenderer {
    mesh: u32,
    material: u32,
}

// Deterministic xorshift so the layout (and thus the cache behavior) is
// identical across runs and machines.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn shuffle<T>(items: &mut [T], rng: &mut Rng) {
    for i in (1..items.len()).rev() {
        let j = rng.below(i + 1);
        items.swap(i, j);
    }
}

struct Bench {
    transforms: Column<Transform>,
    meshes: Column<MeshRenderer>,
    join: JoinIndex,
    // entity index -> row in the mesh column (the id-keyed cross-ref baseline).
    mesh_row_by_index: HashMap<u32, u32>,
    // The renderable subset, transform-and-mesh aligned (archetype ideal).
    dense_t: Vec<Transform>,
    dense_m: Vec<MeshRenderer>,
    // {TRANSFORM, MESH}, the required set a query!(Transform, MeshRenderer) tests.
    required: ComponentMask,
}

// Build `n` entities that all carry a transform; `renderable_permil`/1000 of
// them also carry a mesh. The mesh column is shuffled so its rows do not track
// the transform rows.
fn build(n: usize, renderable_permil: u32, seed: u64) -> Bench {
    let mut entities = Entities::new();
    let mut transforms = Column::new();
    let mut join = JoinIndex::new();
    let mut rng = Rng(seed);
    let mut renderable: Vec<Entity> = Vec::new();

    for i in 0..n {
        let e = entities.alloc();
        let row = transforms.len() as u32;
        transforms.push(e, Transform { m: [i as f32; 16] }, Tick::ZERO);
        join.set(e, transform_id(), row);
        if rng.below(1000) < renderable_permil as usize {
            renderable.push(e);
        }
    }

    shuffle(&mut renderable, &mut rng);

    let mut meshes = Column::new();
    let mut mesh_row_by_index = HashMap::new();
    for &e in &renderable {
        let idx = e.index();
        let row = meshes.len() as u32;
        meshes.push(
            e,
            MeshRenderer {
                mesh: idx,
                material: idx.wrapping_mul(2),
            },
            Tick::ZERO,
        );
        join.set(e, mesh_id(), row);
        mesh_row_by_index.insert(idx, row);
    }

    // The renderable subset in transform-iteration order, so every strategy
    // visits the same pairs and must agree on the checksum.
    let mut dense_t = Vec::new();
    let mut dense_m = Vec::new();
    for (e, t) in transforms.iter_with_entities() {
        if let Some(&row) = mesh_row_by_index.get(&e.index()) {
            dense_t.push(*t);
            dense_m.push(meshes[row as usize]);
        }
    }

    let mut required = ComponentMask::with(transform_id());
    required.insert(mesh_id());

    Bench {
        transforms,
        meshes,
        join,
        mesh_row_by_index,
        dense_t,
        dense_m,
        required,
    }
}

fn combine(acc: u64, t: &Transform, m: &MeshRenderer) -> u64 {
    acc.wrapping_add(t.m[0].to_bits() as u64)
        .wrapping_add(m.mesh as u64)
        .wrapping_add(m.material as u64)
}

fn run_hashmap(b: &Bench) -> u64 {
    let mut acc = 0u64;
    for (e, t) in b.transforms.iter_with_entities() {
        if let Some(&row) = b.mesh_row_by_index.get(&e.index()) {
            acc = combine(acc, t, &b.meshes[row as usize]);
        }
    }
    acc
}

fn run_probe(b: &Bench) -> u64 {
    let mut acc = 0u64;
    for (e, t) in b.transforms.iter_with_entities() {
        if b.join.matches(e, b.required, ComponentMask::EMPTY) {
            let row = b
                .join
                .row(e, mesh_id())
                .expect("a matched entity has a mesh row");
            acc = combine(acc, t, &b.meshes[row as usize]);
        }
    }
    acc
}

fn run_dense(b: &Bench) -> u64 {
    let mut acc = 0u64;
    for (t, m) in b.dense_t.iter().zip(b.dense_m.iter()) {
        acc = combine(acc, t, m);
    }
    acc
}

// One join strategy: walk the world and return a checksum over the joined pairs.
type Strategy = fn(&Bench) -> u64;

#[test]
fn join_strategies_are_equivalent() {
    let b = build(2_000, 750, 0x1234_5678);
    let hashmap = run_hashmap(&b);
    let probe = run_probe(&b);
    let dense = run_dense(&b);
    assert_eq!(
        probe, hashmap,
        "probe join must match the hashmap cross-ref"
    );
    assert_eq!(dense, hashmap, "dense zip must match the hashmap cross-ref");
    assert_ne!(hashmap, 0, "checksum should be non-trivial");
    assert_eq!(
        b.dense_t.len(),
        b.meshes.len(),
        "the dense subset is exactly the renderable entities"
    );
}

#[test]
#[ignore = "microbench; run with `cargo test -p concinnity-eas --release -- --ignored --nocapture join_probe`"]
fn join_probe_microbench() {
    let n = 200_000usize;
    let b = build(n, 750, 0x00C0_FFEE);
    let renderable = b.dense_t.len();
    let reps = 200u32;

    let strategies: [(&str, Strategy); 3] = [
        ("dense_zip (archetype ideal)", run_dense),
        ("joinindex_probe (query!)", run_probe),
        ("hashmap_crossref (today)", run_hashmap),
    ];

    println!("\njoin microbench: {n} entities, {renderable} renderable, {reps} reps/strategy");
    for (name, f) in strategies {
        let warm = std::hint::black_box(f(&b));
        let start = std::time::Instant::now();
        let mut last = 0u64;
        for _ in 0..reps {
            last = std::hint::black_box(f(&b));
        }
        let elapsed = start.elapsed();
        assert_eq!(last, warm, "{name} is not deterministic");
        let per_ns = elapsed.as_secs_f64() * 1e9 / (reps as f64 * renderable as f64);
        let ms = elapsed.as_secs_f64() * 1e3;
        println!("  {name:30} {ms:8.2} ms   {per_ns:6.2} ns/entity");
    }
}
