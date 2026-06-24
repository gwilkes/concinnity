// Reflection probe capture math: the six cube-face view-projection matrices a
// probe renders the scene through, plus the load-time conversion of the
// captured faces into the prefiltered IBL payload the environment sampler
// consumes. Backend-agnostic; the Metal backend drives the actual scene render
// into each face (see metal/probe.rs). DirectX / Vulkan can reuse this math.
//
// Face order and orientation match the engine's cube convention
// (`concinnity-core::build::cubemap` / `environment_map::cube_texel_dir`):
//   0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z, with a face texel at (u,v) in [-1,1]
// looking along `cube_texel_dir(face, u, v)`. Each face's view-projection is
// built so that direction projects to NDC (u, -v) (screen-down is +v), which
// the orientation test pins exactly.

use std::f32::consts::FRAC_PI_2;

// Per-face camera basis (right, up, forward) in world space, derived so that a
// 90-degree view down -forward reproduces `cube_texel_dir`. forward = the face
// axis; right = d/du of the face direction; up = -d/dv.
const FACE_BASIS: [[[f32; 3]; 3]; 6] = [
    // [right, up, forward]
    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], // 0 +X
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], // 1 -X
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], // 2 +Y
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], // 3 -Y
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  // 4 +Z
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], // 5 -Z
];

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// Column-major multiply (out[col][row]), matching the engine's matrix storage.
fn mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

// Right-handed perspective with depth in [0,1], matching metal/math.rs
// `perspective`. 90-degree vertical FOV, square aspect (one cube face).
fn perspective_90(near: f32, far: f32) -> [[f32; 4]; 4] {
    let s = 1.0 / (FRAC_PI_2 / 2.0).tan(); // = 1.0 at 90 degrees
    let zs = far / (near - far);
    [
        [s, 0.0, 0.0, 0.0],
        [0.0, s, 0.0, 0.0],
        [0.0, 0.0, zs, -1.0],
        [0.0, 0.0, zs * near, 0.0],
    ]
}

// World->view matrix for a face's (right, up, forward) basis at `eye`, looking
// down -forward (same form as csm::look_at). Column-major arr[col][row].
fn face_view(eye: [f32; 3], r: [f32; 3], u: [f32; 3], f: [f32; 3]) -> [[f32; 4]; 4] {
    [
        [r[0], u[0], -f[0], 0.0],
        [r[1], u[1], -f[1], 0.0],
        [r[2], u[2], -f[2], 0.0],
        [-dot3(r, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

// The view-projection for cube face `face` (0..6) captured from `eye`.
#[allow(dead_code)]
pub fn face_view_projection(eye: [f32; 3], face: usize, near: f32, far: f32) -> [[f32; 4]; 4] {
    let b = FACE_BASIS[face];
    let view = face_view(eye, b[0], b[1], b[2]);
    mul(perspective_90(near, far), view)
}

// The world->view matrix alone for cube face `face`, captured from `eye`. The
// main pass needs both the combined view-projection (vertex clip transform) and
// the bare view matrix (some shaders reconstruct view-space data), so the probe
// capture builds a `ViewUniforms` from this plus `face_view_projection`.
#[allow(dead_code)]
pub fn face_view_matrix(eye: [f32; 3], face: usize) -> [[f32; 4]; 4] {
    let b = FACE_BASIS[face];
    face_view(eye, b[0], b[1], b[2])
}

// Where one reflection probe is captured from and the influence box it serves.
// Backend-agnostic: the renderer bakes a cube at `position` and, for a surface
// inside [box_min, box_max], selects this probe and parallax-corrects against the
// box. Derived from a declared `ReflectionProbe` asset or `auto_seed_probes`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbePlacement {
    pub position: [f32; 3],
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
}

impl ProbePlacement {
    // Build a placement from an authored `ReflectionProbe` (capture point +
    // half-extents): the box is `position` plus or minus `half_extents`.
    pub fn from_center_extents(position: [f32; 3], half_extents: [f32; 3]) -> ProbePlacement {
        ProbePlacement {
            position,
            box_min: [
                position[0] - half_extents[0],
                position[1] - half_extents[1],
                position[2] - half_extents[2],
            ],
            box_max: [
                position[0] + half_extents[0],
                position[1] + half_extents[1],
                position[2] + half_extents[2],
            ],
        }
    }
}

// Tracks how far a staggered probe bake has progressed. Baking every probe on
// one frame stalls proportionally to the probe count; instead the renderer bakes
// a bounded budget per frame and walks this cursor, so the load cost is spread
// and unbaked probes fall back to the sky until their turn. Indices are handed
// out in order so the baked cube array stays aligned with the placement list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeBakeQueue {
    total: usize,
    next: usize,
}

impl ProbeBakeQueue {
    pub fn new(total: usize) -> ProbeBakeQueue {
        ProbeBakeQueue { total, next: 0 }
    }

    // Whether any placement is still waiting to bake.
    pub fn pending(&self) -> bool {
        self.next < self.total
    }

    // The next placement index to bake, advancing the cursor. `None` when done.
    pub fn take_next(&mut self) -> Option<usize> {
        (self.next < self.total).then(|| {
            let i = self.next;
            self.next += 1;
            i
        })
    }

    // Abandon every remaining placement (e.g. a permanently ineligible world or
    // an unrecoverable bake error). Leaves the queue not `pending`.
    pub fn abort(&mut self) {
        self.next = self.total;
    }
}

// Phase of the single in-flight asynchronous bake. Exactly one probe is baked at
// a time, walking three phases across frames so the render thread never blocks on
// the capture: `Rendering` (six faces submitted, GPU running), `Converting` (faces
// read back, the prefilter convolution running off the render thread), and `Idle`
// (nothing in flight). The renderer holds the GPU resources for the in-flight
// probe; this enum only names which phase it is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakePhase {
    Idle,
    Rendering,
    Converting,
}

// What the renderer should do this frame to advance the asynchronous bake. The
// renderer maps each variant to a side effect: `StartNext` builds the next
// placement's capture buffers + targets (no face submitted yet), `RenderFace`
// submits one cube face (the six are spread one-per-frame so no single frame
// pays the whole capture), `Readback` copies the finished faces back and kicks
// the off-thread convolution, `Install` uploads the convolved cube and advances
// the queue, `Idle` does nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BakeAction {
    Idle,
    StartNext,
    RenderFace,
    Install,
    Readback,
}

// Decide the next action for the single-in-flight asynchronous bake. Pure so the
// transition table is locked by unit tests without a GPU:
//   * while `Rendering`, submit the next cube face while `more_faces` remain (one
//     per frame); once all six are submitted, do nothing until the GPU completion
//     flag `done` is set, then `Readback`;
//   * while `Converting`, do nothing until the off-thread `payload_ready`, then
//     `Install`;
//   * while `Idle`, start the next placement only when one is `queue_pending` and
//     the world is `eligible` to bake this frame (bindless + geometry present +
//     a non-empty cull); otherwise stay `Idle`.
// The invariants this guarantees: never read faces back before all six are
// submitted and the GPU signals completion, never install before the convolution
// finishes, and never begin a second bake while one is already in flight.
pub fn next_bake_action(
    phase: BakePhase,
    done: bool,
    payload_ready: bool,
    queue_pending: bool,
    eligible: bool,
    more_faces: bool,
) -> BakeAction {
    match phase {
        BakePhase::Rendering => {
            if more_faces {
                BakeAction::RenderFace
            } else if done {
                BakeAction::Readback
            } else {
                BakeAction::Idle
            }
        }
        BakePhase::Converting => {
            if payload_ready {
                BakeAction::Install
            } else {
                BakeAction::Idle
            }
        }
        BakePhase::Idle => {
            if queue_pending && eligible {
                BakeAction::StartNext
            } else {
                BakeAction::Idle
            }
        }
    }
}

// Largest number of probes auto-seed places. Bounds the bake cost + probe-cube
// memory for an un-authored world. Must not exceed the renderer's per-frame probe
// bind limit (`metal::uniforms::MAX_PROBES`); asserted by
// `auto_seed_budget_fits_max_probes` in the metal uniforms tests.
pub const AUTO_SEED_BUDGET: usize = 8;

// Horizontal size (metres) each auto-seeded cell aims to cover -- roughly a large
// room / courtyard, so a probe stays locally accurate across its cell. A 24 m
// square scene tiles into the same 2x2 grid the original auto-seed produced.
const AUTO_SEED_CELL_TARGET: f32 = 12.0;

// Voxels along the longest horizontal axis when probing for enclosed interior space;
// the voxel size derives from it. Bounds the detection cost (the grid is also capped
// per axis). Fine enough that a wall a metre or two thick still seals a room.
const INTERIOR_VOXELS_LONG_AXIS: usize = 48;
// Hard cap on voxels per axis so a huge scene cannot blow up the grid.
const INTERIOR_MAX_DIM: usize = 128;
// An empty voxel is "interior" only if at least this many of its six axis directions
// hit solid geometry before leaving the scene. Five captures a sealed or
// single-doorway room (floor + ceiling + three walls) and a fully walled courtyard
// (floor + four walls), while open ground (floor only) and simple overhangs stay
// exterior.
const INTERIOR_MIN_ENCLOSED: u8 = 5;
// Smallest interior region (in voxels) that earns a probe, so a one-voxel pocket
// wedged between props is not mistaken for a room.
const INTERIOR_MIN_CLUSTER: usize = 4;

// Pick grid dimensions `nx * nz <= budget` that stay close to the requested counts
// (so the grid keeps the scene's horizontal aspect). Scales both down by a common
// factor when over budget, then trims the larger dimension for any rounding spill.
fn fit_grid(nx: usize, nz: usize, budget: usize) -> (usize, usize) {
    let (mut nx, mut nz) = (nx.max(1), nz.max(1));
    if nx * nz > budget {
        let scale = (budget as f32 / (nx * nz) as f32).sqrt();
        nx = ((nx as f32 * scale).round() as usize).max(1);
        nz = ((nz as f32 * scale).round() as usize).max(1);
        while nx * nz > budget {
            if nx >= nz {
                nx -= 1;
            } else {
                nz -= 1;
            }
        }
    }
    (nx.max(1), nz.max(1))
}

// Whether a point lies inside any of the occupancy boxes (object world AABBs).
fn point_inside_any(p: [f32; 3], occupancy: &[([f32; 3], [f32; 3])]) -> bool {
    occupancy.iter().any(|(mn, mx)| {
        p[0] >= mn[0]
            && p[0] <= mx[0]
            && p[1] >= mn[1]
            && p[1] <= mx[1]
            && p[2] >= mn[2]
            && p[2] <= mx[2]
    })
}

// Choose a capture point inside the cell `[x0,x1] x [z0,z1]` (at the given eye
// height) that does not sit inside scene geometry. Prefers the cell centre; if that
// is occupied (e.g. inside a wall or building footprint at eye height), tries a few
// vantage points toward the cell's quarters and takes the first open one. Returns
// the centre unchanged when every candidate is occupied, so the result is never
// worse than the un-nudged grid.
fn open_capture_point(
    center: [f32; 3],
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    occupancy: &[([f32; 3], [f32; 3])],
) -> [f32; 3] {
    if !point_inside_any(center, occupancy) {
        return center;
    }
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    for (fx, fz) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
        let p = [lerp(x0, x1, fx), center[1], lerp(z0, z1, fz)];
        if !point_inside_any(p, occupancy) {
            return p;
        }
    }
    center
}

// The interior-detection voxel grid for a scene's bounds: one cube voxel size for
// every axis (so enclosure rays step uniformly), sized so the longest horizontal axis
// gets `INTERIOR_VOXELS_LONG_AXIS` voxels, each axis count capped at `INTERIOR_MAX_DIM`.
// Returns `(voxel_size, nx, ny, nz)`, or `None` for a degenerate (zero-area / zero-
// height) scene.
fn interior_voxel_grid(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> Option<(f32, usize, usize, usize)> {
    let extent = [
        aabb_max[0] - aabb_min[0],
        aabb_max[1] - aabb_min[1],
        aabb_max[2] - aabb_min[2],
    ];
    let long = extent[0].max(extent[2]);
    if long <= 0.0 || extent[1] <= 0.0 {
        return None;
    }
    let vs = (long / INTERIOR_VOXELS_LONG_AXIS as f32).max(0.25);
    let dim = |e: f32| ((e / vs).ceil() as usize).clamp(1, INTERIOR_MAX_DIM);
    Some((vs, dim(extent[0]), dim(extent[1]), dim(extent[2])))
}

// Mark every voxel any object AABB overlaps as solid (rasterise each box into the grid,
// clamped to bounds). Coarse: a watertight single mesh's AABB fills its own interior,
// so this cannot see the hollow -- `solid_from_triangles` is the per-surface answer.
fn solid_from_aabbs(
    aabb_min: [f32; 3],
    vs: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    occupancy: &[([f32; 3], [f32; 3])],
) -> Vec<bool> {
    let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let mut solid = vec![false; nx * ny * nz];
    let to_vx =
        |v: f32, origin: f32, hi: usize| (((v - origin) / vs).floor().max(0.0) as usize).min(hi);
    for (mn, mx) in occupancy {
        let x0 = to_vx(mn[0], aabb_min[0], nx - 1);
        let x1 = to_vx(mx[0], aabb_min[0], nx - 1);
        let y0 = to_vx(mn[1], aabb_min[1], ny - 1);
        let y1 = to_vx(mx[1], aabb_min[1], ny - 1);
        let z0 = to_vx(mn[2], aabb_min[2], nz - 1);
        let z1 = to_vx(mx[2], aabb_min[2], nz - 1);
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    solid[idx(x, y, z)] = true;
                }
            }
        }
    }
    solid
}

// Triangle / axis-aligned-box overlap by the separating-axis theorem (the
// Akenine-Moller "tribox" test, in its plain 13-axis form). `box_c` is the voxel
// centre, `box_h` its half-extent, `tri` a world-space triangle. The 13 candidate
// axes are the 3 box face normals, the triangle face normal, and the 9 edge x
// box-axis cross products; the pair is separated on an axis when the triangle's
// projection interval and the box's `[-r, r]` do not overlap. A degenerate (zero)
// axis projects everything to 0 and never separates, which is the correct no-op.
fn tri_box_overlap(box_c: [f32; 3], box_h: [f32; 3], tri: &[[f32; 3]; 3]) -> bool {
    let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    // Triangle in box-local space (box centred at the origin).
    let v = [sub(tri[0], box_c), sub(tri[1], box_c), sub(tri[2], box_c)];
    let edges = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    let box_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    let separated = |l: [f32; 3]| -> bool {
        let r = box_h[0] * l[0].abs() + box_h[1] * l[1].abs() + box_h[2] * l[2].abs();
        let p0 = dot(l, v[0]);
        let p1 = dot(l, v[1]);
        let p2 = dot(l, v[2]);
        p0.min(p1).min(p2) > r || p0.max(p1).max(p2) < -r
    };

    for a in box_axes {
        if separated(a) {
            return false;
        }
    }
    for e in edges {
        for a in box_axes {
            if separated(cross(e, a)) {
                return false;
            }
        }
    }
    !separated(cross(edges[0], edges[1]))
}

// Surface-voxelise the scene triangles: mark every voxel any triangle actually passes
// through (`tri_box_overlap`). Unlike the AABB rasteriser this leaves the hollow
// interior of a watertight single mesh empty -- exactly the case AABB occupancy cannot
// see -- so the enclosure sweep can find the room. Triangles are world-space; each is
// tested only against the voxels in its own (clamped) AABB.
fn solid_from_triangles(
    aabb_min: [f32; 3],
    vs: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    triangles: &[[[f32; 3]; 3]],
) -> Vec<bool> {
    let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    let mut solid = vec![false; nx * ny * nz];
    // Conservative voxelisation: inflate the test cell by a small voxel-relative margin
    // so a triangle lying exactly on a voxel boundary (an axis-aligned wall coplanar
    // with the grid) is still counted -- the exact SAT would FP-miss it and leave a gap
    // that lets the enclosure sweep leak. The margin is far below one voxel, so it never
    // reaches an interior cell.
    let half = [vs * 0.5 + vs * 1e-3; 3];
    let to_vx =
        |v: f32, origin: f32, hi: usize| (((v - origin) / vs).floor().max(0.0) as usize).min(hi);
    for tri in triangles {
        let mut tmn = tri[0];
        let mut tmx = tri[0];
        for vtx in &tri[1..] {
            for a in 0..3 {
                tmn[a] = tmn[a].min(vtx[a]);
                tmx[a] = tmx[a].max(vtx[a]);
            }
        }
        if !tmn.iter().chain(tmx.iter()).all(|c| c.is_finite()) {
            continue;
        }
        let x0 = to_vx(tmn[0], aabb_min[0], nx - 1);
        let x1 = to_vx(tmx[0], aabb_min[0], nx - 1);
        let y0 = to_vx(tmn[1], aabb_min[1], ny - 1);
        let y1 = to_vx(tmx[1], aabb_min[1], ny - 1);
        let z0 = to_vx(tmn[2], aabb_min[2], nz - 1);
        let z1 = to_vx(tmx[2], aabb_min[2], nz - 1);
        for z in z0..=z1 {
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let i = idx(x, y, z);
                    if solid[i] {
                        continue;
                    }
                    let c = [
                        aabb_min[0] + (x as f32 + 0.5) * vs,
                        aabb_min[1] + (y as f32 + 0.5) * vs,
                        aabb_min[2] + (z as f32 + 0.5) * vs,
                    ];
                    if tri_box_overlap(c, half, tri) {
                        solid[i] = true;
                    }
                }
            }
        }
    }
    solid
}

// Place a probe inside each enclosed interior region (a room or walled courtyard) given
// a precomputed solid-voxel grid. For every empty voxel counts how many of the six axis
// directions hit solid geometry before leaving the bounds (six O(voxels) sweeps); treats
// the well-enclosed empty voxels (`INTERIOR_MIN_ENCLOSED`) as interior; groups them into
// 6-connected regions; and drops one probe at the centre of each region big enough to be
// a room (`INTERIOR_MIN_CLUSTER`), largest first, up to `budget`. Returns empty for an
// open scene (everything reachable from the sky / sides). The `solid` grid comes from
// object AABBs (`seed_interior_probes`) or surface-voxelised triangles
// (`seed_interior_probes_tris`); everything from here on is identical.
fn interior_probes_from_solid(
    aabb_min: [f32; 3],
    vs: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    solid: &[bool],
    budget: usize,
) -> Vec<ProbePlacement> {
    let n = nx * ny * nz;
    let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;

    // For each empty voxel, count axis directions that hit solid before the grid
    // edge, via six linear sweeps (each marks "a solid lies ahead in this direction").
    let mut enclosed = vec![0u8; n];
    for z in 0..nz {
        for y in 0..ny {
            let (mut fwd, mut bwd) = (false, false);
            for x in (0..nx).rev() {
                let i = idx(x, y, z);
                if solid[i] {
                    fwd = true;
                } else if fwd {
                    enclosed[i] += 1;
                }
            }
            for x in 0..nx {
                let i = idx(x, y, z);
                if solid[i] {
                    bwd = true;
                } else if bwd {
                    enclosed[i] += 1;
                }
            }
        }
    }
    for z in 0..nz {
        for x in 0..nx {
            let (mut fwd, mut bwd) = (false, false);
            for y in (0..ny).rev() {
                let i = idx(x, y, z);
                if solid[i] {
                    fwd = true;
                } else if fwd {
                    enclosed[i] += 1;
                }
            }
            for y in 0..ny {
                let i = idx(x, y, z);
                if solid[i] {
                    bwd = true;
                } else if bwd {
                    enclosed[i] += 1;
                }
            }
        }
    }
    for y in 0..ny {
        for x in 0..nx {
            let (mut fwd, mut bwd) = (false, false);
            for z in (0..nz).rev() {
                let i = idx(x, y, z);
                if solid[i] {
                    fwd = true;
                } else if fwd {
                    enclosed[i] += 1;
                }
            }
            for z in 0..nz {
                let i = idx(x, y, z);
                if solid[i] {
                    bwd = true;
                } else if bwd {
                    enclosed[i] += 1;
                }
            }
        }
    }

    let is_interior = |i: usize| !solid[i] && enclosed[i] >= INTERIOR_MIN_ENCLOSED;

    // Group interior voxels into 6-connected regions (rooms).
    let mut label = vec![usize::MAX; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if !is_interior(start) || label[start] != usize::MAX {
            continue;
        }
        let cid = clusters.len();
        let mut members = Vec::new();
        label[start] = cid;
        stack.push(start);
        while let Some(i) = stack.pop() {
            members.push(i);
            let z = i / (nx * ny);
            let y = (i / nx) % ny;
            let x = i % nx;
            let neighbours = [
                (x > 0).then(|| i - 1),
                (x + 1 < nx).then_some(i + 1),
                (y > 0).then(|| i - nx),
                (y + 1 < ny).then_some(i + nx),
                (z > 0).then(|| i - nx * ny),
                (z + 1 < nz).then_some(i + nx * ny),
            ];
            for j in neighbours.into_iter().flatten() {
                if is_interior(j) && label[j] == usize::MAX {
                    label[j] = cid;
                    stack.push(j);
                }
            }
        }
        clusters.push(members);
    }

    // Largest rooms first; skip noise; one probe per room up to budget.
    clusters.retain(|c| c.len() >= INTERIOR_MIN_CLUSTER);
    clusters.sort_by_key(|c| std::cmp::Reverse(c.len()));
    clusters.truncate(budget);

    let voxel_center = |i: usize| {
        let z = i / (nx * ny);
        let y = (i / nx) % ny;
        let x = i % nx;
        [
            aabb_min[0] + (x as f32 + 0.5) * vs,
            aabb_min[1] + (y as f32 + 0.5) * vs,
            aabb_min[2] + (z as f32 + 0.5) * vs,
        ]
    };
    clusters
        .iter()
        .map(|members| {
            // Capture point: the member voxel nearest the region centroid (always an
            // interior voxel, so never inside geometry).
            let inv = 1.0 / members.len() as f32;
            let mut centroid = [0.0f32; 3];
            for &i in members {
                let c = voxel_center(i);
                for a in 0..3 {
                    centroid[a] += c[a] * inv;
                }
            }
            let dist2 = |c: [f32; 3]| {
                (c[0] - centroid[0]).powi(2)
                    + (c[1] - centroid[1]).powi(2)
                    + (c[2] - centroid[2]).powi(2)
            };
            let position = members
                .iter()
                .map(|&i| voxel_center(i))
                .min_by(|a, b| dist2(*a).total_cmp(&dist2(*b)))
                .unwrap_or(centroid);
            // Influence box: the region's voxel bounds, reaching half a voxel out to
            // the enclosing walls.
            let mut box_min = [f32::MAX; 3];
            let mut box_max = [f32::MIN; 3];
            for &i in members {
                let c = voxel_center(i);
                for a in 0..3 {
                    box_min[a] = box_min[a].min(c[a] - vs * 0.5);
                    box_max[a] = box_max[a].max(c[a] + vs * 0.5);
                }
            }
            ProbePlacement {
                position,
                box_min,
                box_max,
            }
        })
        .collect()
}

// Interior probes from object AABB occupancy (the coarse path): a watertight single
// mesh's AABB fills its own interior, so only enclosures built from SEPARATE meshes
// (walls / floor / ceiling / props -- the common case) are detected.
fn seed_interior_probes(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    occupancy: &[([f32; 3], [f32; 3])],
    budget: usize,
) -> Vec<ProbePlacement> {
    if budget == 0 || occupancy.is_empty() {
        return Vec::new();
    }
    let (vs, nx, ny, nz) = match interior_voxel_grid(aabb_min, aabb_max) {
        Some(g) => g,
        None => return Vec::new(),
    };
    let solid = solid_from_aabbs(aabb_min, vs, nx, ny, nz, occupancy);
    interior_probes_from_solid(aabb_min, vs, nx, ny, nz, &solid, budget)
}

// Interior probes from surface-voxelised triangles (the fine path): marks only the
// voxels a triangle actually passes through, so a watertight single mesh reads as a
// hollow shell and its interior room is detected -- the case AABB occupancy misses.
fn seed_interior_probes_tris(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    triangles: &[[[f32; 3]; 3]],
    budget: usize,
) -> Vec<ProbePlacement> {
    if budget == 0 || triangles.is_empty() {
        return Vec::new();
    }
    let (vs, nx, ny, nz) = match interior_voxel_grid(aabb_min, aabb_max) {
        Some(g) => g,
        None => return Vec::new(),
    };
    let solid = solid_from_triangles(aabb_min, vs, nx, ny, nz, triangles);
    interior_probes_from_solid(aabb_min, vs, nx, ny, nz, &solid, budget)
}

// Auto-seed probes from the scene bounds, used when a world declares no
// `ReflectionProbe`. Convenience wrapper over `auto_seed_probes_with_geometry` with no
// triangle geometry: interior detection uses object AABB occupancy (the coarse path).
pub fn auto_seed_probes(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    occupancy: &[([f32; 3], [f32; 3])],
) -> Vec<ProbePlacement> {
    auto_seed_probes_with_geometry(aabb_min, aabb_max, occupancy, &[])
}

// Auto-seed probes from the scene bounds. First places a probe INSIDE each enclosed
// interior region (a room / walled courtyard, where a local capture is most valuable),
// then fills the remaining `AUTO_SEED_BUDGET` with a `seed_grid_probes` grid for broad /
// open coverage. An open scene finds no interiors, so it falls straight through to the
// grid (unchanged); a fully-enclosed scene is mostly rooms; a mixed scene gets both,
// cross-faded by the partition-of-unity blend. When `triangles` is non-empty, interior
// detection surface-voxelises them (so a watertight single mesh's hollow is found);
// when empty, it falls back to the coarse AABB `occupancy`. The grid fill always uses
// the AABB `occupancy` for its open-vantage capture-point nudge. Approximate --
// authored probes give per-space control -- but better than one global cube. Returns
// empty for a degenerate (non-finite or zero-area) scene.
pub fn auto_seed_probes_with_geometry(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    occupancy: &[([f32; 3], [f32; 3])],
    triangles: &[[[f32; 3]; 3]],
) -> Vec<ProbePlacement> {
    let finite = aabb_min
        .iter()
        .chain(aabb_max.iter())
        .all(|c| c.is_finite());
    if !finite || aabb_max[0] <= aabb_min[0] || aabb_max[2] <= aabb_min[2] {
        return Vec::new();
    }
    let mut out = if triangles.is_empty() {
        seed_interior_probes(aabb_min, aabb_max, occupancy, AUTO_SEED_BUDGET)
    } else {
        seed_interior_probes_tris(aabb_min, aabb_max, triangles, AUTO_SEED_BUDGET)
    };
    let remaining = AUTO_SEED_BUDGET.saturating_sub(out.len());
    if remaining > 0 {
        out.extend(seed_grid_probes(aabb_min, aabb_max, occupancy, remaining));
    }
    out
}

// The grid half of auto-seed: tile the scene's horizontal extent into at most
// `budget` cells (sized to `AUTO_SEED_CELL_TARGET`, shaped to the aspect via
// `fit_grid`), each owning its full-height column as the influence box, capture point
// = the cell centre nudged to an open vantage at eye height (`occupancy` = the scene's
// object AABBs) so a probe is not captured from inside a wall.
fn seed_grid_probes(
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    occupancy: &[([f32; 3], [f32; 3])],
    budget: usize,
) -> Vec<ProbePlacement> {
    if budget == 0 {
        return Vec::new();
    }
    let dx = aabb_max[0] - aabb_min[0];
    let dz = aabb_max[2] - aabb_min[2];
    let nx_raw = (dx / AUTO_SEED_CELL_TARGET).ceil().max(1.0) as usize;
    let nz_raw = (dz / AUTO_SEED_CELL_TARGET).ceil().max(1.0) as usize;
    let (nx, nz) = fit_grid(nx_raw, nz_raw, budget);

    let y_eye = probe_eye_point(aabb_min, aabb_max)[1];
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let mut out = Vec::with_capacity(nx * nz);
    for ix in 0..nx {
        for iz in 0..nz {
            let x0 = lerp(aabb_min[0], aabb_max[0], ix as f32 / nx as f32);
            let x1 = lerp(aabb_min[0], aabb_max[0], (ix + 1) as f32 / nx as f32);
            let z0 = lerp(aabb_min[2], aabb_max[2], iz as f32 / nz as f32);
            let z1 = lerp(aabb_min[2], aabb_max[2], (iz + 1) as f32 / nz as f32);
            let center = [(x0 + x1) * 0.5, y_eye, (z0 + z1) * 0.5];
            out.push(ProbePlacement {
                position: open_capture_point(center, x0, x1, z0, z1, occupancy),
                box_min: [x0, aabb_min[1], z0],
                box_max: [x1, aabb_max[1], z1],
            });
        }
    }
    out
}

// Union the world-space AABBs of every scene object into one bounds, skipping
// any box with a non-finite corner (a degenerate / sentinel AABB). Returns
// `None` for an empty scene. The probe eye is then `probe_eye_point` of this.
#[allow(dead_code)]
pub fn fold_world_bounds(
    boxes: impl IntoIterator<Item = ([f32; 3], [f32; 3])>,
) -> Option<([f32; 3], [f32; 3])> {
    let mut acc: Option<([f32; 3], [f32; 3])> = None;
    for (mn, mx) in boxes {
        if !mn.iter().chain(mx.iter()).all(|c| c.is_finite()) {
            continue;
        }
        match &mut acc {
            None => acc = Some((mn, mx)),
            Some((amn, amx)) => {
                for i in 0..3 {
                    amn[i] = amn[i].min(mn[i]);
                    amx[i] = amx[i].max(mx[i]);
                }
            }
        }
    }
    acc
}

// Convert six captured cube faces (each `face_size*face_size` RGBA `f32`, row
// major, in the FACE_BASIS order) into the serialised `ENVM` payload the
// environment sampler consumes: a cosine-convolved irradiance cube + a GGX
// prefilter mip chain. Reuses the exact build-time convolutions (including the
// firefly clamp), so a scene-captured probe and an imported HDR produce
// byte-compatible payloads that flow through the same `upload_environment_map`.
#[allow(dead_code)]
pub fn build_probe_payload(
    faces: &[Vec<f32>; 6],
    face_size: u32,
    irradiance_face: u32,
    prefilter_samples: u32,
    prefilter_clamp: f32,
) -> Vec<u8> {
    use crate::build::environment_map as em;
    let mips = em::max_mip_count(face_size);
    let irradiance = em::compute_irradiance(
        faces,
        face_size,
        irradiance_face,
        em::DEFAULT_IRRADIANCE_PHI_SAMPLES,
        em::DEFAULT_IRRADIANCE_THETA_SAMPLES,
    );
    // A probe cube is sampled only by the specular term (never drawn as a skybox), so
    // clamp mip 0 too: it suppresses a lone blown highlight aliasing into a bright
    // square on a near-mirror surface that falls back to the probe (SSR/RT miss).
    let prefilter = em::compute_prefilter(
        faces,
        face_size,
        mips,
        prefilter_samples,
        prefilter_clamp,
        true,
    );
    em::serialise_payload(irradiance_face, face_size, mips, &irradiance, &prefilter)
}

// Pick the eye point a single scene probe captures from: the horizontal centre
// of the scene bounds, raised to eye height above the floor. A probe serves a
// volume rather than a viewpoint, so centring it degrades most gracefully as a
// first-person camera roams (the captured cube is still parallax-locked to this
// point until box parallax correction lands). Kept pure so an authored probe
// position can later replace this heuristic. `aabb_min`/`aabb_max` are the world
// bounds; +Y is up.
#[allow(dead_code)]
pub fn probe_eye_point(aabb_min: [f32; 3], aabb_max: [f32; 3]) -> [f32; 3] {
    const EYE_HEIGHT: f32 = 1.7;
    let cx = 0.5 * (aabb_min[0] + aabb_max[0]);
    let cz = 0.5 * (aabb_min[2] + aabb_max[2]);
    let floor = aabb_min[1];
    let ceil = aabb_max[1];
    // Eye height off the floor, but never above the scene's own mid-height (so a
    // scene shorter than a person still places the probe inside its bounds).
    let y = (floor + EYE_HEIGHT).min(0.5 * (floor + ceil)).max(floor);
    [cx, y, cz]
}

#[cfg(test)]
mod tests {
    use super::*;

    // The engine's cube convention (mirrors core `cube_texel_dir`): a face texel
    // at (u, v) in [-1, 1] looks along this world direction.
    fn cube_texel_dir(face: usize, u: f32, v: f32) -> [f32; 3] {
        match face {
            0 => [1.0, -v, -u],
            1 => [-1.0, -v, u],
            2 => [u, 1.0, v],
            3 => [u, -1.0, -v],
            4 => [u, -v, 1.0],
            5 => [-u, -v, -1.0],
            _ => unreachable!(),
        }
    }

    fn project(vp: [[f32; 4]; 4], p: [f32; 3]) -> (f32, f32, f32) {
        let mut c = [0.0f32; 4];
        let pv = [p[0], p[1], p[2], 1.0];
        for row in 0..4 {
            for k in 0..4 {
                c[row] += vp[k][row] * pv[k];
            }
        }
        (c[0] / c[3], c[1] / c[3], c[3])
    }

    // Each face's view-projection must map cube_texel_dir(face, u, v) to NDC
    // (u, -v): screen-right is +u, screen-down is +v, exactly the layout the
    // readback stores and the prefilter samples. A flipped or rotated face
    // would break this and is the classic cube-capture bug.
    #[test]
    fn face_view_projection_matches_cube_convention() {
        let eye = [3.0, -1.5, 2.0];
        let samples = [
            (0.0f32, 0.0f32),
            (0.5, 0.0),
            (0.0, 0.5),
            (-0.6, 0.3),
            (0.7, -0.4),
        ];
        for face in 0..6 {
            let vp = face_view_projection(eye, face, 0.05, 100.0);
            for &(u, v) in &samples {
                let d = cube_texel_dir(face, u, v);
                let p = [eye[0] + d[0], eye[1] + d[1], eye[2] + d[2]];
                let (nx, ny, w) = project(vp, p);
                assert!(
                    w > 0.0,
                    "face {face} sample ({u},{v}) behind camera (w={w})"
                );
                assert!(
                    (nx - u).abs() < 1e-4 && (ny - (-v)).abs() < 1e-4,
                    "face {face} ({u},{v}) -> ndc ({nx},{ny}), expected ({u},{})",
                    -v
                );
            }
        }
    }

    #[test]
    fn build_probe_payload_round_trips() {
        // Six small solid faces convolve into a valid ENVM payload that
        // deserialises with the requested sizes.
        let face = 8usize;
        let faces: [Vec<f32>; 6] = std::array::from_fn(|f| {
            let mut v = vec![0.0f32; face * face * 4];
            for px in v.chunks_exact_mut(4) {
                px[0] = f as f32 * 0.1;
                px[1] = 0.2;
                px[2] = 0.3;
                px[3] = 1.0;
            }
            v
        });
        let bytes = build_probe_payload(&faces, face as u32, 8, 16, 12.0);
        let view = crate::build::environment_map::deserialise(&bytes).expect("deserialise");
        assert_eq!(view.prefilter_face, 8);
        assert_eq!(view.irradiance_face, 8);
        assert!(view.prefilter_mips >= 2);
    }

    #[test]
    fn probe_eye_point_centres_at_eye_height() {
        // A tall scene: probe sits at the horizontal centre, eye height off the
        // floor.
        let eye = probe_eye_point([-10.0, 0.0, -4.0], [6.0, 30.0, 12.0]);
        assert!((eye[0] - (-2.0)).abs() < 1e-6, "x not centred: {}", eye[0]);
        assert!((eye[2] - 4.0).abs() < 1e-6, "z not centred: {}", eye[2]);
        assert!((eye[1] - 1.7).abs() < 1e-6, "y not eye height: {}", eye[1]);
    }

    #[test]
    fn probe_eye_point_clamps_to_a_flat_scene() {
        // A scene shorter than a person clamps the probe inside its own bounds.
        let eye = probe_eye_point([0.0, 0.0, 0.0], [2.0, 1.0, 2.0]);
        assert!(
            eye[1] >= 0.0 && eye[1] <= 1.0,
            "y escaped bounds: {}",
            eye[1]
        );
    }

    #[test]
    fn face_view_matrix_composes_to_face_vp() {
        // The exposed bare view matrix, pre-multiplied by the 90-degree
        // projection, must reproduce `face_view_projection` exactly (the probe
        // capture builds ViewUniforms from the two separately).
        let eye = [1.0, 2.0, -3.0];
        for face in 0..6 {
            let vp = face_view_projection(eye, face, 0.1, 50.0);
            let comp = mul(perspective_90(0.1, 50.0), face_view_matrix(eye, face));
            for c in 0..4 {
                for r in 0..4 {
                    assert!(
                        (vp[c][r] - comp[c][r]).abs() < 1e-5,
                        "face {face} [{c}][{r}] mismatch"
                    );
                }
            }
        }
    }

    #[test]
    fn placement_from_center_extents_builds_box() {
        let p = ProbePlacement::from_center_extents([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert_eq!(p.box_min, [-3.0, -3.0, -3.0]);
        assert_eq!(p.box_max, [5.0, 7.0, 9.0]);
        assert_eq!(p.position, [1.0, 2.0, 3.0]);
    }

    // Union of every probe's influence box.
    fn probe_union(probes: &[ProbePlacement]) -> ([f32; 3], [f32; 3]) {
        let mn = probes.iter().fold([f32::MAX; 3], |a, p| {
            std::array::from_fn(|i| a[i].min(p.box_min[i]))
        });
        let mx = probes.iter().fold([f32::MIN; 3], |a, p| {
            std::array::from_fn(|i| a[i].max(p.box_max[i]))
        });
        (mn, mx)
    }

    #[test]
    fn auto_seed_probes_tiles_the_scene() {
        // A 20 m square scene seeds a 2x2 grid whose boxes tile the full extent.
        let probes = auto_seed_probes([-10.0, 0.0, -10.0], [10.0, 6.0, 10.0], &[]);
        assert_eq!(probes.len(), 4);
        let (union_min, union_max) = probe_union(&probes);
        assert_eq!(union_min, [-10.0, 0.0, -10.0]);
        assert_eq!(union_max, [10.0, 6.0, 10.0]);
        // Degenerate scenes seed nothing.
        assert!(auto_seed_probes([0.0; 3], [0.0; 3], &[]).is_empty());
        assert!(auto_seed_probes([f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0], &[]).is_empty());
    }

    #[test]
    fn auto_seed_scales_count_to_scene_size_and_aspect() {
        // A small scene (under one cell each way) seeds a single probe, not a
        // redundant 2x2.
        let small = auto_seed_probes([-3.0, 0.0, -3.0], [3.0, 3.0, 3.0], &[]);
        assert_eq!(small.len(), 1);
        // An elongated scene gets an elongated grid (more cells along the long
        // axis), and the boxes always tile the full extent.
        let long = auto_seed_probes([0.0, 0.0, 0.0], [96.0, 4.0, 12.0], &[]);
        assert!(long.len() > 1 && long.len() <= AUTO_SEED_BUDGET);
        let nx = long.iter().filter(|p| p.box_min[2] == 0.0).count();
        let nz = long.len() / nx;
        assert!(nx > nz, "long axis (x) should have more cells: {nx}x{nz}");
        let (mn, mx) = probe_union(&long);
        assert_eq!(mn, [0.0, 0.0, 0.0]);
        assert_eq!(mx, [96.0, 4.0, 12.0]);
        // A large scene is capped at the budget.
        let big = auto_seed_probes([0.0, 0.0, 0.0], [500.0, 4.0, 500.0], &[]);
        assert!(big.len() <= AUTO_SEED_BUDGET);
    }

    #[test]
    fn auto_seed_nudges_capture_point_out_of_geometry() {
        // One probe (small scene), with a wall-like box covering the cell centre at
        // eye height. The capture point must move out of it but stay in the box.
        let occ = [([-1.0, 0.0, -1.0], [1.0, 5.0, 1.0])];
        let probes = auto_seed_probes([-3.0, 0.0, -3.0], [3.0, 3.0, 3.0], &occ);
        assert_eq!(probes.len(), 1);
        let p = probes[0].position;
        assert!(
            !point_inside_any(p, &occ),
            "capture point {p:?} still inside the occupancy box"
        );
        // Still within the probe's influence box.
        assert!(p[0] >= probes[0].box_min[0] && p[0] <= probes[0].box_max[0]);
        assert!(p[2] >= probes[0].box_min[2] && p[2] <= probes[0].box_max[2]);
        // When every candidate is occupied, the centre is kept (never drops a probe).
        let everywhere = [([-100.0, -100.0, -100.0], [100.0, 100.0, 100.0])];
        let trapped = auto_seed_probes([-3.0, 0.0, -3.0], [3.0, 3.0, 3.0], &everywhere);
        assert_eq!(trapped.len(), 1);
    }

    #[test]
    fn fit_grid_respects_budget_and_aspect() {
        assert_eq!(fit_grid(1, 1, 8), (1, 1));
        assert_eq!(fit_grid(2, 2, 8), (2, 2)); // within budget, unchanged
        let (nx, nz) = fit_grid(9, 2, 8); // over budget, x-heavy
        assert!(nx * nz <= 8 && nx > nz);
        let (nx, nz) = fit_grid(20, 20, 8); // far over budget, square
        assert!(nx * nz <= 8 && nx >= 1 && nz >= 1);
    }

    // The six wall slabs (thickness 1 m) of an axis-aligned room with the given
    // interior bounds, as object occupancy AABBs.
    fn box_room(min: [f32; 3], max: [f32; 3]) -> Vec<([f32; 3], [f32; 3])> {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        vec![
            ([x0, y0 - 1.0, z0], [x1, y0, z1]), // floor
            ([x0, y1, z0], [x1, y1 + 1.0, z1]), // ceiling
            ([x0 - 1.0, y0, z0], [x0, y1, z1]), // -x wall
            ([x1, y0, z0], [x1 + 1.0, y1, z1]), // +x wall
            ([x0, y0, z0 - 1.0], [x1, y1, z0]), // -z wall
            ([x0, y0, z1], [x1, y1, z1 + 1.0]), // +z wall
        ]
    }

    #[test]
    fn seed_interior_probes_finds_a_sealed_room() {
        // A 10x6x10 room inside a larger scene: the only enclosed space is its
        // interior, so exactly one probe lands inside it.
        let room = box_room([0.0, 0.0, 0.0], [10.0, 6.0, 10.0]);
        let probes = seed_interior_probes([-3.0, -3.0, -3.0], [13.0, 9.0, 13.0], &room, 8);
        assert_eq!(probes.len(), 1, "one room -> one interior probe");
        let p = probes[0].position;
        assert!(
            p[0] > 0.0 && p[0] < 10.0 && p[1] > 0.0 && p[1] < 6.0 && p[2] > 0.0 && p[2] < 10.0,
            "probe {p:?} should sit inside the room"
        );
        // The influence box covers (most of) the room interior.
        assert!(probes[0].box_min[0] < 2.0 && probes[0].box_max[0] > 8.0);
    }

    #[test]
    fn seed_interior_probes_ignores_an_open_scene() {
        // A ground slab plus two free-standing pillars: nothing encloses a volume, so
        // no interior probe is seeded (the caller falls back to the grid).
        let open = vec![
            ([-20.0, -1.0, -20.0], [20.0, 0.0, 20.0]), // ground
            ([-5.0, 0.0, -5.0], [-3.0, 6.0, -3.0]),    // pillar
            ([3.0, 0.0, 3.0], [5.0, 6.0, 5.0]),        // pillar
        ];
        let probes = seed_interior_probes([-20.0, -1.0, -20.0], [20.0, 8.0, 20.0], &open, 8);
        assert!(
            probes.is_empty(),
            "open scene seeds no interior probes: {probes:?}"
        );
        // No occupancy at all is likewise empty.
        assert!(seed_interior_probes([0.0, 0.0, 0.0], [10.0, 5.0, 10.0], &[], 8).is_empty());
    }

    #[test]
    fn auto_seed_places_a_room_probe_then_grid() {
        // A room inside the scene -> at least one probe inside it, plus grid fill for
        // the open remainder, all within budget.
        let room = box_room([0.0, 0.0, 0.0], [10.0, 6.0, 10.0]);
        let probes = auto_seed_probes([-12.0, -3.0, -12.0], [22.0, 9.0, 22.0], &room);
        assert!(!probes.is_empty() && probes.len() <= AUTO_SEED_BUDGET);
        let inside_room = probes.iter().any(|p| {
            let q = p.position;
            q[0] > 0.0 && q[0] < 10.0 && q[1] > 0.0 && q[1] < 6.0 && q[2] > 0.0 && q[2] < 10.0
        });
        assert!(
            inside_room,
            "auto-seed should drop a probe inside the room: {probes:?}"
        );
    }

    // 12 triangles (2 per face) of the closed axis-aligned box [min, max]: a watertight
    // single mesh whose AABB is the whole box, but whose triangles only cover the shell.
    fn box_mesh_tris(min: [f32; 3], max: [f32; 3]) -> Vec<[[f32; 3]; 3]> {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        let c = [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ];
        // Each face as four corner indices (a,b,c,d) -> triangles (a,b,c) + (a,c,d).
        let quads = [
            [0, 1, 2, 3], // -z
            [4, 5, 6, 7], // +z
            [0, 3, 7, 4], // -x
            [1, 2, 6, 5], // +x
            [0, 1, 5, 4], // -y
            [3, 2, 6, 7], // +y
        ];
        let mut tris = Vec::with_capacity(12);
        for q in quads {
            tris.push([c[q[0]], c[q[1]], c[q[2]]]);
            tris.push([c[q[0]], c[q[2]], c[q[3]]]);
        }
        tris
    }

    #[test]
    fn tri_box_overlap_detects_intersection_and_separation() {
        let h = [0.5, 0.5, 0.5];
        // A triangle straddling the origin overlaps a unit box centred there.
        let through = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(tri_box_overlap([0.0, 0.0, 0.0], h, &through));
        // Separated from a box well off to the side (a box face-axis separates).
        assert!(!tri_box_overlap([10.0, 0.0, 0.0], h, &through));
        // A horizontal triangle high above is separated by the triangle's own normal
        // axis (the box sits entirely on one side of the triangle's plane).
        let above = [[-1.0, 5.0, -1.0], [3.0, 5.0, -1.0], [0.0, 5.0, 3.0]];
        assert!(!tri_box_overlap([0.0, 0.0, 0.0], h, &above));
        // A tiny triangle fully inside the box overlaps it.
        let inside = [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]];
        assert!(tri_box_overlap([0.0, 0.0, 0.0], h, &inside));
    }

    #[test]
    fn surface_voxels_leave_a_watertight_mesh_hollow() {
        // The exact case AABB occupancy misses: a room modelled as ONE watertight mesh.
        // Its single AABB fills the interior (no room found), but surface-voxelising its
        // triangles leaves the interior empty so the enclosure sweep finds the room.
        let scene_min = [-3.0, -3.0, -3.0];
        let scene_max = [13.0, 9.0, 13.0];
        let room_aabb = vec![([0.0, 0.0, 0.0], [10.0, 6.0, 10.0])];
        let room_tris = box_mesh_tris([0.0, 0.0, 0.0], [10.0, 6.0, 10.0]);

        // AABB occupancy: the solid box has no hollow, so no interior probe.
        let from_aabb = seed_interior_probes(scene_min, scene_max, &room_aabb, 8);
        assert!(
            from_aabb.is_empty(),
            "a watertight mesh's AABB hides its interior: {from_aabb:?}"
        );

        // Triangle occupancy: the shell is hollow, so one probe lands inside the room.
        let from_tris = seed_interior_probes_tris(scene_min, scene_max, &room_tris, 8);
        assert_eq!(from_tris.len(), 1, "the hollow interior earns one probe");
        let p = from_tris[0].position;
        assert!(
            p[0] > 0.0 && p[0] < 10.0 && p[1] > 0.0 && p[1] < 6.0 && p[2] > 0.0 && p[2] < 10.0,
            "probe {p:?} should sit inside the watertight room"
        );

        // The public geometry entry routes through the surface-voxel path.
        let auto = auto_seed_probes_with_geometry(scene_min, scene_max, &room_aabb, &room_tris);
        assert!(
            auto.iter().any(|q| {
                let q = q.position;
                q[0] > 0.0 && q[0] < 10.0 && q[1] > 0.0 && q[1] < 6.0 && q[2] > 0.0 && q[2] < 10.0
            }),
            "auto-seed with geometry drops a probe inside the room: {auto:?}"
        );
        // Empty geometry is exactly the coarse AABB path (back-compatible delegation).
        assert_eq!(
            auto_seed_probes_with_geometry(scene_min, scene_max, &room_aabb, &[]).len(),
            auto_seed_probes(scene_min, scene_max, &room_aabb).len(),
        );
    }

    #[test]
    fn fold_world_bounds_unions_and_skips_nonfinite() {
        let boxes = [
            ([0.0, 0.0, 0.0], [1.0, 2.0, 1.0]),
            ([-3.0, 1.0, -1.0], [0.5, 4.0, 2.0]),
            ([f32::NAN, 0.0, 0.0], [1.0, 1.0, 1.0]), // degenerate: skipped
        ];
        let (mn, mx) = fold_world_bounds(boxes).expect("non-empty");
        assert_eq!(mn, [-3.0, 0.0, -1.0]);
        assert_eq!(mx, [1.0, 4.0, 2.0]);
        assert!(fold_world_bounds(std::iter::empty()).is_none());
    }

    #[test]
    fn face_centres_look_down_their_axis() {
        // The centre texel of each face projects to the NDC origin.
        let eye = [0.0, 0.0, 0.0];
        for face in 0..6 {
            let vp = face_view_projection(eye, face, 0.05, 100.0);
            let d = cube_texel_dir(face, 0.0, 0.0);
            let (nx, ny, w) = project(vp, d);
            assert!(w > 0.0);
            assert!(
                nx.abs() < 1e-5 && ny.abs() < 1e-5,
                "face {face} centre off-origin"
            );
        }
    }

    #[test]
    fn bake_queue_hands_out_indices_in_order() {
        let mut q = ProbeBakeQueue::new(3);
        assert!(q.pending());
        assert_eq!(q.take_next(), Some(0));
        assert_eq!(q.take_next(), Some(1));
        assert!(q.pending());
        assert_eq!(q.take_next(), Some(2));
        assert!(!q.pending());
        assert_eq!(q.take_next(), None);
    }

    #[test]
    fn bake_queue_empty_is_never_pending() {
        let mut q = ProbeBakeQueue::new(0);
        assert!(!q.pending());
        assert_eq!(q.take_next(), None);
    }

    #[test]
    fn bake_queue_abort_skips_the_remainder() {
        let mut q = ProbeBakeQueue::new(4);
        assert_eq!(q.take_next(), Some(0));
        q.abort();
        assert!(!q.pending());
        assert_eq!(q.take_next(), None);
    }

    #[test]
    fn bake_action_idle_starts_only_when_pending_and_eligible() {
        // Idle with work waiting and the world able to bake -> begin.
        assert_eq!(
            next_bake_action(BakePhase::Idle, false, false, true, true, false),
            BakeAction::StartNext
        );
        // Idle but nothing queued -> stay put.
        assert_eq!(
            next_bake_action(BakePhase::Idle, false, false, false, true, false),
            BakeAction::Idle
        );
        // Idle with work queued but the world not eligible this frame (e.g. the
        // cull is still empty while geometry streams in) -> wait, do not start.
        assert_eq!(
            next_bake_action(BakePhase::Idle, false, false, true, false, false),
            BakeAction::Idle
        );
    }

    #[test]
    fn bake_action_rendering_submits_faces_before_waiting_for_completion() {
        // While faces remain, submit the next one (one per frame) -- even if `done`
        // somehow read true, more_faces takes precedence so the capture finishes.
        assert_eq!(
            next_bake_action(BakePhase::Rendering, false, false, true, true, true),
            BakeAction::RenderFace
        );
        // All six submitted, GPU not done yet -> wait.
        assert_eq!(
            next_bake_action(BakePhase::Rendering, false, false, true, true, false),
            BakeAction::Idle
        );
        // All six submitted and the GPU completion flag set -> read them back.
        assert_eq!(
            next_bake_action(BakePhase::Rendering, true, false, true, true, false),
            BakeAction::Readback
        );
    }

    #[test]
    fn bake_action_converting_waits_for_offthread_payload() {
        // The off-thread convolution gates the install: never upload early.
        assert_eq!(
            next_bake_action(BakePhase::Converting, true, false, true, true, false),
            BakeAction::Idle
        );
        assert_eq!(
            next_bake_action(BakePhase::Converting, true, true, false, true, false),
            BakeAction::Install
        );
    }

    #[test]
    fn bake_action_never_starts_a_second_bake_while_one_is_in_flight() {
        // With a probe rendering or converting, a pending queue must not trigger a
        // second StartNext regardless of eligibility -- one bake in flight at a time.
        for phase in [BakePhase::Rendering, BakePhase::Converting] {
            assert_ne!(
                next_bake_action(phase, false, false, true, true, false),
                BakeAction::StartNext
            );
            assert_ne!(
                next_bake_action(phase, false, false, true, true, true),
                BakeAction::StartNext
            );
        }
    }
}
