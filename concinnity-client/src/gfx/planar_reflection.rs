// Planar reflection math: mirror a camera across a world plane and oblique-clip
// the projection so geometry behind the plane never leaks into the reflection.
//
// Backend-agnostic and pure (mirrors gfx/reflection_probe.rs). A reflective flat
// surface (water, a mirror floor) renders the scene a second time from the
// camera reflected across its plane; the reflective surface then samples that
// render projectively. This module produces the matrices that pass needs.
//
// Conventions match metal/math.rs: column-major storage `m[col][row]`, a
// right-handed view looking down -z, and a perspective projection mapping depth
// to [0, 1] (Metal / D3D clip space). A plane is `[nx, ny, nz, d]` with `n`
// unit-length, satisfying `n . p + d = 0` for points on it; `n . p + d > 0` is
// the side the normal points toward.

type Mat4 = [[f32; 4]; 4];
type Vec4 = [f32; 4];

fn mul(a: Mat4, b: Mat4) -> Mat4 {
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

fn mat_vec(m: Mat4, v: Vec4) -> Vec4 {
    let mut out = [0.0f32; 4];
    for row in 0..4 {
        for k in 0..4 {
            out[row] += m[k][row] * v[k];
        }
    }
    out
}

fn transpose(m: Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            out[col][row] = m[row][col];
        }
    }
    out
}

fn dot4(a: Vec4, b: Vec4) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

// General 4x4 inverse via cofactor expansion (column-major in/out). Returns the
// identity for a singular input. Mirrors metal/math.rs `mat4_inverse` so the
// reflected-view plane transform stays backend-agnostic.
fn inverse(m: Mat4) -> Mat4 {
    let a00 = m[0][0];
    let a01 = m[1][0];
    let a02 = m[2][0];
    let a03 = m[3][0];
    let a10 = m[0][1];
    let a11 = m[1][1];
    let a12 = m[2][1];
    let a13 = m[3][1];
    let a20 = m[0][2];
    let a21 = m[1][2];
    let a22 = m[2][2];
    let a23 = m[3][2];
    let a30 = m[0][3];
    let a31 = m[1][3];
    let a32 = m[2][3];
    let a33 = m[3][3];

    let b00 = a00 * a11 - a01 * a10;
    let b01 = a00 * a12 - a02 * a10;
    let b02 = a00 * a13 - a03 * a10;
    let b03 = a01 * a12 - a02 * a11;
    let b04 = a01 * a13 - a03 * a11;
    let b05 = a02 * a13 - a03 * a12;
    let b06 = a20 * a31 - a21 * a30;
    let b07 = a20 * a32 - a22 * a30;
    let b08 = a20 * a33 - a23 * a30;
    let b09 = a21 * a32 - a22 * a31;
    let b10 = a21 * a33 - a23 * a31;
    let b11 = a22 * a33 - a23 * a32;

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if det.abs() < 1e-20 {
        return [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
    }
    let inv_det = 1.0 / det;

    // Row-major inverse entries, re-emitted column-major (out[col][row]).
    let i00 = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
    let i01 = (-a01 * b11 + a02 * b10 - a03 * b09) * inv_det;
    let i02 = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
    let i03 = (-a21 * b05 + a22 * b04 - a23 * b03) * inv_det;
    let i10 = (-a10 * b11 + a12 * b08 - a13 * b07) * inv_det;
    let i11 = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
    let i12 = (-a30 * b05 + a32 * b02 - a33 * b01) * inv_det;
    let i13 = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;
    let i20 = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
    let i21 = (-a00 * b10 + a01 * b08 - a03 * b06) * inv_det;
    let i22 = (a30 * b04 - a31 * b02 + a33 * b00) * inv_det;
    let i23 = (-a20 * b04 + a21 * b02 - a23 * b00) * inv_det;
    let i30 = (-a10 * b09 + a11 * b07 - a12 * b06) * inv_det;
    let i31 = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
    let i32 = (-a30 * b03 + a31 * b01 - a32 * b00) * inv_det;
    let i33 = (a20 * b03 - a21 * b01 + a22 * b00) * inv_det;

    [
        [i00, i10, i20, i30],
        [i01, i11, i21, i31],
        [i02, i12, i22, i32],
        [i03, i13, i23, i33],
    ]
}

// Normalise a plane so its normal is unit length (scaling d to match). A zero
// normal is returned unchanged (degenerate, callers guard separately).
pub(crate) fn normalize_plane(plane: Vec4) -> Vec4 {
    let len = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
    if len < 1e-12 {
        return plane;
    }
    let inv = 1.0 / len;
    [
        plane[0] * inv,
        plane[1] * inv,
        plane[2] * inv,
        plane[3] * inv,
    ]
}

// Householder reflection of world points across `plane` (unit normal). A point
// p maps to p - 2 (n.p + d) n; this 4x4 applies that to homogeneous points.
pub(crate) fn reflection_matrix(plane: Vec4) -> Mat4 {
    let [nx, ny, nz, d] = plane;
    [
        [1.0 - 2.0 * nx * nx, -2.0 * ny * nx, -2.0 * nz * nx, 0.0],
        [-2.0 * nx * ny, 1.0 - 2.0 * ny * ny, -2.0 * nz * ny, 0.0],
        [-2.0 * nx * nz, -2.0 * ny * nz, 1.0 - 2.0 * nz * nz, 0.0],
        [-2.0 * nx * d, -2.0 * ny * d, -2.0 * nz * d, 1.0],
    ]
}

// Reflect a single world point across the plane (the reflected camera eye).
pub(crate) fn reflect_point(p: [f32; 3], plane: Vec4) -> [f32; 3] {
    let dist = plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3];
    [
        p[0] - 2.0 * dist * plane[0],
        p[1] - 2.0 * dist * plane[1],
        p[2] - 2.0 * dist * plane[2],
    ]
}

// The reflected view matrix: reflect a world point across the plane, then apply
// the camera view. Equivalent to rendering the scene from the mirrored camera.
pub(crate) fn reflected_view(view: Mat4, plane: Vec4) -> Mat4 {
    mul(view, reflection_matrix(plane))
}

// Transform a world-space plane into a view space. Planes transform by the
// inverse-transpose of the world->view matrix: plane_view = (V^-1)^T . plane.
pub(crate) fn plane_in_view(plane_world: Vec4, view: Mat4) -> Vec4 {
    mat_vec(transpose(inverse(view)), plane_world)
}

// Oblique near-plane clipping (Lengyel) for a [0, 1]-depth perspective matrix.
// Replaces the projection's z (depth) row so the near clip plane coincides with
// `clip_plane` (given in the projection's view space), clipping everything on the
// negative side of that plane. The far plane is preserved by scaling against the
// frustum corner the plane faces.
//
// Derivation for this depth convention: the near plane is `z_row . p = 0`, so the
// new z-row is `alpha * C` for the clip plane C (any alpha keeps the near plane at
// C). Picking the far frustum corner q = inv(P) . (sgn(Cx), sgn(Cy), 1, 1) and
// requiring it to land on the far plane (ndc.z = 1, i.e. z_row.q = w_row.q = -q.z)
// gives alpha = -q.z / (C . q). For this projection q has the closed form below
// (q.z = -1), so alpha = 1 / (C . q).
pub(crate) fn oblique_projection(proj: Mat4, clip_plane: Vec4) -> Mat4 {
    let xs = proj[0][0];
    let ys = proj[1][1];
    let zs = proj[2][2]; // z-row's z component
    let zs_near = proj[3][2]; // z-row's w component (= zs * near)
    if xs.abs() < 1e-12 || ys.abs() < 1e-12 || zs_near.abs() < 1e-12 {
        return proj;
    }

    let sgn = |v: f32| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    // Back-projected far frustum corner toward the clip plane.
    let q: Vec4 = [
        sgn(clip_plane[0]) / xs,
        sgn(clip_plane[1]) / ys,
        -1.0,
        (1.0 + zs) / zs_near,
    ];
    let denom = dot4(clip_plane, q);
    if denom.abs() < 1e-12 {
        return proj;
    }
    let alpha = 1.0 / denom;

    let mut out = proj;
    // Replace the z (depth) row: row index 2 across all four columns.
    out[0][2] = alpha * clip_plane[0];
    out[1][2] = alpha * clip_plane[1];
    out[2][2] = alpha * clip_plane[2];
    out[3][2] = alpha * clip_plane[3];
    out
}

// Flip a plane so its normal points toward `point` (the kept side faces the
// camera). The reflection matrix is sign-invariant, but the oblique near-plane
// clip is not: it keeps the +n side, so the normal must face the viewer or the
// mirror render clips the wrong half. A no-op when `point` already lies on the
// +n side, which is the horizontal-water-above-camera case, so water renders
// identically with or without this orientation.
pub(crate) fn orient_plane_toward(plane: Vec4, point: [f32; 3]) -> Vec4 {
    let signed = plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3];
    if signed < 0.0 {
        [-plane[0], -plane[1], -plane[2], -plane[3]]
    } else {
        plane
    }
}

// The result of grouping a list of reflection planes into a bounded number of
// distinct slots: `slots[i]` is the slot a plane maps to (`None` when the budget
// is exhausted by earlier distinct planes, i.e. it falls back to the probe cube),
// and `representatives` is the deduplicated plane per slot (`representatives.len()`
// is the number of mirror renders the frame needs).
pub(crate) struct PlanarAssignment {
    pub(crate) slots: Vec<Option<usize>>,
    pub(crate) representatives: Vec<Vec4>,
}

// Group near-coplanar reflection planes so each distinct plane renders one mirror
// pass, capped at `max_slots`. Planes are matched sign-invariantly (a plane and
// its flip are the same surface): two planes share a slot when their unit normals
// are near-parallel and their offset along the normal matches. A plane coplanar
// with an already-assigned slot always reuses it (even past the budget); only a
// NEW distinct plane beyond `max_slots` overflows to `None`. Input order sets slot
// priority, so callers list higher-priority planes (e.g. water) first.
pub(crate) fn assign_planar_slots(planes: &[Vec4], max_slots: usize) -> PlanarAssignment {
    // ~2.6 degrees of normal divergence and 0.1 world units of offset still count
    // as the same plane: tight enough to keep separate walls distinct, loose
    // enough to merge co-planar panes authored with slight slop.
    const NORMAL_DOT_EPS: f32 = 0.999;
    const OFFSET_EPS: f32 = 0.1;

    let mut representatives: Vec<Vec4> = Vec::new();
    let mut slots: Vec<Option<usize>> = Vec::with_capacity(planes.len());
    for &raw in planes {
        let p = normalize_plane(raw);
        let nlen = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if nlen < 1e-6 {
            // Degenerate normal: no usable plane, fall back to the probe cube.
            slots.push(None);
            continue;
        }
        let mut found = None;
        for (i, r) in representatives.iter().enumerate() {
            let d = p[0] * r[0] + p[1] * r[1] + p[2] * r[2];
            if d.abs() >= NORMAL_DOT_EPS {
                // Align the representative to p's sign, then the two are the same
                // surface iff their plane constants match.
                let rd_aligned = if d < 0.0 { -r[3] } else { r[3] };
                if (p[3] - rd_aligned).abs() <= OFFSET_EPS {
                    found = Some(i);
                    break;
                }
            }
        }
        match found {
            Some(i) => slots.push(Some(i)),
            None => {
                if representatives.len() < max_slots {
                    representatives.push(p);
                    slots.push(Some(representatives.len() - 1));
                } else {
                    slots.push(None);
                }
            }
        }
    }
    PlanarAssignment {
        slots,
        representatives,
    }
}

// The matrices a planar reflection pass needs for one mirror plane.
pub(crate) struct PlanarMatrices {
    // Reflected view matrix (world -> mirrored view).
    pub(crate) view: Mat4,
    // Reflected view-projection with oblique near-plane clipping applied.
    pub(crate) view_proj: Mat4,
    // The camera eye reflected across the plane (LOD / view-direction anchor).
    pub(crate) eye: [f32; 3],
}

// Build the mirror view + oblique-clipped view-projection + reflected eye for a
// camera (`view` / `proj` / `cam_pos`) reflecting across `plane_world`. The clip
// plane is nudged a hair below the surface (`clip_bias`, world units along the
// normal) so fragments exactly on the surface are not clipped by precision.
pub(crate) fn planar_matrices(
    view: Mat4,
    proj: Mat4,
    cam_pos: [f32; 3],
    plane_world: Vec4,
    clip_bias: f32,
) -> PlanarMatrices {
    let plane = normalize_plane(plane_world);
    let r_view = reflected_view(view, plane);
    // Push the clip plane slightly toward the kept (normal) side so geometry
    // right at the waterline survives the near-plane test.
    let clip_world = [plane[0], plane[1], plane[2], plane[3] + clip_bias];
    let clip_view = plane_in_view(clip_world, r_view);
    let r_proj = oblique_projection(proj, clip_view);
    PlanarMatrices {
        view: r_view,
        view_proj: mul(r_proj, r_view),
        eye: reflect_point(cam_pos, plane),
    }
}

// Resolve the CPU visible set for a planar mirror render: BVH-cull the cullable
// draw objects against the reflected-camera frustum, then append the always-draw
// fallback (skybox, rooms) so it appears in the reflection too. Mirrors the main
// camera's visible-set resolution but against the reflected frustum, so geometry
// visible only in the reflection (behind or beside the main camera, outside its
// frustum) is captured instead of reusing the main camera's set. `eye` is the
// reflected camera position, consulted only for the leaves' distance-based cull.
// `out` is cleared then refilled, so a caller can reuse one buffer across planes.
pub(crate) fn reflected_visible_set(
    bvh: &crate::gfx::bvh::Bvh,
    reflected_frustum: &crate::gfx::frustum::Frustum,
    eye: [f32; 3],
    always_draw: &[u32],
    out: &mut Vec<u32>,
) {
    out.clear();
    bvh.query(reflected_frustum, eye, |idx| out.push(idx));
    out.sort_unstable();
    out.extend_from_slice(always_draw);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let ys = 1.0 / (fov_y / 2.0).tan();
        let xs = ys / aspect;
        let zs = far / (near - far);
        [
            [xs, 0.0, 0.0, 0.0],
            [0.0, ys, 0.0, 0.0],
            [0.0, 0.0, zs, -1.0],
            [0.0, 0.0, zs * near, 0.0],
        ]
    }

    // Apply a column-major transform to a homogeneous point.
    fn xform(m: Mat4, p: [f32; 4]) -> [f32; 4] {
        mat_vec(m, p)
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn reflection_is_an_involution() {
        // Reflecting twice returns the original point (R * R = I).
        let plane = normalize_plane([0.0, 1.0, 0.0, -2.0]); // y = 2
        let p = [3.0, 5.0, -1.0];
        let once = reflect_point(p, plane);
        let twice = reflect_point(once, plane);
        assert!(approx(twice[0], p[0], 1e-5));
        assert!(approx(twice[1], p[1], 1e-5));
        assert!(approx(twice[2], p[2], 1e-5));
    }

    #[test]
    fn reflection_across_y_plane_flips_height_about_it() {
        // y = 2 plane: a point at y=5 reflects to y=-1 (mirror about 2).
        let plane = normalize_plane([0.0, 1.0, 0.0, -2.0]);
        let r = reflect_point([3.0, 5.0, -1.0], plane);
        assert!(approx(r[0], 3.0, 1e-5));
        assert!(approx(r[1], -1.0, 1e-5));
        assert!(approx(r[2], -1.0, 1e-5));
    }

    #[test]
    fn reflection_matrix_matches_point_reflection() {
        let plane = normalize_plane([0.2, 0.9, -0.3, 1.4]);
        let m = reflection_matrix(plane);
        let p = [1.3, -2.1, 0.7];
        let via_matrix = xform(m, [p[0], p[1], p[2], 1.0]);
        let via_point = reflect_point(p, plane);
        for i in 0..3 {
            assert!(approx(via_matrix[i], via_point[i], 1e-4), "component {i}");
        }
        assert!(approx(via_matrix[3], 1.0, 1e-5));
    }

    #[test]
    fn inverse_round_trips() {
        let m = perspective(1.1, 1.7, 0.2, 80.0);
        let id = mul(m, inverse(m));
        for (c, col) in id.iter().enumerate() {
            for (r, &val) in col.iter().enumerate() {
                let expect = if c == r { 1.0 } else { 0.0 };
                assert!(approx(val, expect, 1e-4), "[{c}][{r}]");
            }
        }
    }

    #[test]
    fn oblique_clip_puts_the_plane_at_the_near_depth() {
        // A view-space plane at z = -5 facing the camera (kept side is farther,
        // z < -5). After oblique clipping the projection, a point ON the plane
        // maps to ndc.z ~= 0, a point in front (far side) to ndc.z in (0, 1), and
        // a point behind the plane to ndc.z < 0 (clipped).
        let proj = perspective(1.2, 1.0, 0.1, 100.0);
        // Plane z = -5: n.p + d = 0 with kept side n.p + d > 0 toward -z (far).
        // Choose C so the far/kept side is positive: C = (0,0,-1,-5) -> for
        // p=(0,0,-50): -(-50)-5 = 45 > 0 (kept); p=(0,0,-2): 2-5 = -3 < 0 (clip).
        let c = [0.0, 0.0, -1.0, -5.0];
        let pobl = oblique_projection(proj, c);

        let ndc_z = |z: f32| {
            let clip = xform(pobl, [0.0, 0.0, z, 1.0]);
            clip[2] / clip[3]
        };
        assert!(approx(ndc_z(-5.0), 0.0, 1e-3), "on-plane ndc.z");
        let front = ndc_z(-50.0);
        assert!(front > 0.0 && front < 1.0, "far side in [0,1]: {front}");
        assert!(ndc_z(-2.0) < 0.0, "near side clipped");
    }

    #[test]
    fn oblique_clip_preserves_x_and_y_projection() {
        // Only the depth row changes; x/y of a projected point are untouched.
        let proj = perspective(1.0, 1.5, 0.1, 50.0);
        let c = [0.0, 0.0, -1.0, -8.0];
        let pobl = oblique_projection(proj, c);
        let p = [2.0, 1.5, -20.0, 1.0];
        let a = xform(proj, p);
        let b = xform(pobl, p);
        assert!(approx(a[0] / a[3], b[0] / b[3], 1e-5), "ndc.x");
        assert!(approx(a[1] / a[3], b[1] / b[3], 1e-5), "ndc.y");
    }

    #[test]
    fn planar_matrices_clip_below_the_water_plane() {
        // A camera above a horizontal water plane (y = 0, normal up). The mirror
        // pass must clip world geometry BELOW the plane (it would otherwise leak
        // into the reflection). Verify a below-water point lands at ndc.z < 0 and
        // an above-water point stays in [0, 1].
        let plane = [0.0, 1.0, 0.0, 0.0]; // y = 0, normal +y (kept side: above)
        // Simple camera at (0, 3, 6) looking toward -z and slightly down. Build a
        // view that just translates (identity rotation is enough for the depth
        // sign test since reflection + projection handle the rest).
        let view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -3.0, -6.0, 1.0],
        ];
        let proj = perspective(1.2, 1.6, 0.1, 100.0);
        let m = planar_matrices(view, proj, [0.0, 3.0, 6.0], plane, 0.0);

        let ndc_z = |p: [f32; 3]| {
            let clip = xform(m.view_proj, [p[0], p[1], p[2], 1.0]);
            clip[2] / clip[3]
        };
        // Above water, in front of the camera: visible (0..1).
        let above = ndc_z([0.0, 2.0, -4.0]);
        assert!(above > 0.0 && above < 1.0, "above-water visible: {above}");
        // Below water, in front of the camera: clipped (ndc.z < 0).
        let below = ndc_z([0.0, -2.0, -4.0]);
        assert!(below < 0.0, "below-water clipped: {below}");
        // The reflected eye sits below the plane (mirror of y = 3).
        assert!(approx(m.eye[1], -3.0, 1e-5), "reflected eye height");
    }

    #[test]
    fn orient_plane_faces_the_camera() {
        // A vertical pane at z = -3 with normal pointing toward -z. A camera in
        // front of it (at +z relative to the pane) must flip the normal so the
        // kept (oblique-clip) side faces the viewer.
        let plane = normalize_plane([0.0, 0.0, -1.0, -3.0]); // n.p + d = 0 -> z = -3
        let cam = [0.0, 1.0, 0.0]; // in front of the pane (z = 0 > -3 side)
        let oriented = orient_plane_toward(plane, cam);
        let signed =
            oriented[0] * cam[0] + oriented[1] * cam[1] + oriented[2] * cam[2] + oriented[3];
        assert!(signed > 0.0, "camera must be on the +normal (kept) side");
        // It flipped the original (which faced away from the camera).
        assert!(approx(oriented[2], 1.0, 1e-5), "normal flipped toward +z");
    }

    #[test]
    fn orient_plane_is_noop_for_water_above_camera() {
        // Horizontal water plane y = 2, normal +y. A camera above it keeps the
        // plane unchanged, so water renders identically with the orientation step.
        let plane = normalize_plane([0.0, 1.0, 0.0, -2.0]);
        let cam = [3.0, 5.0, -1.0]; // above the plane
        let oriented = orient_plane_toward(plane, cam);
        for i in 0..4 {
            assert!(
                approx(oriented[i], plane[i], 1e-6),
                "component {i} unchanged"
            );
        }
    }

    #[test]
    fn assign_slots_dedups_coplanar_and_caps_distinct() {
        // Two coplanar panes (same wall), one distinct wall, plus a third distinct
        // wall that overflows a budget of 2. The coplanar pair shares slot 0, the
        // second wall takes slot 1, the third overflows to None.
        let wall_a0 = [0.0, 0.0, 1.0, -3.0];
        let wall_a1 = [0.0, 0.0, 1.0, -3.05]; // within OFFSET_EPS of a0
        let wall_b = [1.0, 0.0, 0.0, -5.0];
        let wall_c = [0.0, 1.0, 0.0, -1.0];
        let a = assign_planar_slots(&[wall_a0, wall_a1, wall_b, wall_c], 2);
        assert_eq!(a.representatives.len(), 2, "two slots allocated");
        assert_eq!(a.slots[0], Some(0));
        assert_eq!(a.slots[1], Some(0), "coplanar pane reuses slot 0");
        assert_eq!(a.slots[2], Some(1));
        assert_eq!(
            a.slots[3], None,
            "third distinct plane overflows the budget"
        );
    }

    #[test]
    fn assign_slots_is_sign_invariant() {
        // A plane and its flip (opposite normal, opposite offset) are the same
        // surface and must share a slot.
        let front = [0.0, 0.0, 1.0, -3.0];
        let back = [0.0, 0.0, -1.0, 3.0];
        let a = assign_planar_slots(&[front, back], 4);
        assert_eq!(a.representatives.len(), 1, "flip is the same surface");
        assert_eq!(a.slots[0], Some(0));
        assert_eq!(a.slots[1], Some(0));
    }

    #[test]
    fn reflected_frustum_captures_geometry_behind_the_camera() {
        // A vertical mirror at z = -5 (unit normal +z, facing the camera). The
        // camera sits at the origin looking down -z, toward the mirror. An object
        // BEHIND the camera (z = +3) is outside the main frustum, but its
        // reflection is visible in the mirror -- so the reflected-frustum cull must
        // capture it where the main-camera set would miss it (the V1 gap).
        let plane = [0.0, 0.0, 1.0, 5.0]; // n.p + d = 0 -> z = -5
        let view = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let proj = perspective(1.2, 1.6, 0.1, 100.0);
        let cam_pos = [0.0, 0.0, 0.0];
        let m = planar_matrices(view, proj, cam_pos, plane, 0.0);

        // One cullable object behind the camera.
        let bvh = crate::gfx::bvh::Bvh::build(&[crate::gfx::bvh::BvhItem {
            bb_min: [-0.5, -0.5, 2.5],
            bb_max: [0.5, 0.5, 3.5],
            cull_distance: 0.0,
            index: 0,
        }]);

        // The main camera rejects it (behind the near plane).
        let main_frustum = crate::gfx::frustum::Frustum::from_view_projection(proj);
        let mut main_visible = Vec::new();
        bvh.query(&main_frustum, cam_pos, |i| main_visible.push(i));
        assert!(
            !main_visible.contains(&0),
            "object behind the camera must be outside the main frustum"
        );

        // The reflected frustum captures it, and the always-draw fallback is
        // appended after the culled set.
        let reflected_frustum = crate::gfx::frustum::Frustum::from_view_projection(m.view_proj);
        let always = [7u32];
        let mut out = Vec::new();
        reflected_visible_set(&bvh, &reflected_frustum, m.eye, &always, &mut out);
        assert!(
            out.contains(&0),
            "object behind the camera must be visible in the reflection"
        );
        assert_eq!(
            out.last(),
            Some(&7),
            "always-draw fallback appended after the culled set"
        );
    }

    #[test]
    fn reflected_visible_set_reuses_the_output_buffer() {
        // The buffer is cleared each call, so reusing it across planes never leaks
        // a prior plane's culled indices.
        let bvh = crate::gfx::bvh::Bvh::build(&[crate::gfx::bvh::BvhItem {
            bb_min: [-0.5, -0.5, -0.5],
            bb_max: [0.5, 0.5, 0.5],
            cull_distance: 0.0,
            index: 3,
        }]);
        // A frustum that rejects everything (identity clip cube, box far outside).
        let empty_frustum = crate::gfx::frustum::Frustum::from_view_projection([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-100.0, 0.0, 0.0, 1.0],
        ]);
        let mut out = vec![99, 99, 99];
        reflected_visible_set(&bvh, &empty_frustum, [0.0, 0.0, 0.0], &[], &mut out);
        assert!(
            out.is_empty(),
            "stale indices must be cleared before refill"
        );
    }

    #[test]
    fn assign_slots_overflow_still_reuses_existing_slot() {
        // With a budget of 1, a second distinct plane overflows, but a later plane
        // coplanar with slot 0 still maps to slot 0 (dedup precedes the cap).
        let a = assign_planar_slots(
            &[
                [0.0, 0.0, 1.0, -3.0],
                [1.0, 0.0, 0.0, -5.0], // overflow
                [0.0, 0.0, 1.0, -3.0], // coplanar with slot 0
            ],
            1,
        );
        assert_eq!(a.representatives.len(), 1);
        assert_eq!(a.slots[0], Some(0));
        assert_eq!(a.slots[1], None);
        assert_eq!(a.slots[2], Some(0));
    }
}
