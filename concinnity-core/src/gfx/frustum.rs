// src/gfx/frustum.rs
//
// Backend-agnostic frustum culling.
//
// Given a column-major view-projection matrix the six clip-space planes are
// extracted using the Gribb-Hartmann method (left/right/bottom/top/near/far).
// `Frustum::intersects_aabb` returns false only when an axis-aligned bounding
// box is fully outside at least one plane.  False positives are acceptable for
// culling (a few extra draws), false negatives are not, so the test treats
// the box as visible whenever it overlaps any plane.

#[derive(Copy, Clone, Debug)]
pub struct Plane {
    // Plane equation in clip space: dot(normal, p) + d >= 0 == inside.
    pub normal: [f32; 3],
    pub d: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    // Build a frustum from a column-major view-projection matrix.
    // `vp[col][row]`: same layout used by the renderer's ViewUniforms.
    pub fn from_view_projection(vp: [[f32; 4]; 4]) -> Self {
        // Row r of vp = [vp[0][r], vp[1][r], vp[2][r], vp[3][r]].
        let row = |r: usize| -> [f32; 4] { [vp[0][r], vp[1][r], vp[2][r], vp[3][r]] };
        let r0 = row(0);
        let r1 = row(1);
        let r2 = row(2);
        let r3 = row(3);

        let make = |a: [f32; 4], b: [f32; 4], sign: f32| -> Plane {
            let p = [
                a[0] * sign + b[0],
                a[1] * sign + b[1],
                a[2] * sign + b[2],
                a[3] * sign + b[3],
            ];
            normalise_plane(p)
        };

        Self {
            planes: [
                make(r0, r3, 1.0),  // left:   row3 + row0
                make(r0, r3, -1.0), // right:  row3 - row0
                make(r1, r3, 1.0),  // bottom: row3 + row1
                make(r1, r3, -1.0), // top:    row3 - row1
                make(r2, r3, 1.0),  // near:   row3 + row2   (works for 0..1 z and -1..1 z)
                make(r2, r3, -1.0), // far:    row3 - row2
            ],
        }
    }

    // True when the AABB is not entirely outside any plane.
    pub fn intersects_aabb(&self, bb_min: [f32; 3], bb_max: [f32; 3]) -> bool {
        for plane in &self.planes {
            // Pick the AABB corner furthest along the plane normal ("p-vertex"
            // in the SAT against a plane). If that corner is still behind the
            // plane the entire AABB is outside.
            let mut farthest = [0.0f32; 3];
            for (i, n) in plane.normal.iter().enumerate() {
                farthest[i] = if *n >= 0.0 { bb_max[i] } else { bb_min[i] };
            }
            let dist = plane.normal[0] * farthest[0]
                + plane.normal[1] * farthest[1]
                + plane.normal[2] * farthest[2]
                + plane.d;
            if dist < 0.0 {
                return false;
            }
        }
        true
    }
}

fn normalise_plane(p: [f32; 4]) -> Plane {
    let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    let inv = if len > 1e-6 { 1.0 / len } else { 1.0 };
    Plane {
        normal: [p[0] * inv, p[1] * inv, p[2] * inv],
        d: p[3] * inv,
    }
}

// Compute the world-space AABB enclosing a local-space AABB transformed by
// a column-major model matrix.  All eight corners are transformed and
// min/max'd component-wise.
pub fn transform_aabb(
    bb_min: [f32; 3],
    bb_max: [f32; 3],
    model: [[f32; 4]; 4],
) -> ([f32; 3], [f32; 3]) {
    let corners = [
        [bb_min[0], bb_min[1], bb_min[2]],
        [bb_max[0], bb_min[1], bb_min[2]],
        [bb_min[0], bb_max[1], bb_min[2]],
        [bb_max[0], bb_max[1], bb_min[2]],
        [bb_min[0], bb_min[1], bb_max[2]],
        [bb_max[0], bb_min[1], bb_max[2]],
        [bb_min[0], bb_max[1], bb_max[2]],
        [bb_max[0], bb_max[1], bb_max[2]],
    ];
    let mut out_min = [f32::INFINITY; 3];
    let mut out_max = [f32::NEG_INFINITY; 3];
    for c in &corners {
        // column-major mul: out = M * (c.x, c.y, c.z, 1)
        let x = model[0][0] * c[0] + model[1][0] * c[1] + model[2][0] * c[2] + model[3][0];
        let y = model[0][1] * c[0] + model[1][1] * c[1] + model[2][1] * c[2] + model[3][1];
        let z = model[0][2] * c[0] + model[1][2] * c[1] + model[2][2] * c[2] + model[3][2];
        out_min[0] = out_min[0].min(x);
        out_min[1] = out_min[1].min(y);
        out_min[2] = out_min[2].min(z);
        out_max[0] = out_max[0].max(x);
        out_max[1] = out_max[1].max(y);
        out_max[2] = out_max[2].max(z);
    }
    (out_min, out_max)
}

// Squared distance from `cam` to the closest point on the AABB.
// Returns 0 if `cam` is inside.
pub fn aabb_distance_sq(cam: [f32; 3], bb_min: [f32; 3], bb_max: [f32; 3]) -> f32 {
    let mut sq = 0.0f32;
    for i in 0..3 {
        let v = cam[i];
        if v < bb_min[i] {
            let d = bb_min[i] - v;
            sq += d * d;
        } else if v > bb_max[i] {
            let d = v - bb_max[i];
            sq += d * d;
        }
    }
    sq
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity4() -> [[f32; 4]; 4] {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn identity_vp_contains_origin_aabb() {
        // Identity VP defines the [-1,1]^3 clip cube as the visible region.
        let f = Frustum::from_view_projection(identity4());
        assert!(f.intersects_aabb([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]));
    }

    #[test]
    fn identity_vp_rejects_far_aabb() {
        let f = Frustum::from_view_projection(identity4());
        // entirely past the right clip plane
        assert!(!f.intersects_aabb([5.0, -0.5, -0.5], [6.0, 0.5, 0.5]));
    }

    #[test]
    fn transform_aabb_identity_passthrough() {
        let (mn, mx) = transform_aabb([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], identity4());
        assert_eq!(mn, [0.0, 0.0, 0.0]);
        assert_eq!(mx, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn transform_aabb_translates_corners() {
        let mut model = identity4();
        model[3][0] = 5.0;
        model[3][1] = -2.0;
        let (mn, mx) = transform_aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], model);
        assert_eq!(mn, [5.0, -2.0, 0.0]);
        assert_eq!(mx, [6.0, -1.0, 1.0]);
    }

    #[test]
    fn aabb_distance_inside_is_zero() {
        let d = aabb_distance_sq([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn aabb_distance_outside_is_squared() {
        // Camera 3 units to the right of a unit box at origin
        let d = aabb_distance_sq([4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((d - 9.0).abs() < 1e-5);
    }
}
