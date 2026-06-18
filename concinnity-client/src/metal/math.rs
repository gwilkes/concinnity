// Right-handed perspective matching Apple's matrix_perspective_right_hand.
// Depth maps to [0, 1] in Metal NDC. Column-major storage: arr[col][row].
pub(super) fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let ys = 1.0 / (fov_y / 2.0).tan();
    let xs = ys / aspect;
    let zs = far / (near - far); // negative; maps near->0 and far->1 after w-divide
    [
        [xs, 0.0, 0.0, 0.0],
        [0.0, ys, 0.0, 0.0],
        [0.0, 0.0, zs, -1.0],
        [0.0, 0.0, zs * near, 0.0],
    ]
}

// Column-major multiply: out[col][row] = sum_k a[k][row] * b[col][k].
pub(super) fn mat4_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
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

pub(super) const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// General 4×4 matrix inverse via cofactor expansion. Returns `IDENTITY4` for
// a singular input (determinant ≈ 0). Used per-frame by the decal pass to
// invert the view-projection so the shader can reconstruct world position
// from depth. Column-major storage: result[col][row].
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(super) fn mat4_inverse(m: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Read the input as a flat row-major slice for the standard cofactor
    // layout; then re-emit it in the engine's column-major arr[col][row] form.
    let r = |c: usize, row: usize| m[c][row];

    let a00 = r(0, 0);
    let a01 = r(1, 0);
    let a02 = r(2, 0);
    let a03 = r(3, 0);
    let a10 = r(0, 1);
    let a11 = r(1, 1);
    let a12 = r(2, 1);
    let a13 = r(3, 1);
    let a20 = r(0, 2);
    let a21 = r(1, 2);
    let a22 = r(2, 2);
    let a23 = r(3, 2);
    let a30 = r(0, 3);
    let a31 = r(1, 3);
    let a32 = r(2, 3);
    let a33 = r(3, 3);

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
    if det.abs() < 1e-20 || !det.is_finite() {
        return IDENTITY4;
    }
    let inv_det = 1.0 / det;

    // Row-major adjugate / det.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) {
        for col in 0..4 {
            for row in 0..4 {
                assert!(
                    (a[col][row] - b[col][row]).abs() < 1e-3,
                    "mismatch at [{}][{}]: {} vs {}",
                    col,
                    row,
                    a[col][row],
                    b[col][row]
                );
            }
        }
    }

    #[test]
    fn identity_is_multiplicative_unit() {
        let m = [
            [0.8, 0.1, -0.2, 0.0],
            [-0.1, 0.9, 0.3, 0.0],
            [0.2, -0.3, 0.85, 0.0],
            [3.0, -1.5, 7.0, 1.0],
        ];
        approx_eq(mat4_mul(m, IDENTITY4), m);
        approx_eq(mat4_mul(IDENTITY4, m), m);
    }

    #[test]
    fn mat4_inverse_round_trips_with_perspective() {
        let p = perspective(75.0_f32.to_radians(), 1.6, 0.1, 500.0);
        let inv = mat4_inverse(p);
        approx_eq(mat4_mul(p, inv), IDENTITY4);
        approx_eq(mat4_mul(inv, p), IDENTITY4);
    }

    #[test]
    fn mat4_inverse_round_trips_with_view_like_matrix() {
        // A rotation + translation column-major matrix; mat4_inverse must
        // recover its inverse.
        let view: [[f32; 4]; 4] = [
            [0.92388, 0.0, -0.38268, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.38268, 0.0, 0.92388, 0.0],
            [-1.5, -0.7, 4.0, 1.0],
        ];
        let inv = mat4_inverse(view);
        approx_eq(mat4_mul(view, inv), IDENTITY4);
    }

    #[test]
    fn perspective_maps_near_and_far_to_unit_depth() {
        // A point on the near plane maps to NDC depth 0, one on the far plane
        // to NDC depth 1 after the perspective w-divide.
        let proj = perspective(75.0_f32.to_radians(), 1.6, 0.1, 500.0);
        let ndc_depth = |view_z: f32| {
            let z = proj[2][2] * view_z + proj[3][2];
            let w = proj[2][3] * view_z + proj[3][3];
            z / w
        };
        assert!(ndc_depth(-0.1).abs() < 1e-3);
        assert!((ndc_depth(-500.0) - 1.0).abs() < 1e-3);
    }
}
