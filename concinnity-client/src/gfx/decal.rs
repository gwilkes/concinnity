// src/gfx/decal.rs
//
// Backend-agnostic decal helpers. Owns the per-decal model / inverse-model
// matrix math the projected-decal pass needs at runtime, plus the `DecalRecord`
// the backends consume. Decals are stamped onto the scene depth buffer by
// drawing a unit-box volume per decal: the fragment shader reconstructs the
// world-space point of each rasterised pixel from depth and tests whether it
// lies inside the box.

use crate::assets::Decal;

// Per-decal data the renderer consumes each frame. Built once at
// `GraphicsSystem` init from the world's `Decal` components.
//
// `model` is the local→world transform of a unit cube spanning `[-0.5, 0.5]^3`
// in local space. `inv_model` is its inverse; the fragment shader uses it to
// pull a reconstructed world-space sample point back into decal-local space
// and test it against the unit box. `texture_slot` indexes the renderer's
// albedo texture pool; `tint` is RGB×alpha applied to every projected sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecalRecord {
    pub model: [[f32; 4]; 4],
    pub inv_model: [[f32; 4]; 4],
    pub texture_slot: usize,
    pub tint: [f32; 4],
}

impl DecalRecord {
    // World-space AABB enclosing the decal's unit-cube volume. Used by the
    // per-frame frustum-cull skip so a record that lands fully outside the
    // camera frustum costs no draw call. The AABB is the transform of
    // `[-0.5, 0.5]^3` by `model`; for a non-rotated decal this exactly
    // matches the authored `size`, and for a rotated decal it is the
    // minimum AABB enclosing the rotated box.
    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        crate::gfx::frustum::transform_aabb([-0.5; 3], [0.5; 3], self.model)
    }
}

// Build the world-space `model` matrix for a decal: `T(position) * R_yxz *
// S(size)`. Column-major; the inner index is the row. Matches the rotation
// convention used by `Prop::model_matrix`.
pub fn decal_model_matrix(
    position: [f32; 3],
    rotation_deg: [f32; 3],
    size: [f32; 3],
) -> [[f32; 4]; 4] {
    let [px, py, pz] = position;
    let [pitch_deg, yaw_deg, roll_deg] = rotation_deg;
    let [sx, sy, sz] = size;

    let (sp, cp) = (pitch_deg.to_radians().sin(), pitch_deg.to_radians().cos());
    let (sy_, cy) = (yaw_deg.to_radians().sin(), yaw_deg.to_radians().cos());
    let (sr, cr) = (roll_deg.to_radians().sin(), roll_deg.to_radians().cos());

    // R = Ry * Rx * Rz, then scaled component-wise and translated.
    [
        [
            sx * (cy * cr + sy_ * sp * sr),
            sx * (cp * sr),
            sx * (-sy_ * cr + cy * sp * sr),
            0.0,
        ],
        [
            sy * (-cy * sr + sy_ * sp * cr),
            sy * (cp * cr),
            sy * (sy_ * sr + cy * sp * cr),
            0.0,
        ],
        [sz * (sy_ * cp), sz * (-sp), sz * (cy * cp), 0.0],
        [px, py, pz, 1.0],
    ]
}

// Invert an affine TRS matrix of the form built by [`decal_model_matrix`].
// The 3×3 linear part is `R * diag(size)`; its inverse is
// `diag(1/size) * R_transpose`. The translation flips into the inverted
// frame: `-inv_linear * translation`.
//
// Returns `None` when any size component is non-finite or zero, a degenerate
// decal whose volume has collapsed. The renderer skips such decals.
pub fn invert_decal_model(model: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    // Columns of the 3×3 are scaled rotation basis vectors; their lengths are
    // |size_x|, |size_y|, |size_z|.
    let col0 = [model[0][0], model[0][1], model[0][2]];
    let col1 = [model[1][0], model[1][1], model[1][2]];
    let col2 = [model[2][0], model[2][1], model[2][2]];
    let s0 = (col0[0] * col0[0] + col0[1] * col0[1] + col0[2] * col0[2]).sqrt();
    let s1 = (col1[0] * col1[0] + col1[1] * col1[1] + col1[2] * col1[2]).sqrt();
    let s2 = (col2[0] * col2[0] + col2[1] * col2[1] + col2[2] * col2[2]).sqrt();
    if !(s0.is_finite() && s1.is_finite() && s2.is_finite()) || s0 == 0.0 || s1 == 0.0 || s2 == 0.0
    {
        return None;
    }
    // Orthonormal rotation columns recovered from the scaled columns.
    let r0 = [col0[0] / s0, col0[1] / s0, col0[2] / s0];
    let r1 = [col1[0] / s1, col1[1] / s1, col1[2] / s1];
    let r2 = [col2[0] / s2, col2[1] / s2, col2[2] / s2];
    // inverse_linear = diag(1/size) * R^T. Stored column-major as the 3×3
    // upper-left of the inverse matrix.
    let inv_lin = [
        // column 0
        [r0[0] / s0, r1[0] / s1, r2[0] / s2],
        // column 1
        [r0[1] / s0, r1[1] / s1, r2[1] / s2],
        // column 2
        [r0[2] / s0, r1[2] / s1, r2[2] / s2],
    ];
    let t = [model[3][0], model[3][1], model[3][2]];
    let inv_t = [
        -(inv_lin[0][0] * t[0] + inv_lin[1][0] * t[1] + inv_lin[2][0] * t[2]),
        -(inv_lin[0][1] * t[0] + inv_lin[1][1] * t[1] + inv_lin[2][1] * t[2]),
        -(inv_lin[0][2] * t[0] + inv_lin[1][2] * t[1] + inv_lin[2][2] * t[2]),
    ];
    Some([
        [inv_lin[0][0], inv_lin[0][1], inv_lin[0][2], 0.0],
        [inv_lin[1][0], inv_lin[1][1], inv_lin[1][2], 0.0],
        [inv_lin[2][0], inv_lin[2][1], inv_lin[2][2], 0.0],
        [inv_t[0], inv_t[1], inv_t[2], 1.0],
    ])
}

// Resolve a list of `Decal` components into `DecalRecord`s the backend can
// consume. Skips decals whose texture reference is missing, invisible decals,
// and decals whose size is degenerate (any non-positive component).
//
// `texture_name_to_slot` maps a Texture asset id to its slot in the backend's
// albedo texture pool; a decal referencing an unknown texture is logged and
// dropped. A decal with no `texture` falls back to texture slot 0 (the
// renderer's white fallback) so the tint colour still stamps.
pub fn build_decal_records(
    decals: &[&Decal],
    texture_name_to_slot: &std::collections::HashMap<crate::ecs::asset_id::AssetId, usize>,
) -> Vec<DecalRecord> {
    let mut out = Vec::new();
    for d in decals {
        if !d.visible {
            continue;
        }
        if !(d.size[0] > 0.0 && d.size[1] > 0.0 && d.size[2] > 0.0) {
            continue;
        }
        let slot = match d.texture {
            None => 0,
            Some(tex_id) => match texture_name_to_slot.get(&tex_id) {
                Some(&s) => s,
                None => {
                    tracing::error!(
                        "GraphicsSystem: Decal {} references unknown texture {}",
                        d.asset_id,
                        tex_id
                    );
                    continue;
                }
            },
        };
        let model = decal_model_matrix(d.position, d.rotation_deg, d.size);
        let inv_model = match invert_decal_model(model) {
            Some(m) => m,
            None => continue,
        };
        out.push(DecalRecord {
            model,
            inv_model,
            texture_slot: slot,
            tint: d.tint,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat_mul(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
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

    fn near(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> bool {
        a.iter().zip(b.iter()).all(|(ac, bc)| {
            ac.iter()
                .zip(bc.iter())
                .all(|(av, bv)| (av - bv).abs() < 1e-4)
        })
    }

    const IDENTITY: [[f32; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    #[test]
    fn unit_decal_has_identity_model() {
        let m = decal_model_matrix([0.0; 3], [0.0; 3], [1.0; 3]);
        assert!(near(m, IDENTITY));
    }

    #[test]
    fn translation_scale_compose_into_model() {
        let m = decal_model_matrix([2.0, 3.0, -4.0], [0.0; 3], [0.5, 1.0, 2.0]);
        // diag(0.5, 1.0, 2.0) translated by (2, 3, -4).
        let expected: [[f32; 4]; 4] = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [2.0, 3.0, -4.0, 1.0],
        ];
        assert!(near(m, expected));
    }

    #[test]
    fn inverse_round_trips_through_model() {
        let m = decal_model_matrix([1.0, -2.0, 0.5], [30.0, -15.0, 45.0], [0.4, 1.2, 0.7]);
        let inv = invert_decal_model(m).expect("non-degenerate model is invertible");
        assert!(near(mat_mul(m, inv), IDENTITY));
        assert!(near(mat_mul(inv, m), IDENTITY));
    }

    #[test]
    fn degenerate_size_rejects_inverse() {
        let m = decal_model_matrix([0.0; 3], [0.0; 3], [0.0, 1.0, 1.0]);
        assert!(invert_decal_model(m).is_none());
    }

    #[test]
    fn invisible_decal_is_skipped_in_records() {
        let d = Decal {
            visible: false,
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        assert!(build_decal_records(&[&d], &names).is_empty());
    }

    #[test]
    fn degenerate_size_is_skipped_in_records() {
        let d = Decal {
            size: [1.0, 0.0, 1.0],
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        assert!(build_decal_records(&[&d], &names).is_empty());
    }

    #[test]
    fn decal_without_texture_uses_fallback_slot() {
        let d = Decal::default();
        let names = std::collections::HashMap::new();
        let recs = build_decal_records(&[&d], &names);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].texture_slot, 0);
    }

    #[test]
    fn aabb_of_unit_decal_at_origin_is_half_unit_box() {
        let d = Decal::default();
        let names = std::collections::HashMap::new();
        let recs = build_decal_records(&[&d], &names);
        let (mn, mx) = recs[0].aabb();
        assert!((mn[0] + 0.5).abs() < 1e-5 && (mx[0] - 0.5).abs() < 1e-5);
        assert!((mn[1] + 0.5).abs() < 1e-5 && (mx[1] - 0.5).abs() < 1e-5);
        assert!((mn[2] + 0.5).abs() < 1e-5 && (mx[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn aabb_translates_with_decal_position() {
        let d = Decal {
            position: [10.0, 5.0, -3.0],
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_decal_records(&[&d], &names);
        let (mn, mx) = recs[0].aabb();
        // size = [1,1,1] → half-extents 0.5 in every axis.
        assert!((mn[0] - 9.5).abs() < 1e-5 && (mx[0] - 10.5).abs() < 1e-5);
        assert!((mn[1] - 4.5).abs() < 1e-5 && (mx[1] - 5.5).abs() < 1e-5);
        assert!((mn[2] + 3.5).abs() < 1e-5 && (mx[2] + 2.5).abs() < 1e-5);
    }

    #[test]
    fn aabb_expands_under_size() {
        let d = Decal {
            size: [4.0, 0.5, 8.0],
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_decal_records(&[&d], &names);
        let (mn, mx) = recs[0].aabb();
        assert!((mx[0] - mn[0] - 4.0).abs() < 1e-5);
        assert!((mx[1] - mn[1] - 0.5).abs() < 1e-5);
        assert!((mx[2] - mn[2] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_includes_rotated_decal_extents() {
        // 45° yaw about Y on a 2×1×2 box: the local X-Z corners (±1, ±1)
        // rotate to (0, ±√2) and (±√2, 0), so the AABB extends ±√2 along
        // each of X and Z, a span of 2√2 ≈ 2.828.
        let d = Decal {
            size: [2.0, 1.0, 2.0],
            rotation_deg: [0.0, 45.0, 0.0],
            ..Default::default()
        };
        let names = std::collections::HashMap::new();
        let recs = build_decal_records(&[&d], &names);
        let (mn, mx) = recs[0].aabb();
        let span_x = mx[0] - mn[0];
        let span_z = mx[2] - mn[2];
        let expected = 2.0 * std::f32::consts::SQRT_2;
        assert!((span_x - expected).abs() < 1e-4);
        assert!((span_z - expected).abs() < 1e-4);
        // Y axis is unaffected by yaw; still 1 unit tall.
        assert!((mx[1] - mn[1] - 1.0).abs() < 1e-5);
    }
}
