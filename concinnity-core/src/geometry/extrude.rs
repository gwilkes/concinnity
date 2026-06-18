// src/geometry/extrude.rs: extrude a 2D profile in the XZ plane along Y.
//
// Authored by the macOS Mesh Editor: users sketch a polygon in the top-down
// view and pick an extrude height plus an optional uniform corner radius.
// Output is a closed mesh with a flat top, flat bottom, and one flat-shaded
// quad per profile edge.
//
// Args:
//   profile         array of [x, z] pairs (>= 3 points). Either CW or CCW
//                   when viewed from +Y is accepted; the build normalises.
//   height          full extrusion height along Y (default 1.0, must be > 0)
//   corner_radius   optional, default 0.0 (sharp corners)
//   corner_segments optional, default 8; arc samples per rounded corner
//
// Concave (reflex) corners and corners where the radius would exceed the
// available edge length are passed through as-is rather than rounded.

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;
type GeomResult = Result<(Verts, Vec<u16>), String>;

pub(super) fn build_extrude(args: &serde_json::Value) -> GeomResult {
    let profile_raw = args
        .get("profile")
        .and_then(|v| v.as_array())
        .ok_or("extrude requires a `profile` array of [x, z] pairs")?;

    let mut profile: Vec<[f32; 2]> = Vec::with_capacity(profile_raw.len());
    for (i, p) in profile_raw.iter().enumerate() {
        let arr = p
            .as_array()
            .ok_or_else(|| format!("profile[{i}] must be a 2-element [x, z] array"))?;
        if arr.len() < 2 {
            return Err(format!(
                "profile[{i}] must have 2 elements, got {}",
                arr.len()
            ));
        }
        let x = arr[0]
            .as_f64()
            .ok_or_else(|| format!("profile[{i}][0] must be a number"))? as f32;
        let z = arr[1]
            .as_f64()
            .ok_or_else(|| format!("profile[{i}][1] must be a number"))? as f32;
        profile.push([x, z]);
    }

    if profile.len() < 3 {
        return Err(format!(
            "extrude profile must have at least 3 points, got {}",
            profile.len()
        ));
    }

    let height = args.get("height").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    if !height.is_finite() || height <= 0.0 {
        return Err(format!(
            "extrude height must be a positive number, got {height}"
        ));
    }

    let corner_radius = args
        .get("corner_radius")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    if !corner_radius.is_finite() || corner_radius < 0.0 {
        return Err(format!(
            "extrude corner_radius must be non-negative, got {corner_radius}"
        ));
    }
    let corner_segments = (args
        .get("corner_segments")
        .and_then(|v| v.as_u64())
        .unwrap_or(8)
        .max(1)) as usize;

    // Normalise to CCW-math (positive shoelace area in (x, z)). Ear clipping
    // assumes this orientation; the top-cap triangle indices are emitted in
    // reversed winding so the geometric normal still resolves to +Y.
    if signed_area(&profile) < 0.0 {
        profile.reverse();
    }

    if corner_radius > 0.0 {
        profile = round_corners(&profile, corner_radius, corner_segments);
        if profile.len() < 3 {
            return Err("extrude profile collapsed below 3 points after rounding".into());
        }
    }

    let n = profile.len();
    let total_verts = n * 6; // top n + bottom n + 4 per side wall * n walls
    if total_verts > 65536 {
        return Err(format!(
            "extrude profile of {n} points produces {total_verts} vertices, exceeding the u16 limit"
        ));
    }

    let half_h = height / 2.0;
    let top_color = [0.78f32, 0.76, 0.74];
    let bot_color = [0.66f32, 0.64, 0.62];
    let side_color = [0.72f32, 0.70, 0.68];

    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();

    // Top cap (y = +half_h, normal +Y). Planar UV uses XZ directly.
    let top_base = verts.len() as u16;
    for &[x, z] in &profile {
        verts.push(([x, half_h, z], [0.0, 1.0, 0.0], top_color, [x, z]));
    }
    let top_tris = ear_clip(&profile)?;
    for &[a, b, c] in &top_tris {
        // Reversed winding so the face normal matches the per-vertex +Y.
        idxs.extend_from_slice(&[
            top_base + c as u16,
            top_base + b as u16,
            top_base + a as u16,
        ]);
    }

    // Bottom cap (y = -half_h, normal -Y). Original winding gives -Y face normal.
    let bot_base = verts.len() as u16;
    for &[x, z] in &profile {
        verts.push(([x, -half_h, z], [0.0, -1.0, 0.0], bot_color, [x, z]));
    }
    for &[a, b, c] in &top_tris {
        idxs.extend_from_slice(&[
            bot_base + a as u16,
            bot_base + b as u16,
            bot_base + c as u16,
        ]);
    }

    // Side walls. One flat-shaded quad per profile edge with its own normal.
    for i in 0..n {
        let p0 = profile[i];
        let p1 = profile[(i + 1) % n];
        let dx = p1[0] - p0[0];
        let dz = p1[1] - p0[1];
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        // Outward normal for CCW-math polygon: rotate edge direction -90° in XZ.
        let normal = [dz / len, 0.0, -dx / len];

        let base = verts.len() as u16;
        verts.push(([p0[0], -half_h, p0[1]], normal, side_color, [0.0, 0.0]));
        verts.push(([p0[0], half_h, p0[1]], normal, side_color, [0.0, height]));
        verts.push(([p1[0], half_h, p1[1]], normal, side_color, [len, height]));
        verts.push(([p1[0], -half_h, p1[1]], normal, side_color, [len, 0.0]));
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    Ok((verts, idxs))
}

fn signed_area(profile: &[[f32; 2]]) -> f32 {
    let mut a = 0.0f32;
    for i in 0..profile.len() {
        let p = profile[i];
        let q = profile[(i + 1) % profile.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * a
}

// Round each convex corner of a CCW-math polygon with the given radius.
//
// Concave (reflex) corners and corners where the tangent distance would
// exceed half the adjacent edge length are passed through unchanged.
fn round_corners(profile: &[[f32; 2]], radius: f32, segments: usize) -> Vec<[f32; 2]> {
    let n = profile.len();
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(n * (segments + 1));
    for i in 0..n {
        let prev = profile[(i + n - 1) % n];
        let curr = profile[i];
        let next = profile[(i + 1) % n];
        let in_dx = curr[0] - prev[0];
        let in_dz = curr[1] - prev[1];
        let out_dx = next[0] - curr[0];
        let out_dz = next[1] - curr[1];
        let in_len = (in_dx * in_dx + in_dz * in_dz).sqrt();
        let out_len = (out_dx * out_dx + out_dz * out_dz).sqrt();
        if in_len < 1e-6 || out_len < 1e-6 {
            out.push(curr);
            continue;
        }
        let in_ux = in_dx / in_len;
        let in_uz = in_dz / in_len;
        let out_ux = out_dx / out_len;
        let out_uz = out_dz / out_len;
        let cross = in_ux * out_uz - in_uz * out_ux;
        let dot = in_ux * out_ux + in_uz * out_uz;
        if cross < 1e-6 {
            // Straight or right turn (concave): no rounding for this corner.
            out.push(curr);
            continue;
        }
        let phi = dot.clamp(-1.0, 1.0).acos();
        let half_phi = phi / 2.0;
        let tan_half = half_phi.tan();
        if tan_half < 1e-6 {
            out.push(curr);
            continue;
        }
        let t = radius * tan_half;
        let max_t = in_len.min(out_len) * 0.5;
        if t > max_t {
            out.push(curr);
            continue;
        }
        let tin = [curr[0] - t * in_ux, curr[1] - t * in_uz];
        let tout = [curr[0] + t * out_ux, curr[1] + t * out_uz];
        // Arc center sits perpendicular-left of the incoming edge at radius r.
        let cx = tin[0] + radius * (-in_uz);
        let cz = tin[1] + radius * in_ux;
        let start = (tin[1] - cz).atan2(tin[0] - cx);
        let mut delta = (tout[1] - cz).atan2(tout[0] - cx) - start;
        while delta > std::f32::consts::PI {
            delta -= std::f32::consts::TAU;
        }
        while delta < -std::f32::consts::PI {
            delta += std::f32::consts::TAU;
        }
        for s in 0..=segments {
            let theta = start + delta * (s as f32 / segments as f32);
            out.push([cx + radius * theta.cos(), cz + radius * theta.sin()]);
        }
    }
    out
}

// Ear clipping triangulation for a simple CCW-math polygon.
//
// Falls back to a fan triangulation if no ear is found within a generous
// guard; better to deliver a slightly degenerate mesh than to fail the
// build on unusual user input.
fn ear_clip(profile: &[[f32; 2]]) -> Result<Vec<[usize; 3]>, String> {
    let n = profile.len();
    if n < 3 {
        return Err("ear_clip needs at least 3 vertices".into());
    }
    let mut indices: Vec<usize> = (0..n).collect();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(n.saturating_sub(2));
    let mut guard = 0usize;
    while indices.len() > 3 {
        let m = indices.len();
        let mut clipped = false;
        for i in 0..m {
            let i0 = indices[(i + m - 1) % m];
            let i1 = indices[i];
            let i2 = indices[(i + 1) % m];
            let a = profile[i0];
            let b = profile[i1];
            let c = profile[i2];
            let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
            if cross <= 0.0 {
                continue;
            }
            let mut contains = false;
            for &j in indices.iter() {
                if j == i0 || j == i1 || j == i2 {
                    continue;
                }
                if point_in_triangle(profile[j], a, b, c) {
                    contains = true;
                    break;
                }
            }
            if contains {
                continue;
            }
            tris.push([i0, i1, i2]);
            indices.remove(i);
            clipped = true;
            break;
        }
        guard += 1;
        if !clipped || guard > n * n {
            tris.clear();
            for k in 1..n - 1 {
                tris.push([0, k, k + 1]);
            }
            return Ok(tris);
        }
    }
    if indices.len() == 3 {
        tris.push([indices[0], indices[1], indices[2]]);
    }
    Ok(tris)
}

fn point_in_triangle(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let d1 = side_sign(p, a, b);
    let d2 = side_sign(p, b, c);
    let d3 = side_sign(p, c, a);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

fn side_sign(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extrude_args(profile: serde_json::Value, extras: serde_json::Value) -> serde_json::Value {
        let mut obj = serde_json::Map::new();
        obj.insert("generator".into(), "extrude".into());
        obj.insert("profile".into(), profile);
        if let Some(map) = extras.as_object() {
            for (k, v) in map {
                obj.insert(k.clone(), v.clone());
            }
        }
        serde_json::Value::Object(obj)
    }

    #[test]
    fn build_extrude_square() {
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let (verts, idxs) =
            build_extrude(&extrude_args(profile, serde_json::json!({"height": 2.0}))).unwrap();
        assert!(!verts.is_empty());
        assert!(!idxs.is_empty());
        assert_eq!(idxs.len() % 3, 0);
        // Square: top + bottom (4 each) + 4 side walls (4 verts each) = 24 verts.
        assert_eq!(verts.len(), 24);
    }

    #[test]
    fn build_extrude_rejects_too_few_points() {
        let profile = serde_json::json!([[0, 0], [1, 0]]);
        let err = build_extrude(&extrude_args(profile, serde_json::json!({}))).unwrap_err();
        assert!(err.contains("at least 3"));
    }

    #[test]
    fn build_extrude_rejects_zero_height() {
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let err =
            build_extrude(&extrude_args(profile, serde_json::json!({"height": 0.0}))).unwrap_err();
        assert!(err.contains("positive"));
    }

    #[test]
    fn build_extrude_rejects_negative_corner_radius() {
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let err = build_extrude(&extrude_args(
            profile,
            serde_json::json!({"corner_radius": -0.1}),
        ))
        .unwrap_err();
        assert!(err.contains("non-negative"));
    }

    #[test]
    fn build_extrude_with_rounded_corners_expands_profile() {
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let (verts, _) = build_extrude(&extrude_args(
            profile,
            serde_json::json!({"height": 1.0, "corner_radius": 0.2, "corner_segments": 4}),
        ))
        .unwrap();
        // 4 corners × 5 samples each = 20 profile points; total = 6 × 20 = 120.
        assert_eq!(verts.len(), 120);
    }

    #[test]
    fn build_extrude_handles_clockwise_input() {
        // CW input still produces a valid mesh after orientation normalisation.
        let profile = serde_json::json!([[-1, 1], [1, 1], [1, -1], [-1, -1]]);
        let result = build_extrude(&extrude_args(profile, serde_json::json!({"height": 1.0})));
        assert!(result.is_ok());
    }

    #[test]
    fn build_extrude_top_face_geometric_normal_is_up() {
        // Verify the top cap's first triangle winds so cross(e1, e2) ≈ +Y.
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let (verts, idxs) =
            build_extrude(&extrude_args(profile, serde_json::json!({"height": 2.0}))).unwrap();
        let a = verts[idxs[0] as usize].0;
        let b = verts[idxs[1] as usize].0;
        let c = verts[idxs[2] as usize].0;
        let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        assert!(ny > 0.0, "expected top face normal Y > 0, got {ny}");
    }

    #[test]
    fn build_extrude_side_wall_normal_is_outward() {
        // South wall (z = -1) of the unit square should have a -Z outward normal.
        let profile = serde_json::json!([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
        let (verts, _) =
            build_extrude(&extrude_args(profile, serde_json::json!({"height": 1.0}))).unwrap();
        // Top + bottom = 8 verts; first wall verts begin at index 8.
        let n = verts[8].1;
        assert!(
            n[2] < -0.99,
            "expected south wall normal ≈ (0,0,-1), got {n:?}"
        );
    }

    #[test]
    fn round_corners_respects_max_radius() {
        // A triangle with 1-unit edges can't fit a radius-1 round; corners pass through.
        let profile = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let rounded = round_corners(&profile, 1.0, 4);
        assert_eq!(
            rounded.len(),
            3,
            "no corner should round when r > max edge/2"
        );
    }

    #[test]
    fn ear_clip_triangle() {
        let profile = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tris = ear_clip(&profile).unwrap();
        assert_eq!(tris.len(), 1);
    }

    #[test]
    fn ear_clip_concave_polygon() {
        // L-shape (concave) should still produce a valid triangulation.
        let profile = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ];
        let tris = ear_clip(&profile).unwrap();
        assert_eq!(tris.len(), 4); // n - 2 triangles for a simple polygon
    }
}
