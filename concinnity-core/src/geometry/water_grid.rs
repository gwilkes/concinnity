// src/geometry/water_grid.rs: flat tessellated quad for a WaterSurface.
//
// The mesh sits in the XZ plane at Y = 0. All vertical motion comes from the
// per-frame Gerstner displacement applied by the water vertex shader; the
// build-time geometry is just the rest pose. Per-vertex normals are flat
// (+Y); the shader rebuilds them analytically from the wave derivatives.
//
// Parameters:
//   half_width    -- half extent along X    (default 10.0)
//   half_depth    -- half extent along Z    (default 10.0)
//   subdivisions  -- grid resolution per axis (default 64, clamped 8..=255)

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

pub fn build_water_grid(args: &serde_json::Value) -> Result<(Verts, Vec<u16>), String> {
    let half_width = args
        .get("half_width")
        .and_then(|v| v.as_f64())
        .unwrap_or(10.0) as f32;
    let half_depth = args
        .get("half_depth")
        .and_then(|v| v.as_f64())
        .unwrap_or(10.0) as f32;
    let subdivisions = args
        .get("subdivisions")
        .and_then(|v| v.as_u64())
        .unwrap_or(64)
        .clamp(8, 255) as usize;

    let cols = subdivisions + 1;
    let rows = subdivisions + 1;

    if cols * rows > 65536 {
        return Err(format!(
            "water_grid subdivisions {} produces {} vertices, exceeding the u16 limit; use subdivisions ≤ 255",
            subdivisions,
            cols * rows
        ));
    }

    let normal = [0.0f32, 1.0, 0.0];
    let color = [1.0f32, 1.0, 1.0];
    let mut verts: Verts = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let s = col as f32 / subdivisions as f32;
            let t = row as f32 / subdivisions as f32;
            let x = -half_width + s * half_width * 2.0;
            let z = -half_depth + t * half_depth * 2.0;
            verts.push(([x, 0.0, z], normal, color, [s, t]));
        }
    }

    let mut idxs: Vec<u16> = Vec::with_capacity(subdivisions * subdivisions * 6);
    for row in 0..subdivisions {
        for col in 0..subdivisions {
            let tl = (row * cols + col) as u16;
            let tr = tl + 1;
            let bl = tl + cols as u16;
            let br = bl + 1;
            idxs.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
        }
    }

    Ok((verts, idxs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_and_index_counts_match_grid() {
        let args = serde_json::json!({
            "half_width": 5.0,
            "half_depth": 5.0,
            "subdivisions": 8,
        });
        let (verts, idxs) = build_water_grid(&args).expect("builds");
        assert_eq!(verts.len(), 9 * 9);
        assert_eq!(idxs.len(), 8 * 8 * 6);
        // All vertices on Y = 0.
        for v in &verts {
            assert!(v.0[1].abs() < 1e-6);
        }
        // Corner positions span the half-widths.
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        for v in &verts {
            min_x = min_x.min(v.0[0]);
            max_x = max_x.max(v.0[0]);
        }
        assert!((min_x - -5.0).abs() < 1e-5);
        assert!((max_x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn subdivisions_clamps_to_minimum() {
        let args = serde_json::json!({"subdivisions": 2});
        let (verts, _) = build_water_grid(&args).expect("builds");
        // Clamped to 8 → 9x9 grid.
        assert_eq!(verts.len(), 9 * 9);
    }
}
