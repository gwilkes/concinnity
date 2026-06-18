// src/geometry/heightfield.rs: subdivided terrain grid driven by a grayscale
// heightmap image.
//
// Sibling of terrain.rs. Same XZ grid + smooth-normal pass; the only difference
// is the height function: instead of three octaves of LCG-hash noise, this
// generator samples a heightmap image and maps the red channel through the
// configured elevation range.
//
// The source image is decoded by the build crate (this crate links no image
// decoders) and the RGBA pixels are passed in; `compile_heightfield_payload`
// in the parent module is the caller. The geometry is otherwise produced here
// so the collider trailer and tangent / LOD pipeline stay shared with the
// other generators.
//
// Parameters (read from the asset args):
//   half_width    -- half the terrain extent along X (default 64.0)
//   half_depth    -- half the terrain extent along Z (default 64.0)
//   subdivisions  -- grid resolution per axis (default 64, clamped 4..=255)
//   elevation_min -- world Y at image value 0   (default 0.0)
//   elevation_max -- world Y at image value 255 (required)

type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

pub(super) fn build_heightfield_from_pixels(
    args: &serde_json::Value,
    img_w: u32,
    img_h: u32,
    rgba: &[u8],
) -> Result<(Verts, Vec<u16>), String> {
    let half_width = args
        .get("half_width")
        .and_then(|v| v.as_f64())
        .unwrap_or(64.0) as f32;
    let half_depth = args
        .get("half_depth")
        .and_then(|v| v.as_f64())
        .unwrap_or(64.0) as f32;
    let subdivisions = args
        .get("subdivisions")
        .and_then(|v| v.as_u64())
        .unwrap_or(64)
        .clamp(4, 255) as usize;
    let elevation_min = args
        .get("elevation_min")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32;
    let elevation_max = args
        .get("elevation_max")
        .and_then(|v| v.as_f64())
        .ok_or("heightfield generator requires `elevation_max`")? as f32;

    if img_w == 0 || img_h == 0 {
        return Err("heightfield source image has zero extent".to_string());
    }

    let needed = (img_w as usize) * (img_h as usize) * 4;
    if rgba.len() < needed {
        return Err(format!(
            "heightfield source image buffer too small: have {}, need {} for {}x{}",
            rgba.len(),
            needed,
            img_w,
            img_h
        ));
    }

    let cols = subdivisions + 1;
    let rows = subdivisions + 1;

    if cols * rows > 65536 {
        return Err(format!(
            "heightfield subdivisions {} produces {} vertices, exceeding the u16 limit; use subdivisions ≤ 255",
            subdivisions,
            cols * rows
        ));
    }

    let color = [0.55f32, 0.62, 0.42];

    // Pre-sample the heightmap to per-vertex Y. Bilinear filter so the mesh
    // doesn't inherit the heightmap's pixel grid when subdivisions and image
    // resolution differ.
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let s = col as f32 / subdivisions as f32;
            let t = row as f32 / subdivisions as f32;
            let x = -half_width + s * half_width * 2.0;
            let z = -half_depth + t * half_depth * 2.0;
            let y = sample_height_bilinear(rgba, img_w, img_h, s, t, elevation_min, elevation_max);
            positions.push([x, y, z]);
        }
    }

    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; cols * rows];
    for row in 0..subdivisions {
        for col in 0..subdivisions {
            let tl = row * cols + col;
            let tr = tl + 1;
            let bl = tl + cols;
            let br = bl + 1;
            let n1 = super::vec3_face_normal(positions[tl], positions[bl], positions[tr]);
            super::vec3_add(&mut normals[tl], n1);
            super::vec3_add(&mut normals[bl], n1);
            super::vec3_add(&mut normals[tr], n1);
            let n2 = super::vec3_face_normal(positions[tr], positions[bl], positions[br]);
            super::vec3_add(&mut normals[tr], n2);
            super::vec3_add(&mut normals[bl], n2);
            super::vec3_add(&mut normals[br], n2);
        }
    }

    let mut idxs: Vec<u16> = Vec::with_capacity(subdivisions * subdivisions * 6);
    let mut verts: Verts = Vec::with_capacity(cols * rows);

    for i in 0..cols * rows {
        let [x, y, z] = positions[i];
        let normal = super::vec3_normalise(normals[i]);
        verts.push(([x, y, z], normal, color, [x, z]));
    }

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

// Bilinear-sample the heightmap's red channel at normalised UV (s, t) in [0,1]
// and map [0, 255] to [elevation_min, elevation_max].
fn sample_height_bilinear(
    rgba: &[u8],
    img_w: u32,
    img_h: u32,
    s: f32,
    t: f32,
    elevation_min: f32,
    elevation_max: f32,
) -> f32 {
    let fx = s.clamp(0.0, 1.0) * (img_w - 1) as f32;
    let fy = t.clamp(0.0, 1.0) * (img_h - 1) as f32;
    let x0 = fx.floor() as u32;
    let y0 = fy.floor() as u32;
    let x1 = (x0 + 1).min(img_w - 1);
    let y1 = (y0 + 1).min(img_h - 1);
    let sx = fx - x0 as f32;
    let sy = fy - y0 as f32;

    let r = |x: u32, y: u32| -> f32 {
        let idx = (y * img_w + x) as usize * 4;
        rgba[idx] as f32 / 255.0
    };
    let top = r(x0, y0) + (r(x1, y0) - r(x0, y0)) * sx;
    let bot = r(x0, y1) + (r(x1, y1) - r(x0, y1)) * sx;
    let h = top + (bot - top) * sy;
    elevation_min + h * (elevation_max - elevation_min)
}

#[cfg(test)]
mod tests {
    use super::*;

    // A 4×4 grayscale-RGBA buffer with a known ramp pattern across the X axis:
    // row 0 = 0,85,170,255; subsequent rows duplicate. Each pixel is [r,r,r,255].
    fn ramp_4x4_rgba() -> Vec<u8> {
        let mut out = Vec::with_capacity(4 * 4 * 4);
        let row_values: [u8; 4] = [0, 85, 170, 255];
        for _ in 0..4 {
            for v in row_values {
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        out
    }

    #[test]
    fn bilinear_sample_recovers_corner_values() {
        let rgba = ramp_4x4_rgba();
        // Corners of the image map to elevation_min..elevation_max via the
        // ramp's 0 / 255 endpoints (cols 0 and 3 of the row).
        let h_min = sample_height_bilinear(&rgba, 4, 4, 0.0, 0.0, -1.0, 1.0);
        let h_max = sample_height_bilinear(&rgba, 4, 4, 1.0, 0.0, -1.0, 1.0);
        assert!((h_min - -1.0).abs() < 1e-5, "h_min = {}", h_min);
        assert!((h_max - 1.0).abs() < 1e-5, "h_max = {}", h_max);
    }

    #[test]
    fn bilinear_midpoint_interpolates() {
        // Between cols 0 (value 0) and 3 (value 255) at s=0.5 the bilinear
        // sample averages the four neighbours; cols 1 and 2 are 85 and 170
        // (avg 127.5). At s=0.5 / t=0.0, fx = 1.5, so the sample averages
        // cols 1 + 2 = (85 + 170) / 2 = 127.5 → 127.5/255 ≈ 0.5.
        let rgba = ramp_4x4_rgba();
        let h = sample_height_bilinear(&rgba, 4, 4, 0.5, 0.0, 0.0, 1.0);
        assert!((h - 0.5).abs() < 0.01, "h = {}", h);
    }

    // A `w`x`h` grayscale-RGBA buffer whose red channel ramps 0..255 across X
    // so the generated mesh has real elevation variation to sample.
    fn ramp_rgba(w: u32, h: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..h {
            for x in 0..w {
                let v = if w > 1 { (x * 255 / (w - 1)) as u8 } else { 0 };
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        out
    }

    #[test]
    fn requires_elevation_max() {
        // Build a tiny grid; the missing `elevation_max` arg is a hard error.
        let args = serde_json::json!({ "subdivisions": 3 });
        let rgba = ramp_rgba(4, 4);
        let err = build_heightfield_from_pixels(&args, 4, 4, &rgba).unwrap_err();
        assert!(err.contains("elevation_max"), "got: {}", err);
    }

    #[test]
    fn rejects_zero_extent_image() {
        let args = serde_json::json!({ "subdivisions": 3, "elevation_max": 1.0 });
        let err = build_heightfield_from_pixels(&args, 0, 0, &[]).unwrap_err();
        assert!(err.contains("zero extent"), "got: {}", err);
    }

    #[test]
    fn vertex_and_index_counts_match_grid() {
        // subdivisions=4 -> 5x5 = 25 verts, 4*4*2 = 32 tris -> 96 indices.
        let args = serde_json::json!({
            "half_width": 5.0,
            "half_depth": 5.0,
            "subdivisions": 4,
            "elevation_min": 0.0,
            "elevation_max": 10.0,
        });
        let rgba = ramp_rgba(8, 8);
        let (verts, idxs) = build_heightfield_from_pixels(&args, 8, 8, &rgba).expect("builds");
        assert_eq!(verts.len(), 5 * 5);
        assert_eq!(idxs.len(), 4 * 4 * 6);

        // The ramp gives real elevation variation bracketed by the range.
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for v in &verts {
            min_y = min_y.min(v.0[1]);
            max_y = max_y.max(v.0[1]);
        }
        assert!(min_y >= 0.0);
        assert!(max_y <= 10.0);
        assert!(
            max_y > min_y,
            "expected variation but got flat at {}",
            max_y
        );
    }
}
