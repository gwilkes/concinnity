// src/build/wavefront.rs
//
// Parses Wavefront .obj content into deduplicated mesh vertices and indices
// suitable for a Mesh asset. Normals are omitted: the build pipeline computes
// them from the geometry.
//
// Supported directives: v, vt, vn (ignored), f, o, g, s, mtllib, usemtl, #
// Face vertices may be v, v/vt, v/vt/vn, or v//vn.
// Polygons with more than 3 vertices are triangulated with a fan.

use crate::assets::VertexData;
use std::collections::HashMap;

// Neutral grey for vertex color, takes material albedo without tinting.
const NEUTRAL_COLOR: [f32; 3] = [0.75, 0.74, 0.72];

// Sentinel for "no UV index present on this face corner".
const NO_UV: u32 = u32::MAX;

// Parse Wavefront .obj content into (vertices, indices) for a Mesh asset.
//
// Vertices are deduplicated: two face corners sharing the same position and UV
// index map to a single entry in the output vertex list.
pub fn parse_obj(content: &str) -> Result<(Vec<VertexData>, Vec<u16>), String> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    let mut vertex_map: HashMap<(u32, u32), u16> = HashMap::new();
    let mut vertices: Vec<VertexData> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();

    for (line_idx, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let mut tokens = line.split_whitespace();
        let keyword = match tokens.next() {
            Some(k) => k,
            None => continue,
        };

        match keyword {
            "v" => {
                let x = parse_f32(tokens.next(), line_idx, "x")?;
                let y = parse_f32(tokens.next(), line_idx, "y")?;
                let z = parse_f32(tokens.next(), line_idx, "z")?;
                positions.push([x, y, z]);
            }

            "vt" => {
                let u = parse_f32(tokens.next(), line_idx, "u")?;
                // v is optional (defaults to 0)
                let v = tokens
                    .next()
                    .map(|s| s.parse::<f32>().unwrap_or(0.0))
                    .unwrap_or(0.0);
                uvs.push([u, v]);
            }

            // Normals are ignored: the build pipeline recomputes them.
            "vn" => {}

            "f" => {
                let refs: Vec<&str> = tokens.collect();
                if refs.len() < 3 {
                    return Err(format!(
                        "line {}: face with fewer than 3 vertices",
                        line_idx + 1
                    ));
                }

                // Resolve each corner to a deduplicated vertex index.
                let mut corners: Vec<u16> = Vec::with_capacity(refs.len());
                for ref_str in &refs {
                    let key = parse_face_ref(ref_str, &positions, &uvs, line_idx)?;
                    let vi = match vertex_map.get(&key) {
                        Some(&i) => i,
                        None => {
                            if vertices.len() >= u16::MAX as usize {
                                return Err(format!(
                                    "mesh exceeds {} unique vertices (u16 index limit)",
                                    u16::MAX
                                ));
                            }
                            let i = vertices.len() as u16;
                            let uv = if key.1 == NO_UV {
                                [0.0, 0.0]
                            } else {
                                uvs[key.1 as usize]
                            };
                            vertices.push(VertexData {
                                pos: positions[key.0 as usize],
                                color: NEUTRAL_COLOR,
                                uv,
                            });
                            vertex_map.insert(key, i);
                            i
                        }
                    };
                    corners.push(vi);
                }

                // Fan triangulation: (0,1,2), (0,2,3), (0,3,4), …
                for i in 1..corners.len() - 1 {
                    indices.push(corners[0]);
                    indices.push(corners[i]);
                    indices.push(corners[i + 1]);
                }
            }

            // Silently skip all other directives (o, g, s, mtllib, usemtl, …).
            _ => {}
        }
    }

    if vertices.is_empty() {
        return Err("no vertices found in .obj file".to_string());
    }
    if indices.is_empty() {
        return Err("no faces found in .obj file".to_string());
    }

    Ok((vertices, indices))
}

// Parse a face corner reference like "v", "v/vt", "v/vt/vn", or "v//vn".
// Returns a (pos_idx, uv_idx) key using 0-based indices; uv_idx is NO_UV
// when no texture coordinate is specified.
fn parse_face_ref(
    s: &str,
    positions: &[[f32; 3]],
    uvs: &[[f32; 2]],
    line_idx: usize,
) -> Result<(u32, u32), String> {
    let parts: Vec<&str> = s.splitn(3, '/').collect();

    let pos_raw: i32 = parts[0]
        .parse()
        .map_err(|_| format!("line {}: invalid position index '{}'", line_idx + 1, s))?;
    let pos_idx = resolve_index(pos_raw, positions.len(), line_idx, "position")?;

    let uv_idx = if parts.len() >= 2 && !parts[1].is_empty() {
        let raw: i32 = parts[1]
            .parse()
            .map_err(|_| format!("line {}: invalid UV index '{}'", line_idx + 1, s))?;
        resolve_index(raw, uvs.len(), line_idx, "UV")?
    } else {
        NO_UV
    };

    Ok((pos_idx, uv_idx))
}

// Convert a 1-based (or negative relative) OBJ index to a 0-based array index.
fn resolve_index(raw: i32, len: usize, line_idx: usize, kind: &str) -> Result<u32, String> {
    if raw == 0 {
        return Err(format!(
            "line {}: {} index 0 is invalid in OBJ format",
            line_idx + 1,
            kind
        ));
    }
    let idx = if raw > 0 {
        (raw - 1) as usize
    } else {
        let abs = (-raw) as usize;
        if abs > len {
            return Err(format!(
                "line {}: {} index {} out of range ({} {}s defined so far)",
                line_idx + 1,
                kind,
                raw,
                len,
                kind
            ));
        }
        len - abs
    };
    if idx >= len {
        return Err(format!(
            "line {}: {} index {} out of range ({} {}s defined)",
            line_idx + 1,
            kind,
            raw,
            len,
            kind
        ));
    }
    Ok(idx as u32)
}

fn parse_f32(token: Option<&str>, line_idx: usize, field: &str) -> Result<f32, String> {
    token
        .ok_or_else(|| format!("line {}: missing {} component", line_idx + 1, field))?
        .parse::<f32>()
        .map_err(|_| format!("line {}: invalid float for {}", line_idx + 1, field))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_obj() -> &'static str {
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3"
    }

    #[test]
    fn parse_triangle_vertex_count() {
        let (verts, _) = parse_obj(triangle_obj()).unwrap();
        assert_eq!(verts.len(), 3);
    }

    #[test]
    fn parse_triangle_index_count() {
        let (_, idxs) = parse_obj(triangle_obj()).unwrap();
        assert_eq!(idxs.len(), 3);
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn parse_triangle_positions() {
        let (verts, _) = parse_obj(triangle_obj()).unwrap();
        assert_eq!(verts[0].pos, [0.0, 0.0, 0.0]);
        assert_eq!(verts[1].pos, [1.0, 0.0, 0.0]);
        assert_eq!(verts[2].pos, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn parse_triangle_neutral_color() {
        let (verts, _) = parse_obj(triangle_obj()).unwrap();
        for v in &verts {
            assert_eq!(v.color, NEUTRAL_COLOR);
        }
    }

    #[test]
    fn parse_triangle_default_uv() {
        let (verts, _) = parse_obj(triangle_obj()).unwrap();
        for v in &verts {
            assert_eq!(v.uv, [0.0, 0.0]);
        }
    }

    #[test]
    fn parse_quad_triangulates_to_two_triangles() {
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4";
        let (verts, idxs) = parse_obj(obj).unwrap();
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs.len(), 6); // two triangles
    }

    #[test]
    fn parse_uv_coordinates() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.0 0.0\nvt 1.0 0.0\nvt 0.0 1.0\nf 1/1 2/2 3/3";
        let (verts, _) = parse_obj(obj).unwrap();
        assert_eq!(verts[0].uv, [0.0, 0.0]);
        assert_eq!(verts[1].uv, [1.0, 0.0]);
        assert_eq!(verts[2].uv, [0.0, 1.0]);
    }

    #[test]
    fn deduplicates_shared_vertices() {
        // Two triangles sharing an edge: shared corners should map to the same vertex.
        let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4";
        let (verts, idxs) = parse_obj(obj).unwrap();
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs.len(), 6);
    }

    #[test]
    fn same_pos_different_uv_makes_separate_vertices() {
        // Vertex 1 appears at two different UV coords, must not be deduplicated.
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.0 0.0\nvt 0.5 0.5\nvt 1.0 0.0\nf 1/1 2/3 3/2\nf 1/2 2/3 3/1";
        let (verts, _) = parse_obj(obj).unwrap();
        // Vertex 1 used with uv 1 and uv 2 → 2 entries for it
        assert!(verts.len() > 3);
    }

    #[test]
    fn ignores_comments_and_blank_lines() {
        let obj = "# comment\n\nv 0 0 0\nv 1 0 0\nv 0 1 0\n\n# another\nf 1 2 3";
        assert!(parse_obj(obj).is_ok());
    }

    #[test]
    fn ignores_vn_normals() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1";
        let (verts, _) = parse_obj(obj).unwrap();
        assert_eq!(verts.len(), 3);
    }

    #[test]
    fn ignores_unknown_directives() {
        let obj = "mtllib mat.mtl\no MyObject\ng group\ns 1\nusemtl Mat\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3";
        assert!(parse_obj(obj).is_ok());
    }

    #[test]
    fn negative_indices() {
        // After defining 3 vertices, -1 refers to the last one.
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1";
        let (verts, idxs) = parse_obj(obj).unwrap();
        assert_eq!(verts.len(), 3);
        assert_eq!(idxs, vec![0, 1, 2]);
    }

    #[test]
    fn error_on_empty_file() {
        assert!(parse_obj("").is_err());
    }

    #[test]
    fn error_on_positions_only_no_faces() {
        assert!(parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0").is_err());
    }

    #[test]
    fn error_on_out_of_range_index() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99";
        assert!(parse_obj(obj).is_err());
    }

    #[test]
    fn error_on_zero_index() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2";
        assert!(parse_obj(obj).is_err());
    }
}
