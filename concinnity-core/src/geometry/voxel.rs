// src/geometry/voxel.rs
//
// Hidden-face mesher for VoxelChunk assets.
//
// For each block whose palette entry has solid=true, emit a quad for any of
// its six faces whose neighbour is either outside the chunk or non-solid.
// Faces between two solid blocks are skipped entirely, so the interior of a
// filled volume contributes no triangles.
//
// UVs come from the BlockType palette: per-face overrides (uv_top, uv_bottom,
// uv_side) fall back to uv_min/uv_max when None.
//
// Greedy merging of adjacent same-block faces into larger quads is a future
// optimisation; this pass only does hidden-face culling.

// One entry resolved from a VoxelChunk palette.  `None` slots are
// non-solid (air) and emit no geometry; their neighbours treat them as empty.
pub(crate) struct PaletteSlot {
    pub uv_top: [f32; 4],
    pub uv_bottom: [f32; 4],
    pub uv_side: [f32; 4],
}

// Output type for the geometry builders in this module (compatible with the
// rest of the pipeline in src/geometry.rs).
type Verts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;

// Generate hidden-face-culled geometry for a voxel chunk.
//
// `dim` is `[dx, dy, dz]`; `blocks.len()` must equal `dx*dy*dz`. Each block
// id either indexes a `Some(slot)` (solid) or `None` (air). The chunk origin
// is at the local-space origin (`0,0,0` corner); the far corner is at
// `(dx*block_size, dy*block_size, dz*block_size)`.
pub(crate) fn build_voxel_mesh(
    dim: [u32; 3],
    block_size: f32,
    blocks: &[u32],
    palette: &[Option<PaletteSlot>],
) -> Result<(Verts, Vec<u16>), String> {
    let [dx, dy, dz] = [dim[0] as usize, dim[1] as usize, dim[2] as usize];
    let expected = dx.saturating_mul(dy).saturating_mul(dz);
    if blocks.len() != expected {
        return Err(format!(
            "VoxelChunk: blocks length {} does not match dim {}x{}x{} ({} expected)",
            blocks.len(),
            dim[0],
            dim[1],
            dim[2],
            expected
        ));
    }
    for (i, &id) in blocks.iter().enumerate() {
        if (id as usize) >= palette.len() {
            return Err(format!(
                "VoxelChunk: blocks[{}] = {} out of palette range (len {})",
                i,
                id,
                palette.len()
            ));
        }
    }

    let mut verts: Verts = Vec::new();
    let mut idxs: Vec<u16> = Vec::new();
    let color = [0.75f32, 0.74, 0.72];
    let bs = block_size;

    let at = |x: i32, y: i32, z: i32| -> Option<&PaletteSlot> {
        if x < 0 || y < 0 || z < 0 || x >= dx as i32 || y >= dy as i32 || z >= dz as i32 {
            return None;
        }
        let i = (x as usize) + (y as usize) * dx + (z as usize) * dx * dy;
        palette[blocks[i] as usize].as_ref()
    };

    // 6 face emitters. Each writes a quad CCW from outside the block.
    let mut emit_quad = |corners: [[f32; 3]; 4], normal: [f32; 3], uv_rect: [f32; 4]| {
        if verts.len() + 4 > u16::MAX as usize {
            return;
        }
        let base = verts.len() as u16;
        let [u0, v0, u1, v1] = uv_rect;
        // CCW from outside; UVs map a -> (u0,v0), b -> (u1,v0), c -> (u1,v1), d -> (u0,v1).
        let uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
        for (i, p) in corners.iter().enumerate() {
            verts.push((*p, normal, color, uvs[i]));
        }
        idxs.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    };

    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let slot = match at(x as i32, y as i32, z as i32) {
                    Some(s) => s,
                    None => continue,
                };
                let x0 = x as f32 * bs;
                let y0 = y as f32 * bs;
                let z0 = z as f32 * bs;
                let x1 = x0 + bs;
                let y1 = y0 + bs;
                let z1 = z0 + bs;

                // +X
                if at(x as i32 + 1, y as i32, z as i32).is_none() {
                    emit_quad(
                        [[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]],
                        [1.0, 0.0, 0.0],
                        slot.uv_side,
                    );
                }
                // -X
                if at(x as i32 - 1, y as i32, z as i32).is_none() {
                    emit_quad(
                        [[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]],
                        [-1.0, 0.0, 0.0],
                        slot.uv_side,
                    );
                }
                // +Y (top)
                if at(x as i32, y as i32 + 1, z as i32).is_none() {
                    emit_quad(
                        [[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]],
                        [0.0, 1.0, 0.0],
                        slot.uv_top,
                    );
                }
                // -Y (bottom)
                if at(x as i32, y as i32 - 1, z as i32).is_none() {
                    emit_quad(
                        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],
                        [0.0, -1.0, 0.0],
                        slot.uv_bottom,
                    );
                }
                // +Z
                if at(x as i32, y as i32, z as i32 + 1).is_none() {
                    emit_quad(
                        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
                        [0.0, 0.0, 1.0],
                        slot.uv_side,
                    );
                }
                // -Z
                if at(x as i32, y as i32, z as i32 - 1).is_none() {
                    emit_quad(
                        [[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]],
                        [0.0, 0.0, -1.0],
                        slot.uv_side,
                    );
                }
            }
        }
    }

    Ok((verts, idxs))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_slot() -> PaletteSlot {
        PaletteSlot {
            uv_top: [0.0, 0.0, 1.0, 1.0],
            uv_bottom: [0.0, 0.0, 1.0, 1.0],
            uv_side: [0.0, 0.0, 1.0, 1.0],
        }
    }

    #[test]
    fn empty_chunk_produces_no_geometry() {
        let palette = vec![None];
        let (v, i) = build_voxel_mesh([2, 2, 2], 1.0, &[0; 8], &palette).unwrap();
        assert!(v.is_empty());
        assert!(i.is_empty());
    }

    #[test]
    fn single_solid_block_has_six_faces() {
        let palette = vec![None, Some(solid_slot())];
        let (v, i) = build_voxel_mesh([1, 1, 1], 1.0, &[1], &palette).unwrap();
        // 6 faces, 4 verts each, 6 indices each
        assert_eq!(v.len(), 24);
        assert_eq!(i.len(), 36);
    }

    #[test]
    fn interior_faces_are_culled() {
        // 2x1x1 of two solid blocks side by side: shared face hidden, 10 faces total.
        let palette = vec![None, Some(solid_slot())];
        let (v, i) = build_voxel_mesh([2, 1, 1], 1.0, &[1, 1], &palette).unwrap();
        assert_eq!(v.len(), 10 * 4);
        assert_eq!(i.len(), 10 * 6);
    }

    #[test]
    fn fully_filled_cube_has_only_outer_shell() {
        // 3x3x3 of one solid block: 27 blocks, only the 6 outer faces × 9 cells = 54 faces.
        let palette = vec![None, Some(solid_slot())];
        let blocks = vec![1u32; 27];
        let (v, _) = build_voxel_mesh([3, 3, 3], 1.0, &blocks, &palette).unwrap();
        assert_eq!(v.len(), 54 * 4);
    }

    #[test]
    fn air_block_around_solid_emits_all_six_faces() {
        // 3x3x3 with the center cell solid: only 6 faces.
        let palette = vec![None, Some(solid_slot())];
        let mut blocks = vec![0u32; 27];
        let center = 1 + 3 + 9;
        blocks[center] = 1;
        let (v, _) = build_voxel_mesh([3, 3, 3], 1.0, &blocks, &palette).unwrap();
        assert_eq!(v.len(), 6 * 4);
    }

    #[test]
    fn mismatched_blocks_length_errors() {
        let palette = vec![None];
        let result = build_voxel_mesh([2, 2, 2], 1.0, &[0; 4], &palette);
        assert!(result.is_err());
    }

    #[test]
    fn block_index_out_of_palette_range_errors() {
        let palette = vec![None];
        let result = build_voxel_mesh([1, 1, 1], 1.0, &[7], &palette);
        assert!(result.is_err());
    }
}
