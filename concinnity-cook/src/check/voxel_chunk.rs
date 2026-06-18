// Structural validation for VoxelChunk args. Cross-asset palette lookups are
// handled by crate::check::cross_reference::validate_cross_references; this check
// only catches problems we can see from the chunk's own args alone.

pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let dim = args.get("dim").and_then(|v| v.as_array()).ok_or_else(|| {
        format!(
            "Asset '{}': VoxelChunk `dim` must be a [dx, dy, dz] array",
            name
        )
    })?;
    if dim.len() < 3 {
        return Err(format!(
            "Asset '{}': VoxelChunk `dim` must have 3 elements, got {}",
            name,
            dim.len()
        ));
    }
    let dims: [u64; 3] = [
        dim[0].as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk dim[0] must be a non-negative integer",
                name
            )
        })?,
        dim[1].as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk dim[1] must be a non-negative integer",
                name
            )
        })?,
        dim[2].as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk dim[2] must be a non-negative integer",
                name
            )
        })?,
    ];
    let expected = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);

    let blocks = args
        .get("blocks")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk `blocks` must be an array of palette indices",
                name
            )
        })?;
    if (blocks.len() as u64) != expected {
        return Err(format!(
            "Asset '{}': VoxelChunk blocks length {} does not match dim {}x{}x{} ({} expected)",
            name,
            blocks.len(),
            dims[0],
            dims[1],
            dims[2],
            expected
        ));
    }

    let palette_len = args
        .get("palette")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    for (i, entry) in blocks.iter().enumerate() {
        let idx = entry.as_u64().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelChunk blocks[{}] must be a non-negative integer",
                name, i
            )
        })?;
        if palette_len == 0 || (idx as usize) >= palette_len {
            return Err(format!(
                "Asset '{}': VoxelChunk blocks[{}] = {} out of palette range (len {})",
                name, i, idx, palette_len
            ));
        }
    }

    Ok(())
}
