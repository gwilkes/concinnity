// Structural validation for VoxelWorld args. Cross-asset palette/material
// lookups are handled by build/pipeline.rs::validate_cross_references; this
// check only catches problems visible from the world's own args.

pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    // palette: at least air + one solid block so the generator has geometry.
    let palette = args
        .get("palette")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            format!(
                "Asset '{}': VoxelWorld `palette` must be an array of BlockType names",
                name
            )
        })?;
    if palette.len() < 2 {
        return Err(format!(
            "Asset '{}': VoxelWorld `palette` needs at least 2 entries (air + a surface block), got {}",
            name,
            palette.len()
        ));
    }

    // chunk_blocks is optional (defaults apply), but if present must be a
    // 3-element array of positive integers.
    if let Some(cb) = args.get("chunk_blocks") {
        let arr = cb.as_array().ok_or_else(|| {
            format!(
                "Asset '{}': VoxelWorld `chunk_blocks` must be a [dx, dy, dz] array",
                name
            )
        })?;
        if arr.len() != 3 {
            return Err(format!(
                "Asset '{}': VoxelWorld `chunk_blocks` must have 3 elements, got {}",
                name,
                arr.len()
            ));
        }
        for (i, e) in arr.iter().enumerate() {
            let v = e.as_u64().ok_or_else(|| {
                format!(
                    "Asset '{}': VoxelWorld chunk_blocks[{}] must be a non-negative integer",
                    name, i
                )
            })?;
            if v == 0 {
                return Err(format!(
                    "Asset '{}': VoxelWorld chunk_blocks[{}] must be greater than 0",
                    name, i
                ));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_args_pass() {
        let args = serde_json::json!({
            "palette": ["air", "grass", "stone"],
            "chunk_blocks": [16, 24, 16]
        });
        assert!(check("w", &args).is_ok());
    }

    #[test]
    fn missing_palette_fails() {
        assert!(check("w", &serde_json::json!({})).is_err());
    }

    #[test]
    fn single_entry_palette_fails() {
        let args = serde_json::json!({ "palette": ["air"] });
        assert!(check("w", &args).is_err());
    }

    #[test]
    fn zero_chunk_dimension_fails() {
        let args = serde_json::json!({
            "palette": ["air", "stone"],
            "chunk_blocks": [16, 0, 16]
        });
        assert!(check("w", &args).is_err());
    }

    #[test]
    fn wrong_chunk_blocks_length_fails() {
        let args = serde_json::json!({
            "palette": ["air", "stone"],
            "chunk_blocks": [16, 16]
        });
        assert!(check("w", &args).is_err());
    }
}
