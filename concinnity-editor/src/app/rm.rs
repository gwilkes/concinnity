// src/app/rm.rs
// Remove an asset from a world JSONL by its unique `name` field and rebuild.

use crate::world::{WORLD_JSONL, known_names, patch_world_jsonl};
use concinnity_cook::build_from_path;

// Remove the asset named `name` from `world_path` and rebuild.
//
// Errors if `name` is not present. When it isn't, the error message includes
// the known asset names from the world so the caller can suggest a fix.
pub fn rm_at_path(world_path: &str, name: &str) -> std::io::Result<()> {
    let mut removed = false;

    patch_world_jsonl(world_path, |assets| {
        if let Some(i) = assets
            .iter()
            .position(|a| a.get("name").and_then(|v| v.as_str()) == Some(name))
        {
            let asset = assets.remove(i);
            tracing::info!(
                "Removed '{}' (type: {})",
                name,
                asset.get("type").and_then(|v| v.as_str()).unwrap_or("?"),
            );
            removed = true;
        }
    })?;

    if !removed {
        let known = known_names(world_path).unwrap_or_default();
        if known.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "no asset named '{}' in {} (no assets declared)",
                    name, WORLD_JSONL
                ),
            ));
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "no asset named '{}' in {}\nKnown names: {}",
                name,
                WORLD_JSONL,
                known.join(", ")
            ),
        ));
    }

    build_from_path(world_path)
}
