// src/world/config.rs

pub const DEFAULT_MAX_BLOB_BYTES: u64 = 1 << 30;

// Build-time configuration for the world pipeline.
// Constructed by the client; not declared as a world asset.
pub struct WorldConfig {
    pub max_blob_bytes: u64,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            max_blob_bytes: DEFAULT_MAX_BLOB_BYTES,
        }
    }
}
