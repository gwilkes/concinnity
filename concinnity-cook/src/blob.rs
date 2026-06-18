// The blob WRITE half (build output): pack compiled payloads + the def table
// into .cnb files and emit world-lock.json. The READ half (BlobData, load_raw,
// read_cnb, payload_section_start, load_defs) stays in concinnity-core and is
// re-exported here so callers can keep using `blob::...` uniformly.

use std::fs;

use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use concinnity_core::blob::{BLOB_MAGIC, BLOB_VERSION, HEADER_SIZE, LOCK_PATH, PRIMARY_CNB};
use concinnity_core::ecs::{BlobAssetDef, PayloadLocator};

// Re-export the read side from core so `crate::blob::{BlobData, load_raw, ...}`
// resolves for build-crate consumers that read blobs back. `blob_path` is shared
// (used by write_blobs below and by readers).
pub use concinnity_core::blob::{
    BlobData, blob_path, load_defs, load_raw, payload_section_start, read_cnb,
};

use serde::{Deserialize, Serialize};

// Per-blob entry in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct BlobEntry {
    pub path: String,
    pub checksum: String,
    pub payload_bytes: u64,
}

// The resolved build record written alongside the binary blobs
// Human-readable; owned by the build, not the user
#[derive(Debug, Serialize, Deserialize)]
pub struct BlobLock {
    pub built_at: String,
    pub blobs: Vec<BlobEntry>,
    pub pipeline: Vec<String>,
    pub assets: Vec<LockedAsset>,
}

// One asset as recorded in the lock file
#[derive(Debug, Serialize, Deserialize)]
pub struct LockedAsset {
    pub name: String,
    pub kind: String,
    pub discriminant: u8,
    // sha-256 of the asset's serialized args_bytes
    pub args_hash: String,
    // which blob holds this asset's payload, if any
    pub payload_blob: Option<u32>,
}

// The result of a build pack: the blobs written and the path of each
pub struct PackResult {
    pub blob_paths: Vec<String>,
}

// Pack defs and their payloads into one or more blobs
pub fn write_blobs(
    defs: &[BlobAssetDef],
    blob_payloads: &[Vec<u8>],
) -> std::io::Result<PackResult> {
    fs::create_dir_all(concinnity_core::world::CONCINNITY_DATA_DIR)?;

    let mut blob_paths = Vec::new();

    for (idx, payload) in blob_payloads.iter().enumerate() {
        let path = blob_path(idx as u32);
        let defs_to_write: &[BlobAssetDef] = if idx == 0 { defs } else { &[] };
        write_cnb(defs_to_write, payload, &path)?;
        blob_paths.push(path);
    }

    if blob_payloads.is_empty() {
        write_cnb(defs, &[], PRIMARY_CNB)?;
        blob_paths.push(PRIMARY_CNB.to_string());
    }

    Ok(PackResult { blob_paths })
}

// Write a single blob file
fn write_cnb(defs: &[BlobAssetDef], payload: &[u8], path: &str) -> std::io::Result<()> {
    let defs_bytes = postcard::to_stdvec(defs).map_err(|e| std::io::Error::other(e.to_string()))?;

    let defs_len = defs_bytes.len() as u64;

    let mut data = Vec::with_capacity(HEADER_SIZE + defs_bytes.len() + payload.len());
    data.extend_from_slice(&BLOB_MAGIC);
    data.extend_from_slice(&BLOB_VERSION.to_le_bytes());
    data.extend_from_slice(&defs_len.to_le_bytes());
    data.extend_from_slice(&defs_bytes);
    data.extend_from_slice(payload);

    fs::write(path, &data)
}

// PayloadPacker (build step)
pub struct PayloadPacker {
    max_blob_bytes: u64,
    blobs: Vec<Vec<u8>>,
    current_blob: u32,
    current_offset: u64,
}

impl PayloadPacker {
    pub fn new(max_blob_bytes: u64) -> Self {
        Self {
            max_blob_bytes,
            blobs: vec![Vec::new()],
            current_blob: 0,
            current_offset: 0,
        }
    }

    pub fn push(&mut self, data: &[u8]) -> PayloadLocator {
        let len = data.len() as u64;

        if self.current_offset > 0 && self.current_offset + len > self.max_blob_bytes {
            self.blobs.push(Vec::new());
            self.current_blob += 1;
            self.current_offset = 0;
        }

        let offset = self.current_offset;
        self.blobs[self.current_blob as usize].extend_from_slice(data);
        self.current_offset += len;

        PayloadLocator {
            blob_index: self.current_blob,
            offset,
            len,
        }
    }

    pub fn finish(self) -> Vec<Vec<u8>> {
        self.blobs
    }
}

// Lock file
pub fn write_lock(
    pipeline: &[&str],
    named_defs: &[(&str, &BlobAssetDef)],
    blob_paths: &[String],
) -> std::io::Result<()> {
    let mut blobs = Vec::new();
    for path in blob_paths {
        let data = fs::read(path).unwrap_or_default();
        let payload_bytes = if data.len() > HEADER_SIZE {
            let defs_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
            let payload_start = HEADER_SIZE + defs_len;
            (data.len().saturating_sub(payload_start)) as u64
        } else {
            0
        };
        blobs.push(BlobEntry {
            path: path.clone(),
            checksum: checksum(&data),
            payload_bytes,
        });
    }

    let assets = named_defs
        .iter()
        .map(|(name, def)| LockedAsset {
            name: name.to_string(),
            kind: format!("{:?}", def.kind),
            discriminant: def.discriminant,
            args_hash: checksum(&def.args_bytes),
            payload_blob: def.payload.as_ref().map(|p| p.blob_index),
        })
        .collect();

    let lock = BlobLock {
        built_at: now_iso8601(),
        blobs,
        pipeline: pipeline.iter().map(|s| s.to_string()).collect(),
        assets,
    };

    fs::write(LOCK_PATH, serde_json::to_string_pretty(&lock)?)
}

// Helpers
fn checksum(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    format!("{:x}", h.finalize())
}

fn now_iso8601() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .expect("time format")
}
