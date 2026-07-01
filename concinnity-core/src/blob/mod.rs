// The blob READ half: load blob binaries back into memory. The WRITE half
// (packing payloads into .cnb files and emitting world-lock.json) lives in the
// concinnity-cook crate, which imports the consts below.
//
// Layout of a blob binary:
//
//   [ 4 bytes magic ][ 4 bytes version ][ 8 bytes defs_len ][ defs_bytes ][ payload_bytes ... ]
//
// The header is fixed at 16 bytes. `defs_len` is the byte length of the
// postcard-serialized Vec<BlobAssetDef> that follows. Everything after
// defs_len + defs_bytes is the raw payload section, addressed by the
// (blob_index, offset, length) fields inside each BlobAssetDef.
//
// `data/0` is the primary blob. It always holds the full asset registry (defs)
// and may also hold payload bytes for assets packed before the size ceiling.
// Overflow payloads spill into `data/1`, `data/2`, ... as needed.
//
// All blobs share the same header format. Only `data/0` carries a non-empty
// defs section; subsequent blobs have defs_len=0 and are pure payload.
//
// This file never needs to change when a new asset type is added.
pub use crate::ecs::BlobAssetDef;
use crate::ecs::PayloadLocator;
use crate::result::CnResult;
use std::fs;

// Constants
pub const BLOB_MAGIC: [u8; 4] = *b"CNB\0";
pub const BLOB_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 16; // magic(4) + version(4) + defs_len(8)

pub const LOCK_PATH: &str = "world-lock.json";

// Format a blob file path for a given index under `.concinnity/data/`. Blob 0
// is the primary blob (the def table plus the first payload section); higher
// indices are overflow payload blobs.
pub fn blob_path(index: u32) -> String {
    crate::paths::data_dir()
        .join(index.to_string())
        .to_string_lossy()
        .into_owned()
}

// State of one blob file's payload section.
//
// `Unloaded` is the lazy state of an overflow blob: its file is on disk but
// has not been read yet. `Loaded` holds the resident bytes. `Released` means
// a system deliberately freed the payload after consuming it -- reads then
// error rather than reload, since the data is known to be no longer needed.
enum BlobSlot {
    // overflow blob not yet read; the String is its file path
    Unloaded(String),
    // payload section resident in memory
    Loaded(Vec<u8>),
    // payload deliberately released after use; reads error, no reload
    Released,
}

// Holds the raw payload sections of each blob file.
//
// Indexed by `PayloadLocator::blob_index`. Blob 0's payload section is loaded
// eagerly by `load()` -- it carries the defs and the primary payloads and is
// needed immediately. Overflow blobs (1, 2, …) start `Unloaded` and are read
// from disk on demand the first time a locator references them, so a large
// world does not pay the RAM (or I/O) cost of every overflow blob at startup.
//
// Systems call `release(blob_index)` after consuming a blob's payloads (e.g.
// after uploading SPIR-V to the GPU) so the memory is freed promptly.
pub struct BlobData {
    // slots[i] is the payload state of blob i
    slots: Vec<BlobSlot>,
    // True when the payloads came from blob files on disk (the `cn run` path).
    // False for in-memory builds (`cn debug`) and empty stores. The
    // asset-streaming subsystem reads this to decide whether a streamed
    // payload can be re-read from its blob file on demand instead of held
    // RAM-resident -- see `app/texture_stream.rs::DiskPayloadSource`.
    disk_backed: bool,
}

impl BlobData {
    // Build an in-memory store where every section is already resident. Used
    // by the `cn debug` path, which compiles payloads in memory with no blob
    // files, so there is nothing to lazily load. A `None` section is treated
    // as already released.
    pub fn new(payload_sections: Vec<Option<Vec<u8>>>) -> Self {
        let slots = payload_sections
            .into_iter()
            .map(|s| match s {
                Some(bytes) => BlobSlot::Loaded(bytes),
                None => BlobSlot::Released,
            })
            .collect();
        Self {
            slots,
            disk_backed: false,
        }
    }

    // empty store for worlds with no compiled payloads (tests, runtime-only worlds)
    pub fn empty() -> Self {
        Self {
            slots: Vec::new(),
            disk_backed: false,
        }
    }

    // true when the payloads were loaded from blob files on disk, so a
    // streamed payload can be re-read from disk rather than kept in RAM
    pub fn disk_backed(&self) -> bool {
        self.disk_backed
    }

    // read the bytes for a given locator
    //
    // An `Unloaded` overflow blob is read from its file on first access and
    // becomes `Loaded`. Errors if the locator is out of range, the blob was
    // released, or the on-demand load fails.
    #[allow(dead_code)]
    pub fn read(&mut self, locator: &PayloadLocator) -> Result<&[u8], CnResult> {
        let idx = locator.blob_index as usize;
        let slot = self.slots.get_mut(idx).ok_or_else(|| {
            tracing::error!("BlobData: blob {} is out of range", locator.blob_index);
            CnResult::FileIo
        })?;
        if let BlobSlot::Unloaded(path) = slot {
            tracing::debug!(
                "BlobData: lazily loading overflow blob {}",
                locator.blob_index
            );
            let bytes = read_payload_section(&path.clone())?;
            *slot = BlobSlot::Loaded(bytes);
        }

        let section = match &self.slots[idx] {
            BlobSlot::Loaded(bytes) => bytes,
            BlobSlot::Released => {
                tracing::error!("BlobData: blob {} has been released", locator.blob_index);
                return Err(CnResult::FileIo);
            }
            // Unreachable: an Unloaded slot was loaded just above.
            BlobSlot::Unloaded(_) => return Err(CnResult::FileIo),
        };

        let start = locator.offset as usize;
        let end = start.checked_add(locator.len as usize).ok_or_else(|| {
            tracing::error!(
                "BlobData: payload slice offset {} + len {} overflows in blob {}",
                start,
                locator.len,
                locator.blob_index
            );
            CnResult::FileIo
        })?;
        section.get(start..end).ok_or_else(|| {
            tracing::error!(
                "BlobData: payload slice [{}, {}) out of bounds in blob {} (len={})",
                start,
                end,
                locator.blob_index,
                section.len()
            );
            CnResult::FileIo
        })
    }

    // release a blob's in-memory payload once all systems that need it have
    // finished consuming it (e.g. after GPU upload)
    //
    // subsequent `read()` calls for locators in this blob return an error
    // rather than reloading -- the data is known to no longer be needed -- so
    // only call this once you are sure no other system needs it
    #[allow(dead_code)]
    pub fn release(&mut self, blob_index: u32) {
        if let Some(slot) = self.slots.get_mut(blob_index as usize)
            && !matches!(slot, BlobSlot::Released)
        {
            tracing::debug!("BlobData: releasing payload for blob {}", blob_index);
            *slot = BlobSlot::Released;
        }
    }

    // true if the blob's payload is resident in memory right now; an
    // `Unloaded` overflow blob reports false until its first read
    #[allow(dead_code)]
    pub fn is_loaded(&self, blob_index: u32) -> bool {
        matches!(
            self.slots.get(blob_index as usize),
            Some(BlobSlot::Loaded(_))
        )
    }
}

// Read and deserialize asset defs from a blob
// Returns (defs, payload_start_offset)
pub fn read_cnb(path: &str) -> Result<(Vec<BlobAssetDef>, usize), CnResult> {
    let data = fs::read(path).map_err(|e| {
        tracing::error!("Failed to read {}: {}", path, e);
        CnResult::FileIo
    })?;

    if data.len() < HEADER_SIZE {
        tracing::error!("{}: file too short ({} bytes)", path, data.len());
        return Err(CnResult::FileIo);
    }

    let magic = &data[0..4];
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    let defs_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;

    if magic != BLOB_MAGIC {
        tracing::error!("Bad magic in {}: {:?}", path, magic);
        return Err(CnResult::FileIo);
    }
    if version != BLOB_VERSION {
        tracing::error!(
            "Version mismatch in {} (got {}, want {})",
            path,
            version,
            BLOB_VERSION
        );
        return Err(CnResult::FileIo);
    }

    let defs_end = HEADER_SIZE + defs_len;
    if data.len() < defs_end {
        tracing::error!("{}: truncated defs section", path);
        return Err(CnResult::FileIo);
    }

    let defs = postcard::from_bytes(&data[HEADER_SIZE..defs_end]).map_err(|e| {
        tracing::error!("Failed to deserialize defs from {}: {}", path, e);
        CnResult::FileIo
    })?;

    Ok((defs, defs_end))
}

// Byte offset within a blob file at which its payload section begins, i.e.
// just past the 16-byte header and the bincode defs section. Reads only the
// header, so it is cheap to call without loading the whole file -- the
// disk-backed streaming source uses it to turn a `PayloadLocator` offset
// (relative to the payload section) into an absolute file offset.
// Used only by the Metal-driven disk-backed streaming source for now
// (Vulkan/DirectX streaming catch-up is a follow-up).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub fn payload_section_start(path: &str) -> Result<u64, CnResult> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| {
        tracing::error!("Failed to open {}: {}", path, e);
        CnResult::FileIo
    })?;
    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header).map_err(|e| {
        tracing::error!("Failed to read header of {}: {}", path, e);
        CnResult::FileIo
    })?;
    if header[0..4] != BLOB_MAGIC {
        tracing::error!("Bad magic in {}: {:?}", path, &header[0..4]);
        return Err(CnResult::FileIo);
    }
    let defs_len = u64::from_le_bytes(header[8..16].try_into().unwrap());
    Ok(HEADER_SIZE as u64 + defs_len)
}

// Read just the payload section of a .cnb file into memory
fn read_payload_section(path: &str) -> Result<Vec<u8>, CnResult> {
    let data = fs::read(path).map_err(|e| {
        tracing::error!("Failed to read {}: {}", path, e);
        CnResult::FileIo
    })?;
    if data.len() < HEADER_SIZE {
        return Ok(Vec::new());
    }
    let defs_len = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let payload_start = HEADER_SIZE + defs_len;
    Ok(data.get(payload_start..).unwrap_or(&[]).to_vec())
}

// Load the primary blob's defs and the `BlobData` payload store, without
// resolving defs into runtime `Asset`s (that resolution depends on the client
// runtime registry, so it lives in the client `blob::load` shim).
//
// Only blob 0's payload section is read into memory here; overflow blobs
// (index >= 1) start `Unloaded` and `BlobData::read()` reads each from disk
// the first time a locator references it.
pub fn load_raw() -> Result<(Vec<BlobAssetDef>, BlobData), CnResult> {
    let (defs, _header_end) = read_cnb(&blob_path(0))?;

    // determine how many distinct blob indices are referenced so we know
    // which overflow files exist
    let max_blob_index = defs
        .iter()
        .filter_map(|d| d.payload.as_ref())
        .map(|p| p.blob_index)
        .max()
        .unwrap_or(0);

    // Blob 0 is read eagerly -- it is needed immediately. Overflow blobs are
    // left `Unloaded`; `BlobData::read()` pulls each from disk on first use.
    let mut slots: Vec<BlobSlot> = Vec::with_capacity(max_blob_index as usize + 1);
    let blob0_payload = read_payload_section(&blob_path(0))?;
    tracing::debug!("Loaded blob 0 payload ({} bytes)", blob0_payload.len());
    slots.push(BlobSlot::Loaded(blob0_payload));
    for idx in 1..=max_blob_index {
        slots.push(BlobSlot::Unloaded(blob_path(idx)));
    }

    // these sections came from blob files on disk, so the streaming subsystem
    // may re-read a payload from disk instead of holding it RAM-resident
    let blob_data = BlobData {
        slots,
        disk_backed: true,
    };

    Ok((defs, blob_data))
}

// Load defs without resolving (for callers that apply overlays first)
#[allow(dead_code)]
pub fn load_defs() -> Result<Vec<BlobAssetDef>, CnResult> {
    let (defs, _) = read_cnb(&blob_path(0))?;
    Ok(defs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a .cnb byte image inline: 16-byte header (magic + version +
    // defs_len) followed by an empty defs section and the given payload. Mirrors
    // the write-half format that now lives in the build crate, so the read tests
    // do not depend on it.
    fn cnb_bytes(payload: &[u8]) -> Vec<u8> {
        let defs_len: u64 = 0;
        let mut data = Vec::with_capacity(HEADER_SIZE + payload.len());
        data.extend_from_slice(&BLOB_MAGIC);
        data.extend_from_slice(&BLOB_VERSION.to_le_bytes());
        data.extend_from_slice(&defs_len.to_le_bytes());
        data.extend_from_slice(payload);
        data
    }

    #[test]
    fn payload_section_start_skips_header_and_defs() {
        let path = std::env::temp_dir().join(format!("cn_blob_section_{}.cnb", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        fs::write(&path, cnb_bytes(b"payloadbytes")).expect("write blob");

        let start = payload_section_start(&path).expect("section start");
        // the payload must sit exactly at the reported offset
        let data = fs::read(&path).unwrap();
        assert_eq!(&data[start as usize..], b"payloadbytes");

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn payload_section_start_rejects_bad_magic() {
        let path =
            std::env::temp_dir().join(format!("cn_blob_badmagic_{}.cnb", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        fs::write(&path, vec![0u8; HEADER_SIZE]).unwrap();
        assert!(payload_section_start(&path).is_err());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn blob_data_disk_backed_defaults_false() {
        assert!(!BlobData::empty().disk_backed());
        assert!(!BlobData::new(vec![Some(vec![1, 2, 3])]).disk_backed());
    }

    #[test]
    fn read_lazily_loads_an_unloaded_overflow_blob() {
        let path = std::env::temp_dir().join(format!("cn_blob_lazy_{}.cnb", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        fs::write(&path, cnb_bytes(b"hello world")).expect("write blob");

        let mut bd = BlobData {
            slots: vec![
                BlobSlot::Loaded(Vec::new()),     // blob 0
                BlobSlot::Unloaded(path.clone()), // blob 1, not yet read
            ],
            disk_backed: true,
        };
        assert!(!bd.is_loaded(1));

        let loc = PayloadLocator {
            blob_index: 1,
            offset: 6,
            len: 5,
        };
        assert_eq!(bd.read(&loc).expect("read ok"), b"world");
        // the lazy load promoted the slot to resident
        assert!(bd.is_loaded(1));

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn read_errors_on_released_blob() {
        // a `None` section is treated as already released
        let mut bd = BlobData::new(vec![None]);
        let loc = PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 1,
        };
        assert!(bd.read(&loc).is_err());
    }

    #[test]
    fn release_then_read_errors() {
        let mut bd = BlobData::new(vec![Some(b"abcd".to_vec())]);
        let loc = PayloadLocator {
            blob_index: 0,
            offset: 0,
            len: 2,
        };
        assert_eq!(bd.read(&loc).expect("read ok"), b"ab");
        bd.release(0);
        assert!(!bd.is_loaded(0));
        assert!(bd.read(&loc).is_err());
    }

    #[test]
    fn read_errors_on_out_of_range_blob() {
        let mut bd = BlobData::empty();
        let loc = PayloadLocator {
            blob_index: 3,
            offset: 0,
            len: 1,
        };
        assert!(bd.read(&loc).is_err());
    }
}
