// src/app/texture_stream.rs
//
// The `std`-side half of the asset-streaming subsystem.
//
// This owns the background payload-fetch thread and the channels that carry
// work to it, and wraps the `no_std` policy core in `crate::gfx::streaming`.
// The split is deliberate: `gfx::streaming::StreamPlanner` decides *what* to
// stream using only `core` + `alloc`; everything OS-coupled (threads,
// payload I/O) is confined here so a future `no_std` client runtime only has
// to replace this file.
//
// `PayloadSource` is the seam. `MemPayloadSource` serves compiled texture
// payloads already resident in RAM (used by `cn debug`, which builds payloads
// in memory); `DiskPayloadSource` re-reads them from their blob files on disk
// (used by `cn run`, so the bytes never stay RAM-resident). Both plug into the
// same planner and renderer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use crate::gfx::streaming::{StreamPlanner, StreamState};

// A texture payload decoded to GPU-ready RGBA8 pixels.
pub struct DecodedTexture {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

// Fetches and decodes a streamable texture payload by item id.
//
// `Send + Sync` so the background worker thread can own one. Implementors do
// the slow part of streaming (disk read, decompression); the renderer only
// ever sees the finished [`DecodedTexture`].
pub trait PayloadSource: Send + Sync {
    // Decode item `id` into GPU-ready pixels, or return a human-readable
    // error. Called off the main thread.
    fn fetch(&self, id: usize) -> Result<DecodedTexture, String>;
}

// Payload source for the `cn debug` path: compiled texture payloads kept
// resident in RAM.
//
// `cn debug` builds payloads in memory with no blob files on disk, so it
// cannot use [`DiskPayloadSource`]; the bytes stay RAM-resident and this
// streams the *GPU upload* only. `cn run` uses [`DiskPayloadSource`] instead.
pub struct MemPayloadSource {
    payloads: Vec<Vec<u8>>,
}

impl MemPayloadSource {
    // `payloads[id]` is the compiled texture payload for streamable item `id`.
    pub fn new(payloads: Vec<Vec<u8>>) -> Self {
        Self { payloads }
    }
}

impl PayloadSource for MemPayloadSource {
    fn fetch(&self, id: usize) -> Result<DecodedTexture, String> {
        let bytes = self
            .payloads
            .get(id)
            .ok_or_else(|| format!("no payload for streamed texture {}", id))?;
        let (width, height, pixels) = crate::build::texture::deserialise(bytes)?;
        Ok(DecodedTexture {
            width,
            height,
            pixels,
        })
    }
}

// Locates one streamed texture payload inside a blob file on disk.
//
// `file_offset` is absolute into the file -- the payload-section start (past
// the blob header and defs) is already folded in by the caller, so the
// background worker only seeks and reads.
#[derive(Clone)]
pub struct DiskTextureLocator {
    pub path: String,
    pub file_offset: u64,
    pub len: u64,
}

// Disk-backed payload source: re-reads each compiled texture payload
// from its blob file on disk, so the bytes never stay RAM-resident.
//
// This is the counterpart to [`MemPayloadSource`] for the `cn run` path,
// where the world was loaded from blob files that are still on disk. The
// `cn debug` path builds payloads in memory with no blob files, so it must
// keep using [`MemPayloadSource`].
pub struct DiskPayloadSource {
    // locators[id] points streamed item `id` at its bytes in a blob file.
    locators: Vec<DiskTextureLocator>,
}

impl DiskPayloadSource {
    // `locators[id]` locates the compiled payload for streamable item `id`.
    pub fn new(locators: Vec<DiskTextureLocator>) -> Self {
        Self { locators }
    }
}

impl PayloadSource for DiskPayloadSource {
    fn fetch(&self, id: usize) -> Result<DecodedTexture, String> {
        let loc = self
            .locators
            .get(id)
            .ok_or_else(|| format!("no disk locator for streamed texture {}", id))?;
        let mut file = File::open(&loc.path).map_err(|e| format!("open {}: {}", loc.path, e))?;
        file.seek(SeekFrom::Start(loc.file_offset))
            .map_err(|e| format!("seek {} in {}: {}", loc.file_offset, loc.path, e))?;
        let mut bytes = vec![0u8; loc.len as usize];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("read {} bytes from {}: {}", loc.len, loc.path, e))?;
        let (width, height, pixels) = crate::build::texture::deserialise(&bytes)?;
        Ok(DecodedTexture {
            width,
            height,
            pixels,
        })
    }
}

// Outcome of one background load, carried back to the main thread.
struct LoadResult {
    id: usize,
    decoded: Result<DecodedTexture, String>,
}

// Drives streaming of one of the renderer's texture pools.
//
// One instance drives the albedo pool, a second the normal-map pool; the
// type is pool-agnostic -- the caller's `upload` callback routes a completed
// load to the right pool slot.
//
// Owns the [`StreamPlanner`] policy core plus the background fetch thread.
// Each frame the renderer calls [`update_scores`], [`plan_and_dispatch`], and
// [`drain_completed`] in that order.
//
// [`update_scores`]: TextureStreamer::update_scores
// [`plan_and_dispatch`]: TextureStreamer::plan_and_dispatch
// [`drain_completed`]: TextureStreamer::drain_completed
pub struct TextureStreamer {
    planner: StreamPlanner,
    // centers[id] holds the world-space positions of every draw object that
    // samples texture slot `id`; the streaming priority is the squared
    // distance from the camera to the nearest of them.
    centers: Vec<Vec<[f32; 3]>>,
    // Dropped on shutdown to unblock the worker's `recv`.
    request_tx: Option<Sender<usize>>,
    result_rx: Receiver<LoadResult>,
    worker: Option<JoinHandle<()>>,
}

impl TextureStreamer {
    // Spawn the background worker and build a streamer for `centers.len()`
    // texture slots.
    //
    // `centers[id]` lists the draw-object positions that reference slot `id`.
    // `load_budget` caps loads dispatched per frame; `resident_cap` caps how
    // many textures stay resident at once before LRU eviction kicks in.
    pub fn new(
        source: Arc<dyn PayloadSource>,
        centers: Vec<Vec<[f32; 3]>>,
        load_budget: usize,
        resident_cap: usize,
    ) -> Self {
        let planner = StreamPlanner::new(centers.len(), load_budget, resident_cap);
        let (request_tx, request_rx) = std::sync::mpsc::channel::<usize>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<LoadResult>();

        let worker = std::thread::Builder::new()
            .name("cn-texture-stream".to_string())
            .spawn(move || worker_loop(source, request_rx, result_tx))
            .expect("failed to spawn texture-stream worker");

        Self {
            planner,
            centers,
            request_tx: Some(request_tx),
            result_rx,
            worker: Some(worker),
        }
    }

    // Number of streamed texture slots.
    pub fn len(&self) -> usize {
        self.planner.len()
    }

    // Re-score every slot from the camera position and refresh the LRU
    // timestamp of resident slots. Call once per frame before
    // [`plan_and_dispatch`](Self::plan_and_dispatch).
    pub fn update_scores(&mut self, camera: [f32; 3], frame: u64) {
        for id in 0..self.planner.len() {
            self.planner
                .set_score(id, nearest_sq_distance(&self.centers[id], camera));
            if self.planner.state(id) == Some(StreamState::Resident) {
                self.planner.touch(id, frame);
            }
        }
    }

    // Run the planner: dispatch this frame's loads to the worker and return
    // the slots the caller must evict from the GPU.
    pub fn plan_and_dispatch(&mut self) -> Vec<usize> {
        let plan = self.planner.plan();
        for &id in &plan.to_load {
            let sent = self
                .request_tx
                .as_ref()
                .is_some_and(|tx| tx.send(id).is_ok());
            if !sent {
                // Worker gone: revert so the slot is retried rather than
                // stuck Pending forever.
                self.planner.mark_unloaded(id);
            }
        }
        plan.to_evict
    }

    // Apply every completed background load via `upload`, which uploads the
    // decoded pixels into the renderer's texture slot. Returns the number of
    // slots brought resident this call.
    pub fn drain_completed(
        &mut self,
        frame: u64,
        mut upload: impl FnMut(usize, u32, u32, &[u8]),
    ) -> usize {
        let mut applied = 0;
        while let Ok(result) = self.result_rx.try_recv() {
            match result.decoded {
                Ok(tex) => {
                    upload(result.id, tex.width, tex.height, &tex.pixels);
                    self.planner.mark_resident(result.id, frame);
                    applied += 1;
                }
                Err(e) => {
                    tracing::warn!("texture stream: load of slot {} failed: {}", result.id, e);
                    // Treat a failed fetch as terminally resident so the
                    // planner stops retrying; the slot keeps its placeholder.
                    self.planner.mark_resident(result.id, frame);
                }
            }
        }
        applied
    }

    // `(resident, pending, unloaded)` slot counts, for diagnostics.
    pub fn stats(&self) -> (usize, usize, usize) {
        self.planner.counts()
    }
}

impl Drop for TextureStreamer {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's `recv` loop; then join it so a
        // world rebuild does not leak the thread.
        self.request_tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// Background worker: fetch each requested payload and ship the result back.
// Exits when the request channel closes (the streamer was dropped).
fn worker_loop(
    source: Arc<dyn PayloadSource>,
    requests: Receiver<usize>,
    results: Sender<LoadResult>,
) {
    while let Ok(id) = requests.recv() {
        let decoded = source.fetch(id);
        if results.send(LoadResult { id, decoded }).is_err() {
            break;
        }
    }
}

// Squared distance from `camera` to the nearest position in `centers`.
//
// Squared (not true) distance keeps the math `sqrt`-free: ordering is all
// the planner needs. An empty `centers` (a texture referenced by no draw)
// scores 0 so it still streams in promptly rather than stalling forever.
fn nearest_sq_distance(centers: &[[f32; 3]], camera: [f32; 3]) -> f32 {
    let mut nearest = f32::MAX;
    for c in centers {
        let dx = c[0] - camera[0];
        let dy = c[1] - camera[1];
        let dz = c[2] - camera[2];
        let d = dx * dx + dy * dy + dz * dz;
        if d < nearest {
            nearest = d;
        }
    }
    if centers.is_empty() { 0.0 } else { nearest }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_sq_distance_picks_the_closest_center() {
        let centers = [[10.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        // Closest center is at x=3, so squared distance from origin is 9.
        assert_eq!(nearest_sq_distance(&centers, [0.0, 0.0, 0.0]), 9.0);
    }

    #[test]
    fn nearest_sq_distance_of_no_centers_is_zero() {
        assert_eq!(nearest_sq_distance(&[], [5.0, 5.0, 5.0]), 0.0);
    }

    // Build a minimal compiled-texture payload: u32 width, u32 height, RGBA.
    fn make_payload(w: u32, h: u32, fill: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&w.to_le_bytes());
        bytes.extend_from_slice(&h.to_le_bytes());
        bytes.extend(std::iter::repeat_n(fill, (w * h * 4) as usize));
        bytes
    }

    #[test]
    fn mem_payload_source_decodes_a_payload() {
        let source = MemPayloadSource::new(vec![make_payload(2, 1, 0xAB)]);
        let tex = source.fetch(0).expect("fetch ok");
        assert_eq!((tex.width, tex.height), (2, 1));
        assert_eq!(tex.pixels.len(), 2 * 4);
        assert!(tex.pixels.iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn mem_payload_source_errors_on_unknown_id() {
        let source = MemPayloadSource::new(vec![make_payload(1, 1, 0)]);
        assert!(source.fetch(9).is_err());
    }

    #[test]
    fn disk_payload_source_reads_a_payload_at_offset() {
        use std::io::Write;
        let path =
            std::env::temp_dir().join(format!("cn_disk_payload_read_{}.bin", std::process::id()));
        let payload = make_payload(2, 1, 0xCD);
        // arbitrary leading bytes standing in for a blob header + defs section
        let prefix = vec![0u8; 37];
        {
            let mut f = std::fs::File::create(&path).expect("create temp file");
            f.write_all(&prefix).unwrap();
            f.write_all(&payload).unwrap();
        }
        let source = DiskPayloadSource::new(vec![DiskTextureLocator {
            path: path.to_string_lossy().into_owned(),
            file_offset: prefix.len() as u64,
            len: payload.len() as u64,
        }]);
        let tex = source.fetch(0).expect("fetch ok");
        assert_eq!((tex.width, tex.height), (2, 1));
        assert_eq!(tex.pixels.len(), 2 * 4);
        assert!(tex.pixels.iter().all(|&b| b == 0xCD));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn disk_payload_source_errors_on_unknown_id() {
        let source = DiskPayloadSource::new(vec![]);
        assert!(source.fetch(0).is_err());
    }

    #[test]
    fn disk_payload_source_errors_on_missing_file() {
        let source = DiskPayloadSource::new(vec![DiskTextureLocator {
            path: "/nonexistent/cn_disk_payload_missing.bin".to_string(),
            file_offset: 0,
            len: 4,
        }]);
        assert!(source.fetch(0).is_err());
    }

    // A source that yields a fixed 1x1 texture for any id, used to exercise
    // the worker thread without the build pipeline.
    struct ConstSource;
    impl PayloadSource for ConstSource {
        fn fetch(&self, _id: usize) -> Result<DecodedTexture, String> {
            Ok(DecodedTexture {
                width: 1,
                height: 1,
                pixels: vec![1, 2, 3, 4],
            })
        }
    }

    // Pump drain_completed until `want` slots are resident or a deadline hits.
    fn drain_until(streamer: &mut TextureStreamer, frame: u64, want: usize) -> usize {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut uploads = 0;
        while std::time::Instant::now() < deadline {
            uploads += streamer.drain_completed(frame, |_, _, _, _| {});
            if streamer.stats().0 >= want {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        uploads
    }

    #[test]
    fn streamer_loads_nearest_slots_within_budget() {
        let centers = vec![
            vec![[100.0, 0.0, 0.0]], // slot 0: far
            vec![[2.0, 0.0, 0.0]],   // slot 1: near
            vec![[50.0, 0.0, 0.0]],  // slot 2: mid
        ];
        // Budget 1/frame, generous cap.
        let mut streamer = TextureStreamer::new(Arc::new(ConstSource), centers, 1, 8);
        assert_eq!(streamer.len(), 3);

        // Frame 1: nearest slot (1) is dispatched first.
        streamer.update_scores([0.0, 0.0, 0.0], 1);
        let evict = streamer.plan_and_dispatch();
        assert!(evict.is_empty());
        drain_until(&mut streamer, 1, 1);
        assert_eq!(streamer.stats().0, 1);

        // Frame 2: next-nearest (slot 2) follows.
        streamer.update_scores([0.0, 0.0, 0.0], 2);
        streamer.plan_and_dispatch();
        drain_until(&mut streamer, 2, 2);
        assert_eq!(streamer.stats().0, 2);

        // Frame 3: the far slot finishes the set.
        streamer.update_scores([0.0, 0.0, 0.0], 3);
        streamer.plan_and_dispatch();
        drain_until(&mut streamer, 3, 3);
        assert_eq!(streamer.stats(), (3, 0, 0));
    }

    #[test]
    fn upload_callback_receives_decoded_pixels() {
        let centers = vec![vec![[1.0, 0.0, 0.0]]];
        let mut streamer = TextureStreamer::new(Arc::new(ConstSource), centers, 4, 8);
        streamer.update_scores([0.0, 0.0, 0.0], 1);
        streamer.plan_and_dispatch();

        let mut seen: Option<(usize, u32, u32, Vec<u8>)> = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline && seen.is_none() {
            streamer.drain_completed(1, |id, w, h, px| {
                seen = Some((id, w, h, px.to_vec()));
            });
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(seen, Some((0, 1, 1, vec![1, 2, 3, 4])));
    }
}
