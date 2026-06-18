// src/app/mesh_stream.rs
//
// The `std`-side driver for mesh-geometry streaming.
//
// This is the geometry counterpart of `crate::app::texture_stream`: it owns a
// background payload-fetch thread and the channels that carry work to it, and
// wraps the `no_std` policy core in `crate::gfx::streaming`. The split:
// `gfx::streaming::StreamPlanner` decides *what* to stream using only
// `core` + `alloc`; everything OS-coupled -- threads, payload I/O --
// lives here so a future `no_std` client runtime only has to replace
// this file.
//
// `MeshPayloadSource` is the seam. `MemMeshSource` serves mesh geometry kept
// resident in RAM (used by `cn debug`, which builds geometry in memory with no
// disk artifacts); `DiskMeshSource` re-reads it from a scratch file written by
// `write_mesh_scratch` (used by `cn run`, so the geometry never stays a second
// RAM copy past GPU upload). Both plug into the same planner and renderer.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use crate::gfx::mesh_payload::Vertex;
use crate::gfx::streaming::{StreamPlanner, StreamState};

// A mesh payload decoded to GPU-ready vertex and index data.
//
// Index values are mesh-relative (0-based): the renderer's sub-allocator
// places the vertices at an arbitrary offset on upload, so the backend
// rebases the indices onto that region rather than the payload baking in a
// fixed base.
#[derive(Clone)]
pub struct DecodedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
}

// Fetches and decodes a streamable mesh payload by item id.
//
// `Send + Sync` so the background worker thread can own one. Implementors do
// the slow part of streaming (disk read, decompression); the renderer only
// ever sees the finished [`DecodedMesh`].
pub trait MeshPayloadSource: Send + Sync {
    // Decode item `id` into GPU-ready geometry, or return a human-readable
    // error. Called off the main thread.
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String>;
}

// Mesh source for the `cn debug` path: geometry kept resident in RAM.
//
// `cn debug` builds geometry in memory with no disk artifacts, so it cannot
// re-read from a scratch file; the geometry stays RAM-resident and this
// streams the *GPU upload* only. `cn run` uses [`DiskMeshSource`] instead.
pub struct MemMeshSource {
    meshes: Vec<DecodedMesh>,
}

impl MemMeshSource {
    // `meshes[id]` is the geometry for streamable mesh `id`.
    pub fn new(meshes: Vec<DecodedMesh>) -> Self {
        Self { meshes }
    }
}

impl MeshPayloadSource for MemMeshSource {
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String> {
        self.meshes
            .get(id)
            .cloned()
            .ok_or_else(|| format!("no payload for streamed mesh {}", id))
    }
}

// Locates one streamed mesh's geometry record inside the scratch file.
#[derive(Clone)]
pub struct DiskMeshLocator {
    pub file_offset: u64,
    pub len: u64,
}

// Disk-backed mesh source: re-reads each mesh's geometry from a
// scratch file on disk, so the geometry never stays a second RAM copy.
//
// Unlike a streamed texture -- whose compiled payload already sits in a blob
// file the streamer can re-read -- a streamed mesh's geometry only exists as
// a region of the assembled vertex/index buffers, with no discrete on-disk
// payload. [`write_mesh_scratch`] therefore writes the streamed geometry to a
// scratch file once, and this source re-reads each record from it on demand.
// The file is removed when the source is dropped (world rebuild or shutdown).
//
// `cn debug`, which has no disk artifacts, keeps using [`MemMeshSource`].
pub struct DiskMeshSource {
    path: String,
    // locators[id] points streamed mesh `id` at its record in the scratch file
    locators: Vec<DiskMeshLocator>,
}

impl MeshPayloadSource for DiskMeshSource {
    fn fetch(&self, id: usize) -> Result<DecodedMesh, String> {
        let loc = self
            .locators
            .get(id)
            .ok_or_else(|| format!("no disk locator for streamed mesh {}", id))?;
        let mut file = File::open(&self.path).map_err(|e| format!("open {}: {}", self.path, e))?;
        file.seek(SeekFrom::Start(loc.file_offset))
            .map_err(|e| format!("seek {} in {}: {}", loc.file_offset, self.path, e))?;
        let mut bytes = vec![0u8; loc.len as usize];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("read {} bytes from {}: {}", loc.len, self.path, e))?;
        decode_mesh(&bytes)
    }
}

impl Drop for DiskMeshSource {
    fn drop(&mut self) {
        // The scratch file is regenerated on the next world build/run, so a
        // dropped source has no reason to keep it.
        let _ = std::fs::remove_file(&self.path);
    }
}

// Write every streamed mesh's geometry to `path` and return a
// [`DiskMeshSource`] that re-reads each record on demand.
//
// Lets the caller drop the RAM-resident [`DecodedMesh`] payloads: under
// `cn run` the geometry then lives only in the GPU buffers and this scratch
// file, not in a second CPU-side copy.
pub fn write_mesh_scratch(path: String, meshes: &[DecodedMesh]) -> Result<DiskMeshSource, String> {
    let mut file = File::create(&path).map_err(|e| format!("create {}: {}", path, e))?;
    let mut locators = Vec::with_capacity(meshes.len());
    let mut offset: u64 = 0;
    for mesh in meshes {
        let bytes = encode_mesh(mesh);
        file.write_all(&bytes)
            .map_err(|e| format!("write {}: {}", path, e))?;
        locators.push(DiskMeshLocator {
            file_offset: offset,
            len: bytes.len() as u64,
        });
        offset += bytes.len() as u64;
    }
    file.flush().map_err(|e| format!("flush {}: {}", path, e))?;
    Ok(DiskMeshSource { path, locators })
}

// A process-unique scratch-file path in the OS temp directory.
//
// Each call returns a distinct path so a world rebuild's new source does not
// collide with the old one's file (the old [`DiskMeshSource`] removes its own
// file on drop).
pub fn default_scratch_path() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir()
        .join(format!(
            "cn_mesh_scratch_{}_{}.bin",
            std::process::id(),
            seq
        ))
        .to_string_lossy()
        .into_owned()
}

// Serialise one mesh's geometry into the scratch-file record format:
// `u32 vertex_count`, the vertices as 56-byte records, `u32 index_count`,
// the indices as little-endian `u16`s.
fn encode_mesh(mesh: &DecodedMesh) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + mesh.vertices.len() * 56 + 4 + mesh.indices.len() * 2);
    buf.extend_from_slice(&(mesh.vertices.len() as u32).to_le_bytes());
    for v in &mesh.vertices {
        for x in v
            .pos
            .iter()
            .chain(v.normal.iter())
            .chain(v.tangent.iter())
            .chain(v.color.iter())
            .chain(v.uv.iter())
        {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(mesh.indices.len() as u32).to_le_bytes());
    for i in &mesh.indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

// Inverse of [`encode_mesh`]: decode one scratch-file record. Errors rather
// than panics on a truncated record, so a corrupt scratch file fails the
// load loudly instead of reading out of bounds.
fn decode_mesh(bytes: &[u8]) -> Result<DecodedMesh, String> {
    let read_u32 = |at: usize| -> Result<u32, String> {
        bytes
            .get(at..at + 4)
            .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
            .ok_or_else(|| "truncated mesh record".to_string())
    };

    let vertex_count = read_u32(0)? as usize;
    let verts_end = 4 + vertex_count * 56;
    if bytes.len() < verts_end {
        return Err("truncated mesh vertex data".to_string());
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    for v in 0..vertex_count {
        let base = 4 + v * 56;
        let f = |o: usize| f32::from_le_bytes(bytes[base + o..base + o + 4].try_into().unwrap());
        vertices.push(Vertex {
            pos: [f(0), f(4), f(8)],
            normal: [f(12), f(16), f(20)],
            tangent: [f(24), f(28), f(32)],
            color: [f(36), f(40), f(44)],
            uv: [f(48), f(52)],
        });
    }

    let index_count = read_u32(verts_end)? as usize;
    let idx_start = verts_end + 4;
    let idx_end = idx_start + index_count * 2;
    if bytes.len() < idx_end {
        return Err("truncated mesh index data".to_string());
    }
    let mut indices = Vec::with_capacity(index_count);
    for i in 0..index_count {
        let at = idx_start + i * 2;
        indices.push(u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap()));
    }

    Ok(DecodedMesh { vertices, indices })
}

// Outcome of one background load, carried back to the main thread.
struct LoadResult {
    id: usize,
    decoded: Result<DecodedMesh, String>,
}

// Drives streaming of the renderer's static mesh geometry.
//
// Owns the [`StreamPlanner`] policy core plus the background fetch thread.
// Each frame the renderer calls [`update_scores`], [`plan_and_dispatch`], and
// [`drain_completed`] in that order.
//
// [`update_scores`]: MeshStreamer::update_scores
// [`plan_and_dispatch`]: MeshStreamer::plan_and_dispatch
// [`drain_completed`]: MeshStreamer::drain_completed
pub struct MeshStreamer {
    planner: StreamPlanner,
    // centers[id] holds the world-space position(s) used to score streamed
    // mesh `id`; the streaming priority is the squared distance from the
    // camera to the nearest of them.
    centers: Vec<Vec<[f32; 3]>>,
    // Dropped on shutdown to unblock the worker's `recv`.
    request_tx: Option<Sender<usize>>,
    result_rx: Receiver<LoadResult>,
    worker: Option<JoinHandle<()>>,
}

impl MeshStreamer {
    // Spawn the background worker and build a streamer for `centers.len()`
    // meshes.
    //
    // `centers[id]` lists the world-space position(s) used to score mesh
    // `id`. `load_budget` caps loads dispatched per frame; `resident_cap`
    // caps how many meshes stay resident at once before LRU eviction.
    pub fn new(
        source: Arc<dyn MeshPayloadSource>,
        centers: Vec<Vec<[f32; 3]>>,
        load_budget: usize,
        resident_cap: usize,
    ) -> Self {
        let planner = StreamPlanner::new(centers.len(), load_budget, resident_cap);
        let (request_tx, request_rx) = std::sync::mpsc::channel::<usize>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<LoadResult>();

        let worker = std::thread::Builder::new()
            .name("cn-mesh-stream".to_string())
            .spawn(move || worker_loop(source, request_rx, result_tx))
            .expect("failed to spawn mesh-stream worker");

        Self {
            planner,
            centers,
            request_tx: Some(request_tx),
            result_rx,
            worker: Some(worker),
        }
    }

    // Number of streamed meshes.
    pub fn len(&self) -> usize {
        self.planner.len()
    }

    // Re-score every mesh from the camera position and refresh the LRU
    // timestamp of resident meshes. Call once per frame before
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
    // the meshes the caller must evict from the GPU.
    pub fn plan_and_dispatch(&mut self) -> Vec<usize> {
        let plan = self.planner.plan();
        for &id in &plan.to_load {
            let sent = self
                .request_tx
                .as_ref()
                .is_some_and(|tx| tx.send(id).is_ok());
            if !sent {
                // Worker gone -- revert so the mesh is retried rather than
                // stuck Pending forever.
                self.planner.mark_unloaded(id);
            }
        }
        plan.to_evict
    }

    // Apply every completed background load via `upload`, which uploads the
    // decoded geometry into the renderer's mesh region. Returns the number of
    // meshes brought resident this call.
    //
    // `upload` returns `Err` for a *transient* failure -- the shrinkable seed
    // headroom is momentarily full because freed regions still await their
    // retire frame. Such a mesh is rolled back to `Unloaded` so the planner
    // retries it once an eviction's space is reclaimed, rather than being
    // marked resident with no geometry on the GPU. A failed *fetch* (decode /
    // disk error) is terminal and marked resident so the planner stops
    // retrying a payload that will never decode.
    pub fn drain_completed(
        &mut self,
        frame: u64,
        mut upload: impl FnMut(usize, &[Vertex], &[u16]) -> Result<(), String>,
    ) -> usize {
        let mut applied = 0;
        while let Ok(result) = self.result_rx.try_recv() {
            match result.decoded {
                Ok(mesh) => match upload(result.id, &mesh.vertices, &mesh.indices) {
                    Ok(()) => {
                        self.planner.mark_resident(result.id, frame);
                        applied += 1;
                    }
                    Err(e) => {
                        // Transient alloc miss: keep the mesh Unloaded so the
                        // planner re-dispatches it once freed space is
                        // reclaimed.
                        tracing::debug!(
                            "mesh stream: upload of mesh {} deferred: {}",
                            result.id,
                            e
                        );
                        self.planner.mark_unloaded(result.id);
                    }
                },
                Err(e) => {
                    tracing::warn!("mesh stream: load of mesh {} failed: {}", result.id, e);
                    // Treat a failed fetch as terminally resident so the
                    // planner stops retrying; the mesh keeps its empty region.
                    self.planner.mark_resident(result.id, frame);
                }
            }
        }
        applied
    }

    // `(resident, pending, unloaded)` mesh counts -- for diagnostics.
    pub fn stats(&self) -> (usize, usize, usize) {
        self.planner.counts()
    }
}

impl Drop for MeshStreamer {
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
    source: Arc<dyn MeshPayloadSource>,
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
// Squared (not true) distance keeps the math `sqrt`-free -- ordering is all
// the planner needs. An empty `centers` scores 0 so it still streams in
// promptly rather than stalling forever.
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

    fn mk_vertex(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    #[test]
    fn nearest_sq_distance_picks_the_closest_center() {
        let centers = [[10.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        assert_eq!(nearest_sq_distance(&centers, [0.0, 0.0, 0.0]), 9.0);
    }

    #[test]
    fn nearest_sq_distance_of_no_centers_is_zero() {
        assert_eq!(nearest_sq_distance(&[], [5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn mem_mesh_source_serves_a_payload() {
        let source = MemMeshSource::new(vec![DecodedMesh {
            vertices: vec![mk_vertex(1.0), mk_vertex(2.0)],
            indices: vec![0, 1, 0],
        }]);
        let mesh = source.fetch(0).expect("fetch ok");
        assert_eq!(mesh.vertices.len(), 2);
        assert_eq!(mesh.indices, vec![0, 1, 0]);
    }

    #[test]
    fn mem_mesh_source_errors_on_unknown_id() {
        let source = MemMeshSource::new(vec![DecodedMesh {
            vertices: vec![mk_vertex(0.0)],
            indices: vec![0],
        }]);
        assert!(source.fetch(9).is_err());
    }

    // A source yielding a fixed 1-triangle mesh for any id, used to exercise
    // the worker thread without the build pipeline.
    struct ConstSource;
    impl MeshPayloadSource for ConstSource {
        fn fetch(&self, _id: usize) -> Result<DecodedMesh, String> {
            Ok(DecodedMesh {
                vertices: vec![mk_vertex(0.0), mk_vertex(1.0), mk_vertex(2.0)],
                indices: vec![0, 1, 2],
            })
        }
    }

    // Pump drain_completed until `want` meshes are resident or a deadline hits.
    fn drain_until(streamer: &mut MeshStreamer, frame: u64, want: usize) -> usize {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut uploads = 0;
        while std::time::Instant::now() < deadline {
            uploads += streamer.drain_completed(frame, |_, _, _| Ok(()));
            if streamer.stats().0 >= want {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        uploads
    }

    #[test]
    fn streamer_loads_nearest_meshes_within_budget() {
        let centers = vec![
            vec![[100.0, 0.0, 0.0]], // mesh 0: far
            vec![[2.0, 0.0, 0.0]],   // mesh 1: near
            vec![[50.0, 0.0, 0.0]],  // mesh 2: mid
        ];
        // Budget 1/frame, generous cap.
        let mut streamer = MeshStreamer::new(Arc::new(ConstSource), centers, 1, 8);
        assert_eq!(streamer.len(), 3);

        // Frame 1: nearest mesh (1) is dispatched first.
        streamer.update_scores([0.0, 0.0, 0.0], 1);
        let evict = streamer.plan_and_dispatch();
        assert!(evict.is_empty());
        drain_until(&mut streamer, 1, 1);
        assert_eq!(streamer.stats().0, 1);

        // Frame 2: next-nearest (mesh 2) follows.
        streamer.update_scores([0.0, 0.0, 0.0], 2);
        streamer.plan_and_dispatch();
        drain_until(&mut streamer, 2, 2);
        assert_eq!(streamer.stats().0, 2);

        // Frame 3: the far mesh finishes the set.
        streamer.update_scores([0.0, 0.0, 0.0], 3);
        streamer.plan_and_dispatch();
        drain_until(&mut streamer, 3, 3);
        assert_eq!(streamer.stats(), (3, 0, 0));
    }

    #[test]
    fn upload_callback_receives_decoded_geometry() {
        let centers = vec![vec![[1.0, 0.0, 0.0]]];
        let mut streamer = MeshStreamer::new(Arc::new(ConstSource), centers, 4, 8);
        streamer.update_scores([0.0, 0.0, 0.0], 1);
        streamer.plan_and_dispatch();

        let mut seen: Option<(usize, usize, Vec<u16>)> = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline && seen.is_none() {
            streamer.drain_completed(1, |id, verts, idxs| {
                seen = Some((id, verts.len(), idxs.to_vec()));
                Ok(())
            });
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(seen, Some((0, 3, vec![0, 1, 2])));
    }

    #[test]
    fn upload_failure_rolls_back_to_unloaded_for_retry() {
        let centers = vec![vec![[1.0, 0.0, 0.0]]];
        let mut streamer = MeshStreamer::new(Arc::new(ConstSource), centers, 4, 8);

        // Frame 1: dispatch, then fail the upload (transient seed-full miss).
        streamer.update_scores([0.0, 0.0, 0.0], 1);
        streamer.plan_and_dispatch();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut drained = false;
        while std::time::Instant::now() < deadline && !drained {
            streamer.drain_completed(1, |_, _, _| {
                drained = true;
                Err("no free space".to_string())
            });
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(drained, "worker should have produced a result");
        // Not resident: the planner rolled it back to Unloaded for retry.
        assert_eq!(streamer.stats(), (0, 0, 1));

        // Frame 2: re-dispatch + a succeeding upload brings it resident.
        streamer.update_scores([0.0, 0.0, 0.0], 2);
        streamer.plan_and_dispatch();
        drain_until(&mut streamer, 2, 1);
        assert_eq!(streamer.stats().0, 1);
    }

    #[test]
    fn encode_mesh_round_trips_through_decode() {
        let mesh = DecodedMesh {
            vertices: vec![mk_vertex(1.0), mk_vertex(2.0), mk_vertex(3.0)],
            indices: vec![0, 1, 2, 2, 1, 0],
        };
        let decoded = decode_mesh(&encode_mesh(&mesh)).expect("decode ok");
        assert_eq!(decoded.vertices.len(), 3);
        assert_eq!(decoded.indices, vec![0, 1, 2, 2, 1, 0]);
        assert_eq!(decoded.vertices[1].pos, [2.0, 0.0, 0.0]);
        assert_eq!(decoded.vertices[2].normal, [0.0, 1.0, 0.0]);
        assert_eq!(decoded.vertices[0].tangent, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn decode_mesh_errors_on_truncated_record() {
        let bytes = encode_mesh(&DecodedMesh {
            vertices: vec![mk_vertex(0.0)],
            indices: vec![0],
        });
        // dropping the final index byte leaves an incomplete record
        assert!(decode_mesh(&bytes[..bytes.len() - 1]).is_err());
        // a header claiming more vertices than the buffer holds
        assert!(decode_mesh(&[9, 0, 0, 0]).is_err());
        // an empty buffer has not even a vertex count
        assert!(decode_mesh(&[]).is_err());
    }

    #[test]
    fn disk_mesh_source_round_trips_multiple_meshes() {
        let meshes = vec![
            DecodedMesh {
                vertices: vec![mk_vertex(1.0)],
                indices: vec![0],
            },
            DecodedMesh {
                vertices: vec![mk_vertex(2.0), mk_vertex(3.0)],
                indices: vec![0, 1, 0],
            },
        ];
        let source = write_mesh_scratch(default_scratch_path(), &meshes).expect("write scratch");

        let m0 = source.fetch(0).expect("fetch 0");
        assert_eq!(m0.vertices.len(), 1);
        assert_eq!(m0.vertices[0].pos, [1.0, 0.0, 0.0]);
        assert_eq!(m0.indices, vec![0]);

        // mesh 1 lives at a non-zero offset -- exercises the per-record seek
        let m1 = source.fetch(1).expect("fetch 1");
        assert_eq!(m1.vertices.len(), 2);
        assert_eq!(m1.vertices[1].pos, [3.0, 0.0, 0.0]);
        assert_eq!(m1.indices, vec![0, 1, 0]);
    }

    #[test]
    fn disk_mesh_source_errors_on_unknown_id() {
        let source = write_mesh_scratch(default_scratch_path(), &[]).expect("write scratch");
        assert!(source.fetch(0).is_err());
    }

    #[test]
    fn disk_mesh_source_removes_scratch_file_on_drop() {
        let path = default_scratch_path();
        let source = write_mesh_scratch(
            path.clone(),
            &[DecodedMesh {
                vertices: vec![mk_vertex(0.0)],
                indices: vec![0],
            }],
        )
        .expect("write scratch");
        assert!(std::path::Path::new(&path).exists());
        drop(source);
        assert!(!std::path::Path::new(&path).exists());
    }

    #[test]
    fn default_scratch_path_is_unique_per_call() {
        assert_ne!(default_scratch_path(), default_scratch_path());
    }

    #[test]
    fn streamer_loads_from_a_disk_source() {
        let meshes = vec![DecodedMesh {
            vertices: vec![mk_vertex(0.0), mk_vertex(1.0), mk_vertex(2.0)],
            indices: vec![0, 1, 2],
        }];
        let source = write_mesh_scratch(default_scratch_path(), &meshes).expect("write scratch");
        let centers = vec![vec![[1.0, 0.0, 0.0]]];
        let mut streamer = MeshStreamer::new(Arc::new(source), centers, 4, 8);
        streamer.update_scores([0.0, 0.0, 0.0], 1);
        streamer.plan_and_dispatch();

        let mut seen: Option<(usize, usize, Vec<u16>)> = None;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while std::time::Instant::now() < deadline && seen.is_none() {
            streamer.drain_completed(1, |id, verts, idxs| {
                seen = Some((id, verts.len(), idxs.to_vec()));
                Ok(())
            });
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(seen, Some((0, 3, vec![0, 1, 2])));
    }
}
