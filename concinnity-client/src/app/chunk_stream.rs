// src/app/chunk_stream.rs
//
// The `std`-side driver for infinite-world voxel chunk streaming.
//
// This is the chunk counterpart of `crate::app::mesh_stream`: it owns a
// background generation thread and the channels that carry work to it, and
// wraps the `no_std` policy core in `crate::gfx::chunk_window`. The split is
// the same one the rest of the streaming subsystem uses: `ChunkWindow` decides
// *which* chunks to stream using only `core` + `alloc`; everything OS-coupled
// -- the thread, the channels -- lives here so a future `no_std` client
// runtime only has to replace this file.
//
// `ChunkSource` is the seam. The shipped `ProceduralChunkSource` generates a
// chunk's geometry from a seed on demand, so chunks are never RAM- or
// disk-resident: an evicted chunk is simply regenerated if the camera returns.

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread::JoinHandle;

use crate::app::mesh_stream::DecodedMesh;
use crate::geometry::{
    ChunkBlockType, ChunkGenerator, build_chunk_impostor_mesh, build_chunk_mesh,
};
use crate::gfx::chunk_coord::ChunkCoord;
use crate::gfx::chunk_window::{ChunkDetail, ChunkWindow};
use crate::gfx::mesh_payload::Vertex;

// Generates a streamable chunk's geometry by coordinate and detail.
//
// `Send + Sync` so the background worker thread can own one. Implementors do
// the slow part (terrain generation, meshing); the streamer only ever sees
// the finished [`DecodedMesh`]. `detail` selects full voxel geometry
// ([`ChunkDetail::Near`]) or a coarse distant impostor ([`ChunkDetail::Far`]).
pub trait ChunkSource: Send + Sync {
    // Build the geometry for chunk `coord` at `detail`, or return a
    // human-readable error. Called off the main thread.
    fn generate(&self, coord: ChunkCoord, detail: ChunkDetail) -> Result<DecodedMesh, String>;
}

// The shipped chunk source: deterministic procedural terrain.
//
// Wraps a [`ChunkGenerator`] and the resolved block palette; `generate` runs
// the generator and meshes the result. Because generation is a pure function
// of the seed and the coordinate, a chunk that streams out and back in is
// regenerated identically -- no RAM or disk copy is kept.
pub struct ProceduralChunkSource {
    generator: ChunkGenerator,
    palette: Vec<ChunkBlockType>,
    chunk_blocks: [u32; 3],
    block_size: f32,
    // Coarse-grid step (in blocks) for the distant-impostor surface.
    impostor_step: u32,
    // The surface block's atlas UVs, so impostors texture like the full chunks.
    surface_block: ChunkBlockType,
}

impl ProceduralChunkSource {
    // A source for a world with the given seed, chunk dimensions, block size,
    // resolved `BlockType` palette, and distant-impostor coarse step.
    pub fn new(
        seed: u64,
        chunk_blocks: [u32; 3],
        block_size: f32,
        palette: Vec<ChunkBlockType>,
        impostor_step: u32,
    ) -> Self {
        let generator = ChunkGenerator::new(seed, chunk_blocks, palette.len() as u32);
        // The surface block is the one the generator paints the top of each
        // column with; fall back to the first palette entry for a degenerate
        // (single-entry) palette.
        let surface_idx = generator.surface_palette_index() as usize;
        let surface_block = palette
            .get(surface_idx)
            .or_else(|| palette.first())
            .copied()
            .unwrap_or(ChunkBlockType {
                solid: true,
                uv_top: [0.0, 0.0, 1.0, 1.0],
                uv_bottom: [0.0, 0.0, 1.0, 1.0],
                uv_side: [0.0, 0.0, 1.0, 1.0],
            });
        Self {
            generator,
            palette,
            chunk_blocks,
            block_size,
            impostor_step: impostor_step.max(1),
            surface_block,
        }
    }

    // Sample the terrain surface height on the coarse impostor grid: corner
    // (gx, gz) at the world column its clamped local position maps to. Sampling
    // by world column keeps adjacent impostors watertight along their shared
    // edge (both read the same boundary columns).
    fn impostor_heights(&self, coord: ChunkCoord) -> Vec<i32> {
        let step = self.impostor_step;
        let [dx, _dy, dz] = self.chunk_blocks;
        let nx = dx.div_ceil(step);
        let nz = dz.div_ceil(step);
        let base_x = coord.x * dx as i32;
        let base_z = coord.z * dz as i32;
        let mut heights = Vec::with_capacity(((nx + 1) * (nz + 1)) as usize);
        for gz in 0..=nz {
            let lz = (gz * step).min(dz) as i32;
            for gx in 0..=nx {
                let lx = (gx * step).min(dx) as i32;
                heights.push(
                    self.generator
                        .surface_height_world(base_x + lx, base_z + lz),
                );
            }
        }
        heights
    }
}

impl ChunkSource for ProceduralChunkSource {
    fn generate(&self, coord: ChunkCoord, detail: ChunkDetail) -> Result<DecodedMesh, String> {
        let (vertices, indices) = match detail {
            ChunkDetail::Near => {
                let blocks = self.generator.generate(coord);
                build_chunk_mesh(self.chunk_blocks, self.block_size, &blocks, &self.palette)?
            }
            ChunkDetail::Far => {
                let heights = self.impostor_heights(coord);
                build_chunk_impostor_mesh(
                    self.chunk_blocks,
                    self.block_size,
                    self.impostor_step,
                    &heights,
                    self.surface_block.uv_top,
                    self.surface_block.uv_side,
                )?
            }
        };
        Ok(DecodedMesh { vertices, indices })
    }
}

// Outcome of one background generation, carried back to the main thread.
struct LoadResult {
    coord: ChunkCoord,
    decoded: Result<DecodedMesh, String>,
}

// Drives streaming of an infinite voxel world's chunks.
//
// Owns the [`ChunkWindow`] policy core plus the background generation thread.
// Each frame the renderer calls [`plan_and_dispatch`] then [`drain_completed`].
//
// [`plan_and_dispatch`]: ChunkStreamer::plan_and_dispatch
// [`drain_completed`]: ChunkStreamer::drain_completed
pub struct ChunkStreamer {
    window: ChunkWindow,
    // World-space size of one chunk on X / Z -- to map a camera position to
    // its chunk coordinate.
    chunk_w: f32,
    chunk_d: f32,
    // Dropped on shutdown to unblock the worker's `recv`.
    request_tx: Option<Sender<(ChunkCoord, ChunkDetail)>>,
    result_rx: Receiver<LoadResult>,
    worker: Option<JoinHandle<()>>,
}

impl ChunkStreamer {
    // Spawn the background worker and build a streamer.
    //
    // `near_radius` is the full-detail chunk radius, `far_radius` the outer
    // impostor radius (equal to `near_radius` disables impostors),
    // `load_budget` caps generations dispatched per frame, and `chunk_w` /
    // `chunk_d` are one chunk's world-space X / Z size (for the
    // camera-to-chunk mapping).
    pub fn new(
        source: Arc<dyn ChunkSource>,
        near_radius: i32,
        far_radius: i32,
        load_budget: usize,
        chunk_w: f32,
        chunk_d: f32,
    ) -> Self {
        let window = ChunkWindow::new(near_radius, far_radius, load_budget);
        let (request_tx, request_rx) = std::sync::mpsc::channel::<(ChunkCoord, ChunkDetail)>();
        let (result_tx, result_rx) = std::sync::mpsc::channel::<LoadResult>();

        let worker = std::thread::Builder::new()
            .name("cn-chunk-stream".to_string())
            .spawn(move || worker_loop(source, request_rx, result_tx))
            .expect("failed to spawn chunk-stream worker");

        Self {
            window,
            chunk_w,
            chunk_d,
            request_tx: Some(request_tx),
            result_rx,
            worker: Some(worker),
        }
    }

    // The chunk a world-space camera position falls in.
    pub fn camera_chunk(&self, camera: [f32; 3]) -> ChunkCoord {
        ChunkCoord::from_world(camera[0], camera[2], self.chunk_w, self.chunk_d)
    }

    // Run the window policy for a camera in chunk `camera`: dispatch this
    // frame's chunk generations to the worker and return the chunks the
    // caller must remove from the GPU (they have left the view window).
    pub fn plan_and_dispatch(&mut self, camera: ChunkCoord) -> Vec<ChunkCoord> {
        let plan = self.window.plan(camera);
        for &(coord, detail) in &plan.to_load {
            let sent = self
                .request_tx
                .as_ref()
                .is_some_and(|tx| tx.send((coord, detail)).is_ok());
            if !sent {
                // Worker gone -- forget the chunk so it is retried rather than
                // stuck Pending forever.
                self.window.forget(coord);
            }
        }
        plan.to_evict
    }

    // Apply every completed background generation via `upload`, which adds the
    // chunk's geometry to the renderer. Returns the number of chunks brought
    // resident this call.
    //
    // A chunk evicted while its generation was still in flight is dropped --
    // the window no longer tracks it, so its mesh is discarded rather than
    // uploaded into a chunk the camera has already left behind.
    pub fn drain_completed(
        &mut self,
        mut upload: impl FnMut(ChunkCoord, &[Vertex], &[u16]),
    ) -> usize {
        let mut applied = 0;
        while let Ok(result) = self.result_rx.try_recv() {
            if !self.window.is_tracked(result.coord) {
                continue; // evicted mid-flight -- discard
            }
            match result.decoded {
                Ok(mesh) => {
                    upload(result.coord, &mesh.vertices, &mesh.indices);
                    self.window.mark_resident(result.coord);
                    applied += 1;
                }
                Err(e) => {
                    tracing::warn!(
                        "chunk stream: generation of chunk ({},{}) failed: {}",
                        result.coord.x,
                        result.coord.z,
                        e
                    );
                    // Terminally resident so the planner stops retrying a
                    // chunk whose generation deterministically fails.
                    self.window.mark_resident(result.coord);
                }
            }
        }
        applied
    }

    // `(resident, pending)` chunk counts -- for diagnostics.
    pub fn stats(&self) -> (usize, usize) {
        self.window.counts()
    }

    // `(near_resident, far_resident)` chunk counts -- resident full chunks vs
    // resident distant impostors, for diagnostics.
    pub fn detail_counts(&self) -> (usize, usize) {
        self.window.counts_by_detail()
    }
}

impl Drop for ChunkStreamer {
    fn drop(&mut self) {
        // Dropping the sender ends the worker's `recv` loop; then join it so a
        // world rebuild does not leak the thread.
        self.request_tx = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// Background worker: generate each requested chunk and ship the result back.
// Exits when the request channel closes (the streamer was dropped).
fn worker_loop(
    source: Arc<dyn ChunkSource>,
    requests: Receiver<(ChunkCoord, ChunkDetail)>,
    results: Sender<LoadResult>,
) {
    while let Ok((coord, detail)) = requests.recv() {
        let decoded = source.generate(coord, detail);
        if results.send(LoadResult { coord, decoded }).is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cc(x: i32, z: i32) -> ChunkCoord {
        ChunkCoord::new(x, z)
    }

    fn mk_vertex(x: f32) -> Vertex {
        Vertex {
            pos: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        }
    }

    // A source yielding a fixed 1-triangle mesh for any chunk + detail.
    struct ConstSource;
    impl ChunkSource for ConstSource {
        fn generate(
            &self,
            _coord: ChunkCoord,
            _detail: ChunkDetail,
        ) -> Result<DecodedMesh, String> {
            Ok(DecodedMesh {
                vertices: vec![mk_vertex(0.0), mk_vertex(1.0), mk_vertex(2.0)],
                indices: vec![0, 1, 2],
            })
        }
    }

    fn drain_until(streamer: &mut ChunkStreamer, want: usize) -> Vec<ChunkCoord> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        let mut uploaded = Vec::new();
        while std::time::Instant::now() < deadline {
            streamer.drain_completed(|coord, _, _| uploaded.push(coord));
            if streamer.stats().0 >= want {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        uploaded
    }

    #[test]
    fn procedural_source_generates_a_non_empty_chunk() {
        let palette = vec![
            ChunkBlockType {
                solid: false,
                uv_top: [0.0; 4],
                uv_bottom: [0.0; 4],
                uv_side: [0.0; 4],
            },
            ChunkBlockType {
                solid: true,
                uv_top: [0.0, 0.0, 1.0, 1.0],
                uv_bottom: [0.0, 0.0, 1.0, 1.0],
                uv_side: [0.0, 0.0, 1.0, 1.0],
            },
        ];
        let source = ProceduralChunkSource::new(42, [8, 16, 8], 1.0, palette, 4);
        let mesh = source
            .generate(cc(0, 0), ChunkDetail::Near)
            .expect("generate ok");
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        // The Far impostor is a non-empty, far smaller mesh than the full chunk.
        let full = source
            .generate(cc(0, 0), ChunkDetail::Near)
            .expect("full ok");
        let impostor = source
            .generate(cc(0, 0), ChunkDetail::Far)
            .expect("impostor ok");
        assert!(!impostor.vertices.is_empty());
        assert!(
            impostor.vertices.len() < full.vertices.len(),
            "impostor ({}) should be cheaper than full ({})",
            impostor.vertices.len(),
            full.vertices.len()
        );
    }

    #[test]
    fn camera_chunk_maps_world_position_to_a_chunk() {
        let streamer = ChunkStreamer::new(Arc::new(ConstSource), 2, 2, 4, 16.0, 16.0);
        assert_eq!(streamer.camera_chunk([0.0, 5.0, 0.0]), cc(0, 0));
        assert_eq!(streamer.camera_chunk([20.0, 5.0, -1.0]), cc(1, -1));
    }

    #[test]
    fn plan_dispatches_chunks_and_drain_uploads_them() {
        let mut streamer = ChunkStreamer::new(Arc::new(ConstSource), 1, 1, 100, 16.0, 16.0);
        // radius 1 -> a 3x3 window of 9 chunks dispatched at once.
        let evict = streamer.plan_and_dispatch(cc(0, 0));
        assert!(evict.is_empty());
        let uploaded = drain_until(&mut streamer, 9);
        assert_eq!(uploaded.len(), 9);
        assert!(uploaded.contains(&cc(0, 0)));
        assert_eq!(streamer.stats(), (9, 0));
    }

    #[test]
    fn moving_far_evicts_the_old_window() {
        let mut streamer = ChunkStreamer::new(Arc::new(ConstSource), 1, 1, 100, 16.0, 16.0);
        streamer.plan_and_dispatch(cc(0, 0));
        drain_until(&mut streamer, 9);
        // Jump far enough that the whole origin window leaves the evict band.
        let evict = streamer.plan_and_dispatch(cc(50, 0));
        assert!(evict.contains(&cc(0, 0)));
        assert_eq!(evict.len(), 9);
    }
}
