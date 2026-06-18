// GraphicsSystem asset-streaming setup: wires the texture, normal-map, mesh,
// and voxel-world streaming pools onto the backend, plus the stats accessor.

use crate::assets::{BlockType, StreamingConfig, VoxelWorld};
use crate::ecs::asset_id::AssetId;
use crate::gfx::draw_list::MaterialEntry;
use crate::gfx::mesh_payload::Vertex;

use super::helpers::*;
use super::*;

// Worst-case resident chunk count for a streaming VoxelWorld: the bound the
// GPU-cull buffers reserve at init so every resident chunk gets a `GpuObjectData`
// record (chunks fold into the indirect path). The streamer RETAINS a
// chunk until its Chebyshev distance exceeds the evict radius =
// `far_radius + EVICT_HYSTERESIS(=2)` (gfx::chunk_window), where
// `far_radius = impostor_radius()` -- which `VoxelWorld::impostor_radius()` floors
// at `view_radius()`, so this evict-window span is correct whether or not impostors
// are enabled (with impostors off, `far_radius == view_radius`). Peak residency =
// `(2*(far_radius+2)+1)^2`, the SAME `total_chunks` `setup_voxel_world_streaming`
// sizes the geometry headroom for. Capped so a typo radius cannot demand gigabytes
// of record memory (the geometry headroom is the real residency limit anyway).
pub(super) fn chunk_reserve_count(vw: &VoxelWorld) -> usize {
    const MAX_CHUNK_RECORDS: u64 = 65536;
    let far_radius = vw.impostor_radius() as u64;
    let evict_span = 2 * (far_radius + 2) + 1;
    (evict_span * evict_span).min(MAX_CHUNK_RECORDS) as usize
}

impl GraphicsSystem {
    pub(super) fn setup_texture_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        texture_payloads: Vec<Vec<u8>>,
        texture_locators: &[crate::ecs::PayloadLocator],
        disk_backed: bool,
        texture_centers: Vec<Vec<[f32; 3]>>,
    ) {
        let Some(config) = config else { return };
        // When disk-backed the payloads were not retained, so the streamed
        // slot count comes from the locators instead.
        let slot_count = if disk_backed {
            texture_locators.len()
        } else {
            texture_payloads.len()
        };
        if slot_count == 0 {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        // Each backend's update_texture_slot rewrites whichever descriptors,
        // argument-buffers, or per-cluster SRVs sample that slot.
        for slot in 0..slot_count {
            if let Err(e) = backend.evict_texture_slot(slot) {
                tracing::warn!("GraphicsSystem: texture evict slot {}: {}", slot, e);
            }
        }
        let source =
            match build_texture_payload_source(texture_payloads, texture_locators, disk_backed) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("GraphicsSystem: texture streaming source: {}", e);
                    return;
                }
            };
        let streamer = crate::app::texture_stream::TextureStreamer::new(
            source,
            texture_centers,
            config.budget(),
            config.cap(),
        );
        tracing::info!(
            "GraphicsSystem: texture streaming enabled ({} textures, {} source, budget {}/frame, cap {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.budget(),
            config.cap(),
        );
        self.texture_streamer = Some(streamer);
    }

    // Stand up the normal-map streaming subsystem when a StreamingConfig was
    // declared. Mirrors setup_texture_streaming for the normal-map pool: a
    // second TextureStreamer drives it, and streamed item `i` is pool slot
    // `i + 1` (slot 0 is the never-streamed flat-normal fallback). Reuses the
    // shared texture_budget / texture_cap, and the same disk-backed vs
    // RAM-backed payload source choice. Honoured by Metal, Vulkan, and DirectX.
    pub(super) fn setup_normal_map_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        normal_map_payloads: Vec<Vec<u8>>,
        normal_map_locators: &[crate::ecs::PayloadLocator],
        disk_backed: bool,
        normal_map_centers: Vec<Vec<[f32; 3]>>,
    ) {
        let Some(config) = config else { return };
        // When disk-backed the payloads were not retained, so the streamed
        // map count comes from the locators instead.
        let map_count = if disk_backed {
            normal_map_locators.len()
        } else {
            normal_map_payloads.len()
        };
        if map_count == 0 {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        for i in 0..map_count {
            if let Err(e) = backend.evict_normal_map_slot(i + 1) {
                tracing::warn!("GraphicsSystem: normal-map evict slot {}: {}", i + 1, e);
            }
        }
        let source = match build_texture_payload_source(
            normal_map_payloads,
            normal_map_locators,
            disk_backed,
        ) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("GraphicsSystem: normal-map streaming source: {}", e);
                return;
            }
        };
        let streamer = crate::app::texture_stream::TextureStreamer::new(
            source,
            normal_map_centers,
            config.budget(),
            config.cap(),
        );
        tracing::info!(
            "GraphicsSystem: normal-map streaming enabled ({} maps, {} source, budget {}/frame, cap {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.budget(),
            config.cap(),
        );
        self.normal_map_streamer = Some(streamer);
    }

    // Stand up the mesh-geometry streaming subsystem when a StreamingConfig
    // was declared. Every streamed draw's geometry region is zeroed now (via
    // evict_mesh); the streamer brings them back resident over the next
    // frames, nearest first.
    //
    // The payload source depends on where the world came from: a disk-backed
    // `cn run` world writes the streamed geometry to a scratch file and
    // re-reads it from there (no persistent RAM copy), an in-memory `cn debug`
    // world keeps the geometry RAM-resident.
    //
    // Supported on Metal, DirectX, and Vulkan. The args are consumed
    // unconditionally so they never warn as unused on a backend that does not
    // yet stream.
    pub(super) fn setup_mesh_streaming(
        &mut self,
        config: Option<StreamingConfig>,
        mesh_payloads: Vec<crate::app::mesh_stream::DecodedMesh>,
        mesh_centers: Vec<Vec<[f32; 3]>>,
        mesh_draw_indices: Vec<usize>,
        disk_backed: bool,
        seed_region: Option<crate::gfx::mesh_seed::MeshSeedRegion>,
    ) {
        let Some(config) = config else { return };
        if mesh_payloads.is_empty() {
            return;
        }
        let Some(backend) = self.backend.as_deref_mut() else {
            return;
        };
        // Init residency. Two paths:
        //  - Shrinkable seed (`seed_region` present): the streamed geometry was
        //    never baked into the buffers -- compaction already marked each
        //    streamed draw non-resident and reserved one headroom block -- so
        //    seed the sub-allocators with that block. Calling `evict_mesh` here
        //    would zero/free the placeholder offset-0 region and corrupt the
        //    first resident draw, so it must be skipped on this path.
        //  - Full-set seed (`seed_region` absent: a backend without the
        //    shrinkable seed, or no shrink possible): free each streamed mesh's
        //    build-time region into the sub-allocators. retire_frame 0 -- nothing
        //    has been drawn, so the space is reusable immediately.
        match seed_region {
            Some(r) => {
                backend.seed_mesh_streaming(r.vtx_offset, r.vtx_bytes, r.idx_offset, r.idx_bytes);
            }
            None => {
                for &draw_idx in &mesh_draw_indices {
                    if let Err(e) = backend.evict_mesh(draw_idx, 0) {
                        tracing::warn!("GraphicsSystem: mesh evict draw {}: {}", draw_idx, e);
                    }
                }
            }
        }
        // A disk-backed world spills the geometry to a scratch file so the
        // `mesh_payloads` RAM copy can be dropped; `cn debug` keeps it
        // resident since it has no disk artifacts to re-read.
        let source: std::sync::Arc<dyn crate::app::mesh_stream::MeshPayloadSource> = if disk_backed
        {
            let path = crate::app::mesh_stream::default_scratch_path();
            match crate::app::mesh_stream::write_mesh_scratch(path, &mesh_payloads) {
                Ok(s) => std::sync::Arc::new(s),
                Err(e) => {
                    tracing::error!("GraphicsSystem: mesh streaming scratch file: {}", e);
                    return;
                }
            }
        } else {
            std::sync::Arc::new(crate::app::mesh_stream::MemMeshSource::new(mesh_payloads))
        };
        let streamer = crate::app::mesh_stream::MeshStreamer::new(
            source,
            mesh_centers,
            config.mesh_budget(),
            config.mesh_cap(),
        );
        tracing::info!(
            "GraphicsSystem: mesh streaming enabled ({} meshes, {} source, budget {}/frame, cap {})",
            streamer.len(),
            if disk_backed { "disk" } else { "ram" },
            config.mesh_budget(),
            config.mesh_cap(),
        );
        self.mesh_streamer = Some(streamer);
        self.mesh_stream_draw_indices = mesh_draw_indices;
    }

    // Stand up the infinite-world chunk-streaming subsystem when a VoxelWorld
    // was declared. Resolves the block palette and shared material, grows the
    // GPU buffers by a chunk-headroom region, and builds the ChunkStreamer;
    // `step` then generates and uploads chunks around the camera each frame.
    // Supported on Metal, DirectX, and Vulkan. The buffer-growth + SRV/descriptor
    // setup differs per backend (the `setup_chunk_streaming` match below); the
    // palette/material resolution, headroom sizing, and streamer build are
    // backend-agnostic.
    pub(super) fn setup_voxel_world_streaming(
        &mut self,
        voxel_world: Option<VoxelWorld>,
        block_types: &std::collections::HashMap<AssetId, BlockType>,
        material_map: &std::collections::HashMap<AssetId, MaterialEntry>,
    ) {
        let Some(vw) = voxel_world else { return };

        // Resolve the palette: each id is a BlockType; index 0 is air. A
        // missing entry degrades to air rather than failing the world.
        let palette: Vec<crate::geometry::ChunkBlockType> = vw
            .palette
            .iter()
            .map(|id| match block_types.get(id) {
                Some(bt) => block_type_to_chunk(bt),
                None => {
                    tracing::warn!(
                        "GraphicsSystem: VoxelWorld palette entry {} is not a known BlockType",
                        id
                    );
                    crate::geometry::ChunkBlockType {
                        solid: false,
                        uv_top: [0.0; 4],
                        uv_bottom: [0.0; 4],
                        uv_side: [0.0; 4],
                    }
                }
            })
            .collect();

        // Resolve the shared material to texture-pool slots + scalars.
        let (texture_slot, normal_map_slot, material) = vw
            .material
            .and_then(|id| material_map.get(&id).copied())
            .unwrap_or((0, 0, crate::gfx::render_types::MaterialUniforms::DEFAULT));

        let chunk_blocks = vw.chunk_blocks();
        let block_size = vw.block_size();
        let (chunk_w, chunk_d) = vw.chunk_world_size();
        let near_radius = vw.view_radius();
        let far_radius = vw.impostor_radius();
        let impostor_step = vw.impostor_step();

        // Size the chunk buffer headroom for the worst-case resident window.
        // The near band (full voxel meshes) reaches one ring past `near_radius`
        // (the detail-hysteresis transient where a receding chunk is still
        // full); the far band fills the rest of the evict window with cheap
        // impostors. Sizing the two bands separately keeps the impostor radius
        // from demanding full-chunk headroom for hundreds of distant chunks.
        let near_span = 2 * (near_radius as u64 + 1) + 1;
        let near_chunks = near_span * near_span;
        let evict_span = 2 * (far_radius as u64 + 2) + 1;
        let total_chunks = evict_span * evict_span;
        let far_chunks = total_chunks.saturating_sub(near_chunks);

        // Full-detail per-chunk budget: generous face count for rolling terrain;
        // an over-budget chunk fails its add and is logged rather than
        // corrupting GPU memory.
        let faces_per_chunk = (chunk_blocks[0] as u64) * (chunk_blocks[2] as u64) * 4;
        let full_vtx =
            (faces_per_chunk * 4).min(u16::MAX as u64) * std::mem::size_of::<Vertex>() as u64;
        // Shared index buffer is u32-typed; per-mesh indices are widened on
        // upload, so size the chunk headroom for u32 elements.
        let full_idx = faces_per_chunk * 6 * std::mem::size_of::<u32>() as u64;

        // Impostor per-chunk budget: one quad per coarse cell + a perimeter
        // skirt, 4 verts / 6 indices per quad (matches `build_chunk_impostor_mesh`).
        let nx = (chunk_blocks[0] as u64).div_ceil(impostor_step as u64);
        let nz = (chunk_blocks[2] as u64).div_ceil(impostor_step as u64);
        let impostor_quads = nx * nz + 2 * (nx + nz);
        let impostor_vtx = impostor_quads * 4 * std::mem::size_of::<Vertex>() as u64;
        let impostor_idx = impostor_quads * 6 * std::mem::size_of::<u32>() as u64;

        // Cap total headroom so a typo in the radii cannot demand gigabytes of
        // GPU memory.
        const MAX_HEADROOM: u64 = 512 * 1024 * 1024;
        let chunk_vtx_bytes =
            (near_chunks * full_vtx + far_chunks * impostor_vtx).min(MAX_HEADROOM) as usize;
        let chunk_idx_bytes =
            (near_chunks * full_idx + far_chunks * impostor_idx).min(MAX_HEADROOM) as usize;

        // Backend-specific buffer growth + SRV/descriptor setup. Metal binds
        // chunk textures per draw and ignores the slot args (its impl drops
        // them); DirectX and Vulkan bake one shared (albedo, normal)
        // descriptor from the chunk material.
        let setup_result = match self.backend.as_deref_mut() {
            Some(backend) => backend.setup_chunk_streaming(
                chunk_vtx_bytes,
                chunk_idx_bytes,
                texture_slot,
                normal_map_slot,
            ),
            None => return,
        };
        if let Err(e) = setup_result {
            tracing::error!("GraphicsSystem: VoxelWorld chunk streaming: {}", e);
            return;
        }

        let source = std::sync::Arc::new(crate::app::chunk_stream::ProceduralChunkSource::new(
            vw.seed,
            chunk_blocks,
            block_size,
            palette,
            impostor_step,
        ));
        let streamer = crate::app::chunk_stream::ChunkStreamer::new(
            source,
            near_radius,
            far_radius,
            vw.load_budget(),
            chunk_w,
            chunk_d,
        );
        tracing::info!(
            "GraphicsSystem: VoxelWorld streaming enabled (seed {}, {}x{}x{} blocks, near-radius {}, impostor-radius {} (step {}), budget {}/frame, {} KiB chunk headroom)",
            vw.seed,
            chunk_blocks[0],
            chunk_blocks[1],
            chunk_blocks[2],
            near_radius,
            if vw.impostors_enabled() {
                far_radius
            } else {
                0
            },
            impostor_step,
            vw.load_budget(),
            (chunk_vtx_bytes + chunk_idx_bytes) / 1024,
        );
        self.chunk_stream = Some(ChunkStreamState {
            streamer,
            draws: std::collections::BTreeMap::new(),
            chunk_w,
            chunk_d,
            // Seeded at the world origin; the first `step` rebases onto the
            // camera's actual chunk before any chunk is resident, so the
            // seed value never places geometry.
            origin_chunk: crate::gfx::chunk_coord::ChunkCoord::new(0, 0),
            texture_slot,
            normal_map_slot,
            material,
        });
    }

    // `(resident, pending, unloaded)` counts for each active streaming pool.
    // Used by the debug server's `streaming` command for headless checks.
    // (Consumed only by the `cn debug` binary, so dead in a library build.)
    #[allow(dead_code)]
    pub fn streaming_stats(&self) -> StreamingStats {
        StreamingStats {
            texture: self.texture_streamer.as_ref().map(|s| s.stats()),
            normal_map: self.normal_map_streamer.as_ref().map(|s| s.stats()),
            mesh: self.mesh_streamer.as_ref().map(|s| s.stats()),
            chunk: self.chunk_stream.as_ref().map(|cs| cs.streamer.stats()),
        }
    }

    // Shared atomic flag the active backend polls at frame start to trigger a
    // shader rebuild. `Some` only under `cn debug` on backends that ship
    // hot-reload (Metal today); `None` otherwise. The debug server captures
    // this Arc once and uses it to forward `reload-shaders` requests.
    // (Consumed only by the `cn debug` binary, so dead in a library build.)
    #[allow(dead_code)]
    pub fn shader_reload_flag(&self) -> Option<std::sync::Arc<std::sync::atomic::AtomicBool>> {
        self.backend.as_ref().and_then(|b| b.shader_reload_flag())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The chunk record reserve must cover the streamer's worst-case
    // residency, or resident chunks past the reserve get no GPU-driven draw record
    // and render invisibly. The streamer retains a chunk until its Chebyshev
    // distance exceeds `evict_radius = far_radius + EVICT_HYSTERESIS(=2)` (see
    // gfx::chunk_window), with `far_radius = impostor_radius()` (floored at
    // view_radius), so peak residency = `(2*(far_radius+2)+1)^2`. This must hold for
    // impostors-on AND impostors-off worlds (the default is impostor_radius = 0).
    #[test]
    fn chunk_reserve_covers_streamer_evict_window() {
        for (view, impostor) in [(5u32, 0u32), (2, 6), (8, 0), (3, 10), (0, 0), (32, 96)] {
            let vw = VoxelWorld {
                view_radius: view,
                impostor_radius: impostor,
                ..Default::default()
            };
            // `impostor_radius()` floors at `view_radius()`, so this is the real
            // far radius the streamer uses whether or not impostors are enabled.
            let far = vw.impostor_radius() as usize;
            let evict_span = 2 * (far + 2) + 1;
            let bound = (evict_span * evict_span).min(65536);
            assert!(
                chunk_reserve_count(&vw) >= bound,
                "view={view} impostor={impostor}: reserve {} < streamer evict window {}",
                chunk_reserve_count(&vw),
                bound,
            );
        }
    }
}
