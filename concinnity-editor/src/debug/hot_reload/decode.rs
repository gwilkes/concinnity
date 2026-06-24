// src/debug/hot_reload/decode.rs
//
// Off-thread asset decode + apply: spawns the decode + envmap-convolution
// worker threads, then `poll_pending_assets` / `poll_pending_envmap` drain the
// completed work on a later frame and push it through the backend `update_*`
// calls. Keeps the (seconds-long) decode off the render thread.

use crate::gfx::graphics_system::hot_reload_sources::*;

use super::state::*;

// Trigger a hot-reload pass. Spawns at most two worker threads: one for the
// IBL convolution (when an EnvironmentMap is declared) and one for the rest
// of the captured catalogue (textures, ColorLut, file-backed Mesh /
// SkinnedMesh decode). The decode work is CPU-bound (PNG/JPEG inflate, glTF
// parse, vertex normalisation) and used to stall the render thread for
// several seconds on a 43-texture / 14-mesh world; moving it off-thread lets
// the render loop keep drawing while the worker runs. [`poll_pending_assets`]
// + [`poll_pending_envmap`] drain the workers' results on a later frame and
// dispatch them through the matching backend `update_*` calls (those are
// fast, single MTL buffer / texture swaps).
pub fn reload_assets(state: &AssetHotReloadState) {
    spawn_asset_decode_worker(state);
    spawn_envmap_worker(state);
}

// Spawn the texture + ColorLut + Mesh + SkinnedMesh decode worker if any
// source catalogue carries entries. The previous batch's receiver is
// consulted first: if a decode is already in flight, this pass logs and
// skips so the user re-triggers after the result lands. Sources are cloned
// into the worker closure so the worker has no reference back to
// AssetHotReloadState's storage and the render thread keeps mutating its
// own copies freely.
fn spawn_asset_decode_worker(state: &AssetHotReloadState) {
    let has_work = !state.map.entries.is_empty()
        || state.color_lut.is_some()
        || !state.meshes.entries.is_empty()
        || !state.skinned_meshes.entries.is_empty();
    if !has_work {
        return;
    }
    let mut slot = match state.asset_batch_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if slot.is_some() {
        tracing::info!("asset hot-reload: decode batch skipped; previous worker still running");
        return;
    }
    // Snapshot every source the worker needs. Cheap: the maps are a few KB
    // total even on the showcase world.
    let textures = state.map.entries.clone();
    let color_lut = state.color_lut.clone();
    let meshes = state.meshes.entries.clone();
    let skinned = state.skinned_meshes.entries.clone();
    let totals = (
        textures.len(),
        meshes.len(),
        skinned.len(),
        color_lut.is_some(),
    );
    let (tx, rx) = std::sync::mpsc::channel();
    match std::thread::Builder::new()
        .name("cn-asset-reload".into())
        .spawn(move || {
            // Run inside JobPool::install so any nested rayon `par_iter`
            // dispatches to the bounded `available_parallelism() - 1` pool
            // (mirroring the envmap worker). decode_asset_batch itself uses
            // par_iter to fan textures across cores.
            let batch = crate::jobs::pool()
                .install(|| decode_asset_batch(textures, color_lut, meshes, skinned));
            // Receiver dropped means the asset state went away before the
            // decode finished; the send fails silently and the worker exits.
            let _ = tx.send(batch);
        }) {
        Ok(_) => {
            *slot = Some(rx);
            tracing::info!(
                "asset hot-reload: decode batch spawned on worker thread \
                 ({} texture(s), {} Mesh(es), {} SkinnedMesh(es), {} ColorLut)",
                totals.0,
                totals.1,
                totals.2,
                if totals.3 { 1 } else { 0 }
            );
        }
        Err(e) => {
            tracing::error!(
                "asset hot-reload: failed to spawn decode worker: {} \
                 (all assets kept their old payloads)",
                e
            );
        }
    }
}

// Spawn the EnvironmentMap convolution worker if an envmap is declared.
// Same pattern as the asset decode worker but kept on its own slot because
// the convolution often takes seconds (much longer than texture / mesh
// decode) and the user benefits from re-triggering envmap and asset
// reloads independently.
fn spawn_envmap_worker(state: &AssetHotReloadState) {
    let Some(env_map) = state.environment_map.as_ref() else {
        return;
    };
    let mut slot = match state.env_map_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if slot.is_some() {
        tracing::info!(
            "asset hot-reload: EnvironmentMap reload skipped for '{}'; previous \
             convolution still in flight on a worker thread",
            env_map.resolved_path
        );
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let env_map_copy = env_map.clone();
    match std::thread::Builder::new()
        .name("cn-envmap-reload".into())
        .spawn(move || {
            let result = crate::jobs::pool().install(|| {
                crate::build::environment_map::decode_source(
                    &env_map_copy.resolved_path,
                    env_map_copy.prefilter_face_size,
                    env_map_copy.irradiance_face_size,
                    env_map_copy.prefilter_samples,
                    env_map_copy.prefilter_clamp,
                )
            });
            let _ = tx.send(result);
        }) {
        Ok(_) => {
            *slot = Some(rx);
            tracing::info!(
                "asset hot-reload: EnvironmentMap convolution for '{}' spawned \
                 on worker thread (render loop keeps drawing; result will land \
                 on a later frame via poll_pending_envmap)",
                env_map.resolved_path
            );
        }
        Err(e) => {
            tracing::error!(
                "asset hot-reload: failed to spawn EnvironmentMap worker thread \
                 for '{}': {} (IBL kept its old payload)",
                env_map.resolved_path,
                e
            );
        }
    }
}

// Worker body: re-decode every captured source from disk and pack the
// results into a `DecodedAssetBatch`. Per-entry failures are logged at
// `error` and counted in `decode_failures` but never abort the rest of the
// batch: a half-written file is picked up on the next reload when the
// writer finishes. `parsed_glb_cache` amortises `parse_glb` across every
// texture / mesh / skinned mesh that shares the same `.glb` (the showcase
// world fans 43 textures + 35 meshes out of a single 43 MB ABeautifulGame
// chess set, so without the cache the parse cost dominates the worker).
pub(super) fn decode_asset_batch(
    textures: Vec<TextureSourceEntry>,
    color_lut: Option<ColorLutSource>,
    meshes: Vec<MeshSourceEntry>,
    skinned: Vec<SkinnedMeshSourceEntry>,
) -> DecodedAssetBatch {
    use rayon::prelude::*;
    use std::sync::Mutex;

    let mut batch = DecodedAssetBatch::default();
    // Shared parsed-glb cache across textures / meshes / skinned meshes.
    // Wrapped in a Mutex so the par_iter texture loop can populate it from
    // worker threads; contention only fires the first time each unique
    // source is seen.
    let parsed_glb_cache: Mutex<std::collections::HashMap<String, gltf::Gltf>> =
        Mutex::new(std::collections::HashMap::new());

    // Helper: fetch (or parse + cache) the glTF doc for `source`. Returns a
    // cloned `gltf::Gltf`: the underlying doc is reference-counted internally
    // by the `gltf` crate, so the clone is cheap.
    let load_glb = |source: &str| -> Result<gltf::Gltf, String> {
        if let Some(doc) = parsed_glb_cache.lock().unwrap().get(source) {
            return Ok(doc.clone());
        }
        let doc = concinnity_cook::glb::parse_glb(source)?;
        parsed_glb_cache
            .lock()
            .unwrap()
            .insert(source.to_string(), doc.clone());
        Ok(doc)
    };

    // Decode textures in parallel. Each entry is independent; the worker
    // pool's bounded thread count keeps the render thread breathing room.
    let texture_results: Vec<_> = textures
        .par_iter()
        .map(|entry| {
            let decoded = if entry.source.to_lowercase().ends_with(".glb") {
                match load_glb(&entry.source) {
                    Ok(doc) => concinnity_cook::texture::decode_glb_image_from_doc(
                        &doc,
                        &entry.source,
                        entry.image_index,
                    ),
                    Err(e) => Err(e),
                }
            } else {
                concinnity_cook::texture::decode_source(&entry.source, entry.image_index)
            };
            (entry.clone(), decoded)
        })
        .collect();
    for (entry, decoded) in texture_results {
        match decoded {
            Ok((w, h, px)) => batch.textures.push(DecodedTexture {
                slot: entry.slot,
                kind: entry.kind,
                width: w,
                height: h,
                pixels: px,
                source: entry.source,
            }),
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode '{}': {} (slot {} kept its old pixels)",
                    entry.source,
                    e,
                    entry.slot
                );
                batch.decode_failures += 1;
            }
        }
    }

    // ColorLut: single source, decoded serially.
    if let Some(lut) = color_lut {
        match concinnity_cook::color_lut::decode_source(&lut.resolved_path) {
            Ok((size, data)) => {
                batch.color_lut = Some(DecodedColorLut {
                    size,
                    data,
                    source: lut.resolved_path,
                });
            }
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode ColorLut '{}': {} (LUT kept its \
                     old payload)",
                    lut.resolved_path,
                    e
                );
                batch.decode_failures += 1;
            }
        }
    }

    // Static Meshes: re-import each via the shared glb cache. Serial: the
    // per-mesh decode is fast enough that the parallel-decode-then-collect
    // shape isn't worth the bookkeeping (most meshes share their parsed
    // doc, so the parse is amortised regardless).
    for (entry_idx, entry) in meshes.iter().enumerate() {
        let doc = match load_glb(&entry.source) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to parse Mesh source '{}': {} \
                     ({} draw slot(s) kept their old geometry)",
                    entry.source,
                    e,
                    entry.draw_indices.len()
                );
                batch.decode_failures += 1;
                continue;
            }
        };
        match concinnity_cook::mesh_reimport::decode_mesh_from_parsed_glb(
            &doc,
            &entry.source,
            entry.primitive_index,
            entry.lod_levels,
            &entry.lod_distances,
        ) {
            Ok((verts, idxs, lods)) => batch.meshes.push(DecodedMesh {
                entry_idx,
                vertices: verts,
                indices: idxs,
                lod_alternates: lods,
            }),
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode Mesh primitive {} from \
                     '{}': {} ({} draw slot(s) kept their old geometry)",
                    entry.primitive_index,
                    entry.source,
                    e,
                    entry.draw_indices.len()
                );
                batch.decode_failures += 1;
            }
        }
    }

    // Skinned meshes: same pattern as static.
    for (entry_idx, entry) in skinned.iter().enumerate() {
        let doc = match load_glb(&entry.source) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to parse SkinnedMesh source '{}': {} \
                     (skinned slot {} kept its old geometry)",
                    entry.source,
                    e,
                    entry.skinned_index
                );
                batch.decode_failures += 1;
                continue;
            }
        };
        match concinnity_cook::mesh_reimport::decode_skinned_from_parsed_glb(&doc, &entry.source) {
            Ok((verts, idxs, skeleton)) => batch.skinned_meshes.push(DecodedSkinnedMesh {
                entry_idx,
                vertices: verts,
                indices: idxs,
                skeleton,
            }),
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: failed to decode SkinnedMesh '{}': {} \
                     (skinned slot {} kept its old geometry)",
                    entry.source,
                    e,
                    entry.skinned_index
                );
                batch.decode_failures += 1;
            }
        }
    }

    batch
}

// Refresh `SkinnedMeshSourceEntry`s' captured `vertex_base` / `vertex_count`
// / `index_count` from the post-rebuild layouts returned by
// [`crate::gfx::backend::RenderBackend::rebuild_skinned_geometry`]. Entries
// whose `skinned_index` is not in `layouts` are left untouched (the backend
// did not include them in the rebuild), but a typical Metal rebuild
// returns one layout per slot since the whole shared buffer is re-laid out,
// so every entry gets refreshed. Extracted from the apply path so the
// (testable) mutation is independent of the backend call.
pub(super) fn apply_skinned_layouts_to_entries(
    entries: &mut [SkinnedMeshSourceEntry],
    layouts: &[crate::gfx::backend::SkinnedSlotLayout],
) {
    let layout_by_skinned: std::collections::HashMap<
        usize,
        &crate::gfx::backend::SkinnedSlotLayout,
    > = layouts.iter().map(|l| (l.skinned_index, l)).collect();
    for entry in entries.iter_mut() {
        if let Some(layout) = layout_by_skinned.get(&entry.skinned_index) {
            entry.vertex_base = layout.vertex_base;
            entry.vertex_count = layout.vertex_count;
            entry.index_count = layout.index_count;
        }
    }
}

// Try to receive a completed off-thread asset decode batch and push every
// item through the matching backend `update_*` call. Called every frame from
// `GraphicsSystem::step`; cheap when nothing is in flight (a Mutex lock +
// `None` check). Per-item dispatches:
//
// - Textures → `update_texture_slot` / `update_normal_map_slot`.
// - ColorLut → `update_color_lut`.
// - Static Meshes → `update_mesh_geometry` per draw slot when the slot's
//   init-time vertex / index counts (and per-LOD counts) still match the
//   re-imported geometry; otherwise the whole entry's
//   `DrawGeometryUpdate`s are accumulated and applied in a single
//   `rebuild_static_geometry` call after the in-place loop.
// - SkinnedMeshes → `update_skinned_mesh_geometry` per slot when the
//   slot's init-time vertex / index counts still match the re-imported
//   geometry; otherwise the whole entry's
//   `SkinnedDrawGeometryUpdate`s are accumulated and applied in a single
//   `rebuild_skinned_geometry` call after the in-place loop. Joint-count
//   changes resize the backend's per-slot joint-matrix buffers via
//   `update_skinned_skeleton` and queue a `PendingSkeletonUpdate` for the
//   next `GraphicsSystem::step` to refresh the ECS-owned `SkeletonPose`.
//
// Returns `true` when a batch was applied (success or failure), `false`
// when still waiting or nothing scheduled.
pub fn poll_pending_assets(
    state: &mut AssetHotReloadState,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> bool {
    let mut slot = match state.asset_batch_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let rx = match slot.as_ref() {
        Some(r) => r,
        None => return false,
    };
    let batch = match rx.try_recv() {
        Ok(b) => b,
        Err(std::sync::mpsc::TryRecvError::Empty) => return false,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            *slot = None;
            tracing::error!(
                "asset hot-reload: decode worker disconnected without sending a \
                 batch (assets kept their old payloads)"
            );
            return true;
        }
    };
    *slot = None;
    drop(slot);

    let mut reloaded = 0usize;
    let mut failed = batch.decode_failures;

    for tex in &batch.textures {
        let result = match tex.kind {
            TextureKind::Albedo => {
                backend.update_texture_slot(tex.slot, tex.width, tex.height, &tex.pixels)
            }
            TextureKind::NormalMap => {
                backend.update_normal_map_slot(tex.slot, tex.width, tex.height, &tex.pixels)
            }
        };
        match result {
            Ok(()) => reloaded += 1,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: backend rejected slot {} update for '{}': {}",
                    tex.slot,
                    tex.source,
                    e
                );
                failed += 1;
            }
        }
    }

    if let Some(lut) = &batch.color_lut {
        match backend.update_color_lut(lut.size, &lut.data) {
            Ok(()) => reloaded += 1,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: backend rejected ColorLut update for '{}': {}",
                    lut.source,
                    e
                );
                failed += 1;
            }
        }
    }

    // Static meshes: same in-place vs rebuild dispatch the inline path
    // used to do, just driven by the worker's pre-decoded results.
    let mut rebuild_changes: Vec<crate::gfx::backend::DrawGeometryUpdate> = Vec::new();
    let mut rebuild_entry_count = 0usize;
    for dm in &batch.meshes {
        let entry = match state.meshes.entries.get(dm.entry_idx) {
            Some(e) => e,
            None => {
                tracing::warn!(
                    "asset hot-reload: decoded Mesh entry_idx {} out of range (source \
                     map mutated mid-pass?); skipping",
                    dm.entry_idx
                );
                failed += 1;
                continue;
            }
        };
        let size_changed = entry.draw_indices.iter().any(|&draw_idx| {
            let base_changed = match backend.draw_geometry_size(draw_idx) {
                Some((v, i)) => v != dm.vertices.len() || i != dm.indices.len(),
                None => false,
            };
            let lod_changed = match backend.draw_lod_index_counts(draw_idx) {
                Some(counts) => {
                    counts.len() != dm.lod_alternates.len()
                        || counts
                            .iter()
                            .zip(dm.lod_alternates.iter())
                            .any(|(c, (_, idx))| *c != idx.len())
                }
                None => false,
            };
            base_changed || lod_changed
        });
        if size_changed {
            rebuild_entry_count += 1;
            for &draw_idx in &entry.draw_indices {
                rebuild_changes.push(crate::gfx::backend::DrawGeometryUpdate {
                    draw_idx,
                    vertices: dm.vertices.clone(),
                    indices: dm.indices.clone(),
                    lod_alternates: dm.lod_alternates.clone(),
                });
            }
            continue;
        }
        let mut slot_failures = 0usize;
        for &draw_idx in &entry.draw_indices {
            if let Err(e) = backend.update_mesh_geometry(
                draw_idx,
                &dm.vertices,
                &dm.indices,
                &dm.lod_alternates,
            ) {
                tracing::error!(
                    "asset hot-reload: backend rejected mesh update for draw {} \
                     (source '{}', primitive {}): {}",
                    draw_idx,
                    entry.source,
                    entry.primitive_index,
                    e
                );
                slot_failures += 1;
            }
        }
        if slot_failures == 0 {
            reloaded += 1;
        } else {
            failed += 1;
        }
    }
    if !rebuild_changes.is_empty() {
        tracing::info!(
            "asset hot-reload: {} Mesh entry/entries had size changes; rebuilding \
             shared static-geometry buffers ({} draw slot(s) affected)",
            rebuild_entry_count,
            rebuild_changes.len()
        );
        match backend.rebuild_static_geometry(rebuild_changes) {
            Ok(()) => reloaded += rebuild_entry_count,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: rebuild_static_geometry failed: {} \
                     ({} Mesh entry/entries kept their old geometry)",
                    e,
                    rebuild_entry_count
                );
                failed += rebuild_entry_count;
            }
        }
    }

    // Skinned meshes: same in-place vs rebuild dispatch the static-Mesh
    // loop above does, plus a parallel skeleton-shape pass that resizes
    // each affected slot's joint-matrix buffers and queues a SkeletonPose
    // refresh for the next step. Joint-count and vertex/index counts change
    // independently: a tweak to a single joint's bind translation keeps
    // both counts the same, while a re-export with a different skeleton may
    // also rewrite the mesh's vertex count, so the two checks are
    // independent and either path may fire.
    let mut skinned_rebuild_changes: Vec<crate::gfx::backend::SkinnedDrawGeometryUpdate> =
        Vec::new();
    let mut skinned_rebuild_entries: Vec<usize> = Vec::new();
    // Per-slot new joint count, parallel to the corresponding decoded
    // entry; applied after the rebuild dispatch (success path only).
    let mut joint_count_changes: Vec<(usize /*entry_idx*/, usize /*new_count*/)> = Vec::new();
    for ds in &batch.skinned_meshes {
        let entry = match state.skinned_meshes.entries.get(ds.entry_idx) {
            Some(e) => e,
            None => {
                tracing::warn!(
                    "asset hot-reload: decoded SkinnedMesh entry_idx {} out of range; \
                     skipping",
                    ds.entry_idx
                );
                failed += 1;
                continue;
            }
        };
        let new_joint_count = ds.skeleton.len().min(crate::gfx::render_types::MAX_JOINTS);
        let joint_count_changed = new_joint_count != entry.joint_count;
        let size_changed =
            ds.vertices.len() != entry.vertex_count || ds.indices.len() != entry.index_count;
        if joint_count_changed {
            joint_count_changes.push((ds.entry_idx, new_joint_count));
            state.pending_skeleton_updates.push(PendingSkeletonUpdate {
                skinned_index: entry.skinned_index,
                new_skeleton: crate::assets::build_skeleton_from_joint_defs(&ds.skeleton),
            });
            tracing::info!(
                "asset hot-reload: SkinnedMesh '{}' joint count changed ({} → {}), \
                 resizing joint buffers and refreshing SkeletonPose",
                entry.source,
                entry.joint_count,
                new_joint_count
            );
        }
        if size_changed {
            skinned_rebuild_entries.push(ds.entry_idx);
            skinned_rebuild_changes.push(crate::gfx::backend::SkinnedDrawGeometryUpdate {
                skinned_index: entry.skinned_index,
                vertices: ds.vertices.clone(),
                indices: ds.indices.clone(),
            });
            continue;
        }
        match backend.update_skinned_mesh_geometry(
            entry.skinned_index,
            entry.vertex_base,
            &ds.vertices,
            &ds.indices,
        ) {
            Ok(()) => reloaded += 1,
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: backend rejected SkinnedMesh update for \
                     skinned slot {} (source '{}'): {}",
                    entry.skinned_index,
                    entry.source,
                    e
                );
                failed += 1;
            }
        }
    }
    if !skinned_rebuild_changes.is_empty() {
        tracing::info!(
            "asset hot-reload: {} SkinnedMesh entry/entries had size changes; \
             rebuilding shared skinned vertex / index buffers",
            skinned_rebuild_changes.len()
        );
        match backend.rebuild_skinned_geometry(skinned_rebuild_changes) {
            Ok(new_layouts) => {
                // Every slot's vertex_base may have shifted (the rebuild
                // re-packs from scratch). Refresh every captured source
                // entry's layout so the next reload pass's size check uses
                // current state, and so a future
                // `update_skinned_mesh_geometry` call uses the new
                // vertex_base.
                apply_skinned_layouts_to_entries(&mut state.skinned_meshes.entries, &new_layouts);
                reloaded += skinned_rebuild_entries.len();
            }
            Err(e) => {
                tracing::error!(
                    "asset hot-reload: rebuild_skinned_geometry failed: {} \
                     ({} SkinnedMesh entry/entries kept their old geometry)",
                    e,
                    skinned_rebuild_entries.len()
                );
                failed += skinned_rebuild_entries.len();
                // The geometry rebuild failed, so the slot kept its old
                // vertices/indices: discard any queued skeleton update
                // tied to a failed rebuild so the SkeletonPose stays in
                // sync with the unchanged geometry.
                let failed_skinned: std::collections::HashSet<usize> = skinned_rebuild_entries
                    .iter()
                    .filter_map(|&i| state.skinned_meshes.entries.get(i))
                    .map(|e| e.skinned_index)
                    .collect();
                state
                    .pending_skeleton_updates
                    .retain(|u| !failed_skinned.contains(&u.skinned_index));
                // Also drop the joint-count changes for the failed slots so
                // the apply-to-entries step below doesn't write a stale
                // joint_count into the source map.
                joint_count_changes.retain(|(entry_idx, _)| {
                    state
                        .skinned_meshes
                        .entries
                        .get(*entry_idx)
                        .map(|e| !failed_skinned.contains(&e.skinned_index))
                        .unwrap_or(true)
                });
            }
        }
    }
    // Apply queued joint-count changes: resize each affected slot's
    // backend joint-matrix buffers and update the source map's
    // captured `joint_count`. Runs after the (possibly failing) rebuild
    // dispatch so failed rebuilds can prune their joint-count writes
    // before they land.
    for (entry_idx, new_count) in &joint_count_changes {
        let entry = match state.skinned_meshes.entries.get_mut(*entry_idx) {
            Some(e) => e,
            None => continue,
        };
        if let Err(e) = backend.update_skinned_skeleton(entry.skinned_index, *new_count) {
            tracing::error!(
                "asset hot-reload: backend rejected joint-count update for skinned slot {} \
                 (source '{}'): {} (joint_count stays at {})",
                entry.skinned_index,
                entry.source,
                e,
                entry.joint_count
            );
            // Drop the queued SkeletonPose update so it doesn't desync
            // from the unchanged backend joint_count.
            let stale_skinned_index = entry.skinned_index;
            state
                .pending_skeleton_updates
                .retain(|u| u.skinned_index != stale_skinned_index);
            failed += 1;
        } else {
            entry.joint_count = *new_count;
        }
    }

    tracing::info!(
        "asset hot-reload: applied off-thread batch, reloaded {} asset(s) ({} failed)",
        reloaded,
        failed
    );
    true
}

// Try to receive a completed off-thread EnvironmentMap convolution and push
// the payload to the backend. Called every frame from `GraphicsSystem::step`;
// cheap when nothing is in flight (a Mutex lock + `None` check). Returns
// `true` when something was consumed (success or failure) so the caller can
// log a single line per result; `false` when still waiting or nothing
// scheduled.
pub fn poll_pending_envmap(
    state: &AssetHotReloadState,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> bool {
    let mut slot = match state.env_map_inflight.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let rx = match slot.as_ref() {
        Some(r) => r,
        None => return false,
    };
    match rx.try_recv() {
        Ok(Ok(payload)) => {
            *slot = None;
            // Drop the lock before the (potentially long) backend call so a
            // re-entrant `reload_assets` invocation from the same step
            // doesn't deadlock. Backend update is fast (it's a single
            // cubemap-pair swap) so dropping is just defensive.
            drop(slot);
            match backend.update_environment_map(&payload) {
                Ok(()) => {
                    tracing::info!(
                        "asset hot-reload: EnvironmentMap convolution completed \
                         off-thread and applied to IBL cubes"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        "asset hot-reload: backend rejected off-thread EnvironmentMap \
                         payload: {} (IBL kept its old payload)",
                        e
                    );
                }
            }
            true
        }
        Ok(Err(msg)) => {
            *slot = None;
            tracing::error!(
                "asset hot-reload: off-thread EnvironmentMap convolution failed: {} \
                 (IBL kept its old payload)",
                msg
            );
            true
        }
        Err(std::sync::mpsc::TryRecvError::Empty) => false,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
            *slot = None;
            tracing::error!(
                "asset hot-reload: EnvironmentMap worker thread disconnected without \
                 sending a result (IBL kept its old payload)"
            );
            true
        }
    }
}
