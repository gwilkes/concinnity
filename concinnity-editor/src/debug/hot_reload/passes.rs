// src/debug/hot_reload/passes.rs
//
// The world.jsonl / ProceduralMesh / VolumetricFog / world-loaded ShaderStage
// reload passes: re-read the on-disk source, diff against the captured state,
// and apply changes through the backend. Each returns a small tally the drive
// logs.

use crate::gfx::graphics_system::hot_reload_sources::*;

// Per-reload tally returned by [`reload_world`]: the counts each caller logs.
// `added_props` is the list of newly authored `Prop`s the caller must push
// into the ECS (the helper cannot see `PipelineContext`).
#[derive(Default, Debug)]
pub struct WorldReloadResult {
    // Transform slots whose world matrix was pushed via `update_model`.
    pub transforms_applied: usize,
    // Prop slots removed from the world: their draws were hidden via
    // `update_visibility(_, false)`.
    pub removed: usize,
    // Prop slots newly added: a draw object was cloned per sub-mesh from an
    // existing template, bookkeeping vecs were grown.
    pub added: usize,
    // Prop slots whose non-transform args (material, texture, cull_distance,
    // scene, parent) were re-applied to existing draw slots.
    pub modified: usize,
    // Prop slots skipped because their mesh / model is not in the init world,
    // `prefab` was authored, or another arg change cannot be applied without
    // a full process restart. Logged separately so the user can tell which
    // edits require a relaunch.
    pub restart_required: usize,
    // Newly authored Props to push into the ECS via `ctx.push`. Caller-side
    // because the helper has no `PipelineContext` borrow.
    pub added_props: Vec<crate::assets::Prop>,
}

// Re-read `world.jsonl` from disk and apply every diffable change to the
// live scene: transforms, adds, removes, and non-transform arg edits
// (material, texture, cull_distance, scene, parent). Mesh / model / prefab
// changes and adds whose mesh isn't already in the world are detected and
// counted as `restart_required`.
//
// Mutates the caller's per-prop bookkeeping vecs in place: hidden draws stay
// in `prop_draw_indices` (the chunk-streaming pattern), adds grow every vec
// by one entry per new Prop. The `tracked_props` snapshot is also kept in
// sync so a subsequent reload sees the post-edit state.
//
// Adds that reference a mesh / model already in the world clone an existing
// draw object's geometry region: no new bytes go into the shared vertex /
// index buffers, and the new draw is non-cullable (sentinel AABB). Adds
// that reference an unknown mesh are logged + counted as `restart_required`
// since this V1 cannot load new geometry.
pub(super) fn reload_world(
    path: &str,
    tracked_props: &mut Vec<crate::assets::Prop>,
    prop_parents: &mut Vec<Option<usize>>,
    prop_draw_indices: &mut Vec<Vec<usize>>,
    prop_scene: &mut Vec<Option<crate::ecs::asset_id::AssetId>>,
    world_reload: &crate::gfx::graphics_system::WorldReloadState,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> WorldReloadResult {
    let mut result = WorldReloadResult::default();

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                "world.jsonl hot-reload: failed to read '{}': {} (no edits applied)",
                path,
                e
            );
            return result;
        }
    };
    // Use the same expansion the build pipeline runs at init so synthetic
    // entries (prefab expansions, LightRig children, companion-injected
    // systems, etc.) appear in the parsed list and the diff doesn't
    // misclassify them as removed-from-jsonl on every reload.
    let entries = match concinnity_cook::world::expand_world_from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                "world.jsonl hot-reload: failed to parse '{}': {} (no edits applied)",
                path,
                e
            );
            return result;
        }
    };

    // Build name -> Prop map from the new JSONL. The interner returns the same
    // AssetId for an existing Prop name (matches tracked_props by id) and
    // fresh ids for new names (which become adds).
    let mut new_by_id: std::collections::HashMap<
        crate::ecs::asset_id::AssetId,
        crate::assets::Prop,
    > = std::collections::HashMap::new();
    for entry in &entries {
        let asset_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if asset_type != "Prop" {
            continue;
        }
        let Some(name) = entry.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let args = entry
            .get("args")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let parsed: crate::assets::Prop = match serde_json::from_value(args) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "world.jsonl hot-reload: failed to parse Prop '{}' args: {} (kept old state)",
                    name,
                    e
                );
                continue;
            }
        };
        let asset_id = crate::ecs::asset_id::intern(name);
        let mut with_id = parsed;
        with_id.asset_id = asset_id;
        // Resolve the prop's `scene` from its name (`<scene>_*`) the same way
        // build/pipeline.rs::resolve_scene_refs does at build time, so a Prop
        // added by hot-reload routes through SceneReel correctly. Only
        // applied when the JSONL didn't carry an explicit `scene` arg.
        if with_id.scene.is_none() {
            with_id.scene = resolve_scene_from_name(name, &world_reload.scene_names);
        }
        new_by_id.insert(asset_id, with_id);
    }

    // Per-prop diff against tracked_props. The index loop walks
    // tracked_props in place because we mutate prop_draw_indices / parents /
    // scene / tracked_props by the same index.
    for prop_idx in 0..tracked_props.len() {
        let init_id = tracked_props[prop_idx].asset_id;
        match new_by_id.remove(&init_id) {
            // Existing prop survives: apply diffable edits.
            Some(new_prop) => {
                let init_prop = &tracked_props[prop_idx];

                // 1. Non-transform edits that this V1 still cannot apply.
                let mut needs_restart = false;
                if init_prop.mesh != new_prop.mesh
                    || init_prop.model != new_prop.model
                    || init_prop.prefab != new_prop.prefab
                {
                    tracing::warn!(
                        "world.jsonl hot-reload: Prop '{}' mesh/model/prefab changed; \
                         needs a full restart to take effect",
                        init_id
                    );
                    needs_restart = true;
                }

                let draw_idxs = prop_draw_indices.get(prop_idx).cloned().unwrap_or_default();

                // 2. Material / texture edits: rewrite per draw slot.
                if init_prop.material != new_prop.material || init_prop.texture != new_prop.texture
                {
                    if let Some(model_id) = new_prop.model {
                        // Multi-mesh model: each submesh's material is fixed
                        // by the Model definition, not the Prop, so a Prop
                        // material edit only matters when there is no model.
                        // Log + leave alone.
                        let _ = model_id;
                    } else if let Some((tex_slot, nm_slot, uniforms)) =
                        resolve_material_or_texture(&new_prop, world_reload)
                    {
                        for &draw_idx in &draw_idxs {
                            backend.set_draw_material(draw_idx, uniforms, tex_slot, nm_slot);
                        }
                    } else {
                        tracing::warn!(
                            "world.jsonl hot-reload: Prop '{}' new material/texture is not \
                             in the init asset set; keeping previous material",
                            init_id
                        );
                    }
                }

                // 3. Cull-distance edit: per draw slot.
                if (init_prop.cull_distance - new_prop.cull_distance).abs() > f32::EPSILON {
                    for &draw_idx in &draw_idxs {
                        backend.set_draw_cull_distance(draw_idx, new_prop.cull_distance);
                    }
                }

                // 4. Parent / scene bookkeeping. Parent re-resolution updates
                // the per-prop parent index so the per-frame transform loop
                // picks the new chain up automatically. Scene goes through
                // prop_scene and SceneReel.
                if init_prop.parent != new_prop.parent {
                    let new_parent_idx = new_prop
                        .parent
                        .and_then(|pid| tracked_props.iter().position(|p| p.asset_id == pid));
                    if new_prop.parent.is_some() && new_parent_idx.is_none() {
                        tracing::warn!(
                            "world.jsonl hot-reload: Prop '{}' parent not found in tracked \
                             props; leaving parent unchanged",
                            init_id
                        );
                    } else {
                        prop_parents[prop_idx] = new_parent_idx;
                    }
                }
                if init_prop.scene != new_prop.scene {
                    prop_scene[prop_idx] = new_prop.scene;
                }

                // Track in counts: any change-but-not-restart counts as
                // modified; restart-required is its own bucket. Transforms
                // are always re-pushed below.
                let transform_changed = init_prop.position != new_prop.position
                    || init_prop.rotation_deg != new_prop.rotation_deg
                    || init_prop.scale != new_prop.scale;
                let non_transform_changed = init_prop.material != new_prop.material
                    || init_prop.texture != new_prop.texture
                    || (init_prop.cull_distance - new_prop.cull_distance).abs() > f32::EPSILON
                    || init_prop.parent != new_prop.parent
                    || init_prop.scene != new_prop.scene;
                if non_transform_changed && !needs_restart {
                    result.modified += 1;
                }
                if needs_restart {
                    result.restart_required += 1;
                }

                // Write the new args back into tracked_props so a subsequent
                // reload diffs against the post-edit state, except mesh /
                // model / prefab, which only take effect on restart and
                // therefore stay as the live init values.
                tracked_props[prop_idx].position = new_prop.position;
                tracked_props[prop_idx].rotation_deg = new_prop.rotation_deg;
                tracked_props[prop_idx].scale = new_prop.scale;
                tracked_props[prop_idx].material = new_prop.material;
                tracked_props[prop_idx].texture = new_prop.texture;
                tracked_props[prop_idx].cull_distance = new_prop.cull_distance;
                tracked_props[prop_idx].parent = new_prop.parent;
                tracked_props[prop_idx].scene = new_prop.scene;
                tracked_props[prop_idx].interactable = new_prop.interactable;
                tracked_props[prop_idx].pickup = new_prop.pickup;
                tracked_props[prop_idx].collider = new_prop.collider.clone();
                let _ = transform_changed; // counted by transforms_applied below
            }
            // Init prop missing from the new JSONL: treat as removed.
            None => {
                if let Some(draw_idxs) = prop_draw_indices.get_mut(prop_idx) {
                    for &draw_idx in draw_idxs.iter() {
                        backend.update_visibility(draw_idx, false);
                    }
                    draw_idxs.clear();
                }
                // The Prop component stays in the ECS (no remove API), but
                // its draw slots are hidden and the prop_draw_indices entry
                // is empty so the per-frame transform push is a no-op.
                result.removed += 1;
            }
        }
    }

    // The remaining entries in new_by_id are adds (names that weren't in
    // tracked_props). Apply them in deterministic order so logs are stable
    // across runs.
    let mut additions: Vec<(crate::ecs::asset_id::AssetId, crate::assets::Prop)> =
        new_by_id.into_iter().collect();
    additions.sort_by_key(|(id, _)| *id);
    for (id, new_prop) in additions {
        if !new_prop.prefab.is_empty() {
            tracing::warn!(
                "world.jsonl hot-reload: Prop '{}' authors a prefab; needs a full restart",
                id
            );
            result.restart_required += 1;
            continue;
        }

        // Figure out which template draw(s) to clone from.
        let template_draws: Vec<(
            usize,
            crate::gfx::render_types::MaterialUniforms,
            usize,
            usize,
        )> = if let Some(model_id) = new_prop.model {
            let submeshes = match world_reload.model_map.get(&model_id) {
                Some(s) => s,
                None => {
                    tracing::warn!(
                        "world.jsonl hot-reload: Prop '{}' model '{}' unknown; needs restart",
                        id,
                        model_id
                    );
                    result.restart_required += 1;
                    continue;
                }
            };
            let mut subs = Vec::with_capacity(submeshes.len());
            let mut ok = true;
            for sub in submeshes {
                let Some(sub_mesh) = sub.mesh else {
                    ok = false;
                    break;
                };
                let Some(&src_draw) = world_reload.mesh_id_to_draw.get(&sub_mesh) else {
                    ok = false;
                    break;
                };
                let (tex, nm, uni) = match sub.material {
                    Some(mat_id) => match world_reload.material_map.get(&mat_id) {
                        Some(v) => *v,
                        None => {
                            ok = false;
                            break;
                        }
                    },
                    None => (0, 0, crate::gfx::render_types::MaterialUniforms::DEFAULT),
                };
                subs.push((src_draw, uni, tex, nm));
            }
            if !ok {
                tracing::warn!(
                    "world.jsonl hot-reload: Prop '{}' model '{}' references assets not \
                         in the init world; needs restart",
                    id,
                    model_id
                );
                result.restart_required += 1;
                continue;
            }
            subs
        } else if let Some(mesh_id) = new_prop.mesh {
            let Some(&src_draw) = world_reload.mesh_id_to_draw.get(&mesh_id) else {
                tracing::warn!(
                    "world.jsonl hot-reload: Prop '{}' mesh '{}' unknown; needs restart",
                    id,
                    mesh_id
                );
                result.restart_required += 1;
                continue;
            };
            let (tex, nm, uni) = match resolve_material_or_texture(&new_prop, world_reload) {
                Some(v) => v,
                None => {
                    tracing::warn!(
                        "world.jsonl hot-reload: Prop '{}' material/texture not in init \
                             asset set; needs restart",
                        id
                    );
                    result.restart_required += 1;
                    continue;
                }
            };
            vec![(src_draw, uni, tex, nm)]
        } else {
            tracing::warn!(
                "world.jsonl hot-reload: Prop '{}' has no mesh or model; ignored",
                id
            );
            result.restart_required += 1;
            continue;
        };

        // Resolve parent against the post-edit tracked_props order. If the
        // parent was also added in this batch, it has not been pushed yet:
        // fall back to no parent and warn.
        let parent_idx = new_prop
            .parent
            .and_then(|pid| tracked_props.iter().position(|p| p.asset_id == pid));
        if let Some(parent) = new_prop.parent
            && parent_idx.is_none()
        {
            tracing::warn!(
                "world.jsonl hot-reload: added Prop '{}' parent '{}' missing from tracked \
                 props; placing in world space",
                id,
                parent
            );
        }

        // Compute the new prop's world matrix the same way init does:
        // append a slot, then compute_world_matrices over the grown list.
        // Indices in template_draws are stable (we never remove drawObjects).
        let new_prop_idx = tracked_props.len();
        tracked_props.push(new_prop.clone());
        prop_parents.push(parent_idx);
        prop_scene.push(new_prop.scene);

        let tracked_refs: Vec<&crate::assets::Prop> = tracked_props.iter().collect();
        let world_mats = crate::gfx::draw_list::compute_world_matrices(&tracked_refs, prop_parents);
        let model_mat = world_mats[new_prop_idx];

        // Clone each sub-draw with the new transform / material / cull
        // distance and remember its new draw_idx for bookkeeping.
        let mut new_draw_idxs: Vec<usize> = Vec::with_capacity(template_draws.len());
        let mut clone_failed = false;
        for (src_draw, uni, tex_slot, nm_slot) in template_draws {
            match backend.clone_static_draw_object(
                src_draw,
                model_mat,
                tex_slot,
                nm_slot,
                uni,
                new_prop.cull_distance,
            ) {
                Ok(new_idx) => new_draw_idxs.push(new_idx),
                Err(e) => {
                    tracing::warn!(
                        "world.jsonl hot-reload: clone_static_draw_object for Prop '{}' failed: {}",
                        id,
                        e
                    );
                    clone_failed = true;
                    break;
                }
            }
        }
        if clone_failed {
            // Roll back the bookkeeping vecs to keep them parallel; leave
            // any partial draws hidden so they don't render.
            for &di in &new_draw_idxs {
                backend.update_visibility(di, false);
            }
            tracked_props.pop();
            prop_parents.pop();
            prop_scene.pop();
            result.restart_required += 1;
            continue;
        }
        prop_draw_indices.push(new_draw_idxs);
        result.added += 1;
        result.added_props.push(new_prop);
    }

    // Push every tracked Prop's fresh world matrix. compute_world_matrices
    // sees the post-edit tracked_props + prop_parents, so parent changes and
    // adds (with their cloned-then-positioned draws) all reflect here.
    let tracked_refs: Vec<&crate::assets::Prop> = tracked_props.iter().collect();
    let worlds = crate::gfx::draw_list::compute_world_matrices(&tracked_refs, prop_parents);
    for (prop_idx, mat) in worlds.iter().enumerate() {
        let Some(draw_idxs) = prop_draw_indices.get(prop_idx) else {
            continue;
        };
        if draw_idxs.is_empty() {
            continue;
        }
        for &draw_idx in draw_idxs {
            backend.update_model(draw_idx, *mat);
        }
        result.transforms_applied += 1;
    }

    result
}

// Per-reload tally for the volumetric-fog path. Sibling of
// [`WorldReloadResult`]; counts are surfaced separately so a single info!
// line per asset class keeps the log readable.
#[derive(Default, Debug)]
pub struct FogReloadResult {
    // True when the resolved `Option<FogSettings>` differed from what was
    // last pushed; the trait call fired and the dedupe state advanced.
    pub updated: bool,
}

// Re-read `world.jsonl` and push the first declared `VolumetricFog` through
// [`RenderBackend::update_fog_settings`]. Disabled / missing assets push
// `None`. Each authored asset runs through the same clamp chain as init:
// [`crate::assets::VolumetricFog::from_args`] (asset-side floors) then
// [`crate::gfx::volumetric_fog::FogSettings::resolve`] (gfx-side ceilings),
// so a reload cannot land out-of-range values.
//
// `last_pushed` is the dedupe state owned by the caller; the function
// compares the freshly-resolved value against it and only fires the trait
// call (and reports `updated = true`) on a real change. Backend-agnostic:
// rides on the `update_fog_settings` trait, which has a default no-op on
// backends that have not yet implemented runtime fog mutation.
pub(super) fn reload_volumetric_fog(
    path: &str,
    last_pushed: &mut Option<crate::gfx::volumetric_fog::FogSettings>,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> FogReloadResult {
    use crate::ecs::Component;
    let mut result = FogReloadResult::default();

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                "VolumetricFog hot-reload: failed to read '{}': {} (no update)",
                path,
                e,
            );
            return result;
        }
    };
    let entries = match concinnity_cook::world::expand_world_from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                "VolumetricFog hot-reload: failed to parse '{}': {} (no update)",
                path,
                e,
            );
            return result;
        }
    };

    // First declared VolumetricFog wins; mirrors the init drain in
    // `run_init`. A missing entry or one with `enabled = false` resolves to
    // `None`, which disables the pass.
    let mut resolved: Option<crate::gfx::volumetric_fog::FogSettings> = None;
    for entry in &entries {
        let asset_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if asset_type != "VolumetricFog" {
            continue;
        }
        let args = entry
            .get("args")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let parsed: crate::assets::VolumetricFog = match serde_json::from_value(args) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "VolumetricFog hot-reload: failed to parse args: {} (kept previous state)",
                    e,
                );
                return result;
            }
        };
        let clamped = crate::assets::VolumetricFog::from_args(parsed);
        if clamped.enabled {
            resolved = Some(crate::gfx::volumetric_fog::FogSettings::resolve(
                clamped.color,
                clamped.density,
                clamped.height_falloff,
                clamped.height_reference,
                clamped.max_distance,
                clamped.phase_g,
                clamped.ambient,
            ));
        }
        break;
    }

    if resolved != *last_pushed {
        backend.update_fog_settings(resolved);
        *last_pushed = resolved;
        result.updated = true;
    }
    result
}

// Per-reload tally for the procedural-mesh path. Sibling of
// [`WorldReloadResult`]; counts are surfaced separately so a single info!
// line per asset class keeps the log readable.
#[derive(Default, Debug)]
pub struct ProceduralMeshReloadResult {
    // Procedural meshes whose args differed from the captured snapshot and
    // whose regenerated geometry was pushed to the backend (in-place
    // `update_mesh_geometry` or batched `rebuild_static_geometry`).
    pub regenerated: usize,
    // Procedural meshes whose args matched the snapshot: nothing pushed.
    pub unchanged: usize,
    // Procedural meshes whose generator returned an error (left at the
    // pre-reload geometry).
    pub failed: usize,
}

// Re-read `world.jsonl` and, for every captured `ProceduralMesh`, diff its
// current generator args against the init-time snapshot. When the args
// differ, re-run `compile_mesh_payload` + `deserialise_with_lods` to get
// fresh vertices / indices / LOD alternates, then dispatch to
// `update_mesh_geometry` (same vertex/index/LOD counts) or batch into a
// single `rebuild_static_geometry` (size changed); mirrors the file-backed
// `Mesh` reload's in-place-vs-rebuild dispatch but driven by JSONL args
// instead of a `.glb` file. Updates the captured args on success so a
// subsequent reload diffs against the post-edit state.
//
// Backend-agnostic: rides on the existing `update_mesh_geometry` /
// `rebuild_static_geometry` trait surface, so it lights up on every backend
// as soon as those have real implementations (Metal today; default no-ops on
// Vulkan + DirectX).
pub(super) fn reload_procedural_meshes(
    path: &str,
    procedural_meshes: &mut ProceduralMeshSourceMap,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> ProceduralMeshReloadResult {
    let mut result = ProceduralMeshReloadResult::default();
    if procedural_meshes.is_empty() {
        return result;
    }

    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!(
                "ProceduralMesh hot-reload: failed to read '{}': {} (no regen)",
                path,
                e
            );
            return result;
        }
    };
    // Expand prefabs / etc. so auto-injected ProceduralMeshes match the
    // init-time captured set; otherwise an unchanged entry shows up as
    // "missing from JSONL" every reload.
    let entries = match concinnity_cook::world::expand_world_from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                "ProceduralMesh hot-reload: failed to parse '{}': {} (no regen)",
                path,
                e
            );
            return result;
        }
    };

    // Build a name -> args map from the new JSONL for O(N) per-entry lookup.
    // Normalised by `ProceduralMesh::deserialize → serialize` so the diff
    // against the captured args (which went through the same round-trip at
    // init) sees identical filled-in defaults.
    let mut new_args_by_name: std::collections::HashMap<String, serde_json::Value> =
        std::collections::HashMap::new();
    for entry in &entries {
        let asset_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if asset_type != "ProceduralMesh" {
            continue;
        }
        let Some(name) = entry.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let raw_args = entry
            .get("args")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let parsed: crate::assets::ProceduralMesh = match serde_json::from_value(raw_args.clone()) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    "ProceduralMesh hot-reload: failed to parse '{}' args: {} \
                         (kept old geometry)",
                    name,
                    e
                );
                continue;
            }
        };
        let normalised = match serde_json::to_value(&parsed) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    "ProceduralMesh hot-reload: failed to re-serialise '{}' args: {} \
                     (kept old geometry)",
                    name,
                    e
                );
                continue;
            }
        };
        new_args_by_name.insert(name.to_string(), normalised);
    }

    // Collect rebuild changes batched into a single rebuild call (mirrors the
    // file-backed Mesh path). Each entry either commits in place, queues a
    // rebuild change, or is unchanged.
    let mut rebuild_changes: Vec<crate::gfx::backend::DrawGeometryUpdate> = Vec::new();
    let mut rebuild_entry_indices: Vec<usize> = Vec::new();
    // Args to write back into the source map after a successful update,
    // staged here so a failed regen / rebuild doesn't clobber the captured
    // value (the diff next reload would then miss the still-pending edit).
    let mut staged_args: Vec<(usize, serde_json::Value)> = Vec::new();

    for (entry_idx, entry) in procedural_meshes.entries.iter().enumerate() {
        let Some(new_args) = new_args_by_name.get(&entry.name) else {
            // No matching entry in the new JSONL: treat as removed-from-jsonl.
            // We deliberately do not destroy the existing draws here: a Prop
            // referencing this mesh is still rendering through its draw slot.
            // The world.jsonl Prop diff (sibling `reload_world` pass) is the
            // path that hides those props if they're gone too.
            result.unchanged += 1;
            continue;
        };
        if new_args == &entry.args {
            result.unchanged += 1;
            continue;
        }

        // Regenerate from the new args. Routed through the build wrapper so a
        // live-edited `heightfield` ProceduralMesh still decodes its source
        // image (core's compile_mesh_payload links no image decoders).
        let payload = match concinnity_cook::mesh_compile::compile_mesh_payload(new_args) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    "ProceduralMesh hot-reload: regen failed for '{}': {} (kept old geometry)",
                    entry.name,
                    e
                );
                result.failed += 1;
                continue;
            }
        };
        let (vertices, indices, lod_alternates) =
            match crate::gfx::mesh_payload::deserialise_with_lods(&payload) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(
                        "ProceduralMesh hot-reload: deserialise failed for '{}': {} \
                         (kept old geometry)",
                        entry.name,
                        e
                    );
                    result.failed += 1;
                    continue;
                }
            };

        // Match the file-backed Mesh dispatch: size-changing regens go through
        // `rebuild_static_geometry`, size-matched go through
        // `update_mesh_geometry`. The size check probes any one of the draw
        // slots: they were built from the same source so they all carry the
        // same vertex / index / LOD counts.
        let size_changed = entry.draw_indices.iter().any(|&draw_idx| {
            let base_changed = match backend.draw_geometry_size(draw_idx) {
                Some((v, i)) => v != vertices.len() || i != indices.len(),
                None => false,
            };
            let lod_changed = match backend.draw_lod_index_counts(draw_idx) {
                Some(counts) => {
                    counts.len() != lod_alternates.len()
                        || counts
                            .iter()
                            .zip(lod_alternates.iter())
                            .any(|(c, (_, idx))| *c != idx.len())
                }
                None => false,
            };
            base_changed || lod_changed
        });

        if size_changed {
            for &draw_idx in &entry.draw_indices {
                rebuild_changes.push(crate::gfx::backend::DrawGeometryUpdate {
                    draw_idx,
                    vertices: vertices.clone(),
                    indices: indices.clone(),
                    lod_alternates: lod_alternates.clone(),
                });
            }
            rebuild_entry_indices.push(entry_idx);
            staged_args.push((entry_idx, new_args.clone()));
            continue;
        }

        let mut slot_failures = 0usize;
        for &draw_idx in &entry.draw_indices {
            if let Err(e) =
                backend.update_mesh_geometry(draw_idx, &vertices, &indices, &lod_alternates)
            {
                tracing::warn!(
                    "ProceduralMesh hot-reload: backend rejected update for '{}' draw {}: \
                     {}",
                    entry.name,
                    draw_idx,
                    e
                );
                slot_failures += 1;
            }
        }
        if slot_failures == 0 {
            result.regenerated += 1;
            staged_args.push((entry_idx, new_args.clone()));
        } else {
            result.failed += 1;
        }
    }

    if !rebuild_changes.is_empty() {
        match backend.rebuild_static_geometry(rebuild_changes) {
            Ok(()) => {
                result.regenerated += rebuild_entry_indices.len();
            }
            Err(e) => {
                tracing::error!(
                    "ProceduralMesh hot-reload: rebuild_static_geometry failed: {} \
                     ({} entry/entries kept old geometry)",
                    e,
                    rebuild_entry_indices.len()
                );
                result.failed += rebuild_entry_indices.len();
                // Drop the staged-args writes for the failed rebuild so the
                // captured args still reflect what the backend is actually
                // rendering.
                let failed_idxs: std::collections::HashSet<usize> =
                    rebuild_entry_indices.iter().copied().collect();
                staged_args.retain(|(idx, _)| !failed_idxs.contains(idx));
            }
        }
    }

    for (entry_idx, new_args) in staged_args {
        if let Some(e) = procedural_meshes.entries.get_mut(entry_idx) {
            e.args = new_args;
        }
    }

    result
}

// Per-reload tally for the world-loaded shader-stage path. Counts are
// surfaced separately from the asset-payload / world / procedural-mesh
// tallies so a single info line per asset class keeps the log readable.
#[derive(Default, Debug)]
pub struct ShaderStageReloadResult {
    // `ShaderStage` sources that re-compiled successfully and contributed
    // to the backend pipeline rebuild. Counts every kind that the caller
    // re-read off disk, including stages whose source bytes were
    // byte-for-byte unchanged (a no-op shader save still re-compiles,
    // detecting a no-op would mean caching every shader's byte payload).
    pub recompiled: usize,
    // `ShaderStage` sources that failed to compile (logged per-stage in
    // the helper). The backend rebuild is skipped entirely on any failure:
    // the live pipelines stay bound and the next save can recover.
    pub failed: usize,
    // `true` when the backend pipeline rebuild ran and swapped fresh
    // pipelines into the live context. `false` when one or more stages
    // failed to compile or the backend rejected the rebuild: the next
    // save retries from scratch.
    pub pipelines_rebuilt: bool,
}

// Re-compile every captured world-loaded `ShaderStage` source from disk and
// hand the fresh bytes to the backend for a main / instanced / shadow
// pipeline rebuild. Sibling of [`reload_shaders`] in
// [`crate::metal::hot_reload`] (which targets the engine's bundled shader
// directory) but driven by the asset hot-reload
// watcher when one of the world's authored shader sources changes.
//
// Each captured kind goes through
// [`concinnity_cook::shader::compile_shader`]; on any per-stage compile
// failure the helper logs and aborts the whole pass without touching the
// backend, so a typo in one shader never desyncs the others. When every
// stage compiles cleanly the bytes are forwarded through
// [`crate::gfx::backend::RenderBackend::update_world_shader_pipelines`],
// which runs the rebuild-then-swap dance on Metal (and is a no-op on
// Vulkan + DirectX until those backends grow an impl).
//
// Does not detect "unchanged source": every fire recompiles every
// captured stage. A `.metal` save is rare enough (and the compile cheap
// enough) that the savings from per-source mtime tracking would not be
// worth the additional state.
pub(super) fn reload_shader_stages(
    shader_stages: &ShaderStageSourceMap,
    backend: &mut dyn crate::gfx::backend::RenderBackend,
) -> ShaderStageReloadResult {
    use crate::assets::shader_stage::ShaderKind;

    let mut result = ShaderStageReloadResult::default();
    if shader_stages.is_empty() {
        return result;
    }

    let mut compiled: std::collections::HashMap<ShaderKind, Vec<u8>> =
        std::collections::HashMap::new();
    for entry in &shader_stages.entries {
        let compile_args = concinnity_cook::shader::ShaderCompileArgs {
            source_path: entry.resolved_path.clone(),
            // Asset name is used by the metal compiler to derive temp paths
            // for the `.air` / `.metallib` intermediates. The source's bare
            // filename (without extension) is a stable per-stage identifier:
            // two stages on different kinds with the same source file
            // are rare in practice but the kind suffix keeps the paths
            // disjoint if it ever happens.
            asset_name: format!(
                "{}_{}",
                std::path::Path::new(&entry.resolved_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("shader_stage"),
                match entry.kind {
                    ShaderKind::Vertex => "vert",
                    ShaderKind::Fragment => "frag",
                    ShaderKind::VertexInstanced => "vert_inst",
                }
            ),
            kind: entry.kind.compile_kind().to_string(),
        };
        match concinnity_cook::shader::compile_shader(compile_args) {
            Ok(bytes) => {
                compiled.insert(entry.kind.clone(), bytes);
                result.recompiled += 1;
            }
            Err(e) => {
                tracing::error!(
                    "ShaderStage hot-reload: failed to compile '{}' ({:?}): {} \
                     (live pipelines kept their previous source)",
                    entry.resolved_path,
                    entry.kind,
                    e
                );
                result.failed += 1;
            }
        }
    }
    if result.failed > 0 {
        return result;
    }

    let vert = compiled.get(&ShaderKind::Vertex).map(|v| v.as_slice());
    let frag = compiled.get(&ShaderKind::Fragment).map(|v| v.as_slice());
    let vert_inst = compiled
        .get(&ShaderKind::VertexInstanced)
        .map(|v| v.as_slice());

    // Shadow shaders are engine-internal: never world-asset hot-reloaded here.
    match backend.update_world_shader_pipelines(vert, frag, None, vert_inst) {
        Ok(()) => {
            result.pipelines_rebuilt = true;
        }
        Err(e) => {
            tracing::error!(
                "ShaderStage hot-reload: backend pipeline rebuild rejected: {} \
                 (live pipelines kept their previous source)",
                e
            );
        }
    }
    result
}

// Resolve a Prop's (texture_slot, normal_map_slot, MaterialUniforms) the
// same way [`crate::gfx::draw_list::build_draw_list`] does at init time,
// using only the maps captured into [`crate::gfx::graphics_system::WorldReloadState`]. Returns
// `None` when the prop references a Material / Texture that wasn't in the
// init world.
pub(super) fn resolve_material_or_texture(
    prop: &crate::assets::Prop,
    world_reload: &crate::gfx::graphics_system::WorldReloadState,
) -> Option<(usize, usize, crate::gfx::render_types::MaterialUniforms)> {
    if let Some(mat_id) = prop.material {
        world_reload.material_map.get(&mat_id).copied()
    } else if let Some(tex_id) = prop.texture {
        let slot = *world_reload.texture_name_to_slot.get(&tex_id)?;
        Some((slot, 0, crate::gfx::render_types::MaterialUniforms::DEFAULT))
    } else {
        Some((0, 0, crate::gfx::render_types::MaterialUniforms::DEFAULT))
    }
}

// Resolve a Prop's `scene` from its name using the `<scene>_*` convention
// applied by [`crate::build::pipeline::resolve_scene_refs`] at build time:
// match each declared scene name `sn` against `name.starts_with("{sn}_")`,
// returning the interned id of the first match. Returns `None` when no
// declared scene's prefix matches: the Prop is then unscoped (always
// visible), same as build-time behavior.
pub(super) fn resolve_scene_from_name(
    name: &str,
    scene_names: &[String],
) -> Option<crate::ecs::asset_id::AssetId> {
    for sn in scene_names {
        if name.starts_with(&format!("{sn}_")) {
            return Some(crate::ecs::asset_id::intern(sn));
        }
    }
    None
}
