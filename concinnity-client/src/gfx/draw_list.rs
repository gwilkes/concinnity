// src/gfx/draw_list.rs
//
// Render-prep helpers that consume asset components and produce GPU-ready data.
// None of these functions hold or borrow a backend handle.

use crate::assets::{
    File, FileKind, InstancedProp, Mesh, ProceduralMesh, Prop, Room, SubMeshRef, VoxelChunk,
};
use crate::ecs::PipelineContext;
use crate::ecs::asset_id::AssetId;
use crate::gfx::mesh_payload::Vertex;
use crate::gfx::render_types::{DrawObject, InstancedCluster, LodSlice, MaterialUniforms};

pub(crate) const IDENTITY4: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

// (albedo_slot, normal_map_slot, gpu material uniforms), passed through build_draw_list.
pub(crate) type MaterialEntry = (usize, usize, MaterialUniforms);

// Geometry decoded for one Room: the asset, its vertices, LOD0 indices, and
// LOD alternates (switch_distance, indices).
pub(crate) type RoomGeometry = (Room, Vec<Vertex>, Vec<u16>, Vec<(f32, Vec<u16>)>);

// Mesh-geometry lookup tables from `load_mesh_geometry`: loaded meshes, their
// source metadata, and the always-resident mesh id set.
pub(crate) type MeshGeometryMaps = (
    std::collections::HashMap<AssetId, LoadedMesh>,
    std::collections::HashMap<AssetId, MeshSourceMeta>,
    std::collections::HashSet<AssetId>,
);

// Output of `build_draw_list`: shared vertex/index buffers, draw objects,
// GPU-instanced clusters, the per-prop draw-index table, and the mesh-id to
// draw-slot map for hot-reload.
pub(crate) type DrawListData = (
    Vec<Vertex>,
    Vec<u32>,
    Vec<DrawObject>,
    Vec<InstancedCluster>,
    Vec<Vec<usize>>,
    std::collections::HashMap<AssetId, Vec<usize>>,
);

// One appended mesh's placement in the shared buffers: vertex_offset,
// vertex_count, index_offset, index_count, LOD slices, and local AABB min/max.
type AppendedMesh = (
    usize,
    usize,
    usize,
    usize,
    Vec<LodSlice>,
    [f32; 3],
    [f32; 3],
);

// Sentinel AABB used when a draw object opts out of culling (e.g. unbounded
// skybox geometry). Both metal and vulkan/directx backends should treat any
// non-finite component as "always draw".
const UNCULLED_BB: ([f32; 3], [f32; 3]) = (
    [f32::NAN, f32::NAN, f32::NAN],
    [f32::NAN, f32::NAN, f32::NAN],
);

fn local_bounds(verts: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    if verts.is_empty() {
        return UNCULLED_BB;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for v in verts {
        for i in 0..3 {
            mn[i] = mn[i].min(v.pos[i]);
            mx[i] = mx[i].max(v.pos[i]);
        }
    }
    (mn, mx)
}

// Props whose transform can change at runtime are pulled out of the BVH and
// always drawn (after a per-object frustum test). The BVH is built once at
// init and does not refit, so anything that moves would otherwise risk being
// culled incorrectly: its model matrix updates but its init-time AABB
// remains, so the wrong region of space is tested against the frustum.
// `pickup` and `interactable` move via Camera3DSystem; `parent.is_some()` may
// inherit a parent transform that changes; `collider.is_some()` rides a
// PhysicsSystem rigid body that may translate / rotate every step (and even
// a "static" collider is cheap to mark dynamic; the per-object frustum test
// is O(1) and the BVH would not have refit it either way).
fn is_dynamic_prop(prop: &Prop) -> bool {
    prop.pickup || prop.interactable || prop.parent.is_some() || prop.collider.is_some()
}

// Column-major 4×4 matrix multiply: result = a * b; layout m[col][row].
fn mat_mul4(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for col in 0..4 {
        for row in 0..4 {
            for k in 0..4 {
                out[col][row] += a[k][row] * b[col][k];
            }
        }
    }
    out
}

// Resolve world matrices for a flat list of Props that may form a parent-child
// hierarchy. `parents[i]` is the index of prop i's parent, or None for roots.
//
// Iterates until all props are resolved or no further progress is possible
// (which indicates a cycle; cyclic props fall back to their local matrix).
pub fn compute_world_matrices(props: &[&Prop], parents: &[Option<usize>]) -> Vec<[[f32; 4]; 4]> {
    let n = props.len();
    let mut world: Vec<Option<[[f32; 4]; 4]>> = vec![None; n];

    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            if world[i].is_some() {
                continue;
            }
            let local = props[i].model_matrix();
            let w = match parents.get(i).copied().flatten() {
                None => local,
                Some(pi) => match world.get(pi).copied().flatten() {
                    Some(pw) => mat_mul4(pw, local),
                    None => continue, // parent not yet resolved
                },
            };
            world[i] = Some(w);
            changed = true;
        }
    }

    // Any unresolved props (from cycles) fall back to their local matrix.
    world
        .into_iter()
        .enumerate()
        .map(|(i, w)| w.unwrap_or_else(|| props[i].model_matrix()))
        .collect()
}

// Resolve and write each entity's GlobalTransform from its Transform and parent
// chain. The entity-keyed analogue of compute_world_matrices: roots use their
// local matrix, children compose parent-world * local, and cyclic parents fall
// back to their local matrix. Reads Transform and Parent, writes GlobalTransform
// (seeded on every transform entity at init).
pub(crate) fn propagate_transforms(ctx: &mut crate::ecs::PipelineContext) {
    use crate::assets::{GlobalTransform, Parent, Transform};
    use crate::ecs::Entity;
    use std::collections::HashMap;

    let parents: HashMap<Entity, Entity> = ctx
        .query_with_entity::<Parent>()
        .map(|(entity, parent)| (entity, parent.0))
        .collect();
    let locals: Vec<(Entity, [[f32; 4]; 4])> = ctx
        .query_with_entity::<Transform>()
        .map(|(entity, transform)| (entity, transform.model_matrix()))
        .collect();

    // Fixed-point resolution: keep a pass running while any entity newly
    // resolves; stop on a pass with no progress (a cycle) or once all are done.
    let mut world: HashMap<Entity, [[f32; 4]; 4]> = HashMap::with_capacity(locals.len());
    loop {
        let mut progressed = false;
        for (entity, local) in &locals {
            if world.contains_key(entity) {
                continue;
            }
            let resolved = match parents.get(entity) {
                None => Some(*local),
                Some(parent) => world.get(parent).map(|pw| mat_mul4(*pw, *local)),
            };
            if let Some(matrix) = resolved {
                world.insert(*entity, matrix);
                progressed = true;
            }
        }
        if !progressed || world.len() == locals.len() {
            break;
        }
    }
    // Cyclic entities fall back to their local matrix.
    for (entity, local) in &locals {
        world.entry(*entity).or_insert(*local);
    }

    for (entity, matrix) in world {
        if let Some(global) = ctx.get_mut::<GlobalTransform>(entity) {
            global.0 = matrix;
        }
    }
}

// Decoded mesh geometry plus its optional LOD trailer. Returned by
// `load_mesh_geometry` and consumed by `build_draw_list`. The `vertices`
// slice is shared across LOD0 and every alternate; vertex-clustering
// decimation reuses the original vertex set and only generates new index
// lists. Empty `lod_alternates` means the mesh declared `lod_levels <= 1`
// (or the build dropped degenerate decimations); the runtime then keeps
// the single LOD0 slice.
pub(crate) struct LoadedMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,
    pub lod_alternates: Vec<(f32, Vec<u16>)>,
}

// Hot-reload source metadata for a file-backed `Mesh`. Captured by
// `load_mesh_geometry` before the Mesh is drained and consumed; the
// `(asset_id, source, primitive_index, lod_levels, lod_distances)` tuple is
// later cross-referenced against `build_draw_list`'s mesh_id → draw_indices
// map to build the runtime
// [`MeshSourceMap`](crate::gfx::graphics_system::hot_reload_sources::MeshSourceMap).
pub(crate) struct MeshSourceMeta {
    pub source: String,
    pub primitive_index: u32,
    pub lod_levels: u32,
    pub lod_distances: Vec<f32>,
}

// Decode all Mesh, ProceduralMesh, and mesh-kind File payloads into a
// name-keyed geometry table. Returns None if any payload is missing or
// malformed. Also returns a per-asset-id source-meta map for file-backed Mesh
// declarations under `cn debug`, used by the hot-reload watcher to know what
// to re-import, and the set of mesh ids whose props must always stay resident
// (skybox-class geometry that encloses the camera).
pub(crate) fn load_mesh_geometry(ctx: &mut PipelineContext) -> Option<MeshGeometryMaps> {
    let raw_meshes = ctx.drain::<Mesh>();
    // ProceduralMesh components are cloned rather than drained: PhysicsSystem
    // inits after GraphicsSystem and resolves its `terrain_mesh` reference by
    // querying ProceduralMesh for the live heightfield args. Same precedent as
    // [`crate::assets::audio_clip::audio_clip_blob_indices`]: leave the
    // component in place so a later init step can still read it.
    let proc_meshes: Vec<ProceduralMesh> = ctx.query::<ProceduralMesh>().cloned().collect();
    let voxel_chunks = ctx.drain::<VoxelChunk>();
    let file_assets = ctx.drain::<File>();
    let file_meshes: Vec<&File> = file_assets
        .iter()
        .filter(|f| f.kind.as_ref().map(FileKind::is_mesh).unwrap_or(false))
        .collect();

    // Skybox-generated meshes enclose the camera, so any prop using one must
    // opt out of frustum culling AND streaming residency (per the
    // StreamingConfig docstring's "skybox always stays resident" promise),
    // collected here while the ProceduralMesh args are in scope.
    let always_resident_meshes: std::collections::HashSet<AssetId> = proc_meshes
        .iter()
        .filter(|pm| pm.generator == "skybox")
        .map(|pm| pm.asset_id)
        .collect();

    if raw_meshes.is_empty()
        && proc_meshes.is_empty()
        && voxel_chunks.is_empty()
        && file_meshes.is_empty()
    {
        // Room path can carry the scene without any explicit Mesh/ProceduralMesh
        tracing::warn!(
            "GraphicsSystem: no Mesh, ProceduralMesh, VoxelChunk, or mesh-kind File components found"
        );
    }

    let capture_sources = crate::app::dev_flags::enabled();
    let mut mesh_sources: std::collections::HashMap<AssetId, MeshSourceMeta> =
        std::collections::HashMap::new();
    if capture_sources {
        for m in &raw_meshes {
            if !m.source.is_empty() {
                mesh_sources.insert(
                    m.asset_id,
                    MeshSourceMeta {
                        source: m.source.clone(),
                        primitive_index: m.primitive_index,
                        lod_levels: m.lod_levels,
                        lod_distances: m.lod_distances.clone(),
                    },
                );
            }
        }
    }

    let mut geometry: std::collections::HashMap<AssetId, LoadedMesh> =
        std::collections::HashMap::new();

    macro_rules! load_meshes {
        ($label:expr_2021, $items:expr_2021) => {
            for (i, mesh) in $items.iter().enumerate() {
                let locator = match &mesh.locator {
                    Some(l) => l,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: {}[{}] {} has no compiled payload",
                            $label,
                            i,
                            mesh.asset_id
                        );
                        return None;
                    }
                };
                let bytes = match ctx.read_payload(locator) {
                    Ok(b) => b.to_vec(),
                    Err(e) => {
                        tracing::error!(
                            "GraphicsSystem: failed to read {} payload: {:?}",
                            $label,
                            e
                        );
                        return None;
                    }
                };
                // `deserialise_with_lods` parses the optional LOD trailer
                // when the build emitted one and falls back to an empty
                // alternates vec for legacy single-LOD payloads.
                match crate::gfx::mesh_payload::deserialise_with_lods(&bytes) {
                    Ok((verts, idxs, alternates)) => {
                        geometry.insert(
                            mesh.asset_id,
                            LoadedMesh {
                                vertices: verts,
                                indices: idxs,
                                lod_alternates: alternates,
                            },
                        );
                    }
                    Err(e) => {
                        tracing::error!("GraphicsSystem: malformed {} payload: {}", $label, e);
                        return None;
                    }
                }
            }
        };
    }
    load_meshes!("Mesh", raw_meshes);
    load_meshes!("ProceduralMesh", proc_meshes);
    load_meshes!("VoxelChunk", voxel_chunks);
    load_meshes!("File", file_meshes);

    Some((geometry, mesh_sources, always_resident_meshes))
}

// Decode all Room mesh payloads and collect blob indices for the release step.
// Returns None if any payload is missing or malformed (error already logged).
pub(crate) fn load_room_geometry(
    ctx: &mut PipelineContext,
) -> Option<(Vec<RoomGeometry>, Vec<u32>)> {
    let rooms = ctx.drain::<Room>();
    let mut room_geometry: Vec<RoomGeometry> = Vec::new();
    let mut blob_indices: Vec<u32> = Vec::new();

    for (i, room) in rooms.into_iter().enumerate() {
        let locator = match &room.locator {
            Some(l) => l.clone(),
            None => {
                tracing::error!(
                    "GraphicsSystem: Room[{}] {} has no compiled payload -- did the build succeed?",
                    i,
                    room.asset_id
                );
                return None;
            }
        };
        blob_indices.push(locator.blob_index);
        let bytes = match ctx.read_payload(&locator) {
            Ok(b) => b.to_vec(),
            Err(e) => {
                tracing::error!(
                    "GraphicsSystem: failed to read Room {} payload: {:?}",
                    room.asset_id,
                    e
                );
                return None;
            }
        };
        match crate::gfx::mesh_payload::deserialise_with_lods(&bytes) {
            Ok((verts, idxs, alternates)) => room_geometry.push((room, verts, idxs, alternates)),
            Err(e) => {
                tracing::error!("GraphicsSystem: malformed Room payload: {}", e);
                return None;
            }
        }
    }

    Some((room_geometry, blob_indices))
}

// Assemble the shared vertex/index buffers and per-object draw records from all
// scene geometry (props, unreferenced meshes, rooms). Also returns the per-prop
// draw-index table for runtime model-matrix updates and the GPU-instanced
// cluster list (one entry per InstancedProp).
// Returns None if any referenced asset is missing (error already logged).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_draw_list(
    props: &[&Prop],
    instanced_props: &[InstancedProp],
    world_mats: &[[[f32; 4]; 4]],
    model_map: &std::collections::HashMap<AssetId, Vec<SubMeshRef>>,
    mesh_geometry: &std::collections::HashMap<AssetId, LoadedMesh>,
    room_geometry: &[RoomGeometry],
    texture_name_to_slot: &std::collections::HashMap<AssetId, usize>,
    material_map: &std::collections::HashMap<AssetId, MaterialEntry>,
    always_resident_meshes: &std::collections::HashSet<AssetId>,
) -> Option<DrawListData> {
    let mut all_vertices: Vec<Vertex> = Vec::new();
    let mut all_indices: Vec<u32> = Vec::new();
    let mut draw_objects: Vec<DrawObject> = Vec::new();
    let mut instanced_clusters: Vec<InstancedCluster> = Vec::new();
    let mut prop_draw_indices: Vec<Vec<usize>> = Vec::new();
    // Map every Mesh / ProceduralMesh / etc. asset to the draw slots that
    // received a copy of its geometry. Hot-reload (`cn debug` only) walks this
    // to know which slots to overwrite when the source `.glb` changes. One
    // entry per Mesh asset; the `Vec<usize>` accumulates every push since a
    // Mesh shared by N `Prop`s yields N independent draw objects.
    let mut mesh_id_to_draws: std::collections::HashMap<AssetId, Vec<usize>> =
        std::collections::HashMap::new();

    // track explicitly referenced mesh ids so unreferenced ones get auto-rendered
    let mut referenced: std::collections::HashSet<AssetId> = std::collections::HashSet::new();
    for prop in props {
        if let Some(mesh_id) = prop.mesh {
            referenced.insert(mesh_id);
        }
        if let Some(model_id) = prop.model
            && let Some(submeshes) = model_map.get(&model_id)
        {
            for sub in submeshes {
                if let Some(sub_mesh) = sub.mesh {
                    referenced.insert(sub_mesh);
                }
            }
        }
    }
    for inst in instanced_props {
        if let Some(mesh_id) = inst.mesh {
            referenced.insert(mesh_id);
        }
    }

    // append_mesh: add a mesh into the shared buffers by id, return
    // (vertex_offset, vertex_count, index_offset, index_count, lod_slices,
    // local_bb_min, local_bb_max). `lod_slices` is empty for legacy
    // single-LOD meshes; otherwise each entry is a `LodSlice` pointing at the
    // alternate's rebased indices in `all_indices`, paired with its switch
    // distance. Every LOD alternate reuses the same `vertex_offset` /
    // `vertex_count` since clustering decimation does not modify the vertex
    // set.
    let mut append_mesh = |id: AssetId| -> Option<AppendedMesh> {
        let loaded = mesh_geometry.get(&id)?;
        let vertex_byte_offset = all_vertices.len() * std::mem::size_of::<Vertex>();
        let index_elem_offset = all_indices.len();
        let base = all_vertices.len() as u32;
        let (bb_min, bb_max) = local_bounds(&loaded.vertices);
        all_vertices.extend_from_slice(&loaded.vertices);
        all_indices.extend(loaded.indices.iter().map(|i| u32::from(*i) + base));
        let mut lod_slices: Vec<LodSlice> = Vec::with_capacity(loaded.lod_alternates.len());
        for (switch_distance, alt_idx) in &loaded.lod_alternates {
            let alt_offset = all_indices.len();
            all_indices.extend(alt_idx.iter().map(|i| u32::from(*i) + base));
            lod_slices.push(LodSlice {
                index_offset: alt_offset,
                index_count: alt_idx.len(),
                switch_distance: *switch_distance,
            });
        }
        Some((
            vertex_byte_offset,
            loaded.vertices.len(),
            index_elem_offset,
            loaded.indices.len(),
            lod_slices,
            bb_min,
            bb_max,
        ))
    };

    for (prop_idx, prop) in props.iter().enumerate() {
        let model_mat = world_mats[prop_idx];
        let mut prop_idxs: Vec<usize> = Vec::new();

        if let Some(model_id) = prop.model {
            // multi-mesh model path: one draw object per sub-mesh
            let submeshes = match model_map.get(&model_id) {
                Some(s) => s,
                None => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} references unknown model {} -- add a Model asset with that id",
                        prop.asset_id,
                        model_id
                    );
                    return None;
                }
            };
            for sub in submeshes {
                let sub_mesh_id = match sub.mesh {
                    Some(m) => m,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Model {} has a sub-mesh with no mesh",
                            model_id
                        );
                        return None;
                    }
                };
                let (
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    lod_alternates,
                    local_min,
                    local_max,
                ) = match append_mesh(sub_mesh_id) {
                    Some(t) => t,
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Model {} sub-mesh {} not found -- add a Mesh or ProceduralMesh asset with that id",
                            model_id,
                            sub_mesh_id
                        );
                        return None;
                    }
                };
                let (texture_slot, normal_map_slot, material) = match sub.material {
                    Some(mat_id) => match material_map.get(&mat_id) {
                        Some(&(slot, nms, u)) => (slot, nms, u),
                        None => {
                            tracing::error!(
                                "GraphicsSystem: Model {} sub-mesh material {} not found",
                                model_id,
                                mat_id
                            );
                            return None;
                        }
                    },
                    None => (0, 0, MaterialUniforms::DEFAULT),
                };
                let (bb_min, bb_max) =
                    if is_dynamic_prop(prop) || always_resident_meshes.contains(&sub_mesh_id) {
                        UNCULLED_BB
                    } else {
                        crate::gfx::frustum::transform_aabb(local_min, local_max, model_mat)
                    };
                prop_idxs.push(draw_objects.len());
                mesh_id_to_draws
                    .entry(sub_mesh_id)
                    .or_default()
                    .push(draw_objects.len());
                draw_objects.push(DrawObject {
                    vertex_offset,
                    vertex_count,
                    index_offset,
                    index_count,
                    // Static geometry: indices are absolute into the shared
                    // vertex buffer, so no per-draw base.
                    base_vertex: 0,
                    model: model_mat,
                    texture_slot,
                    normal_map_slot,
                    material,
                    visible: true,
                    resident: true,
                    bb_min,
                    bb_max,
                    cull_distance: prop.cull_distance,
                    lod_alternates,
                });
            }
        } else {
            // single-mesh path
            let mesh_id = match prop.mesh {
                Some(m) => m,
                None => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} has neither a model nor a mesh",
                        prop.asset_id
                    );
                    return None;
                }
            };
            let (
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                lod_alternates,
                local_min,
                local_max,
            ) = match append_mesh(mesh_id) {
                Some(t) => t,
                None => {
                    tracing::error!(
                        "GraphicsSystem: Prop {} references unknown mesh {} -- add a Mesh or ProceduralMesh asset with that id",
                        prop.asset_id,
                        mesh_id
                    );
                    return None;
                }
            };
            let (texture_slot, normal_map_slot, material) = if let Some(mat_id) = prop.material {
                match material_map.get(&mat_id) {
                    Some(&(slot, nms, uniforms)) => (slot, nms, uniforms),
                    None => {
                        tracing::error!(
                            "GraphicsSystem: Prop {} references unknown material {} -- add a Material asset with that id",
                            prop.asset_id,
                            mat_id
                        );
                        return None;
                    }
                }
            } else if let Some(tex_id) = prop.texture {
                let slot = *texture_name_to_slot.get(&tex_id).unwrap_or(&0);
                (slot, 0, MaterialUniforms::DEFAULT)
            } else {
                (0, 0, MaterialUniforms::DEFAULT)
            };
            let (bb_min, bb_max) =
                if is_dynamic_prop(prop) || always_resident_meshes.contains(&mesh_id) {
                    UNCULLED_BB
                } else {
                    crate::gfx::frustum::transform_aabb(local_min, local_max, model_mat)
                };
            prop_idxs.push(draw_objects.len());
            mesh_id_to_draws
                .entry(mesh_id)
                .or_default()
                .push(draw_objects.len());
            draw_objects.push(DrawObject {
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                base_vertex: 0,
                model: model_mat,
                texture_slot,
                normal_map_slot,
                material,
                visible: true,
                resident: true,
                bb_min,
                bb_max,
                cull_distance: prop.cull_distance,
                lod_alternates,
            });
        }

        prop_draw_indices.push(prop_idxs);
    }

    // InstancedProp -> one GPU-instanced cluster per InstancedProp.
    // The cluster mesh is appended once; per-instance model matrices are
    // resolved up front and uploaded to the GPU each frame. The cluster's
    // union AABB is used as a single frustum-cull test for the whole batch.
    for inst in instanced_props {
        let mesh_id = match inst.mesh {
            Some(m) if !inst.instances.is_empty() => m,
            _ => continue,
        };
        // Instanced clusters carry the mesh's LOD alternates and bucket
        // their per-instance matrices by camera distance at draw time;
        // see [`InstancedCluster::lod_buckets`].
        let (
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            lod_alternates,
            local_min,
            local_max,
        ) = match append_mesh(mesh_id) {
            Some(t) => t,
            None => {
                tracing::error!(
                    "GraphicsSystem: InstancedProp {} references unknown mesh {}",
                    inst.asset_id,
                    mesh_id
                );
                return None;
            }
        };
        let (texture_slot, normal_map_slot, material) = if let Some(mat_id) = inst.material {
            match material_map.get(&mat_id) {
                Some(&(slot, nms, uniforms)) => (slot, nms, uniforms),
                None => {
                    tracing::error!(
                        "GraphicsSystem: InstancedProp {} references unknown material {}",
                        inst.asset_id,
                        mat_id
                    );
                    return None;
                }
            }
        } else if let Some(tex_id) = inst.texture {
            let slot = *texture_name_to_slot.get(&tex_id).unwrap_or(&0);
            (slot, 0, MaterialUniforms::DEFAULT)
        } else {
            (0, 0, MaterialUniforms::DEFAULT)
        };

        let mut instance_mats: Vec<[[f32; 4]; 4]> = Vec::with_capacity(inst.instances.len());
        let mut cluster_min = [f32::INFINITY; 3];
        let mut cluster_max = [f32::NEG_INFINITY; 3];
        for i in 0..inst.instances.len() {
            let Some(model_mat) = inst.instance_model_matrix(i) else {
                continue;
            };
            let (bb_min, bb_max) =
                crate::gfx::frustum::transform_aabb(local_min, local_max, model_mat);
            for k in 0..3 {
                cluster_min[k] = cluster_min[k].min(bb_min[k]);
                cluster_max[k] = cluster_max[k].max(bb_max[k]);
            }
            instance_mats.push(model_mat);
        }
        if instance_mats.is_empty() {
            continue;
        }

        instanced_clusters.push(InstancedCluster {
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            texture_slot,
            normal_map_slot,
            material,
            cluster_bb_min: cluster_min,
            cluster_bb_max: cluster_max,
            local_bb_min: local_min,
            local_bb_max: local_max,
            cull_distance: inst.cull_distance,
            instances: instance_mats,
            lod_alternates,
        });
    }

    // unreferenced meshes (e.g. a standalone room): identity model matrix, slot 0.
    // These are drawn unconditionally; culling is disabled via the sentinel AABB.
    for mesh_id in mesh_geometry.keys().copied().collect::<Vec<_>>() {
        if referenced.contains(&mesh_id) {
            continue;
        }
        if let Some((
            vertex_offset,
            vertex_count,
            index_offset,
            index_count,
            lod_alternates,
            _,
            _,
        )) = append_mesh(mesh_id)
        {
            // Auto-rendered unreferenced meshes (e.g. a standalone room mesh)
            // are non-cullable, so distance-keyed LOD swaps make no sense
            // here. Drop any alternates the build emitted; the LOD0 draw is
            // the only one that will fire.
            let _ = lod_alternates;
            mesh_id_to_draws
                .entry(mesh_id)
                .or_default()
                .push(draw_objects.len());
            draw_objects.push(DrawObject {
                vertex_offset,
                vertex_count,
                index_offset,
                index_count,
                base_vertex: 0,
                model: IDENTITY4,
                texture_slot: 0,
                normal_map_slot: 0,
                material: MaterialUniforms::DEFAULT,
                visible: true,
                resident: true,
                bb_min: UNCULLED_BB.0,
                bb_max: UNCULLED_BB.1,
                cull_distance: 0.0,
                lod_alternates: Vec::new(),
            });
        }
    }

    // Room components placed at the world origin with optional texture.
    // Rooms also opt out of culling (they enclose the camera). LOD picks
    // come from camera-to-origin distance per [`crate::gfx::lod::camera_distance`]'s
    // sentinel-AABB fallback, so practical swaps only fire if the camera
    // wanders far from the world origin.
    for (room, verts, idxs, room_lods) in room_geometry {
        let vertex_byte_offset = all_vertices.len() * std::mem::size_of::<Vertex>();
        let index_elem_offset = all_indices.len();
        let base = all_vertices.len() as u32;
        all_vertices.extend_from_slice(verts);
        all_indices.extend(idxs.iter().map(|i| u32::from(*i) + base));
        let mut lod_slices: Vec<LodSlice> = Vec::with_capacity(room_lods.len());
        for (switch_distance, alt_idx) in room_lods {
            let alt_offset = all_indices.len();
            all_indices.extend(alt_idx.iter().map(|i| u32::from(*i) + base));
            lod_slices.push(LodSlice {
                index_offset: alt_offset,
                index_count: alt_idx.len(),
                switch_distance: *switch_distance,
            });
        }
        let texture_slot = match room.effective_texture() {
            None => 0,
            Some(id) => *texture_name_to_slot.get(&id).unwrap_or(&0),
        };
        draw_objects.push(DrawObject {
            vertex_offset: vertex_byte_offset,
            vertex_count: verts.len(),
            index_offset: index_elem_offset,
            index_count: idxs.len(),
            base_vertex: 0,
            model: IDENTITY4,
            texture_slot,
            normal_map_slot: 0,
            material: MaterialUniforms::DEFAULT,
            visible: true,
            resident: true,
            bb_min: UNCULLED_BB.0,
            bb_max: UNCULLED_BB.1,
            cull_distance: 0.0,
            lod_alternates: lod_slices,
        });
    }

    Some((
        all_vertices,
        all_indices,
        draw_objects,
        instanced_clusters,
        prop_draw_indices,
        mesh_id_to_draws,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::Prop;

    fn make_prop(position: [f32; 3]) -> Prop {
        Prop {
            asset_id: AssetId::default(),
            model: None,
            mesh: None,
            material: None,
            texture: None,
            position,
            rotation_deg: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
            collider: None,
            interactable: false,
            pickup: false,
            parent: None,
            scene: None,
            prefab: String::new(),
            cull_distance: 0.0,
            is_held: false,
        }
    }

    fn prop_with_transform(t: &crate::assets::Transform) -> Prop {
        Prop {
            position: t.position,
            rotation_deg: t.rotation_deg,
            scale: t.scale,
            ..make_prop([0.0; 3])
        }
    }

    // The decomposed transform path must produce matrices bit-identical to the
    // legacy positional path, so the per-frame model-matrix push is unchanged.
    #[test]
    fn propagate_transforms_matches_compute_world_matrices() {
        use crate::assets::{GlobalTransform, Parent, Transform};
        use crate::blob::BlobData;
        use crate::ecs::{ComponentStorage, PipelineContext, Resources};
        use crate::gfx::profile::FrameProfile;

        let parent_t = Transform {
            position: [1.0, 2.0, 3.0],
            rotation_deg: [0.0, 30.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        };
        let child_t = Transform {
            position: [0.0, 0.0, 1.0],
            rotation_deg: [10.0, 0.0, 5.0],
            scale: [2.0, 2.0, 2.0],
        };

        let mut components = ComponentStorage::default();
        let mut blob = BlobData::empty();
        let mut profile = FrameProfile::default();
        let mut resources = Resources::new();
        let mut ctx = PipelineContext {
            components: &mut components,
            blob: &mut blob,
            profile: &mut profile,
            resources: &mut resources,
        };

        // A child parented to a root, each with its own GlobalTransform to write.
        let parent_e = ctx.components.spawn();
        ctx.insert(parent_e, parent_t);
        ctx.insert(parent_e, GlobalTransform::default());
        let child_e = ctx.components.spawn();
        ctx.insert(child_e, child_t);
        ctx.insert(child_e, Parent(parent_e));
        ctx.insert(child_e, GlobalTransform::default());

        propagate_transforms(&mut ctx);

        // Reference: the legacy positional path over the same two transforms,
        // child (index 1) parented to parent (index 0).
        let parent_prop = prop_with_transform(&parent_t);
        let child_prop = prop_with_transform(&child_t);
        let refs = vec![&parent_prop, &child_prop];
        let ref_mats = compute_world_matrices(&refs, &[None, Some(0)]);

        let parent_g = ctx.components.get::<GlobalTransform>(parent_e).unwrap().0;
        let child_g = ctx.components.get::<GlobalTransform>(child_e).unwrap().0;
        assert_eq!(parent_g, ref_mats[0], "root GlobalTransform matches legacy");
        assert_eq!(child_g, ref_mats[1], "child GlobalTransform matches legacy");
    }

    #[test]
    fn is_dynamic_prop_flags_collider_bearing_props() {
        // A plain static prop stays in the BVH.
        let plain = make_prop([0.0; 3]);
        assert!(!is_dynamic_prop(&plain));

        // A prop with a collider rides a physics body that may translate /
        // rotate every step; the init-time BVH AABB would stale out, so it
        // must opt out of the BVH and pay the per-frame frustum test instead.
        // Reproduces the "physics ball disappears when the camera gets close"
        // bug: the ball rolled past its init AABB, so the BVH culled it even
        // when it was right in front of the camera.
        let mut with_collider = make_prop([0.0; 3]);
        with_collider.collider = Some(crate::assets::PropCollider::default());
        assert!(is_dynamic_prop(&with_collider));

        // The pre-existing dynamic flags (pickup / interactable / parent)
        // still mark a prop dynamic on their own.
        let mut pickup = make_prop([0.0; 3]);
        pickup.pickup = true;
        assert!(is_dynamic_prop(&pickup));

        let mut interactable = make_prop([0.0; 3]);
        interactable.interactable = true;
        assert!(is_dynamic_prop(&interactable));

        let mut parented = make_prop([0.0; 3]);
        parented.parent = Some(AssetId(1));
        assert!(is_dynamic_prop(&parented));
    }

    #[test]
    fn mat_mul4_identity_is_no_op() {
        let m = [
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
            [10.0, 11.0, 12.0, 1.0],
        ];
        assert_eq!(mat_mul4(m, IDENTITY4), m);
        assert_eq!(mat_mul4(IDENTITY4, m), m);
    }

    #[test]
    fn mat_mul4_translations_compose() {
        // T(1,0,0) * T(0,1,0) should give combined translation (1,1,0).
        // Column-major: translation is in col 3.
        let tx = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ];
        let ty = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ];
        let result = mat_mul4(tx, ty);
        assert_eq!(result[3], [1.0, 1.0, 0.0, 1.0]);
        assert_eq!(result[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(result[1], [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn compute_world_matrices_flat_list() {
        let a = make_prop([1.0, 0.0, 0.0]);
        let b = make_prop([0.0, 2.0, 0.0]);
        let props: Vec<&Prop> = vec![&a, &b];
        let mats = compute_world_matrices(&props, &[None, None]);
        assert_eq!(mats[0], a.model_matrix());
        assert_eq!(mats[1], b.model_matrix());
    }

    #[test]
    fn compute_world_matrices_child_inherits_parent_translation() {
        let parent = make_prop([1.0, 0.0, 0.0]);
        let child = make_prop([0.0, 1.0, 0.0]);
        let props: Vec<&Prop> = vec![&parent, &child];
        let mats = compute_world_matrices(&props, &[None, Some(0)]);
        // parent unchanged
        assert_eq!(mats[0], parent.model_matrix());
        // child world translation = parent_pos + child_pos = (1,1,0)
        // col 3 carries translation in column-major layout
        assert!((mats[1][3][0] - 1.0).abs() < 1e-5);
        assert!((mats[1][3][1] - 1.0).abs() < 1e-5);
        assert!((mats[1][3][2]).abs() < 1e-5);
    }

    #[test]
    fn compute_world_matrices_cycle_falls_back_to_local() {
        let a = make_prop([1.0, 0.0, 0.0]);
        let b = make_prop([0.0, 1.0, 0.0]);
        let props: Vec<&Prop> = vec![&a, &b];
        // Mutual cycle: a's parent is b, b's parent is a; neither can resolve.
        let mats = compute_world_matrices(&props, &[Some(1), Some(0)]);
        assert_eq!(mats[0], a.model_matrix());
        assert_eq!(mats[1], b.model_matrix());
    }

    fn unit_quad_mesh() -> LoadedMesh {
        // Axis-aligned unit cube centred at origin; bounds = [-0.5, 0.5]^3.
        let mk = |x, y, z| Vertex {
            pos: [x, y, z],
            normal: [0.0, 1.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0],
            uv: [0.0, 0.0],
        };
        let v = vec![
            mk(-0.5, -0.5, -0.5),
            mk(0.5, -0.5, -0.5),
            mk(0.5, 0.5, -0.5),
            mk(-0.5, 0.5, -0.5),
        ];
        let i = vec![0u16, 1, 2, 0, 2, 3];
        LoadedMesh {
            vertices: v,
            indices: i,
            lod_alternates: Vec::new(),
        }
    }

    #[test]
    fn build_draw_list_emits_one_cluster_for_instanced_prop() {
        let mut mesh_geometry = std::collections::HashMap::new();
        mesh_geometry.insert(AssetId(0), unit_quad_mesh());

        let inst = crate::assets::InstancedProp {
            asset_id: AssetId::default(),
            mesh: Some(AssetId(0)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            instances: vec![
                crate::assets::instanced_prop::InstanceTransform {
                    position: [0.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
                crate::assets::instanced_prop::InstanceTransform {
                    position: [5.0, 0.0, 0.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
                crate::assets::instanced_prop::InstanceTransform {
                    position: [-3.0, 0.0, 2.0],
                    rotation_deg: [0.0; 3],
                    scale: [1.0; 3],
                },
            ],
        };

        let (verts, idxs, draw_objects, clusters, _prop_idxs, mesh_id_to_draws) = build_draw_list(
            &[],
            &[inst],
            &[],
            &std::collections::HashMap::new(),
            &mesh_geometry,
            &[],
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("build_draw_list");

        // Cluster mesh appended exactly once into the shared buffers.
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs.len(), 6);
        // InstancedProp meshes go into clusters, not draw_objects; the
        // hot-reload map (which only tracks draw_objects-backed pushes) stays
        // empty for this scene.
        assert!(mesh_id_to_draws.is_empty());

        // Each instance no longer emits its own DrawObject; the cluster carries
        // every transform.
        assert!(draw_objects.is_empty());
        assert_eq!(clusters.len(), 1);
        let c = &clusters[0];
        assert_eq!(c.instances.len(), 3);
        assert_eq!(c.index_count, 6);

        // Union AABB over all per-instance world AABBs. The unit_quad_mesh
        // is planar at z=-0.5, so each instance contributes a flat slab in z;
        // x and y span the quad extent [-0.5, 0.5].
        assert!((c.cluster_bb_min[0] - (-3.5)).abs() < 1e-5);
        assert!((c.cluster_bb_max[0] - 5.5).abs() < 1e-5);
        assert!((c.cluster_bb_min[1] - (-0.5)).abs() < 1e-5);
        assert!((c.cluster_bb_max[1] - 0.5).abs() < 1e-5);
        // z: instances at z=0 give [-0.5,-0.5]; instance at z=2 gives [1.5,1.5];
        // union is [-0.5, 1.5].
        assert!((c.cluster_bb_min[2] - (-0.5)).abs() < 1e-5);
        assert!((c.cluster_bb_max[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn build_draw_list_skips_empty_instanced_prop() {
        let mut mesh_geometry = std::collections::HashMap::new();
        mesh_geometry.insert(AssetId(0), unit_quad_mesh());

        let inst = crate::assets::InstancedProp {
            asset_id: AssetId::default(),
            mesh: Some(AssetId(0)),
            material: None,
            texture: None,
            cull_distance: 0.0,
            instances: Vec::new(),
        };

        let (_verts, _idxs, draw_objects, clusters, _prop_idxs, _mesh_id_to_draws) =
            build_draw_list(
                &[],
                &[inst],
                &[],
                &std::collections::HashMap::new(),
                &mesh_geometry,
                &[],
                &std::collections::HashMap::new(),
                &std::collections::HashMap::new(),
                &std::collections::HashSet::new(),
            )
            .expect("build_draw_list");

        assert!(draw_objects.is_empty());
        assert!(clusters.is_empty());
    }

    #[test]
    fn always_resident_mesh_forces_uncullable_bb_on_static_prop() {
        // A static prop with no dynamic flags would normally get a finite AABB
        // and be picked up by the streamer's `obj.cullable()` selection. When
        // its mesh is in the always_resident_meshes set (e.g. the auto-generated
        // skybox), the bb is forced to NaN so the prop opts out of frustum
        // culling and of mesh streaming. This is what the StreamingConfig
        // docstring promises for the skybox.
        let mut mesh_geometry = std::collections::HashMap::new();
        mesh_geometry.insert(AssetId(0), unit_quad_mesh());

        let mut prop = make_prop([0.0; 3]);
        prop.mesh = Some(AssetId(0));
        let props_refs: Vec<&Prop> = vec![&prop];
        let world_mats = compute_world_matrices(&props_refs, &[None]);

        let mut always_resident = std::collections::HashSet::new();
        always_resident.insert(AssetId(0));

        let (_v, _i, draw_objects, _c, _p, _m) = build_draw_list(
            &props_refs,
            &[],
            &world_mats,
            &std::collections::HashMap::new(),
            &mesh_geometry,
            &[],
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
            &always_resident,
        )
        .expect("build_draw_list");

        assert_eq!(draw_objects.len(), 1);
        // UNCULLED_BB is all-NaN; `cullable()` returns false in that case.
        assert!(draw_objects[0].bb_min[0].is_nan());
        assert!(draw_objects[0].bb_max[0].is_nan());
        assert!(!draw_objects[0].cullable());
    }
}
