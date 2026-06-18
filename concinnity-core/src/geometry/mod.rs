// src/geometry.rs
//
// Mesh geometry generators and the build-time compile functions that invoke
// them. The two public entry points are compile_mesh_payload() and
// compile_room_payload(); everything else is private to this module tree.
//
// Adding a new generator
// 1. Add a branch to the match in compile_mesh_payload().
// 2. Add a pub(super) build_* function in the appropriate submodule.
// 3. No other files need to change.

// Procedural voxel-chunk generation. Consumed only by the Metal backend's
// chunk-streaming path for now, so the module is compiled but
// unreferenced on non-macOS builds.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
mod chunk_gen;
mod extrude;
pub mod glass_quad;
// The heightfield generator builds its grid from a source image the build crate
// decodes and hands in (see compile_heightfield_payload); the generation, the
// collider trailer, and the shared tangent / LOD pipeline stay here.
mod heightfield;
mod primitives;
mod room;
mod skybox;
mod terrain;
mod voxel;
pub mod water_grid;

#[cfg_attr(not(target_os = "macos"), allow(unused_imports))]
pub use chunk_gen::{ChunkBlockType, ChunkGenerator};

// Interleaved CPU vertex tuples the geometry generators produce before packing
// into the GPU `Vertex`. The submodules expose `Verts = Vec<Vert>`; the build
// helpers here also need the tangent-bearing and raw inline forms:
//   Vert    = position, normal, color, uv
//   VertT   = position, normal, tangent, color, uv
//   RawVert = position, color, uv (inline input, before normals are derived)
type Vert = ([f32; 3], [f32; 3], [f32; 3], [f32; 2]);
type VertT = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 2]);
type RawVert = ([f32; 3], [f32; 3], [f32; 2]);

// LOD alternates: (switch_distance, index buffer) pairs.
type LodAlternates = Vec<(f32, Vec<u16>)>;

// Convert a payload-form joint back into the args-form `JointDef`.
fn payload_joint_to_def(j: crate::gfx::mesh_payload::PayloadJoint) -> crate::assets::JointDef {
    crate::assets::JointDef {
        name: j.name,
        parent: j.parent,
        translation: j.translation,
        rotation_deg: j.rotation_deg,
        scale: j.scale,
    }
}

// Convert an args-form `JointDef` into the payload joint form.
fn joint_def_to_payload(j: &crate::assets::JointDef) -> crate::gfx::mesh_payload::PayloadJoint {
    crate::gfx::mesh_payload::PayloadJoint {
        name: j.name.clone(),
        parent: j.parent,
        translation: j.translation,
        rotation_deg: j.rotation_deg,
        scale: j.scale,
    }
}

// Convert a payload-joint vec to the args-form vec the runtime
// `build_skeleton_from_joint_defs` consumes. Public so the client runtime
// init path can call it without re-implementing the field mapping.
pub fn payload_joints_to_defs(
    joints: Vec<crate::gfx::mesh_payload::PayloadJoint>,
) -> Vec<crate::assets::JointDef> {
    joints.into_iter().map(payload_joint_to_def).collect()
}

// Compile a Mesh component's JSON args into a packed binary payload.
pub fn compile_mesh_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let generator = args.get("generator").and_then(|v| v.as_str()).unwrap_or("");

    let (vertices, indices): (Vec<Vert>, Vec<u16>) = match generator {
        "room" => room::build_room(args)?,
        "box" => primitives::build_box(args)?,
        "cylinder" => primitives::build_cylinder(args)?,
        "plane" => primitives::build_plane(args)?,
        "sphere" => primitives::build_sphere(args)?,
        "terrain" => terrain::build_terrain(args)?,
        "heightfield" => {
            // The heightfield generator needs the source image decoded, which
            // the build crate does before calling compile_heightfield_payload.
            return Err(
                "the `heightfield` generator decodes a source image; compile it \
                 through the build crate's heightfield path"
                    .to_string(),
            );
        }
        "water_grid" => water_grid::build_water_grid(args)?,
        "skybox" => skybox::build_skybox(args)?,
        "extrude" => extrude::build_extrude(args)?,
        // empty generator string = inline vertex data supplied directly in the blob
        "" => build_inline(args)?,
        other => return Err(format!("unknown mesh generator '{other}'")),
    };

    finish_mesh_payload(vertices, indices, args)
}

// Compile a heightfield mesh from a pre-decoded source image into a packed
// payload with a baked collider height grid. The build crate decodes the source
// image (this crate links no image decoders) and passes the RGBA pixels in. The
// collider grid is the mesh's own per-vertex Y in row-major order, so the
// physics terrain collider tracks the rendered surface vertex-for-vertex with
// no runtime image decode.
pub fn compile_heightfield_payload(
    args: &serde_json::Value,
    img_w: u32,
    img_h: u32,
    rgba: Vec<u8>,
) -> Result<Vec<u8>, String> {
    let (vertices, indices) =
        heightfield::build_heightfield_from_pixels(args, img_w, img_h, &rgba)?;
    // Read the LOD0 mesh's per-vertex Y for the collider grid before the payload
    // tail consumes the vertices.
    let (n, heights) = heightfield_collider_grid(&vertices)?;
    let mut payload = finish_mesh_payload(vertices, indices, args)?;
    payload.extend_from_slice(&crate::gfx::mesh_payload::serialise_heightfield_trailer(
        n, n, &heights,
    ));
    Ok(payload)
}

// Shared payload tail for every generator: derive tangents from the UV
// gradient, build the optional LOD alternate index lists, and serialise the
// packed mesh payload (with the LOD trailer when alternates are present).
//
// `lod_levels` (read from `args`) is the total count including LOD0;
// `build_lod_alternates` generates `levels - 1` decimated index lists via
// vertex clustering on the LOD0 vertex set, each paired with a switch distance.
// `lod_distances` either gives explicit thresholds or, when empty, the build
// derives a default doubling sequence from the bounding-sphere radius.
// `lod_levels = 1` (the default) produces no trailer and emits the legacy
// single-LOD payload byte-for-byte.
fn finish_mesh_payload(
    vertices: Vec<Vert>,
    indices: Vec<u16>,
    args: &serde_json::Value,
) -> Result<Vec<u8>, String> {
    let tangents = compute_tangents(&vertices, &indices);
    let verts5: Vec<VertT> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();
    let alternates = build_lod_alternates(args, &verts5, &indices)?;
    Ok(crate::gfx::mesh_payload::serialise_with_lods(
        &verts5,
        &indices,
        &alternates,
    ))
}

// Extract the square collider height grid from a heightfield mesh's vertices.
// The generator emits exactly `(subdivisions + 1)²` vertices in row-major
// order with each vertex's Y already mapped through the elevation range, so the
// grid is just those Y values; `n` is the side length.
fn heightfield_collider_grid(verts: &[Vert]) -> Result<(usize, Vec<f32>), String> {
    let n = verts.len().isqrt();
    if n * n != verts.len() {
        return Err(format!(
            "heightfield mesh has {} vertices, not a square grid",
            verts.len()
        ));
    }
    let heights = verts.iter().map(|v| v.0[1]).collect();
    Ok((n, heights))
}

// Build the per-LOD `(switch_distance, indices)` list for a mesh, honouring
// the `lod_levels` and `lod_distances` args. Returns an empty `Vec` when
// `lod_levels <= 1` so the payload writer emits a legacy single-LOD blob.
fn build_lod_alternates(
    args: &serde_json::Value,
    verts: &[VertT],
    indices: &[u16],
) -> Result<LodAlternates, String> {
    let lod_levels = args
        .get("lod_levels")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(1)
        .clamp(1, 8);
    if lod_levels <= 1 {
        return Ok(Vec::new());
    }
    let alt_count = (lod_levels - 1) as usize;

    // Read explicit thresholds or derive them from the bounding-sphere
    // radius. The default cascade doubles per level, which keeps successive
    // LODs visibly apart when seen in the showcase's free-fly camera.
    let explicit_distances: Vec<f32> = args
        .get("lod_distances")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_f64().map(|n| n as f32))
                .collect()
        })
        .unwrap_or_default();
    if !explicit_distances.is_empty() && explicit_distances.len() != alt_count {
        return Err(format!(
            "lod_distances has {} entries but lod_levels = {} expects {}",
            explicit_distances.len(),
            lod_levels,
            alt_count,
        ));
    }
    let radius = bounding_sphere_radius(verts);

    let positions: Vec<[f32; 3]> = verts.iter().map(|(p, _, _, _, _)| *p).collect();
    let lod0_tri_count = indices.len() / 3;
    let mut out = Vec::with_capacity(alt_count);
    for level in 1..lod_levels {
        let target = crate::gfx::lod::target_tri_count_for_level(lod0_tri_count, level);
        let idx = crate::gfx::lod::decimate_by_qem(&positions, indices, target);
        // Drop LOD if the decimator collapsed everything (degenerate input);
        // remaining levels would also be empty so we stop early.
        if idx.is_empty() {
            break;
        }
        let distance = if explicit_distances.is_empty() {
            crate::gfx::lod::default_distance_for_level(radius, level)
        } else {
            explicit_distances[(level - 1) as usize]
        };
        out.push((distance, idx));
    }
    Ok(out)
}

// Bounding-sphere radius around the mesh AABB centre. Cheap upper bound on
// the per-vertex distance to centre, used to seed default LOD thresholds.
fn bounding_sphere_radius(verts: &[VertT]) -> f32 {
    if verts.is_empty() {
        return 1.0;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for v in verts {
        let p = v.0;
        for k in 0..3 {
            mn[k] = mn[k].min(p[k]);
            mx[k] = mx[k].max(p[k]);
        }
    }
    let dx = mx[0] - mn[0];
    let dy = mx[1] - mn[1];
    let dz = mx[2] - mn[2];
    (0.5 * (dx * dx + dy * dy + dz * dz).sqrt()).max(0.25)
}

// Compile typed vertex and index data into a packed binary mesh payload.
//
// Normals are computed from triangle geometry; tangents from UV gradients.
// Shared by the inline Mesh path and file-backed mesh formats (e.g. OBJ).
pub fn compile_mesh_from_vertex_data(
    vertex_data: &[crate::assets::VertexData],
    indices: &[u16],
) -> Vec<u8> {
    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; vertex_data.len()];
    let tris = indices.len() / 3;
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        if ia >= vertex_data.len() || ib >= vertex_data.len() || ic >= vertex_data.len() {
            continue;
        }
        let n = vec3_face_normal(
            vertex_data[ia].pos,
            vertex_data[ib].pos,
            vertex_data[ic].pos,
        );
        vec3_add(&mut normals[ia], n);
        vec3_add(&mut normals[ib], n);
        vec3_add(&mut normals[ic], n);
    }
    let vertices: Vec<Vert> = vertex_data
        .iter()
        .enumerate()
        .map(|(i, v)| (v.pos, vec3_normalise(normals[i]), v.color, v.uv))
        .collect();
    let tangents = compute_tangents(&vertices, indices);
    let verts5: Vec<_> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();
    crate::gfx::mesh_payload::serialise(&verts5, indices)
}

// Compile a skinned mesh payload with optional LOD alternates. `lod_levels`
// includes LOD0 (so `1` emits no alternates); `lod_distances` may be empty
// (use a doubling cascade derived from the bounding sphere) or must hold
// exactly `lod_levels - 1` thresholds. Decimation is QEM half-edge collapse
// against the LOD0 vertex set, mirroring the static [`build_lod_alternates`]
// path.
pub fn compile_skinned_mesh_payload_with_lods(
    vertex_data: &[crate::assets::SkinnedVertexData],
    indices: &[u16],
    skeleton: &[crate::assets::JointDef],
    lod_levels: u32,
    lod_distances: &[f32],
) -> Result<Vec<u8>, String> {
    if vertex_data.is_empty() {
        return Err("SkinnedMesh requires at least one vertex".to_string());
    }
    let tris = indices.len() / 3;
    for t in 0..tris {
        for k in 0..3 {
            if indices[t * 3 + k] as usize >= vertex_data.len() {
                return Err(format!("SkinnedMesh index out of range in triangle {t}"));
            }
        }
    }

    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; vertex_data.len()];
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        let n = vec3_face_normal(
            vertex_data[ia].pos,
            vertex_data[ib].pos,
            vertex_data[ic].pos,
        );
        vec3_add(&mut normals[ia], n);
        vec3_add(&mut normals[ib], n);
        vec3_add(&mut normals[ic], n);
    }
    let pnt: Vec<Vert> = vertex_data
        .iter()
        .enumerate()
        .map(|(i, v)| (v.pos, vec3_normalise(normals[i]), v.color, v.uv))
        .collect();
    let tangents = compute_tangents(&pnt, indices);

    let skinned: Vec<crate::gfx::mesh_payload::SkinnedVertex> = vertex_data
        .iter()
        .zip(pnt.iter())
        .zip(tangents)
        .map(|((v, (pos, normal, color, uv)), tangent)| {
            let sum: f32 = v.weights.iter().sum();
            let weights = if sum > 1e-6 {
                [
                    v.weights[0] / sum,
                    v.weights[1] / sum,
                    v.weights[2] / sum,
                    v.weights[3] / sum,
                ]
            } else {
                // No weights authored: bind fully to the first joint.
                [1.0, 0.0, 0.0, 0.0]
            };
            crate::gfx::mesh_payload::SkinnedVertex {
                pos: *pos,
                normal: *normal,
                tangent,
                color: *color,
                uv: *uv,
                joints: [
                    v.joints[0] as u16,
                    v.joints[1] as u16,
                    v.joints[2] as u16,
                    v.joints[3] as u16,
                ],
                weights,
            }
        })
        .collect();

    let payload_joints: Vec<crate::gfx::mesh_payload::PayloadJoint> =
        skeleton.iter().map(joint_def_to_payload).collect();

    // Bake LOD alternates against the skinned vertex set. Half-edge QEM
    // preserves the vertex range so the runtime can share one SkinnedVertex
    // buffer across LOD0 and every alternate; only the per-LOD index list
    // varies.
    let alternates = if lod_levels > 1 {
        let alt_count = (lod_levels - 1) as usize;
        if !lod_distances.is_empty() && lod_distances.len() != alt_count {
            return Err(format!(
                "lod_distances has {} entries but lod_levels = {} expects {}",
                lod_distances.len(),
                lod_levels,
                alt_count,
            ));
        }
        let positions: Vec<[f32; 3]> = skinned.iter().map(|v| v.pos).collect();
        let radius = skinned_bounding_sphere_radius(&skinned);
        let lod0_tris = indices.len() / 3;
        let mut out: Vec<(f32, Vec<u16>)> = Vec::with_capacity(alt_count);
        for level in 1..lod_levels {
            let target = crate::gfx::lod::target_tri_count_for_level(lod0_tris, level);
            let idx = crate::gfx::lod::decimate_by_qem(&positions, indices, target);
            if idx.is_empty() {
                break;
            }
            let distance = if lod_distances.is_empty() {
                crate::gfx::lod::default_distance_for_level(radius, level)
            } else {
                lod_distances[(level - 1) as usize]
            };
            out.push((distance, idx));
        }
        out
    } else {
        Vec::new()
    };

    Ok(crate::gfx::mesh_payload::serialise_skinned_with_lods(
        &skinned,
        indices,
        &payload_joints,
        &alternates,
    ))
}

fn skinned_bounding_sphere_radius(verts: &[crate::gfx::mesh_payload::SkinnedVertex]) -> f32 {
    if verts.is_empty() {
        return 1.0;
    }
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for v in verts {
        for k in 0..3 {
            mn[k] = mn[k].min(v.pos[k]);
            mx[k] = mx[k].max(v.pos[k]);
        }
    }
    let dx = mx[0] - mn[0];
    let dy = mx[1] - mn[1];
    let dz = mx[2] - mn[2];
    (0.5 * (dx * dx + dy * dy + dz * dz).sqrt()).max(0.25)
}

// Compile a VoxelChunk component into a packed binary mesh payload.
//
// `palette_lookup` resolves each palette entry name to its BlockType args
// (typically by scanning the world.jsonl asset list). Non-solid entries
// (`solid: false`) become `None` in the resolved palette and emit no faces.
pub fn compile_voxel_chunk_payload<F>(
    args: &serde_json::Value,
    mut palette_lookup: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&str) -> Option<serde_json::Value>,
{
    let dim = parse_u32x3(args.get("dim"), "dim")?;
    let block_size = args
        .get("block_size")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    let palette_names: Vec<String> = args
        .get("palette")
        .and_then(|v| v.as_array())
        .ok_or("VoxelChunk: `palette` must be an array of BlockType names")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| "VoxelChunk: palette entries must be strings".to_string())
                .map(str::to_string)
        })
        .collect::<Result<_, _>>()?;
    let blocks: Vec<u32> = args
        .get("blocks")
        .and_then(|v| v.as_array())
        .ok_or("VoxelChunk: `blocks` must be an array of palette indices")?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_u64()
                .map(|x| x as u32)
                .ok_or_else(|| format!("VoxelChunk: blocks[{i}] must be a non-negative integer"))
        })
        .collect::<Result<_, _>>()?;

    let mut palette: Vec<Option<voxel::PaletteSlot>> = Vec::with_capacity(palette_names.len());
    for name in &palette_names {
        let bt_args = palette_lookup(name).ok_or_else(|| {
            format!("VoxelChunk: palette entry '{name}' has no matching BlockType asset")
        })?;
        palette.push(resolve_block_type(&bt_args));
    }

    let (vertices, indices) = voxel::build_voxel_mesh(dim, block_size, &blocks, &palette)?;
    let tangents = compute_tangents(&vertices, &indices);
    let verts5: Vec<_> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();

    // Optional LOD alternates: each level halves the triangle budget via
    // QEM half-edge collapse. Default off; opt in with `lod_levels: 2..=8`
    // on the VoxelChunk asset. Routed through the same payload trailer
    // as Mesh / ProceduralMesh so the runtime LoadedMesh path picks them
    // up unchanged.
    let alternates = build_lod_alternates(args, &verts5, &indices)?;
    Ok(crate::gfx::mesh_payload::serialise_with_lods(
        &verts5,
        &indices,
        &alternates,
    ))
}

// Build a renderable mesh for one procedurally generated chunk.
//
// The runtime counterpart of `compile_voxel_chunk_payload`: it takes a
// chunk's already-generated block array and resolved palette and returns
// interleaved `Vertex` geometry directly, with no on-disk payload in between.
// Chunk streaming (`app::chunk_stream`) calls this on its background thread.
pub fn build_chunk_mesh(
    dim: [u32; 3],
    block_size: f32,
    blocks: &[u32],
    palette: &[ChunkBlockType],
) -> Result<(Vec<crate::gfx::mesh_payload::Vertex>, Vec<u16>), String> {
    let slots: Vec<Option<voxel::PaletteSlot>> = palette
        .iter()
        .map(|b| {
            if b.solid {
                Some(voxel::PaletteSlot {
                    uv_top: b.uv_top,
                    uv_bottom: b.uv_bottom,
                    uv_side: b.uv_side,
                })
            } else {
                None
            }
        })
        .collect();
    let (verts, indices) = voxel::build_voxel_mesh(dim, block_size, blocks, &slots)?;
    let tangents = compute_tangents(&verts, &indices);
    let vertices = verts
        .into_iter()
        .zip(tangents)
        .map(
            |((pos, normal, color, uv), tangent)| crate::gfx::mesh_payload::Vertex {
                pos,
                normal,
                tangent,
                color,
                uv,
            },
        )
        .collect();
    Ok((vertices, indices))
}

// Build a coarse "impostor" mesh for one distant chunk from its terrain
// surface heights.
//
// Where [`build_chunk_mesh`] emits every visible voxel face, this stands a
// far-away chunk in for a fraction of the triangles: the surface height
// sampled on a coarse `step`-block grid becomes a low-poly top surface (one
// quad per coarse cell), wrapped by a perimeter skirt that drops to the chunk
// floor to hide the gap against a nearer full-detail neighbour or the world
// edge. Side and subsurface geometry are dropped: invisible at impostor
// distance.
//
// `heights[gz * (nx + 1) + gx]` is the surface block index at coarse corner
// `(gx, gz)`, where `nx = ceil(dx / step)`, `nz = ceil(dz / step)`, and corner
// `gx`'s local block column is `min(gx * step, dx)` (the last corner lands on
// the chunk's far edge so adjacent impostors share it exactly). The caller
// samples those heights from [`ChunkGenerator::surface_height_world`] at the
// matching world columns, which keeps neighbouring impostors watertight.
// `top_uv` / `side_uv` are the surface block's atlas rects.
pub fn build_chunk_impostor_mesh(
    dim: [u32; 3],
    block_size: f32,
    step: u32,
    heights: &[i32],
    top_uv: [f32; 4],
    side_uv: [f32; 4],
) -> Result<(Vec<crate::gfx::mesh_payload::Vertex>, Vec<u16>), String> {
    let step = step.max(1);
    let [dx, _dy, dz] = dim;
    let nx = dx.div_ceil(step);
    let nz = dz.div_ceil(step);
    let cols = (nx + 1) as usize;
    let expected = ((nx + 1) * (nz + 1)) as usize;
    if heights.len() != expected {
        return Err(format!(
            "impostor mesh: expected {} height samples for a {}x{} coarse grid, got {}",
            expected,
            nx + 1,
            nz + 1,
            heights.len()
        ));
    }
    let bs = block_size;
    // Local position of coarse corner gx / gz. The last corner clamps to the
    // chunk's far edge (dx / dz) so a non-dividing `step` still closes the mesh
    // exactly on the boundary shared with the next chunk.
    let cx = |gx: u32| ((gx * step).min(dx) as f32) * bs;
    let cz = |gz: u32| ((gz * step).min(dz) as f32) * bs;
    // Top of the surface block at corner (gx, gz): the +1 matches the full
    // mesher, whose top face of block `h` sits at `(h + 1) * block_size`.
    let surf_y = |gx: u32, gz: u32| ((heights[gz as usize * cols + gx as usize] + 1) as f32) * bs;

    type RawVerts = Vec<([f32; 3], [f32; 3], [f32; 3], [f32; 2])>;
    let mut verts: RawVerts = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let color = [0.75f32, 0.74, 0.72];

    // CCW-from-outside quad, matching `build_voxel_mesh`'s winding + UV mapping.
    let mut emit_quad = |corners: [[f32; 3]; 4], normal: [f32; 3], uv_rect: [f32; 4]| {
        if verts.len() + 4 > u16::MAX as usize {
            return;
        }
        let base = verts.len() as u16;
        let [u0, v0, u1, v1] = uv_rect;
        let uvs = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]];
        for (i, p) in corners.iter().enumerate() {
            verts.push((*p, normal, color, uvs[i]));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    };

    // Top surface: one up-facing quad per coarse cell. Each cell carries its
    // own 4 vertices; adjacent cells sample identical corner heights, so the
    // duplicated corner vertices coincide and the surface stays watertight.
    let n_up = [0.0, 1.0, 0.0];
    for gz in 0..nz {
        for gx in 0..nx {
            emit_quad(
                [
                    [cx(gx), surf_y(gx, gz + 1), cz(gz + 1)],
                    [cx(gx + 1), surf_y(gx + 1, gz + 1), cz(gz + 1)],
                    [cx(gx + 1), surf_y(gx + 1, gz), cz(gz)],
                    [cx(gx), surf_y(gx, gz), cz(gz)],
                ],
                n_up,
                top_uv,
            );
        }
    }

    // Perimeter skirt: vertical quads from the surface edge down to the chunk
    // floor (y = 0), one per boundary segment, facing outward. Hides the seam
    // where a coarse impostor abuts a nearer full chunk (or the world edge).
    let x_max = (dx as f32) * bs;
    let z_max = (dz as f32) * bs;
    for gx in 0..nx {
        // -Z edge (z = 0), outward normal -Z.
        emit_quad(
            [
                [cx(gx + 1), 0.0, 0.0],
                [cx(gx), 0.0, 0.0],
                [cx(gx), surf_y(gx, 0), 0.0],
                [cx(gx + 1), surf_y(gx + 1, 0), 0.0],
            ],
            [0.0, 0.0, -1.0],
            side_uv,
        );
        // +Z edge (z = z_max), outward normal +Z.
        emit_quad(
            [
                [cx(gx), 0.0, z_max],
                [cx(gx + 1), 0.0, z_max],
                [cx(gx + 1), surf_y(gx + 1, nz), z_max],
                [cx(gx), surf_y(gx, nz), z_max],
            ],
            [0.0, 0.0, 1.0],
            side_uv,
        );
    }
    for gz in 0..nz {
        // -X edge (x = 0), outward normal -X.
        emit_quad(
            [
                [0.0, 0.0, cz(gz)],
                [0.0, 0.0, cz(gz + 1)],
                [0.0, surf_y(0, gz + 1), cz(gz + 1)],
                [0.0, surf_y(0, gz), cz(gz)],
            ],
            [-1.0, 0.0, 0.0],
            side_uv,
        );
        // +X edge (x = x_max), outward normal +X.
        emit_quad(
            [
                [x_max, 0.0, cz(gz + 1)],
                [x_max, 0.0, cz(gz)],
                [x_max, surf_y(nx, gz), cz(gz)],
                [x_max, surf_y(nx, gz + 1), cz(gz + 1)],
            ],
            [1.0, 0.0, 0.0],
            side_uv,
        );
    }

    let tangents = compute_tangents(&verts, &indices);
    let vertices = verts
        .into_iter()
        .zip(tangents)
        .map(
            |((pos, normal, color, uv), tangent)| crate::gfx::mesh_payload::Vertex {
                pos,
                normal,
                tangent,
                color,
                uv,
            },
        )
        .collect();
    Ok((vertices, indices))
}

fn resolve_block_type(bt_args: &serde_json::Value) -> Option<voxel::PaletteSlot> {
    let solid = bt_args
        .get("solid")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    if !solid {
        return None;
    }
    let uv_min = parse_f32x2(bt_args.get("uv_min"), "uv_min").unwrap_or([0.0, 0.0]);
    let uv_max = parse_f32x2(bt_args.get("uv_max"), "uv_max").unwrap_or([1.0, 1.0]);
    let default_rect = [uv_min[0], uv_min[1], uv_max[0], uv_max[1]];
    let parse_rect = |v: Option<&serde_json::Value>| -> [f32; 4] {
        v.and_then(|x| x.as_array())
            .and_then(|a| {
                if a.len() < 4 {
                    return None;
                }
                let mut out = [0.0f32; 4];
                for (i, e) in out.iter_mut().enumerate() {
                    *e = a[i].as_f64()? as f32;
                }
                Some(out)
            })
            .unwrap_or(default_rect)
    };
    Some(voxel::PaletteSlot {
        uv_top: parse_rect(bt_args.get("uv_top")),
        uv_bottom: parse_rect(bt_args.get("uv_bottom")),
        uv_side: parse_rect(bt_args.get("uv_side")),
    })
}

fn parse_u32x3(v: Option<&serde_json::Value>, label: &str) -> Result<[u32; 3], String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("{label} must be an array of 3 non-negative integers"))?;
    if arr.len() < 3 {
        return Err(format!("{label} must have 3 elements, got {}", arr.len()));
    }
    let f = |i: usize| -> Result<u32, String> {
        arr[i]
            .as_u64()
            .map(|x| x as u32)
            .ok_or_else(|| format!("{label}[{i}] must be a non-negative integer"))
    };
    Ok([f(0)?, f(1)?, f(2)?])
}

// Compile a Room component's JSON args into a packed binary payload.
//
// Handles the `size: [width, depth, height]` shorthand in addition to the
// `half_width` / `half_depth` / `ceiling_height` fields. Texture references
// in the args are ignored here; they are resolved by the build pipeline and
// GraphicsSystem at load time.
pub fn compile_room_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let (half_width, half_depth, ceiling_height) = if let Some(size) = args
        .get("size")
        .and_then(|v| v.as_array())
        .filter(|a| a.len() >= 3)
    {
        let w = size[0].as_f64().unwrap_or(16.0) as f32;
        let d = size[1].as_f64().unwrap_or(20.0) as f32;
        let h = size[2].as_f64().unwrap_or(3.5) as f32;
        (w / 2.0, d / 2.0, h)
    } else {
        let hw = args
            .get("half_width")
            .and_then(|v| v.as_f64())
            .unwrap_or(8.0) as f32;
        let hd = args
            .get("half_depth")
            .and_then(|v| v.as_f64())
            .unwrap_or(10.0) as f32;
        let ch = args
            .get("ceiling_height")
            .and_then(|v| v.as_f64())
            .unwrap_or(3.5) as f32;
        (hw, hd, ch)
    };

    let (vertices, indices) =
        room::build_room_geometry(half_width, half_depth, 0.0, ceiling_height);
    let tangents = compute_tangents(&vertices, &indices);
    let verts5: Vec<_> = vertices
        .into_iter()
        .zip(tangents)
        .map(|((pos, normal, color, uv), tangent)| (pos, normal, tangent, color, uv))
        .collect();
    let alternates = build_lod_alternates(args, &verts5, &indices)?;
    Ok(crate::gfx::mesh_payload::serialise_with_lods(
        &verts5,
        &indices,
        &alternates,
    ))
}

// Builds geometry from inline vertex/index data in the blob JSON.
//
// Vertices may be specified in one of two forms:
//
//   Named fields (preferred):
//     {"pos": [x, y, z], "color": [r, g, b], "uv": [u, v]}
//
//   Flat array (legacy, still accepted):
//     [x, y, z, r, g, b, u, v]   (8 values) or [x, y, z, r, g, b] (6 values, uv defaults to 0)
//
// Normals are computed automatically from the triangle data.
fn build_inline(args: &serde_json::Value) -> Result<(Vec<Vert>, Vec<u16>), String> {
    let verts = args
        .get("vertices")
        .and_then(|v| v.as_array())
        .ok_or("inline Mesh requires a `vertices` array")?;

    let idxs = args
        .get("indices")
        .and_then(|v| v.as_array())
        .ok_or("inline Mesh requires an `indices` array")?;

    let parsed: Vec<RawVert> = verts
        .iter()
        .enumerate()
        .map(|(i, v)| parse_vertex(v, i))
        .collect::<Result<Vec<_>, _>>()?;

    let indices: Vec<u16> = idxs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v.as_u64()
                .map(|x| x as u16)
                .ok_or_else(|| format!("index[{i}] must be an integer"))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut normals: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; parsed.len()];
    let tris = indices.len() / 3;
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        if ia >= parsed.len() || ib >= parsed.len() || ic >= parsed.len() {
            return Err(format!("index out of range in triangle {t}"));
        }
        let n = vec3_face_normal(parsed[ia].0, parsed[ib].0, parsed[ic].0);
        vec3_add(&mut normals[ia], n);
        vec3_add(&mut normals[ib], n);
        vec3_add(&mut normals[ic], n);
    }

    let vertices = parsed
        .into_iter()
        .enumerate()
        .map(|(i, (pos, color, uv))| (pos, vec3_normalise(normals[i]), color, uv))
        .collect();

    Ok((vertices, indices))
}

// Compute a per-vertex tangent vector for every vertex in the mesh.
//
// For each triangle the tangent is derived from the UV gradient. Contributions
// are accumulated at each shared vertex and then Gram-Schmidt orthogonalized
// against the existing normal. Degenerate UV triangles fall back to an
// arbitrary perpendicular so the TBN matrix is always well-defined.
fn compute_tangents(vertices: &[Vert], indices: &[u16]) -> Vec<[f32; 3]> {
    let n = vertices.len();
    let mut accum: Vec<[f32; 3]> = vec![[0.0; 3]; n];

    let tris = indices.len() / 3;
    for t in 0..tris {
        let ia = indices[t * 3] as usize;
        let ib = indices[t * 3 + 1] as usize;
        let ic = indices[t * 3 + 2] as usize;
        if ia >= n || ib >= n || ic >= n {
            continue;
        }
        let (pa, _, _, uva) = vertices[ia];
        let (pb, _, _, uvb) = vertices[ib];
        let (pc, _, _, uvc) = vertices[ic];

        let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let du1 = uvb[0] - uva[0];
        let dv1 = uvb[1] - uva[1];
        let du2 = uvc[0] - uva[0];
        let dv2 = uvc[1] - uva[1];

        let denom = du1 * dv2 - du2 * dv1;
        let tangent = if denom.abs() < 1e-8 {
            arbitrary_tangent(vertices[ia].1)
        } else {
            let r = 1.0 / denom;
            [
                (e1[0] * dv2 - e2[0] * dv1) * r,
                (e1[1] * dv2 - e2[1] * dv1) * r,
                (e1[2] * dv2 - e2[2] * dv1) * r,
            ]
        };

        vec3_add(&mut accum[ia], tangent);
        vec3_add(&mut accum[ib], tangent);
        vec3_add(&mut accum[ic], tangent);
    }

    vertices
        .iter()
        .zip(accum)
        .map(|((_, normal, _, _), raw)| {
            let dot = raw[0] * normal[0] + raw[1] * normal[1] + raw[2] * normal[2];
            let t = [
                raw[0] - dot * normal[0],
                raw[1] - dot * normal[1],
                raw[2] - dot * normal[2],
            ];
            vec3_normalise(t)
        })
        .collect()
}

// Returns an arbitrary unit vector perpendicular to `normal`.
fn arbitrary_tangent(normal: [f32; 3]) -> [f32; 3] {
    let up = if normal[0].abs() <= normal[1].abs() && normal[0].abs() <= normal[2].abs() {
        [1.0f32, 0.0, 0.0]
    } else if normal[1].abs() <= normal[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    };
    let t = [
        up[1] * normal[2] - up[2] * normal[1],
        up[2] * normal[0] - up[0] * normal[2],
        up[0] * normal[1] - up[1] * normal[0],
    ];
    vec3_normalise(t)
}

pub(super) fn vec3_face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    vec3_normalise(n)
}

pub(super) fn vec3_add(dst: &mut [f32; 3], src: [f32; 3]) {
    dst[0] += src[0];
    dst[1] += src[1];
    dst[2] += src[2];
}

pub(super) fn vec3_normalise(n: [f32; 3]) -> [f32; 3] {
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len < 1e-6 {
        [0.0, 1.0, 0.0]
    } else {
        [n[0] / len, n[1] / len, n[2] / len]
    }
}

fn parse_vertex(v: &serde_json::Value, idx: usize) -> Result<RawVert, String> {
    if let Some(obj) = v.as_object() {
        let pos = parse_f32x3(obj.get("pos"), &format!("vertex[{idx}].pos"))?;
        let color = parse_f32x3(obj.get("color"), &format!("vertex[{idx}].color"))?;
        let uv = if let Some(u) = obj.get("uv") {
            parse_f32x2(Some(u), &format!("vertex[{idx}].uv"))?
        } else {
            [0.0, 0.0]
        };
        Ok((pos, color, uv))
    } else if let Some(arr) = v.as_array() {
        if arr.len() < 6 {
            return Err(format!(
                "vertex[{idx}] flat array must have 6 or 8 elements, got {}",
                arr.len()
            ));
        }
        let f = |i: usize| -> Result<f32, String> {
            arr[i]
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("vertex[{idx}][{i}] must be a number"))
        };
        let uv = if arr.len() >= 8 {
            [f(6)?, f(7)?]
        } else {
            [0.0, 0.0]
        };
        Ok(([f(0)?, f(1)?, f(2)?], [f(3)?, f(4)?, f(5)?], uv))
    } else {
        Err(format!(
            "vertex[{idx}] must be an object or a 6- or 8-element array"
        ))
    }
}

pub(super) fn parse_f32x3(v: Option<&serde_json::Value>, label: &str) -> Result<[f32; 3], String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("{label} must be an array of 3 numbers"))?;
    if arr.len() < 3 {
        return Err(format!("{label} must have 3 elements, got {}", arr.len()));
    }
    let f = |i: usize| -> Result<f32, String> {
        arr[i]
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("{label}[{i}] must be a number"))
    };
    Ok([f(0)?, f(1)?, f(2)?])
}

fn parse_f32x2(v: Option<&serde_json::Value>, label: &str) -> Result<[f32; 2], String> {
    let arr = v
        .and_then(|x| x.as_array())
        .ok_or_else(|| format!("{label} must be an array of 2 numbers"))?;
    if arr.len() < 2 {
        return Err(format!("{label} must have 2 elements, got {}", arr.len()));
    }
    let f = |i: usize| -> Result<f32, String> {
        arr[i]
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("{label}[{i}] must be a number"))
    };
    Ok([f(0)?, f(1)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    // A flat coarse height grid of `(nx+1)*(nz+1)` corners all at height `h`.
    fn flat_heights(dim: [u32; 3], step: u32, h: i32) -> Vec<i32> {
        let nx = dim[0].div_ceil(step);
        let nz = dim[2].div_ceil(step);
        vec![h; ((nx + 1) * (nz + 1)) as usize]
    }

    #[test]
    fn impostor_mesh_counts_match_cells_plus_skirt() {
        // 16x_x16 chunk, step 4 -> nx = nz = 4 coarse cells.
        let dim = [16, 24, 16];
        let step = 4;
        let heights = flat_heights(dim, step, 5);
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, i) = build_chunk_impostor_mesh(dim, 1.0, step, &heights, uv, uv).expect("impostor");
        // top quads = 4*4 = 16; skirt quads = 2*(4+4) = 16; total 32 quads.
        let quads = 16 + 16;
        assert_eq!(v.len(), quads * 4);
        assert_eq!(i.len(), quads * 6);
        // Far cheaper than a full chunk's tens-of-thousands of vertices.
        assert!(v.len() < 200);
    }

    #[test]
    fn impostor_top_surface_sits_above_the_surface_block() {
        // A flat surface at block height 5 puts the top face at (5+1)*bs.
        let dim = [8, 16, 8];
        let bs = 2.0;
        let heights = flat_heights(dim, 4, 5);
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, _) = build_chunk_impostor_mesh(dim, bs, 4, &heights, uv, uv).expect("impostor");
        let top_y = (5 + 1) as f32 * bs;
        // The two coarse cells produce a top surface entirely at top_y; the
        // skirt drops to y = 0. Both extremes must be present.
        assert!(v.iter().any(|vert| (vert.pos[1] - top_y).abs() < 1e-4));
        assert!(v.iter().any(|vert| vert.pos[1].abs() < 1e-4));
        // Nothing pokes above the surface.
        assert!(v.iter().all(|vert| vert.pos[1] <= top_y + 1e-4));
    }

    #[test]
    fn impostor_spans_the_full_chunk_footprint() {
        // The mesh must cover [0, dx*bs] x [0, dz*bs] so it tiles seamlessly
        // with neighbouring chunks placed one chunk apart.
        let dim = [16, 24, 16];
        let bs = 1.0;
        let heights = flat_heights(dim, 4, 3);
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, _) = build_chunk_impostor_mesh(dim, bs, 4, &heights, uv, uv).expect("impostor");
        let max_x = v.iter().map(|vert| vert.pos[0]).fold(0.0f32, f32::max);
        let max_z = v.iter().map(|vert| vert.pos[2]).fold(0.0f32, f32::max);
        assert!((max_x - 16.0).abs() < 1e-4, "x extent {}", max_x);
        assert!((max_z - 16.0).abs() < 1e-4, "z extent {}", max_z);
    }

    #[test]
    fn impostor_rejects_a_mismatched_height_grid() {
        let dim = [16, 24, 16];
        let uv = [0.0, 0.0, 1.0, 1.0];
        // step 4 wants a 5x5 grid (25 samples); give it 9.
        let bad = vec![0i32; 9];
        assert!(build_chunk_impostor_mesh(dim, 1.0, 4, &bad, uv, uv).is_err());
    }

    #[test]
    fn impostor_with_step_exceeding_chunk_collapses_to_one_cell() {
        // step >= chunk dims -> nx = nz = 1: a single top quad + 4 skirt quads.
        let dim = [16, 24, 16];
        let heights = flat_heights(dim, 32, 2); // 2x2 corner grid
        let uv = [0.0, 0.0, 1.0, 1.0];
        let (v, i) = build_chunk_impostor_mesh(dim, 1.0, 32, &heights, uv, uv).expect("impostor");
        let quads = 1 + 2 * (1 + 1); // 1 top + 4 skirt
        assert_eq!(v.len(), quads * 4);
        assert_eq!(i.len(), quads * 6);
    }

    #[test]
    fn compile_mesh_payload_room_generator_succeeds() {
        let args = serde_json::json!({"generator": "room"});
        assert!(compile_mesh_payload(&args).is_ok());
    }

    #[test]
    fn compile_mesh_payload_sphere_generator_succeeds() {
        let args =
            serde_json::json!({"generator": "sphere", "radius": 1.0, "rings": 6, "segments": 8});
        assert!(compile_mesh_payload(&args).is_ok());
    }

    #[test]
    fn compile_mesh_payload_unknown_generator_errors() {
        let args = serde_json::json!({"generator": "nonexistent"});
        assert!(compile_mesh_payload(&args).is_err());
    }

    #[test]
    fn compile_mesh_payload_known_generators_ok() {
        for name in &[
            "room", "box", "cylinder", "plane", "sphere", "terrain", "skybox",
        ] {
            let args = serde_json::json!({"generator": name});
            assert!(
                compile_mesh_payload(&args).is_ok(),
                "expected ok for generator '{name}'"
            );
        }
    }

    #[test]
    fn compile_mesh_payload_extrude_generator_succeeds() {
        let args = serde_json::json!({
            "generator": "extrude",
            "profile": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
            "height": 1.0
        });
        assert!(compile_mesh_payload(&args).is_ok());
    }

    #[test]
    fn compile_mesh_payload_unknown_generator_name_errors() {
        let args = serde_json::json!({"generator": "teapot"});
        let err = compile_mesh_payload(&args).unwrap_err();
        assert!(
            err.contains("teapot"),
            "expected error mentioning 'teapot', got: {err}"
        );
    }

    #[test]
    fn compile_room_payload_size_shorthand() {
        let args = serde_json::json!({"size": [16.0, 20.0, 3.5]});
        let result = compile_room_payload(&args);
        assert!(result.is_ok());
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn compile_room_payload_half_extents() {
        let args =
            serde_json::json!({"half_width": 8.0, "half_depth": 10.0, "ceiling_height": 3.5});
        let args_size = serde_json::json!({"size": [16.0, 20.0, 3.5]});
        assert_eq!(
            compile_room_payload(&args).unwrap(),
            compile_room_payload(&args_size).unwrap()
        );
    }

    #[test]
    fn compile_room_payload_texture_fields_are_ignored() {
        let args = serde_json::json!({
            "size": [16.0, 20.0, 3.5],
            "wall_texture": "brick",
            "floor_texture": "concrete",
            "ceiling_texture": "checker"
        });
        assert!(compile_room_payload(&args).is_ok());
    }
}
