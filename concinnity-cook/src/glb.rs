// src/glb.rs
//
// Binary glTF (.glb) container parsing: turns a `.glb` file into the engine's
// inline mesh / skeleton / animation forms. Only the `.glb` container is
// handled: buffer data must travel in the embedded GLB binary chunk, so a
// `.gltf` with external or base64-URI buffers is rejected.
//
// glTF stores a skin's joints in an arbitrary order; this engine's `JointDef`
// list requires parents before children. Joints are therefore topologically
// reordered and a remap table rewrites both each joint's parent index and
// every vertex's `JOINTS_0` binding into the new index space.
//
// This is the decode half of the glTF pipeline. The asset-level desugar
// wrappers (`import_skinned_glb`, `import_glb_animation`, ...) live in
// `crate::gltf` and call into here.

use std::collections::HashMap;

use crate::assets::{JointDef, SkinnedVertexData, VertexData};
use crate::gfx::skinning::{JointPose, euler_yxz_from_quat};

// Neutral grey for vertex color, matches the wavefront/OBJ importer so
// imported geometry takes the material albedo without tinting.
const NEUTRAL_COLOR: [f32; 3] = [0.75, 0.74, 0.72];

// The inline `SkinnedMesh` fields produced from a glTF file.
pub struct ImportedSkinnedMesh {
    pub vertices: Vec<SkinnedVertexData>,
    pub indices: Vec<u16>,
    pub skeleton: Vec<JointDef>,
}

// Same as [`import_skinned_glb`] but takes a pre-parsed glTF document. The
// asset hot-reload pass uses this directly so it can amortise the `.glb`
// parse across every Mesh / SkinnedMesh entry that references the same file.
pub fn import_skinned_from_doc(
    doc: &gltf::Gltf,
    source: &str,
) -> Result<ImportedSkinnedMesh, String> {
    let blob = doc.blob.as_deref();

    // The skinned mesh is the first node carrying both a mesh and a skin.
    let node = doc
        .document
        .nodes()
        .find(|n| n.mesh().is_some() && n.skin().is_some())
        .ok_or_else(|| format!("'{}': no node with both a mesh and a skin", source))?;
    let mesh = node.mesh().unwrap();
    let skin = node.skin().unwrap();

    let skeleton = import_skeleton(&skin)?;
    let (vertices, indices) = import_geometry(&mesh, blob, &skeleton.remap)?;

    Ok(ImportedSkinnedMesh {
        vertices,
        indices,
        skeleton: skeleton.joints,
    })
}

// Parse a binary glTF file from disk. Shared by the skinned and static
// importers; the desugar pass uses this directly so it can memoize one GLB
// across many primitive/material/image lookups.
pub fn parse_glb(source: &str) -> Result<gltf::Gltf, String> {
    let path = resolve_source(source);
    let bytes = std::fs::read(&path).map_err(|e| format!("failed to read '{}': {}", path, e))?;
    gltf::Gltf::from_slice(&bytes)
        .map_err(|e| format!("'{}': not a valid glTF/GLB file: {}", path, e))
}

// Import the indexed primitive (flattened across glTF meshes in declaration
// order) from a parsed `.glb` into a static `Mesh`'s `(vertices, indices)`.
// UVs are taken from `TEXCOORD_0` when present; vertex colors fall back to
// neutral grey so the material albedo controls surface color.
//
// Only TRIANGLES topology is supported. POINTS/LINES/strip variants
// error out so a regression is obvious rather than silently mis-rendered.
//
// Errors if the primitive's vertex count or any index exceeds Concinnity's
// u16 index limit; `cn add` pre-splits oversized primitives at add time so
// the desugar pass never encounters one.
pub fn import_static_glb_primitive_from_doc(
    doc: &gltf::Gltf,
    source: &str,
    primitive_index: u32,
) -> Result<(Vec<VertexData>, Vec<u16>), String> {
    let (vertices, indices_u32) = read_primitive_geometry(doc, source, primitive_index)?;
    let mut indices: Vec<u16> = Vec::with_capacity(indices_u32.len());
    for v in indices_u32 {
        if v > u16::MAX as u32 {
            return Err(format!(
                "'{}': primitive {} exceeds the {}-vertex u16 index limit; \
                 import via `cn add` to auto-split it",
                source,
                primitive_index,
                u16::MAX
            ));
        }
        indices.push(v as u16);
    }
    Ok((vertices, indices))
}

// Read a primitive's vertices and u32-indexed triangle list from a parsed
// `.glb`. The shared backbone of [`import_static_glb_primitive_from_doc`]
// and the `cn add` splitting path; neither caller needs to repeat the
// triangle-topology check or attribute reads.
pub fn read_primitive_geometry(
    doc: &gltf::Gltf,
    source: &str,
    primitive_index: u32,
) -> Result<(Vec<VertexData>, Vec<u32>), String> {
    let blob = doc.blob.as_deref();

    let primitive = doc
        .document
        .meshes()
        .flat_map(|m| m.primitives())
        .nth(primitive_index as usize)
        .ok_or_else(|| {
            format!(
                "'{}': primitive_index {} is out of range",
                source, primitive_index
            )
        })?;

    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return Err(format!(
            "'{}': primitive {} uses topology {:?}; only TRIANGLES is supported",
            source,
            primitive_index,
            primitive.mode()
        ));
    }

    // GLB-only: buffer data must be the embedded binary chunk. External or
    // base64 buffer URIs resolve to None and fail the POSITION read below.
    let get_buffer = |buffer: gltf::Buffer<'_>| -> Option<&[u8]> {
        match buffer.source() {
            gltf::buffer::Source::Bin => blob,
            gltf::buffer::Source::Uri(_) => None,
        }
    };
    let reader = primitive.reader(get_buffer);

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| {
            format!(
                "'{}': primitive {} has no POSITION data (external .gltf buffers \
                 are unsupported, re-export as .glb)",
                source, primitive_index
            )
        })?
        .collect();
    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|t| t.into_f32().collect())
        .unwrap_or_default();
    let colors: Vec<[f32; 3]> = reader
        .read_colors(0)
        .map(|c| c.into_rgb_f32().collect())
        .unwrap_or_default();

    let mut vertices: Vec<VertexData> = Vec::with_capacity(positions.len());
    for (i, &pos) in positions.iter().enumerate() {
        vertices.push(VertexData {
            pos,
            color: colors.get(i).copied().unwrap_or(NEUTRAL_COLOR),
            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
        });
    }

    let indices: Vec<u32> = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };

    if vertices.is_empty() {
        return Err(format!(
            "'{}': primitive {} has no vertices",
            source, primitive_index
        ));
    }
    // Indices come straight from the (untrusted) glTF index buffer. Reject any
    // that reference past the vertex array here so downstream consumers like
    // split_into_u16_chunks can index `vertices` without bounds checks.
    if let Some(&bad) = indices.iter().find(|&&i| i as usize >= vertices.len()) {
        return Err(format!(
            "'{}': primitive {} index {} out of range ({} vertices)",
            source,
            primitive_index,
            bad,
            vertices.len()
        ));
    }
    Ok((vertices, indices))
}

// Split an oversized triangle list (any vertex count, u32 indices) into
// chunks that each fit in u16. Each chunk has its own vertex buffer containing
// only the vertices its triangles use; indices are remapped to that local
// buffer. The split is greedy by triangle order: fast and stable, with no
// attempt at locality optimisation. Vertices are duplicated across chunks
// when a triangle straddles a flush boundary; for chess-piece geometry the
// duplication is well under one percent.
pub fn split_into_u16_chunks(
    vertices: &[VertexData],
    indices: &[u32],
) -> Vec<(Vec<VertexData>, Vec<u16>)> {
    let limit: usize = u16::MAX as usize + 1;
    let mut chunks: Vec<(Vec<VertexData>, Vec<u16>)> = Vec::new();
    let mut cur_verts: Vec<VertexData> = Vec::new();
    let mut cur_indices: Vec<u16> = Vec::new();
    let mut remap: std::collections::HashMap<u32, u16> = std::collections::HashMap::new();

    for tri in indices.chunks_exact(3) {
        let new_in_tri = tri.iter().filter(|&&v| !remap.contains_key(&v)).count();
        if !cur_verts.is_empty() && cur_verts.len() + new_in_tri > limit {
            chunks.push((
                std::mem::take(&mut cur_verts),
                std::mem::take(&mut cur_indices),
            ));
            remap.clear();
        }
        for &v in tri {
            let local = *remap.entry(v).or_insert_with(|| {
                let idx = cur_verts.len() as u16;
                cur_verts.push(vertices[v as usize].clone());
                idx
            });
            cur_indices.push(local);
        }
    }
    if !cur_verts.is_empty() {
        chunks.push((cur_verts, cur_indices));
    }
    chunks
}

// Resolve a `SkinnedMesh.source` to an on-disk path. A bare filename (no
// directory component) is searched for under the fetched-assets directory
// (the same resolution `ColorLut` and `EnvironmentMap` sources use) while a
// path with a directory component is taken as-is, so a relative or absolute
// path still works for local test worlds.
pub fn resolve_source(source: &str) -> String {
    let bare = std::path::Path::new(source)
        .parent()
        .map(|d| d.as_os_str().is_empty())
        .unwrap_or(true);
    if !bare {
        return source.to_string();
    }
    if let Some(found) = concinnity_core::world::preset::find_in_assets(source) {
        return found;
    }
    concinnity_core::paths::assets_dir()
        .join(source)
        .to_string_lossy()
        .into_owned()
}

// A skeleton reordered into parents-before-children order, plus the lookup
// tables an animation importer needs to resolve a glTF channel target back
// to a joint in this skeleton.
pub struct ImportedSkeleton {
    pub joints: Vec<JointDef>,
    // `remap[skin_joint_index] = topologically-sorted index`.
    pub remap: Vec<usize>,
    // `node_to_joint[glTF_node_index] = skin_joint_index`. An animation
    // channel whose target node is missing from this map is targeting a
    // non-joint node and should be dropped.
    pub node_to_joint: HashMap<usize, usize>,
}

// Build the engine's skeleton (joints in parents-before-children order) from
// a glTF skin. Public so the animation importer can reuse the remap +
// node-to-joint table without re-deriving them.
pub fn import_skeleton(skin: &gltf::Skin<'_>) -> Result<ImportedSkeleton, String> {
    let joint_nodes: Vec<gltf::Node<'_>> = skin.joints().collect();
    let n = joint_nodes.len();
    if n == 0 {
        return Err("glTF skin has no joints".to_string());
    }

    // glTF node index -> skin-joint index, so a node's children can be
    // resolved back to joints.
    let node_to_joint: HashMap<usize, usize> = joint_nodes
        .iter()
        .enumerate()
        .map(|(sj, node)| (node.index(), sj))
        .collect();

    // A joint's parent is whichever joint lists it as a child. Joints not
    // claimed by any other joint are roots. A self-parent is dropped.
    let mut parents: Vec<Option<usize>> = vec![None; n];
    for (sj, node) in joint_nodes.iter().enumerate() {
        for child in node.children() {
            if let Some(&cj) = node_to_joint.get(&child.index())
                && cj != sj
            {
                parents[cj] = Some(sj);
            }
        }
    }

    let (order, remap) = topological_order(&parents);

    let joints = order
        .iter()
        .map(|&sj| {
            let node = &joint_nodes[sj];
            let (translation, rotation, scale) = node.transform().decomposed();
            JointDef {
                name: node.name().unwrap_or("").to_string(),
                parent: parents[sj].map_or(-1, |p| remap[p] as i32),
                translation,
                rotation_deg: euler_yxz_from_quat(rotation),
                scale,
            }
        })
        .collect();

    Ok(ImportedSkeleton {
        joints,
        remap,
        node_to_joint,
    })
}

// Order joint indices so every parent precedes its children, and return the
// inverse remap (skin-joint index -> sorted index). A joint whose parent
// chain cannot be resolved (a cycle, which a valid glTF skin never has) is
// emitted in its original order as a safety fallback so the function always
// returns a total order over every joint.
fn topological_order(parents: &[Option<usize>]) -> (Vec<usize>, Vec<usize>) {
    let n = parents.len();
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut emitted = vec![false; n];

    loop {
        let progress = order.len();
        for (s, parent) in parents.iter().enumerate() {
            if emitted[s] {
                continue;
            }
            let ready = match *parent {
                None => true,
                Some(p) => p >= n || emitted[p],
            };
            if ready {
                emitted[s] = true;
                order.push(s);
            }
        }
        if order.len() == progress {
            break;
        }
    }
    // Stragglers (a cycle): emit in original order rather than dropping them.
    for (s, done) in emitted.iter().enumerate() {
        if !done {
            order.push(s);
        }
    }

    let mut remap = vec![0usize; n];
    for (new_idx, &s) in order.iter().enumerate() {
        remap[s] = new_idx;
    }
    (order, remap)
}

fn import_geometry(
    mesh: &gltf::Mesh<'_>,
    blob: Option<&[u8]>,
    remap: &[usize],
) -> Result<(Vec<SkinnedVertexData>, Vec<u16>), String> {
    let mut vertices: Vec<SkinnedVertexData> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();

    // GLB-only: buffer data must be the embedded binary chunk. An external or
    // base64-URI buffer resolves to `None` and fails the POSITION read below.
    let get_buffer = |buffer: gltf::Buffer<'_>| -> Option<&[u8]> {
        match buffer.source() {
            gltf::buffer::Source::Bin => blob,
            gltf::buffer::Source::Uri(_) => None,
        }
    };

    for primitive in mesh.primitives() {
        let reader = primitive.reader(get_buffer);

        // A primitive with no JOINTS_0 is static geometry, skip it; the
        // SkinnedMesh asset only carries skinned vertices.
        let joints: Vec<[u16; 4]> = match reader.read_joints(0) {
            Some(j) => j.into_u16().collect(),
            None => continue,
        };
        let positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or_else(|| {
                "skinned primitive has no POSITION data (external .gltf buffers \
                 are unsupported, re-export as .glb)"
                    .to_string()
            })?
            .collect();
        let weights: Vec<[f32; 4]> = reader
            .read_weights(0)
            .ok_or_else(|| "skinned primitive missing WEIGHTS_0".to_string())?
            .into_f32()
            .collect();
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|t| t.into_f32().collect())
            .unwrap_or_default();
        let colors: Vec<[f32; 3]> = reader
            .read_colors(0)
            .map(|c| c.into_rgb_f32().collect())
            .unwrap_or_default();

        let base = vertices.len() as u32;
        for (i, &pos) in positions.iter().enumerate() {
            let raw = joints.get(i).copied().unwrap_or([0; 4]);
            let bound = |j: u16| -> u32 {
                // A JOINTS_0 index always indexes the skin's joint array;
                // remap it into the topologically-sorted index space.
                remap.get(j as usize).map_or(0, |&r| r as u32)
            };
            vertices.push(SkinnedVertexData {
                pos,
                color: colors.get(i).copied().unwrap_or([1.0, 1.0, 1.0]),
                uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                joints: [bound(raw[0]), bound(raw[1]), bound(raw[2]), bound(raw[3])],
                weights: weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]),
            });
        }

        let push_index = |indices: &mut Vec<u16>, v: u32| -> Result<(), String> {
            let abs = base + v;
            if abs > u16::MAX as u32 {
                return Err(format!(
                    "imported skinned mesh exceeds the {}-vertex u16 index limit",
                    u16::MAX
                ));
            }
            indices.push(abs as u16);
            Ok(())
        };
        match reader.read_indices() {
            Some(idx) => {
                for v in idx.into_u32() {
                    push_index(&mut indices, v)?;
                }
            }
            None => {
                // A non-indexed primitive draws vertices sequentially.
                for v in 0..positions.len() as u32 {
                    push_index(&mut indices, v)?;
                }
            }
        }
    }

    if vertices.is_empty() {
        return Err("glTF mesh has no skinned primitives (no JOINTS_0)".to_string());
    }
    Ok((vertices, indices))
}

// glTF animation import

// A single keyframe extracted from a glTF animation channel.
#[derive(Debug, Clone, Copy)]
pub struct ImportedKeyframe {
    pub time: f32,
    pub pose: JointPose,
}

// Per-joint channel of an imported animation.
#[derive(Debug, Clone)]
pub struct ImportedAnimationTrack {
    // Index in the engine's parents-before-children joint array.
    pub joint: usize,
    pub keys: Vec<ImportedKeyframe>,
}

// One animation extracted from a glTF file.
#[derive(Debug, Clone)]
pub struct ImportedAnimation {
    // glTF-side name; empty if the source did not name the clip.
    pub name: String,
    // Clip length in seconds: the largest sample time across all channels.
    pub duration: f32,
    // Joint-targeted channels, deduplicated and merged across translation /
    // rotation / scale targets so each joint has at most one entry.
    pub tracks: Vec<ImportedAnimationTrack>,
}

// Same as [`import_glb_animations`] but takes a pre-parsed glTF document.
// The asset hot-reload pass uses this directly so a single reload pass can
// amortise the `.glb` parse across every Animation entry that references
// the same file.
pub fn import_glb_animations_from_doc(
    doc: &gltf::Gltf,
    source: &str,
) -> Result<Vec<ImportedAnimation>, String> {
    let skin = first_skin(doc, source)?;
    let skeleton = import_skeleton(&skin)?;
    let blob = doc.blob.as_deref();
    Ok(doc
        .document
        .animations()
        .map(|anim| import_animation(&anim, &skeleton, blob))
        .collect())
}

// Resolve a `(animation_name, animation_index)` pair on a pre-parsed `.glb`,
// returning the selected clip. `animation_name` takes precedence: when
// non-empty, looks up the matching clip by name; otherwise falls back to
// `animation_index`. Used by the asset hot-reload pass to mirror the
// desugar pass's selection logic exactly so a reload picks the same clip
// the build chose at compile time.
pub fn import_glb_animation_from_doc(
    doc: &gltf::Gltf,
    source: &str,
    animation_index: u32,
    animation_name: &str,
) -> Result<ImportedAnimation, String> {
    let mut anims = import_glb_animations_from_doc(doc, source)?;
    let idx = if !animation_name.is_empty() {
        anims
            .iter()
            .position(|a| a.name == animation_name)
            .ok_or_else(|| {
                format!(
                    "'{}': no animation named '{}' (file has {} clip{})",
                    source,
                    animation_name,
                    anims.len(),
                    if anims.len() == 1 { "" } else { "s" }
                )
            })?
    } else {
        let i = animation_index as usize;
        if i >= anims.len() {
            return Err(format!(
                "'{}': animation_index {} out of range (file has {} animation{})",
                source,
                animation_index,
                anims.len(),
                if anims.len() == 1 { "" } else { "s" }
            ));
        }
        i
    };
    Ok(anims.swap_remove(idx))
}

// First node in `doc` carrying both a mesh and a skin, the same node the
// skinned-mesh importer picks, so both importers share one skeleton view.
fn first_skin<'a>(doc: &'a gltf::Gltf, path: &str) -> Result<gltf::Skin<'a>, String> {
    doc.document
        .nodes()
        .find(|n| n.mesh().is_some() && n.skin().is_some())
        .and_then(|n| n.skin())
        .ok_or_else(|| format!("'{}': no node with both a mesh and a skin", path))
}

// Build one `ImportedAnimation` from a glTF animation, dropping channels that
// target non-joint nodes. The skeleton's `remap` rewrites skin-joint indices
// into the engine's parents-before-children order.
fn import_animation(
    anim: &gltf::Animation<'_>,
    skeleton: &ImportedSkeleton,
    blob: Option<&[u8]>,
) -> ImportedAnimation {
    // joint index -> JointPose per sample time. Each channel writes only its
    // own property (T/R/S) and leaves the others at the bind pose, so we seed
    // every joint pose from the bind transform before walking channels.
    let bind_pose = |j: usize| -> JointPose {
        let def = &skeleton.joints[j];
        JointPose {
            translation: def.translation,
            rotation_deg: def.rotation_deg,
            scale: def.scale,
        }
    };

    // joint index -> (time -> pose)
    let mut tracks: HashMap<usize, Vec<(f32, JointPose)>> = HashMap::new();
    let mut max_time: f32 = 0.0;

    for channel in anim.channels() {
        let target_node = channel.target().node().index();
        let Some(&skin_joint) = skeleton.node_to_joint.get(&target_node) else {
            // Channel targets a non-joint (camera, prop, mesh node), drop.
            continue;
        };
        let joint_idx = skeleton
            .remap
            .get(skin_joint)
            .copied()
            .unwrap_or(skin_joint);
        let reader = channel.reader(|buf| match buf.source() {
            gltf::buffer::Source::Bin => blob,
            gltf::buffer::Source::Uri(_) => None,
        });
        let times: Vec<f32> = match reader.read_inputs() {
            Some(t) => t.collect(),
            None => continue,
        };
        for &t in &times {
            if t > max_time {
                max_time = t;
            }
        }

        let interpolation = channel.sampler().interpolation();
        let entry = tracks.entry(joint_idx).or_default();
        let bind = bind_pose(joint_idx);
        let upsert = |entry: &mut Vec<(f32, JointPose)>, time: f32| -> usize {
            // Same-time pose merges across T/R/S channels.
            if let Some(pos) = entry.iter().position(|(t, _)| (*t - time).abs() < 1e-6) {
                pos
            } else {
                entry.push((time, bind));
                entry.len() - 1
            }
        };

        match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Translations(it)) => {
                let values: Vec<[f32; 3]> = it.collect();
                let samples = sampled(&times, &values, interpolation);
                for (time, t) in samples {
                    let i = upsert(entry, time);
                    entry[i].1.translation = t;
                }
            }
            Some(gltf::animation::util::ReadOutputs::Rotations(rot)) => {
                let values: Vec<[f32; 4]> = rot.into_f32().collect();
                let samples = sampled(&times, &values, interpolation);
                for (time, q) in samples {
                    let i = upsert(entry, time);
                    entry[i].1.rotation_deg = euler_yxz_from_quat(q);
                }
            }
            Some(gltf::animation::util::ReadOutputs::Scales(it)) => {
                let values: Vec<[f32; 3]> = it.collect();
                let samples = sampled(&times, &values, interpolation);
                for (time, s) in samples {
                    let i = upsert(entry, time);
                    entry[i].1.scale = s;
                }
            }
            // Morph targets are not yet a Concinnity asset type; drop quietly.
            Some(gltf::animation::util::ReadOutputs::MorphTargetWeights(_)) | None => continue,
        }
    }

    // Sort each joint's keys by time so the runtime's linear-scan sampling
    // walks them in order.
    let mut sorted_tracks: Vec<ImportedAnimationTrack> = tracks
        .into_iter()
        .map(|(joint, mut keys)| {
            keys.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            ImportedAnimationTrack {
                joint,
                keys: keys
                    .into_iter()
                    .map(|(time, pose)| ImportedKeyframe { time, pose })
                    .collect(),
            }
        })
        .collect();
    sorted_tracks.sort_by_key(|t| t.joint);

    ImportedAnimation {
        name: anim.name().unwrap_or("").to_string(),
        duration: max_time.max(1e-3),
        tracks: sorted_tracks,
    }
}

// Pair each input time with its output value, accounting for glTF's three
// interpolation modes:
//
// - `LINEAR`: 1 value per time, pass-through.
// - `STEP`: 1 value per time, also pass-through: the runtime's linear
//   interp between two equal values is a no-op, and successive different
//   values blend over their gap. This is mildly lossy for step animations
//   (e.g. an on/off blink) but never misaligned at the keyframes.
// - `CUBICSPLINE`: 3 values per time (`in_tangent`, `value`, `out_tangent`).
//   We take only `value` and treat it as `LINEAR`; tangent-driven shapes
//   degrade but joint positions at each keyframe are correct.
//
// Mismatched lengths (a malformed file) return an empty vec rather than
// panicking.
fn sampled<T: Copy>(
    times: &[f32],
    values: &[T],
    interp: gltf::animation::Interpolation,
) -> Vec<(f32, T)> {
    use gltf::animation::Interpolation;
    match interp {
        Interpolation::Linear | Interpolation::Step => {
            if values.len() != times.len() {
                return Vec::new();
            }
            times.iter().copied().zip(values.iter().copied()).collect()
        }
        Interpolation::CubicSpline => {
            if values.len() != times.len() * 3 {
                return Vec::new();
            }
            times
                .iter()
                .copied()
                .enumerate()
                .map(|(i, t)| (t, values[i * 3 + 1]))
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topological_order_keeps_an_already_sorted_chain() {
        let parents = [None, Some(0), Some(1)];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order, vec![0, 1, 2]);
        assert_eq!(remap, vec![0, 1, 2]);
    }

    #[test]
    fn topological_order_sorts_children_after_parents() {
        // A 3-chain authored child-first: joint 0's parent is 1, 1's is 2,
        // 2 is the root. The sorted order must emit 2, then 1, then 0.
        let parents = [Some(1), Some(2), None];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order, vec![2, 1, 0]);
        assert_eq!(remap, vec![2, 1, 0]);
        // Every parent now precedes its child under the remap.
        for (sj, p) in parents.iter().enumerate() {
            if let Some(&p) = p.as_ref() {
                assert!(remap[p] < remap[sj], "parent {p} not before child {sj}");
            }
        }
    }

    #[test]
    fn topological_order_handles_a_forest_with_multiple_roots() {
        // Two independent roots, each with one child.
        let parents = [None, Some(0), None, Some(2)];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order.len(), 4);
        for (sj, p) in parents.iter().enumerate() {
            if let Some(&p) = p.as_ref() {
                assert!(remap[p] < remap[sj]);
            }
        }
    }

    #[test]
    fn topological_order_does_not_drop_joints_in_a_cycle() {
        // A cycle (never valid glTF) must still yield a total order so no
        // joint or its vertex bindings are silently lost.
        let parents = [Some(1), Some(0)];
        let (order, _) = topological_order(&parents);
        assert_eq!(order.len(), 2);
        let mut seen = order.clone();
        seen.sort();
        assert_eq!(seen, vec![0, 1]);
    }

    #[test]
    fn topological_order_treats_an_out_of_range_parent_as_a_root() {
        let parents = [Some(9), Some(0)];
        let (order, remap) = topological_order(&parents);
        assert_eq!(order.len(), 2);
        assert!(remap[0] < remap[1]);
    }

    use gltf::animation::Interpolation;

    #[test]
    fn sampled_linear_pairs_times_with_values() {
        let times = [0.0_f32, 0.5, 1.0];
        let values = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let out = sampled(&times, &values, Interpolation::Linear);
        assert_eq!(out.len(), 3);
        assert_eq!(out[1], (0.5, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn sampled_step_is_pass_through() {
        let times = [0.0_f32, 1.0];
        let values = [[1.0_f32; 3], [2.0_f32; 3]];
        let out = sampled(&times, &values, Interpolation::Step);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn sampled_cubicspline_takes_middle_value_of_each_triplet() {
        // CubicSpline emits 3 values per time: in_tangent, value, out_tangent.
        let times = [0.0_f32, 0.5];
        // Times[0]'s triplet: in=11, val=12, out=13. Times[1]'s: 21, 22, 23.
        let values = [
            [11.0_f32; 3],
            [12.0; 3],
            [13.0; 3],
            [21.0; 3],
            [22.0; 3],
            [23.0; 3],
        ];
        let out = sampled(&times, &values, Interpolation::CubicSpline);
        assert_eq!(out, vec![(0.0, [12.0; 3]), (0.5, [22.0; 3])]);
    }

    #[test]
    fn sampled_with_mismatched_lengths_returns_empty() {
        let times = [0.0_f32, 1.0];
        let values: Vec<[f32; 3]> = vec![[0.0; 3]];
        assert!(sampled(&times, &values, Interpolation::Linear).is_empty());
        assert!(sampled(&times, &values, Interpolation::CubicSpline).is_empty());
    }
}
