// src/gfx/mesh_payload.rs
//
// Canonical vertex type and the binary serialisation format shared between
// the build step (build_mesh.rs writes) and GraphicsSystem (reads).
//
// Format (little-endian):
//   u32  vertex_count
//   vertex_count * 56 bytes   float3 pos + float3 normal + float3 tangent + float3 color + float2 uv (14 x f32)
//   u32  index_count                              // LOD0 indices
//   index_count  * 2 bytes    u16 indices
//   optional LOD trailer
//   4 bytes                   ascii "LODS" magic (absent for legacy / single-LOD payloads)
//   u32  alt_count            // number of additional LODs beyond LOD0
//   alt_count × {
//     f32  switch_distance    // camera-distance threshold (LOD i+1 applies at d >= switch_distance)
//     u32  index_count
//     index_count * 2 bytes   u16 indices
//   }
//
// `deserialise` reads only the LOD0 indices and ignores any trailer, so old
// readers keep working unchanged. `deserialise_with_lods` reads the trailer
// when present and returns the additional LODs alongside LOD0.

// Vertex layout shared by all mesh producers and both GPU backends.
// Repr(C) so it can be cast directly to GPU buffer memory.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Vertex {
    pub pos: [f32; 3],
    // Object-space surface normal, normalised. Transformed to world space in
    // the vertex shader. Used for diffuse lighting in the fragment shader.
    pub normal: [f32; 3],
    // Object-space tangent vector (U direction of the normal map). Transformed
    // to world space in the vertex shader. Used to build the TBN matrix for
    // tangent-space normal mapping.
    pub tangent: [f32; 3],
    pub color: [f32; 3],
    // Texture coordinates in [0, 1] space.  (0,0) is top-left.
    pub uv: [f32; 2],
}

// Interleaved vertex tuple the payload format stores: position, normal,
// tangent, color, uv.
type VertTuple = ([f32; 3], [f32; 3], [f32; 3], [f32; 3], [f32; 2]);

// LOD alternates: (switch_distance, index buffer) pairs (LOD1..N).
type LodAlternates = Vec<(f32, Vec<u16>)>;

// Deserialised static mesh: vertices, LOD0 indices, and LOD alternates.
type DeserialisedStatic = (Vec<Vertex>, Vec<u16>, LodAlternates);

// Deserialised skinned mesh: vertices, indices, and the bind-pose skeleton.
type DeserialisedSkinned = (Vec<SkinnedVertex>, Vec<u16>, Vec<PayloadJoint>);

// Deserialised skinned mesh plus its LOD alternates.
type DeserialisedSkinnedLods = (
    Vec<SkinnedVertex>,
    Vec<u16>,
    Vec<PayloadJoint>,
    LodAlternates,
);

// Serialise vertex and index slices into the packed binary payload format.
// Each vertex tuple is (pos, normal, tangent, color, uv).
pub fn serialise(vertices: &[VertTuple], indices: &[u16]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + vertices.len() * 56 + 4 + indices.len() * 2);
    buf.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    for (pos, normal, tangent, color, uv) in vertices {
        for x in pos
            .iter()
            .chain(normal.iter())
            .chain(tangent.iter())
            .chain(color.iter())
            .chain(uv.iter())
        {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for i in indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

// Magic header for the optional LOD trailer. Absent in legacy payloads so
// `deserialise` keeps working without changes.
const LODS_MAGIC: &[u8; 4] = b"LODS";

// Serialise a multi-LOD mesh payload. `indices` is LOD0; `lod_alternates`
// is the list of additional LODs (LOD1..N), each paired with the
// camera-distance threshold that triggers a switch to it. When
// `lod_alternates` is empty this is byte-identical to the single-LOD
// `serialise` output, so the build can call this unconditionally.
pub fn serialise_with_lods(
    vertices: &[VertTuple],
    indices: &[u16],
    lod_alternates: &[(f32, Vec<u16>)],
) -> Vec<u8> {
    let mut buf = serialise(vertices, indices);
    if lod_alternates.is_empty() {
        return buf;
    }
    buf.extend_from_slice(LODS_MAGIC);
    buf.extend_from_slice(&(lod_alternates.len() as u32).to_le_bytes());
    for (distance, idx) in lod_alternates {
        buf.extend_from_slice(&distance.to_le_bytes());
        buf.extend_from_slice(&(idx.len() as u32).to_le_bytes());
        for i in idx {
            buf.extend_from_slice(&i.to_le_bytes());
        }
    }
    buf
}

// Magic header for the optional baked-heightfield collider trailer. Rides
// after the (optional) LOD trailer on a `heightfield`-generator ProceduralMesh
// payload so the physics terrain collider can read a ready-made height grid
// instead of decoding the source image at runtime. `deserialise` and
// `deserialise_with_lods` stop after the LOD block and ignore these bytes, so
// the render path is unaffected and legacy payloads keep loading unchanged.
const HFLD_MAGIC: &[u8; 4] = b"HFLD";

// A baked heightfield collider grid: `rows` x `cols` world-space heights in
// row-major order (row index increases along +Z, column index along +X),
// matching the vertex order the heightfield mesh generator emits.
pub struct HeightfieldGrid {
    pub rows: usize,
    pub cols: usize,
    pub heights: Vec<f32>,
}

// Serialise a baked-heightfield collider trailer: `"HFLD"` magic, `u32 rows`,
// `u32 cols`, then `rows * cols` little-endian f32 heights in row-major order.
// Appended to a heightfield ProceduralMesh payload after the optional LOD
// trailer.
pub fn serialise_heightfield_trailer(rows: usize, cols: usize, heights: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 4 + 4 + heights.len() * 4);
    buf.extend_from_slice(HFLD_MAGIC);
    buf.extend_from_slice(&(rows as u32).to_le_bytes());
    buf.extend_from_slice(&(cols as u32).to_le_bytes());
    for h in heights {
        buf.extend_from_slice(&h.to_le_bytes());
    }
    buf
}

// Decode the baked-heightfield trailer from a static mesh payload, if present.
// The trailer rides at the very end, so this walks past the vertex, LOD0
// index, and optional LOD blocks positionally before reading the `"HFLD"`
// block. Returns `Ok(None)` for any payload without the trailer (i.e. every
// non-heightfield mesh) so callers can treat absence as "no baked collider".
pub fn deserialise_heightfield(bytes: &[u8]) -> Result<Option<HeightfieldGrid>, String> {
    let mut cur = 0usize;

    let read_u32 = |cur: &mut usize| -> Result<u32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!("unexpected end of mesh payload at offset {}", *cur));
        }
        let v = u32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };
    let skip = |cur: &mut usize, n: usize| -> Result<(), String> {
        let end = cur.checked_add(n).ok_or("mesh payload length overflow")?;
        if end > bytes.len() {
            return Err(format!(
                "mesh payload too short: need {} bytes, have {}",
                end,
                bytes.len()
            ));
        }
        *cur = end;
        Ok(())
    };

    // Vertex block (56 bytes each), then LOD0 indices (2 bytes each).
    let vertex_count = read_u32(&mut cur)? as usize;
    skip(&mut cur, vertex_count * 56)?;
    let index_count = read_u32(&mut cur)? as usize;
    skip(&mut cur, index_count * 2)?;

    // Optional LOD trailer: skip the whole block when present so the cursor
    // lands on the HFLD trailer (if any) that follows it.
    if cur + 4 <= bytes.len() && &bytes[cur..cur + 4] == LODS_MAGIC {
        cur += 4;
        let alt_count = read_u32(&mut cur)? as usize;
        for _ in 0..alt_count {
            skip(&mut cur, 4)?; // switch distance (f32)
            let n = read_u32(&mut cur)? as usize;
            skip(&mut cur, n * 2)?;
        }
    }

    // Optional HFLD trailer.
    if cur + 4 > bytes.len() || &bytes[cur..cur + 4] != HFLD_MAGIC {
        return Ok(None);
    }
    cur += 4;
    let rows = read_u32(&mut cur)? as usize;
    let cols = read_u32(&mut cur)? as usize;
    let count = rows
        .checked_mul(cols)
        .ok_or("heightfield trailer grid size overflow")?;
    let mut heights = Vec::with_capacity(count);
    for _ in 0..count {
        let end = cur + 4;
        if end > bytes.len() {
            return Err(format!(
                "heightfield trailer too short for {} x {} grid",
                rows, cols
            ));
        }
        heights.push(f32::from_le_bytes(bytes[cur..end].try_into().unwrap()));
        cur = end;
    }
    Ok(Some(HeightfieldGrid {
        rows,
        cols,
        heights,
    }))
}

// Vertex layout for skeletally animated meshes. A superset of `Vertex`: the
// same 56-byte static attributes plus four joint indices and four blend
// weights. `repr(C)`, 80 bytes, so it casts directly to a GPU buffer.
//
// The vertex shader skins `pos` / `normal` / `tangent` by blending up to four
// joint matrices: `sum(weights[k] * joint[joints[k]] * v)`. Weights that sum
// to less than 1 leave the remainder un-skinned; the build step normalises
// them so this never happens for authored meshes.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct SkinnedVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
    // Indices into the skeleton's joint array, one per blend weight.
    pub joints: [u16; 4],
    // Blend weights, parallel to `joints`. Normalised at build time.
    pub weights: [f32; 4],
}

// Magic header for the skinned-mesh binary payload. Distinguishes a skinned
// blob from the headerless static `Vertex` format so a mismatched payload
// fails loudly instead of being misread.
const SKINNED_MAGIC: &[u8; 4] = b"SKMV";

// One joint of a skinned mesh's bind-pose skeleton, as stored in the
// compiled payload. Mirrors `assets::skinned_mesh::JointDef` but lives in
// `gfx` so the payload format stays self-contained: the build/runtime
// boundaries convert between the two. Parents must appear before their
// children, so the runtime can walk the array once when building the
// `Skeleton`.
#[derive(Clone, Debug, PartialEq)]
pub struct PayloadJoint {
    pub name: String,
    pub parent: i32,
    pub translation: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
}

// Serialise skinned vertices, indices, and bind-pose skeleton into a packed
// binary payload.
//
// Format (little-endian): `"SKMV"` magic, `u32 vertex_count`,
// `vertex_count * 80` bytes of interleaved `SkinnedVertex` data,
// `u32 index_count`, `index_count * 2` bytes of u16 indices,
// `u32 joint_count`, then `joint_count` joint records, each:
// `u32 name_byte_len`, name UTF-8 bytes, `i32 parent`,
// `f32×3 translation`, `f32×3 rotation_deg`, `f32×3 scale`.
//
// The skeleton block is always present (possibly with `joint_count == 0`),
// so a payload deserialises into a self-contained runtime view, no need
// for the args JSON to carry the skeleton alongside.
//
// Calls [`serialise_skinned_with_lods`] with an empty alternates list, so
// the on-wire format is identical to the legacy single-LOD payload when
// no alternates are present.
#[cfg(test)]
pub fn serialise_skinned(
    vertices: &[SkinnedVertex],
    indices: &[u16],
    joints: &[PayloadJoint],
) -> Vec<u8> {
    serialise_skinned_with_lods(vertices, indices, joints, &[])
}

// Serialise a multi-LOD skinned mesh. The optional LOD trailer rides
// after the joint block: magic `"LODS"`, `u32 alt_count`, then per
// alternate `f32 switch_distance`, `u32 index_count`,
// `index_count * 2` bytes of u16 indices. Empty `lod_alternates` matches
// the single-LOD [`serialise_skinned`] output byte-for-byte.
pub fn serialise_skinned_with_lods(
    vertices: &[SkinnedVertex],
    indices: &[u16],
    joints: &[PayloadJoint],
    lod_alternates: &[(f32, Vec<u16>)],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 4 + vertices.len() * 80 + 4 + indices.len() * 2 + 4);
    buf.extend_from_slice(SKINNED_MAGIC);
    buf.extend_from_slice(&(vertices.len() as u32).to_le_bytes());
    for v in vertices {
        for f in v
            .pos
            .iter()
            .chain(v.normal.iter())
            .chain(v.tangent.iter())
            .chain(v.color.iter())
            .chain(v.uv.iter())
        {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        for j in v.joints {
            buf.extend_from_slice(&j.to_le_bytes());
        }
        for w in v.weights {
            buf.extend_from_slice(&w.to_le_bytes());
        }
    }
    buf.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    for i in indices {
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf.extend_from_slice(&(joints.len() as u32).to_le_bytes());
    for j in joints {
        let name_bytes = j.name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&j.parent.to_le_bytes());
        for x in j
            .translation
            .iter()
            .chain(j.rotation_deg.iter())
            .chain(j.scale.iter())
        {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    if !lod_alternates.is_empty() {
        buf.extend_from_slice(LODS_MAGIC);
        buf.extend_from_slice(&(lod_alternates.len() as u32).to_le_bytes());
        for (distance, idx) in lod_alternates {
            buf.extend_from_slice(&distance.to_le_bytes());
            buf.extend_from_slice(&(idx.len() as u32).to_le_bytes());
            for i in idx {
                buf.extend_from_slice(&i.to_le_bytes());
            }
        }
    }
    buf
}

// Deserialise a packed skinned-mesh payload produced by `serialise_skinned`.
// The returned skeleton lives in the payload; the args JSON no longer needs
// to carry it. The optional LOD trailer is parsed and discarded; callers
// who need LOD alternates should use [`deserialise_skinned_with_lods`].
pub fn deserialise_skinned(bytes: &[u8]) -> Result<DeserialisedSkinned, String> {
    let (v, i, j, _) = deserialise_skinned_with_lods(bytes)?;
    Ok((v, i, j))
}

// Deserialise a packed skinned-mesh payload, also returning any optional
// LOD trailer. Mirrors [`deserialise_with_lods`] for static meshes:
// legacy single-LOD payloads have no trailer and produce an empty
// alternates vec.
pub fn deserialise_skinned_with_lods(bytes: &[u8]) -> Result<DeserialisedSkinnedLods, String> {
    if bytes.len() < 8 || &bytes[0..4] != SKINNED_MAGIC {
        return Err("skinned mesh payload missing SKMV magic header".to_string());
    }
    let mut cur = 4usize;

    let read_u32 = |cur: &mut usize| -> Result<u32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!(
                "unexpected end of skinned mesh payload at offset {}",
                *cur
            ));
        }
        let v = u32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };
    let read_i32 = |cur: &mut usize| -> Result<i32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!(
                "unexpected end of skinned mesh payload at offset {}",
                *cur
            ));
        }
        let v = i32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };
    let read_f32 = |cur: &mut usize| -> Result<f32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!(
                "unexpected end of skinned mesh payload at offset {}",
                *cur
            ));
        }
        let v = f32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };

    let vertex_count = read_u32(&mut cur)? as usize;
    let vertex_bytes = vertex_count * 80;
    if cur + vertex_bytes > bytes.len() {
        return Err(format!(
            "skinned mesh payload too short for {} vertices (need {} bytes, have {})",
            vertex_count,
            cur + vertex_bytes,
            bytes.len()
        ));
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut f = [0f32; 14];
        for x in &mut f {
            let end = cur + 4;
            *x = f32::from_le_bytes(bytes[cur..end].try_into().unwrap());
            cur = end;
        }
        let mut joints = [0u16; 4];
        for j in &mut joints {
            let end = cur + 2;
            *j = u16::from_le_bytes(bytes[cur..end].try_into().unwrap());
            cur = end;
        }
        let mut weights = [0f32; 4];
        for w in &mut weights {
            let end = cur + 4;
            *w = f32::from_le_bytes(bytes[cur..end].try_into().unwrap());
            cur = end;
        }
        vertices.push(SkinnedVertex {
            pos: [f[0], f[1], f[2]],
            normal: [f[3], f[4], f[5]],
            tangent: [f[6], f[7], f[8]],
            color: [f[9], f[10], f[11]],
            uv: [f[12], f[13]],
            joints,
            weights,
        });
    }

    let index_count = read_u32(&mut cur)? as usize;
    let index_bytes = index_count * 2;
    if cur + index_bytes > bytes.len() {
        return Err(format!(
            "skinned mesh payload too short for {} indices (need {} bytes, have {})",
            index_count,
            cur + index_bytes,
            bytes.len()
        ));
    }
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        let end = cur + 2;
        indices.push(u16::from_le_bytes(bytes[cur..end].try_into().unwrap()));
        cur = end;
    }

    let joint_count = read_u32(&mut cur)? as usize;
    let mut joints_out = Vec::with_capacity(joint_count);
    for _ in 0..joint_count {
        let name_len = read_u32(&mut cur)? as usize;
        let name_end = cur + name_len;
        if name_end > bytes.len() {
            return Err(format!(
                "skinned mesh payload too short for joint name (need {} bytes, have {})",
                name_end,
                bytes.len()
            ));
        }
        let name = std::str::from_utf8(&bytes[cur..name_end])
            .map_err(|e| format!("joint name is not valid utf-8: {}", e))?
            .to_string();
        cur = name_end;
        let parent = read_i32(&mut cur)?;
        let mut t = [0f32; 3];
        for x in &mut t {
            *x = read_f32(&mut cur)?;
        }
        let mut r = [0f32; 3];
        for x in &mut r {
            *x = read_f32(&mut cur)?;
        }
        let mut s = [0f32; 3];
        for x in &mut s {
            *x = read_f32(&mut cur)?;
        }
        joints_out.push(PayloadJoint {
            name,
            parent,
            translation: t,
            rotation_deg: r,
            scale: s,
        });
    }

    // Optional LOD trailer (mirrors the static-mesh format): legacy
    // single-LOD payloads end at the joint block; if the next four bytes
    // are the `LODS` magic, the alternates follow.
    let mut alternates: Vec<(f32, Vec<u16>)> = Vec::new();
    if cur + 4 <= bytes.len() && &bytes[cur..cur + 4] == LODS_MAGIC {
        cur += 4;
        let alt_count = read_u32(&mut cur)? as usize;
        alternates.reserve(alt_count);
        for _ in 0..alt_count {
            let distance = read_f32(&mut cur)?;
            let n = read_u32(&mut cur)? as usize;
            let bytes_needed = n * 2;
            if cur + bytes_needed > bytes.len() {
                return Err(format!(
                    "skinned mesh payload too short for {} LOD indices (need {} bytes, have {})",
                    n,
                    cur + bytes_needed,
                    bytes.len()
                ));
            }
            let mut alt: Vec<u16> = Vec::with_capacity(n);
            for _ in 0..n {
                let end = cur + 2;
                alt.push(u16::from_le_bytes(bytes[cur..end].try_into().unwrap()));
                cur = end;
            }
            alternates.push((distance, alt));
        }
    }

    Ok((vertices, indices, joints_out, alternates))
}

// Deserialise a packed payload, also returning any optional LOD trailer.
// Legacy single-LOD payloads have no trailer and produce an empty
// alternates vec; multi-LOD payloads parse the `"LODS"` block after the
// LOD0 indices and return one entry per additional level. The order is
// preserved: `alternates[i]` is LOD `i + 1` and applies at camera
// distance ≥ `alternates[i].0`.
pub fn deserialise_with_lods(bytes: &[u8]) -> Result<DeserialisedStatic, String> {
    let mut cur = 0usize;

    let read_u32 = |cur: &mut usize| -> Result<u32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!("unexpected end of mesh payload at offset {}", *cur));
        }
        let v = u32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };
    let read_f32 = |cur: &mut usize| -> Result<f32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!("unexpected end of mesh payload at offset {}", *cur));
        }
        let v = f32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };

    let vertex_count = read_u32(&mut cur)? as usize;
    let vertex_bytes = vertex_count * 56;
    if cur + vertex_bytes > bytes.len() {
        return Err(format!(
            "mesh payload too short for {} vertices (need {} bytes, have {})",
            vertex_count,
            cur + vertex_bytes,
            bytes.len()
        ));
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut f = [0f32; 14];
        for x in &mut f {
            *x = read_f32(&mut cur)?;
        }
        vertices.push(Vertex {
            pos: [f[0], f[1], f[2]],
            normal: [f[3], f[4], f[5]],
            tangent: [f[6], f[7], f[8]],
            color: [f[9], f[10], f[11]],
            uv: [f[12], f[13]],
        });
    }

    let read_indices = |cur: &mut usize, n: usize| -> Result<Vec<u16>, String> {
        let needed = n * 2;
        if *cur + needed > bytes.len() {
            return Err(format!(
                "mesh payload too short for {} indices (need {} bytes, have {})",
                n,
                *cur + needed,
                bytes.len()
            ));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let end = *cur + 2;
            out.push(u16::from_le_bytes(bytes[*cur..end].try_into().unwrap()));
            *cur = end;
        }
        Ok(out)
    };

    let index_count = read_u32(&mut cur)? as usize;
    let indices = read_indices(&mut cur, index_count)?;

    // Optional LOD trailer. The legacy single-LOD payload ends here; check
    // for the `LODS` magic before reading anything more.
    let mut alternates = Vec::new();
    if cur + 4 <= bytes.len() && &bytes[cur..cur + 4] == LODS_MAGIC {
        cur += 4;
        let alt_count = read_u32(&mut cur)? as usize;
        alternates.reserve(alt_count);
        for _ in 0..alt_count {
            let distance = read_f32(&mut cur)?;
            let n = read_u32(&mut cur)? as usize;
            let alt = read_indices(&mut cur, n)?;
            alternates.push((distance, alt));
        }
    }

    Ok((vertices, indices, alternates))
}

// Deserialise a packed payload back into typed vertex and index vecs (static).
#[cfg(test)]
pub fn deserialise(bytes: &[u8]) -> Result<(Vec<Vertex>, Vec<u16>), String> {
    let mut cur = 0usize;

    let read_u32 = |cur: &mut usize| -> Result<u32, String> {
        let end = *cur + 4;
        if end > bytes.len() {
            return Err(format!("unexpected end of mesh payload at offset {}", *cur));
        }
        let v = u32::from_le_bytes(bytes[*cur..end].try_into().unwrap());
        *cur = end;
        Ok(v)
    };

    let vertex_count = read_u32(&mut cur)? as usize;
    // 14 floats per vertex: float3 pos + float3 normal + float3 tangent + float3 color + float2 uv
    let vertex_bytes = vertex_count * 56;
    if cur + vertex_bytes > bytes.len() {
        return Err(format!(
            "mesh payload too short for {} vertices (need {} bytes, have {})",
            vertex_count,
            cur + vertex_bytes,
            bytes.len()
        ));
    }
    let mut vertices = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let mut f = [0f32; 14];
        for x in &mut f {
            let end = cur + 4;
            *x = f32::from_le_bytes(bytes[cur..end].try_into().unwrap());
            cur = end;
        }
        vertices.push(Vertex {
            pos: [f[0], f[1], f[2]],
            normal: [f[3], f[4], f[5]],
            tangent: [f[6], f[7], f[8]],
            color: [f[9], f[10], f[11]],
            uv: [f[12], f[13]],
        });
    }

    let index_count = read_u32(&mut cur)? as usize;
    let index_bytes = index_count * 2;
    if cur + index_bytes > bytes.len() {
        return Err(format!(
            "mesh payload too short for {} indices (need {} bytes, have {})",
            index_count,
            cur + index_bytes,
            bytes.len()
        ));
    }
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        let end = cur + 2;
        indices.push(u16::from_le_bytes(bytes[cur..end].try_into().unwrap()));
        cur = end;
    }

    Ok((vertices, indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_skinned() -> Vec<SkinnedVertex> {
        vec![
            SkinnedVertex {
                pos: [1.0, 2.0, 3.0],
                normal: [0.0, 1.0, 0.0],
                tangent: [1.0, 0.0, 0.0],
                color: [0.5, 0.6, 0.7],
                uv: [0.25, 0.75],
                joints: [0, 1, 2, 3],
                weights: [0.5, 0.3, 0.2, 0.0],
            },
            SkinnedVertex {
                pos: [-4.0, 5.0, -6.0],
                normal: [0.0, 0.0, 1.0],
                tangent: [0.0, 1.0, 0.0],
                color: [1.0, 1.0, 1.0],
                uv: [0.0, 1.0],
                joints: [7, 0, 0, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        ]
    }

    fn sample_skeleton() -> Vec<PayloadJoint> {
        vec![
            PayloadJoint {
                name: "root".to_string(),
                parent: -1,
                translation: [0.0, 0.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
            PayloadJoint {
                name: "tip".to_string(),
                parent: 0,
                translation: [0.0, 1.0, 0.0],
                rotation_deg: [0.0, 0.0, 0.0],
                scale: [1.0, 1.0, 1.0],
            },
        ]
    }

    #[test]
    fn skinned_roundtrip_preserves_data() {
        let verts = sample_skinned();
        let idxs = vec![0u16, 1, 0];
        let skel = sample_skeleton();
        let bytes = serialise_skinned(&verts, &idxs, &skel);
        let (out_v, out_i, out_s) = deserialise_skinned(&bytes).expect("deserialise");
        assert_eq!(out_v, verts);
        assert_eq!(out_i, idxs);
        assert_eq!(out_s, skel);
    }

    #[test]
    fn skinned_roundtrip_with_empty_skeleton_keeps_trailer_present() {
        // joint_count == 0 still emits the u32 length prefix, so the format
        // is uniform regardless of whether the asset declared a skeleton.
        let verts = sample_skinned();
        let idxs = vec![0u16, 1, 0];
        let bytes = serialise_skinned(&verts, &idxs, &[]);
        let (out_v, out_i, out_s) = deserialise_skinned(&bytes).expect("deserialise");
        assert_eq!(out_v, verts);
        assert_eq!(out_i, idxs);
        assert!(out_s.is_empty());
    }

    #[test]
    fn skinned_payload_size_is_predictable() {
        // magic + vert_count + 2*vertex + idx_count + 3*idx + joint_count
        // + per-joint: name_len + name + parent + 3*vec3.
        let skel = sample_skeleton();
        let bytes = serialise_skinned(&sample_skinned(), &[0u16, 1, 0], &skel);
        let per_joint = skel
            .iter()
            .map(|j| 4 + j.name.len() + 4 + 12 + 12 + 12)
            .sum::<usize>();
        assert_eq!(bytes.len(), 4 + 4 + 2 * 80 + 4 + 3 * 2 + 4 + per_joint);
    }

    #[test]
    fn vertex_layout_matches_msl() {
        // `Vertex` is read through a pointer by the RT skinning kernel
        // (`VtxOut` in rt_skin.metal: five packed_float* fields, 56-byte
        // stride) and as the static RT vertex format, so the field offsets
        // must match exactly. The main/shadow passes consume it through a
        // vertex descriptor declaring the same 0/12/24/36/48 attribute offsets.
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<Vertex>(), 56);
        assert_eq!(offset_of!(Vertex, pos), 0);
        assert_eq!(offset_of!(Vertex, normal), 12);
        assert_eq!(offset_of!(Vertex, tangent), 24);
        assert_eq!(offset_of!(Vertex, color), 36);
        assert_eq!(offset_of!(Vertex, uv), 48);
    }

    #[test]
    fn skinned_vertex_layout_matches_msl() {
        // `SkinnedVertex` is read through a pointer by the RT skinning kernel
        // (`SkinnedVtxIn` in rt_skin.metal), whose packed_float* + ushort[4] +
        // packed_float4 fields must line up byte-for-byte with this 80-byte
        // struct. The main/shadow skinned passes consume it through a vertex
        // descriptor declaring the same attribute offsets.
        use std::mem::{offset_of, size_of};
        assert_eq!(size_of::<SkinnedVertex>(), 80);
        assert_eq!(offset_of!(SkinnedVertex, pos), 0);
        assert_eq!(offset_of!(SkinnedVertex, normal), 12);
        assert_eq!(offset_of!(SkinnedVertex, tangent), 24);
        assert_eq!(offset_of!(SkinnedVertex, color), 36);
        assert_eq!(offset_of!(SkinnedVertex, uv), 48);
        assert_eq!(offset_of!(SkinnedVertex, joints), 56);
        assert_eq!(offset_of!(SkinnedVertex, weights), 64);
    }

    #[test]
    fn deserialise_skinned_rejects_missing_magic() {
        // The static payload format has no magic header, so feeding one in
        // must be rejected rather than silently misread.
        let static_bytes = serialise(&[([0.0; 3], [0.0; 3], [0.0; 3], [1.0; 3], [0.0; 2])], &[]);
        assert!(deserialise_skinned(&static_bytes).is_err());
    }

    fn sample_static_verts() -> Vec<VertTuple> {
        vec![
            (
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0; 3],
                [0.0, 0.0],
            ),
            (
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0; 3],
                [1.0, 0.0],
            ),
            (
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0; 3],
                [0.0, 1.0],
            ),
        ]
    }

    #[test]
    fn serialise_with_no_lods_matches_legacy_format() {
        let verts = sample_static_verts();
        let idx = vec![0u16, 1, 2];
        let legacy = serialise(&verts, &idx);
        let with_lods = serialise_with_lods(&verts, &idx, &[]);
        assert_eq!(legacy, with_lods, "no alternates → no trailer bytes");
    }

    #[test]
    fn lod_trailer_roundtrip_preserves_distances_and_indices() {
        let verts = sample_static_verts();
        let lod0 = vec![0u16, 1, 2];
        let alternates = vec![(8.0_f32, vec![0u16, 2, 1]), (25.0_f32, vec![0u16, 1, 2])];
        let bytes = serialise_with_lods(&verts, &lod0, &alternates);
        let (out_v, out_idx, out_alts) = deserialise_with_lods(&bytes).expect("deserialise");
        assert_eq!(out_v.len(), verts.len());
        assert_eq!(out_idx, lod0);
        assert_eq!(out_alts.len(), 2);
        assert_eq!(out_alts[0].0, 8.0);
        assert_eq!(out_alts[0].1, vec![0u16, 2, 1]);
        assert_eq!(out_alts[1].0, 25.0);
        assert_eq!(out_alts[1].1, vec![0u16, 1, 2]);
    }

    #[test]
    fn legacy_payload_has_no_alternates() {
        // A payload written by the single-LOD `serialise` must deserialise via
        // `deserialise_with_lods` with an empty alternates vec: backward
        // compatibility for every existing on-disk blob.
        let verts = sample_static_verts();
        let idx = vec![0u16, 1, 2];
        let bytes = serialise(&verts, &idx);
        let (_, _, alts) = deserialise_with_lods(&bytes).expect("deserialise");
        assert!(alts.is_empty());
    }

    #[test]
    fn heightfield_trailer_roundtrips_without_lods() {
        let verts = sample_static_verts();
        let idx = vec![0u16, 1, 2];
        let heights = vec![0.0f32, 1.0, 2.0, 3.0];
        let mut bytes = serialise_with_lods(&verts, &idx, &[]);
        bytes.extend_from_slice(&serialise_heightfield_trailer(2, 2, &heights));

        let grid = deserialise_heightfield(&bytes)
            .expect("parse")
            .expect("trailer present");
        assert_eq!(grid.rows, 2);
        assert_eq!(grid.cols, 2);
        assert_eq!(grid.heights, heights);

        // The render path ignores the trailer entirely.
        let (out_v, out_i, out_alts) = deserialise_with_lods(&bytes).expect("render path");
        assert_eq!(out_v.len(), verts.len());
        assert_eq!(out_i, idx);
        assert!(out_alts.is_empty());
    }

    #[test]
    fn heightfield_trailer_roundtrips_after_lod_trailer() {
        let verts = sample_static_verts();
        let lod0 = vec![0u16, 1, 2];
        let alternates = vec![(8.0_f32, vec![0u16, 2, 1]), (25.0_f32, vec![0u16, 1, 2])];
        let heights = vec![-1.0f32, 0.5, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0];
        let mut bytes = serialise_with_lods(&verts, &lod0, &alternates);
        bytes.extend_from_slice(&serialise_heightfield_trailer(3, 3, &heights));

        // Both trailers parse independently from the same payload.
        let (_, out_i, out_alts) = deserialise_with_lods(&bytes).expect("render path");
        assert_eq!(out_i, lod0);
        assert_eq!(out_alts.len(), 2);

        let grid = deserialise_heightfield(&bytes)
            .expect("parse")
            .expect("trailer present");
        assert_eq!((grid.rows, grid.cols), (3, 3));
        assert_eq!(grid.heights, heights);
    }

    #[test]
    fn no_heightfield_trailer_returns_none() {
        let verts = sample_static_verts();
        let bytes = serialise_with_lods(&verts, &[0u16, 1, 2], &[(10.0, vec![0u16, 2, 1])]);
        assert!(deserialise_heightfield(&bytes).expect("parse").is_none());
    }

    #[test]
    fn legacy_deserialise_still_works_on_multi_lod_payload() {
        // The legacy `deserialise` reader must keep ignoring the LODS
        // trailer so any code path that didn't migrate yet still loads
        // LOD0 from a multi-LOD payload.
        let verts = sample_static_verts();
        let lod0 = vec![0u16, 1, 2];
        let bytes = serialise_with_lods(&verts, &lod0, &[(10.0, vec![0u16, 2, 1])]);
        let (out_v, out_idx) = deserialise(&bytes).expect("legacy reader");
        assert_eq!(out_v.len(), verts.len());
        assert_eq!(out_idx, lod0);
    }
}
