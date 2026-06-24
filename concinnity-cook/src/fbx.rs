// src/build/fbx.rs
//
// Imports a binary FBX scene (v7.4 / v7.5) into engine-native assets. The
// scene is reduced to three flat lists: materials (with their resolved texture
// paths), geometry primitives (one per material group within a mesh, in
// geometry-local space), and props (one per scene node that carries geometry,
// holding the node's world transform decomposed into translation / Euler
// rotation / scale and the indices of its primitives).
//
// Only what the renderer needs is extracted: positions, the first UV set, and
// per-polygon material assignment. Normals and tangents are recomputed by the
// mesh build, so FBX normal layers are ignored. Geometry is kept in local space
// and the node transform travels on the prop, mirroring the glTF import.
//
// FBX texture slots are mapped from the connection property they bind to:
// DiffuseColor -> albedo, NormalMap -> normal, SpecularColor -> packed ORM
// (occlusion/roughness/metalness), EmissiveColor -> emissive.

use std::collections::HashMap;
use std::path::Path;

use fbxcel::low::v7400::AttributeValue;
use fbxcel::tree::any::AnyTree;
use fbxcel::tree::v7400::NodeHandle;

use crate::assets::VertexData;
use crate::gfx::skinning::{IDENTITY, Mat4, decompose, euler_yxz_from_quat, mat4_mul};

// Neutral grey vertex colour so imported geometry takes the material albedo
// unmodified, matching the glTF and OBJ importers.
const NEUTRAL_COLOR: [f32; 3] = [0.75, 0.74, 0.72];

// A material with its scalar factors and resolved on-disk texture paths.
pub struct FbxMaterial {
    pub name: String,
    pub albedo: Option<String>,
    pub normal: Option<String>,
    pub orm: Option<String>,
    pub emissive: Option<String>,
    pub diffuse: [f32; 3],
    pub emissive_factor: [f32; 3],
    // Surface opacity in [0, 1]; 1 = fully opaque. Read from the FBX Opacity /
    // TransparencyFactor property (glass/windows export < 1). 1.0 when absent.
    pub opacity: f32,
}

// One material group of a mesh: geometry-local vertices and triangle indices.
pub struct FbxPrimitive {
    pub vertices: Vec<VertexData>,
    pub indices: Vec<u32>,
    pub material: Option<usize>,
}

// A scene node carrying geometry: a world transform plus the primitives drawn
// at that transform.
pub struct FbxProp {
    pub name: String,
    pub position: [f32; 3],
    pub rotation_deg: [f32; 3],
    pub scale: [f32; 3],
    pub primitives: Vec<usize>,
}

// The fully extracted scene.
pub struct FbxScene {
    pub materials: Vec<FbxMaterial>,
    pub primitives: Vec<FbxPrimitive>,
    pub props: Vec<FbxProp>,
    // World-space bounding box (min, max) of all geometry, for framing a camera.
    pub aabb: Option<([f32; 3], [f32; 3])>,
}

// Vertex count of a primitive, for deciding u16 chunk splitting.
pub fn primitive_vertex_count(scene: &FbxScene, index: u32) -> Option<usize> {
    scene
        .primitives
        .get(index as usize)
        .map(|p| p.vertices.len())
}

// Raw geometry of a primitive (clones out of the parsed scene), in the same
// `(Vec<VertexData>, Vec<u32>)` shape the glTF importer produces so the shared
// `split_into_u16_chunks` chunker applies unchanged.
pub fn read_primitive_geometry(
    scene: &FbxScene,
    index: u32,
) -> Result<(Vec<VertexData>, Vec<u32>), String> {
    scene
        .primitives
        .get(index as usize)
        .map(|p| (p.vertices.clone(), p.indices.clone()))
        .ok_or_else(|| format!("fbx primitive_index {index} out of range"))
}

fn attr_i64(a: &AttributeValue) -> Option<i64> {
    match a {
        AttributeValue::I64(v) => Some(*v),
        AttributeValue::I32(v) => Some(*v as i64),
        _ => None,
    }
}

fn attr_str(a: &AttributeValue) -> Option<&str> {
    match a {
        AttributeValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

fn attr_f64(a: &AttributeValue) -> Option<f64> {
    match a {
        AttributeValue::F64(v) => Some(*v),
        AttributeValue::F32(v) => Some(*v as f64),
        AttributeValue::I32(v) => Some(*v as f64),
        AttributeValue::I64(v) => Some(*v as f64),
        _ => None,
    }
}

fn arr_f64<'a>(n: &NodeHandle<'a>) -> Option<&'a [f64]> {
    match n.attributes().first()? {
        AttributeValue::ArrF64(v) => Some(v),
        _ => None,
    }
}

fn arr_i32<'a>(n: &NodeHandle<'a>) -> Option<&'a [i32]> {
    match n.attributes().first()? {
        AttributeValue::ArrI32(v) => Some(v),
        _ => None,
    }
}

// First string attribute of a named child node (e.g. RelativeFilename).
fn child_str<'a>(n: &NodeHandle<'a>, name: &str) -> Option<&'a str> {
    n.first_child_by_name(name)
        .and_then(|c| c.attributes().first().and_then(attr_str))
}

// Object id (first attribute) and clean name (second attribute, trimmed at the
// FBX `name\0\u{1}Class` separator).
fn object_id(n: &NodeHandle) -> Option<i64> {
    n.attributes().first().and_then(attr_i64)
}

fn object_name(n: &NodeHandle) -> String {
    n.attributes()
        .get(1)
        .and_then(attr_str)
        .map(|s| s.split('\u{0}').next().unwrap_or(s).to_string())
        .unwrap_or_default()
}

// Read a `Properties70` P entry's three numeric values (x, y, z at attribute
// indices 4..7).
fn prop_vec3(p70: &NodeHandle, name: &str) -> Option<[f64; 3]> {
    for p in p70.children_by_name("P") {
        let a = p.attributes();
        if a.first().and_then(attr_str) == Some(name) {
            return Some([
                a.get(4).and_then(attr_f64)?,
                a.get(5).and_then(attr_f64)?,
                a.get(6).and_then(attr_f64)?,
            ]);
        }
    }
    None
}

// Read a `Properties70` P entry's single numeric value (at attribute index 4).
fn prop_scalar(p70: &NodeHandle, name: &str) -> Option<f64> {
    for p in p70.children_by_name("P") {
        let a = p.attributes();
        if a.first().and_then(attr_str) == Some(name) {
            return a.get(4).and_then(attr_f64);
        }
    }
    None
}

fn translate(t: [f64; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[3][0] = t[0] as f32;
    m[3][1] = t[1] as f32;
    m[3][2] = t[2] as f32;
    m
}

fn scale_mat(s: [f64; 3]) -> Mat4 {
    let mut m = IDENTITY;
    m[0][0] = s[0] as f32;
    m[1][1] = s[1] as f32;
    m[2][2] = s[2] as f32;
    m
}

fn rot_x(deg: f64) -> Mat4 {
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let mut m = IDENTITY;
    m[1][1] = c;
    m[1][2] = s;
    m[2][1] = -s;
    m[2][2] = c;
    m
}

fn rot_y(deg: f64) -> Mat4 {
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let mut m = IDENTITY;
    m[0][0] = c;
    m[0][2] = -s;
    m[2][0] = s;
    m[2][2] = c;
    m
}

fn rot_z(deg: f64) -> Mat4 {
    let (s, c) = (deg as f32).to_radians().sin_cos();
    let mut m = IDENTITY;
    m[0][0] = c;
    m[0][1] = s;
    m[1][0] = -s;
    m[1][1] = c;
    m
}

// Compose an FBX node transform T * R * S. FBX `eEulerXYZ` applies X then Y
// then Z, which is Rz * Ry * Rx for column vectors.
fn trs_matrix(t: [f64; 3], r_deg: [f64; 3], s: [f64; 3]) -> Mat4 {
    let r = mat4_mul(rot_z(r_deg[2]), mat4_mul(rot_y(r_deg[1]), rot_x(r_deg[0])));
    mat4_mul(translate(t), mat4_mul(r, scale_mat(s)))
}

// (node-local matrix, geometric-offset matrix) for a Model node.
fn local_matrices(model: &NodeHandle) -> (Mat4, Mat4) {
    let Some(p70) = model.first_child_by_name("Properties70") else {
        return (IDENTITY, IDENTITY);
    };
    let t = prop_vec3(&p70, "Lcl Translation").unwrap_or([0.0, 0.0, 0.0]);
    let r = prop_vec3(&p70, "Lcl Rotation").unwrap_or([0.0, 0.0, 0.0]);
    let s = prop_vec3(&p70, "Lcl Scaling").unwrap_or([1.0, 1.0, 1.0]);
    let gt = prop_vec3(&p70, "GeometricTranslation").unwrap_or([0.0, 0.0, 0.0]);
    let gr = prop_vec3(&p70, "GeometricRotation").unwrap_or([0.0, 0.0, 0.0]);
    let gs = prop_vec3(&p70, "GeometricScaling").unwrap_or([1.0, 1.0, 1.0]);
    (trs_matrix(t, r, s), trs_matrix(gt, gr, gs))
}

// World matrix of a model, walking the parent chain (depth is shallow in
// practice; most nodes parent directly to the scene root id 0).
fn world_matrix(id: i64, locals: &HashMap<i64, (Mat4, Mat4)>, parents: &HashMap<i64, i64>) -> Mat4 {
    let local = locals.get(&id).map(|x| x.0).unwrap_or(IDENTITY);
    match parents.get(&id) {
        Some(&p) if p != 0 && locals.contains_key(&p) => {
            mat4_mul(world_matrix(p, locals, parents), local)
        }
        _ => local,
    }
}

// Resolve an FBX texture's RelativeFilename to a path under the FBX directory.
fn resolve_texture_path(tex: &NodeHandle, fbx_dir: &Path) -> Option<String> {
    let rel = child_str(tex, "RelativeFilename").or_else(|| child_str(tex, "FileName"))?;
    let rel = rel.replace('\\', "/");
    let relp = Path::new(&rel);
    let joined = if relp.is_absolute() {
        // Absolute author-machine path: keep only the file name under the local dir.
        fbx_dir.join(relp.file_name().map(Path::new).unwrap_or(relp))
    } else {
        fbx_dir.join(relp)
    };
    Some(joined.to_string_lossy().into_owned())
}

// Parse a binary FBX file into an [`FbxScene`].
pub fn parse_fbx(path: &str) -> Result<FbxScene, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("could not open '{path}': {e}"))?;
    let reader = std::io::BufReader::new(file);
    let tree = match AnyTree::from_seekable_reader(reader)
        .map_err(|e| format!("'{path}': not a valid FBX file: {e}"))?
    {
        AnyTree::V7400(_ver, tree, _footer) => tree,
        _ => return Err(format!("'{path}': unsupported FBX version")),
    };

    let root = tree.root();
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| format!("'{path}': FBX has no Objects section"))?;

    // Index objects by id and type.
    let mut geom_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut model_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut material_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    let mut texture_by_id: HashMap<i64, NodeHandle> = HashMap::new();
    for c in objects.children() {
        let Some(id) = object_id(&c) else { continue };
        match c.name() {
            "Geometry" => {
                geom_by_id.insert(id, c);
            }
            "Model" => {
                model_by_id.insert(id, c);
            }
            "Material" => {
                material_by_id.insert(id, c);
            }
            "Texture" => {
                texture_by_id.insert(id, c);
            }
            _ => {}
        }
    }

    // Connections: object-object (hierarchy / geometry / material) and
    // object-property (texture -> material slot).
    let mut model_parent: HashMap<i64, i64> = HashMap::new();
    let mut model_geometry: HashMap<i64, i64> = HashMap::new();
    let mut model_materials: HashMap<i64, Vec<i64>> = HashMap::new();
    let mut mat_textures: HashMap<i64, Vec<(i64, String)>> = HashMap::new();
    if let Some(conns) = root.first_child_by_name("Connections") {
        for c in conns.children_by_name("C") {
            let a = c.attributes();
            let ty = a.first().and_then(attr_str).unwrap_or("");
            let (Some(child), Some(parent)) =
                (a.get(1).and_then(attr_i64), a.get(2).and_then(attr_i64))
            else {
                continue;
            };
            match ty {
                "OO" => {
                    if model_by_id.contains_key(&child) {
                        model_parent.insert(child, parent);
                    }
                    if geom_by_id.contains_key(&child) && model_by_id.contains_key(&parent) {
                        model_geometry.insert(parent, child);
                    }
                    if material_by_id.contains_key(&child) && model_by_id.contains_key(&parent) {
                        model_materials.entry(parent).or_default().push(child);
                    }
                }
                "OP" if texture_by_id.contains_key(&child)
                    && material_by_id.contains_key(&parent) =>
                {
                    let prop = a.get(3).and_then(attr_str).unwrap_or("").to_string();
                    mat_textures.entry(parent).or_default().push((child, prop));
                }
                _ => {}
            }
        }
    }

    let fbx_dir = Path::new(path)
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();

    // Materials, in object-declaration order (deterministic).
    let mut material_index: HashMap<i64, usize> = HashMap::new();
    let mut materials: Vec<FbxMaterial> = Vec::new();
    for c in objects.children().filter(|c| c.name() == "Material") {
        let Some(id) = object_id(&c) else { continue };
        let p70 = c.first_child_by_name("Properties70");
        let diffuse = p70
            .and_then(|p| prop_vec3(&p, "DiffuseColor"))
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .unwrap_or([1.0, 1.0, 1.0]);
        let emissive_factor = p70
            .and_then(|p| prop_vec3(&p, "EmissiveColor").or_else(|| prop_vec3(&p, "Emissive")))
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .unwrap_or([0.0, 0.0, 0.0]);
        // Opacity: `Opacity` (1 = opaque) preferred; else `TransparencyFactor`
        // (0 = opaque, so invert). Absent -> fully opaque.
        let opacity = p70
            .and_then(|p| {
                prop_scalar(&p, "Opacity")
                    .or_else(|| prop_scalar(&p, "TransparencyFactor").map(|t| 1.0 - t))
            })
            .map(|v| v.clamp(0.0, 1.0) as f32)
            .unwrap_or(1.0);

        let mut mat = FbxMaterial {
            name: object_name(&c),
            albedo: None,
            normal: None,
            orm: None,
            emissive: None,
            diffuse,
            emissive_factor,
            opacity,
        };
        if let Some(texs) = mat_textures.get(&id) {
            for (tex_id, prop) in texs {
                let Some(tex) = texture_by_id.get(tex_id) else {
                    continue;
                };
                let path = resolve_texture_path(tex, &fbx_dir);
                match prop.as_str() {
                    "DiffuseColor" => mat.albedo = path,
                    "NormalMap" => mat.normal = path,
                    "SpecularColor" => mat.orm = path,
                    "EmissiveColor" => mat.emissive = path,
                    _ => {}
                }
            }
        }
        material_index.insert(id, materials.len());
        materials.push(mat);
    }

    // Per-model local matrices (computed once for the world walk).
    let locals: HashMap<i64, (Mat4, Mat4)> = model_by_id
        .iter()
        .map(|(id, h)| (*id, local_matrices(h)))
        .collect();

    // Props + primitives, iterating models in object order for a stable
    // primitive index space (desugar and `cn add` must agree).
    let mut primitives: Vec<FbxPrimitive> = Vec::new();
    let mut props: Vec<FbxProp> = Vec::new();
    let mut aabb: Option<([f32; 3], [f32; 3])> = None;
    for model in objects.children().filter(|c| c.name() == "Model") {
        let Some(model_id) = object_id(&model) else {
            continue;
        };
        let Some(geom_id) = model_geometry.get(&model_id) else {
            continue;
        };
        let Some(geom) = geom_by_id.get(geom_id) else {
            continue;
        };

        let node_world = world_matrix(model_id, &locals, &model_parent);
        let geometric = locals.get(&model_id).map(|x| x.1).unwrap_or(IDENTITY);
        let mesh_world = mat4_mul(node_world, geometric);
        let (position, quat, scale) = decompose(mesh_world);
        let rotation_deg = euler_yxz_from_quat(quat);

        let local_mats: Vec<Option<usize>> = model_materials
            .get(&model_id)
            .map(|ids| ids.iter().map(|m| material_index.get(m).copied()).collect())
            .unwrap_or_default();

        let groups = extract_geometry_groups(geom, &local_mats);
        let mut prim_indices = Vec::new();
        for (material, vertices, indices) in groups {
            if vertices.is_empty() || indices.is_empty() {
                continue;
            }
            expand_aabb(&mut aabb, &vertices, mesh_world);
            prim_indices.push(primitives.len());
            primitives.push(FbxPrimitive {
                vertices,
                indices,
                material,
            });
        }
        if prim_indices.is_empty() {
            continue;
        }
        props.push(FbxProp {
            name: object_name(&model),
            position,
            rotation_deg,
            scale,
            primitives: prim_indices,
        });
    }

    Ok(FbxScene {
        materials,
        primitives,
        props,
        aabb,
    })
}

// Transform a point by a column-major matrix.
fn transform_point(m: Mat4, p: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

// Expand `aabb` to include a primitive's vertices after `world` transform. The
// primitive's local min/max is transformed at all eight corners so any rotation
// or non-uniform scale is captured.
fn expand_aabb(aabb: &mut Option<([f32; 3], [f32; 3])>, vertices: &[VertexData], world: Mat4) {
    if vertices.is_empty() {
        return;
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in vertices {
        for i in 0..3 {
            lo[i] = lo[i].min(v.pos[i]);
            hi[i] = hi[i].max(v.pos[i]);
        }
    }
    let corners = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ];
    for c in corners {
        let w = transform_point(world, c);
        match aabb {
            None => *aabb = Some((w, w)),
            Some((min, max)) => {
                for i in 0..3 {
                    min[i] = min[i].min(w[i]);
                    max[i] = max[i].max(w[i]);
                }
            }
        }
    }
}

// Decode the control-point index of a polygon-vertex entry. The last vertex of
// each polygon is stored as the bitwise complement of its index.
fn decode_pvi(raw: i32) -> (i32, bool) {
    if raw < 0 { (!raw, true) } else { (raw, false) }
}

// Look up a polygon-vertex UV, flipping V from FBX bottom-up to top-down.
fn lookup_uv(uv: Option<&[f64]>, indexed: bool, uv_index: Option<&[i32]>, pv: usize) -> [f32; 2] {
    let Some(uv) = uv else { return [0.0, 0.0] };
    let k = if indexed {
        uv_index
            .and_then(|ui| ui.get(pv))
            .copied()
            .unwrap_or(0)
            .max(0) as usize
    } else {
        pv
    };
    let u = uv.get(k * 2).copied().unwrap_or(0.0) as f32;
    let v = uv.get(k * 2 + 1).copied().unwrap_or(0.0) as f32;
    [u, 1.0 - v]
}

// Split a geometry into per-material primitives: triangulate polygons (fan) and
// deduplicate vertices by (control point, UV index) so UV seams stay sharp.
fn extract_geometry_groups(
    geom: &NodeHandle,
    local_mats: &[Option<usize>],
) -> Vec<(Option<usize>, Vec<VertexData>, Vec<u32>)> {
    let positions = geom
        .first_child_by_name("Vertices")
        .as_ref()
        .and_then(arr_f64);
    let pvi = geom
        .first_child_by_name("PolygonVertexIndex")
        .as_ref()
        .and_then(arr_i32);
    let (Some(positions), Some(pvi)) = (positions, pvi) else {
        return Vec::new();
    };

    // First texture UV layer (prefer the one named "TextureUV").
    let uv_layer = geom
        .children_by_name("LayerElementUV")
        .find(|l| child_str(l, "Name") == Some("TextureUV"))
        .or_else(|| geom.children_by_name("LayerElementUV").next());
    let (uv, uv_index, uv_indexed) = match uv_layer {
        Some(l) => (
            l.first_child_by_name("UV").as_ref().and_then(arr_f64),
            l.first_child_by_name("UVIndex").as_ref().and_then(arr_i32),
            child_str(&l, "ReferenceInformationType") == Some("IndexToDirect"),
        ),
        None => (None, None, false),
    };

    // Per-polygon material assignment.
    let mat_layer = geom.first_child_by_name("LayerElementMaterial");
    let mat_all_same = mat_layer
        .map(|l| child_str(&l, "MappingInformationType") == Some("AllSame"))
        .unwrap_or(true);
    let mat_per_poly = mat_layer.and_then(|l| {
        l.first_child_by_name("Materials")
            .as_ref()
            .and_then(arr_i32)
    });

    type Group = (HashMap<(i32, i32), u32>, Vec<VertexData>, Vec<u32>);
    let mut groups: HashMap<Option<usize>, Group> = HashMap::new();
    let mut corners: Vec<(i32, usize)> = Vec::new();
    let mut poly = 0usize;

    for (pv, &raw) in pvi.iter().enumerate() {
        let (cp, end) = decode_pvi(raw);
        corners.push((cp, pv));
        if !end {
            continue;
        }

        let local_mat = if mat_all_same {
            mat_per_poly.and_then(|m| m.first()).copied()
        } else {
            mat_per_poly.and_then(|m| m.get(poly)).copied()
        };
        let material = match local_mat {
            Some(li) if li >= 0 => local_mats.get(li as usize).copied().flatten(),
            _ => None,
        };

        let group = groups
            .entry(material)
            .or_insert_with(|| (HashMap::new(), Vec::new(), Vec::new()));
        let mut corner_idx: Vec<u32> = Vec::with_capacity(corners.len());
        for &(cp, pv) in &corners {
            let uv_key = if uv_indexed {
                uv_index.and_then(|ui| ui.get(pv)).copied().unwrap_or(0)
            } else {
                pv as i32
            };
            let key = (cp, uv_key);
            let out = if let Some(&i) = group.0.get(&key) {
                i
            } else {
                let c = (cp as usize) * 3;
                let pos = [
                    positions.get(c).copied().unwrap_or(0.0) as f32,
                    positions.get(c + 1).copied().unwrap_or(0.0) as f32,
                    positions.get(c + 2).copied().unwrap_or(0.0) as f32,
                ];
                let vd = VertexData {
                    pos,
                    color: NEUTRAL_COLOR,
                    uv: lookup_uv(uv, uv_indexed, uv_index, pv),
                };
                let i = group.1.len() as u32;
                group.1.push(vd);
                group.0.insert(key, i);
                i
            };
            corner_idx.push(out);
        }
        // Fan triangulation.
        for k in 1..corner_idx.len().saturating_sub(1) {
            group.2.push(corner_idx[0]);
            group.2.push(corner_idx[k]);
            group.2.push(corner_idx[k + 1]);
        }

        corners.clear();
        poly += 1;
    }

    // Deterministic primitive order: by material index, untextured group last.
    let mut out: Vec<(Option<usize>, Vec<VertexData>, Vec<u32>)> =
        groups.into_iter().map(|(m, (_, v, i))| (m, v, i)).collect();
    out.sort_by_key(|(m, _, _)| m.map(|x| x as i64).unwrap_or(i64::MAX));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_pvi_marks_polygon_end() {
        assert_eq!(decode_pvi(5), (5, false));
        // ~5 == -6 marks the last vertex of a polygon and decodes back to 5.
        assert_eq!(decode_pvi(!5), (5, true));
        assert_eq!(decode_pvi(0), (0, false));
        assert_eq!(decode_pvi(!0), (0, true));
    }

    #[test]
    fn lookup_uv_flips_v() {
        let uv = [0.25f64, 0.75];
        // Direct mapping, pv 0 -> (0.25, 1 - 0.75).
        let out = lookup_uv(Some(&uv), false, None, 0);
        assert!((out[0] - 0.25).abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn lookup_uv_indexed() {
        let uv = [0.0f64, 0.0, 0.5, 0.5];
        let idx = [1i32];
        // IndexToDirect: pv 0 -> uv[idx[0]=1] = (0.5, 0.5) -> v flipped to 0.5.
        let out = lookup_uv(Some(&uv), true, Some(&idx), 0);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
    }

    // rot_x(90) sends +Y to +Z (column-major, right-handed).
    #[test]
    fn rot_x_90_maps_y_to_z() {
        let m = rot_x(90.0);
        // Column 1 is the image of the Y basis vector.
        assert!((m[1][1]).abs() < 1e-6);
        assert!((m[1][2] - 1.0).abs() < 1e-6);
    }

    // A pure translation decomposes back to the same position.
    #[test]
    fn trs_translation_round_trips() {
        let m = trs_matrix([3.0, -2.0, 5.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let (t, _q, s) = decompose(m);
        assert!((t[0] - 3.0).abs() < 1e-4);
        assert!((t[1] + 2.0).abs() < 1e-4);
        assert!((t[2] - 5.0).abs() < 1e-4);
        assert!((s[0] - 1.0).abs() < 1e-4);
    }
}
