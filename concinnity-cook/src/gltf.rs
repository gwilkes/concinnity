// src/build/gltf.rs
//
// Imports a skinned mesh + skeleton from a binary glTF (.glb) file into the
// inline `SkinnedMesh` asset fields. Only the `.glb` container is handled:
// buffer data must travel in the embedded GLB binary chunk, so a `.gltf` with
// external or base64-URI buffers is rejected. glTF animations are not
// imported: the mesh lands in its bind pose.
//
// glTF stores a skin's joints in an arbitrary order; this engine's `JointDef`
// list requires parents before children. Joints are therefore topologically
// reordered and a remap table rewrites both each joint's parent index and
// every vertex's `JOINTS_0` binding into the new index space.

// The `.glb` container decode lives in `crate::glb`; the asset-level desugar
// wrappers here call into it for parsing, shared geometry reads, and the
// Imported* payload types.
use crate::glb::{
    ImportedAnimation, ImportedSkinnedMesh, import_glb_animations_from_doc,
    import_skinned_from_doc, parse_glb, resolve_source,
};

// Parse a binary glTF (`.glb`) file into inline `SkinnedMesh` geometry plus a
// parents-before-children skeleton. The first node carrying both a mesh and a
// skin is imported; other nodes, materials, cameras, and animations are
// ignored.
pub fn import_skinned_glb(source: &str) -> Result<ImportedSkinnedMesh, String> {
    let doc = parse_glb(source)?;
    import_skinned_from_doc(&doc, source)
}

// Vertex count for the indexed primitive without reading any vertex data,
// used by `cn add` to decide whether a primitive fits Concinnity's u16 index
// limit or needs splitting.
pub fn primitive_vertex_count(doc: &gltf::Gltf, primitive_index: u32) -> Option<usize> {
    doc.document
        .meshes()
        .flat_map(|m| m.primitives())
        .nth(primitive_index as usize)
        .and_then(|p| p.get(&gltf::Semantic::Positions))
        .map(|a| a.count())
}

// Import every animation in a `.glb` whose channels target joints of the
// file's first skinned node. Channels whose target node is not a skin joint,
// or whose interpolation method we cannot honour, are dropped silently;
// per-clip warnings would spam build output for files that mix joint and
// non-joint animations (e.g. character + camera).
pub fn import_glb_animations(source: &str) -> Result<Vec<ImportedAnimation>, String> {
    let doc = parse_glb(source)?;
    import_glb_animations_from_doc(&doc, source)
}

// Import a single animation by its glTF index. Index out of range is a hard
// error; the user authored an animation entry the file does not contain.
pub fn import_glb_animation(source: &str, index: usize) -> Result<ImportedAnimation, String> {
    let mut anims = import_glb_animations(source)?;
    if index >= anims.len() {
        return Err(format!(
            "'{}': animation_index {} out of range (file has {} animation{})",
            source,
            index,
            anims.len(),
            if anims.len() == 1 { "" } else { "s" }
        ));
    }
    Ok(anims.swap_remove(index))
}

// Names of every animation in a `.glb`, in file declaration order. Useful
// for the desugar pass when the user authored `animation_name` instead of
// `animation_index` and we need to look up the index.
pub fn glb_animation_names(source: &str) -> Result<Vec<String>, String> {
    let path = resolve_source(source);
    let bytes = std::fs::read(&path).map_err(|e| format!("failed to read '{}': {}", path, e))?;
    let doc = gltf::Gltf::from_slice(&bytes)
        .map_err(|e| format!("'{}': not a valid glTF/GLB file: {}", path, e))?;
    Ok(doc
        .document
        .animations()
        .map(|a| a.name().unwrap_or("").to_string())
        .collect())
}
