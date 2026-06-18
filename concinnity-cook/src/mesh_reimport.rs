// src/mesh_reimport.rs
//
// Asset hot-reload (`cn debug`) decode helpers: re-import a file-backed Mesh /
// SkinnedMesh from a pre-parsed glTF document into the runtime Vertex /
// SkinnedVertex form, mirroring the build pipeline so a hot-reloaded mesh is
// byte-identical to a fresh `cn build`. The runtime crate links no image / glTF
// decoders, so these live here in the build crate; the editor's debug server
// drives them.

use concinnity_core::geometry::{
    compile_mesh_payload, compile_skinned_mesh_payload_with_lods, payload_joints_to_defs,
};
use concinnity_core::gfx::mesh_payload::{deserialise_skinned, deserialise_with_lods};

// LOD alternates: (switch_distance, index buffer) pairs.
type LodAlternates = Vec<(f32, Vec<u16>)>;

// Imported skinned mesh: runtime vertices, indices, and the bind-pose skeleton.
type SkinnedImport = (
    Vec<concinnity_core::gfx::mesh_payload::SkinnedVertex>,
    Vec<u16>,
    Vec<concinnity_core::assets::JointDef>,
);

// Decode a file-backed `Mesh` primitive from a pre-parsed glTF document the
// same way the build pipeline does at compile time, returning the runtime
// `Vertex` / index form with normals + tangents + optional LOD alternates baked
// in. Used by the asset hot-reload path (`cn debug` only); production reads the
// compiled payload from a blob locator and goes through
// `deserialise_with_lods` instead.
//
// The caller is responsible for parsing the `.glb` (via [`crate::glb::parse_glb`])
// so a single reload pass can amortise the parse across every `Mesh` that
// references the same file: `ABeautifulGame` alone fans 35+ Mesh assets out of
// one `.glb`.
//
// `primitive_index` selects which primitive (flattened across glTF meshes) to
// import; `lod_levels` and `lod_distances` mirror the asset declaration so the
// reload produces a byte-identical payload to the build pass. The third
// component of the result is empty for `lod_levels <= 1`.
pub fn decode_mesh_from_parsed_glb(
    doc: &gltf::Gltf,
    source: &str,
    primitive_index: u32,
    lod_levels: u32,
    lod_distances: &[f32],
) -> Result<
    (
        Vec<concinnity_core::gfx::mesh_payload::Vertex>,
        Vec<u16>,
        LodAlternates,
    ),
    String,
> {
    let (vertex_data, indices) =
        crate::glb::import_static_glb_primitive_from_doc(doc, source, primitive_index)?;
    // Rebuild the JSON args the desugar pass would have produced, then run the
    // existing compile + deserialise cycle. This keeps the runtime path
    // byte-identical to the build pass so any difference is a build bug, not a
    // reload-only divergence.
    let args = serde_json::json!({
        "vertices": vertex_data,
        "indices": indices,
        "lod_levels": lod_levels,
        "lod_distances": lod_distances,
    });
    let payload = compile_mesh_payload(&args)?;
    deserialise_with_lods(&payload)
}

// Decode a file-backed `SkinnedMesh` from a pre-parsed glTF document the same
// way the build pipeline does at compile time, returning the runtime
// `SkinnedVertex` / index form (normals + tangents baked in) plus the imported
// bind-pose skeleton. Used by the asset hot-reload path (`cn debug` only);
// production reads the compiled payload from a blob locator and goes through
// `deserialise_skinned` instead.
//
// The caller is responsible for parsing the `.glb` (via [`crate::glb::parse_glb`])
// so a single reload pass can amortise the parse across every `Mesh` /
// `SkinnedMesh` that references the same file. The skeleton is returned in the
// same `JointDef` form the `SkinnedMesh` asset args carry; the reload helper
// checks it against the init-time joint count before pushing to the GPU.
pub fn decode_skinned_from_parsed_glb(
    doc: &gltf::Gltf,
    source: &str,
) -> Result<SkinnedImport, String> {
    let imported = crate::glb::import_skinned_from_doc(doc, source)?;
    let payload = compile_skinned_mesh_payload_with_lods(
        &imported.vertices,
        &imported.indices,
        &imported.skeleton,
        1,
        &[],
    )?;
    let (verts, idxs, payload_joints) = deserialise_skinned(&payload)?;
    let skeleton = payload_joints_to_defs(payload_joints);
    Ok((verts, idxs, skeleton))
}
