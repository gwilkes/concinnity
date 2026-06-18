// asset_impls/skinned_mesh.rs

use concinnity_core::assets::JointDef;
use concinnity_core::assets::SkinnedMesh;

impl crate::asset::BuildAsset for SkinnedMesh {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let mesh: SkinnedMesh = serde_json::from_value(args.clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        // `skeleton` is no longer a field on the runtime struct; the
        // desugar pass writes it into the args JSON for glTF-sourced
        // meshes, and inline-authored worlds may carry it directly. Read
        // it straight from the JSON so it can be baked into the compiled
        // payload alongside vertices and indices.
        let skeleton: Vec<JointDef> = match args.get("skeleton") {
            Some(v) => serde_json::from_value(v.clone()).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("SkinnedMesh: invalid skeleton args: {e}"),
                )
            })?,
            None => Vec::new(),
        };
        let lod_levels = mesh.lod_levels.clamp(1, 8);
        crate::geometry::compile_skinned_mesh_payload_with_lods(
            &mesh.vertices,
            &mesh.indices,
            &skeleton,
            lod_levels,
            &mesh.lod_distances,
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
