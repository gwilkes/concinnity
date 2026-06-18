// asset_impls/procedural_mesh.rs

use concinnity_core::assets::ProceduralMesh;

impl crate::asset::BuildAsset for ProceduralMesh {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::mesh_compile::compile_mesh_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
