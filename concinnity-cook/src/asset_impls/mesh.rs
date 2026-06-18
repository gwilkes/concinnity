// asset_impls/mesh.rs

use concinnity_core::assets::Mesh;

impl crate::asset::BuildAsset for Mesh {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::mesh_compile::compile_mesh_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
