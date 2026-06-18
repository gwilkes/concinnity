// asset_impls/cubemap_texture.rs

use concinnity_core::assets::CubemapTexture;

impl crate::asset::BuildAsset for CubemapTexture {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::cubemap::compile_cubemap_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
