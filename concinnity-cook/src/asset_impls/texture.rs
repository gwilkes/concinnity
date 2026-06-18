// asset_impls/texture.rs

use concinnity_core::assets::Texture;

impl crate::asset::BuildAsset for Texture {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::texture::compile_texture_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
