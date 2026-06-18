// asset_impls/font.rs

use concinnity_core::assets::Font;

impl crate::asset::BuildAsset for Font {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::font::compile_font_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
