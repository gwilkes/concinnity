// asset_impls/color_lut.rs

use concinnity_core::assets::ColorLut;

impl crate::asset::BuildAsset for ColorLut {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::color_lut::compile_color_lut_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
