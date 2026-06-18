// asset_impls/environment_map.rs

use concinnity_core::assets::EnvironmentMap;

impl crate::asset::BuildAsset for EnvironmentMap {
    fn compile_payload(
        args: &serde_json::Value,
        _ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        crate::environment_map::compile_environment_map_payload(args)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
