// asset_impls/voxel_chunk.rs

use concinnity_core::assets::VoxelChunk;

impl crate::asset::BuildAsset for VoxelChunk {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let palette_lookup = |bt_name: &str| {
            ctx.all_assets
                .iter()
                .find(|a| {
                    let tn = a.asset_type.to_lowercase().replace('_', "");
                    (tn == "blocktype" || tn == "block") && a.name == bt_name
                })
                .map(|a| a.args.clone())
        };
        crate::geometry::compile_voxel_chunk_payload(args, palette_lookup)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
