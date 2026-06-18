// asset_impls/file.rs

use concinnity_core::assets::File;
use concinnity_core::assets::FileKind;

impl crate::asset::BuildAsset for File {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let path = args.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let kind_str = args.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        let kind = FileKind::from_ext(kind_str).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}': unsupported File kind '{}'", ctx.name, kind_str),
            )
        })?;
        crate::file::compile_file_payload(path, &kind)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}
