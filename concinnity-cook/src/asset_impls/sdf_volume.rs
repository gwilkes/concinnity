// asset_impls/sdf_volume.rs

use concinnity_core::assets::SdfVolume;
use concinnity_core::assets::sdf_volume::{current_platform_source_arg, resolve_source_path};

impl crate::asset::BuildAsset for SdfVolume {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        // Only the current backend's shader is required: a volume that
        // declares an `.hlsl` source (or an `hlsl`-only map) contributes
        // nothing the Metal build can compile, so it is a hard error here
        // rather than an attempt to read a file the backend never needs.
        let platform_key = concinnity_core::build::Platform::current().key();
        let raw = current_platform_source_arg(args).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "SdfVolume '{}': no fragment shader source for backend \"{}\" \
                     (declare `fragment_shaders.{}` or a `fragment_shader` path \
                     with a matching extension)",
                    ctx.name, platform_key, platform_key
                ),
            )
        })?;

        let source_path = resolve_source_path(&raw, ctx).unwrap_or_else(|| raw.clone());

        // No MSL compilation here: the runtime backend prepends the
        // engine-shipped helpers + appends the template and compiles
        // via `newLibraryWithSource_options_error` (matching how every
        // other Metal feature pass loads its MSL). We just transport
        // the user source bytes through the blob so `cn run` worlds
        // don't need the file on disk.
        std::fs::read(&source_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!(
                    "SdfVolume '{}': failed to read fragment shader '{}': {}",
                    ctx.name, source_path, e
                ),
            )
        })
    }

    // The cache's generic JSON-string walk only resolves bare filenames
    // (via `find_in_assets`) and cwd-relative paths. `fragment_shader` is
    // typically a path with a directory component (e.g.
    // `"shaders/chrome_blob.metal"`) under the source-tree `assets/` dir,
    // which the generic walk misses. Without this override, editing the
    // .metal file would not invalidate the cache and stale bytes would
    // replay forever.
    fn source_files(args: &serde_json::Value, ctx: &crate::asset::BuildCtx<'_>) -> Vec<String> {
        let Some(raw) = current_platform_source_arg(args) else {
            return Vec::new();
        };
        resolve_source_path(&raw, ctx).into_iter().collect()
    }
}
