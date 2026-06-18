// asset_impls/shader_stage.rs

use concinnity_core::assets::ShaderKind;
use concinnity_core::assets::ShaderStage;
use concinnity_core::assets::shader_stage::{
    declares_only_builtin_sources, platform_key, resolve_source_from_args, resolve_source_path_for,
};

impl crate::asset::BuildAsset for ShaderStage {
    fn compile_payload(
        args: &serde_json::Value,
        ctx: &crate::asset::BuildCtx<'_>,
    ) -> std::io::Result<Vec<u8>> {
        let shader_kind: ShaderKind = args
            .get("kind")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or(ShaderKind::Vertex);

        let resolved = resolve_source_from_args(args);

        // On Linux/Vulkan, missing per-platform sources are not fatal: the
        // Vulkan backend ships inline GLSL for every required stage and
        // compiles it whenever the payload bytes aren't valid SPIR-V.
        if resolved.is_none() && platform_key() == "glsl" {
            // The bundled default shader set declares only metal/hlsl built-ins
            // and renders via the backend's inline GLSL by design, so stay
            // quiet for it. Only a custom stage that forgot its glsl variant
            // (some non-built-in source) is worth flagging -- it won't render
            // as authored on Vulkan.
            if !declares_only_builtin_sources(args) {
                tracing::warn!(
                    "Asset '{}': no shader source for platform \"glsl\", falling back to built-in GLSL",
                    ctx.name
                );
            }
            return Ok(vec![]);
        }

        let raw = resolved.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Compiled asset '{}': no shader source for platform \"{}\"",
                    ctx.name,
                    platform_key()
                ),
            )
        })?;

        // `resolve_source_path_for` stays in core and is typed against core's
        // `BuildCtx`; this crate's `BuildCtx` is a distinct (field-identical)
        // type, so bridge it across the crate boundary by value.
        let core_ctx = concinnity_core::build::BuildCtx {
            name: ctx.name,
            artifacts_dir: ctx.artifacts_dir,
            all_assets: ctx.all_assets,
        };
        let source_path = resolve_source_path_for(&raw, &core_ctx);

        let compile_args = crate::shader::ShaderCompileArgs {
            source_path,
            asset_name: ctx.name.to_string(),
            kind: shader_kind.compile_kind().to_string(),
        };
        crate::shader::compile_shader(compile_args).map_err(|e| {
            std::io::Error::other(format!("Asset '{}' compile error: {}", ctx.name, e))
        })
    }

    // The cache's generic JSON-string walk only finds bare filenames via
    // `find_in_assets` (which walks `.concinnity/assets/`). A `sources` entry
    // with a directory component, or a bare filename that lives in
    // `<artifacts_dir>` instead of `.concinnity/assets/`, is missed. Built-in
    // shader names short-circuit through the generic walk's `builtin:` path
    // so we skip them here.
    fn source_files(args: &serde_json::Value, ctx: &crate::asset::BuildCtx<'_>) -> Vec<String> {
        let Some(raw) = resolve_source_from_args(args) else {
            return Vec::new();
        };
        if concinnity_core::build::shader::builtin_shader_source(&raw).is_some() {
            return Vec::new();
        }
        // See `compile_payload`: bridge this crate's `BuildCtx` to core's,
        // which `resolve_source_path_for` is typed against.
        let core_ctx = concinnity_core::build::BuildCtx {
            name: ctx.name,
            artifacts_dir: ctx.artifacts_dir,
            all_assets: ctx.all_assets,
        };
        let path = resolve_source_path_for(&raw, &core_ctx);
        if std::path::Path::new(&path).exists() {
            vec![path]
        } else {
            Vec::new()
        }
    }
}
