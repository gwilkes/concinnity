// Dispatches to the correct backend compiler based on the source file extension.
//   .metal -> xcrun metal + xcrun metallib -> raw .metallib bytes (macOS only)
//   all others -> glslc or shaderc -> SPIR-V bytes
//
// Built-in shader sources are embedded into the binary at compile time so the
// runtime never depends on `assets/` for them. Any caller-supplied source path
// whose bare filename matches a built-in name resolves to the embedded bytes.
// Source is handed straight to the platform compiler (over stdin or in
// memory); no shader source file is written to disk.
//
// The built-in source table (`builtin_shader_source` and the embedded const
// strings) stays in concinnity-core; this crate's compile path resolves bare
// built-in filenames through `concinnity_core::build::shader::builtin_shader_source`.

use concinnity_core::build::shader::builtin_shader_source;

#[derive(Debug, Clone)]
pub struct ShaderCompileArgs {
    pub source_path: String,
    pub asset_name: String,
    #[allow(dead_code)]
    pub kind: String,
}

// Backend hook the build pipeline calls after a user shader compiles, so a
// render backend can validate that the shader's engine-provided buffer structs
// (per-frame uniforms, object data, lights, ...) have the same memory layout
// as the engine's `#[repr(C)]` Rust structs. Catches CPU/GPU layout mismatches
// at `cn build` with a clear message instead of as a GPU page fault at `cn run`.
//
// The build pipeline (this crate) is backend-agnostic and never links a GPU
// API, so the actual reflection lives in the client's render backend, which
// installs an implementation via [`set_shader_build_validator`]. When none is
// registered (a core-only build, a non-macOS host, the server) the call is a
// no-op and the build is unaffected.
pub trait ShaderBuildValidator: Send + Sync {
    // Validate one compiled shader stage. `source` is the shader source text,
    // `kind` the compile kind (`"vertex"` or `"fragment"`; a shadow stage
    // compiles as `"vertex"` and is told apart by its entry-point name),
    // `asset_name` the declaring asset. Return `Err(msg)` to fail the build.
    fn validate_metal(&self, source: &str, kind: &str, asset_name: &str) -> Result<(), String>;
}

static SHADER_BUILD_VALIDATOR: std::sync::OnceLock<Box<dyn ShaderBuildValidator>> =
    std::sync::OnceLock::new();

// Register the process-wide shader build validator. The first registration
// wins; later calls are ignored, so build entry points can call this
// unconditionally (the backend installs exactly one validator).
pub fn set_shader_build_validator(validator: Box<dyn ShaderBuildValidator>) {
    let _ = SHADER_BUILD_VALIDATOR.set(validator);
}

// Run the registered validator against a just-compiled `.metal` source. A no-op
// when no validator is registered or when the source is an engine built-in
// (built-ins are correct by construction and covered by the `*_layout_matches_msl`
// unit tests; only user-authored sources are checked). A validation error is
// surfaced as an `InvalidData` build error so `cn build` fails.
//
// Only reached from `compile_metal`, which is `#[cfg(backend_metal)]`; on the
// other backends it's exercised solely by the unit tests, so allow dead code
// there rather than gating the function (the tests call it unconditionally).
#[cfg_attr(not(backend_metal), allow(dead_code))]
fn validate_compiled_metal(source: &str, args: &ShaderCompileArgs) -> Result<(), std::io::Error> {
    if builtin_shader_source(&args.source_path).is_some() {
        return Ok(());
    }
    let Some(validator) = SHADER_BUILD_VALIDATOR.get() else {
        return Ok(());
    };
    validator
        .validate_metal(source, &args.kind, &args.asset_name)
        .map_err(|msg| std::io::Error::new(std::io::ErrorKind::InvalidData, msg))
}

pub fn compile_shader(args: ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    let ext = std::path::Path::new(&args.source_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // .metal/.hlsl source is passed to the backend in memory; .glsl is only
    // ever caller-supplied (never a built-in), so that path keeps reading
    // from its own file.
    match ext {
        "metal" => compile_metal(&read_shader_source(&args.source_path)?, &args),
        "hlsl" => compile_hlsl(&read_shader_source(&args.source_path)?, &args),
        _ => compile_glsl(args),
    }
}

// Resolve a shader source path to its text. A bare filename matching an
// engine-shipped shader resolves to the source embedded at compile time;
// built-ins always win, so a stale copy under assets/ can't shadow one. Any
// other path is read from disk.
fn read_shader_source(source_path: &str) -> Result<String, std::io::Error> {
    match builtin_shader_source(source_path) {
        Some(src) => Ok(src.to_string()),
        None => std::fs::read_to_string(source_path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("Failed to read shader source '{}': {}", source_path, e),
            )
        }),
    }
}

#[cfg(backend_metal)]
fn compile_metal(source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use std::fs;
    use std::io::Write;
    use std::process::Stdio;

    let data_dir = crate::paths::data_dir();
    let air_path = format!("{}/{}.air", data_dir.display(), args.asset_name);
    let lib_path = format!("{}/{}.metallib", data_dir.display(), args.asset_name);

    fs::create_dir_all(&data_dir)?;

    // Feed the source to `xcrun metal` over stdin (`-x metal` selects the
    // language since stdin has no extension, `-` is the stdin input) so no
    // shader source file is written to disk. The .air and .metallib it emits
    // are intermediate artifacts under .concinnity/data/, removed once read.
    let mut metal = std::process::Command::new("xcrun")
        .args(["metal", "-x", "metal", "-c", "-", "-o", &air_path])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    metal
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(source.as_bytes())?;
    let metal_output = metal.wait_with_output()?;

    if !metal_output.status.success() {
        let _ = fs::remove_file(&air_path);
        return Err(std::io::Error::other(format!(
            "xcrun metal failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&metal_output.stdout),
            String::from_utf8_lossy(&metal_output.stderr),
        )));
    }

    let lib_output = std::process::Command::new("xcrun")
        .args(["metallib", &air_path, "-o", &lib_path])
        .output()?;

    let _ = fs::remove_file(&air_path);

    if !lib_output.status.success() {
        let _ = fs::remove_file(&lib_path);
        return Err(std::io::Error::other(format!(
            "xcrun metallib failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&lib_output.stdout),
            String::from_utf8_lossy(&lib_output.stderr),
        )));
    }

    let bytes = fs::read(&lib_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read metallib '{}': {}", lib_path, e),
        )
    })?;

    let _ = fs::remove_file(&lib_path);

    // The shader compiled; now check that every engine-provided buffer struct it
    // reads matches the engine's layout. A mismatch fails the build here rather
    // than faulting the GPU at run time.
    validate_compiled_metal(source, args)?;

    Ok(bytes)
}

#[cfg(not(backend_metal))]
fn compile_metal(_source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!(
            "Asset '{}': .metal shaders can only be compiled on macOS",
            args.asset_name
        ),
    ))
}

// Vulkan build: compile GLSL in-process via the shaderc crate.
#[cfg(backend_vk)]
fn compile_glsl(args: ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use shaderc::{CompileOptions, Compiler, EnvVersion, OptimizationLevel, ShaderKind, TargetEnv};

    let source = std::fs::read_to_string(&args.source_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("Failed to read '{}': {}", args.source_path, e),
        )
    })?;

    let compiler = Compiler::new()
        .map_err(|e| std::io::Error::other(format!("shaderc init failed: {}", e)))?;

    let mut options = CompileOptions::new()
        .map_err(|e| std::io::Error::other(format!("shaderc options init: {}", e)))?;
    options.set_optimization_level(OptimizationLevel::Performance);
    // Target Vulkan 1.2 so the emitted module is SPIR-V 1.5 (not the 1.6 a 1.3
    // target produces). The runtime instance is created at Vulkan 1.2 (see
    // `concinnity_client::vulkan::init`), and `vkCreateShaderModule` rejects a SPIR-V version
    // newer than the instance's, so a world-authored ShaderStage compiled to
    // SPIR-V 1.6 fails to load. 1.5 loads cleanly on a 1.2-or-newer instance.
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);

    let kind = match args.kind.to_lowercase().as_str() {
        "vertex" | "vert" => ShaderKind::Vertex,
        "fragment" | "frag" => ShaderKind::Fragment,
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported shader kind: '{}'", args.kind),
            ));
        }
    };

    let artifact = compiler
        .compile_into_spirv(&source, kind, &args.source_path, "main", Some(&options))
        .map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset '{}' compile error:\n{}", args.asset_name, e),
            )
        })?;

    if artifact.get_num_warnings() > 0 {
        tracing::warn!(
            "Asset '{}' GLSL warnings:\n{}",
            args.asset_name,
            artifact.get_warning_messages()
        );
    }

    let spv_bytes: Vec<u8> = artifact
        .as_binary()
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();

    Ok(spv_bytes)
}

// DirectX build: worlds use HLSL, so GLSL is never compiled.
#[cfg(backend_dx)]
fn compile_glsl(args: ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!(
            "Asset '{}': GLSL/SPIR-V compilation is not supported by the DirectX backend (use HLSL)",
            args.asset_name
        ),
    ))
}

#[cfg(backend_dx)]
fn compile_hlsl(source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    use windows::Win32::Graphics::Direct3D::Fxc::{
        D3DCOMPILE_DEBUG, D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR,
        D3DCOMPILE_SKIP_OPTIMIZATION, D3DCompile,
    };

    let target = match args.kind.to_lowercase().as_str() {
        "fragment" | "frag" => "ps_5_1",
        _ => "vs_5_1",
    };

    let src_c = std::ffi::CString::new(source).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("hlsl src: {e}"))
    })?;
    let entry_c = std::ffi::CString::new("main").unwrap();
    let target_c = std::ffi::CString::new(target).unwrap();

    // Force column-major matrix storage so matrices inside `StructuredBuffer`
    // structs read as column-major (matching Rust's upload layout). FXC
    // silently ignores `#pragma pack_matrix(column_major)` for SRV-resident
    // matrices and defaults them to row_major; without this flag a custom
    // shader that reads e.g. an instance-matrix StructuredBuffer would see
    // every transform transposed. Mirrors the same flag in
    // `concinnity_client::directx::pipeline::compile_hlsl`.
    let flags = if cfg!(debug_assertions) {
        D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
    } else {
        D3DCOMPILE_OPTIMIZATION_LEVEL3 | D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR
    };

    let mut blob: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;
    let mut error: Option<windows::Win32::Graphics::Direct3D::ID3DBlob> = None;

    let result = unsafe {
        D3DCompile(
            src_c.as_ptr() as *const std::ffi::c_void,
            source.len(),
            None,
            None,
            None,
            windows::core::PCSTR(entry_c.as_ptr() as *const u8),
            windows::core::PCSTR(target_c.as_ptr() as *const u8),
            flags,
            0,
            &mut blob,
            Some(&mut error),
        )
    };

    if result.is_err() {
        let msg = error
            .as_ref()
            .map(|e| {
                let ptr = unsafe { e.GetBufferPointer() } as *const u8;
                let len = unsafe { e.GetBufferSize() };
                String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(ptr, len) })
                    .into_owned()
            })
            .unwrap_or_else(|| "unknown compile error".to_string());
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Asset '{}' compile error ({target}):\n{msg}",
                args.asset_name
            ),
        ));
    }

    let b = blob.ok_or_else(|| {
        std::io::Error::other(format!(
            "Asset '{}': D3DCompile returned no blob",
            args.asset_name
        ))
    })?;
    let ptr = unsafe { b.GetBufferPointer() } as *const u8;
    let len = unsafe { b.GetBufferSize() };
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
}

#[cfg(not(backend_dx))]
fn compile_hlsl(_source: &str, args: &ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!(
            "Asset '{}': .hlsl shaders are only supported by the DirectX backend",
            args.asset_name
        ),
    ))
}

// macOS (Metal backend) compiles GLSL by shelling out to glslc.
#[cfg(backend_metal)]
fn compile_glsl(args: ShaderCompileArgs) -> Result<Vec<u8>, std::io::Error> {
    let data_dir = crate::paths::data_dir();
    let out_path = format!("{}/{}.spv", data_dir.display(), args.asset_name);
    std::fs::create_dir_all(&data_dir)?;

    let output = std::process::Command::new("glslc")
        .args([
            "--target-env=vulkan1.0",
            "-fshader-stage",
            &args.kind,
            &args.source_path,
            "-o",
            &out_path,
        ])
        .output()?;

    if !output.status.success() {
        let _ = std::fs::remove_file(&out_path);
        return Err(std::io::Error::other(format!(
            "glslc failed for '{}':\n--- stdout ---\n{}\n--- stderr ---\n{}",
            args.asset_name,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )));
    }

    let bytes = std::fs::read(&out_path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("failed to read SPIR-V '{}': {}", out_path, e),
        )
    })?;

    let _ = std::fs::remove_file(&out_path);

    Ok(bytes)
}

#[cfg(test)]
mod hook_tests {
    use super::*;

    // A validator that only objects to one sentinel asset name, so registering
    // it process-wide (the OnceLock can only be set once per test binary) leaves
    // every other asset's build untouched.
    struct SentinelValidator;
    const SENTINEL: &str = "__layout_hook_sentinel__";

    impl ShaderBuildValidator for SentinelValidator {
        fn validate_metal(
            &self,
            _source: &str,
            _kind: &str,
            asset_name: &str,
        ) -> Result<(), String> {
            if asset_name == SENTINEL {
                Err("sentinel layout mismatch".to_string())
            } else {
                Ok(())
            }
        }
    }

    fn args(asset_name: &str, source_path: &str) -> ShaderCompileArgs {
        ShaderCompileArgs {
            source_path: source_path.to_string(),
            asset_name: asset_name.to_string(),
            kind: "fragment".to_string(),
        }
    }

    #[test]
    fn validator_hook_dispatches_and_skips_builtins() {
        set_shader_build_validator(Box::new(SentinelValidator));

        // A user source for the sentinel asset surfaces the validator's error.
        let err = validate_compiled_metal("frag source", &args(SENTINEL, "user_frag.metal"))
            .expect_err("sentinel must fail");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("sentinel layout mismatch"));

        // A built-in source is skipped even for the sentinel asset name.
        validate_compiled_metal("frag source", &args(SENTINEL, "default.metal"))
            .expect("built-ins are never validated");

        // Any other asset passes through cleanly.
        validate_compiled_metal("frag source", &args("ok_asset", "user_frag.metal"))
            .expect("non-sentinel assets pass");
    }
}
