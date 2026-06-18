// Shared build-script support for the workspace.
//
// Two responsibilities, both previously copy-pasted between the runtime crate's
// build script and the editor crate's build script (and missing entirely from
// the example binaries, which is why they failed to link against the runtime's
// DLSS code on Windows):
//
// 1. Resolve the rendering backend once and emit it as a single cfg
//    (`backend_metal` / `backend_dx` / `backend_vk`) the source gates on.
//
// 2. Detect the optional graphics SDKs and emit the cfgs the renderer gates on
//    (`agility_sdk_configured`, `ffx_sdk_bundled`, `xess_sdk_bundled`,
//    `ngx_sdk_bundled`, `dxc_bundled`). For a package that produces a final
//    binary (the editor, an example) this also copies the runtime DLLs next to
//    the .exe and links the NGX import lib; for the runtime rlib's own test
//    binaries only the NGX link is needed (no DLL copy), selected with
//    `SdkOptions { bundle_dlls: false }`.
//
// Every function here emits `cargo::` directives on stdout, which Cargo
// attributes to the build script of whichever package called in. That is what
// lets an example binary's build script pick up the same NGX link and DLL
// bundling the editor's does, without duplicating any of this logic.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Dx,
    Vk,
}

impl Backend {
    fn cfg_name(self) -> &'static str {
        match self {
            Backend::Metal => "backend_metal",
            Backend::Dx => "backend_dx",
            Backend::Vk => "backend_vk",
        }
    }
}

// Options for the SDK setup. `bundle_dlls` distinguishes a package that produces
// a final binary (true: copy runtime DLLs next to the .exe, emit the Agility
// linker exports) from the runtime rlib's own test binaries (false: link the
// NGX import lib and emit the gating cfgs, but place no DLLs).
#[derive(Clone, Copy, Debug)]
pub struct SdkOptions {
    pub bundle_dlls: bool,
}

// Resolve the backend from the target OS and whether the `vulkan` feature is on.
// macOS is always Metal; Windows defaults to DirectX and opts into Vulkan with
// the feature; everything else (Linux) is Vulkan.
pub fn resolve_backend(target_os: &str, vulkan: bool) -> Backend {
    match (target_os, vulkan) {
        ("macos", _) => Backend::Metal,
        ("windows", false) => Backend::Dx,
        _ => Backend::Vk,
    }
}

// Declare every cfg the renderer source gates on so `--check-cfg` does not warn.
// A package only needs this if its own source references one of these cfgs.
pub fn emit_check_cfgs() {
    for cfg in [
        "backend_metal",
        "backend_dx",
        "backend_vk",
        "agility_sdk_configured",
        "ffx_sdk_bundled",
        "xess_sdk_bundled",
        "ngx_sdk_bundled",
        "dxc_bundled",
    ] {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
    }
}

// Resolve the backend from the Cargo-provided environment and emit the
// `rustc-cfg` for it, returning the choice so the caller can branch.
pub fn emit_backend_cfg() -> Backend {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();
    let backend = resolve_backend(&target_os, vulkan);
    println!("cargo::rustc-cfg={}", backend.cfg_name());
    backend
}

// Set up the optional graphics SDKs for the given backend. On a non-Windows
// target (or the Metal backend) this is a no-op: none of these SDKs apply.
pub fn setup_graphics_sdks(backend: Backend, opts: SdkOptions) {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match backend {
        Backend::Dx => {
            setup_agility_sdk(opts.bundle_dlls);
            setup_fidelityfx_dx_sdk(opts.bundle_dlls);
            setup_xess_sdk(opts.bundle_dlls);
            setup_dlss_sdk(opts.bundle_dlls);
            if opts.bundle_dlls {
                setup_dxc_sdk();
            }
        }
        Backend::Vk if target_os == "windows" => {
            // DLSS (NGX) and XeSS expose Vulkan entry points from the same
            // binaries the DirectX backend uses, so the setup helpers are
            // backend-agnostic and reused here. GLFW links dynamically on the
            // Windows Vulkan build, so its DLL is bundled too.
            if opts.bundle_dlls {
                setup_glfw_runtime();
            }
            setup_fidelityfx_vk_sdk(opts.bundle_dlls);
            setup_dlss_sdk(opts.bundle_dlls);
            setup_xess_sdk(opts.bundle_dlls);
        }
        _ => {}
    }
}

// Default SDK install roots, overridable via the matching env var.
const DEFAULT_AGILITY_SDK_ROOT: &str = "C:\\microsoft.direct3d.d3d12.1.619.3";
const DEFAULT_FIDELITYFX_SDK_ROOT: &str = "C:\\FidelityFX-SDK-v1.1.4";
const DEFAULT_XESS_SDK_ROOT: &str = "C:\\XeSS_SDK_3.0.1";
const DEFAULT_STREAMLINE_SDK_ROOT: &str = "C:\\streamline-sdk-v2.11.1";

// Returns true when `var` is unset or set to anything other than "0". Each SDK
// probe uses this so it defaults to ON and can be opted out of with `<VAR>=0`.
fn enabled(var: &str) -> bool {
    println!("cargo::rerun-if-env-changed={var}");
    std::env::var(var).ok().as_deref() != Some("0")
}

// Microsoft's Agility SDK D3D12 runtime. The DLL copy and the binary's
// `D3D12SDKVersion`/`D3D12SDKPath` exports are only emitted when bundling for a
// final binary; the `agility_sdk_configured` cfg is always emitted when the SDK
// is present so the runtime FSR3 gate matches what the binary actually carries.
fn setup_agility_sdk(bundle_dlls: bool) {
    if !enabled("CN_ENABLE_AGILITY_SDK") {
        if bundle_dlls {
            println!(
                "cargo::warning=Agility SDK setup skipped (CN_ENABLE_AGILITY_SDK=0); \
                 binary will use the OS-bundled D3D12 runtime"
            );
        }
        return;
    }
    println!("cargo::rerun-if-env-changed=AGILITY_SDK_ROOT");

    let sdk_root =
        std::env::var("AGILITY_SDK_ROOT").unwrap_or_else(|_| DEFAULT_AGILITY_SDK_ROOT.to_string());
    let sdk_bin = PathBuf::from(&sdk_root)
        .join("build")
        .join("native")
        .join("bin")
        .join("x64");
    let core_dll = sdk_bin.join("D3D12Core.dll");

    if !core_dll.exists() {
        if bundle_dlls {
            println!(
                "cargo::warning=Agility SDK not found at {} - set AGILITY_SDK_ROOT \
                 or install the `microsoft.direct3d.d3d12` NuGet package. FidelityFX \
                 FSR3 will be unavailable (the binary falls back to the OS-bundled \
                 D3D12 runtime).",
                sdk_bin.display()
            );
        }
        return;
    }

    if bundle_dlls {
        // `D3D12SDKPath = ".\\D3D12\\"` resolves relative to the .exe, so the
        // DLLs must live in `<target>/<profile>/D3D12/`.
        let Some(profile_dir) = target_profile_dir() else {
            return;
        };
        let d3d12_dir = profile_dir.join("D3D12");
        if let Err(e) = std::fs::create_dir_all(&d3d12_dir) {
            println!(
                "cargo::warning=Agility SDK: could not create {}: {e}",
                d3d12_dir.display()
            );
            return;
        }
        for dll in ["D3D12Core.dll", "d3d12SDKLayers.dll"] {
            let src = sdk_bin.join(dll);
            let dst = d3d12_dir.join(dll);
            if let Err(e) = std::fs::copy(&src, &dst) {
                println!(
                    "cargo::warning=Agility SDK: could not copy {} -> {}: {e}",
                    src.display(),
                    dst.display()
                );
                return;
            }
            println!("cargo::rerun-if-changed={}", src.display());
        }

        // Export the two symbols `d3d12.dll` reads at process start. `,DATA` is
        // critical: without it the linker inserts a code thunk that `d3d12.dll`
        // would dereference as a pointer. The symbols themselves are defined as
        // `#[used]` statics in the binary crate's source.
        println!("cargo::rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo::rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }

    println!("cargo::rustc-cfg=agility_sdk_configured");
}

// AMD FidelityFX DX12 upscaler runtime. The renderer loads the DLL with
// `LoadLibrary` at runtime, so bundling only copies it next to the .exe; the
// `ffx_sdk_bundled` cfg is emitted when the SDK is present regardless.
fn setup_fidelityfx_dx_sdk(bundle_dlls: bool) {
    if !enabled("CN_ENABLE_FFX_FSR3") {
        if bundle_dlls {
            println!(
                "cargo::warning=FidelityFX SDK bundling skipped (CN_ENABLE_FFX_FSR3=0); \
                 temporal upscaling will be unavailable unless amd_fidelityfx_dx12.dll \
                 is on PATH at runtime"
            );
        }
        return;
    }
    println!("cargo::rerun-if-env-changed=FIDELITYFX_SDK_ROOT");

    let sdk_root = std::env::var("FIDELITYFX_SDK_ROOT")
        .unwrap_or_else(|_| DEFAULT_FIDELITYFX_SDK_ROOT.to_string());
    let dll_src = PathBuf::from(&sdk_root)
        .join("bin")
        .join("amd_fidelityfx_dx12.dll");
    if !dll_src.exists() {
        if bundle_dlls {
            println!(
                "cargo::warning=FidelityFX SDK not found at {} - set FIDELITYFX_SDK_ROOT \
                 or install the SDK. Temporal upscaling will be unavailable unless \
                 amd_fidelityfx_dx12.dll is on PATH at runtime.",
                dll_src.display()
            );
        }
        return;
    }

    if bundle_dlls && !copy_next_to_exe(&dll_src, "amd_fidelityfx_dx12.dll") {
        return;
    }
    println!("cargo::rustc-cfg=ffx_sdk_bundled");
}

// AMD FidelityFX Vulkan upscaler runtime. Prefers the in-repo patched DLL under
// `concinnity-client/third_party/ffx/` (carries the FSR3 rw_luma_history format
// fix), falling back to the stock SDK copy.
fn setup_fidelityfx_vk_sdk(bundle_dlls: bool) {
    if !enabled("CN_ENABLE_FFX_FSR3") {
        if bundle_dlls {
            println!(
                "cargo::warning=FidelityFX SDK bundling skipped (CN_ENABLE_FFX_FSR3=0); \
                 Vulkan temporal upscaling will be unavailable unless amd_fidelityfx_vk.dll \
                 is on PATH at runtime"
            );
        }
        return;
    }
    println!("cargo::rerun-if-env-changed=FIDELITYFX_SDK_ROOT");

    // The vendored DLL lives at a fixed location relative to the workspace root,
    // resolved by walking up from the caller's manifest so this works no matter
    // which package's build script called in.
    let vendored = workspace_root().map(|root| {
        root.join("concinnity-client")
            .join("third_party")
            .join("ffx")
            .join("amd_fidelityfx_vk.dll")
    });
    let sdk_root = std::env::var("FIDELITYFX_SDK_ROOT")
        .unwrap_or_else(|_| DEFAULT_FIDELITYFX_SDK_ROOT.to_string());
    let sdk_dll = PathBuf::from(&sdk_root)
        .join("bin")
        .join("amd_fidelityfx_vk.dll");

    let dll_src = match vendored {
        Some(v) if v.exists() => v,
        _ => sdk_dll,
    };
    if !dll_src.exists() {
        if bundle_dlls {
            println!(
                "cargo::warning=FidelityFX VK runtime not found ({}). Set FIDELITYFX_SDK_ROOT, \
                 run scripts/setup_ffx_vk_dll.ps1, or put amd_fidelityfx_vk.dll on PATH at \
                 runtime; Vulkan temporal upscaling will fall back to native resolution.",
                dll_src.display()
            );
        }
        return;
    }

    if bundle_dlls && !copy_next_to_exe(&dll_src, "amd_fidelityfx_vk.dll") {
        return;
    }
    println!("cargo::rustc-cfg=ffx_sdk_bundled");
}

// Intel XeSS upscaler runtime. Pure `LoadLibrary` at runtime, so bundling only
// copies the DLL; the cfg gates the copy and a log.
fn setup_xess_sdk(bundle_dlls: bool) {
    if !enabled("CN_ENABLE_XESS") {
        if bundle_dlls {
            println!(
                "cargo::warning=XeSS SDK bundling skipped (CN_ENABLE_XESS=0); the XeSS \
                 upscaler will be unavailable unless libxess.dll is on PATH at runtime"
            );
        }
        return;
    }
    println!("cargo::rerun-if-env-changed=XESS_SDK_ROOT");

    let sdk_root =
        std::env::var("XESS_SDK_ROOT").unwrap_or_else(|_| DEFAULT_XESS_SDK_ROOT.to_string());
    let dll_src = PathBuf::from(&sdk_root).join("bin").join("libxess.dll");
    if !dll_src.exists() {
        if bundle_dlls {
            println!(
                "cargo::warning=XeSS SDK not found at {} - set XESS_SDK_ROOT or install \
                 the SDK. The XeSS upscaler backend will be unavailable unless \
                 libxess.dll is on PATH at runtime.",
                dll_src.display()
            );
        }
        return;
    }

    if bundle_dlls && !copy_next_to_exe(&dll_src, "libxess.dll") {
        return;
    }
    println!("cargo::rustc-cfg=xess_sdk_bundled");
}

// DLSS via raw NGX. The import lib is always linked when present (the DLSS code
// compiled into the runtime rlib references its symbols, so every final binary
// and the rlib's own tests must resolve them). When bundling for a final binary
// the feature DLL `nvngx_dlss.dll` is also copied next to the .exe.
fn setup_dlss_sdk(bundle_dlls: bool) {
    if !enabled("CN_ENABLE_DLSS") {
        if bundle_dlls {
            println!(
                "cargo::warning=DLSS (NGX) setup skipped (CN_ENABLE_DLSS=0); the DLSS \
                 upscaler backend will be unavailable"
            );
        }
        return;
    }
    println!("cargo::rerun-if-env-changed=STREAMLINE_SDK_ROOT");

    let sdk_root = std::env::var("STREAMLINE_SDK_ROOT")
        .unwrap_or_else(|_| DEFAULT_STREAMLINE_SDK_ROOT.to_string());
    let ngx_lib = PathBuf::from(&sdk_root)
        .join("external")
        .join("ngx-sdk")
        .join("lib")
        .join("Windows_x86_64")
        .join("nvsdk_ngx_d.lib");
    if !ngx_lib.exists() {
        if bundle_dlls {
            println!(
                "cargo::warning=NGX import lib not found at {} - set STREAMLINE_SDK_ROOT. \
                 The DLSS upscaler backend will be unavailable.",
                ngx_lib.display()
            );
        }
        return;
    }

    // Pass the NGX static import lib straight to the linker for the final
    // artifact (a build-script `rustc-link-lib` does not reliably propagate, and
    // `rustc-link-arg` is scoped to the calling package's own targets, so each
    // package that links the DLSS code must emit this itself).
    println!("cargo::rustc-link-arg={}", ngx_lib.display());
    println!("cargo::rerun-if-changed={}", ngx_lib.display());
    println!("cargo::rustc-cfg=ngx_sdk_bundled");

    if bundle_dlls {
        let dll_src = PathBuf::from(&sdk_root)
            .join("bin")
            .join("x64")
            .join("nvngx_dlss.dll");
        if !dll_src.exists() {
            println!(
                "cargo::warning=NGX feature DLL not found at {} - DLSS will fail to \
                 create its feature at runtime.",
                dll_src.display()
            );
            return;
        }
        copy_next_to_exe(&dll_src, "nvngx_dlss.dll");
    }
}

// DirectX Shader Compiler (`dxcompiler.dll` + `dxil.dll`) for the runtime DXC
// path that compiles the inline ray-tracing reflection shader. Copy-only, so
// only relevant when bundling for a final binary.
fn setup_dxc_sdk() {
    if !enabled("CN_ENABLE_DXC") {
        println!(
            "cargo::warning=DXC bundling skipped (CN_ENABLE_DXC=0); hardware \
             ray-traced reflections will be unavailable unless dxcompiler.dll + \
             dxil.dll are on PATH at runtime (the renderer falls back to SSR)"
        );
        return;
    }
    println!("cargo::rerun-if-env-changed=DXC_SDK_ROOT");

    let Some(dxc_dir) = find_dxc_dir() else {
        println!(
            "cargo::warning=dxcompiler.dll + dxil.dll not found - set DXC_SDK_ROOT \
             to a directory containing them, or install the Windows SDK. Hardware \
             ray-traced reflections will be unavailable (the renderer falls back \
             to SSR)."
        );
        return;
    };

    for dll in ["dxcompiler.dll", "dxil.dll"] {
        let src = dxc_dir.join(dll);
        if !copy_next_to_exe(&src, dll) {
            return;
        }
    }
    println!("cargo::rustc-cfg=dxc_bundled");
}

// GLFW runtime DLL for the Windows Vulkan build (GLFW links dynamically there).
// Copy-only.
fn setup_glfw_runtime() {
    if !enabled("CN_ENABLE_GLFW_DLL") {
        println!(
            "cargo::warning=GLFW runtime bundling skipped (CN_ENABLE_GLFW_DLL=0); \
             the Vulkan client needs glfw3.dll beside the .exe or on PATH to launch"
        );
        return;
    }

    let Some(profile_dir) = target_profile_dir() else {
        return;
    };
    let build_dir = profile_dir.join("build");
    let Some(dll_src) = find_glfw_dll(&build_dir) else {
        println!(
            "cargo::warning=glfw3.dll not found under {} - the Vulkan client may fail \
             to launch with a missing-DLL error. It is produced by the `glfw-sys` \
             dependency; a clean rebuild usually regenerates it.",
            build_dir.display()
        );
        return;
    };
    copy_next_to_exe(&dll_src, "glfw3.dll");
}

// Copy `src` to `<target>/<profile>/<file_name>` so `LoadLibrary` (which
// searches the .exe directory first) finds it. Returns false on failure after
// emitting a `cargo::warning`.
fn copy_next_to_exe(src: &Path, file_name: &str) -> bool {
    let Some(profile_dir) = target_profile_dir() else {
        return false;
    };
    let dst = profile_dir.join(file_name);
    if let Err(e) = std::fs::copy(src, &dst) {
        println!(
            "cargo::warning=could not copy {} -> {}: {e}",
            src.display(),
            dst.display()
        );
        return false;
    }
    println!("cargo::rerun-if-changed={}", src.display());
    true
}

// `<target>/<profile>/`, where the final binaries and bundled DLLs land. Derived
// from `OUT_DIR` (`<target>/<profile>/build/<pkg>-<hash>/out/`).
fn target_profile_dir() -> Option<PathBuf> {
    let out_dir = std::env::var("OUT_DIR").ok()?;
    profile_dir_from_out_dir(Path::new(&out_dir)).map(|p| p.to_path_buf())
}

// Pure form of the above: walk up to `<target>/<profile>/` from an `OUT_DIR`.
fn profile_dir_from_out_dir(out_dir: &Path) -> Option<&Path> {
    out_dir.ancestors().nth(3)
}

// Locate the workspace root by walking up from the caller's manifest until a
// `Cargo.toml` declaring `[workspace]` is found.
fn workspace_root() -> Option<PathBuf> {
    let start = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    find_ancestor_with(Path::new(&start), |dir| {
        std::fs::read_to_string(dir.join("Cargo.toml"))
            .map(|c| c.contains("[workspace]"))
            .unwrap_or(false)
    })
}

// Walk `start` and its ancestors, returning the first that satisfies `pred`.
fn find_ancestor_with(start: &Path, pred: impl Fn(&Path) -> bool) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if pred(&dir) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

// Locate a directory holding both `dxcompiler.dll` and `dxil.dll`: the
// `DXC_SDK_ROOT` override, else the highest-versioned Windows SDK `x64` bin that
// carries both.
fn find_dxc_dir() -> Option<PathBuf> {
    let has_both = |d: &Path| d.join("dxcompiler.dll").exists() && d.join("dxil.dll").exists();

    if let Ok(root) = std::env::var("DXC_SDK_ROOT") {
        let dir = PathBuf::from(root);
        if has_both(&dir) {
            return Some(dir);
        }
    }

    let sdk_bin = PathBuf::from("C:\\Program Files (x86)\\Windows Kits\\10\\bin");
    let versions = sorted_version_dirs(
        std::fs::read_dir(&sdk_bin)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
    );
    versions
        .into_iter()
        .rev()
        .map(|ver| ver.join("x64"))
        .find(|x64| has_both(x64))
}

// Sort version directories ascending so the newest is last. Lexicographic is
// adequate because every Windows SDK entry is `10.0.NNNNN.0` (equal width).
fn sorted_version_dirs(mut dirs: Vec<PathBuf>) -> Vec<PathBuf> {
    dirs.sort();
    dirs
}

// Locate `glfw3.dll` inside the `glfw-sys` build output under
// `build/glfw-sys-*/out/`, preferring the newest-MSVC `lib-vc2022` variant.
fn find_glfw_dll(build_dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(build_dir).ok()?.flatten() {
        let p = entry.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        if p.is_dir() && name.starts_with("glfw-sys-") {
            collect_files_named(&p.join("out"), "glfw3.dll", &mut candidates, 0);
        }
    }
    prefer_path_containing(candidates, "lib-vc2022")
}

// From a set of candidates, prefer the first whose path contains `needle`,
// otherwise the first overall.
fn prefer_path_containing(candidates: Vec<PathBuf>, needle: &str) -> Option<PathBuf> {
    candidates
        .iter()
        .find(|p| p.to_string_lossy().contains(needle))
        .cloned()
        .or_else(|| candidates.into_iter().next())
}

// Recursively collect files named `name` under `dir`, depth-bounded so a stray
// symlink loop cannot run away.
fn collect_files_named(dir: &Path, name: &str, out: &mut Vec<PathBuf>, depth: usize) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_files_named(&p, name, out, depth + 1);
        } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
            out.push(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_resolution_covers_every_target() {
        assert_eq!(resolve_backend("macos", false), Backend::Metal);
        assert_eq!(resolve_backend("macos", true), Backend::Metal);
        assert_eq!(resolve_backend("windows", false), Backend::Dx);
        assert_eq!(resolve_backend("windows", true), Backend::Vk);
        assert_eq!(resolve_backend("linux", false), Backend::Vk);
        assert_eq!(resolve_backend("linux", true), Backend::Vk);
    }

    #[test]
    fn backend_cfg_names_are_stable() {
        assert_eq!(Backend::Metal.cfg_name(), "backend_metal");
        assert_eq!(Backend::Dx.cfg_name(), "backend_dx");
        assert_eq!(Backend::Vk.cfg_name(), "backend_vk");
    }

    #[test]
    fn profile_dir_walks_up_from_out_dir() {
        // Backslash paths only parse into components on Windows; on other hosts
        // `Path` treats the whole string as one component, so this case is
        // Windows-only. The Unix-style case below runs everywhere.
        #[cfg(windows)]
        {
            let out =
                Path::new("C:\\proj\\target\\release\\build\\concinnity-client-abcd1234\\out");
            assert_eq!(
                profile_dir_from_out_dir(out),
                Some(Path::new("C:\\proj\\target\\release"))
            );
        }

        let out_debug = Path::new("/proj/target/debug/build/bistro-deadbeef/out");
        assert_eq!(
            profile_dir_from_out_dir(out_debug),
            Some(Path::new("/proj/target/debug"))
        );
    }

    #[test]
    fn profile_dir_none_when_too_shallow() {
        assert_eq!(profile_dir_from_out_dir(Path::new("out")), None);
    }

    #[test]
    fn ancestor_search_finds_marked_dir() {
        let start = Path::new("/a/b/c/d");
        let hit = find_ancestor_with(start, |p| p == Path::new("/a/b"));
        assert_eq!(hit, Some(PathBuf::from("/a/b")));

        let miss = find_ancestor_with(start, |p| p == Path::new("/x"));
        assert_eq!(miss, None);
    }

    #[test]
    fn version_dirs_sort_oldest_to_newest() {
        let dirs = vec![
            PathBuf::from("10.0.22621.0"),
            PathBuf::from("10.0.19041.0"),
            PathBuf::from("10.0.20348.0"),
        ];
        let sorted = sorted_version_dirs(dirs);
        assert_eq!(
            sorted,
            vec![
                PathBuf::from("10.0.19041.0"),
                PathBuf::from("10.0.20348.0"),
                PathBuf::from("10.0.22621.0"),
            ]
        );
    }

    #[test]
    fn glfw_candidate_prefers_vc2022() {
        let candidates = vec![
            PathBuf::from("/out/lib-vc2019/glfw3.dll"),
            PathBuf::from("/out/lib-vc2022/glfw3.dll"),
        ];
        assert_eq!(
            prefer_path_containing(candidates, "lib-vc2022"),
            Some(PathBuf::from("/out/lib-vc2022/glfw3.dll"))
        );
    }

    #[test]
    fn glfw_candidate_falls_back_to_first() {
        let candidates = vec![
            PathBuf::from("/out/lib-mingw/glfw3.dll"),
            PathBuf::from("/out/lib-other/glfw3.dll"),
        ];
        assert_eq!(
            prefer_path_containing(candidates.clone(), "lib-vc2022"),
            Some(PathBuf::from("/out/lib-mingw/glfw3.dll"))
        );
    }

    #[test]
    fn glfw_candidate_empty_is_none() {
        assert_eq!(prefer_path_containing(Vec::new(), "lib-vc2022"), None);
    }
}
