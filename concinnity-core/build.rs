// build.rs
//
// Resolves the rendering backend cfg the same way the client crate's build.rs
// does, so the backend-conditional data layouts in `gfx` (e.g. the Metal-only
// repr(C) structs, the Vulkan-only camera helpers) compile against the same
// backend the client and server pick for the target:
//   backend_metal  macOS (always)
//   backend_dx     Windows, default
//   backend_vk     Linux (always), or Windows with the `vulkan` feature
// The choice must stay in lockstep with concinnity-client/build.rs.
fn main() {
    println!("cargo::rustc-check-cfg=cfg(backend_metal)");
    println!("cargo::rustc-check-cfg=cfg(backend_dx)");
    println!("cargo::rustc-check-cfg=cfg(backend_vk)");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();

    let backend = match (target_os.as_str(), vulkan) {
        ("macos", _) => "backend_metal",
        ("windows", false) => "backend_dx",
        _ => "backend_vk",
    };
    println!("cargo::rustc-cfg={backend}");
}
