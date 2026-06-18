// build.rs
//
// Three jobs:
//
// 1. Resolve the rendering backend once and expose it as a single cfg the crate
//    gates on (`backend_metal` / `backend_dx` / `backend_vk`), via the shared
//    `concinnity-toolchain` helper.
//
// 2. Detect the optional graphics SDKs and, because this crate owns the final
//    `concinnity` binary and the FFI cdylib, bundle their runtime DLLs next to
//    the artifact and emit the Agility linker exports (the `D3D12SDKVersion` /
//    `D3D12SDKPath` statics those exports name are defined in src/main.rs). That
//    is `SdkOptions { bundle_dlls: true }`.
//
// 3. Generate the C FFI header (../include/concinnity.h) from the extern "C"
//    surface in src/ffi.rs via cbindgen.

use std::path::Path;

use concinnity_toolchain::{SdkOptions, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: true });
    generate_ffi_header();
}

// Generate the C FFI header (../include/concinnity.h) from the extern "C"
// surface in src/ffi.rs via cbindgen.
fn generate_ffi_header() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ffi_src = Path::new(&crate_dir).join("src/ffi.rs");
    let out_dir = Path::new(&crate_dir).join("../include");
    let out_header = out_dir.join("concinnity.h");

    println!("cargo::rerun-if-changed=src/ffi.rs");
    println!("cargo::rerun-if-changed=cbindgen.toml");

    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        println!("cargo::warning=cbindgen: cannot create {out_dir:?}: {e}");
        return;
    }

    let config = cbindgen::Config::from_root_or_default(&crate_dir);
    match cbindgen::Builder::new()
        .with_src(&ffi_src)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&out_header);
        }
        Err(e) => {
            println!("cargo::warning=cbindgen: failed to generate {out_header:?}: {e}");
        }
    }
}
