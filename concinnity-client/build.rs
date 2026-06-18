// build.rs
//
// The runtime crate (rlib). Two jobs, both delegated to the shared
// `concinnity-toolchain` build helper:
//
// 1. Resolve the rendering backend once and expose it as a single cfg the crate
//    gates on (`backend_metal` / `backend_dx` / `backend_vk`).
//
// 2. Detect the optional upscaler SDKs and emit the cfgs the renderer gates on.
//    This crate produces only an rlib (consumed by the editor and the examples)
//    plus its own test binaries, so it does NOT bundle runtime DLLs next to a
//    binary (that belongs to whichever package owns the final artifact). The one
//    link directive kept is the NGX import lib: the DLSS modules are
//    `#[cfg(ngx_sdk_bundled)]`, so when that cfg is on they compile into the lib
//    and must resolve their NGX symbols when this crate's test binaries link.
//    That is `SdkOptions { bundle_dlls: false }`.

use concinnity_toolchain::{SdkOptions, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: false });
}
