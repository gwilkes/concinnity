// build.rs
//
// The Bistro example produces a final binary that links the runtime crate, so
// it has to do the same graphics-SDK setup the editor binary does: link the NGX
// import lib (the runtime's DLSS modules reference its symbols) and bundle the
// runtime DLLs next to the .exe. Without this the binary fails to link against
// the runtime's DLSS code on Windows. All of it is delegated to the shared
// `concinnity-toolchain` helper with `bundle_dlls: true`.
//
// The backend cfg is emitted because main.rs gates the Agility `D3D12SDKVersion`
// / `D3D12SDKPath` export statics on `#[cfg(backend_dx)]`.

use concinnity_toolchain::{SdkOptions, emit_backend_cfg, emit_check_cfgs, setup_graphics_sdks};

fn main() {
    emit_check_cfgs();
    let backend = emit_backend_cfg();
    setup_graphics_sdks(backend, SdkOptions { bundle_dlls: true });
}
