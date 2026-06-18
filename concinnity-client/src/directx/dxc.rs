// src/directx/dxc.rs
//
// Runtime HLSL -> DXIL compilation via the DirectX Shader Compiler
// (`dxcompiler.dll`). The built-in FXC path (`pipeline::compile_hlsl`) tops out
// at shader model 5.1, which cannot express DXR 1.1 inline ray tracing
// (`RayQuery`); the hardware-ray-traced reflection shader needs SM 6.5, so it is
// compiled here instead.
//
// `dxcompiler.dll` (and its companion `dxil.dll`, which DXC invokes to validate
// + sign the produced container so the D3D12 runtime accepts it) are bundled
// next to the .exe by `build.rs::setup_dxc_sdk`. Loading is best-effort: when the
// DLL is absent the compile returns an `Err` and the caller leaves RT reflections
// off, falling back to the screen-space SSR path. The library is loaded per
// compile and dropped at the end (so the COM objects release into a still-loaded
// DLL); RT shaders compile only at init + on hot-reload, so the load cost is
// irrelevant.

use windows::Win32::Graphics::Direct3D::Dxc::{
    CLSID_DxcCompiler, DXC_CP_UTF8, DxcBuffer, IDxcBlob, IDxcBlobUtf8, IDxcResult,
};
use windows::core::{GUID, HRESULT, Interface, PCWSTR};

// Signature of `dxcompiler.dll`'s exported `DxcCreateInstance`. Resolved by
// name at runtime so the build never link-depends on `dxcompiler.lib`.
type DxcCreateInstanceFn =
    unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut std::ffi::c_void) -> HRESULT;

// Encode a Rust string as a null-terminated UTF-16 buffer for a `PCWSTR` arg.
fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

// Compile `source` (UTF-8 HLSL) to a signed DXIL blob via DXC. `entry` is the
// entry-point name; `target` is a shader-model profile string such as `"ps_6_5"`
// or `"vs_6_5"`. Returns the DXIL container bytes ready for a D3D12
// `D3D12_SHADER_BYTECODE`, or an `Err` carrying the DXC diagnostic text (or the
// reason the compiler could not be loaded).
pub(super) fn compile_hlsl_dxc(source: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
    // SAFETY: `dxcompiler.dll` is a normal Win32 DLL; loading it and resolving
    // `DxcCreateInstance` is the documented usage. The library is held in `_lib`
    // until the end of the function so every COM `Release` (compiler / result /
    // blob drops) runs while the DLL is still mapped.
    let lib = unsafe { libloading::Library::new("dxcompiler.dll") }
        .map_err(|e| format!("load dxcompiler.dll: {e} (DXC not bundled; RT falls back to SSR)"))?;
    let create: libloading::Symbol<DxcCreateInstanceFn> =
        unsafe { lib.get(b"DxcCreateInstance\0") }
            .map_err(|e| format!("resolve DxcCreateInstance: {e}"))?;

    let compiler: windows::Win32::Graphics::Direct3D::Dxc::IDxcCompiler3 = unsafe {
        let mut ptr = std::ptr::null_mut();
        let hr = create(
            &CLSID_DxcCompiler,
            &windows::Win32::Graphics::Direct3D::Dxc::IDxcCompiler3::IID,
            &mut ptr,
        );
        hr.ok()
            .map_err(|e| format!("DxcCreateInstance(compiler): {e}"))?;
        windows::Win32::Graphics::Direct3D::Dxc::IDxcCompiler3::from_raw(ptr)
    };

    // Build the argument list. `-Zpc` forces column-major matrix packing to
    // match the `#pragma pack_matrix(column_major)` the shader sets and the
    // column-major `[[f32; 4]; 4]` layouts the Rust side uploads (same default
    // the FXC path's `D3DCOMPILE_PACK_MATRIX_COLUMN_MAJOR` flag enforces).
    let entry_w = to_wide(entry);
    let target_w = to_wide(target);
    let args_owned: Vec<Vec<u16>> = vec![
        to_wide("-E"),
        entry_w,
        to_wide("-T"),
        target_w,
        to_wide("-Zpc"),
    ];
    let args: Vec<PCWSTR> = args_owned.iter().map(|w| PCWSTR(w.as_ptr())).collect();

    let buffer = DxcBuffer {
        Ptr: source.as_ptr() as *const std::ffi::c_void,
        Size: source.len(),
        Encoding: DXC_CP_UTF8.0,
    };

    let result: IDxcResult = unsafe { compiler.Compile(&buffer, Some(&args), None) }
        .map_err(|e| format!("DXC compile {target}: {e}"))?;

    // A non-zero status means the shader failed to compile; surface the DXC
    // diagnostic text (the error blob), mirroring the FXC path's error dump.
    let status: HRESULT =
        unsafe { result.GetStatus() }.map_err(|e| format!("DXC GetStatus {target}: {e}"))?;
    if status.is_err() {
        let mut errors: Option<IDxcBlobUtf8> = None;
        let _ = unsafe {
            result.GetOutput::<IDxcBlobUtf8>(
                windows::Win32::Graphics::Direct3D::Dxc::DXC_OUT_ERRORS,
                &mut errors as *mut _ as *mut _,
                std::ptr::null_mut(),
            )
        };
        let msg = errors
            .map(|b| {
                let ptr = unsafe { b.GetStringPointer() };
                if ptr.is_null() {
                    String::new()
                } else {
                    unsafe { std::ffi::CStr::from_ptr(ptr.0 as *const i8) }
                        .to_string_lossy()
                        .into_owned()
                }
            })
            .unwrap_or_else(|| "unknown DXC error".to_string());
        return Err(format!("DXC compile {target} ({entry}): {msg}"));
    }

    // The object output is the signed DXIL container ready for D3D12.
    let object: IDxcBlob =
        unsafe { result.GetResult() }.map_err(|e| format!("DXC GetResult {target}: {e}"))?;
    let ptr = unsafe { object.GetBufferPointer() } as *const u8;
    let len = unsafe { object.GetBufferSize() };
    if ptr.is_null() || len == 0 {
        return Err(format!("DXC compile {target}: empty object blob"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec();
    // `lib` (declared first) drops last, after `object` / `result` / `compiler`
    // release their COM refs into the still-mapped DLL. Keep it owned until here.
    Ok(bytes)
}
