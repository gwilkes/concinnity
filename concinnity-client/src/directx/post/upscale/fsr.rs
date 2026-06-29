// src/directx/post/upscale.rs
//
// AMD FidelityFX FSR3 temporal upscaling for the D3D12 backend. Mirrors
// `metal/post/upscale.rs` (which wraps `MTLFXTemporalScaler`). The engine
// renders the 3D scene at a fraction of drawable size and this pass
// reconstructs a drawable-resolution image the bloom + composite stack
// consumes.
//
// **FFX SDK integration.** This module wraps the
// [AMD FidelityFX SDK v1.1.x unified `ffx_api`](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK)
// at runtime. The runtime DLL is `amd_fidelityfx_dx12.dll`; we
// `LoadLibraryA` it on demand (it must be on `PATH`, the SDK's `bin/`
// directory) and load the five C entry points (`ffxCreateContext` /
// `ffxDestroyContext` / `ffxConfigure` / `ffxQuery` / `ffxDispatch`) via
// `GetProcAddress`. Failure to find the DLL or any entry point logs a
// warning and the caller falls back to native-resolution rendering. The
// FFI bindings live inline at the top of this file because the API
// surface is small (five entry points, ~10 structs) and concentrated.
//
// The scaler does temporal accumulation itself, so the existing TAA pass
// is bypassed while upscaling is on (`PostProcessConfig.aa_mode` is
// ignored). The unified G-buffer pre-pass still runs; FSR consumes its
// motion + depth targets. Projection jitter is still applied, but per FSR's
// `ffxQueryDescUpscaleGetJitterOffset`, not the engine's stock Halton
// sequence (FSR's jitter sequence is tuned to its temporal kernel).
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{CStr, c_void};
use std::ptr;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::core::{Interface, PCSTR};

// FFX API bindings (subset)
//
// Layouts match `C:\FidelityFX-SDK-v1.1.4\ffx-api\include\ffx_api\*.h`.
// Verified against v1.1.4; bump the FFX SDK and re-check.

type ffxContext = *mut c_void;
type ffxReturnCode_t = u32;

const FFX_API_RETURN_OK: u32 = 0;

#[repr(C)]
struct ffxApiHeader {
    // Discriminator (one of `FFX_API_*_DESC_TYPE_*` u64 constants).
    ty: u64,
    // Pointer to next struct in chain (null if none).
    p_next: *mut ffxApiHeader,
}

const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12: u64 = 0x0000002;
const FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE: u64 = 0x00010000;
const FFX_API_DISPATCH_DESC_TYPE_UPSCALE: u64 = 0x00010001;
const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: u64 = 0x00010004;
const FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: u64 = 0x00010005;

const FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1: u64 = 0x0000001;
const FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE: u32 = 0xfffffff;

#[repr(C)]
struct ffxConfigureDescGlobalDebug1 {
    header: ffxApiHeader,
    fp_message: FfxApiMessage,
    debug_level: u32,
}

// Bitmask values from `enum FfxApiCreateContextUpscaleFlags`.
const FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE: u32 = 1 << 0;
const FFX_UPSCALE_ENABLE_DEPTH_INVERTED: u32 = 1 << 3;
const FFX_UPSCALE_ENABLE_DEPTH_INFINITE: u32 = 1 << 4;
const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;

// FfxApiResourceType
const FFX_API_RESOURCE_TYPE_TEXTURE2D: u32 = 2;

// FfxApiResourceUsage (the unused `RENDERTARGET` bit is part of the
// enum surface and kept for completeness even though FSR's inputs are
// all READ_ONLY / UAV / DEPTHTARGET).
const FFX_API_RESOURCE_USAGE_READ_ONLY: u32 = 0;
const FFX_API_RESOURCE_USAGE_UAV: u32 = 1 << 1;
const FFX_API_RESOURCE_USAGE_DEPTHTARGET: u32 = 1 << 2;

// FfxApiResourceState
const FFX_API_RESOURCE_STATE_UNORDERED_ACCESS: u32 = 1 << 1;
const FFX_API_RESOURCE_STATE_COMPUTE_READ: u32 = 1 << 2;

// FfxApiSurfaceFormat
const FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: u32 = 4;
const FFX_API_SURFACE_FORMAT_R32_FLOAT: u32 = 28;
const FFX_API_SURFACE_FORMAT_R16G16_FLOAT: u32 = 18;

#[repr(C)]
#[derive(Clone, Copy)]
struct FfxApiDimensions2D {
    width: u32,
    height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfxApiFloatCoords2D {
    x: f32,
    y: f32,
}

#[repr(C)]
struct FfxApiResourceDescription {
    ty: u32,     // FfxApiResourceType
    format: u32, // FfxApiSurfaceFormat
    width_or_size: u32,
    height_or_stride: u32,
    depth_or_alignment: u32,
    mip_count: u32,
    flags: u32,
    usage: u32,
}

#[repr(C)]
struct FfxApiResource {
    resource: *mut c_void, // ID3D12Resource*
    description: FfxApiResourceDescription,
    state: u32, // FfxApiResourceState
}

impl FfxApiResource {
    fn empty() -> Self {
        Self {
            resource: ptr::null_mut(),
            description: FfxApiResourceDescription {
                ty: 0,
                format: 0,
                width_or_size: 0,
                height_or_stride: 0,
                depth_or_alignment: 0,
                mip_count: 0,
                flags: 0,
                usage: 0,
            },
            state: 0,
        }
    }
}

#[repr(C)]
struct ffxCreateBackendDX12Desc {
    header: ffxApiHeader,
    device: *mut c_void, // ID3D12Device*
}

type FfxApiMessage = Option<extern "C" fn(ty: u32, message: *const u16)>;

// Tracing-backed message sink for FFX errors / warnings. Routes
// FFX's internal diagnostics into the engine's `tracing` log so we
// can see what `ffxCreateContext` is unhappy about instead of just
// taking the C++ exception abort.
extern "C" fn ffx_message_sink(ty: u32, message: *const u16) {
    if message.is_null() {
        return;
    }
    let mut len = 0usize;
    let mut p = message;
    // SAFETY: FFX guarantees null-termination.
    unsafe {
        while *p != 0 {
            len += 1;
            p = p.add(1);
        }
    }
    let slice = unsafe { std::slice::from_raw_parts(message, len) };
    let text = String::from_utf16_lossy(slice);
    match ty {
        // FFX_API_MESSAGE_TYPE_ERROR
        0 => tracing::error!("FFX: {text}"),
        // FFX_API_MESSAGE_TYPE_WARNING
        1 => tracing::warn!("FFX: {text}"),
        other => tracing::info!("FFX[{other}]: {text}"),
    }
}

#[repr(C)]
struct ffxCreateContextDescUpscale {
    header: ffxApiHeader,
    flags: u32,
    max_render_size: FfxApiDimensions2D,
    max_upscale_size: FfxApiDimensions2D,
    fp_message: FfxApiMessage,
}

#[repr(C)]
struct ffxDispatchDescUpscale {
    header: ffxApiHeader,
    command_list: *mut c_void, // ID3D12GraphicsCommandList*
    color: FfxApiResource,
    depth: FfxApiResource,
    motion_vectors: FfxApiResource,
    exposure: FfxApiResource,
    reactive: FfxApiResource,
    transparency_and_composition: FfxApiResource,
    output: FfxApiResource,
    jitter_offset: FfxApiFloatCoords2D,
    motion_vector_scale: FfxApiFloatCoords2D,
    render_size: FfxApiDimensions2D,
    upscale_size: FfxApiDimensions2D,
    enable_sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera_near: f32,
    camera_far: f32,
    camera_fov_angle_vertical: f32,
    view_space_to_meters_factor: f32,
    flags: u32,
}

#[repr(C)]
struct ffxQueryDescUpscaleGetJitterPhaseCount {
    header: ffxApiHeader,
    render_width: u32,
    display_width: u32,
    out_phase_count: *mut i32,
}

#[repr(C)]
struct ffxQueryDescUpscaleGetJitterOffset {
    header: ffxApiHeader,
    index: i32,
    phase_count: i32,
    out_x: *mut f32,
    out_y: *mut f32,
}

#[repr(C)]
struct ffxAllocationCallbacks {
    user_data: *mut c_void,
    alloc: *mut c_void,
    dealloc: *mut c_void,
}

type PfnFfxCreateContext = unsafe extern "C" fn(
    context: *mut ffxContext,
    desc: *mut ffxApiHeader,
    mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t;
type PfnFfxDestroyContext = unsafe extern "C" fn(
    context: *mut ffxContext,
    mem_cb: *const ffxAllocationCallbacks,
) -> ffxReturnCode_t;
type PfnFfxQuery =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *mut ffxApiHeader) -> ffxReturnCode_t;
type PfnFfxDispatch =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxApiHeader) -> ffxReturnCode_t;

type PfnFfxConfigure =
    unsafe extern "C" fn(context: *mut ffxContext, desc: *const ffxApiHeader) -> ffxReturnCode_t;

struct FfxApi {
    #[allow(dead_code)] // Held to keep the DLL loaded for the context's lifetime.
    module: HMODULE,
    create_context: PfnFfxCreateContext,
    destroy_context: PfnFfxDestroyContext,
    configure: PfnFfxConfigure,
    query: PfnFfxQuery,
    dispatch: PfnFfxDispatch,
}

impl FfxApi {
    // Load `amd_fidelityfx_dx12.dll` from the system search path
    // (`%PATH%`, the SDK's `bin/` directory) and resolve the five
    // entry points. Returns `None` on any failure; the caller logs and
    // falls back to native-resolution rendering. The DLL handle is
    // held in the returned struct so the function pointers stay valid
    // for the FFX context's lifetime; nothing in this code path calls
    // `FreeLibrary`.
    fn load() -> Option<Self> {
        let module =
            unsafe { LoadLibraryA(PCSTR(c"amd_fidelityfx_dx12.dll".as_ptr() as *const u8)) }
                .ok()?;
        let resolve = |name: &CStr| -> Option<*const c_void> {
            unsafe {
                GetProcAddress(module, PCSTR(name.as_ptr() as *const u8))
                    .map(|p| p as *const c_void)
            }
        };
        // SAFETY: each entry point's prototype matches the FFX header.
        // GetProcAddress returns null on missing exports, which we surface
        // as None via the `?` operator.
        unsafe {
            Some(FfxApi {
                module,
                create_context: std::mem::transmute::<*const c_void, PfnFfxCreateContext>(resolve(
                    c"ffxCreateContext",
                )?),
                destroy_context: std::mem::transmute::<*const c_void, PfnFfxDestroyContext>(
                    resolve(c"ffxDestroyContext")?,
                ),
                configure: std::mem::transmute::<*const c_void, PfnFfxConfigure>(resolve(
                    c"ffxConfigure",
                )?),
                query: std::mem::transmute::<*const c_void, PfnFfxQuery>(resolve(c"ffxQuery")?),
                dispatch: std::mem::transmute::<*const c_void, PfnFfxDispatch>(resolve(
                    c"ffxDispatch",
                )?),
            })
        }
    }
}

// FsrUpscaler

// Owns the FFX upscale context, the output texture the bloom + composite
// stack consumes (at output resolution), and the FFX function pointer
// table. The context internally owns its own D3D12 pipeline + resources;
// we just feed it per-frame inputs through `ffxDispatch` from
// `encode_upscale`.
pub(in crate::directx) struct FsrUpscaler {
    ffx: FfxApi,
    ctx: ffxContext,

    // Output texture (output-res RGBA16Float, ALLOW_UNORDERED_ACCESS).
    // FFX writes here; the bloom + composite passes sample it.
    pub(in crate::directx) output: ID3D12Resource,
    // SRV gpu handle the post stack samples through (heap slot reserved
    // at init).
    pub(in crate::directx) output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // CPU descriptor handles for the output texture's UAV + SRV. Held so a
    // window resize can recreate the output texture at the new drawable
    // size and rewrite both views into the same pre-reserved heap slots.
    pub(in crate::directx) output_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(in crate::directx) output_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    // Per-axis render-to-output ratio resolved from
    // `PostProcessConfig.upscale_quality`. A resize recomputes the render
    // dims as `output * upscale_scale`, and the upscaler is rebuilt for
    // the new (render, output) pair.
    pub(in crate::directx) upscale_scale: f32,

    // Render-resolution dims passed to FFX every frame (its `renderSize`).
    pub(in crate::directx) render_width: u32,
    pub(in crate::directx) render_height: u32,
    // Output-resolution dims (FFX's `upscaleSize`).
    pub(in crate::directx) output_width: u32,
    pub(in crate::directx) output_height: u32,

    // Number of FFX-prescribed jitter phases for the current
    // (render_width, output_width) pair. The engine reads
    // `jitter_index % phase_count` per frame and asks FFX for the
    // matching offset via `ffxQuery`.
    pub(in crate::directx) jitter_phase_count: i32,
    // `true` when the upscaler was just rebuilt (first frame or after
    // a resize) translates into the FFX dispatch's `reset` flag so
    // the temporal history starts fresh.
    pub(in crate::directx) reset_pending: std::cell::Cell<bool>,
    // Tracks whether `output` is currently in `UNORDERED_ACCESS` (the
    // state init leaves it in and FSR dispatch needs) or
    // `PIXEL_SHADER_RESOURCE` (the state after dispatch ends, where
    // bloom + composite can sample it). Toggled by `encode_upscale`
    // so the worker's top-of-frame barrier is correct on every frame
    // after the first.
    pub(in crate::directx) output_is_psr: std::cell::Cell<bool>,
}

// The FFX context handle + loaded function pointers are raw COM / C pointers
// used only on the render thread; the trait's `Send` bound is satisfied
// unsafely, same as the rest of `DxContext`.
unsafe impl Send for FsrUpscaler {}

// FFX takes a `ID3D12Device*` as a raw COM pointer. Pull it out of the
// `windows`-crate wrapper using `Interface::as_raw`. The lifetime is
// tied to the live `ID3D12Device` the caller still holds; FFX stores
// the pointer and uses it across the context lifetime, so the device
// must stay alive.
fn device_raw(device: &ID3D12Device) -> *mut c_void {
    device.as_raw()
}

fn cmd_list_raw(cmd: &ID3D12GraphicsCommandList) -> *mut c_void {
    cmd.as_raw()
}

fn resource_raw(res: &ID3D12Resource) -> *mut c_void {
    res.as_raw()
}

impl FsrUpscaler {
    // Try to construct an upscaler at the given output resolution +
    // quality. Returns `Ok(None)` when FFX is not available (DLL miss,
    // any entry point missing, context init failed); the caller logs a
    // warning and renders at native resolution. Returns `Ok(Some(...))`
    // on success.
    //
    // `upscale_scale` is the per-axis render-to-output ratio from
    // `PostProcessConfig.upscale_quality` (≤ 1.0). The scene is rendered
    // at `output * upscale_scale` and FSR reconstructs the full output
    // resolution. A `1.0` ratio degenerates to a TAA replacement (no
    // upscale).
    pub(in crate::directx) fn try_new(
        device: &ID3D12Device,
        output_width: u32,
        output_height: u32,
        upscale_scale: f32,
        output_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<Option<Self>, String> {
        // FFX FSR3 needs the Agility SDK configured at build time.
        // Microsoft's `d3d12.dll` reads `D3D12SDKVersion` +
        // `D3D12SDKPath` exports from the host EXE at process start and
        // loads a recent `D3D12Core.dll` from `target/{profile}/D3D12/`;
        // without this the OS-bundled (older) D3D12 runtime is used and
        // `ffxCreateContext` throws (FFX needs SM 6.6+ and post-1.610
        // Agility features). `build.rs` sets `cfg(agility_sdk_configured)`
        // when it locates the SDK and emits the linker exports.
        //
        // Loading `amd_fidelityfx_dx12.dll` itself is gracefully optional:
        // `build.rs` bundles it next to the .exe when found (emitting
        // `cfg(ffx_sdk_bundled)`), but at runtime we just try
        // `FfxApi::load()`; that works for both the bundled DLL and a
        // user-supplied DLL on PATH. Missing DLL → native-res fallback.
        if !cfg!(agility_sdk_configured) {
            tracing::warn!(
                "FidelityFX FSR3: skipping FFX init: `build.rs` did not configure \
                 Microsoft's Agility SDK at build time (set AGILITY_SDK_ROOT or \
                 install the `microsoft.direct3d.d3d12` NuGet package, and ensure \
                 CN_ENABLE_AGILITY_SDK is not 0). Rendering at native resolution."
            );
            return Ok(None);
        }

        let ffx = match FfxApi::load() {
            Some(api) => api,
            None => {
                if cfg!(ffx_sdk_bundled) {
                    tracing::warn!(
                        "FidelityFX FSR3: amd_fidelityfx_dx12.dll was bundled at \
                         build time but failed to load at runtime, falling back \
                         to native-resolution rendering"
                    );
                } else {
                    tracing::warn!(
                        "FidelityFX FSR3: amd_fidelityfx_dx12.dll not found (build.rs \
                         did not bundle it; set FIDELITYFX_SDK_ROOT or put the DLL on \
                         PATH). Falling back to native-resolution rendering."
                    );
                }
                return Ok(None);
            }
        };

        // Resolution split: the scene renders at `output * upscale_scale`
        // and FSR reconstructs the output resolution. FSR's temporal
        // kernel supports up to a 3x per-axis upscale (ratio ≥ 1/3); clamp
        // the requested scale into `[1/3, 1]` so an out-of-range quality
        // preset can't ask FFX for an unsupported ratio. A `1.0` scale
        // makes `render == output` (TAA-replacement mode).
        let scale = if upscale_scale > 0.0 {
            upscale_scale.clamp(1.0 / 3.0, 1.0)
        } else {
            1.0
        };
        let render_width = (((output_width as f32) * scale).round() as u32).max(1);
        let render_height = (((output_height as f32) * scale).round() as u32).max(1);

        // Build the create-context descriptor chain: backend DX12 →
        // upscale spec. FFX walks the `p_next` chain and matches each
        // type against its loaded providers.
        let mut backend = ffxCreateBackendDX12Desc {
            header: ffxApiHeader {
                ty: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12,
                p_next: ptr::null_mut(),
            },
            device: device_raw(device),
        };
        let mut upscale = ffxCreateContextDescUpscale {
            header: ffxApiHeader {
                ty: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                p_next: &mut backend.header as *mut ffxApiHeader,
            },
            // HDR linear input, depth in [0, 1] with reverse-Z not in use
            // (the engine writes 0 at near, 1 at far; Direct3D default).
            // The depth-infinite flag would force FFX's heuristic for
            // skybox masking; we have a real far plane so leave it off.
            // Auto-exposure: FFX computes its own mid-grey heuristic
            // from the colour buffer when this is on, useful because
            // the engine's `PostProcessConfig.auto_exposure` runs after
            // upscaling, so the input scene is *un*-exposed.
            flags: FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE,
            max_render_size: FfxApiDimensions2D {
                width: render_width,
                height: render_height,
            },
            max_upscale_size: FfxApiDimensions2D {
                width: output_width,
                height: output_height,
            },
            fp_message: Some(ffx_message_sink),
        };

        let mut ctx: ffxContext = ptr::null_mut();
        let rc = unsafe {
            (ffx.create_context)(
                &mut ctx,
                &mut upscale.header as *mut ffxApiHeader,
                ptr::null(),
            )
        };
        if rc != FFX_API_RETURN_OK || ctx.is_null() {
            tracing::warn!(
                "FidelityFX FSR3: ffxCreateContext returned {rc}; falling back to native"
            );
            return Ok(None);
        }

        // Raise FFX's diagnostic verbosity on the new context. The per-context
        // message sink is already installed from the create descriptor's
        // `fp_message` (so context-creation errors route through tracing too);
        // this only bumps the level to verbose. `ffxConfigure` dereferences the
        // context's provider, so it must run after `ffxCreateContext`, not
        // before (a null context returns FFX_API_RETURN_ERROR_PARAMETER).
        let mut global_debug = ffxConfigureDescGlobalDebug1 {
            header: ffxApiHeader {
                ty: FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1,
                p_next: ptr::null_mut(),
            },
            fp_message: Some(ffx_message_sink),
            debug_level: FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE,
        };
        let rc_dbg =
            unsafe { (ffx.configure)(&mut ctx, &global_debug.header as *const ffxApiHeader) };
        let _ = &mut global_debug;
        if rc_dbg != FFX_API_RETURN_OK {
            tracing::warn!("FidelityFX FSR3: global debug configure returned {rc_dbg} (non-fatal)");
        }
        tracing::info!(
            "FidelityFX FSR3: context created: render {}x{} -> upscale {}x{} (scale {:.3})",
            render_width,
            render_height,
            output_width,
            output_height,
            scale
        );
        let _ = FFX_UPSCALE_ENABLE_DEPTH_INVERTED;
        let _ = FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
        let _ = FFX_API_RESOURCE_USAGE_DEPTHTARGET;
        let _ = FFX_API_RESOURCE_USAGE_READ_ONLY;
        let _ = FFX_API_SURFACE_FORMAT_R32_FLOAT;

        // Query the jitter phase count once at init; it depends on the
        // ratio, which is fixed for this upscaler instance.
        let mut phase_count: i32 = 0;
        let mut jpc_desc = ffxQueryDescUpscaleGetJitterPhaseCount {
            header: ffxApiHeader {
                ty: FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT,
                p_next: ptr::null_mut(),
            },
            render_width,
            display_width: output_width,
            out_phase_count: &mut phase_count,
        };
        let rc = unsafe { (ffx.query)(&mut ctx, &mut jpc_desc.header as *mut ffxApiHeader) };
        if rc != FFX_API_RETURN_OK || phase_count <= 0 {
            tracing::warn!(
                "FidelityFX FSR3: jitter-phase-count query returned {rc} (phase_count={phase_count})"
            );
            phase_count = 8;
        }

        // Create the output texture FFX writes into: output-res
        // RGBA16Float UAV. Plus an SRV the bloom + composite stack
        // samples from. Both descriptors live in dedicated SRV-heap
        // slots reserved by init.
        let output = super::create_output_texture(device, output_width, output_height)?;
        super::write_output_uav(device, &output, output_uav_cpu);
        super::write_output_srv(device, &output, output_srv_cpu);
        let _ = output_uav_cpu; // descriptor written; the FFX context owns its own UAV inside.

        Ok(Some(FsrUpscaler {
            ffx,
            ctx,
            output,
            output_srv_gpu,
            output_uav_cpu,
            output_srv_cpu,
            upscale_scale: scale,
            render_width,
            render_height,
            output_width,
            output_height,
            jitter_phase_count: phase_count,
            reset_pending: std::cell::Cell::new(true),
            output_is_psr: std::cell::Cell::new(false),
        }))
    }
}

impl super::UpscaleBackend for FsrUpscaler {
    fn render_dims(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }
    fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }
    fn upscale_scale(&self) -> f32 {
        self.upscale_scale
    }
    fn output_srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        self.output_srv_gpu
    }
    fn output_descriptors(
        &self,
    ) -> (
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_GPU_DESCRIPTOR_HANDLE,
    ) {
        (
            self.output_uav_cpu,
            self.output_srv_cpu,
            self.output_srv_gpu,
        )
    }
    fn output_resource(&self) -> &ID3D12Resource {
        &self.output
    }
    fn output_is_psr(&self) -> bool {
        self.output_is_psr.get()
    }
    fn set_output_is_psr(&self, v: bool) {
        self.output_is_psr.set(v);
    }

    // Query FFX for the sub-pixel jitter offset matching this frame's
    // jitter index. Driven from `draw_frame` so the same offset feeds
    // the camera projection (jittered VP) and the FFX dispatch.
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        let mut jx = 0.0_f32;
        let mut jy = 0.0_f32;
        let index = (frame_index as i32).rem_euclid(self.jitter_phase_count.max(1));
        let mut desc = ffxQueryDescUpscaleGetJitterOffset {
            header: ffxApiHeader {
                ty: FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET,
                p_next: ptr::null_mut(),
            },
            index,
            phase_count: self.jitter_phase_count,
            out_x: &mut jx,
            out_y: &mut jy,
        };
        let rc = unsafe {
            // `query` mutates the (caller-owned) context handle if the
            // call type requires it; for the GETJITTEROFFSET case the
            // call is read-only but the C API still takes ctx by
            // pointer-to-pointer. Cast away const at the call site.
            (self.ffx.query)(
                &self.ctx as *const ffxContext as *mut ffxContext,
                &mut desc.header as *mut ffxApiHeader,
            )
        };
        if rc != FFX_API_RETURN_OK {
            return [0.0, 0.0];
        }
        [jx, jy]
    }

    // Dispatch the FFX upscale onto `cmd`. Caller has already
    // transitioned `color` to `NON_PIXEL_SHADER_RESOURCE`, `depth` to
    // `NON_PIXEL_SHADER_RESOURCE`, `motion_vectors` to
    // `NON_PIXEL_SHADER_RESOURCE`, and `output` to `UNORDERED_ACCESS`;
    // FFX recurses through its own internal barriers from these
    // claimed states. Returns the upscaler's output texture so the
    // caller can transition it for the next consumer (bloom / composite
    // read).
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        color: &ID3D12Resource,
        depth: &ID3D12Resource,
        motion_vectors: &ID3D12Resource,
        jitter_offset: [f32; 2],
        frame_time_delta_ms: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y_radians: f32,
    ) -> Result<(), String> {
        let render_size = FfxApiDimensions2D {
            width: self.render_width,
            height: self.render_height,
        };
        let upscale_size = FfxApiDimensions2D {
            width: self.output_width,
            height: self.output_height,
        };

        // Wrap each D3D12 resource as an FfxApiResource. We claim the
        // states the caller transitioned them into; FFX uses these as
        // the assumed "in" states for its internal barrier dance.
        let mk_input =
            |res: &ID3D12Resource, format: u32, usage: u32, state: u32, width: u32, height: u32| {
                FfxApiResource {
                    resource: resource_raw(res),
                    description: FfxApiResourceDescription {
                        ty: FFX_API_RESOURCE_TYPE_TEXTURE2D,
                        format,
                        width_or_size: width,
                        height_or_stride: height,
                        depth_or_alignment: 1,
                        mip_count: 1,
                        flags: 0,
                        usage,
                    },
                    state,
                }
            };

        let color_res = mk_input(
            color,
            FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            FFX_API_RESOURCE_USAGE_READ_ONLY,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            self.render_width,
            self.render_height,
        );
        let depth_res = mk_input(
            depth,
            FFX_API_SURFACE_FORMAT_R32_FLOAT,
            FFX_API_RESOURCE_USAGE_DEPTHTARGET,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            self.render_width,
            self.render_height,
        );
        let mv_res = mk_input(
            motion_vectors,
            FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
            FFX_API_RESOURCE_USAGE_READ_ONLY,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            self.render_width,
            self.render_height,
        );
        let output_res = mk_input(
            &self.output,
            FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            FFX_API_RESOURCE_USAGE_UAV,
            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
            self.output_width,
            self.output_height,
        );

        let reset = self.reset_pending.replace(false);

        let mut desc = ffxDispatchDescUpscale {
            header: ffxApiHeader {
                ty: FFX_API_DISPATCH_DESC_TYPE_UPSCALE,
                p_next: ptr::null_mut(),
            },
            command_list: cmd_list_raw(cmd),
            color: color_res,
            depth: depth_res,
            motion_vectors: mv_res,
            exposure: FfxApiResource::empty(),
            reactive: FfxApiResource::empty(),
            transparency_and_composition: FfxApiResource::empty(),
            output: output_res,
            jitter_offset: FfxApiFloatCoords2D {
                x: jitter_offset[0],
                y: jitter_offset[1],
            },
            // Motion vectors are stored as `prev_uv - cur_uv` in UV
            // space (RG16Float). FSR expects them in input-pixel
            // coordinates, so per-axis scale = the render-resolution
            // extent.
            motion_vector_scale: FfxApiFloatCoords2D {
                x: self.render_width as f32,
                y: self.render_height as f32,
            },
            render_size,
            upscale_size,
            enable_sharpening: false,
            sharpness: 0.0,
            frame_time_delta: frame_time_delta_ms,
            pre_exposure: 1.0,
            reset,
            camera_near,
            camera_far,
            camera_fov_angle_vertical: camera_fov_y_radians,
            view_space_to_meters_factor: 1.0,
            flags: 0,
        };

        let rc = unsafe {
            (self.ffx.dispatch)(
                &self.ctx as *const ffxContext as *mut ffxContext,
                &desc.header as *const ffxApiHeader,
            )
        };
        // Touch `desc` past the dispatch so the compiler doesn't
        // re-order any of its initialisation past the call.
        let _ = &mut desc.header;
        if rc != FFX_API_RETURN_OK {
            return Err(format!("ffxDispatch (upscale) returned {rc}"));
        }
        Ok(())
    }
}

impl crate::directx::context::DxContext {
    // Encode the FSR3 temporal upscale onto `cmd`. Runs after SSR
    // resolve / Fog / ParticlesDraw (so the input scene is the
    // fully-decorated post-SSR colour) and before Bloom + Composite
    // (which sample the upscaler's output via `scene_srv_for_post`).
    // Caller (the executor) already routed this onto the
    // `PassId::Upscale` per-pass cmd list.
    //
    // The dispatch passes the scene's `render_size` (the off-screen scene
    // resolution) and the larger `upscale_size` (drawable resolution);
    // FSR reconstructs the latter from the former. When the quality preset
    // resolves to a `1.0` scale the two are equal and FSR degenerates to a
    // TAA replacement.
    pub(in crate::directx) fn encode_upscale(
        &self,
        cmd: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
        params: &crate::directx::graph_exec::GraphFrameParams<'_>,
    ) -> Result<(), String> {
        use windows::Win32::Graphics::Direct3D12::*;
        let upscaler = match &self.upscale.backend {
            Some(u) => u,
            None => return Ok(()),
        };
        // One-time info log: confirms the worker arm is firing. Tracing
        // is rate-limited via a static AtomicBool so we don't spam every
        // frame.
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            let (rw, rh) = upscaler.render_dims();
            let (ow, oh) = upscaler.output_dims();
            tracing::info!(
                "temporal upscaling: first encode_upscale firing (render_size={rw}x{rh}, upscale_size={ow}x{oh})"
            );
        }
        let gb = match &self.gbuffer {
            Some(g) => g,
            None => {
                // Init forces the G-buffer to be built when upscale is on (it
                // owns the velocity + depth FSR consumes); the only way to hit
                // this branch is a programming error in init.
                return Err(
                    "Upscale enabled but G-buffer resources (velocity / depth) are missing".into(),
                );
            }
        };

        // Inputs:
        //   scene  : the post-SSR scene the bloom/composite stack would
        //            normally sample. SSR resolve writes into
        //            `ssr.resolve.output` when SSR is on; otherwise the head
        //            of the hdr_resolve RMW chain is the raw resolved HDR
        //            target (which SSGI, if on, has already composited into).
        //            In both cases the resource is in PIXEL_SHADER_RESOURCE
        //            state after its writer.
        //   depth  : `gbuffer.depth` (single-sample D32F at render-res;
        //            the unified G-buffer pre-pass writes it).
        //   mv     : `gbuffer.velocity` (RG16F screen-space motion).
        let (scene_res, scene_was_in_psr) = match self.ssr.as_ref().and_then(|s| s.resolve.as_ref())
        {
            Some(r) => (r.output.clone(), true),
            None => match &self.hdr.resolve {
                Some(r) => (r.clone(), true),
                None => (self.hdr.color.clone(), true),
            },
        };
        let _ = scene_was_in_psr;

        // Transition inputs PSR → NON_PSR for FSR's compute reads, and
        // remember to restore them PSR-side after (so any later passes,
        // and the end-of-frame restores in `record_frame`, find them
        // where they expect). The upscaler's output texture stays in
        // UNORDERED_ACCESS the entire frame except for the bloom /
        // composite sample window, which we flip to NON_PSR after the
        // dispatch.
        let to_npsr_scene = transition_barrier(
            &scene_res,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let to_npsr_velo = transition_barrier(
            &gb.velocity,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        // gbuffer.depth is in DEPTH_WRITE after the G-buffer pre-pass (no
        // intermediate transition). Flip it to NON_PSR for FSR's read.
        let to_npsr_depth = transition_barrier(
            &gb.depth,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let mut barriers = vec![to_npsr_scene, to_npsr_velo, to_npsr_depth];
        // After the *previous* frame's dispatch we left `output` in
        // PIXEL_SHADER_RESOURCE so Bloom + Composite could sample it.
        // FSR's dispatch needs it back in UNORDERED_ACCESS; flip it
        // unless this is the first frame (output starts in UAV at init).
        if upscaler.output_is_psr() {
            barriers.push(transition_barrier(
                upscaler.output_resource(),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
        }
        unsafe { cmd.ResourceBarrier(&barriers) };

        // Per-frame inputs FSR consumes. Clamp dt to a sane range:
        // `upscale_prev_elapsed` initialises to 0.0, so the first
        // frame's raw `(now - prev)` is whatever elapsed-since-startup
        // happens to be (often seconds), and a stalled frame can yield
        // arbitrarily large values too. FSR's heuristics expect
        // frame-time-ish numbers; clamp to [1, 100] ms (10-1000 Hz).
        let jitter = self.upscale.jitter.get();
        let now = params.elapsed;
        let prev = self.upscale.prev_elapsed.replace(now);
        let dt_ms = ((now - prev) * 1000.0).clamp(1.0, 100.0);

        let near = params.near.max(1e-3);
        let far = params.far.max(near + 1.0);
        let fov_y = params.fov_y_radians;

        upscaler.dispatch(
            cmd,
            &scene_res,
            &gb.depth,
            &gb.velocity,
            jitter,
            dt_ms,
            near,
            far,
            fov_y,
        )?;

        // Restore the inputs to the states downstream consumers (and
        // the end-of-frame restores) expect: scene + G-buffer velocity
        // PSR, G-buffer depth back to DEPTH_WRITE.
        let from_npsr_scene = transition_barrier(
            &scene_res,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let from_npsr_velo = transition_barrier(
            &gb.velocity,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        let from_npsr_depth = transition_barrier(
            &gb.depth,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        // Flip the upscaler output UAV → PIXEL_SHADER_RESOURCE so
        // Bloom + Composite can sample it. Track the new state on the
        // upscaler so the next frame's top-of-encode barrier knows to
        // flip back.
        let output_to_psr = transition_barrier(
            upscaler.output_resource(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        upscaler.set_output_is_psr(true);
        unsafe {
            cmd.ResourceBarrier(&[
                from_npsr_scene,
                from_npsr_velo,
                from_npsr_depth,
                output_to_psr,
            ])
        };

        Ok(())
    }
}

fn transition_barrier(
    resource: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    before: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
    after: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER {
    crate::directx::texture::transition_barrier(resource, before, after)
}

impl Drop for FsrUpscaler {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // Destroy via the loaded entry point; the DLL stays
            // mapped until our `FfxApi` drops (which is just a
            // few-byte struct, not a resource).
            unsafe {
                let _ = (self.ffx.destroy_context)(&mut self.ctx, ptr::null());
            }
            self.ctx = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // Validates the FFX struct layouts match the C definitions byte-for-byte.
    // If the FFX SDK is bumped and these sizes change, the FFI is wrong.
    #[test]
    fn ffx_struct_sizes_match_sdk_v114() {
        // ffxApiHeader: ty + p_next = 8 + 8 = 16
        assert_eq!(size_of::<ffxApiHeader>(), 16);
        // FfxApiDimensions2D: 2 * u32 = 8
        assert_eq!(size_of::<FfxApiDimensions2D>(), 8);
        // FfxApiFloatCoords2D: 2 * f32 = 8
        assert_eq!(size_of::<FfxApiFloatCoords2D>(), 8);
        // FfxApiResourceDescription: 8 * u32 = 32
        assert_eq!(size_of::<FfxApiResourceDescription>(), 32);
        // FfxApiResource: void* + description + state + pad
        //   = 8 + 32 + 4 + 4 = 48 (last u32 + 4-byte tail padding to 8-byte alignment)
        assert_eq!(size_of::<FfxApiResource>(), 48);
    }
}
