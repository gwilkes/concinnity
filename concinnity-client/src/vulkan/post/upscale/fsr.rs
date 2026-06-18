// src/vulkan/post/upscale/fsr.rs
//
// AMD FidelityFX FSR temporal upscaling for the Vulkan backend. Mirrors
// `directx/post/upscale/fsr.rs` (the FFX `ffx_api` is shared between the DX12
// and Vulkan backends; only the backend-create descriptor and the resource
// handle types differ). One of the three `VkUpscaleBackend` implementations;
// the cross-vendor default that every other backend falls back to.
//
// **FFX SDK integration.** Wraps the AMD FidelityFX SDK v1.1.x unified
// `ffx_api` at runtime. The runtime library is `amd_fidelityfx_vk.dll`
// (Windows) / `libamd_fidelityfx_vk.so` (Linux); it is loaded on demand via
// `libloading` and the five C entry points (`ffxCreateContext` /
// `ffxDestroyContext` / `ffxConfigure` / `ffxQuery` / `ffxDispatch`) are
// resolved by symbol. Failure to find the library or any entry point logs a
// warning and `try_new` returns `None`; `build_upscaler` then falls back to the
// next backend / native-resolution rendering. The FFI bindings live inline
// because the surface is small (five entry points, ~10 structs); the only delta
// from the DX module is `ffxCreateBackendVKDesc` and the handles being
// `VkDevice` / `VkPhysicalDevice` / `VkImage` / `VkCommandBuffer`.
//
// The scaler does temporal accumulation itself, so the TAA resolve is bypassed
// while upscaling is on (the frame graph drops `TaaResolve` and runs `Upscale`
// in its slot). The velocity pre-pass still runs; FSR consumes its
// render-resolution motion + depth targets. Projection jitter is still applied,
// but per FSR's `ffxQueryDescUpscaleGetJitterOffset`, not the engine's stock
// Halton sequence (FSR's jitter is tuned to its temporal kernel).
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::cell::Cell;
use std::ffi::c_void;
use std::ptr;

use ash::vk::Handle;
use ash::{Device, vk};

use super::{UpscaleImage, VkUpscaleBackend};
use crate::vulkan::texture::GpuImage;

// FFX API bindings (subset)
//
// Layouts match `C:\FidelityFX-SDK-v1.1.4\ffx-api\include\ffx_api\*.h` and
// `.../vk/ffx_api_vk.h`. Verified against v1.1.4; bump the FFX SDK and
// re-check (the `ffx_struct_sizes_match_sdk_v114` test guards drift).

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

// Vulkan backend (the one delta from the DX module, where this is
// `FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 = 0x2`).
const FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK: u64 = 0x0000003;
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
const FFX_UPSCALE_ENABLE_AUTO_EXPOSURE: u32 = 1 << 5;

// FfxApiResourceType
const FFX_API_RESOURCE_TYPE_TEXTURE2D: u32 = 2;

// FfxApiResourceUsage
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
    resource: *mut c_void, // VkImage handle (as a raw pointer-sized value)
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

// Vulkan backend-create descriptor. `ffx_api_vk.h`:
//   struct ffxCreateBackendVKDesc {
//       ffxCreateContextDescHeader header;
//       VkDevice                   vkDevice;
//       VkPhysicalDevice           vkPhysicalDevice;
//       PFN_vkGetDeviceProcAddr    vkDeviceProcAddr;
//   };
// The three handles are passed as raw pointer-sized values; the FFX VK
// backend loads its own VK entry points through `vkDeviceProcAddr`.
#[repr(C)]
struct ffxCreateBackendVKDesc {
    header: ffxApiHeader,
    vk_device: *mut c_void,
    vk_physical_device: *mut c_void,
    vk_device_proc_addr: *mut c_void,
}

type FfxApiMessage = Option<extern "C" fn(ty: u32, message: *const u16)>;

// Tracing-backed message sink for FFX errors / warnings (Windows only: FFX
// passes `wchar_t*`, which is 2 bytes on Windows but 4 on Linux, so the
// u16 decode is only correct there; on other platforms we pass `None` and
// rely on FFX return codes).
#[cfg(windows)]
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
        0 => tracing::error!("FFX: {text}"),
        1 => tracing::warn!("FFX: {text}"),
        other => tracing::info!("FFX[{other}]: {text}"),
    }
}

// The per-context / global message callback, present only on Windows.
fn message_callback() -> FfxApiMessage {
    #[cfg(windows)]
    {
        Some(ffx_message_sink)
    }
    #[cfg(not(windows))]
    {
        None
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
    command_list: *mut c_void, // VkCommandBuffer handle
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
    // Held to keep the library mapped for the context's lifetime. The function
    // pointers below are copied out of it and stay valid as long as this stays
    // alive.
    _lib: libloading::Library,
    create_context: PfnFfxCreateContext,
    destroy_context: PfnFfxDestroyContext,
    configure: PfnFfxConfigure,
    query: PfnFfxQuery,
    dispatch: PfnFfxDispatch,
}

impl FfxApi {
    // Load the FFX Vulkan runtime and resolve the five entry points. Returns
    // `None` on any failure; the caller logs and falls back.
    fn load() -> Option<Self> {
        let lib_name = if cfg!(windows) {
            "amd_fidelityfx_vk.dll"
        } else {
            "libamd_fidelityfx_vk.so"
        };
        // SAFETY: loading a system library + reading well-known C symbols.
        // The symbols' prototypes match the FFX header; GetProcAddress-style
        // misses surface as `None` via `?`.
        unsafe {
            let lib = libloading::Library::new(lib_name).ok()?;
            let create_context = *lib.get::<PfnFfxCreateContext>(b"ffxCreateContext\0").ok()?;
            let destroy_context = *lib
                .get::<PfnFfxDestroyContext>(b"ffxDestroyContext\0")
                .ok()?;
            let configure = *lib.get::<PfnFfxConfigure>(b"ffxConfigure\0").ok()?;
            let query = *lib.get::<PfnFfxQuery>(b"ffxQuery\0").ok()?;
            let dispatch = *lib.get::<PfnFfxDispatch>(b"ffxDispatch\0").ok()?;
            Some(FfxApi {
                _lib: lib,
                create_context,
                destroy_context,
                configure,
                query,
                dispatch,
            })
        }
    }
}

// Owns the FFX upscale context, the display-resolution output texture the bloom
// + composite stack samples, and the FFX function-pointer table. The FFX
// context internally owns its own Vulkan pipelines + history resources; we feed
// it per-frame inputs through `ffxDispatch` from `encode_upscale`.
pub(in crate::vulkan) struct FsrUpscaler {
    ffx: FfxApi,
    ctx: ffxContext,

    // Output texture FFX writes (display-res RGBA16F, STORAGE | SAMPLED). Lives
    // in `GENERAL` while FFX writes it and `SHADER_READ_ONLY_OPTIMAL` while
    // bloom + composite sample it; `output_layout` tracks which.
    output: GpuImage,
    output_layout: Cell<vk::ImageLayout>,

    // Render-resolution dims passed to FFX every frame (its `renderSize`).
    render_width: u32,
    render_height: u32,
    // Output- (display-) resolution dims (FFX's `upscaleSize`).
    output_width: u32,
    output_height: u32,
    // Per-axis render-to-output ratio actually used (clamped).
    upscale_scale: f32,

    // Number of FFX-prescribed jitter phases for this (render, output) pair.
    jitter_phase_count: i32,
    // This frame's FSR-prescribed jitter offset (render-pixel units), set by
    // `draw.rs` before the parallel fan-out and read by `encode_upscale`.
    jitter: Cell<[f32; 2]>,
    // Previous frame's elapsed-seconds stamp, for the FFX frame delta.
    prev_elapsed: Cell<f32>,
    // `true` on the first frame / after a resize: forces FFX's `reset` so the
    // temporal history starts fresh.
    reset_pending: Cell<bool>,
}

// The FFX context handle + loaded function pointers are raw C pointers used only
// on the render thread (the upscale pass is recorded by exactly one
// parallel-encoder worker per frame); the `Send` bound is satisfied unsafely,
// same as the rest of `VkContext`.
unsafe impl Send for FsrUpscaler {}

impl FsrUpscaler {
    // Try to construct an FSR upscaler at the given output resolution + quality.
    // Returns `Ok(None)` when FFX is unavailable (library miss, any entry point
    // missing, context init failed); `build_upscaler` then falls through.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_new(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        output_width: u32,
        output_height: u32,
        upscale_scale: f32,
    ) -> Result<Option<Self>, String> {
        let ffx = match FfxApi::load() {
            Some(api) => api,
            None => {
                if cfg!(ffx_sdk_bundled) {
                    tracing::warn!(
                        "FidelityFX FSR (Vulkan): amd_fidelityfx_vk.dll was bundled at build \
                         time but failed to load at runtime; trying the next backend"
                    );
                } else {
                    tracing::warn!(
                        "FidelityFX FSR (Vulkan): amd_fidelityfx_vk.dll not found (build.rs did \
                         not bundle it; set FIDELITYFX_SDK_ROOT or put the DLL on PATH). \
                         Trying the next backend."
                    );
                }
                return Ok(None);
            }
        };

        let (render_width, render_height, scale) =
            super::resolve_render_dims(output_width, output_height, upscale_scale);

        // Build the create-context descriptor chain: backend VK -> upscale spec.
        let mut backend = ffxCreateBackendVKDesc {
            header: ffxApiHeader {
                ty: FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK,
                p_next: ptr::null_mut(),
            },
            vk_device: device.handle().as_raw() as usize as *mut c_void,
            vk_physical_device: physical_device.as_raw() as usize as *mut c_void,
            vk_device_proc_addr: instance.fp_v1_0().get_device_proc_addr as usize as *mut c_void,
        };
        let mut upscale = ffxCreateContextDescUpscale {
            header: ffxApiHeader {
                ty: FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE,
                p_next: &mut backend.header as *mut ffxApiHeader,
            },
            // HDR linear input; FFX runs its own auto-exposure heuristic from
            // the colour buffer. Depth is the standard Vulkan [0, 1] range (not
            // reverse-Z), so no depth-inverted / depth-infinite flags.
            flags: FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE,
            // `maxRenderSize` is the upper bound on the per-frame render size;
            // the FSR sample sets it to the display size, so mirror that. The
            // actual reduced render size is passed per-frame in the dispatch.
            max_render_size: FfxApiDimensions2D {
                width: output_width,
                height: output_height,
            },
            max_upscale_size: FfxApiDimensions2D {
                width: output_width,
                height: output_height,
            },
            fp_message: message_callback(),
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
                "FidelityFX FSR (Vulkan): ffxCreateContext returned {rc}; trying the next backend"
            );
            return Ok(None);
        }

        // Raise FFX's diagnostic verbosity on the new context. `ffxConfigure`
        // dereferences the context's provider, so it must run after
        // `ffxCreateContext`.
        let mut global_debug = ffxConfigureDescGlobalDebug1 {
            header: ffxApiHeader {
                ty: FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1,
                p_next: ptr::null_mut(),
            },
            fp_message: message_callback(),
            debug_level: FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE,
        };
        let rc_dbg =
            unsafe { (ffx.configure)(&mut ctx, &global_debug.header as *const ffxApiHeader) };
        let _ = &mut global_debug;
        if rc_dbg != FFX_API_RETURN_OK {
            tracing::warn!(
                "FidelityFX FSR (Vulkan): global debug configure returned {rc_dbg} (non-fatal)"
            );
        }
        tracing::info!(
            "FidelityFX FSR (Vulkan): context created: render {}x{} -> upscale {}x{} (scale {:.3})",
            render_width,
            render_height,
            output_width,
            output_height,
            scale
        );

        // Query the jitter phase count once; it depends on the (fixed) ratio.
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
                "FidelityFX FSR (Vulkan): jitter-phase-count query returned {rc} (phase_count={phase_count})"
            );
            phase_count = 8;
        }

        // Output texture FFX writes into.
        let output = match super::create_output_image(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            output_width,
            output_height,
        ) {
            Ok(img) => img,
            Err(e) => {
                unsafe {
                    let _ = (ffx.destroy_context)(&mut ctx, ptr::null());
                }
                return Err(e);
            }
        };

        Ok(Some(FsrUpscaler {
            ffx,
            ctx,
            output,
            output_layout: Cell::new(vk::ImageLayout::GENERAL),
            render_width,
            render_height,
            output_width,
            output_height,
            upscale_scale: scale,
            jitter_phase_count: phase_count,
            jitter: Cell::new([0.0, 0.0]),
            prev_elapsed: Cell::new(0.0),
            reset_pending: Cell::new(true),
        }))
    }
}

impl VkUpscaleBackend for FsrUpscaler {
    fn render_dims(&self) -> (u32, u32) {
        (self.render_width, self.render_height)
    }
    fn output_dims(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }
    fn scale(&self) -> f32 {
        self.upscale_scale
    }
    fn output_image(&self) -> &GpuImage {
        &self.output
    }
    fn output_layout(&self) -> vk::ImageLayout {
        self.output_layout.get()
    }
    fn set_output_layout(&self, layout: vk::ImageLayout) {
        self.output_layout.set(layout);
    }
    fn set_jitter(&self, offset: [f32; 2]) {
        self.jitter.set(offset);
    }
    fn jitter(&self) -> [f32; 2] {
        self.jitter.get()
    }

    // Query FFX for the sub-pixel jitter offset matching this frame's index.
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

    // Record the FFX upscale dispatch onto `cmd`. The inputs must already be in
    // the layouts/states `encode_upscale` arranged; FFX records its own internal
    // barriers from the declared states. FSR consumes only the raw image handles
    // (FFX is told each input's format + render dims).
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        color: &UpscaleImage,
        depth: &UpscaleImage,
        motion: &UpscaleImage,
        jitter_offset: [f32; 2],
        elapsed: f32,
        camera_near: f32,
        camera_far: f32,
        camera_fov_y_radians: f32,
    ) -> Result<(), String> {
        let mk = |image: vk::Image, format: u32, usage: u32, state: u32, w: u32, h: u32| {
            FfxApiResource {
                resource: image.as_raw() as usize as *mut c_void,
                description: FfxApiResourceDescription {
                    ty: FFX_API_RESOURCE_TYPE_TEXTURE2D,
                    format,
                    width_or_size: w,
                    height_or_stride: h,
                    depth_or_alignment: 1,
                    mip_count: 1,
                    flags: 0,
                    usage,
                },
                state,
            }
        };

        let color_res = mk(
            color.image,
            FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            FFX_API_RESOURCE_USAGE_READ_ONLY,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            self.render_width,
            self.render_height,
        );
        let depth_res = mk(
            depth.image,
            FFX_API_SURFACE_FORMAT_R32_FLOAT,
            FFX_API_RESOURCE_USAGE_DEPTHTARGET,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            self.render_width,
            self.render_height,
        );
        let mv_res = mk(
            motion.image,
            FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
            FFX_API_RESOURCE_USAGE_READ_ONLY,
            FFX_API_RESOURCE_STATE_COMPUTE_READ,
            self.render_width,
            self.render_height,
        );
        let output_res = mk(
            self.output.image,
            FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
            FFX_API_RESOURCE_USAGE_UAV,
            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS,
            self.output_width,
            self.output_height,
        );

        let reset = self.reset_pending.replace(false);
        let dt_ms = super::frame_delta_ms(&self.prev_elapsed, elapsed);

        let mut desc = ffxDispatchDescUpscale {
            header: ffxApiHeader {
                ty: FFX_API_DISPATCH_DESC_TYPE_UPSCALE,
                p_next: ptr::null_mut(),
            },
            command_list: cmd.as_raw() as usize as *mut c_void,
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
            // Velocity is stored as `prev_uv - cur_uv` in UV space (RG16F); FSR
            // expects motion in input-pixel coords, so per-axis scale = the
            // render-resolution extent.
            motion_vector_scale: FfxApiFloatCoords2D {
                x: self.render_width as f32,
                y: self.render_height as f32,
            },
            render_size: FfxApiDimensions2D {
                width: self.render_width,
                height: self.render_height,
            },
            upscale_size: FfxApiDimensions2D {
                width: self.output_width,
                height: self.output_height,
            },
            enable_sharpening: false,
            sharpness: 0.0,
            frame_time_delta: dt_ms,
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
        let _ = &mut desc.header;
        if rc != FFX_API_RETURN_OK {
            return Err(format!("ffxDispatch (upscale, vulkan) returned {rc}"));
        }
        Ok(())
    }

    fn destroy(&mut self, device: &Device) {
        if !self.ctx.is_null() {
            unsafe {
                let _ = (self.ffx.destroy_context)(&mut self.ctx, ptr::null());
            }
            self.ctx = ptr::null_mut();
        }
        self.output.destroy(device);
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
        assert_eq!(size_of::<ffxApiHeader>(), 16);
        assert_eq!(size_of::<FfxApiDimensions2D>(), 8);
        assert_eq!(size_of::<FfxApiFloatCoords2D>(), 8);
        assert_eq!(size_of::<FfxApiResourceDescription>(), 32);
        // void* + description + state + tail pad = 8 + 32 + 4 + 4 = 48.
        assert_eq!(size_of::<FfxApiResource>(), 48);
        // header (16) + 3 pointer-sized handles = 16 + 24 = 40.
        assert_eq!(size_of::<ffxCreateBackendVKDesc>(), 40);
    }
}
