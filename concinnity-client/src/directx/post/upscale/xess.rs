// src/directx/post/upscale/xess.rs
//
// Intel XeSS temporal upscaling for the D3D12 backend. One of the three
// `UpscaleBackend` implementations; runs cross-vendor (Arc XMX + the DP4a
// fallback on other GPUs). The runtime DLL is `libxess.dll`, loaded on demand
// via `LoadLibraryA` (bundled next to the .exe by `build.rs` when the XeSS SDK
// is found, else searched on PATH). Failure to load logs a warning and the
// caller falls through to the next backend / native rendering. The FFI
// bindings are inline (small, concentrated API), validated against XeSS SDK
// 3.0.1 by the size/offset asserts in the tests.
#![allow(non_camel_case_types)]

use std::ffi::{CStr, c_void};
use std::ptr;

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};
use windows::core::{Interface, PCSTR};

// XeSS API bindings (subset). Layouts match
// `C:\XeSS_SDK_3.0.1\inc\xess\{xess.h,xess_d3d12.h}` (v3.0.1). `XESS_PACK_B()`
// is `pack(8)`, a no-op on x86_64 where every field is already <= 8-aligned, so
// `#[repr(C)]` matches byte-for-byte (asserted in tests).

// xess_result_t: 0 == success, negative == error, positive == warning.
const XESS_RESULT_SUCCESS: i32 = 0;

// xess_quality_settings_t (a C enum, ABI int).
const XESS_QUALITY_SETTING_ULTRA_PERFORMANCE: i32 = 100;
const XESS_QUALITY_SETTING_PERFORMANCE: i32 = 101;
const XESS_QUALITY_SETTING_BALANCED: i32 = 102;
const XESS_QUALITY_SETTING_QUALITY: i32 = 103;
const XESS_QUALITY_SETTING_AA: i32 = 106;

// xess_init_flags_t (bitmask). The engine feeds HDR linear colour, low-res
// (render-resolution) UV motion vectors scaled to pixels via SetVelocityScale,
// and depth in [0,1] with 0 = near (NOT inverted). So the only flag we set is
// auto-exposure (the scene is un-exposed pre-upscale, matching the FSR path).
const XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE: u32 = 1 << 8;

type xess_context_handle_t = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct xess_2d_t {
    x: u32,
    y: u32,
}

#[repr(C)]
struct xess_d3d12_init_params_t {
    output_resolution: xess_2d_t,
    quality_setting: i32,
    init_flags: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
    p_temp_buffer_heap: *mut c_void,
    buffer_heap_offset: u64,
    p_temp_texture_heap: *mut c_void,
    texture_heap_offset: u64,
    p_pipeline_library: *mut c_void,
}

#[repr(C)]
struct xess_d3d12_execute_params_t {
    p_color_texture: *mut c_void,
    p_velocity_texture: *mut c_void,
    p_depth_texture: *mut c_void,
    p_exposure_scale_texture: *mut c_void,
    p_responsive_pixel_mask_texture: *mut c_void,
    p_output_texture: *mut c_void,
    jitter_offset_x: f32,
    jitter_offset_y: f32,
    exposure_scale: f32,
    reset_history: u32,
    input_width: u32,
    input_height: u32,
    input_color_base: xess_2d_t,
    input_motion_vector_base: xess_2d_t,
    input_depth_base: xess_2d_t,
    input_responsive_mask_base: xess_2d_t,
    reserved0: xess_2d_t,
    output_color_base: xess_2d_t,
    p_descriptor_heap: *mut c_void,
    descriptor_heap_offset: u32,
}

type PfnXessD3D12CreateContext =
    unsafe extern "C" fn(device: *mut c_void, out_ctx: *mut xess_context_handle_t) -> i32;
type PfnXessD3D12BuildPipelines = unsafe extern "C" fn(
    ctx: xess_context_handle_t,
    pipeline_lib: *mut c_void,
    blocking: bool,
    init_flags: u32,
) -> i32;
type PfnXessD3D12Init = unsafe extern "C" fn(
    ctx: xess_context_handle_t,
    params: *const xess_d3d12_init_params_t,
) -> i32;
type PfnXessD3D12Execute = unsafe extern "C" fn(
    ctx: xess_context_handle_t,
    cmd: *mut c_void,
    params: *const xess_d3d12_execute_params_t,
) -> i32;
type PfnXessDestroyContext = unsafe extern "C" fn(ctx: xess_context_handle_t) -> i32;
type PfnXessSetVelocityScale =
    unsafe extern "C" fn(ctx: xess_context_handle_t, x: f32, y: f32) -> i32;

struct XessApi {
    #[allow(dead_code)] // Held to keep the DLL loaded for the context's lifetime.
    module: HMODULE,
    create_context: PfnXessD3D12CreateContext,
    build_pipelines: PfnXessD3D12BuildPipelines,
    init: PfnXessD3D12Init,
    execute: PfnXessD3D12Execute,
    destroy_context: PfnXessDestroyContext,
    set_velocity_scale: PfnXessSetVelocityScale,
}

impl XessApi {
    // Load `libxess.dll` and resolve the entry points. Returns `None` on any
    // failure; the caller logs and falls through. The DLL handle is held so the
    // function pointers stay valid for the context's lifetime.
    fn load() -> Option<Self> {
        let module = unsafe { LoadLibraryA(PCSTR(c"libxess.dll".as_ptr() as *const u8)) }.ok()?;
        let resolve = |name: &CStr| -> Option<*const c_void> {
            unsafe {
                GetProcAddress(module, PCSTR(name.as_ptr() as *const u8))
                    .map(|p| p as *const c_void)
            }
        };
        // SAFETY: each prototype matches the XeSS header for SDK 3.0.1.
        unsafe {
            Some(XessApi {
                module,
                create_context: std::mem::transmute::<*const c_void, PfnXessD3D12CreateContext>(
                    resolve(c"xessD3D12CreateContext")?,
                ),
                build_pipelines: std::mem::transmute::<*const c_void, PfnXessD3D12BuildPipelines>(
                    resolve(c"xessD3D12BuildPipelines")?,
                ),
                init: std::mem::transmute::<*const c_void, PfnXessD3D12Init>(resolve(
                    c"xessD3D12Init",
                )?),
                execute: std::mem::transmute::<*const c_void, PfnXessD3D12Execute>(resolve(
                    c"xessD3D12Execute",
                )?),
                destroy_context: std::mem::transmute::<*const c_void, PfnXessDestroyContext>(
                    resolve(c"xessDestroyContext")?,
                ),
                set_velocity_scale: std::mem::transmute::<*const c_void, PfnXessSetVelocityScale>(
                    resolve(c"xessSetVelocityScale")?,
                ),
            })
        }
    }
}

fn device_raw(device: &ID3D12Device) -> *mut c_void {
    device.as_raw()
}
fn cmd_list_raw(cmd: &ID3D12GraphicsCommandList) -> *mut c_void {
    cmd.as_raw()
}
fn resource_raw(res: &ID3D12Resource) -> *mut c_void {
    res.as_raw()
}

// Map the engine's per-axis render-to-output ratio to the nearest XeSS quality
// preset. The preset is a hint for XeSS's internal model selection; the actual
// render dims are `output * scale` (passed via `inputWidth/Height`). Pure; unit
// tested.
fn quality_from_scale(scale: f32) -> i32 {
    if scale >= 0.99 {
        XESS_QUALITY_SETTING_AA
    } else if scale >= 0.62 {
        XESS_QUALITY_SETTING_QUALITY
    } else if scale >= 0.55 {
        XESS_QUALITY_SETTING_BALANCED
    } else if scale >= 0.42 {
        XESS_QUALITY_SETTING_PERFORMANCE
    } else {
        XESS_QUALITY_SETTING_ULTRA_PERFORMANCE
    }
}

// Owns the XeSS context, the output texture the bloom + composite stack
// consumes (at output resolution), and the loaded function table. Mirrors
// `FsrUpscaler`.
pub(in crate::directx) struct XessUpscaler {
    xess: XessApi,
    ctx: xess_context_handle_t,
    output: ID3D12Resource,
    output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    output_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    output_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    upscale_scale: f32,
    render_width: u32,
    render_height: u32,
    output_width: u32,
    output_height: u32,
    reset_pending: std::cell::Cell<bool>,
    output_is_psr: std::cell::Cell<bool>,
}

// The XeSS context handle + loaded function pointers are raw C pointers used
// only on the render thread; the trait's `Send` bound is satisfied unsafely,
// same as the rest of `DxContext`.
unsafe impl Send for XessUpscaler {}

impl XessUpscaler {
    // Try to construct an XeSS upscaler. Returns `Ok(None)` when XeSS is
    // unavailable (DLL miss / context init failure); the caller falls through.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn try_new(
        device: &ID3D12Device,
        output_width: u32,
        output_height: u32,
        upscale_scale: f32,
        output_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<Option<Self>, String> {
        let xess = match XessApi::load() {
            Some(api) => api,
            None => {
                tracing::warn!(
                    "XeSS: libxess.dll not found (build.rs did not bundle it; set \
                     XESS_SDK_ROOT or put the DLL on PATH). Trying the next upscaler."
                );
                return Ok(None);
            }
        };

        // Same render/output split as FSR: render at `output * scale`, clamp
        // into [1/3, 1] (XeSS supports up to ~3x per-axis).
        let scale = if upscale_scale > 0.0 {
            upscale_scale.clamp(1.0 / 3.0, 1.0)
        } else {
            1.0
        };
        let render_width = (((output_width as f32) * scale).round() as u32).max(1);
        let render_height = (((output_height as f32) * scale).round() as u32).max(1);

        let mut ctx: xess_context_handle_t = ptr::null_mut();
        let rc = unsafe { (xess.create_context)(device_raw(device), &mut ctx) };
        if rc != XESS_RESULT_SUCCESS || ctx.is_null() {
            tracing::warn!("XeSS: xessD3D12CreateContext returned {rc}; trying the next upscaler");
            return Ok(None);
        }

        let init_flags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
        let rc = unsafe { (xess.build_pipelines)(ctx, ptr::null_mut(), true, init_flags) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS: xessD3D12BuildPipelines returned {rc}; trying the next upscaler");
            unsafe { (xess.destroy_context)(ctx) };
            return Ok(None);
        }

        let init_params = xess_d3d12_init_params_t {
            output_resolution: xess_2d_t {
                x: output_width,
                y: output_height,
            },
            quality_setting: quality_from_scale(scale),
            init_flags,
            creation_node_mask: 0,
            visible_node_mask: 0,
            p_temp_buffer_heap: ptr::null_mut(),
            buffer_heap_offset: 0,
            p_temp_texture_heap: ptr::null_mut(),
            texture_heap_offset: 0,
            p_pipeline_library: ptr::null_mut(),
        };
        let rc = unsafe { (xess.init)(ctx, &init_params) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS: xessD3D12Init returned {rc}; trying the next upscaler");
            unsafe { (xess.destroy_context)(ctx) };
            return Ok(None);
        }

        // Motion vectors are RG16F `prev_uv - cur_uv` in UV space; XeSS expects
        // pixel-space velocity (default low-res), so scale by the render extent.
        let rc =
            unsafe { (xess.set_velocity_scale)(ctx, render_width as f32, render_height as f32) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS: xessSetVelocityScale returned {rc} (non-fatal)");
        }

        let output = super::create_output_texture(device, output_width, output_height)?;
        super::write_output_uav(device, &output, output_uav_cpu);
        super::write_output_srv(device, &output, output_srv_cpu);

        tracing::info!(
            "XeSS: context created: render {render_width}x{render_height} -> upscale \
             {output_width}x{output_height} (scale {scale:.3})"
        );

        Ok(Some(XessUpscaler {
            xess,
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
            reset_pending: std::cell::Cell::new(true),
            output_is_psr: std::cell::Cell::new(false),
        }))
    }
}

impl super::UpscaleBackend for XessUpscaler {
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

    // XeSS prescribes no jitter sequence; the engine's Halton-2/3 (shared with
    // the camera projection) drives both. XeSS wants the offset in [-0.5, 0.5].
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        super::halton_jitter_offset(frame_index)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        color: &ID3D12Resource,
        depth: &ID3D12Resource,
        motion_vectors: &ID3D12Resource,
        jitter_offset: [f32; 2],
        _frame_time_delta_ms: f32,
        _camera_near: f32,
        _camera_far: f32,
        _camera_fov_y_radians: f32,
    ) -> Result<(), String> {
        let reset = self.reset_pending.replace(false);
        let zero = xess_2d_t { x: 0, y: 0 };
        let params = xess_d3d12_execute_params_t {
            p_color_texture: resource_raw(color),
            p_velocity_texture: resource_raw(motion_vectors),
            p_depth_texture: resource_raw(depth),
            p_exposure_scale_texture: ptr::null_mut(),
            p_responsive_pixel_mask_texture: ptr::null_mut(),
            p_output_texture: resource_raw(&self.output),
            jitter_offset_x: jitter_offset[0],
            jitter_offset_y: jitter_offset[1],
            exposure_scale: 1.0,
            reset_history: if reset { 1 } else { 0 },
            input_width: self.render_width,
            input_height: self.render_height,
            input_color_base: zero,
            input_motion_vector_base: zero,
            input_depth_base: zero,
            input_responsive_mask_base: zero,
            reserved0: zero,
            output_color_base: zero,
            p_descriptor_heap: ptr::null_mut(),
            descriptor_heap_offset: 0,
        };
        let rc = unsafe { (self.xess.execute)(self.ctx, cmd_list_raw(cmd), &params) };
        if rc != XESS_RESULT_SUCCESS {
            return Err(format!("xessD3D12Execute returned {rc}"));
        }
        Ok(())
    }
}

impl Drop for XessUpscaler {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                let _ = (self.xess.destroy_context)(self.ctx);
            }
            self.ctx = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn xess_struct_sizes_match_sdk_v301() {
        assert_eq!(size_of::<xess_2d_t>(), 8);
        assert_eq!(size_of::<xess_d3d12_init_params_t>(), 64);
        assert_eq!(size_of::<xess_d3d12_execute_params_t>(), 136);

        assert_eq!(offset_of!(xess_d3d12_init_params_t, quality_setting), 8);
        assert_eq!(offset_of!(xess_d3d12_init_params_t, init_flags), 12);
        assert_eq!(offset_of!(xess_d3d12_init_params_t, p_temp_buffer_heap), 24);
        assert_eq!(offset_of!(xess_d3d12_init_params_t, p_pipeline_library), 56);

        assert_eq!(
            offset_of!(xess_d3d12_execute_params_t, p_output_texture),
            40
        );
        assert_eq!(offset_of!(xess_d3d12_execute_params_t, jitter_offset_x), 48);
        assert_eq!(offset_of!(xess_d3d12_execute_params_t, reset_history), 60);
        assert_eq!(offset_of!(xess_d3d12_execute_params_t, input_width), 64);
        assert_eq!(
            offset_of!(xess_d3d12_execute_params_t, input_color_base),
            72
        );
        assert_eq!(
            offset_of!(xess_d3d12_execute_params_t, p_descriptor_heap),
            120
        );
        assert_eq!(
            offset_of!(xess_d3d12_execute_params_t, descriptor_heap_offset),
            128
        );
    }

    #[test]
    fn xess_quality_mapping_is_monotonic_by_scale() {
        assert_eq!(quality_from_scale(1.0), XESS_QUALITY_SETTING_AA);
        assert_eq!(quality_from_scale(2.0 / 3.0), XESS_QUALITY_SETTING_QUALITY);
        assert_eq!(quality_from_scale(0.587), XESS_QUALITY_SETTING_BALANCED);
        assert_eq!(quality_from_scale(0.5), XESS_QUALITY_SETTING_PERFORMANCE);
        assert_eq!(
            quality_from_scale(1.0 / 3.0),
            XESS_QUALITY_SETTING_ULTRA_PERFORMANCE
        );
    }
}
