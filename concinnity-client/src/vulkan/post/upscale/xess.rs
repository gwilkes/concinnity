// src/vulkan/post/upscale/xess.rs
//
// Intel XeSS temporal upscaling for the Vulkan backend. One of the three
// `VkUpscaleBackend` implementations; runs cross-vendor (Arc XMX + the DP4a
// fallback on other GPUs). The runtime DLL is `libxess.dll` (it carries the
// `xessVK*` entry points alongside the D3D12 ones), loaded on demand via
// `libloading` (cross-platform, mirroring the FSR module's choice over the DX
// path's `LoadLibraryA`). Failure to load logs a warning and `try_new` returns
// `None`; `build_upscaler` then falls through.
//
// Unlike FSR, XeSS needs Vulkan instance + device extensions (and a device
// feature chain) enabled at instance / device creation. Those are queried up
// front through `XessExtQuery` (held by `UpscaleSdk`, see `mod.rs`); this module
// only creates the upscale context after the device exists.
//
// The FFI bindings are inline (small, concentrated API), validated against XeSS
// SDK 3.0.1 (`inc/xess/{xess.h,xess_vk.h}`) by the size/offset asserts in the
// tests. `XESS_PACK_B()` is `pack(8)`, a no-op on x86_64 where every field is
// already <= 8-aligned, so `#[repr(C)]` matches byte-for-byte.
#![allow(non_camel_case_types)]

use std::cell::Cell;
use std::ffi::{CString, c_char, c_void};
use std::ptr;

use ash::{Device, vk};

use super::{UpscaleImage, VkUpscaleBackend, copy_ext_names};
use crate::vulkan::context::HDR_FORMAT;
use crate::vulkan::texture::GpuImage;

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
// and depth in [0,1] with 0 = near (NOT inverted). So the only flag set is
// auto-exposure (the scene is un-exposed pre-upscale, matching the FSR path).
const XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE: u32 = 1 << 8;

type xess_context_handle_t = *mut c_void;

// xess_coord_t is a typedef of xess_2d_t (xess.h:87): { uint32_t x, y }.
#[repr(C)]
#[derive(Clone, Copy)]
struct xess_2d_t {
    x: u32,
    y: u32,
}

// xess_vk_image_view_info (xess_vk.h). VkImageView / VkImage are
// non-dispatchable 64-bit handles; VkImageSubresourceRange is 5 u32s (20 B);
// VkFormat is an ABI int. 48 bytes under pack(8).
#[repr(C)]
#[derive(Clone, Copy)]
struct xess_vk_image_view_info {
    image_view: vk::ImageView,
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

impl xess_vk_image_view_info {
    // An absent optional input (exposure scale / responsive mask): null
    // handles, no flags set so XeSS ignores it.
    fn empty() -> Self {
        Self {
            image_view: vk::ImageView::null(),
            image: vk::Image::null(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::empty(),
                base_mip_level: 0,
                level_count: 0,
                base_array_layer: 0,
                layer_count: 0,
            },
            format: vk::Format::UNDEFINED,
            width: 0,
            height: 0,
        }
    }

    fn from_input(img: &UpscaleImage) -> Self {
        Self {
            image_view: img.view,
            image: img.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: img.aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            format: img.format,
            width: img.width,
            height: img.height,
        }
    }
}

#[repr(C)]
struct xess_vk_init_params_t {
    output_resolution: xess_2d_t,
    quality_setting: i32,
    init_flags: u32,
    creation_node_mask: u32,
    visible_node_mask: u32,
    temp_buffer_heap: vk::DeviceMemory,
    buffer_heap_offset: u64,
    temp_texture_heap: vk::DeviceMemory,
    texture_heap_offset: u64,
    pipeline_cache: vk::PipelineCache,
}

#[repr(C)]
struct xess_vk_execute_params_t {
    color_texture: xess_vk_image_view_info,
    velocity_texture: xess_vk_image_view_info,
    depth_texture: xess_vk_image_view_info,
    exposure_scale_texture: xess_vk_image_view_info,
    responsive_pixel_mask_texture: xess_vk_image_view_info,
    output_texture: xess_vk_image_view_info,
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
}

// Extension / context entry points. XESS_API is a bare dllimport (no explicit
// __cdecl/__stdcall), so on x86_64 Windows `extern "C"` is the only convention.
type PfnXessVKGetRequiredInstanceExtensions =
    unsafe extern "C" fn(*mut u32, *mut *const *const c_char, *mut u32) -> i32;
type PfnXessVKGetRequiredDeviceExtensions = unsafe extern "C" fn(
    vk::Instance,
    vk::PhysicalDevice,
    *mut u32,
    *mut *const *const c_char,
) -> i32;
type PfnXessVKGetRequiredDeviceFeatures =
    unsafe extern "C" fn(vk::Instance, vk::PhysicalDevice, *mut *mut c_void) -> i32;
type PfnXessVKCreateContext = unsafe extern "C" fn(
    vk::Instance,
    vk::PhysicalDevice,
    vk::Device,
    *mut xess_context_handle_t,
) -> i32;
type PfnXessVKBuildPipelines =
    unsafe extern "C" fn(xess_context_handle_t, vk::PipelineCache, bool, u32) -> i32;
type PfnXessVKInit =
    unsafe extern "C" fn(xess_context_handle_t, *const xess_vk_init_params_t) -> i32;
type PfnXessVKExecute = unsafe extern "C" fn(
    xess_context_handle_t,
    vk::CommandBuffer,
    *const xess_vk_execute_params_t,
) -> i32;
type PfnXessDestroyContext = unsafe extern "C" fn(xess_context_handle_t) -> i32;
type PfnXessSetVelocityScale = unsafe extern "C" fn(xess_context_handle_t, f32, f32) -> i32;

fn lib_name() -> &'static str {
    if cfg!(windows) {
        "libxess.dll"
    } else {
        "libxess.so"
    }
}

// Pre-device extension / feature queries (XeSS-specific). Held by `UpscaleSdk`
// across instance + device creation so the SDK-owned device-feature chain
// (`device_features`) stays mapped through `vkCreateDevice`.
pub(super) struct XessExtQuery {
    _lib: libloading::Library,
    get_instance_exts: PfnXessVKGetRequiredInstanceExtensions,
    get_device_exts: PfnXessVKGetRequiredDeviceExtensions,
    get_device_features: PfnXessVKGetRequiredDeviceFeatures,
}

impl XessExtQuery {
    pub(super) fn load() -> Option<Self> {
        // SAFETY: loading a system library + reading well-known C symbols whose
        // prototypes match the XeSS SDK 3.0.1 headers; misses surface as `None`.
        unsafe {
            let lib = libloading::Library::new(lib_name()).ok()?;
            let get_instance_exts = *lib
                .get::<PfnXessVKGetRequiredInstanceExtensions>(
                    b"xessVKGetRequiredInstanceExtensions\0",
                )
                .ok()?;
            let get_device_exts = *lib
                .get::<PfnXessVKGetRequiredDeviceExtensions>(b"xessVKGetRequiredDeviceExtensions\0")
                .ok()?;
            let get_device_features = *lib
                .get::<PfnXessVKGetRequiredDeviceFeatures>(b"xessVKGetRequiredDeviceFeatures\0")
                .ok()?;
            Some(XessExtQuery {
                _lib: lib,
                get_instance_exts,
                get_device_exts,
                get_device_features,
            })
        }
    }

    // Returns XeSS's required instance extensions and the minimum Vulkan API
    // version it needs (XeSS 3.x shaders use SPV_KHR_integer_dot_product, which
    // requires a Vulkan 1.3 environment). The caller raises the instance
    // `apiVersion` to at least this (clamped to loader support).
    pub(super) fn instance_extensions(&self) -> (Vec<CString>, u32) {
        let mut count: u32 = 0;
        let mut exts: *const *const c_char = ptr::null();
        let mut min_api: u32 = 0;
        let rc = unsafe { (self.get_instance_exts)(&mut count, &mut exts, &mut min_api) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS: xessVKGetRequiredInstanceExtensions returned {rc}");
            return (Vec::new(), 0);
        }
        (unsafe { copy_ext_names(count, exts) }, min_api)
    }

    pub(super) fn device_extensions(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Vec<CString> {
        let mut count: u32 = 0;
        let mut exts: *const *const c_char = ptr::null();
        let rc = unsafe {
            (self.get_device_exts)(instance.handle(), physical_device, &mut count, &mut exts)
        };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS: xessVKGetRequiredDeviceExtensions returned {rc}");
            return Vec::new();
        }
        unsafe { copy_ext_names(count, exts) }
    }

    // Patch the device-feature `pNext` chain with XeSS's required features and
    // return the (possibly new) chain head, to be set as `VkDeviceCreateInfo.pNext`.
    // `head` is the caller's existing chain; the returned memory the SDK adds is
    // owned by libxess.dll and valid while `self` lives. On failure the caller's
    // `head` is returned unchanged.
    pub(super) fn device_features(
        &self,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        head: *mut c_void,
    ) -> *mut c_void {
        let mut chain = head;
        let rc =
            unsafe { (self.get_device_features)(instance.handle(), physical_device, &mut chain) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!(
                "XeSS: xessVKGetRequiredDeviceFeatures returned {rc}; using base features"
            );
            return head;
        }
        chain
    }
}

// The full XeSS context API, loaded once at context creation.
struct XessApi {
    _lib: libloading::Library,
    create_context: PfnXessVKCreateContext,
    build_pipelines: PfnXessVKBuildPipelines,
    init: PfnXessVKInit,
    execute: PfnXessVKExecute,
    destroy_context: PfnXessDestroyContext,
    set_velocity_scale: PfnXessSetVelocityScale,
}

impl XessApi {
    fn load() -> Option<Self> {
        // SAFETY: see `XessExtQuery::load`.
        unsafe {
            let lib = libloading::Library::new(lib_name()).ok()?;
            let create_context = *lib
                .get::<PfnXessVKCreateContext>(b"xessVKCreateContext\0")
                .ok()?;
            let build_pipelines = *lib
                .get::<PfnXessVKBuildPipelines>(b"xessVKBuildPipelines\0")
                .ok()?;
            let init = *lib.get::<PfnXessVKInit>(b"xessVKInit\0").ok()?;
            let execute = *lib.get::<PfnXessVKExecute>(b"xessVKExecute\0").ok()?;
            let destroy_context = *lib
                .get::<PfnXessDestroyContext>(b"xessDestroyContext\0")
                .ok()?;
            let set_velocity_scale = *lib
                .get::<PfnXessSetVelocityScale>(b"xessSetVelocityScale\0")
                .ok()?;
            Some(XessApi {
                _lib: lib,
                create_context,
                build_pipelines,
                init,
                execute,
                destroy_context,
                set_velocity_scale,
            })
        }
    }
}

// Map the engine's per-axis render-to-output ratio to the nearest XeSS quality
// preset. The preset is a hint for XeSS's internal model selection; the actual
// render dims are `output * scale` (passed via `inputWidth/Height`). Pure; unit
// tested. Mirrors `directx::post::upscale::xess::quality_from_scale`.
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

// Owns the XeSS context, the output texture the bloom + composite stack samples
// (at output resolution), and the loaded function table.
pub(in crate::vulkan) struct XessUpscaler {
    xess: XessApi,
    ctx: xess_context_handle_t,

    output: GpuImage,
    output_layout: Cell<vk::ImageLayout>,

    render_width: u32,
    render_height: u32,
    output_width: u32,
    output_height: u32,
    upscale_scale: f32,

    jitter: Cell<[f32; 2]>,
    reset_pending: Cell<bool>,
}

// The XeSS context handle + loaded function pointers are raw C pointers used
// only on the render thread; the trait's `Send` bound is satisfied unsafely,
// same as the rest of `VkContext`.
unsafe impl Send for XessUpscaler {}

impl XessUpscaler {
    // Try to construct an XeSS upscaler. Returns `Ok(None)` when XeSS is
    // unavailable (DLL miss / context init failure); `build_upscaler` falls
    // through. Assumes the XeSS instance / device extensions + features were
    // enabled at creation time (via `UpscaleSdk`); if they were not, the
    // context create / init below fails and we fall back.
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
        let xess = match XessApi::load() {
            Some(api) => api,
            None => {
                tracing::warn!(
                    "XeSS (Vulkan): libxess.dll not found (build.rs did not bundle it; set \
                     XESS_SDK_ROOT or put the DLL on PATH). Trying the next backend."
                );
                return Ok(None);
            }
        };

        let (render_width, render_height, scale) =
            super::resolve_render_dims(output_width, output_height, upscale_scale);

        let mut ctx: xess_context_handle_t = ptr::null_mut();
        let rc = unsafe {
            (xess.create_context)(
                instance.handle(),
                physical_device,
                device.handle(),
                &mut ctx,
            )
        };
        if rc != XESS_RESULT_SUCCESS || ctx.is_null() {
            tracing::warn!(
                "XeSS (Vulkan): xessVKCreateContext returned {rc}; trying the next backend"
            );
            return Ok(None);
        }

        let init_flags = XESS_INIT_FLAG_ENABLE_AUTOEXPOSURE;
        let rc =
            unsafe { (xess.build_pipelines)(ctx, vk::PipelineCache::null(), true, init_flags) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!(
                "XeSS (Vulkan): xessVKBuildPipelines returned {rc}; trying the next backend"
            );
            unsafe { (xess.destroy_context)(ctx) };
            return Ok(None);
        }

        let init_params = xess_vk_init_params_t {
            output_resolution: xess_2d_t {
                x: output_width,
                y: output_height,
            },
            quality_setting: quality_from_scale(scale),
            init_flags,
            creation_node_mask: 0,
            visible_node_mask: 0,
            temp_buffer_heap: vk::DeviceMemory::null(),
            buffer_heap_offset: 0,
            temp_texture_heap: vk::DeviceMemory::null(),
            texture_heap_offset: 0,
            pipeline_cache: vk::PipelineCache::null(),
        };
        let rc = unsafe { (xess.init)(ctx, &init_params) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS (Vulkan): xessVKInit returned {rc}; trying the next backend");
            unsafe { (xess.destroy_context)(ctx) };
            return Ok(None);
        }

        // Motion vectors are RG16F `prev_uv - cur_uv` in UV space; XeSS expects
        // pixel-space velocity (default low-res), so scale by the render extent.
        let rc =
            unsafe { (xess.set_velocity_scale)(ctx, render_width as f32, render_height as f32) };
        if rc != XESS_RESULT_SUCCESS {
            tracing::warn!("XeSS (Vulkan): xessSetVelocityScale returned {rc} (non-fatal)");
        }

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
                unsafe { (xess.destroy_context)(ctx) };
                return Err(e);
            }
        };

        tracing::info!(
            "XeSS (Vulkan): context created: render {render_width}x{render_height} -> upscale \
             {output_width}x{output_height} (scale {scale:.3})"
        );

        Ok(Some(XessUpscaler {
            xess,
            ctx,
            output,
            output_layout: Cell::new(vk::ImageLayout::GENERAL),
            render_width,
            render_height,
            output_width,
            output_height,
            upscale_scale: scale,
            jitter: Cell::new([0.0, 0.0]),
            reset_pending: Cell::new(true),
        }))
    }
}

impl VkUpscaleBackend for XessUpscaler {
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

    // XeSS prescribes no jitter sequence; the engine's Halton-2/3 (shared with
    // the camera projection) drives both. XeSS wants the offset in [-0.5, 0.5].
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2] {
        super::halton_jitter_offset(frame_index)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        color: &UpscaleImage,
        depth: &UpscaleImage,
        motion: &UpscaleImage,
        jitter_offset: [f32; 2],
        _elapsed: f32,
        _camera_near: f32,
        _camera_far: f32,
        _camera_fov_y_radians: f32,
    ) -> Result<(), String> {
        let reset = self.reset_pending.replace(false);
        let zero = xess_2d_t { x: 0, y: 0 };
        let output_view = xess_vk_image_view_info {
            image_view: self.output.view,
            image: self.output.image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            format: HDR_FORMAT,
            width: self.output_width,
            height: self.output_height,
        };
        let params = xess_vk_execute_params_t {
            color_texture: xess_vk_image_view_info::from_input(color),
            velocity_texture: xess_vk_image_view_info::from_input(motion),
            depth_texture: xess_vk_image_view_info::from_input(depth),
            exposure_scale_texture: xess_vk_image_view_info::empty(),
            responsive_pixel_mask_texture: xess_vk_image_view_info::empty(),
            output_texture: output_view,
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
        };
        let rc = unsafe { (self.xess.execute)(self.ctx, cmd, &params) };
        if rc != XESS_RESULT_SUCCESS {
            return Err(format!("xessVKExecute returned {rc}"));
        }
        Ok(())
    }

    fn destroy(&mut self, device: &Device) {
        if !self.ctx.is_null() {
            unsafe {
                let _ = (self.xess.destroy_context)(self.ctx);
            }
            self.ctx = ptr::null_mut();
        }
        self.output.destroy(device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    // Pin the XeSS VK struct layouts against the SDK 3.0.1 headers. These differ
    // from the D3D12 structs (image-view-info vs raw resource pointers), so they
    // are the most likely silent ABI mismatch.
    #[test]
    fn xess_vk_struct_sizes_match_sdk_v301() {
        // VkImageView(8) + VkImage(8) + VkImageSubresourceRange(20) +
        // VkFormat(4) + width(4) + height(4) = 48.
        assert_eq!(size_of::<xess_vk_image_view_info>(), 48);
        assert_eq!(offset_of!(xess_vk_image_view_info, image_view), 0);
        assert_eq!(offset_of!(xess_vk_image_view_info, image), 8);
        assert_eq!(offset_of!(xess_vk_image_view_info, subresource_range), 16);
        assert_eq!(offset_of!(xess_vk_image_view_info, format), 36);
        assert_eq!(offset_of!(xess_vk_image_view_info, width), 40);
        assert_eq!(offset_of!(xess_vk_image_view_info, height), 44);

        // xess_2d_t = 2 u32.
        assert_eq!(size_of::<xess_2d_t>(), 8);

        // init params: outputResolution(8) + quality(4) + flags(4) + 2 masks(8)
        // + 3 handles(24) + 2 offsets(16) = 64.
        assert_eq!(size_of::<xess_vk_init_params_t>(), 64);
        assert_eq!(offset_of!(xess_vk_init_params_t, quality_setting), 8);
        assert_eq!(offset_of!(xess_vk_init_params_t, init_flags), 12);
        assert_eq!(offset_of!(xess_vk_init_params_t, temp_buffer_heap), 24);
        assert_eq!(offset_of!(xess_vk_init_params_t, pipeline_cache), 56);

        // execute params: 6 image-view-infos (288) + 3 floats + reset + 2 dims
        // (24) + 6 coords (48) = 360.
        assert_eq!(size_of::<xess_vk_execute_params_t>(), 360);
        assert_eq!(offset_of!(xess_vk_execute_params_t, output_texture), 240);
        assert_eq!(offset_of!(xess_vk_execute_params_t, jitter_offset_x), 288);
        assert_eq!(offset_of!(xess_vk_execute_params_t, reset_history), 300);
        assert_eq!(offset_of!(xess_vk_execute_params_t, input_width), 304);
        assert_eq!(offset_of!(xess_vk_execute_params_t, input_color_base), 312);
        assert_eq!(offset_of!(xess_vk_execute_params_t, output_color_base), 352);
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
