// src/vulkan/post/upscale/dlss.rs
//
// NVIDIA DLSS temporal upscaling for the Vulkan backend, via the raw NGX API
// (`NVSDK_NGX_VULKAN_*`). One of the three `VkUpscaleBackend` implementations;
// RTX-only. Compiled only when `build.rs` finds the NGX SDK and emits
// `cfg(ngx_sdk_bundled)` (which also links `nvsdk_ngx_d.lib` and bundles
// `nvngx_dlss.dll` next to the .exe). When the SDK is absent the whole module
// is cfg'd out and `build_upscaler` never resolves to DLSS.
//
// Mirrors `directx/post/upscale/dlss.rs`: same NGX parameter-bag flow
// (CreateFeature / EvaluateFeature record onto a command buffer), same engine
// identity, same perf-quality mapping. The Vulkan deltas are the
// `NVSDK_NGX_VULKAN_*` entry points (vs `NVSDK_NGX_D3D12_*`), the device /
// instance extensions DLSS needs at creation time (queried via
// `required_extensions`, enabled by `UpscaleSdk`), and resources passed as
// `NVSDK_NGX_Resource_VK` through `SetVoidPointer` (vs `SetD3d12Resource`). It
// also differs in exposure handling: it supplies an explicit 1.0 exposure
// texture (NVIDIA's recommended path over auto-exposure) instead of the
// auto-exposure flag the D3D12 path uses. The scene is un-exposed pre-upscale
// (exposure + tonemap run after the upscale), so 1.0 is the identity value.
// Validated against NGX SDK 1.5.0 by the constant + layout asserts in the tests.
#![allow(non_snake_case)]

use std::cell::Cell;
use std::ffi::{CString, c_char, c_void};
use std::ptr;

use ash::{Device, vk};

use super::{UpscaleImage, VkUpscaleBackend};
use crate::vulkan::context::HDR_FORMAT;
use crate::vulkan::texture::{GpuImage, create_image, create_image_view, one_shot_submit};

// NGX result: 0x1 is success; failure codes share the 0xBAD00000 high bits.
const NVSDK_NGX_RESULT_FAIL: u32 = 0xBAD0_0000;
fn ngx_succeeded(v: u32) -> bool {
    (v & 0xFFF0_0000) != NVSDK_NGX_RESULT_FAIL
}

const NVSDK_NGX_VERSION_API: i32 = 0x0000_0015; // 1.5.0
const NVSDK_NGX_ENGINE_TYPE_CUSTOM: i32 = 0;
const NVSDK_NGX_FEATURE_SUPERSAMPLING: i32 = 1;
// NVSDK_NGX_Resource_VK_Type (sequential from 0).
const NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW: i32 = 0;

// NVSDK_NGX_PerfQuality_Value (sequential from 0).
const PERF_MAX_PERF: i32 = 0;
const PERF_BALANCED: i32 = 1;
const PERF_MAX_QUALITY: i32 = 2;
const PERF_ULTRA_PERFORMANCE: i32 = 3;
const PERF_DLAA: i32 = 5;

// NVSDK_NGX_DLSS_Feature_Flags. Engine depth is 0 = near (NOT inverted), HDR
// linear input, low-res UV motion vectors. So: IsHDR; DepthInverted stays off.
// Auto-exposure is deliberately left off: the dispatch supplies an explicit 1.0
// exposure texture instead (NVIDIA's recommended path over auto-exposure), which
// NGX uses only while this flag is clear.
const DLSS_FLAG_IS_HDR: i32 = 1 << 0;

// NVSDK_NGX_Parameter name strings (NUL-terminated; from nvsdk_ngx_params.h).
const P_WIDTH: &[u8] = b"Width\0";
const P_HEIGHT: &[u8] = b"Height\0";
const P_OUT_WIDTH: &[u8] = b"OutWidth\0";
const P_OUT_HEIGHT: &[u8] = b"OutHeight\0";
const P_PERF_QUALITY: &[u8] = b"PerfQualityValue\0";
const P_CREATE_FLAGS: &[u8] = b"DLSS.Feature.Create.Flags\0";
const P_ENABLE_OUTPUT_SUBRECTS: &[u8] = b"DLSS.Enable.Output.Subrects\0";
const P_CREATION_NODE_MASK: &[u8] = b"CreationNodeMask\0";
const P_VISIBILITY_NODE_MASK: &[u8] = b"VisibilityNodeMask\0";
const P_SUPERSAMPLING_AVAILABLE: &[u8] = b"SuperSampling.Available\0";
const P_COLOR: &[u8] = b"Color\0";
const P_OUTPUT: &[u8] = b"Output\0";
const P_DEPTH: &[u8] = b"Depth\0";
const P_MOTION_VECTORS: &[u8] = b"MotionVectors\0";
const P_EXPOSURE_TEXTURE: &[u8] = b"ExposureTexture\0";
const P_JITTER_X: &[u8] = b"Jitter.Offset.X\0";
const P_JITTER_Y: &[u8] = b"Jitter.Offset.Y\0";
const P_MV_SCALE_X: &[u8] = b"MV.Scale.X\0";
const P_MV_SCALE_Y: &[u8] = b"MV.Scale.Y\0";
const P_RESET: &[u8] = b"Reset\0";
const P_SUBRECT_WIDTH: &[u8] = b"DLSS.Render.Subrect.Dimensions.Width\0";
const P_SUBRECT_HEIGHT: &[u8] = b"DLSS.Render.Subrect.Dimensions.Height\0";
const P_SHARPNESS: &[u8] = b"Sharpness\0";

// Engine identity for NGX. Any GUID-like project id avoids needing an
// NVIDIA-assigned application id.
const PROJECT_ID: &[u8] = b"5f2e1a64-9c3b-4d7e-8a1f-2b6c0d9e7f30\0";
const ENGINE_VERSION: &[u8] = b"1.0.0\0";

// NVSDK_NGX_ImageViewInfo_VK (nvsdk_ngx_vk.h). 48 bytes; same shape as XeSS's
// image-view-info: handles(16) + VkImageSubresourceRange(20) + format(4) +
// width(4) + height(4).
#[repr(C)]
#[derive(Clone, Copy)]
struct NVSDK_NGX_ImageViewInfo_VK {
    image_view: vk::ImageView,
    image: vk::Image,
    subresource_range: vk::ImageSubresourceRange,
    format: vk::Format,
    width: u32,
    height: u32,
}

// NVSDK_NGX_Resource_VK (nvsdk_ngx_vk.h). The C struct's first member is a union
// of an `ImageViewInfo_VK` and a `BufferInfo_VK`; the image-view arm is the
// larger (48 vs 16) and the only one DLSS uses here, so it sizes the union and
// is embedded directly. 56 bytes: union(48) + Type(4) + ReadWrite(1) + pad.
#[repr(C)]
struct NVSDK_NGX_Resource_VK {
    image_view_info: NVSDK_NGX_ImageViewInfo_VK,
    ty: i32,
    read_write: bool,
}

fn make_resource(
    img_view: vk::ImageView,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    format: vk::Format,
    width: u32,
    height: u32,
    read_write: bool,
) -> NVSDK_NGX_Resource_VK {
    NVSDK_NGX_Resource_VK {
        image_view_info: NVSDK_NGX_ImageViewInfo_VK {
            image_view: img_view,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            format,
            width,
            height,
        },
        ty: NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW,
        read_write,
    }
}

// Create a small device-local input image, clear it to `clear`, and leave it in
// GENERAL (the layout NGX reads all resources in). Used for the supplied
// exposure texture, written once here and never touched again, so the one-time
// clear barrier covers all later NGX reads. SAMPLED | STORAGE usage matches
// however NGX binds it (descriptor type unknown to us).
#[allow(clippy::too_many_arguments)]
fn create_cleared_input(
    instance: &ash::Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    width: u32,
    height: u32,
    format: vk::Format,
    clear: vk::ClearColorValue,
) -> Result<GpuImage, String> {
    let (image, memory) = create_image(
        instance,
        device,
        physical_device,
        width.max(1),
        height.max(1),
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        vk::SampleCountFlags::TYPE_1,
    )?;
    let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
    let range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    one_shot_submit(device, command_pool, queue, |cmd| {
        super::image_barrier(
            device,
            cmd,
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            device.cmd_clear_color_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear,
                std::slice::from_ref(&range),
            );
        }
        super::image_barrier(
            device,
            cmd,
            image,
            vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ,
        );
    })?;
    Ok(GpuImage {
        image,
        memory,
        view,
        aux_views: Vec::new(),
    })
}

// NGX entry points, exported (unmangled, C linkage) from nvsdk_ngx_d.lib. The
// `NVSDK_NGX_Parameter` / `NVSDK_NGX_Handle` bags are opaque pointers; the VK
// handles ride as ash's repr(transparent) wrappers. Init/CreateFeature/Evaluate
// are the Vulkan variants; the `NVSDK_NGX_Parameter_*` setters are API-agnostic
// core exports (shared with the D3D12 path).
unsafe extern "C" {
    fn NVSDK_NGX_VULKAN_Init_with_ProjectID(
        project_id: *const u8,
        engine_type: i32,
        engine_version: *const u8,
        app_data_path: *const u16,
        instance: vk::Instance,
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        gipa: *const c_void,
        gdpa: *const c_void,
        feature_info: *const c_void,
        sdk_version: i32,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_Shutdown1(device: vk::Device) -> u32;
    fn NVSDK_NGX_VULKAN_GetCapabilityParameters(out_params: *mut *mut c_void) -> u32;
    fn NVSDK_NGX_VULKAN_DestroyParameters(params: *mut c_void) -> u32;
    // CreateFeature1 takes the VkDevice (vs CreateFeature), letting NGX set up
    // its internal resources against the device during the init command buffer
    // instead of lazily on first evaluate. The NGX helper header prefers it
    // whenever a device is available, and it avoids first-frame UNDEFINED ->
    // GENERAL validation errors on NGX's internal textures.
    fn NVSDK_NGX_VULKAN_CreateFeature1(
        device: vk::Device,
        cmd: vk::CommandBuffer,
        feature_id: i32,
        params: *const c_void,
        out_handle: *mut *mut c_void,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_ReleaseFeature(handle: *mut c_void) -> u32;
    fn NVSDK_NGX_VULKAN_EvaluateFeature_C(
        cmd: vk::CommandBuffer,
        handle: *const c_void,
        params: *const c_void,
        callback: *const c_void,
    ) -> u32;
    fn NVSDK_NGX_VULKAN_RequiredExtensions(
        out_inst_count: *mut u32,
        out_inst_exts: *mut *const *const c_char,
        out_dev_count: *mut u32,
        out_dev_exts: *mut *const *const c_char,
    ) -> u32;
    fn NVSDK_NGX_Parameter_SetUI(params: *mut c_void, name: *const u8, value: u32);
    fn NVSDK_NGX_Parameter_SetI(params: *mut c_void, name: *const u8, value: i32);
    fn NVSDK_NGX_Parameter_SetF(params: *mut c_void, name: *const u8, value: f32);
    fn NVSDK_NGX_Parameter_SetVoidPointer(params: *mut c_void, name: *const u8, value: *mut c_void);
    fn NVSDK_NGX_Parameter_GetUI(params: *mut c_void, name: *const u8, out: *mut u32) -> u32;
}

// Instance + device extensions NGX requires, queried before instance / device
// creation. Returns `(instance_exts, device_exts)` as owned `CString`s, or
// `None` if the query fails. Static-linked, so no DLL load is needed.
pub(super) fn required_extensions() -> Option<(Vec<CString>, Vec<CString>)> {
    let mut inst_count: u32 = 0;
    let mut inst_exts: *const *const c_char = ptr::null();
    let mut dev_count: u32 = 0;
    let mut dev_exts: *const *const c_char = ptr::null();
    let rc = unsafe {
        NVSDK_NGX_VULKAN_RequiredExtensions(
            &mut inst_count,
            &mut inst_exts,
            &mut dev_count,
            &mut dev_exts,
        )
    };
    if !ngx_succeeded(rc) {
        tracing::warn!("DLSS: NVSDK_NGX_VULKAN_RequiredExtensions returned {rc:#x}");
        return None;
    }
    let inst = unsafe { super::copy_ext_names(inst_count, inst_exts) };
    let dev = unsafe { super::copy_ext_names(dev_count, dev_exts) };
    Some((inst, dev))
}

// Map the engine's per-axis render-to-output ratio to the nearest DLSS
// performance/quality preset. Pure; unit tested. Mirrors
// `directx::post::upscale::dlss::perf_quality_from_scale`.
fn perf_quality_from_scale(scale: f32) -> i32 {
    if scale >= 0.99 {
        PERF_DLAA
    } else if scale >= 0.62 {
        PERF_MAX_QUALITY
    } else if scale >= 0.55 {
        PERF_BALANCED
    } else if scale >= 0.42 {
        PERF_MAX_PERF
    } else {
        PERF_ULTRA_PERFORMANCE
    }
}

// Owns the NGX feature handle + parameter bag, the output texture the bloom +
// composite stack consumes, and the device handle (held for `Shutdown1`).
pub(in crate::vulkan) struct DlssUpscaler {
    device: vk::Device,
    params: *mut c_void,
    handle: *mut c_void,

    output: GpuImage,
    output_layout: Cell<vk::ImageLayout>,

    // Engine-supplied exposure texture (1x1 R32F = 1.0), created once in GENERAL
    // and bound every dispatch. With auto-exposure off NGX uses this identity
    // exposure (the scene is un-exposed pre-upscale).
    exposure: GpuImage,

    render_width: u32,
    render_height: u32,
    output_width: u32,
    output_height: u32,
    upscale_scale: f32,

    jitter: Cell<[f32; 2]>,
    reset_pending: Cell<bool>,
}

// The NGX handle / parameter bag are render-thread-only raw pointers; the
// trait's `Send` bound is satisfied unsafely, same as the rest of `VkContext`.
unsafe impl Send for DlssUpscaler {}

impl DlssUpscaler {
    // Try to construct a DLSS upscaler. Returns `Ok(None)` when DLSS is
    // unavailable (NGX init failure, GPU lacks DLSS, feature-create failure);
    // `build_upscaler` falls through. NGX `CreateFeature` records onto a command
    // buffer, so this submits a one-shot init buffer to `queue`. Assumes the NGX
    // instance / device extensions were enabled at creation (via `UpscaleSdk`).
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
        let (render_width, render_height, scale) =
            super::resolve_render_dims(output_width, output_height, upscale_scale);

        // NGX writes logs / data into the app-data path; use the working dir.
        let app_path: Vec<u16> = ".".encode_utf16().chain(std::iter::once(0)).collect();
        let rc = unsafe {
            NVSDK_NGX_VULKAN_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                ENGINE_VERSION.as_ptr(),
                app_path.as_ptr(),
                instance.handle(),
                physical_device,
                device.handle(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                NVSDK_NGX_VERSION_API,
            )
        };
        if !ngx_succeeded(rc) {
            tracing::warn!(
                "DLSS (Vulkan): NVSDK_NGX_VULKAN_Init returned {rc:#x} (NGX unavailable / not \
                 RTX). Trying the next backend."
            );
            return Ok(None);
        }

        let mut params: *mut c_void = ptr::null_mut();
        let rc = unsafe { NVSDK_NGX_VULKAN_GetCapabilityParameters(&mut params) };
        if !ngx_succeeded(rc) || params.is_null() {
            tracing::warn!(
                "DLSS (Vulkan): GetCapabilityParameters returned {rc:#x}; trying next backend"
            );
            unsafe { NVSDK_NGX_VULKAN_Shutdown1(device.handle()) };
            return Ok(None);
        }

        // Authoritative DLSS-support gate for this GPU + driver.
        let mut available: u32 = 0;
        let rc = unsafe {
            NVSDK_NGX_Parameter_GetUI(params, P_SUPERSAMPLING_AVAILABLE.as_ptr(), &mut available)
        };
        if !ngx_succeeded(rc) || available == 0 {
            tracing::warn!(
                "DLSS (Vulkan): SuperSampling not available on this GPU; trying next backend"
            );
            unsafe {
                NVSDK_NGX_VULKAN_DestroyParameters(params);
                NVSDK_NGX_VULKAN_Shutdown1(device.handle());
            }
            return Ok(None);
        }

        // Feature-create parameters.
        unsafe {
            NVSDK_NGX_Parameter_SetUI(params, P_WIDTH.as_ptr(), render_width);
            NVSDK_NGX_Parameter_SetUI(params, P_HEIGHT.as_ptr(), render_height);
            NVSDK_NGX_Parameter_SetUI(params, P_OUT_WIDTH.as_ptr(), output_width);
            NVSDK_NGX_Parameter_SetUI(params, P_OUT_HEIGHT.as_ptr(), output_height);
            NVSDK_NGX_Parameter_SetI(
                params,
                P_PERF_QUALITY.as_ptr(),
                perf_quality_from_scale(scale),
            );
            NVSDK_NGX_Parameter_SetI(params, P_CREATE_FLAGS.as_ptr(), DLSS_FLAG_IS_HDR);
            NVSDK_NGX_Parameter_SetI(params, P_ENABLE_OUTPUT_SUBRECTS.as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, P_CREATION_NODE_MASK.as_ptr(), 1);
            NVSDK_NGX_Parameter_SetUI(params, P_VISIBILITY_NODE_MASK.as_ptr(), 1);
        }

        // CreateFeature1 records onto a command buffer; submit a one-shot init
        // (the submit is fence-waited, so the feature + its internal resources
        // are ready before the first frame's evaluate).
        let mut handle: *mut c_void = ptr::null_mut();
        let mut create_rc: u32 = NVSDK_NGX_RESULT_FAIL;
        one_shot_submit(device, command_pool, queue, |cmd| {
            create_rc = unsafe {
                NVSDK_NGX_VULKAN_CreateFeature1(
                    device.handle(),
                    cmd,
                    NVSDK_NGX_FEATURE_SUPERSAMPLING,
                    params,
                    &mut handle,
                )
            };
        })?;
        if !ngx_succeeded(create_rc) || handle.is_null() {
            tracing::warn!(
                "DLSS (Vulkan): CreateFeature returned {create_rc:#x}; trying next backend"
            );
            unsafe {
                NVSDK_NGX_VULKAN_DestroyParameters(params);
                NVSDK_NGX_VULKAN_Shutdown1(device.handle());
            }
            return Ok(None);
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
                unsafe {
                    NVSDK_NGX_VULKAN_ReleaseFeature(handle);
                    NVSDK_NGX_VULKAN_DestroyParameters(params);
                    NVSDK_NGX_VULKAN_Shutdown1(device.handle());
                }
                return Err(e);
            }
        };

        // Supplied 1.0 exposure texture (see the struct field). Created in GENERAL
        // so the first evaluate finds it ready; on failure tear down everything
        // built so far.
        let exposure = match create_cleared_input(
            instance,
            device,
            physical_device,
            command_pool,
            queue,
            1,
            1,
            vk::Format::R32_SFLOAT,
            vk::ClearColorValue {
                float32: [1.0, 0.0, 0.0, 0.0],
            },
        ) {
            Ok(img) => img,
            Err(e) => {
                unsafe {
                    NVSDK_NGX_VULKAN_ReleaseFeature(handle);
                    NVSDK_NGX_VULKAN_DestroyParameters(params);
                    NVSDK_NGX_VULKAN_Shutdown1(device.handle());
                }
                output.destroy(device);
                return Err(e);
            }
        };

        tracing::info!(
            "DLSS (Vulkan): feature created: render {render_width}x{render_height} -> upscale \
             {output_width}x{output_height} (scale {scale:.3})"
        );

        Ok(Some(DlssUpscaler {
            device: device.handle(),
            params,
            handle,
            output,
            output_layout: Cell::new(vk::ImageLayout::GENERAL),
            exposure,
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

impl VkUpscaleBackend for DlssUpscaler {
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

    // DLSS prescribes no jitter sequence; the engine's Halton-2/3 (shared with
    // the camera projection) drives both.
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
        // These must outlive the EvaluateFeature call (NGX reads them during the
        // record); they are bound by pointer through SetVoidPointer.
        let mut color_res = make_resource(
            color.view,
            color.image,
            color.aspect,
            color.format,
            color.width,
            color.height,
            false,
        );
        let mut depth_res = make_resource(
            depth.view,
            depth.image,
            depth.aspect,
            depth.format,
            depth.width,
            depth.height,
            false,
        );
        let mut motion_res = make_resource(
            motion.view,
            motion.image,
            motion.aspect,
            motion.format,
            motion.width,
            motion.height,
            false,
        );
        let mut output_res = make_resource(
            self.output.view,
            self.output.image,
            vk::ImageAspectFlags::COLOR,
            HDR_FORMAT,
            self.output_width,
            self.output_height,
            true,
        );
        let mut exposure_res = make_resource(
            self.exposure.view,
            self.exposure.image,
            vk::ImageAspectFlags::COLOR,
            vk::Format::R32_SFLOAT,
            1,
            1,
            false,
        );
        unsafe {
            let p = self.params;
            NVSDK_NGX_Parameter_SetVoidPointer(
                p,
                P_COLOR.as_ptr(),
                &mut color_res as *mut _ as *mut c_void,
            );
            NVSDK_NGX_Parameter_SetVoidPointer(
                p,
                P_OUTPUT.as_ptr(),
                &mut output_res as *mut _ as *mut c_void,
            );
            NVSDK_NGX_Parameter_SetVoidPointer(
                p,
                P_DEPTH.as_ptr(),
                &mut depth_res as *mut _ as *mut c_void,
            );
            NVSDK_NGX_Parameter_SetVoidPointer(
                p,
                P_MOTION_VECTORS.as_ptr(),
                &mut motion_res as *mut _ as *mut c_void,
            );
            NVSDK_NGX_Parameter_SetVoidPointer(
                p,
                P_EXPOSURE_TEXTURE.as_ptr(),
                &mut exposure_res as *mut _ as *mut c_void,
            );
            NVSDK_NGX_Parameter_SetF(p, P_JITTER_X.as_ptr(), jitter_offset[0]);
            NVSDK_NGX_Parameter_SetF(p, P_JITTER_Y.as_ptr(), jitter_offset[1]);
            // RG16F motion vectors are `prev_uv - cur_uv` in UV space; DLSS
            // wants pixel-space, so scale by the render extent (same as FSR).
            NVSDK_NGX_Parameter_SetF(p, P_MV_SCALE_X.as_ptr(), self.render_width as f32);
            NVSDK_NGX_Parameter_SetF(p, P_MV_SCALE_Y.as_ptr(), self.render_height as f32);
            NVSDK_NGX_Parameter_SetI(p, P_RESET.as_ptr(), if reset { 1 } else { 0 });
            NVSDK_NGX_Parameter_SetUI(p, P_SUBRECT_WIDTH.as_ptr(), self.render_width);
            NVSDK_NGX_Parameter_SetUI(p, P_SUBRECT_HEIGHT.as_ptr(), self.render_height);
            NVSDK_NGX_Parameter_SetF(p, P_SHARPNESS.as_ptr(), 0.0);
        }
        let rc = unsafe {
            NVSDK_NGX_VULKAN_EvaluateFeature_C(cmd, self.handle, self.params, ptr::null())
        };
        if !ngx_succeeded(rc) {
            return Err(format!("NVSDK_NGX_VULKAN_EvaluateFeature returned {rc:#x}"));
        }
        Ok(())
    }

    fn destroy(&mut self, device: &Device) {
        unsafe {
            if !self.handle.is_null() {
                NVSDK_NGX_VULKAN_ReleaseFeature(self.handle);
                self.handle = ptr::null_mut();
            }
            if !self.params.is_null() {
                NVSDK_NGX_VULKAN_DestroyParameters(self.params);
                self.params = ptr::null_mut();
            }
            NVSDK_NGX_VULKAN_Shutdown1(self.device);
        }
        self.output.destroy(device);
        self.exposure.destroy(device);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn ngx_vk_resource_layout_matches_sdk() {
        // ImageViewInfo: handles(16) + VkImageSubresourceRange(20) + format(4)
        // + width(4) + height(4) = 48.
        assert_eq!(size_of::<NVSDK_NGX_ImageViewInfo_VK>(), 48);
        assert_eq!(offset_of!(NVSDK_NGX_ImageViewInfo_VK, image_view), 0);
        assert_eq!(offset_of!(NVSDK_NGX_ImageViewInfo_VK, image), 8);
        assert_eq!(
            offset_of!(NVSDK_NGX_ImageViewInfo_VK, subresource_range),
            16
        );
        assert_eq!(offset_of!(NVSDK_NGX_ImageViewInfo_VK, format), 36);
        assert_eq!(offset_of!(NVSDK_NGX_ImageViewInfo_VK, width), 40);
        assert_eq!(offset_of!(NVSDK_NGX_ImageViewInfo_VK, height), 44);

        // Resource_VK: union(48, sized by ImageViewInfo) + Type(4) + ReadWrite(1)
        // + tail pad = 56.
        assert_eq!(size_of::<NVSDK_NGX_Resource_VK>(), 56);
        assert_eq!(offset_of!(NVSDK_NGX_Resource_VK, ty), 48);
        assert_eq!(offset_of!(NVSDK_NGX_Resource_VK, read_write), 52);
    }

    #[test]
    fn ngx_constants_match_sdk() {
        assert!(ngx_succeeded(0x1)); // NVSDK_NGX_Result_Success
        assert!(!ngx_succeeded(0xBAD0_0005)); // a FAIL code
        assert_eq!(NVSDK_NGX_VERSION_API, 0x0000_0015);
        assert_eq!(NVSDK_NGX_FEATURE_SUPERSAMPLING, 1);
        assert_eq!(NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW, 0);
        assert_eq!(PERF_MAX_PERF, 0);
        assert_eq!(PERF_MAX_QUALITY, 2);
        assert_eq!(PERF_ULTRA_PERFORMANCE, 3);
        assert_eq!(PERF_DLAA, 5);
        assert_eq!(DLSS_FLAG_IS_HDR, 1);
    }

    #[test]
    fn dlss_perf_quality_mapping_by_scale() {
        assert_eq!(perf_quality_from_scale(1.0), PERF_DLAA);
        assert_eq!(perf_quality_from_scale(2.0 / 3.0), PERF_MAX_QUALITY);
        assert_eq!(perf_quality_from_scale(0.587), PERF_BALANCED);
        assert_eq!(perf_quality_from_scale(0.5), PERF_MAX_PERF);
        assert_eq!(perf_quality_from_scale(1.0 / 3.0), PERF_ULTRA_PERFORMANCE);
    }
}
