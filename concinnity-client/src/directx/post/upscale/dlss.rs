// src/directx/post/upscale/dlss.rs
//
// NVIDIA DLSS temporal upscaling for the D3D12 backend, via the raw NGX API
// (`NVSDK_NGX_D3D12_*`). One of the three `UpscaleBackend` implementations;
// RTX-only. Compiled only when `build.rs` finds the NGX SDK and emits
// `cfg(ngx_sdk_bundled)` (which also links `nvsdk_ngx_d.lib` and bundles
// `nvngx_dlss.dll` next to the .exe). When the SDK is absent the whole module
// is cfg'd out and `build_upscaler` never resolves to DLSS.
//
// NGX is a parameter-bag API: a feature is created + evaluated by setting
// named parameters on an `NVSDK_NGX_Parameter` and calling CreateFeature /
// EvaluateFeature, both of which record onto a command list (mirroring the FFX
// "evaluate on a command list" model). The bindings are inline `extern "C"`
// (linked from the static lib), validated against NGX SDK 1.5.0 by the
// constant asserts in the tests.
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::ptr;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

// NGX result: 0x1 is success; failure codes share the 0xBAD00000 high bits.
const NVSDK_NGX_RESULT_FAIL: u32 = 0xBAD0_0000;
fn ngx_succeeded(v: u32) -> bool {
    (v & 0xFFF0_0000) != NVSDK_NGX_RESULT_FAIL
}

const NVSDK_NGX_VERSION_API: i32 = 0x0000_0015; // 1.5.0
const NVSDK_NGX_ENGINE_TYPE_CUSTOM: i32 = 0;
const NVSDK_NGX_FEATURE_SUPERSAMPLING: i32 = 1;

// NVSDK_NGX_PerfQuality_Value (sequential from 0).
const PERF_MAX_PERF: i32 = 0;
const PERF_BALANCED: i32 = 1;
const PERF_MAX_QUALITY: i32 = 2;
const PERF_ULTRA_PERFORMANCE: i32 = 3;
const PERF_DLAA: i32 = 5;

// NVSDK_NGX_DLSS_Feature_Flags. Engine depth is 0 = near (NOT inverted), HDR
// linear input, low-res UV motion vectors. So: IsHDR + AutoExposure (the scene
// is un-exposed pre-upscale, matching the FSR path); DepthInverted stays off.
const DLSS_FLAG_IS_HDR: i32 = 1 << 0;
const DLSS_FLAG_AUTO_EXPOSURE: i32 = 1 << 6;

// NVSDK_NGX_Parameter name strings (NUL-terminated; from nvsdk_ngx_defs.h).
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

// NGX entry points, exported (unmangled, __cdecl) from nvsdk_ngx_d.lib.
// `NVSDK_NGX_Parameter` / `NVSDK_NGX_Handle` / `ID3D12*` are opaque pointers
// from Rust's side.
unsafe extern "C" {
    fn NVSDK_NGX_D3D12_Init_with_ProjectID(
        project_id: *const u8,
        engine_type: i32,
        engine_version: *const u8,
        app_data_path: *const u16,
        device: *mut c_void,
        feature_info: *const c_void,
        sdk_version: i32,
    ) -> u32;
    fn NVSDK_NGX_D3D12_Shutdown1(device: *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_GetCapabilityParameters(out_params: *mut *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_DestroyParameters(params: *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_CreateFeature(
        cmd: *mut c_void,
        feature_id: i32,
        params: *const c_void,
        out_handle: *mut *mut c_void,
    ) -> u32;
    fn NVSDK_NGX_D3D12_ReleaseFeature(handle: *mut c_void) -> u32;
    fn NVSDK_NGX_D3D12_EvaluateFeature_C(
        cmd: *mut c_void,
        handle: *const c_void,
        params: *const c_void,
        callback: *const c_void,
    ) -> u32;
    fn NVSDK_NGX_Parameter_SetUI(params: *mut c_void, name: *const u8, value: u32);
    fn NVSDK_NGX_Parameter_SetI(params: *mut c_void, name: *const u8, value: i32);
    fn NVSDK_NGX_Parameter_SetF(params: *mut c_void, name: *const u8, value: f32);
    fn NVSDK_NGX_Parameter_SetD3d12Resource(params: *mut c_void, name: *const u8, res: *mut c_void);
    fn NVSDK_NGX_Parameter_GetUI(params: *mut c_void, name: *const u8, out: *mut u32) -> u32;
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

// Map the engine's per-axis render-to-output ratio to the nearest DLSS
// performance/quality preset. Pure; unit tested.
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
// composite stack consumes, and the device (held for `Shutdown1` on drop).
// Mirrors `FsrUpscaler`.
pub(in crate::directx) struct DlssUpscaler {
    device: ID3D12Device,
    params: *mut c_void,
    handle: *mut c_void,
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

// The NGX handle / parameter bag / device are render-thread-only; the trait's
// `Send` bound is satisfied unsafely, same as the rest of `DxContext`.
unsafe impl Send for DlssUpscaler {}

impl DlssUpscaler {
    // Try to construct a DLSS upscaler. Returns `Ok(None)` when DLSS is
    // unavailable (NGX init failure, GPU lacks DLSS, feature-create failure);
    // the caller falls through. NGX `CreateFeature` records onto a command
    // list, so this submits a one-shot init list to `command_queue`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::directx) fn try_new(
        device: &ID3D12Device,
        command_queue: &ID3D12CommandQueue,
        output_width: u32,
        output_height: u32,
        upscale_scale: f32,
        output_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    ) -> Result<Option<Self>, String> {
        let scale = if upscale_scale > 0.0 {
            upscale_scale.clamp(1.0 / 3.0, 1.0)
        } else {
            1.0
        };
        let render_width = (((output_width as f32) * scale).round() as u32).max(1);
        let render_height = (((output_height as f32) * scale).round() as u32).max(1);

        // NGX writes logs / data into the app-data path; use the working dir.
        let app_path: Vec<u16> = ".".encode_utf16().chain(std::iter::once(0)).collect();
        let rc = unsafe {
            NVSDK_NGX_D3D12_Init_with_ProjectID(
                PROJECT_ID.as_ptr(),
                NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                ENGINE_VERSION.as_ptr(),
                app_path.as_ptr(),
                device_raw(device),
                ptr::null(),
                NVSDK_NGX_VERSION_API,
            )
        };
        if !ngx_succeeded(rc) {
            tracing::warn!(
                "DLSS: NVSDK_NGX_D3D12_Init returned {rc:#x} (NGX unavailable / not RTX). \
                 Trying the next upscaler."
            );
            return Ok(None);
        }

        let mut params: *mut c_void = ptr::null_mut();
        let rc = unsafe { NVSDK_NGX_D3D12_GetCapabilityParameters(&mut params) };
        if !ngx_succeeded(rc) || params.is_null() {
            tracing::warn!(
                "DLSS: GetCapabilityParameters returned {rc:#x}; trying the next upscaler"
            );
            unsafe { NVSDK_NGX_D3D12_Shutdown1(device_raw(device)) };
            return Ok(None);
        }

        // Authoritative DLSS-support gate for this GPU + driver.
        let mut available: u32 = 0;
        let rc = unsafe {
            NVSDK_NGX_Parameter_GetUI(params, P_SUPERSAMPLING_AVAILABLE.as_ptr(), &mut available)
        };
        if !ngx_succeeded(rc) || available == 0 {
            tracing::warn!(
                "DLSS: SuperSampling not available on this GPU; trying the next upscaler"
            );
            unsafe {
                NVSDK_NGX_D3D12_DestroyParameters(params);
                NVSDK_NGX_D3D12_Shutdown1(device_raw(device));
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
            NVSDK_NGX_Parameter_SetI(
                params,
                P_CREATE_FLAGS.as_ptr(),
                DLSS_FLAG_IS_HDR | DLSS_FLAG_AUTO_EXPOSURE,
            );
            NVSDK_NGX_Parameter_SetI(params, P_ENABLE_OUTPUT_SUBRECTS.as_ptr(), 0);
            NVSDK_NGX_Parameter_SetUI(params, P_CREATION_NODE_MASK.as_ptr(), 1);
            NVSDK_NGX_Parameter_SetUI(params, P_VISIBILITY_NODE_MASK.as_ptr(), 1);
        }

        // CreateFeature records onto a command list; submit a one-shot init list.
        let mut handle: *mut c_void = ptr::null_mut();
        let mut create_rc: u32 = NVSDK_NGX_RESULT_FAIL;
        crate::directx::texture::one_shot_submit(device, command_queue, |cmd| {
            create_rc = unsafe {
                NVSDK_NGX_D3D12_CreateFeature(
                    cmd_list_raw(cmd),
                    NVSDK_NGX_FEATURE_SUPERSAMPLING,
                    params,
                    &mut handle,
                )
            };
        })?;
        if !ngx_succeeded(create_rc) || handle.is_null() {
            tracing::warn!("DLSS: CreateFeature returned {create_rc:#x}; trying the next upscaler");
            unsafe {
                NVSDK_NGX_D3D12_DestroyParameters(params);
                NVSDK_NGX_D3D12_Shutdown1(device_raw(device));
            }
            return Ok(None);
        }

        let output = super::create_output_texture(device, output_width, output_height)?;
        super::write_output_uav(device, &output, output_uav_cpu);
        super::write_output_srv(device, &output, output_srv_cpu);

        tracing::info!(
            "DLSS: feature created: render {render_width}x{render_height} -> upscale \
             {output_width}x{output_height} (scale {scale:.3})"
        );

        Ok(Some(DlssUpscaler {
            device: device.clone(),
            params,
            handle,
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

impl super::UpscaleBackend for DlssUpscaler {
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

    // DLSS prescribes no jitter sequence; the engine's Halton-2/3 (shared with
    // the camera projection) drives both.
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
        unsafe {
            NVSDK_NGX_Parameter_SetD3d12Resource(
                self.params,
                P_COLOR.as_ptr(),
                resource_raw(color),
            );
            NVSDK_NGX_Parameter_SetD3d12Resource(
                self.params,
                P_OUTPUT.as_ptr(),
                resource_raw(&self.output),
            );
            NVSDK_NGX_Parameter_SetD3d12Resource(
                self.params,
                P_DEPTH.as_ptr(),
                resource_raw(depth),
            );
            NVSDK_NGX_Parameter_SetD3d12Resource(
                self.params,
                P_MOTION_VECTORS.as_ptr(),
                resource_raw(motion_vectors),
            );
            NVSDK_NGX_Parameter_SetF(self.params, P_JITTER_X.as_ptr(), jitter_offset[0]);
            NVSDK_NGX_Parameter_SetF(self.params, P_JITTER_Y.as_ptr(), jitter_offset[1]);
            // RG16F motion vectors are `prev_uv - cur_uv` in UV space; DLSS
            // wants pixel-space, so scale by the render extent (same as FSR).
            NVSDK_NGX_Parameter_SetF(self.params, P_MV_SCALE_X.as_ptr(), self.render_width as f32);
            NVSDK_NGX_Parameter_SetF(
                self.params,
                P_MV_SCALE_Y.as_ptr(),
                self.render_height as f32,
            );
            NVSDK_NGX_Parameter_SetI(self.params, P_RESET.as_ptr(), if reset { 1 } else { 0 });
            NVSDK_NGX_Parameter_SetUI(self.params, P_SUBRECT_WIDTH.as_ptr(), self.render_width);
            NVSDK_NGX_Parameter_SetUI(self.params, P_SUBRECT_HEIGHT.as_ptr(), self.render_height);
            NVSDK_NGX_Parameter_SetF(self.params, P_SHARPNESS.as_ptr(), 0.0);
        }
        let rc = unsafe {
            NVSDK_NGX_D3D12_EvaluateFeature_C(
                cmd_list_raw(cmd),
                self.handle,
                self.params,
                ptr::null(),
            )
        };
        if !ngx_succeeded(rc) {
            return Err(format!("NVSDK_NGX_D3D12_EvaluateFeature returned {rc:#x}"));
        }
        Ok(())
    }
}

impl Drop for DlssUpscaler {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                NVSDK_NGX_D3D12_ReleaseFeature(self.handle);
                self.handle = ptr::null_mut();
            }
            if !self.params.is_null() {
                NVSDK_NGX_D3D12_DestroyParameters(self.params);
                self.params = ptr::null_mut();
            }
            NVSDK_NGX_D3D12_Shutdown1(device_raw(&self.device));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngx_constants_match_sdk() {
        assert!(ngx_succeeded(0x1)); // NVSDK_NGX_Result_Success
        assert!(!ngx_succeeded(0xBAD0_0005)); // a FAIL code
        assert_eq!(NVSDK_NGX_VERSION_API, 0x0000_0015);
        assert_eq!(NVSDK_NGX_FEATURE_SUPERSAMPLING, 1);
        assert_eq!(PERF_MAX_PERF, 0);
        assert_eq!(PERF_MAX_QUALITY, 2);
        assert_eq!(PERF_ULTRA_PERFORMANCE, 3);
        assert_eq!(PERF_DLAA, 5);
        assert_eq!(DLSS_FLAG_IS_HDR, 1);
        assert_eq!(DLSS_FLAG_AUTO_EXPOSURE, 64);
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
