// src/directx/post/upscale/mod.rs
//
// Temporal upscaling for the D3D12 backend. The engine renders the 3D scene at
// a fraction of drawable size and the `PassId::Upscale` pass reconstructs a
// drawable-resolution image the bloom + composite stack consumes.
//
// Three interchangeable backends sit behind the `UpscaleBackend` trait:
//   fsr   AMD FidelityFX FSR3 (cross-vendor; the default fallback)
//   dlss  NVIDIA DLSS via raw NGX (RTX only; cfg(ngx_sdk_bundled))
//   xess  Intel XeSS (cross-vendor DP4a + Arc XMX)
// `build_upscaler` resolves the requested `UpscalerBackend` against runtime
// availability and constructs the first that initialises, falling back to
// native-resolution rendering when none is available. The shared per-frame
// `encode_upscale` (in fsr.rs) drives whichever backend is active through the
// trait; only the inner vendor evaluate differs.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::assets::UpscalerBackend;

#[cfg(ngx_sdk_bundled)]
mod dlss;
mod fsr;
mod xess;

// One temporal-upscaling backend. The shared `encode_upscale` transitions the
// scene / depth / motion inputs and the output texture, then calls `dispatch`;
// each backend records its vendor upscale onto the supplied command list.
pub(in crate::directx) trait UpscaleBackend: Send {
    // Off-screen scene render dimensions (the backend's input size).
    fn render_dims(&self) -> (u32, u32);
    // Drawable (output) dimensions the backend reconstructs.
    fn output_dims(&self) -> (u32, u32);
    // Per-axis render-to-output ratio resolved from the quality preset.
    fn upscale_scale(&self) -> f32;
    // SRV the bloom + composite stack samples as the scene.
    fn output_srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE;
    // The output texture's (uav_cpu, srv_cpu, srv_gpu) heap handles, so a
    // resize can rebuild the backend into the same pre-reserved slots.
    fn output_descriptors(
        &self,
    ) -> (
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_CPU_DESCRIPTOR_HANDLE,
        D3D12_GPU_DESCRIPTOR_HANDLE,
    );
    // The output texture (transitioned UAV <-> PSR around the dispatch).
    fn output_resource(&self) -> &ID3D12Resource;
    // Whether `output` currently rests in PIXEL_SHADER_RESOURCE (the
    // post-dispatch sample window) vs UNORDERED_ACCESS. Tracked across frames
    // by `encode_upscale`.
    fn output_is_psr(&self) -> bool;
    fn set_output_is_psr(&self, v: bool);
    // Sub-pixel jitter for this frame's index, shared with the camera
    // projection so the jittered VP and the upscale agree.
    fn jitter_offset(&self, frame_index: u32) -> [f32; 2];
    // Record the upscale onto `cmd`. Inputs are claimed in the states
    // `encode_upscale` transitioned them into (color / depth / motion in
    // NON_PIXEL_SHADER_RESOURCE, output in UNORDERED_ACCESS).
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
    ) -> Result<(), String>;
}

// Output texture helpers (shared by every backend): an output-resolution
// RGBA16Float UAV the backend writes + an SRV the bloom / composite stack
// samples. Created in UNORDERED_ACCESS.

const UPSCALE_OUTPUT_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

fn create_output_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: UPSCALE_OUTPUT_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create upscale output texture: {e}"))?;
    tex_opt.ok_or_else(|| "create upscale output texture returned None".to_string())
}

fn write_output_uav(device: &ID3D12Device, res: &ID3D12Resource, cpu: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: UPSCALE_OUTPUT_FORMAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    unsafe { device.CreateUnorderedAccessView(res, None, Some(&desc), cpu) };
}

fn write_output_srv(device: &ID3D12Device, res: &ID3D12Resource, cpu: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: UPSCALE_OUTPUT_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(res, Some(&desc), cpu) };
}

// Sub-pixel jitter shared by the DLSS + XeSS backends (FSR queries its own
// FFX-prescribed sequence instead). A 16-phase Halton-2/3 sequence in
// [-0.5, 0.5] render-pixel units; the same value jitters the camera projection
// (see `draw_frame`) so the rasterised scene and the upscale agree.
pub(super) fn halton_jitter_offset(frame_index: u32) -> [f32; 2] {
    let idx = (frame_index % 16) + 1;
    [radical_inverse(idx, 2) - 0.5, radical_inverse(idx, 3) - 0.5]
}

// Van der Corput radical inverse of `i` in the given base, in [0, 1).
fn radical_inverse(mut i: u32, base: u32) -> f32 {
    let inv_base = 1.0 / base as f32;
    let mut f = 1.0_f32;
    let mut r = 0.0_f32;
    while i > 0 {
        f *= inv_base;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

// Backend selection

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::directx) enum ResolvedBackend {
    Fsr3,
    Dlss,
    Xess,
    Native,
}

// Ordered candidate list for a requested backend + availability: the
// explicitly requested one first (when available), then the Auto priority
// order (DLSS, XeSS, FSR3), then Native (always last, always available).
fn backend_order(
    requested: UpscalerBackend,
    dlss_avail: bool,
    xess_avail: bool,
    fsr3_avail: bool,
) -> Vec<ResolvedBackend> {
    let mut order: Vec<ResolvedBackend> = Vec::new();
    match requested {
        UpscalerBackend::Dlss if dlss_avail => order.push(ResolvedBackend::Dlss),
        UpscalerBackend::Xess if xess_avail => order.push(ResolvedBackend::Xess),
        UpscalerBackend::Fsr3 if fsr3_avail => order.push(ResolvedBackend::Fsr3),
        _ => {}
    }
    for (cand, avail) in [
        (ResolvedBackend::Dlss, dlss_avail),
        (ResolvedBackend::Xess, xess_avail),
        (ResolvedBackend::Fsr3, fsr3_avail),
    ] {
        if avail && !order.contains(&cand) {
            order.push(cand);
        }
    }
    order.push(ResolvedBackend::Native);
    order
}

// Construct the upscaler for the requested backend, falling through the
// candidate order on any `try_new` that returns `None` (DLL miss, unsupported
// GPU, context-init failure). Returns the boxed backend (or `None` for native
// rendering) and the tag that actually built, so a resize rebuilds the same one.
#[allow(clippy::too_many_arguments)]
pub(in crate::directx) fn build_upscaler(
    device: &ID3D12Device,
    command_queue: &ID3D12CommandQueue,
    output_width: u32,
    output_height: u32,
    upscale_scale: f32,
    output_uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    output_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    output_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    requested: UpscalerBackend,
) -> Result<(Option<Box<dyn UpscaleBackend>>, ResolvedBackend), String> {
    // `command_queue` is only consumed by the DLSS init path (NGX CreateFeature
    // records onto a command list); silence the unused-binding lint otherwise.
    #[cfg(not(ngx_sdk_bundled))]
    let _ = command_queue;

    let dlss_avail = cfg!(ngx_sdk_bundled);
    let xess_avail = cfg!(xess_sdk_bundled);
    let fsr3_avail = cfg!(agility_sdk_configured);

    for cand in backend_order(requested, dlss_avail, xess_avail, fsr3_avail) {
        let built: Option<Box<dyn UpscaleBackend>> = match cand {
            ResolvedBackend::Fsr3 => fsr::FsrUpscaler::try_new(
                device,
                output_width,
                output_height,
                upscale_scale,
                output_uav_cpu,
                output_srv_cpu,
                output_srv_gpu,
            )?
            .map(|u| Box::new(u) as Box<dyn UpscaleBackend>),
            ResolvedBackend::Xess => xess::XessUpscaler::try_new(
                device,
                output_width,
                output_height,
                upscale_scale,
                output_uav_cpu,
                output_srv_cpu,
                output_srv_gpu,
            )?
            .map(|u| Box::new(u) as Box<dyn UpscaleBackend>),
            ResolvedBackend::Dlss => {
                #[cfg(ngx_sdk_bundled)]
                {
                    dlss::DlssUpscaler::try_new(
                        device,
                        command_queue,
                        output_width,
                        output_height,
                        upscale_scale,
                        output_uav_cpu,
                        output_srv_cpu,
                        output_srv_gpu,
                    )?
                    .map(|u| Box::new(u) as Box<dyn UpscaleBackend>)
                }
                #[cfg(not(ngx_sdk_bundled))]
                {
                    None
                }
            }
            ResolvedBackend::Native => None,
        };
        if let Some(b) = built {
            tracing::info!(
                "temporal upscaling: using {cand:?} backend (output {output_width}x{output_height})"
            );
            return Ok((Some(b), cand));
        }
        if cand != ResolvedBackend::Native {
            tracing::warn!("temporal upscaling: {cand:?} unavailable, trying next backend");
        }
    }
    tracing::info!("temporal upscaling: no backend available, rendering at native resolution");
    Ok((None, ResolvedBackend::Native))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assets::UpscalerBackend as B;

    // The backend a request resolves to: the first candidate in the order.
    fn resolved(req: B, dlss: bool, xess: bool, fsr3: bool) -> ResolvedBackend {
        backend_order(req, dlss, xess, fsr3)[0]
    }

    #[test]
    fn auto_prefers_dlss_then_xess_then_fsr3_then_native() {
        assert_eq!(resolved(B::Auto, true, true, true), ResolvedBackend::Dlss);
        assert_eq!(resolved(B::Auto, false, true, true), ResolvedBackend::Xess);
        assert_eq!(resolved(B::Auto, false, false, true), ResolvedBackend::Fsr3);
        assert_eq!(
            resolved(B::Auto, false, false, false),
            ResolvedBackend::Native
        );
    }

    #[test]
    fn explicit_choice_used_when_available() {
        assert_eq!(resolved(B::Dlss, true, true, true), ResolvedBackend::Dlss);
        assert_eq!(resolved(B::Xess, true, true, true), ResolvedBackend::Xess);
        assert_eq!(resolved(B::Fsr3, true, true, true), ResolvedBackend::Fsr3);
    }

    #[test]
    fn halton_jitter_is_centered_and_bounded() {
        for f in 0..64u32 {
            let [x, y] = halton_jitter_offset(f);
            assert!((-0.5..0.5).contains(&x), "x={x} out of range");
            assert!((-0.5..0.5).contains(&y), "y={y} out of range");
        }
        // radical_inverse(1, 2) = 0.5 -> offset 0.0; (1,3) = 1/3 -> -1/6.
        let [x, y] = halton_jitter_offset(0);
        assert!((x - 0.0).abs() < 1e-6);
        assert!((y - (1.0 / 3.0 - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn explicit_choice_falls_through_when_unavailable() {
        // Requested DLSS unavailable falls to the next available (XeSS).
        assert_eq!(resolved(B::Dlss, false, true, true), ResolvedBackend::Xess);
        // Requested XeSS unavailable, only FSR3 left.
        assert_eq!(resolved(B::Xess, false, false, true), ResolvedBackend::Fsr3);
        // Requested FSR3 unavailable, nothing left.
        assert_eq!(
            resolved(B::Fsr3, false, false, false),
            ResolvedBackend::Native
        );
    }
}
