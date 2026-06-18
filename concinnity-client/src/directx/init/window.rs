// src/directx/init/window.rs
//
// Bootstrap for DxContext: Win32 window registration, DXGI factory, adapter
// selection, D3D12 device + (optional) debug info-queue, command queue, MSAA
// support query, and swapchain creation. Returns a `DeviceAndWindow` bundle
// that init/mod.rs unpacks into the constructor's local state.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::UI::WindowsAndMessaging::{GWLP_USERDATA, SetWindowLongPtrW};
use windows::core::Interface;

use crate::directx::context::FRAMES;
use crate::directx::texture::HDR_FORMAT;
use crate::directx::window::{WindowState, create_window};
use crate::gfx::hdr_output::HdrOutputMode;

pub(super) struct DeviceAndWindow {
    pub win_state: Box<WindowState>,
    pub device: ID3D12Device,
    pub info_queue: Option<ID3D12InfoQueue>,
    pub command_queue: ID3D12CommandQueue,
    pub swapchain: IDXGISwapChain3,
    pub swapchain_format: DXGI_FORMAT,
    // Whether the swapchain was created with DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
    // (vsync off + tearing supported). Drives the present sync interval / flags
    // and must be mirrored in every `ResizeBuffers` call.
    pub allow_tearing: bool,
    pub msaa_samples: u32,
    // Adapter cast to `IDXGIAdapter3` so the profiler overlay can call
    // `QueryVideoMemoryInfo` for the VRAM chip. `None` on adapters that don't
    // expose the v3 interface (very old WDDM 1.x drivers); the HUD then reads
    // `VRAM 0 MB` and the rest of the overlay still works.
    pub adapter: Option<IDXGIAdapter3>,
    // Resolved swapchain colour-output mode. When `Hdr`, the swapchain was
    // created in `RGBA16Float` and `SetColorSpace1` was called with the
    // matching colour space: scRGB linear
    // (`DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`) for
    // `HdrEncoding::ExtendedLinear`, HDR10 PQ
    // (`DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`) for `HdrEncoding::Pq`.
    // The composite shader's `hdr_output` branch emits the matching
    // envelope; its `pq_output` flag picks scRGB-linear passthrough vs the
    // SMPTE ST 2084 in-shader encode. PQ-not-supported worlds fall back to
    // extended-linear and the encoding is rewritten on the returned mode so
    // the caller's `post_process.pq_output` setup stays consistent.
    pub hdr_mode: HdrOutputMode,
}

pub(super) fn setup(
    title: &str,
    width: u32,
    height: u32,
    validation: bool,
    vsync: bool,
    hdr_display_requested: bool,
    hdr_pq_requested: bool,
) -> Result<DeviceAndWindow, String> {
    // Validation / debug layer
    if validation
        && let Ok(debug) = unsafe {
            let mut d: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut d).map(|_| d)
        }
        && let Some(d) = debug
    {
        unsafe { d.EnableDebugLayer() };
    }

    // Win32 window
    let (hwnd, mut win_state) = create_window(title, width, height)?;

    // Register for raw mouse input (for captured-cursor delta).
    let rid = windows::Win32::UI::Input::RAWINPUTDEVICE {
        usUsagePage: 0x01,
        usUsage: 0x02, // mouse
        dwFlags: windows::Win32::UI::Input::RIDEV_INPUTSINK,
        hwndTarget: hwnd,
    };
    let _ = unsafe {
        windows::Win32::UI::Input::RegisterRawInputDevices(
            &[rid],
            std::mem::size_of::<windows::Win32::UI::Input::RAWINPUTDEVICE>() as u32,
        )
    };

    // Store win_state pointer in GWLP_USERDATA so wnd_proc can reach it.
    unsafe {
        SetWindowLongPtrW(
            hwnd,
            GWLP_USERDATA,
            &mut *win_state as *mut WindowState as isize,
        )
    };

    // DXGI factory
    // DXGI_CREATE_FACTORY_DEBUG requires the Windows "Graphics Tools" optional
    // feature; fall back silently if it isn't installed.
    let factory: IDXGIFactory4 = if validation {
        unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_DEBUG) }
            .or_else(|_| unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) })
            .map_err(|e| format!("CreateDXGIFactory2: {e}"))?
    } else {
        unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .map_err(|e| format!("CreateDXGIFactory2: {e}"))?
    };

    let adapter = pick_adapter(&factory)?;
    // Try to upcast to IDXGIAdapter3 for QueryVideoMemoryInfo. Optional; pre-
    // WDDM 2.0 drivers may not expose it, in which case the VRAM chip stays
    // at 0 MB and the rest of the overlay still works.
    let adapter3: Option<IDXGIAdapter3> = adapter.cast().ok();

    let mut device_opt: Option<ID3D12Device> = None;
    unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device_opt) }
        .map_err(|e| format!("D3D12CreateDevice: {e}"))?;
    let device = device_opt.ok_or("D3D12CreateDevice returned None")?;

    // Validation message sink
    // Disable break-on-error so the debug layer doesn't terminate the process
    // before we can log the message via tracing.
    let info_queue: Option<ID3D12InfoQueue> = if validation {
        device.cast::<ID3D12InfoQueue>().ok().inspect(|iq| unsafe {
            let _ = iq.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, false);
            let _ = iq.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, false);
            let _ = iq.SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, false);
        })
    } else {
        None
    };

    // Command queue
    let queue_desc = D3D12_COMMAND_QUEUE_DESC {
        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
        ..Default::default()
    };
    let command_queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }
        .map_err(|e| format!("CreateCommandQueue: {e}"))?;

    // MSAA support check (queried against HDR_FORMAT; the swapchain is always 1x).
    let msaa_samples = query_msaa_samples(&device);

    // HDR-output detection. Walk the adapter's outputs, find the highest
    // max-EDR multiplier reported by any HDR-capable output, and feed it
    // through the asset-side `hdr_display` toggle. EDR support is per-output
    // (per-monitor), so the answer depends on which display the window will
    // land on; before the swapchain exists we have to scan every output and
    // pick the best. A capable output reports `ColorSpace ==
    // DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020` (the standard "HDR
    // available" signal); `MaxLuminance` in cd/m² divided by the 80-nit SDR
    // reference white gives the EDR multiplier the resolver expects.
    let max_edr = measure_max_edr(&adapter);
    // `mut` so a later PQ-not-supported fallback can downgrade the encoding
    // before the returned `hdr_mode` reaches the caller; see the
    // SetColorSpace1 block below.
    let mut hdr_mode = HdrOutputMode::resolve(hdr_display_requested, hdr_pq_requested, max_edr);
    if hdr_display_requested && !hdr_mode.is_hdr() {
        tracing::warn!(
            "HDR display requested but the active adapter's outputs report max EDR \
             multiplier {:.3}, falling back to SDR (BGRA8Unorm) output",
            max_edr
        );
    } else if hdr_mode.is_hdr() {
        tracing::info!(
            "HDR display output enabled: max EDR multiplier {:.3} on the active adapter",
            max_edr
        );
    }

    // Swapchain
    // SDR: BGRA8Unorm (the historical default). HDR: RGBA16Float so the
    // compositor receives linear extended-range floats. The scRGB colour space
    // is set via `SetColorSpace1` after the cast to `IDXGISwapChain3` below;
    // CreateSwapChain itself does not take a colour-space parameter.
    let swapchain_format = if hdr_mode.is_hdr() {
        HDR_SWAPCHAIN_FORMAT
    } else {
        DXGI_FORMAT_B8G8R8A8_UNORM
    };
    // Tearing support gates true uncapped (no-vsync) presentation on the flip
    // model: without it, a windowed flip swapchain paces to the display refresh
    // even at sync interval 0. Probe it via IDXGIFactory5; if vsync is requested
    // or the factory / hardware doesn't support tearing, create the swapchain
    // without the flag (refresh-paced fallback).
    let allow_tearing = !vsync
        && factory
            .cast::<IDXGIFactory5>()
            .ok()
            .map(|f5| {
                // CheckFeatureSupport writes a 4-byte Win32 BOOL; an i32 matches
                // its layout without pulling in the BOOL type.
                let mut data: i32 = 0;
                unsafe {
                    f5.CheckFeatureSupport(
                        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        &mut data as *mut _ as *mut core::ffi::c_void,
                        std::mem::size_of::<i32>() as u32,
                    )
                }
                .is_ok()
                    && data != 0
            })
            .unwrap_or(false);
    let sc_desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: swapchain_format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: FRAMES as u32,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
        Flags: if allow_tearing {
            DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32
        } else {
            0
        },
        ..Default::default()
    };
    let sc_base: IDXGISwapChain1 =
        unsafe { factory.CreateSwapChainForHwnd(&command_queue, hwnd, &sc_desc, None, None) }
            .map_err(|e| format!("CreateSwapChain: {e}"))?;
    let swapchain: IDXGISwapChain3 = sc_base
        .cast()
        .map_err(|e| format!("SwapChain3 cast: {e}"))?;

    // On the HDR path, tell DXGI which colour space the swapchain pixels
    // carry. Two flavours, picked by `hdr_mode.encoding`:
    //
    //   - `ExtendedLinear` → `DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`
    //     (scRGB linear). The compositor maps `1.0` to SDR reference white
    //     and gives the panel headroom above that, same envelope as Metal's
    //     extended-linear Display P3, but Rec.709 primaries. The standard
    //     DXGI linear-HDR signal supported on every recent driver.
    //
    //   - `Pq` → `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020` (HDR10 PQ).
    //     The shader emits SMPTE ST 2084 PQ-encoded values directly; the
    //     panel decodes via the PQ EOTF. SDR reference white maps to 203
    //     nits per BT.2408.
    //
    // PQ-not-supported worlds gracefully fall back to scRGB linear (the
    // shader's `pq_output` flag is cleared in [`super::DxContext::new`] when
    // that fallback fires so the shader doesn't double-encode). If neither
    // signal is advertised by the swapchain, leave at the platform default
    // and warn; the resolver path has already gated us on the panel
    // reporting HDR headroom, so the format itself works; only the precise
    // encoding hand-off may be wrong.
    if hdr_mode.is_hdr() {
        let want_pq = matches!(
            hdr_mode,
            HdrOutputMode::Hdr {
                encoding: crate::gfx::hdr_output::HdrEncoding::Pq,
                ..
            }
        );
        let primary = if want_pq {
            HDR_PQ_COLOR_SPACE
        } else {
            HDR_LINEAR_COLOR_SPACE
        };
        let primary_label = if want_pq { "HDR10 PQ" } else { "scRGB linear" };
        let primary_support = unsafe { swapchain.CheckColorSpaceSupport(primary) }.unwrap_or(0);
        let primary_ok =
            (primary_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32) != 0;
        let mut applied = false;
        if primary_ok {
            if let Err(e) = unsafe { swapchain.SetColorSpace1(primary) } {
                tracing::warn!(
                    "HDR display enabled but SetColorSpace1({primary_label}) failed ({e}); \
                     the compositor may still treat the RGBA16Float swapchain as sRGB; HDR \
                     output may look desaturated"
                );
            } else {
                applied = true;
            }
        }
        if !applied && want_pq {
            // PQ requested but unsupported: fall back to scRGB linear so
            // the renderer still drives the panel's HDR headroom (the
            // shader's `pq_output` flag is cleared below).
            let fallback_support =
                unsafe { swapchain.CheckColorSpaceSupport(HDR_LINEAR_COLOR_SPACE) }.unwrap_or(0);
            if (fallback_support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT.0 as u32) != 0 {
                tracing::warn!(
                    "HDR display + hdr_pq:true requested but the swapchain does not advertise \
                     HDR10 PQ support (CheckColorSpaceSupport flags = {primary_support:#x}); \
                     falling back to scRGB linear extended-range output"
                );
                if let Err(e) = unsafe { swapchain.SetColorSpace1(HDR_LINEAR_COLOR_SPACE) } {
                    tracing::warn!(
                        "scRGB linear fallback also failed ({e}); leaving the swapchain at \
                         its default colour space"
                    );
                } else {
                    // Rewrite the encoding so the caller's
                    // `post_process.pq_output` flag stays in sync with what
                    // the swapchain is actually expecting.
                    hdr_mode = HdrOutputMode::Hdr {
                        max_edr: match hdr_mode {
                            HdrOutputMode::Hdr { max_edr, .. } => max_edr,
                            HdrOutputMode::Sdr => unreachable!(),
                        },
                        encoding: crate::gfx::hdr_output::HdrEncoding::ExtendedLinear,
                    };
                }
            } else {
                tracing::warn!(
                    "HDR display + hdr_pq:true requested but neither HDR10 PQ \
                     (flags={primary_support:#x}) nor scRGB linear (flags={fallback_support:#x}) \
                     are advertised by this swapchain; leaving the swapchain at its default \
                     colour space"
                );
            }
        } else if !applied {
            tracing::warn!(
                "HDR display enabled but the swapchain does not advertise {primary_label} \
                 support (CheckColorSpaceSupport flags = {primary_support:#x}); leaving the \
                 swapchain at its default colour space"
            );
        }
    }

    // Disable Alt+Enter fullscreen toggle.
    unsafe { factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }.ok();

    Ok(DeviceAndWindow {
        win_state,
        device,
        info_queue,
        command_queue,
        swapchain,
        swapchain_format,
        allow_tearing,
        msaa_samples,
        adapter: adapter3,
        hdr_mode,
    })
}

// Swapchain pixel format on the HDR path. RGBA16Float gives the compositor
// the headroom to drive values past SDR reference white without crushing
// precision; an 8-bit format cannot represent the extended range.
const HDR_SWAPCHAIN_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// DXGI colour space for scRGB linear: Rec.709 primaries, gamma 1.0 (linear),
// extended range. `1.0` is SDR reference white; values above that map to the
// panel's HDR headroom. The standard DXGI signal for "linear HDR", supported
// on every recent driver alongside the older HDR10 (PQ) path.
const HDR_LINEAR_COLOR_SPACE: DXGI_COLOR_SPACE_TYPE = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;

// DXGI colour space for HDR10 PQ: Rec.2020 primaries, SMPTE ST 2084 EOTF
// (PQ), absolute-luminance signal capped at 10,000 cd/m². The shader emits
// PQ-encoded values directly; the panel decodes via the PQ EOTF. SDR
// reference white maps to 203 nits per BT.2408.
const HDR_PQ_COLOR_SPACE: DXGI_COLOR_SPACE_TYPE = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;

// Largest extended-range colour-component multiplier any output on this
// adapter reports. Returns `1.0` on an SDR-only adapter (or when no output
// is available, a head-less unit test) so the resolver stays on the SDR
// path. An HDR output's `MaxLuminance` is in cd/m²; SDR reference white is
// 80 nits, so the EDR multiplier is `MaxLuminance / 80.0`.
fn measure_max_edr(adapter: &IDXGIAdapter1) -> f32 {
    const SDR_REFERENCE_NITS: f32 = 80.0;
    let mut best: f32 = 1.0;
    let mut i: u32 = 0;
    loop {
        let output: IDXGIOutput = match unsafe { adapter.EnumOutputs(i) } {
            Ok(o) => o,
            Err(_) => break,
        };
        i += 1;
        let output6: IDXGIOutput6 = match output.cast() {
            Ok(o) => o,
            Err(_) => continue, // pre-Windows-10 output, no HDR info
        };
        let desc1 = match unsafe { output6.GetDesc1() } {
            Ok(d) => d,
            Err(_) => continue,
        };
        // The colour-space field is the canonical "HDR available" signal:
        // PQ here means the output is wired to an HDR-capable display.
        let hdr_advertised = desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
        if !hdr_advertised {
            continue;
        }
        let nits = desc1.MaxLuminance;
        if nits.is_finite() && nits > 0.0 {
            let edr = nits / SDR_REFERENCE_NITS;
            if edr > best {
                best = edr;
            }
        }
    }
    best
}

fn pick_adapter(factory: &IDXGIFactory4) -> Result<IDXGIAdapter1, String> {
    let mut i = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
        let desc = unsafe { adapter.GetDesc1() }.map_err(|e| format!("GetDesc1: {e}"))?;
        // Skip the software (WARP) adapter.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
            i += 1;
            continue;
        }
        // Check D3D12 support.
        if unsafe {
            D3D12CreateDevice(
                &adapter,
                D3D_FEATURE_LEVEL_11_0,
                std::ptr::null_mut::<Option<ID3D12Device>>(),
            )
        }
        .is_ok()
        {
            return Ok(adapter);
        }
        i += 1;
    }
    Err("no suitable D3D12 adapter found".to_string())
}

fn query_msaa_samples(device: &ID3D12Device) -> u32 {
    // Queried against the off-screen HDR target's format: that is the only
    // multisampled render target; the swapchain backbuffer is always 1x.
    for &count in &[4u32, 2] {
        let mut data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: HDR_FORMAT,
            SampleCount: count,
            Flags: D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE,
            NumQualityLevels: 0,
        };
        if unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                &mut data as *mut _ as *mut std::ffi::c_void,
                std::mem::size_of::<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS>() as u32,
            )
        }
        .is_ok()
            && data.NumQualityLevels > 0
        {
            return count;
        }
    }
    1
}
