// src/directx/gpu_profile.rs
//
// Pre-backend GPU performance probe for DirectX. Enumerates the DXGI adapter the
// renderer would pick (the first non-software adapter that supports D3D12) and
// classifies it WITHOUT creating a device, swapchain, or pipelines, so the
// auto-config quality ceiling can be resolved before the backend (and its render
// targets) are built. Mirrors `DxContext::gpu_profile` exactly (vendor id +
// dedicated VRAM) and the standalone `metal/gpu_profile.rs` pattern. Returns
// `UNKNOWN` on any failure (no factory, no suitable adapter, desc query fails),
// which the resolver treats as "no clamp".

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::gfx::backend::{GpuClassInput, GpuProfile, GpuVendor, classify_tier};

pub(crate) fn probe_gpu_profile() -> GpuProfile {
    probe_adapter().unwrap_or(GpuProfile::UNKNOWN)
}

// Build a throwaway DXGI factory (no debug flag, so it never needs the
// "Graphics Tools" optional feature), pick the first non-software adapter that
// supports D3D12, and classify it. No device is created: the D3D12 support check
// passes a null device-out pointer, a pure capability query. Mirrors the
// `pick_adapter` loop in `init/window.rs`.
fn probe_adapter() -> Option<GpuProfile> {
    let factory: IDXGIFactory4 =
        unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.ok()?;
    let mut i = 0u32;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
        i += 1;
        let desc = match unsafe { adapter.GetDesc1() } {
            Ok(d) => d,
            Err(_) => continue,
        };
        // Skip the software (WARP) adapter; it is not the render target.
        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
            continue;
        }
        // D3D12 support as a pure capability check (a null device-out pointer
        // builds no device).
        let supported = unsafe {
            D3D12CreateDevice(
                &adapter,
                D3D_FEATURE_LEVEL_11_0,
                std::ptr::null_mut::<Option<ID3D12Device>>(),
            )
        }
        .is_ok();
        if supported {
            return Some(adapter_profile(&desc));
        }
    }
    None
}

// Classify a chosen adapter's description, mirroring `DxContext::gpu_profile`.
fn adapter_profile(desc: &DXGI_ADAPTER_DESC1) -> GpuProfile {
    let vendor = match desc.VendorId {
        0x10DE => GpuVendor::Nvidia,
        0x1002 => GpuVendor::Amd,
        0x8086 => GpuVendor::Intel,
        _ => GpuVendor::Other,
    };
    let dedicated = desc.DedicatedVideoMemory as u64;
    // A discrete GPU has dedicated VRAM; an integrated part reports little or
    // none. A small floor keeps a few MB of carve-out from reading as discrete.
    let discrete = dedicated >= (256u64 << 20);
    let tier = classify_tier(&GpuClassInput {
        vendor,
        memory_budget_bytes: dedicated,
        discrete,
        apple_family: 0,
    });
    GpuProfile {
        vendor,
        tier,
        memory_budget_bytes: dedicated,
        unified_memory: !discrete,
        discrete,
    }
}
