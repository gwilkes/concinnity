// src/metal/gpu_profile.rs
//
// Coarse GPU performance profile from the live MTLDevice, for default-quality
// selection. Every signal is read straight off the device (no extra GPU work):
// the unified-memory flag, the recommended working-set as the memory budget, and
// the highest supported Apple GPU family as the generation. The shared
// classify_tier (gfx/backend.rs) maps these to a tier so every backend
// classifies the same way.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLGPUFamily};

use crate::gfx::backend::{GpuClassInput, GpuProfile, GpuVendor, classify_tier};

// Highest Apple GPU family the device supports, as a generation rank (7 = M1 ..
// 10 = M4), or 0 when the device reports no Apple family (e.g. an AMD dGPU on an
// Intel Mac). Probed top-down so a newer device returns its own generation.
fn apple_gpu_family(device: &ProtocolObject<dyn MTLDevice>) -> u8 {
    for (family, rank) in [
        (MTLGPUFamily::Apple10, 10u8),
        (MTLGPUFamily::Apple9, 9),
        (MTLGPUFamily::Apple8, 8),
        (MTLGPUFamily::Apple7, 7),
    ] {
        if device.supportsFamily(family) {
            return rank;
        }
    }
    0
}

pub(crate) fn device_profile(device: &ProtocolObject<dyn MTLDevice>) -> GpuProfile {
    let unified = device.hasUnifiedMemory();
    let budget = device.recommendedMaxWorkingSetSize();
    let family = apple_gpu_family(device);
    // Apple silicon (or any device reporting an Apple GPU family) is Apple;
    // otherwise an AMD / Intel dGPU on an Intel Mac, classified by VRAM.
    let vendor = if family >= 7 || unified {
        GpuVendor::Apple
    } else {
        GpuVendor::Other
    };
    let discrete = !device.isLowPower();
    let tier = classify_tier(&GpuClassInput {
        vendor,
        memory_budget_bytes: budget,
        discrete,
        apple_family: family,
    });
    GpuProfile {
        vendor,
        tier,
        memory_budget_bytes: budget,
        unified_memory: unified,
        discrete,
    }
}
