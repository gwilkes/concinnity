#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Number of histogram bins the build kernel writes into and the average kernel
// reduces. Must match gfx::auto_exposure::HISTOGRAM_BINS on the CPU side.
constant uint HISTOGRAM_BINS = 256u;

// Auto-exposure parameters. Layout matches AutoExposureParams in
// metal/auto_exposure.rs. 16 bytes.
struct AutoExposureParams {
    // Lowest log2(luminance) the histogram bins span. Pixels darker than this
    // fall in bin 0 and are dropped by the average pass.
    float lum_log2_min;
    // Width of the log2(luminance) range the histogram covers, i.e.
    // `LUM_LOG2_MAX - LUM_LOG2_MIN`. The build kernel uses this to map a
    // log-luminance value to a bin index, and the average kernel uses it to
    // recover the bin centre during the weighted reduction.
    float lum_log2_range;
    // Pre-computed `HISTOGRAM_BINS / lum_log2_range`. The build kernel
    // multiplies the centred log-luminance by this to get a bin index, saving
    // a per-pixel divide.
    float lum_to_bin_scale;
    float _pad;
};

// One thread per HDR-resolve pixel. Each threadgroup keeps a local 256-bin
// histogram in threadgroup memory, atomically increments its own bin per
// pixel, and atomically merges the local counts into the global histogram on
// exit. The local stage absorbs most of the contention so the per-frame
// global atomics scale to large HDR resolves cheaply.
kernel void histogram_build(
    texture2d<float, access::read>  hdr       [[texture(0)]],
    device   atomic_uint           *histogram [[buffer(0)]],
    constant AutoExposureParams    &params    [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint  tid [[thread_index_in_threadgroup]]
) {
    threadgroup atomic_uint local_hist[HISTOGRAM_BINS];
    // First 256 threads of the group clear their slot. Larger groups would
    // need a loop, but 16x16 == 256 matches exactly.
    if (tid < HISTOGRAM_BINS) {
        atomic_store_explicit(&local_hist[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint w = hdr.get_width();
    uint h = hdr.get_height();
    if (gid.x < w && gid.y < h) {
        float3 c = hdr.read(gid).rgb;
        // Rec. 709 luminance. The 1.0e-6 floor keeps log2 finite on a fully
        // black pixel, which would otherwise fall in bin 0 via -inf clamp.
        float lum = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1.0e-6);
        float lum_log2 = clamp(
            log2(lum),
            params.lum_log2_min,
            params.lum_log2_min + params.lum_log2_range
        );
        float t = (lum_log2 - params.lum_log2_min) * params.lum_to_bin_scale;
        uint bin = min(uint(t), HISTOGRAM_BINS - 1u);
        atomic_fetch_add_explicit(&local_hist[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < HISTOGRAM_BINS) {
        uint count = atomic_load_explicit(&local_hist[tid], memory_order_relaxed);
        if (count > 0u) {
            atomic_fetch_add_explicit(&histogram[tid], count, memory_order_relaxed);
        }
    }
}

// One threadgroup of 256 threads. Each thread reads its bin's count, clears
// the bin for the next frame, then participates in a parallel reduction that
// writes the count-weighted average log-luminance to `output[0]`. Bin 0 is
// treated as "below sensor floor" and weighted out, so a mostly-black frame
// still produces a finite EV instead of dragging the average to the floor.
kernel void histogram_average(
    device   atomic_uint        *histogram [[buffer(0)]],
    device   float              *output    [[buffer(1)]],
    constant AutoExposureParams &params    [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint  counts  [HISTOGRAM_BINS];
    threadgroup float weighted[HISTOGRAM_BINS];

    uint count = atomic_load_explicit(&histogram[tid], memory_order_relaxed);
    // Clear for the next frame's build pass. Safe to do here because every
    // thread has already loaded its bin and the reduction below operates on
    // the threadgroup-local copies.
    atomic_store_explicit(&histogram[tid], 0u, memory_order_relaxed);

    // Drop the sub-floor bin so mostly-black frames don't peg the average.
    uint effective_count = (tid == 0u) ? 0u : count;
    float step = params.lum_log2_range / float(HISTOGRAM_BINS);
    float centre = params.lum_log2_min + (float(tid) + 0.5) * step;
    counts[tid] = effective_count;
    weighted[tid] = centre * float(effective_count);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction (256 -> 1).
    for (uint stride = HISTOGRAM_BINS / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            counts[tid]   += counts[tid + stride];
            weighted[tid] += weighted[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        output[0] = (counts[0] > 0u)
            ? (weighted[0] / float(counts[0]))
            : params.lum_log2_min;
    }
}
