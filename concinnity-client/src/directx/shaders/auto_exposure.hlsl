// HLSL port of metal/shaders/auto_exposure.metal. Two compute entry points:
//   * build   - one thread per HDR-resolve pixel; threadgroup-local 256-bin
//               atomic histogram, merged into the global histogram at the end.
//   * average - one threadgroup of 256 threads reducing the histogram to a
//               single average log-luminance value, then clearing it for the
//               next frame.
//
// `HISTOGRAM_BINS` and the AutoExposureParams layout (16 bytes) must match
// `gfx::auto_exposure::HISTOGRAM_BINS` and the inline struct the host pushes
// as root constants.

#pragma pack_matrix(column_major)

static const uint HISTOGRAM_BINS = 256;

cbuffer AutoExposureParams : register(b0)
{
    // Lowest log2(luminance) the histogram bins span. Pixels darker than this
    // fall in bin 0 and are dropped by the average pass.
    float lum_log2_min;
    // Width of the log2(luminance) range the histogram covers
    // (`LUM_LOG2_MAX - LUM_LOG2_MIN`).
    float lum_log2_range;
    // Precomputed `HISTOGRAM_BINS / lum_log2_range`. The build kernel multiplies
    // the centred log-luminance by this to get a bin index.
    float lum_to_bin_scale;
    float _pad;
};

Texture2D<float4>       hdr_texture : register(t0);
RWStructuredBuffer<uint> histogram   : register(u0);
RWStructuredBuffer<float> output_avg : register(u1);

groupshared uint local_hist[HISTOGRAM_BINS];

// One thread per HDR-resolve pixel. Each threadgroup keeps a local 256-bin
// histogram in groupshared memory, atomically increments its own bin per
// pixel, and atomically merges the local counts into the global histogram on
// exit. The local stage absorbs most of the contention so per-frame global
// atomics scale cheaply.
[numthreads(16, 16, 1)]
void build(
    uint3 gid : SV_DispatchThreadID,
    uint  tid : SV_GroupIndex)
{
    if (tid < HISTOGRAM_BINS)
    {
        local_hist[tid] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint w, h;
    hdr_texture.GetDimensions(w, h);
    if (gid.x < w && gid.y < h)
    {
        float3 c = hdr_texture.Load(int3(int(gid.x), int(gid.y), 0)).rgb;
        // Rec. 709 luminance. The 1.0e-6 floor keeps log2 finite on a fully
        // black pixel, which would otherwise fall in bin 0 via -inf clamp.
        float lum = max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1.0e-6);
        float lum_log2 = clamp(
            log2(lum),
            lum_log2_min,
            lum_log2_min + lum_log2_range);
        float t = (lum_log2 - lum_log2_min) * lum_to_bin_scale;
        uint bin = min(uint(t), HISTOGRAM_BINS - 1u);
        uint _dummy;
        InterlockedAdd(local_hist[bin], 1u, _dummy);
    }
    GroupMemoryBarrierWithGroupSync();

    if (tid < HISTOGRAM_BINS)
    {
        uint count = local_hist[tid];
        if (count > 0u)
        {
            uint _dummy;
            InterlockedAdd(histogram[tid], count, _dummy);
        }
    }
}

groupshared uint  reduce_counts[HISTOGRAM_BINS];
groupshared float reduce_weighted[HISTOGRAM_BINS];

// One threadgroup of 256 threads. Each thread reads its bin's count, clears
// the bin for the next frame's build pass, then participates in a parallel
// reduction that writes the count-weighted average log-luminance to
// `output_avg[0]`. Bin 0 is treated as "below sensor floor" and weighted out,
// so a mostly-black frame still produces a finite EV.
[numthreads(HISTOGRAM_BINS, 1, 1)]
void average(uint tid : SV_GroupIndex)
{
    uint count = histogram[tid];
    // Clear for the next frame's build pass. Safe because every thread has
    // already read its bin and the reduction below operates on the
    // groupshared copies.
    histogram[tid] = 0u;

    // Drop the sub-floor bin so mostly-black frames don't peg the average.
    uint effective_count = (tid == 0u) ? 0u : count;
    float step = lum_log2_range / float(HISTOGRAM_BINS);
    float centre = lum_log2_min + (float(tid) + 0.5) * step;
    reduce_counts[tid]   = effective_count;
    reduce_weighted[tid] = centre * float(effective_count);
    GroupMemoryBarrierWithGroupSync();

    [unroll] for (uint stride = HISTOGRAM_BINS / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            reduce_counts[tid]   += reduce_counts[tid + stride];
            reduce_weighted[tid] += reduce_weighted[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        output_avg[0] = (reduce_counts[0] > 0u)
            ? (reduce_weighted[0] / float(reduce_counts[0]))
            : lum_log2_min;
    }
}
