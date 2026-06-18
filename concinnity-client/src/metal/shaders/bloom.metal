#include <metal_stdlib>
using namespace metal;

struct BloomVtxOut {
    float4 position [[position]];
    float2 uv;
};

struct PostUniforms {
    float bloom_intensity;
    float bloom_threshold;
    float bloom_knee;
    float exposure;
    float vignette;
    float lut_strength;
};

// Upper bound on a single bloom source sample. IBL skyboxes can be many orders
// of magnitude brighter than surface shading (e.g. an EV24 HDR); without a
// clamp one ultra-bright texel blurs out and washes the whole frame white.
constant float BLOOM_CLAMP = 16.0;

vertex BloomVtxOut bloom_vertex_main(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    BloomVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

inline float bloom_luma(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

// Karis weight: attenuate bright 2x2 groups so a single firefly does not
// dominate the downsample. Applied only on the first (prefilter) downsample.
inline float karis_weight(float3 c) {
    return 1.0 / (1.0 + bloom_luma(c));
}

// 13-tap downsample (Jimenez/CoD). When `karis` is set, the 13 taps are
// grouped into five overlapping 2x2 boxes and luma-weighted.
inline float3 downsample_13(texture2d<float> src, sampler smp,
                            float2 uv, float2 texel, bool karis) {
    float3 a = src.sample(smp, uv + texel * float2(-2.0, -2.0)).rgb;
    float3 b = src.sample(smp, uv + texel * float2( 0.0, -2.0)).rgb;
    float3 c = src.sample(smp, uv + texel * float2( 2.0, -2.0)).rgb;
    float3 d = src.sample(smp, uv + texel * float2(-2.0,  0.0)).rgb;
    float3 e = src.sample(smp, uv + texel * float2( 0.0,  0.0)).rgb;
    float3 f = src.sample(smp, uv + texel * float2( 2.0,  0.0)).rgb;
    float3 g = src.sample(smp, uv + texel * float2(-2.0,  2.0)).rgb;
    float3 h = src.sample(smp, uv + texel * float2( 0.0,  2.0)).rgb;
    float3 i = src.sample(smp, uv + texel * float2( 2.0,  2.0)).rgb;
    float3 j = src.sample(smp, uv + texel * float2(-1.0, -1.0)).rgb;
    float3 k = src.sample(smp, uv + texel * float2( 1.0, -1.0)).rgb;
    float3 l = src.sample(smp, uv + texel * float2(-1.0,  1.0)).rgb;
    float3 m = src.sample(smp, uv + texel * float2( 1.0,  1.0)).rgb;

    if (karis) {
        float3 g0 = (a + b + d + e) * (0.125 / 4.0);
        float3 g1 = (b + c + e + f) * (0.125 / 4.0);
        float3 g2 = (d + e + g + h) * (0.125 / 4.0);
        float3 g3 = (e + f + h + i) * (0.125 / 4.0);
        float3 g4 = (j + k + l + m) * (0.5 / 4.0);
        g0 *= karis_weight(g0);
        g1 *= karis_weight(g1);
        g2 *= karis_weight(g2);
        g3 *= karis_weight(g3);
        g4 *= karis_weight(g4);
        return g0 + g1 + g2 + g3 + g4;
    }

    float3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
    return result;
}

// 9-tap tent upsample filter (weights 1 2 1 / 2 4 2 / 1 2 1, /16).
inline float3 upsample_tent(texture2d<float> src, sampler smp,
                            float2 uv, float2 texel) {
    float3 sum = float3(0.0);
    sum += src.sample(smp, uv + texel * float2(-1.0, -1.0)).rgb * 1.0;
    sum += src.sample(smp, uv + texel * float2( 0.0, -1.0)).rgb * 2.0;
    sum += src.sample(smp, uv + texel * float2( 1.0, -1.0)).rgb * 1.0;
    sum += src.sample(smp, uv + texel * float2(-1.0,  0.0)).rgb * 2.0;
    sum += src.sample(smp, uv + texel * float2( 0.0,  0.0)).rgb * 4.0;
    sum += src.sample(smp, uv + texel * float2( 1.0,  0.0)).rgb * 2.0;
    sum += src.sample(smp, uv + texel * float2(-1.0,  1.0)).rgb * 1.0;
    sum += src.sample(smp, uv + texel * float2( 0.0,  1.0)).rgb * 2.0;
    sum += src.sample(smp, uv + texel * float2( 1.0,  1.0)).rgb * 1.0;
    return sum * (1.0 / 16.0);
}

// Quadratic soft-knee threshold (CoD/Unity): pixels above `threshold`
// contribute fully, pixels within `knee` below it ramp in smoothly.
inline float3 prefilter_threshold(float3 c, float threshold, float knee) {
    float br = max(c.r, max(c.g, c.b));
    float kn = max(knee, 1e-4);
    float soft = clamp(br - threshold + kn, 0.0, 2.0 * kn);
    soft = (soft * soft) / (4.0 * kn);
    float contrib = max(soft, br - threshold) / max(br, 1e-4);
    return c * max(contrib, 0.0);
}

fragment float4 bloom_prefilter_fragment(
    BloomVtxOut in [[stage_in]],
    texture2d<float> src [[texture(0)]],
    sampler smp [[sampler(0)]],
    constant PostUniforms& post [[buffer(0)]]
) {
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float3 c = downsample_13(src, smp, in.uv, texel, true);
    // Exposure applies before the threshold so the soft-knee cut tracks the
    // exposed scene the composite pass tonemaps, not the raw HDR radiance.
    c *= post.exposure;
    c = min(c, float3(BLOOM_CLAMP));
    c = prefilter_threshold(c, post.bloom_threshold, post.bloom_knee);
    return float4(c, 1.0);
}

fragment float4 bloom_downsample_fragment(
    BloomVtxOut in [[stage_in]],
    texture2d<float> src [[texture(0)]],
    sampler smp [[sampler(0)]]
) {
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    return float4(downsample_13(src, smp, in.uv, texel, false), 1.0);
}

fragment float4 bloom_upsample_fragment(
    BloomVtxOut in [[stage_in]],
    texture2d<float> src [[texture(0)]],
    sampler smp [[sampler(0)]]
) {
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    return float4(upsample_tent(src, smp, in.uv, texel), 1.0);
}
