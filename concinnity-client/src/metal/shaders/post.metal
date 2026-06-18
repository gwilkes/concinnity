#include <metal_stdlib>
using namespace metal;

struct PostVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle generated from vertex_id 0..2 - no vertex buffer
// needed. Covers the viewport with one big triangle whose UVs map [0,1]
// across the visible area.
vertex PostVtxOut post_vertex_main(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    PostVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    // Flip Y so the sampled HDR target matches NDC orientation.
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

// Narkowicz 2015 ACES fit - closed-form approximation of the ACES RRT+ODT.
inline float3 aces_narkowicz(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// FXAA luma weighting on a linear sRGB pixel.
inline float fxaa_luma(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

// Post-process tunables. Layout matches render_types::PostProcessParams.
struct PostUniforms {
    float bloom_intensity;
    float bloom_threshold;
    float bloom_knee;
    float exposure;
    float vignette;
    float lut_strength;
    // 0.0 = SDR path (ACES + gamma 2.2 + FXAA + ColorLut into BGRA8Unorm).
    // 1.0 = HDR EDR path (linear extended-range values into RGBA16Float;
    // ACES / gamma / FXAA / LUT are all skipped because they assume a
    // display-referred output range).
    float hdr_output;
    // Inside the HDR branch: 0.0 = scRGB-linear passthrough, 1.0 = PQ
    // encode (SMPTE ST 2084) for HDR10 panels.
    float pq_output;
};

// SDR reference white in cd/m² (nits). BT.2408 recommends 203 nits as the
// HDR mixing reference; classic D-Cinema uses 100. 203 keeps SDR content
// from looking dim alongside HDR highlights and matches the value mainstream
// HDR pipelines (Windows, modern macOS) use as the linear-to-PQ mapping
// reference.
constant float PQ_SDR_REFERENCE_NITS = 203.0;

// PQ inverse-EOTF (a.k.a. PQ OETF / encode). Constants from SMPTE ST 2084 /
// ITU-R BT.2100. Maps absolute luminance in cd/m² (capped at 10000) onto
// the perceptually-quantized signal in [0, 1] the HDR10 panel expects.
inline float3 pq_encode(float3 L_nits) {
    constexpr float m1 = 0.1593017578125;     // = 2610 / 16384
    constexpr float m2 = 78.84375;            // = 2523 / 4096 * 128
    constexpr float c1 = 0.8359375;           // = 3424 / 4096
    constexpr float c2 = 18.8515625;          // = 2413 / 4096 * 32
    constexpr float c3 = 18.6875;             // = 2392 / 4096 * 32
    float3 L_n = clamp(L_nits * (1.0 / 10000.0), 0.0, 1.0);
    float3 Lm1 = pow(L_n, m1);
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2);
}

// Sample a 3D colour-grading LUT with the tonemapped, display-referred sRGB
// colour. The half-texel correction maps an input of 0 / 1 to the centres of
// the first / last texels so trilinear filtering stays accurate edge to edge;
// a 2x2x2 identity LUT then reproduces the input exactly.
inline float3 apply_lut(texture3d<float> lut, sampler smp, float3 c) {
    float n = float(lut.get_width());
    float3 uvw = clamp(c, 0.0, 1.0) * ((n - 1.0) / n) + (0.5 / n);
    return lut.sample(smp, uvw).rgb;
}

// Scene colour = exposed HDR resolve + bloom. Exposure scales the HDR tap
// only - the bloom mip already carries the exposure applied in the prefilter.
// Bloom mip 0 is half-res; the linear-clamp sampler upscales it smoothly. The
// sample is skipped when bloom is disabled so an uninitialised bloom target is
// never read.
inline float3 scene_sample(texture2d<float> hdr, texture2d<float> bloom,
                           sampler smp, float2 uv, float bloom_intensity,
                           float exposure) {
    float3 c = hdr.sample(smp, uv).rgb * exposure;
    if (bloom_intensity > 0.0) {
        c += bloom.sample(smp, uv).rgb * bloom_intensity;
    }
    return c;
}

// Smooth radial corner darkening. `strength` 0 disables it; 1 fully darkens
// the corners. The squared-distance falloff keeps the centre untouched.
inline float vignette_factor(float2 uv, float strength) {
    float2 d = uv - 0.5;
    float dist = dot(d, d) * 2.0;
    return 1.0 - strength * smoothstep(0.25, 1.0, dist);
}

fragment float4 post_fragment_main(
    PostVtxOut in [[stage_in]],
    texture2d<float> hdr [[texture(0)]],
    texture2d<float> bloom [[texture(1)]],
    texture3d<float> lut [[texture(2)]],
    sampler smp [[sampler(0)]],
    constant PostUniforms& post [[buffer(0)]]
) {
    float2 uv = in.uv;
    float2 tex_size = float2(hdr.get_width(), hdr.get_height());
    float2 inv_size = 1.0 / tex_size;
    float bi = post.bloom_intensity;
    float ex = post.exposure;
    float vig = vignette_factor(uv, post.vignette);

    // 1) Composite bloom onto the HDR scene, then ACES tonemap + gamma encode.
    float3 hdr_c = scene_sample(hdr, bloom, smp, uv, bi, ex);

    // HDR EDR output: skip ACES + gamma + FXAA + LUT. The CAMetalLayer is
    // configured for a Display P3 EDR colour space with
    // `wantsExtendedDynamicRangeContent = true`. Two sub-paths:
    //
    //   - scRGB-linear (extended-linear Display P3): output linear values
    //     where `1.0` is SDR reference white. The OS compositor handles the
    //     final encode for whatever the panel needs.
    //   - PQ (Display P3 PQ, SMPTE ST 2084): output PQ-encoded values
    //     directly. The panel decodes via the PQ EOTF. SDR reference white
    //     maps to PQ_SDR_REFERENCE_NITS so the same scene reads at the same
    //     apparent brightness as the scRGB-linear path.
    //
    // Vignette is applied in linear space before the optional PQ encode
    // because it is a multiplicative falloff defined on luminance.
    if (post.hdr_output > 0.5) {
        float3 hdr_vig = hdr_c * vig;
        if (post.pq_output > 0.5) {
            return float4(pq_encode(hdr_vig * PQ_SDR_REFERENCE_NITS), 1.0);
        }
        return float4(hdr_vig, 1.0);
    }

    float3 ldr_c = pow(aces_narkowicz(hdr_c), float3(1.0 / 2.2));

    // 2) FXAA 3.11-style edge detection on the encoded image. Each neighbour
    //    is composited with bloom and remapped through the same tonemap +
    //    gamma so luma compares stay consistent with the centre sample.
    float3 n  = pow(aces_narkowicz(scene_sample(hdr, bloom, smp, uv + float2(0.0, -inv_size.y), bi, ex)), float3(1.0 / 2.2));
    float3 s  = pow(aces_narkowicz(scene_sample(hdr, bloom, smp, uv + float2(0.0,  inv_size.y), bi, ex)), float3(1.0 / 2.2));
    float3 e  = pow(aces_narkowicz(scene_sample(hdr, bloom, smp, uv + float2( inv_size.x, 0.0), bi, ex)), float3(1.0 / 2.2));
    float3 w  = pow(aces_narkowicz(scene_sample(hdr, bloom, smp, uv + float2(-inv_size.x, 0.0), bi, ex)), float3(1.0 / 2.2));

    float lN = fxaa_luma(n);
    float lS = fxaa_luma(s);
    float lE = fxaa_luma(e);
    float lW = fxaa_luma(w);
    float lC = fxaa_luma(ldr_c);

    float l_min = min(lC, min(min(lN, lS), min(lE, lW)));
    float l_max = max(lC, max(max(lN, lS), max(lE, lW)));
    float l_range = l_max - l_min;

    // Skip flat regions - FXAA edge threshold (relative + absolute).
    if (l_range < max(0.0312, l_max * 0.125)) {
        float3 graded = mix(ldr_c, apply_lut(lut, smp, ldr_c), post.lut_strength);
        return float4(graded * vig, 1.0);
    }

    // Pick the dominant edge direction (horizontal vs vertical) and step
    // half a texel along the perpendicular for a 2-tap blur.
    float horz_diff = abs(lN + lS - 2.0 * lC) * 2.0 + abs(lE + lW - 2.0 * lC);
    float vert_diff = abs(lE + lW - 2.0 * lC) * 2.0 + abs(lN + lS - 2.0 * lC);
    bool horizontal = horz_diff >= vert_diff;

    float2 step_dir = horizontal ? float2(0.0, inv_size.y) : float2(inv_size.x, 0.0);
    float3 a = pow(aces_narkowicz(scene_sample(hdr, bloom, smp, uv + step_dir * 0.5, bi, ex)), float3(1.0 / 2.2));
    float3 b = pow(aces_narkowicz(scene_sample(hdr, bloom, smp, uv - step_dir * 0.5, bi, ex)), float3(1.0 / 2.2));
    float3 blended = (ldr_c + a + b) * (1.0 / 3.0);

    // Colour-grade the FXAA-resolved LDR colour, then vignette. Grading on the
    // display-referred result keeps the LUT independent of exposure / tonemap;
    // the vignette stays last so it darkens the final composited colour.
    float3 graded = mix(blended, apply_lut(lut, smp, blended), post.lut_strength);
    return float4(graded * vig, 1.0);
}
