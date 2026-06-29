Texture2D    hdr_tex        : register(t0);
// Bloom mip 0 (half-res). Always bound - when bloom is disabled the sample is
// skipped, so an unwritten target is never read.
Texture2D    bloom_tex      : register(t1);
// 3D colour-grading LUT. Always bound - a 2x2x2 identity LUT stands in when the
// world declares no ColorLut, so the grade is a no-op at any lut_strength.
Texture3D    lut_tex        : register(t2);
SamplerState linear_sampler : register(s0);

// Post-process tunables. Layout matches gfx::render_types::PostProcessParams
// (32 bytes - 8 floats). `hdr_output` is the runtime EDR toggle: `0.0` keeps
// the SDR ACES + gamma + FXAA + LUT chain; `> 0.5` skips them and emits the
// HDR scene with the matching encoding. Inside the HDR branch, `pq_output`
// picks scRGB-linear passthrough (`0.0` - the compositor handles the panel
// encode) vs SMPTE ST 2084 PQ-encode (`> 0.5` - for HDR10 panels expecting
// PQ-encoded values directly). Always 0.0 on the SDR path.
cbuffer PostBlock : register(b0)
{
    float bloom_intensity;
    float bloom_threshold;
    float bloom_knee;
    float exposure;
    float vignette;
    float lut_strength;
    float hdr_output;
    float pq_output;
    // 1.0 = run the FXAA edge filter on the SDR path; 0.0 = skip it (the Off
    // anti-aliasing mode). Ignored on the HDR path, which never runs FXAA.
    float fxaa;
};

// SDR reference white in cd/m² (nits). BT.2408 recommends 203 nits as the
// HDR mixing reference; classic D-Cinema uses 100. 203 keeps SDR content
// from looking dim alongside HDR highlights and matches the value mainstream
// HDR pipelines (Windows, modern macOS) use as the linear-to-PQ mapping
// reference.
static const float PQ_SDR_REFERENCE_NITS = 203.0;

// PQ inverse-EOTF (a.k.a. PQ OETF / encode). Constants from SMPTE ST 2084 /
// ITU-R BT.2100. Maps absolute luminance in cd/m² (capped at 10000) onto
// the perceptually-quantized signal in [0, 1] the HDR10 panel expects.
// Mirrors `pq_encode` in metal/shaders/post.metal.
float3 pq_encode(float3 L_nits)
{
    const float m1 = 0.1593017578125;     // = 2610 / 16384
    const float m2 = 78.84375;            // = 2523 / 4096 * 128
    const float c1 = 0.8359375;           // = 3424 / 4096
    const float c2 = 18.8515625;          // = 2413 / 4096 * 32
    const float c3 = 18.6875;             // = 2392 / 4096 * 32
    float3 L_n = clamp(L_nits * (1.0 / 10000.0), 0.0, 1.0);
    float3 Lm1 = pow(L_n, m1);
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), m2);
}

struct PsIn
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

// Narkowicz 2015 ACES fit - closed-form approximation of the ACES RRT+ODT.
float3 aces_narkowicz(float3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// FXAA luma weighting on a display-referred sRGB pixel.
float fxaa_luma(float3 rgb)
{
    return dot(rgb, float3(0.299, 0.587, 0.114));
}

// Scene colour = exposed HDR scene + bloom. Exposure scales the HDR tap only -
// the bloom mip already carries the exposure applied in the prefilter.
float3 scene_sample(float2 uv)
{
    float3 c = hdr_tex.SampleLevel(linear_sampler, uv, 0.0).rgb * exposure;
    if (bloom_intensity > 0.0)
    {
        c += bloom_tex.SampleLevel(linear_sampler, uv, 0.0).rgb * bloom_intensity;
    }
    return c;
}

// Scene sample -> ACES tonemap -> gamma 2.2 encode.
float3 tonemap(float2 uv)
{
    return pow(aces_narkowicz(scene_sample(uv)), float3(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2));
}

// Smooth radial corner darkening. `vignette` 0 disables it.
float vignette_factor(float2 uv)
{
    float2 d = uv - 0.5;
    float dist = dot(d, d) * 2.0;
    return 1.0 - vignette * smoothstep(0.25, 1.0, dist);
}

// Sample the 3D colour-grading LUT with the tonemapped, display-referred sRGB
// colour. The half-texel correction maps an input of 0 / 1 to the centres of
// the first / last texels so trilinear filtering stays accurate edge to edge;
// a 2x2x2 identity LUT then reproduces the input exactly. Mirrors `apply_lut`
// in metal/pipeline.rs and the Vulkan COMPOSITE_FRAG_GLSL.
float3 apply_lut(float3 c)
{
    uint w, h, d;
    lut_tex.GetDimensions(w, h, d);
    float n = (float)w;
    float3 uvw = clamp(c, 0.0, 1.0) * ((n - 1.0) / n) + (0.5 / n);
    return lut_tex.SampleLevel(linear_sampler, uvw, 0.0).rgb;
}

// Blend the LUT-graded colour over the input by `lut_strength`. Grading the
// display-referred LDR result keeps the LUT independent of exposure / tonemap.
float3 grade(float3 c)
{
    return lerp(c, apply_lut(c), lut_strength);
}

float4 main(PsIn p) : SV_TARGET
{
    uint w, h;
    hdr_tex.GetDimensions(w, h);
    float2 inv_size = 1.0 / float2((float)w, (float)h);

    // HDR EDR output: skip ACES + gamma + FXAA + LUT. Two flavours, picked
    // by `pq_output`:
    //
    //   - `pq_output <= 0.5` - scRGB linear. The swapchain was created with
    //     `DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709`; the compositor wants
    //     linear extended-range values where `1.0` is SDR reference white
    //     and values above that drive the panel's HDR headroom.
    //   - `pq_output  > 0.5` - HDR10 PQ. The swapchain was created with
    //     `DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020`; the panel decodes
    //     via the PQ EOTF, so we encode in-shader. SDR reference white maps
    //     to 203 nits per BT.2408.
    //
    // Vignette is applied in linear space before the optional PQ encode
    // because it is a multiplicative falloff defined on luminance. Mirrors
    // the HDR branch in metal/shaders/post.metal.
    if (hdr_output > 0.5)
    {
        float3 hdr_c = scene_sample(p.uv);
        float vig_hdr = vignette_factor(p.uv);
        float3 hdr_vig = hdr_c * vig_hdr;
        if (pq_output > 0.5)
        {
            return float4(pq_encode(hdr_vig * PQ_SDR_REFERENCE_NITS), 1.0);
        }
        return float4(hdr_vig, 1.0);
    }

    // Tonemap the centre sample and its 4-neighbourhood; each neighbour goes
    // through the same tonemap so the FXAA luma compares stay consistent.
    float3 c = tonemap(p.uv);

    // FXAA gated by `fxaa` (off for the Off anti-aliasing mode). When disabled,
    // grade + vignette the tonemapped centre directly and skip the neighbour
    // samples the edge filter would otherwise take.
    if (fxaa < 0.5)
    {
        return float4(grade(c) * vignette_factor(p.uv), 1.0);
    }

    float3 n = tonemap(p.uv + float2(0.0, -inv_size.y));
    float3 s = tonemap(p.uv + float2(0.0,  inv_size.y));
    float3 e = tonemap(p.uv + float2( inv_size.x, 0.0));
    float3 w_ = tonemap(p.uv + float2(-inv_size.x, 0.0));

    float lC = fxaa_luma(c);
    float lN = fxaa_luma(n);
    float lS = fxaa_luma(s);
    float lE = fxaa_luma(e);
    float lW = fxaa_luma(w_);

    float l_min = min(lC, min(min(lN, lS), min(lE, lW)));
    float l_max = max(lC, max(max(lN, lS), max(lE, lW)));
    float l_range = l_max - l_min;

    float vig = vignette_factor(p.uv);

    // Flat regions skip the blur entirely; they are still colour-graded and
    // vignetted.
    if (l_range < max(0.0312, l_max * 0.125))
    {
        return float4(grade(c) * vig, 1.0);
    }

    // Pick the dominant edge direction, then step half a texel along the
    // perpendicular for a 2-tap blur averaged with the centre.
    float horz_diff = abs(lN + lS - 2.0 * lC) * 2.0 + abs(lE + lW - 2.0 * lC);
    float vert_diff = abs(lE + lW - 2.0 * lC) * 2.0 + abs(lN + lS - 2.0 * lC);
    bool horizontal = horz_diff >= vert_diff;

    float2 step_dir = horizontal ? float2(0.0, inv_size.y)
                                 : float2(inv_size.x, 0.0);
    float3 a = tonemap(p.uv + step_dir * 0.5);
    float3 b = tonemap(p.uv - step_dir * 0.5);
    float3 blended = (c + a + b) * (1.0 / 3.0);

    // Grade the FXAA-resolved colour, then vignette last so it darkens the
    // final composited result.
    return float4(grade(blended) * vig, 1.0);
}
