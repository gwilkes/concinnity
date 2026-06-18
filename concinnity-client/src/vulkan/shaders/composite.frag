#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D hdr_tex;
// Bloom mip 0 (half-res). Always bound - when bloom is disabled the sample is
// skipped, so an unwritten target is never read.
layout(set = 0, binding = 1) uniform sampler2D bloom_tex;
// 3D colour-grading LUT. Always bound - a 2x2x2 identity LUT stands in when the
// world declares no ColorLut, so the grade is a no-op at any lut_strength.
layout(set = 0, binding = 2) uniform sampler3D lut_tex;

// Post-process tunables. Layout matches gfx::render_types::PostProcessParams
// (32 bytes - 8 floats). `hdr_output > 0.5` switches to the EDR path: exposed
// HDR scene goes out without the ACES + gamma + FXAA + LUT chain. Inside that
// branch `pq_output` picks the encoding: `<= 0.5` emits scRGB-linear
// passthrough (Rec.709 primaries, gamma 1.0; the compositor maps `1.0` to SDR
// reference white and values above drive the panel headroom), `> 0.5`
// PQ-encodes (SMPTE ST 2084) in-shader for an HDR10 panel expecting
// PQ-encoded values directly. Always 0.0 on the SDR path.
layout(push_constant) uniform PostBlock {
    float bloom_intensity;
    float bloom_threshold;
    float bloom_knee;
    float exposure;
    float vignette;
    float lut_strength;
    float hdr_output;
    float pq_output;
} post;

// SDR reference white in cd/m2 (nits). BT.2408 recommends 203 nits as the HDR
// mixing reference; it keeps SDR content from looking dim alongside HDR
// highlights and matches the value mainstream HDR pipelines use for the
// linear-to-PQ mapping reference.
const float PQ_SDR_REFERENCE_NITS = 203.0;

// PQ inverse-EOTF (PQ OETF / encode). Constants from SMPTE ST 2084 / ITU-R
// BT.2100. Maps absolute luminance in cd/m2 (capped at 10000) onto the
// perceptually-quantized signal in [0, 1] the HDR10 panel expects. Mirrors
// `pq_encode` in directx/shaders/composite_frag.hlsl and metal/shaders/post.metal.
vec3 pq_encode(vec3 L_nits) {
    const float m1 = 0.1593017578125;     // = 2610 / 16384
    const float m2 = 78.84375;            // = 2523 / 4096 * 128
    const float c1 = 0.8359375;           // = 3424 / 4096
    const float c2 = 18.8515625;          // = 2413 / 4096 * 32
    const float c3 = 18.6875;             // = 2392 / 4096 * 32
    vec3 L_n = clamp(L_nits * (1.0 / 10000.0), 0.0, 1.0);
    vec3 Lm1 = pow(L_n, vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

// Narkowicz 2015 ACES fit - closed-form approximation of the ACES RRT+ODT.
vec3 aces_narkowicz(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// FXAA luma weighting on a display-referred sRGB pixel.
float fxaa_luma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

// Scene colour = exposed HDR resolve + bloom. Exposure scales the HDR tap
// only - the bloom mip already carries the exposure applied in the prefilter.
vec3 scene_sample(vec2 uv) {
    vec3 c = texture(hdr_tex, uv).rgb * post.exposure;
    if (post.bloom_intensity > 0.0) {
        c += texture(bloom_tex, uv).rgb * post.bloom_intensity;
    }
    return c;
}

// Scene sample → ACES tonemap → gamma 2.2 encode.
vec3 tonemap(vec2 uv) {
    return pow(aces_narkowicz(scene_sample(uv)), vec3(1.0 / 2.2));
}

// Smooth radial corner darkening. `strength` 0 disables it.
float vignette_factor(vec2 uv) {
    vec2 d = uv - 0.5;
    float dist = dot(d, d) * 2.0;
    return 1.0 - post.vignette * smoothstep(0.25, 1.0, dist);
}

// Sample the 3D colour-grading LUT with the tonemapped, display-referred sRGB
// colour. The half-texel correction maps an input of 0 / 1 to the centres of
// the first / last texels so trilinear filtering stays accurate edge to edge;
// a 2x2x2 identity LUT then reproduces the input exactly. Mirrors `apply_lut`
// in metal/pipeline.rs.
vec3 apply_lut(vec3 c) {
    float n = float(textureSize(lut_tex, 0).x);
    vec3 uvw = clamp(c, 0.0, 1.0) * ((n - 1.0) / n) + (0.5 / n);
    return texture(lut_tex, uvw).rgb;
}

// Blend the LUT-graded colour over the input by `lut_strength`. Grading the
// display-referred LDR result keeps the LUT independent of exposure / tonemap.
vec3 grade(vec3 c) {
    return mix(c, apply_lut(c), post.lut_strength);
}

void main() {
    vec2 inv_size = 1.0 / vec2(textureSize(hdr_tex, 0));

    // HDR EDR output: skip ACES + gamma + FXAA + LUT. Two flavours, picked by
    // `pq_output`:
    //
    //   - `pq_output <= 0.5` - scRGB linear. The swapchain was created with
    //     `VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT`; the compositor wants
    //     linear extended-range values where `1.0` is SDR reference white and
    //     values above that drive the panel's HDR headroom.
    //   - `pq_output  > 0.5` - HDR10 PQ. The swapchain was created with
    //     `VK_COLOR_SPACE_HDR10_ST2084_EXT`; the panel decodes via the PQ
    //     EOTF, so we encode in-shader. SDR reference white maps to 203 nits
    //     per BT.2408.
    //
    // Vignette is applied in linear space before the optional PQ encode because
    // it is a multiplicative falloff defined on luminance. Mirrors the HDR
    // branch in directx/shaders/composite_frag.hlsl and metal/shaders/post.metal.
    if (post.hdr_output > 0.5) {
        vec3 hdr_c = scene_sample(frag_uv);
        float vig_hdr = vignette_factor(frag_uv);
        vec3 hdr_vig = hdr_c * vig_hdr;
        if (post.pq_output > 0.5) {
            out_color = vec4(pq_encode(hdr_vig * PQ_SDR_REFERENCE_NITS), 1.0);
        } else {
            out_color = vec4(hdr_vig, 1.0);
        }
        return;
    }

    // 1) Composite bloom onto the HDR scene, then ACES tonemap + gamma encode
    //    the centre sample and its 4-neighbourhood. Each neighbour is remapped
    //    through the same tonemap so luma compares stay consistent.
    vec3 c = tonemap(frag_uv);
    vec3 n = tonemap(frag_uv + vec2(0.0, -inv_size.y));
    vec3 s = tonemap(frag_uv + vec2(0.0,  inv_size.y));
    vec3 e = tonemap(frag_uv + vec2( inv_size.x, 0.0));
    vec3 w = tonemap(frag_uv + vec2(-inv_size.x, 0.0));

    float lC = fxaa_luma(c);
    float lN = fxaa_luma(n);
    float lS = fxaa_luma(s);
    float lE = fxaa_luma(e);
    float lW = fxaa_luma(w);

    float l_min = min(lC, min(min(lN, lS), min(lE, lW)));
    float l_max = max(lC, max(max(lN, lS), max(lE, lW)));
    float l_range = l_max - l_min;

    float vig = vignette_factor(frag_uv);

    // 2) FXAA 3.11-style edge filter. Flat regions skip the blur entirely;
    //    they are still colour-graded and vignetted.
    if (l_range < max(0.0312, l_max * 0.125)) {
        out_color = vec4(grade(c) * vig, 1.0);
        return;
    }

    // Pick the dominant edge direction, then step half a texel along the
    // perpendicular for a 2-tap blur averaged with the centre.
    float horz_diff = abs(lN + lS - 2.0 * lC) * 2.0 + abs(lE + lW - 2.0 * lC);
    float vert_diff = abs(lE + lW - 2.0 * lC) * 2.0 + abs(lN + lS - 2.0 * lC);
    bool horizontal = horz_diff >= vert_diff;

    vec2 step_dir = horizontal ? vec2(0.0, inv_size.y) : vec2(inv_size.x, 0.0);
    vec3 a = tonemap(frag_uv + step_dir * 0.5);
    vec3 b = tonemap(frag_uv - step_dir * 0.5);
    vec3 blended = (c + a + b) * (1.0 / 3.0);

    // Grade the FXAA-resolved colour, then vignette last so it darkens the
    // final composited result.
    out_color = vec4(grade(blended) * vig, 1.0);
}
