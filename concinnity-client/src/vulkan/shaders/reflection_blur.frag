#version 450

// Reflection composite, pass 1 (reduced resolution): weight-averages the SSR / RT
// resolve target over a roughness-scaled cone into the reduced-resolution blur
// target. This is the expensive multi-tap part (up to 17 taps), so running it at a
// fraction of the pixels is the saving; the composite pass bilinear-upsamples it.
// Ports src/metal/shaders/reflection_composite.metal::reflection_blur_fragment.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// The resolve target (rgb = reflected radiance, a = Fresnel/gloss composite weight)
// and the G-buffer roughness the cone radius keys off.
layout(set = 0, binding = 0) uniform sampler2D reflection;
layout(set = 0, binding = 1) uniform sampler2D rough_tex;

// Surfaces rougher than this get no reflection (weight already 0); below it the
// blur cone ramps from a sharp mirror at 0 to REFL_BLUR_MAX. Matches the resolve
// gloss gate (SSR_ROUGH_CUT / RT_ROUGH_CUT in ssr_resolve.frag / rt_reflections.frag);
// keep the three in sync.
const float REFLECTION_ROUGHNESS_CUT = 0.6;

// Largest blur-cone radius in UV at the cut. This is the composite blur cone, a
// separate quantity from the resolve's own in-tap gather radius.
const float REFL_BLUR_MAX = 0.02;

// Two 8-tap rings (45-degree steps) at half and full radius approximate the
// widening glossy reflection cone.
const vec2 REFL_RING[8] = vec2[8](
    vec2( 1.0,         0.0       ), vec2( 0.70710678,  0.70710678),
    vec2( 0.0,         1.0       ), vec2(-0.70710678,  0.70710678),
    vec2(-1.0,         0.0       ), vec2(-0.70710678, -0.70710678),
    vec2( 0.0,        -1.0       ), vec2( 0.70710678, -0.70710678)
);

void main() {
    vec4  c         = texture(reflection, frag_uv);
    float roughness = texture(rough_tex, frag_uv).r;
    float radius    = clamp(roughness / REFLECTION_ROUGHNESS_CUT, 0.0, 1.0) * REFL_BLUR_MAX;

    // Weight each tap by its own coverage so weightless (non-reflecting) taps
    // cannot drag their colour in; a reflective edge then fades smoothly.
    vec3  sum_rw = c.rgb * c.a;
    float sum_w  = c.a;
    float taps   = 1.0;
    if (radius > 1e-5) {
        for (int ring = 0; ring < 2; ring++) {
            float rr = radius * (ring == 0 ? 0.5 : 1.0);
            for (int i = 0; i < 8; i++) {
                vec4 t = texture(reflection, frag_uv + REFL_RING[i] * rr);
                sum_rw += t.rgb * t.a;
                sum_w  += t.a;
                taps   += 1.0;
            }
        }
    }
    vec3  blurred = sum_w > 1e-4 ? sum_rw / sum_w : c.rgb;
    float weight  = sum_w / taps;
    out_color = vec4(blurred, weight);
}
