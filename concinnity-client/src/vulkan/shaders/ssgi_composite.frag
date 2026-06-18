#version 450

// SSGI composite: a depth-aware box blur over the noisy gather output that the
// pipeline then additively blends (ONE / ONE) into the scene, scaled by the
// authored intensity. Background pixels emit zero so the sky is untouched.
//
// Translated 1:1 from src/directx/shaders/ssgi.hlsl::ssgi_composite_frag /
// src/metal/shaders/ssgi.metal::ssgi_composite_fragment.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// Layout matches gfx::render_types::SsgiParams (32 bytes). Only `intensity` is
// read here, but the whole block is shared with the gather pass.
layout(push_constant) uniform SsgiParamsBlock {
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // Rays / steps loop bounds (read by the gather pass; unused here). Named to
    // match the shared 32-byte SsgiParams layout rather than left as padding.
    float rays;
    float steps;
} params;

// binding 0 = the noisy gather output; binding 1 = the SSR pre-pass G-buffer
// (depth in .a) for the depth-similarity weighting.
layout(set = 0, binding = 0) uniform sampler2D gi_tex;
layout(set = 0, binding = 1) uniform sampler2D gbuffer;

// Depth-aware blur footprint: a (2R+1)^2 box weighted by depth similarity, so
// the indirect term denoises without bleeding across silhouettes.
const int SSGI_BLUR_RADIUS = 2;

void main() {
    float center_depth = texture(gbuffer, frag_uv).a;
    if (center_depth <= 0.0) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec2 texel = 1.0 / vec2(textureSize(gi_tex, 0));
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int dy = -SSGI_BLUR_RADIUS; dy <= SSGI_BLUR_RADIUS; dy++) {
        for (int dx = -SSGI_BLUR_RADIUS; dx <= SSGI_BLUR_RADIUS; dx++) {
            vec2 uv = frag_uv + vec2(float(dx), float(dy)) * texel;
            float d = texture(gbuffer, uv).a;
            if (d <= 0.0) continue;                      // skip background taps
            // Depth-similarity weight: taps on a different surface fall off
            // sharply so the indirect term doesn't bleed across edges.
            float dd = abs(d - center_depth);
            float w = exp2(-dd * 8.0);
            sum  += texture(gi_tex, uv).rgb * w;
            wsum += w;
        }
    }
    vec3 gi = wsum > 0.0 ? sum / wsum : texture(gi_tex, frag_uv).rgb;
    out_color = vec4(gi * params.intensity, 1.0);
}
