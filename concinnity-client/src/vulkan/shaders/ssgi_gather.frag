#version 450

// SSGI gather: per pixel, cast `rays` cosine-weighted hemisphere rays around
// the surface normal, screen-march each against the SSR pre-pass G-buffer, and
// accumulate the lit scene colour at each on-screen hit into the off-screen `gi`
// target. Misses contribute nothing (the IBL ambient already covers the
// off-screen / sky term). The cosine-weighted importance sampling folds the
// cos(theta) / pdf factor away, so the estimate of the (albedo-free) indirect
// irradiance is just the mean hit radiance.
//
// Translated 1:1 from src/directx/shaders/ssgi.hlsl::ssgi_gather_frag /
// src/metal/shaders/ssgi.metal::ssgi_gather_fragment. The view-space
// reconstruction + projection match ssr_resolve.frag (ssgi_view_pos /
// ssgi_project), so the gather agrees with the G-buffer the SSR pre-pass wrote
// (rgb = unit view normal, a = -view_z).

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// Layout matches gfx::render_types::SsgiParams (32 bytes).
layout(push_constant) uniform SsgiParamsBlock {
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // Hemisphere rays per pixel + ray-march samples per ray, read as int loop
    // bounds below (carried as f32 to keep the 32-byte layout). Mirrors
    // ssgi.metal, which reads int(p.rays) / int(p.steps).
    float rays;
    float steps;
} params;

// binding 0 = lit scene radiance (the bounce-radiance source); binding 1 = the
// SSR pre-pass G-buffer (rgb = view normal, a = linear view depth).
layout(set = 0, binding = 0) uniform sampler2D scene;
layout(set = 0, binding = 1) uniform sampler2D gbuffer;

// Origin offset along the surface normal (x stride) so a ray doesn't
// immediately self-intersect the surface it starts on.
const float SSGI_NORMAL_BIAS = 0.5;
const float SSGI_PI = 3.14159265359;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos / ssao_view_pos.
vec3 ssgi_view_pos(vec2 uv, float depth, float tan_y, float asp) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * tan_y * asp, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
vec2 ssgi_project(vec3 q, float tan_y, float asp) {
    float inv = 1.0 / max(-q.z, 1e-4);
    vec2 ndc = vec2(q.x * inv / (tan_y * asp), q.y * inv / tan_y);
    return vec2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Interleaved gradient noise: a cheap per-pixel hash in [0, 1). Decorrelates the
// hemisphere sampling spatially; the depth-aware blur + TAA clean up the
// residual high-frequency noise.
float ssgi_ign(vec2 p) {
    return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
}

// Van der Corput radical inverse (base 2), for the low-discrepancy ray set.
float ssgi_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 2^32
}

void main() {
    vec4 c     = texture(gbuffer, frag_uv);
    float depth = c.a;
    if (depth <= 0.0) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0); // background / sky
        return;
    }

    vec3 N = normalize(c.xyz);
    vec3 P = ssgi_view_pos(frag_uv, depth, params.tan_half_fov_y, params.aspect);

    // Orthonormal basis around the view-space normal.
    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);

    float jitter = ssgi_ign(gl_FragCoord.xy);
    vec3 origin = P + N * (params.stride * SSGI_NORMAL_BIAS);

    // Hemisphere rays + march samples per ray, read from the push constant
    // (floored at 1). Mirrors ssgi.metal:102-103.
    int ray_count  = max(1, int(params.rays));
    int step_count = max(1, int(params.steps));

    vec3 indirect = vec3(0.0);
    for (int i = 0; i < ray_count; i++) {
        // Stratified cosine-weighted hemisphere sample, jittered per pixel.
        float u1 = (float(i) + jitter) / float(ray_count);
        float u2 = fract(ssgi_vdc(uint(i + 1)) + jitter);
        float r   = sqrt(u1);
        float phi = 2.0 * SSGI_PI * u2;
        vec3 d_t = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        vec3 d   = normalize(T * d_t.x + B * d_t.y + N * d_t.z);

        vec3 step_v = d * params.stride;
        vec3 q = origin;
        for (int s = 0; s < step_count; s++) {
            q += step_v;
            if (q.z >= 0.0) break;                       // crossed the camera plane
            vec2 uv = ssgi_project(q, params.tan_half_fov_y, params.aspect);
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
            float scene_depth = texture(gbuffer, uv).a;
            if (scene_depth <= 0.0) continue;            // sky here -- keep marching
            float diff = (-q.z) - scene_depth;           // > 0: ray is behind the surface
            if (diff > 0.0 && diff < params.thickness) {
                indirect += texture(scene, uv).rgb;      // bounced radiance
                break;
            }
        }
    }

    indirect *= (1.0 / float(ray_count));
    out_color = vec4(indirect, 1.0);
}
