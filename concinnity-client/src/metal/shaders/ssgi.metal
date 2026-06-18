#include <metal_stdlib>
using namespace metal;

struct SsgiVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle generated from vertex_id 0..2 - no vertex buffer.
vertex SsgiVtxOut ssgi_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    SsgiVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

// buffer(0): SSGI tunables. Layout matches render_types::SsgiParams.
struct SsgiParams {
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // Rays cast per pixel over the hemisphere, and march samples per ray. Both
    // arrive as floats and are read as int loop bounds; the Rust side derives
    // the stride from `steps`.
    float rays;
    float steps;
};

// Origin offset along the surface normal (× stride) so a ray doesn't
// immediately self-intersect the surface it starts on.
constant float SSGI_NORMAL_BIAS = 0.5;
constant float SSGI_PI = 3.14159265359;
// Depth-aware blur footprint (composite pass): a (2R+1)² box weighted by
// depth similarity, so the indirect term denoises without bleeding across
// silhouettes.
constant int   SSGI_BLUR_RADIUS = 2;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos / ssao_view_pos.
static float3 ssgi_view_pos(float2 uv, float depth, float tan_y, float aspect) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
static float2 ssgi_project(float3 q, float tan_y, float aspect) {
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (tan_y * aspect), q.y * inv / tan_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Interleaved gradient noise: a cheap per-pixel hash in [0, 1). Decorrelates
// the hemisphere sampling spatially; the depth-aware blur + TAA clean up the
// residual high-frequency noise.
static float ssgi_ign(float2 p) {
    return fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
}

// Van der Corput radical inverse (base 2), for the low-discrepancy ray set.
static float ssgi_vdc(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 2^32
}

// Gather pass: per pixel, cast SSGI_RAYS cosine-weighted hemisphere rays around
// the surface normal, screen-march each against the SSR pre-pass G-buffer, and
// accumulate the lit scene colour at each on-screen hit. Misses contribute
// nothing (the IBL ambient already covers the off-screen / sky term). The
// cosine-weighted importance sampling folds the cos θ / pdf factor away, so the
// estimate of the (albedo-free) indirect irradiance is just the mean hit
// radiance.
fragment float4 ssgi_gather_fragment(
    SsgiVtxOut in              [[stage_in]],
    constant SsgiParams &p     [[buffer(0)]],
    texture2d<float>    scene  [[texture(0)]],
    texture2d<float>    gbuffer[[texture(1)]],
    sampler smp                [[sampler(0)]]
) {
    float4 c     = gbuffer.sample(smp, in.uv);
    float  depth = c.a;
    if (depth <= 0.0) return float4(0.0, 0.0, 0.0, 1.0); // background / sky

    float3 N = normalize(c.xyz);
    float3 P = ssgi_view_pos(in.uv, depth, p.tan_half_fov_y, p.aspect);

    // Orthonormal basis around the view-space normal.
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T  = normalize(cross(up, N));
    float3 B  = cross(N, T);

    float jitter = ssgi_ign(in.position.xy);
    float3 origin = P + N * (p.stride * SSGI_NORMAL_BIAS);

    int rays  = max(1, int(p.rays));
    int steps = max(1, int(p.steps));

    float3 indirect = float3(0.0);
    for (int i = 0; i < rays; i++) {
        // Stratified cosine-weighted hemisphere sample, jittered per pixel.
        float u1 = (float(i) + jitter) / float(rays);
        float u2 = fract(ssgi_vdc(uint(i + 1)) + jitter);
        float r   = sqrt(u1);
        float phi = 2.0 * SSGI_PI * u2;
        float3 d_t = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        float3 d   = normalize(T * d_t.x + B * d_t.y + N * d_t.z);

        float3 step_v = d * p.stride;
        float3 q = origin;
        for (int s = 0; s < steps; s++) {
            q += step_v;
            if (q.z >= 0.0) break;                       // crossed camera plane
            float2 uv = ssgi_project(q, p.tan_half_fov_y, p.aspect);
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
            float scene_depth = gbuffer.sample(smp, uv).a;
            if (scene_depth <= 0.0) continue;            // sky here - keep marching
            float diff = (-q.z) - scene_depth;           // > 0: ray is behind the surface
            if (diff > 0.0 && diff < p.thickness) {
                indirect += scene.sample(smp, uv).rgb;   // bounced radiance
                break;
            }
        }
    }

    indirect *= (1.0 / float(rays));
    return float4(indirect, 1.0);
}

// Composite pass: a depth-aware box blur over the noisy gather output that the
// pipeline then additively blends (ONE / ONE) into the scene, scaled by the
// authored intensity. Background pixels emit zero so the sky is untouched.
fragment float4 ssgi_composite_fragment(
    SsgiVtxOut in              [[stage_in]],
    constant SsgiParams &p     [[buffer(0)]],
    texture2d<float>    gi_tex [[texture(0)]],
    texture2d<float>    gbuffer[[texture(1)]],
    sampler smp                [[sampler(0)]]
) {
    float center_depth = gbuffer.sample(smp, in.uv).a;
    if (center_depth <= 0.0) return float4(0.0, 0.0, 0.0, 1.0);

    float2 texel = float2(1.0 / float(gi_tex.get_width()),
                          1.0 / float(gi_tex.get_height()));
    float3 sum = float3(0.0);
    float  wsum = 0.0;
    for (int dy = -SSGI_BLUR_RADIUS; dy <= SSGI_BLUR_RADIUS; dy++) {
        for (int dx = -SSGI_BLUR_RADIUS; dx <= SSGI_BLUR_RADIUS; dx++) {
            float2 uv = in.uv + float2(float(dx), float(dy)) * texel;
            float d = gbuffer.sample(smp, uv).a;
            if (d <= 0.0) continue;                      // skip background taps
            // Depth-similarity weight: taps on a different surface fall off
            // sharply so the indirect term doesn't bleed across edges.
            float dd = abs(d - center_depth);
            float w = exp2(-dd * 8.0);
            sum  += gi_tex.sample(smp, uv).rgb * w;
            wsum += w;
        }
    }
    float3 gi = wsum > 0.0 ? sum / wsum : gi_tex.sample(smp, in.uv).rgb;
    return float4(gi * p.intensity, 1.0);
}
