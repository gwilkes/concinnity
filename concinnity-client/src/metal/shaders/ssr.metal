#include <metal_stdlib>
using namespace metal;

struct SsrVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle generated from vertex_id 0..2 - no vertex buffer.
vertex SsrVtxOut ssr_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    SsrVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

// buffer(0): SSR tunables. Layout matches render_types::SsrParams.
struct SsrParams {
    float    intensity;
    float    max_distance;
    float    tan_half_fov_y;
    float    aspect;
    float    stride;
    float    thickness;
    // IBL prefilter cubemap mip count; 0 means no EnvironmentMap is bound and
    // the cube fallback is skipped (missed rays keep the base shading).
    float    prefilter_mip_count;
    float    _pad;
    // View-space to world-space rotation; turns the view-space reflection ray
    // into the world-space direction the prefilter cubemap is sampled with.
    float4x4 inv_view_rot;
};

constant int   SSR_MAX_STEPS = 48;
constant int   SSR_REFINE    = 5;
// Surfaces rougher than this get no SSR; glossiness ramps in below it.
constant float SSR_ROUGH_CUT = 0.6;
// Dielectric base reflectance (water, glass, polished stone) for the Fresnel.
constant float SSR_F0        = 0.04;
// UV margin over which a hit near the screen border fades out.
constant float SSR_EDGE_FADE = 0.12;
// Largest screen-space (UV) blur radius, reached as roughness approaches the
// cut-off. The reflected scene colour is gathered over a disk this wide so a
// glossy-but-not-mirror surface reflects a blurred image; a near-zero
// roughness shrinks the radius to a single sharp tap.
constant float SSR_BLUR_MAX  = 0.018;

// Eight evenly spaced offsets on the unit circle (cos/sin of k * 45 deg) for
// the roughness-keyed reflection gather.
constant float2 SSR_BLUR_RING[8] = {
    float2( 1.0,         0.0       ), float2( 0.70710678,  0.70710678),
    float2( 0.0,         1.0       ), float2(-0.70710678,  0.70710678),
    float2(-1.0,         0.0       ), float2(-0.70710678, -0.70710678),
    float2( 0.0,        -1.0       ), float2( 0.70710678, -0.70710678),
};

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// The inverse of ssr_project; matches ssao_view_pos in the SSAO kernel.
static float3 ssr_view_pos(float2 uv, float depth, float tan_y, float aspect) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * aspect, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
static float2 ssr_project(float3 q, float tan_y, float aspect) {
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (tan_y * aspect), q.y * inv / tan_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Gather the reflected scene colour over a roughness-scaled disk. A zero
// radius is one sharp tap (mirror); a wider radius averages an eight-tap ring
// plus the centre, approximating the wider reflection cone of a rough surface.
static float3 ssr_gather(texture2d<float> scene, sampler smp, float2 uv, float radius) {
    float3 c = scene.sample(smp, uv).rgb;
    if (radius <= 1e-5) {
        return c;
    }
    for (int i = 0; i < 8; i++) {
        c += scene.sample(smp, uv + SSR_BLUR_RING[i] * radius).rgb;
    }
    return c * (1.0 / 9.0);
}

fragment float4 ssr_resolve_fragment(
    SsrVtxOut in                  [[stage_in]],
    constant SsrParams &p         [[buffer(0)]],
    texture2d<float>   scene      [[texture(0)]],
    texture2d<float>   gbuffer    [[texture(1)]],
    texture2d<float>   rough_tex  [[texture(2)]],
    texturecube<float> prefilter  [[texture(3)]],
    sampler smp                   [[sampler(0)]],
    sampler cube_smp              [[sampler(1)]]
) {
    float3 base  = scene.sample(smp, in.uv).rgb;
    float4 c     = gbuffer.sample(smp, in.uv);
    float  depth = c.a;
    if (depth <= 0.0) return float4(base, 1.0);     // background / sky

    float roughness = rough_tex.sample(smp, in.uv).r;
    // Glossy surfaces reflect sharply; rough ones get nothing.
    float gloss = saturate((SSR_ROUGH_CUT - roughness) / SSR_ROUGH_CUT);
    if (gloss <= 0.0) return float4(base, 1.0);

    float3 N = normalize(c.xyz);
    float3 P = ssr_view_pos(in.uv, depth, p.tan_half_fov_y, p.aspect);
    float3 V = normalize(-P);                       // P in view space, camera at origin
    float3 R = reflect(-V, N);                      // reflected ray direction

    // Environment fallback: the reflection the IBL prefilter cubemap gives in
    // the reflected direction, sampled at a roughness-keyed mip so a rougher
    // surface reflects a blurrier environment (matching the main pass). With
    // no EnvironmentMap bound there is nothing to fall back to, so the
    // environment stays the base shading and missed rays behave as before.
    bool   ibl = p.prefilter_mip_count > 0.5;
    float3 env = base;
    if (ibl) {
        float3 r_world = (p.inv_view_rot * float4(R, 0.0)).xyz;
        float  lod     = roughness * (p.prefilter_mip_count - 1.0);
        env = prefilter.sample(cube_smp, r_world, level(lod)).rgb;
    }

    float3 step_v = R * p.stride;
    float3 q = P;
    bool   hit = false;
    float2 hit_uv = in.uv;
    int    steps_taken = SSR_MAX_STEPS;
    for (int i = 0; i < SSR_MAX_STEPS; i++) {
        q += step_v;
        if (q.z >= 0.0) { steps_taken = i; break; } // crossed the camera plane
        float2 uv = ssr_project(q, p.tan_half_fov_y, p.aspect);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            steps_taken = i;
            break;
        }
        float scene_depth = gbuffer.sample(smp, uv).a;
        if (scene_depth <= 0.0) continue;           // sky here - keep marching
        float diff = (-q.z) - scene_depth;          // > 0: ray is behind the surface
        if (diff > 0.0 && diff < p.thickness) {
            // Binary-search refine between the last two samples.
            float3 lo = q - step_v;
            float3 hi = q;
            for (int r = 0; r < SSR_REFINE; r++) {
                float3 mid = (lo + hi) * 0.5;
                float2 muv = ssr_project(mid, p.tan_half_fov_y, p.aspect);
                float  sd  = gbuffer.sample(smp, muv).a;
                if (sd > 0.0 && (-mid.z) - sd > 0.0) hi = mid; else lo = mid;
            }
            hit_uv = ssr_project(hi, p.tan_half_fov_y, p.aspect);
            hit = true;
            steps_taken = i;
            break;
        }
    }

    // The reflected colour: a screen-space hit gathered over a roughness-scaled
    // disk, or the environment cube when the ray missed. A hit near the screen
    // border or at the end of its march fades toward the environment rather
    // than snapping flat to the base shading.
    float  blur_radius = (roughness / SSR_ROUGH_CUT) * SSR_BLUR_MAX;
    float3 reflected;
    if (hit) {
        float3 hit_color = ssr_gather(scene, smp, hit_uv, blur_radius);
        float2 e = smoothstep(0.0, SSR_EDGE_FADE, hit_uv)
                 * smoothstep(0.0, SSR_EDGE_FADE, 1.0 - hit_uv);
        float edge = e.x * e.y;
        float march = float(steps_taken) / float(SSR_MAX_STEPS);
        float dist_fade = 1.0 - smoothstep(0.7, 1.0, march);
        reflected = mix(env, hit_color, edge * dist_fade);
    } else {
        reflected = env;
    }

    float ndv     = saturate(dot(N, V));
    float fresnel = SSR_F0 + (1.0 - SSR_F0) * pow(1.0 - ndv, 5.0);
    float w = saturate(fresnel * gloss * p.intensity);
    return float4(mix(base, reflected, w), 1.0);
}
