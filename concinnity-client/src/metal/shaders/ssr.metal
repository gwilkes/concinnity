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
    // Camera-to-world transform (the rigid inverse of the view matrix): its 3x3
    // turns the view-space reflection ray into the world-space direction the
    // cubemap is sampled with, and its translation column lets the resolve rebuild
    // the world-space surface position the reflection probe box-projects against.
    float4x4 inv_view;
};

// Reflection-probe set, bound at buffer(1). Layout mirrors `ProbeSet` /
// `ProbeUniforms` in default.metal (and metal::uniforms): a missed reflection ray
// falls back to the local scene-captured probe (box-projected) instead of the
// foreign sky cube, the same source the forward IBL specular term uses. `count`
// is 0 in worlds with no baked probe, where the resolve keeps the sky fallback.
constant constexpr uint  MAX_PROBES         = 8u;
constant constexpr float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms {
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

struct ProbeSet {
    uint count;
    // Three scalar uints, NOT a uint3: a uint3 is 16-byte aligned in MSL, which
    // pushes `probes` to offset 32 (struct 416 bytes) and mismatches the CPU-side
    // metal::uniforms::ProbeSet, whose [u32; 3] keeps `probes` at offset 16 (struct
    // 400 bytes). The static_assert locks the 400-byte layout at shader-compile time.
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};
static_assert(sizeof(ProbeSet) == 400,
              "ProbeSet must be 400 bytes to match the CPU-side metal::uniforms::ProbeSet");

// Box-parallax sample of one probe cube: intersect the world-space reflection ray
// with the probe's influence box and re-anchor the sample at that hit relative to
// the capture point, so a static cube tracks the camera. Mirrors default.metal's
// `sample_probe_radiance`; the `dist > 0` guard keeps a blended secondary box the
// surface has already left from sampling backward.
static float3 sample_probe_radiance(
    texturecube<float>      probe_cube,
    constant ProbeUniforms &probe,
    float3                  world_pos,
    float3                  R,
    float                   lod,
    sampler                 cube_sampler
) {
    float3 sample_dir = R;
    if (probe.box_min.w > 0.5) {
        float3 inv_r = 1.0 / R;
        float3 t_max = (probe.box_max.xyz - world_pos) * inv_r;
        float3 t_min = (probe.box_min.xyz - world_pos) * inv_r;
        float3 t_far = max(t_max, t_min);
        float dist = min(min(t_far.x, t_far.y), t_far.z);
        if (dist > 0.0) {
            float3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    return probe_cube.sample(cube_sampler, sample_dir, bias(lod)).rgb;
}

// Reflection-probe radiance for `world_pos` along world-space ray `R`. Weights
// every probe whose influence box covers `world_pos` and returns the
// weight-normalised sum of their box-projected samples (partition of unity): each
// probe's weight is `smoothstep(-margin, margin, sd)` of its signed box distance, so
// overlapping boxes cross-fade smoothly (no pop at a 3-way overlap line) and a single
// covering box reduces to one sample. Falls back to the nearest by capture distance
// where no box covers. Matches default.metal's `probe_set_specular`.
static float3 probe_set_specular(
    constant ProbeSet                    &set,
    array<texturecube<float>, MAX_PROBES> probes,
    float3                                world_pos,
    float3                                R,
    float                                 lod,
    sampler                               cube_sampler
) {
    float3 acc = float3(0.0);
    float wsum = 0.0;
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint i = 0u; i < set.count; i++) {
        float3 c = 0.5 * (set.probes[i].box_min.xyz + set.probes[i].box_max.xyz);
        float3 he = 0.5 * (set.probes[i].box_max.xyz - set.probes[i].box_min.xyz);
        float3 q = abs(world_pos - c) - he;
        float sd = -(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0));
        float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
        float w = smoothstep(-margin, margin, sd);
        if (w > 0.0) {
            acc += w * sample_probe_radiance(
                           probes[i], set.probes[i], world_pos, R, lod, cube_sampler);
            wsum += w;
        }
        float d = distance(world_pos, set.probes[i].probe_pos.xyz);
        if (d < near_d) {
            near_d = d;
            near_i = i;
        }
    }
    if (wsum > 0.0) {
        return acc / wsum;
    }
    return sample_probe_radiance(
        probes[near_i], set.probes[near_i], world_pos, R, lod, cube_sampler);
}

constant int   SSR_MAX_STEPS = 48;
constant int   SSR_REFINE    = 5;
// Surfaces rougher than REFLECTION_ROUGHNESS_CUT get no SSR; glossiness ramps in
// below it. The cut is injected as a shared `constant` (see pipeline.rs /
// concinnity_core::gfx::ssr::REFLECTION_ROUGHNESS_CUT) so the SSR, RT, and
// composite passes can never disagree on it.
// Dielectric base reflectance (water, glass, polished stone) for the Fresnel.
constant float SSR_F0        = 0.04;
// UV margin over which a hit near the screen border fades out.
constant float SSR_EDGE_FADE = 0.12;

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

fragment float4 ssr_resolve_fragment(
    SsrVtxOut in                  [[stage_in]],
    constant SsrParams &p         [[buffer(0)]],
    texture2d<float>   scene      [[texture(0)]],
    texture2d<float>   gbuffer    [[texture(1)]],
    texture2d<float>   rough_tex  [[texture(2)]],
    texturecube<float> prefilter  [[texture(3)]],
    array<texturecube<float>, MAX_PROBES> probes [[texture(4)]],
    constant ProbeSet &probe_set  [[buffer(1)]],
    sampler smp                   [[sampler(0)]],
    sampler cube_smp              [[sampler(1)]]
) {
    float3 base  = scene.sample(smp, in.uv).rgb;
    float4 c     = gbuffer.sample(smp, in.uv);
    float  depth = c.a;
    if (depth <= 0.0) return float4(0.0);           // background / sky -> no reflection

    float roughness = rough_tex.sample(smp, in.uv).r;
    // Glossy surfaces reflect sharply; rough ones get nothing.
    float gloss = saturate((REFLECTION_ROUGHNESS_CUT - roughness) / REFLECTION_ROUGHNESS_CUT);
    if (gloss <= 0.0) return float4(0.0);           // too rough -> no reflection

    float3 N = normalize(c.xyz);
    float3 P = ssr_view_pos(in.uv, depth, p.tan_half_fov_y, p.aspect);
    float3 V = normalize(-P);                       // P in view space, camera at origin
    float3 R = reflect(-V, N);                      // reflected ray direction

    // Environment fallback for a missed (or screen-edge) ray, in the reflected
    // direction at a roughness-keyed mip so a rougher surface reflects a blurrier
    // environment. With a baked reflection probe this is the local scene capture,
    // box-projected from the surface so it reflects real surrounding geometry
    // rather than the foreign sky HDR (the same source the forward IBL specular
    // term uses); otherwise it is the IBL prefilter cube. With no EnvironmentMap
    // bound there is nothing to fall back to, so missed rays keep the base shading.
    bool   ibl = p.prefilter_mip_count > 0.5;
    float3 env = base;
    if (ibl) {
        float3 r_world = (p.inv_view * float4(R, 0.0)).xyz;
        float  lod     = roughness * (p.prefilter_mip_count - 1.0);
        if (probe_set.count > 0u) {
            float3 world_pos = (p.inv_view * float4(P, 1.0)).xyz;
            env = probe_set_specular(probe_set, probes, world_pos, r_world, lod, cube_smp);
        } else {
            env = prefilter.sample(cube_smp, r_world, level(lod)).rgb;
        }
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

    // The reflected colour: the screen-space hit (a single sharp tap - the
    // reflection composite blurs it by roughness), or the environment cube when
    // the ray missed. A hit near the screen border or at the end of its march
    // fades toward the environment rather than snapping flat to the base shading.
    float3 reflected;
    if (hit) {
        float3 hit_color = scene.sample(smp, hit_uv).rgb;
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
    // Reflected radiance (.rgb) + composite weight (.a). The reflection
    // composite blurs this by surface roughness and blends it over the scene.
    return float4(reflected, w);
}
