// probe_common.hlsl
//
// Shared reflection-probe sampling for the DirectX forward / SSR / RT / glass
// shaders. Mirrors `default.metal`'s `sample_probe_radiance` + `probe_set_specular`
// (reflection_probes.md sections 4 + 6): a box-parallax cube sample blended across
// every probe whose influence box covers the surface (partition of unity), falling
// back to the nearest probe when outside all boxes.
//
// Concatenated (not #included -- the DX HLSL path has no include handler) ahead of
// each consuming shader, which therefore must NOT redeclare `cube_sampler`,
// `probe_cubes`, or the `ProbeBlock` cbuffer. The registers default to t7 / b4 / s2
// (free in the bindless main shader); a consumer that needs different slots #defines
// PROBE_CUBES_REGISTER / PROBE_SET_REGISTER / PROBE_SAMPLER_REGISTER before this file.

#pragma pack_matrix(column_major)

#ifndef MAX_PROBES
#define MAX_PROBES 8
#endif
#ifndef PROBE_CUBES_REGISTER
#define PROBE_CUBES_REGISTER t7
#endif
#ifndef PROBE_SET_REGISTER
#define PROBE_SET_REGISTER b4
#endif
#ifndef PROBE_SAMPLER_REGISTER
#define PROBE_SAMPLER_REGISTER s2
#endif

// Cross-fade width (fraction of the smallest box half-extent) over which a probe's
// weight ramps from 0 (a margin outside its box) to 1 (on the box surface and in).
// Must match `metal::uniforms` / `default.metal` PROBE_BLEND_MARGIN.
static const float PROBE_BLEND_MARGIN = 0.2;

// One probe's parallax box (matches `directx::probe_uniforms::ProbeUniforms`,
// 48 bytes). `box_min.w` is the enabled flag (1 = box parallax on).
struct ProbeUniforms
{
    float4 box_min;
    float4 box_max;
    float4 probe_pos;
};

// The live probe set (matches `directx::probe_uniforms::ProbeSet`, 400 bytes). The
// three scalar pad uints (NOT a uint3) keep `probes` at byte offset 16; a uint3
// would push it to 32 and silently shift every probe by one float4, disabling box
// parallax (reflection_probes.md section 7).
struct ProbeSet
{
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};

SamplerState cube_sampler : register(PROBE_SAMPLER_REGISTER);
TextureCube  probe_cubes[MAX_PROBES] : register(PROBE_CUBES_REGISTER);
cbuffer ProbeBlock : register(PROBE_SET_REGISTER)
{
    ProbeSet probes;
};

// Box-parallax sample of one probe cube: intersect the reflection ray `R` with the
// probe's influence box and re-anchor the sample direction at that hit relative to
// the capture point, so a static cube tracks a moving camera. Falls back to the raw
// ray when the probe has no baked box (`box_min.w <= 0.5`) or the box does not lie
// ahead of the ray. `lod` selects the roughness mip via SampleBias (the reflection
// vector's screen-space footprint widens it at grazing / distant angles).
float3 sample_probe_radiance(TextureCube cube, ProbeUniforms probe, float3 world_pos,
                             float3 R, float lod)
{
    float3 sample_dir = R;
    if (probe.box_min.w > 0.5)
    {
        float3 inv_r = 1.0 / R;
        float3 t_max = (probe.box_max.xyz - world_pos) * inv_r;
        float3 t_min = (probe.box_min.xyz - world_pos) * inv_r;
        float3 t_far = max(t_max, t_min);
        float dist = min(min(t_far.x, t_far.y), t_far.z);
        if (dist > 0.0)
        {
            float3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    return cube.SampleBias(cube_sampler, sample_dir, lod).rgb;
}

// Probe radiance for `world_pos` along world-space ray `R`, blended across every
// probe covering the point (partition of unity). Each probe gets a smoothstep
// weight from its signed box distance (1 deep inside, 0.5 on the surface, 0 a margin
// outside); the result is the weight-normalised sum of each probe's box-projected
// sample. Where no box covers, falls back to the nearest probe by capture distance.
float3 probe_set_specular(ProbeSet set, float3 world_pos, float3 R, float lod)
{
    float3 acc = float3(0.0, 0.0, 0.0);
    float wsum = 0.0;
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint i = 0u; i < set.count; i++)
    {
        float3 c = 0.5 * (set.probes[i].box_min.xyz + set.probes[i].box_max.xyz);
        float3 he = 0.5 * (set.probes[i].box_max.xyz - set.probes[i].box_min.xyz);
        // Signed distance to the box surface: positive inside, negative outside.
        float3 q = abs(world_pos - c) - he;
        float sd = -(length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0));
        float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
        float w = smoothstep(-margin, margin, sd);
        if (w > 0.0)
        {
            acc += w * sample_probe_radiance(
                           probe_cubes[i], set.probes[i], world_pos, R, lod);
            wsum += w;
        }
        float d = distance(world_pos, set.probes[i].probe_pos.xyz);
        if (d < near_d)
        {
            near_d = d;
            near_i = i;
        }
    }
    if (wsum > 0.0)
    {
        return acc / wsum;
    }
    return sample_probe_radiance(
        probe_cubes[near_i], set.probes[near_i], world_pos, R, lod);
}
