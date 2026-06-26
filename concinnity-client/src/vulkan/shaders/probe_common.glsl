// probe_common.glsl
//
// Shared reflection-probe sampling for the Vulkan forward shader (later: the SSR
// / RT / glass paths). Mirrors directx/shaders/probe_common.hlsl + default.metal's
// sample_probe_radiance + probe_set_specular (reflection_probes.md sections 4 + 6):
// a box-parallax cube sample blended across every probe whose influence box covers
// the surface (partition of unity), falling back to the nearest probe by capture
// distance when the surface lies outside every box.
//
// shaderc has no #include resolver, so this file is substituted into the consuming
// shader at its PROBE_COMMON marker; the MAX_PROBES token is replaced with the bind
// count (probe_uniforms::MAX_PROBES), and PROBE_DESC_SET with the descriptor-set
// index the global set (which owns bindings 7 + 8) is bound at in that shader -- 0
// for the forward bindless pass (the global set IS set 0 there), or a higher index
// for the SSR / RT passes that bind the global set as an extra set. The consumer
// must NOT redeclare probe_set / probe_cubes. The ProbeBlock UBO matches
// probe_uniforms::ProbeSet byte-for-byte: `count` is padded with THREE scalar uints
// (NOT a uvec3, which std140 would 16-byte-align, pushing probes to offset 32 and
// silently disabling box parallax).

const float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms {
    vec4 box_min;
    vec4 box_max;
    vec4 probe_pos;
};

layout(std140, set = {PROBE_DESC_SET}, binding = 7) uniform ProbeBlock {
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[{MAX_PROBES}];
} probe_set;

layout(set = {PROBE_DESC_SET}, binding = 8) uniform samplerCube probe_cubes[{MAX_PROBES}];

// Box-parallax sample of probe cube `i`: intersect the reflection ray `R` with the
// probe's influence box and re-anchor the sample direction at that hit relative to
// the capture point, so a static captured cube tracks a moving camera. Falls back
// to the raw ray when the probe has no baked box (box_min.w <= 0.5) or the box does
// not lie ahead of the ray. `lod` is passed as the texture bias (the reflection
// vector's screen-space footprint widens the mip at grazing / distant angles), the
// same SampleBias semantics the prefilter-cube tap uses.
vec3 sample_probe_radiance(uint i, ProbeUniforms probe, vec3 world_pos, vec3 R, float lod) {
    vec3 sample_dir = R;
    if (probe.box_min.w > 0.5) {
        vec3 inv_r = 1.0 / R;
        vec3 t_max = (probe.box_max.xyz - world_pos) * inv_r;
        vec3 t_min = (probe.box_min.xyz - world_pos) * inv_r;
        vec3 t_far = max(t_max, t_min);
        float dist = min(min(t_far.x, t_far.y), t_far.z);
        if (dist > 0.0) {
            vec3 hit = world_pos + R * dist;
            sample_dir = hit - probe.probe_pos.xyz;
        }
    }
    return texture(probe_cubes[i], sample_dir, lod).rgb;
}

// Probe radiance for `world_pos` along world-space ray `R`, blended across every
// probe covering the point (partition of unity). Each probe gets a smoothstep
// weight from its signed box distance (1 deep inside, 0.5 on the surface, 0 a
// margin outside); the result is the weight-normalised sum of each probe's
// box-projected sample. Where no box covers, falls back to the nearest probe by
// capture distance.
vec3 probe_set_specular(vec3 world_pos, vec3 R, float lod) {
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    float near_d = 1e30;
    uint near_i = 0u;
    for (uint i = 0u; i < probe_set.count; i++) {
        vec3 c = 0.5 * (probe_set.probes[i].box_min.xyz + probe_set.probes[i].box_max.xyz);
        vec3 he = 0.5 * (probe_set.probes[i].box_max.xyz - probe_set.probes[i].box_min.xyz);
        // Signed distance to the box surface: positive inside, negative outside.
        vec3 q = abs(world_pos - c) - he;
        float sd = -(length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0));
        float margin = max(PROBE_BLEND_MARGIN * min(he.x, min(he.y, he.z)), 1e-4);
        float w = smoothstep(-margin, margin, sd);
        if (w > 0.0) {
            acc += w * sample_probe_radiance(i, probe_set.probes[i], world_pos, R, lod);
            wsum += w;
        }
        float d = distance(world_pos, probe_set.probes[i].probe_pos.xyz);
        if (d < near_d) {
            near_d = d;
            near_i = i;
        }
    }
    if (wsum > 0.0) {
        return acc / wsum;
    }
    return sample_probe_radiance(near_i, probe_set.probes[near_i], world_pos, R, lod);
}
