#include <metal_stdlib>
using namespace metal;

// --- Glass panel pass ---
//
// The simplest consumer of the engine's transparent pass: a flat, fixed
// rectangular pane. Runs in the same `PassId::Transparent` slot as water,
// after SSR resolve and before TAA. The build-time quad is already in world
// space (see geometry::glass_quad), so the vertex shader only projects it.
//
// The fragment shader:
//   - Discards where nearer opaque geometry occludes the pane (manual depth
//     test against the resolved main depth - the transparent pass binds no
//     depth attachment).
//   - Refracts: samples the pre-transparent scene snapshot at a screen offset
//     perturbed by the (view-facing) pane normal, then tints it.
//   - Reflects: when the planar reflection is active (the scene re-rendered
//     mirrored across this pane's plane, resolve at texture(11)) samples it at
//     the fragment's own screen UV for a sharp scene-correct mirror; otherwise
//     samples the box-projected reflection-probe set (the local scene capture)
//     along the mirror direction, falling back to the sky prefilter cube. Then
//     mixes reflection over refraction by a Schlick Fresnel term (~4% head-on,
//     full mirror at grazing).
//
// Output is straight-alpha blended (SRC_ALPHA / ONE_MINUS_SRC_ALPHA) by the
// pipeline. Shares `TransparentView` (buffer 5) with every other transparent
// draw; `GlassParams` (buffer 6) is per-pane. The reflection sources are bound
// once by `encode_transparent`: the sky prefilter cube at texture(2), the
// reflection-probe cubes at texture(3..3+MAX_PROBES), the cube sampler at
// sampler(1), and the probe set at buffer(7).

struct TransparentView {
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    float    _pad;
};

struct GlassParams {
    // Vec3 fields stored as float4 (.w unused) so the Rust [f32; 4] layout is
    // byte-identical regardless of MSL float3 packing.
    float4 centre;  // world-space pane centre
    float4 normal;  // unit pane normal (facing direction)
    float4 tint;    // colour multiplied into the refracted scene
    float  opacity;
    float  refraction_strength;
    float  fresnel_power;
    float  prefilter_mip_count; // mips in the sky prefilter cube; 0 = none bound
    // Planar reflection control: x = strength (>0.5 selects the planar reflection
    // resolve at texture(11) over the probe/sky cube). A float4 so the struct
    // stays 16-byte aligned and matches the CPU-side metal::uniforms::GlassParams.
    float4 planar;
};

// Reflection-probe set, bound at buffer(7) + texture(3..3+MAX_PROBES). Mirrors
// `ProbeSet` / `ProbeUniforms` in default.metal / rt_reflections.metal (and
// metal::uniforms). Lets glass sample the LOCAL box-projected scene capture
// instead of only the foreign sky cube - the same source the forward IBL
// specular term and the RT-miss fallback use. `count` is 0 in worlds with no
// baked probe, where the surface keeps the sky prefilter.
constant constexpr uint  MAX_PROBES         = 8u;
constant constexpr float PROBE_BLEND_MARGIN = 0.2;

struct ProbeUniforms {
    float4 box_min;   // xyz = influence-box min, w = enabled
    float4 box_max;   // xyz = influence-box max
    float4 probe_pos; // xyz = capture position
};

struct ProbeSet {
    uint count;
    // Three SCALAR uints, NOT a uint3 (which would be 16-byte aligned and push
    // `probes` to offset 32 / struct 416), so `probes` stays at offset 16 and
    // the struct is 400 bytes, matching the CPU-side metal::uniforms::ProbeSet.
    uint _pad0;
    uint _pad1;
    uint _pad2;
    ProbeUniforms probes[MAX_PROBES];
};
static_assert(sizeof(ProbeSet) == 400,
              "ProbeSet must be 400 bytes to match the CPU-side metal::uniforms::ProbeSet");

// Box-parallax sample of one probe cube: intersect the world-space reflection
// ray with the probe's influence box and re-anchor the sample at that hit
// relative to the capture point, so a static cube tracks the camera. Mirrors
// default.metal / rt_reflections.metal `sample_probe_radiance`.
static float3 sample_probe_radiance(
    texturecube<float>      probe_cube,
    constant ProbeUniforms &probe,
    float3                  world_pos,
    float3                  R,
    float                   lod,
    sampler                 cube_sampler)
{
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
// every probe whose influence box covers `world_pos` (partition of unity) and
// returns the weight-normalised sum of their box-projected samples; falls back
// to the nearest probe by capture distance where no box covers. Mirrors
// default.metal / rt_reflections.metal `probe_set_specular`.
static float3 probe_set_specular(
    constant ProbeSet                    &set,
    array<texturecube<float>, MAX_PROBES> probes,
    float3                                world_pos,
    float3                                R,
    float                                 lod,
    sampler                               cube_sampler)
{
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

struct GlassVtxIn {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct GlassVtxOut {
    float4 position [[position]];
    float3 world_pos;
};

vertex GlassVtxOut glass_vertex(
    GlassVtxIn            in [[stage_in]],
    constant TransparentView &v [[buffer(5)]],
    constant GlassParams  &p [[buffer(6)]])
{
    GlassVtxOut out;
    // Quad vertices are pre-transformed into world space at build time.
    out.world_pos = in.pos;
    out.position = v.vp * float4(in.pos, 1.0);
    return out;
}

fragment float4 glass_fragment(
    GlassVtxOut                       in            [[stage_in]],
    constant TransparentView         &v             [[buffer(5)]],
    constant GlassParams             &p             [[buffer(6)]],
    constant ProbeSet                &probe_set     [[buffer(7)]],
    texture2d<float, access::sample>  scene_color   [[texture(0)]],
    depth2d<float>                    scene_depth   [[texture(1)]],
    texturecube<float, access::sample> prefilter    [[texture(2)]],
    array<texturecube<float>, MAX_PROBES> probes    [[texture(3)]],
    texture2d<float, access::sample>  planar_reflection [[texture(11)]],
    sampler                           scene_sampler [[sampler(0)]],
    sampler                           cube_sampler  [[sampler(1)]])
{
    float3 view_dir = normalize(v.camera_pos.xyz - in.world_pos);
    // Two-sided: orient the normal toward the viewer so a pane lit from
    // behind still Fresnels correctly.
    float3 normal = normalize(p.normal.xyz);
    if (dot(normal, view_dir) < 0.0) {
        normal = -normal;
    }

    float2 viewport = max(v.viewport, float2(1.0));
    float2 frag_uv = float2(in.position.x / viewport.x,
                            in.position.y / viewport.y);

    // Manual depth occlusion: discard where the resolved scene depth at this
    // pixel is nearer than the pane (the pass has no hardware depth test).
    uint2 self_pixel = min(uint2(in.position.xy), uint2(viewport) - uint2(1));
    float scene_self_depth01 = scene_depth.read(self_pixel);
    if (scene_self_depth01 < in.position.z) {
        discard_fragment();
    }

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    float2 refract_uv = clamp(frag_uv + normal.xy * p.refraction_strength,
                              float2(0.001), float2(0.999));
    float3 refracted = scene_color.sample(scene_sampler, refract_uv).rgb * p.tint.rgb;

    // Reflection along the mirror direction. When the planar reflection is active
    // (`planar.x > 0.5`: the scene re-rendered mirrored across this pane's plane)
    // sample it projectively at this fragment's own screen UV - a flat pane is a
    // perfect mirror, so the mirrored render lands exactly under the reflector and
    // needs no distortion. Otherwise fall back to the local box-projected probe
    // set (the scene capture) when the world has baked probes, else the sky
    // prefilter cube, else a white rim so a probe-less, env-less world still reads
    // as glass. A pane is smooth, so every path is sharp (mip 0).
    float3 R = reflect(-view_dir, normal);
    float3 reflection;
    if (p.planar.x > 0.5) {
        reflection = planar_reflection.sample(scene_sampler, frag_uv).rgb;
    } else if (probe_set.count > 0u) {
        reflection = probe_set_specular(probe_set, probes, in.world_pos, R, 0.0, cube_sampler);
    } else if (p.prefilter_mip_count > 0.5) {
        reflection = prefilter.sample(cube_sampler, R, level(0.0)).rgb;
    } else {
        reflection = float3(1.0);
    }

    // Schlick Fresnel (F0 = 0.04 dielectric) drives the reflection/refraction
    // blend: ~4% head-on, rising to a full mirror at grazing. `fresnel_power`
    // stays the author's grazing-rim shaping control for the opacity ramp.
    float n_dot_v = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - n_dot_v, max(p.fresnel_power, 1e-3));
    float refl_weight = saturate(0.04 + 0.96 * rim);
    float3 colour = mix(refracted, reflection, refl_weight);
    float alpha = saturate(mix(p.opacity, 1.0, rim));

    return float4(colour, alpha);
}
