#include <metal_stdlib>
using namespace metal;

// --- Water surface pass ---
//
// Runs as the engine's transparent pass: after SSR resolve, before TAA. For
// each WaterSurface the vertex shader displaces a flat tessellated XZ grid
// by a sum of Gerstner waves (analytic position + tangent + bitangent →
// world-space normal). The fragment shader composites:
//
//   - Refraction: sample the resolved opaque scene at a screen-space offset
//     perturbed by the surface normal's XZ → bend the seabed under waves.
//   - Depth-based tint: read main depth, derive water column thickness from
//     the difference, mix shallow→deep colour by `1 - exp(-depth / falloff)`.
//   - Foam: brighten the surface where the water column thickness falls
//     below `foam_width` (intersection lines + shoreline).
//   - Reflection: sample the IBL prefilter cubemap at the reflected view
//     direction, mipped by `roughness`.
//   - Fresnel: mix refraction + tint vs. reflection by the standard
//     `(1 - dot(N, V))^fresnel_power` curve.
//
// Output is straight-alpha blended into scene_pre_taa via the pipeline's
// (SrcAlpha, OneMinusSrcAlpha) blend state.

constant uint MAX_WATER_WAVES = 4;

struct WaterWave {
    // Packed [direction.x, direction.y, amplitude, wavelength]
    float4 dir_amp_wave;
    // Packed [speed, steepness, 0, 0]
    float4 speed_steep_pad;
};

struct WaterView {
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    float    _pad;
};

struct WaterParams {
    // Vec3 fields stored as float4 (with the .w slot unused) so the Rust
    // `[f32; 4]` array layout is byte-identical regardless of MSL's
    // `float3` packing rules - those vary by compiler version and have
    // already burned us once. `.xyz` / `.rgb` reads pick the meaningful
    // components.
    float4   centre;
    float4   deep_colour;
    float4   shallow_colour;
    // Scalar block - tightly packed, 8 floats × 4B = 32 bytes.
    float    depth_falloff;
    float    foam_width;
    float    foam_intensity;
    float    fresnel_power;
    float    roughness;
    float    refraction_strength;
    uint     wave_count;
    float    prefilter_mip_count;
    // Wave coefficients (16-byte aligned).
    WaterWave waves[MAX_WATER_WAVES];
    // Planar reflection control: x = strength (>0.5 selects the planar
    // reflection over the probe/sky cube), y = wave-normal distortion scale.
    float4   planar;
};

struct WaterVtxIn {
    float3 pos     [[attribute(0)]];
    float3 normal  [[attribute(1)]];
    float3 tangent [[attribute(2)]];
    float3 color   [[attribute(3)]];
    float2 uv      [[attribute(4)]];
};

struct WaterVtxOut {
    float4 position    [[position]];
    float3 world_pos;
    float3 world_normal;
};

// Sum up to MAX_WATER_WAVES Gerstner waves at a flat-rest XZ position. Each
// wave produces a horizontal pinch + vertical sinusoid; analytic derivatives
// against (x, z) give the world-space normal at the displaced point.
static float3 gerstner_displace(
    float2 rest_xz,
    constant WaterParams &p,
    float time,
    thread float3 &out_normal)
{
    float3 displaced = float3(rest_xz.x, 0.0, rest_xz.y);
    float3 binormal = float3(1.0, 0.0, 0.0); // ∂P/∂x
    float3 tangent  = float3(0.0, 0.0, 1.0); // ∂P/∂z

    for (uint i = 0; i < p.wave_count && i < MAX_WATER_WAVES; ++i) {
        float2 dir = normalize(float2(p.waves[i].dir_amp_wave.x,
                                       p.waves[i].dir_amp_wave.y));
        float amp = p.waves[i].dir_amp_wave.z;
        float wavelen = max(p.waves[i].dir_amp_wave.w, 1e-3);
        float speed = p.waves[i].speed_steep_pad.x;
        float steep = clamp(p.waves[i].speed_steep_pad.y, 0.0, 1.0);

        float k = 2.0 * M_PI_F / wavelen;
        float phase = k * (dir.x * rest_xz.x + dir.y * rest_xz.y) - speed * k * time;
        float c = cos(phase);
        float s = sin(phase);

        // Q controls horizontal pinch; steepness * amplitude per Wave-A-spec
        // is the natural choppy parameterisation but we'll clamp to avoid
        // self-intersection.
        float q = steep / (k * amp * max(float(p.wave_count), 1.0));

        displaced.x += q * amp * dir.x * c;
        displaced.z += q * amp * dir.y * c;
        displaced.y += amp * s;

        // Analytic partials of the Gerstner formulation. Reference: NVIDIA
        // "Effective Water Simulation from Physical Models" GPU Gems 1.
        float wa = k * amp;
        binormal.x += -q * dir.x * dir.x * wa * s;
        binormal.y +=  dir.x * wa * c;
        binormal.z += -q * dir.x * dir.y * wa * s;

        tangent.x += -q * dir.x * dir.y * wa * s;
        tangent.y +=  dir.y * wa * c;
        tangent.z += -q * dir.y * dir.y * wa * s;
    }

    out_normal = normalize(cross(tangent, binormal));
    return displaced;
}

vertex WaterVtxOut water_vertex(
    WaterVtxIn          in [[stage_in]],
    constant WaterView  &v [[buffer(5)]],
    constant WaterParams &p [[buffer(6)]])
{
    WaterVtxOut out;

    float2 rest_xz = float2(in.pos.x + p.centre.x, in.pos.z + p.centre.z);
    float3 n;
    float3 displaced = gerstner_displace(rest_xz, p, v.time, n);
    displaced.y += p.centre.y;

    out.world_pos = displaced;
    out.world_normal = n;
    out.position = v.vp * float4(displaced, 1.0);
    return out;
}

// Map a screen-space NDC point + sampled non-linear depth to view-space
// linear depth (positive = away from camera). Used to derive water column
// thickness from the main depth attachment.
static float view_linear_depth(float ndc_x, float ndc_y, float depth01,
                                constant WaterView &v)
{
    float4 clip = float4(ndc_x, ndc_y, depth01, 1.0);
    float4 world = v.inv_vp * clip;
    world /= world.w;
    return distance(world.xyz, v.camera_pos.xyz);
}

// Reflection-probe set, bound at buffer(7) + texture(3..3+MAX_PROBES). Mirrors
// `ProbeSet` / `ProbeUniforms` in default.metal / rt_reflections.metal (and
// metal::uniforms). Lets water sample the LOCAL box-projected scene capture (a
// pond reflects nearby geometry, not just the sky) - the same source the
// forward IBL specular term and the RT-miss fallback use. `count` is 0 in
// worlds with no baked probe, where the surface keeps the sky prefilter.
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

fragment float4 water_fragment(
    WaterVtxOut                       in           [[stage_in]],
    constant WaterView               &v            [[buffer(5)]],
    constant WaterParams             &p            [[buffer(6)]],
    constant ProbeSet                &probe_set    [[buffer(7)]],
    texture2d<float, access::sample>  scene_color  [[texture(0)]],
    depth2d<float>                    scene_depth  [[texture(1)]],
    texturecube<float, access::sample> prefilter   [[texture(2)]],
    array<texturecube<float>, MAX_PROBES> probes   [[texture(3)]],
    texture2d<float, access::sample>  planar_reflection [[texture(11)]],
    sampler                            scene_sampler [[sampler(0)]],
    sampler                            cube_sampler  [[sampler(1)]])
{
    float3 view_dir = normalize(v.camera_pos.xyz - in.world_pos);
    float3 normal = normalize(in.world_normal);

    // Screen-space UV of the current fragment.
    float2 viewport = max(v.viewport, float2(1.0));
    float2 frag_uv = float2(in.position.x / viewport.x,
                            in.position.y / viewport.y);

    // Manual depth occlusion: the water pass binds no depth attachment so
    // foreground geometry cannot occlude us via hardware depth test. Read
    // the main depth at this fragment's pixel and discard when the scene
    // sample is closer than the water surface itself.
    uint2 self_pixel = uint2(in.position.xy);
    self_pixel = min(self_pixel, uint2(viewport) - uint2(1));
    float scene_self_depth01 = scene_depth.read(self_pixel);
    float frag_depth01 = in.position.z;
    if (scene_self_depth01 < frag_depth01) {
        discard_fragment();
    }

    // Refraction sample: perturb the lookup by the surface normal's XZ.
    float2 refract_uv = clamp(frag_uv + normal.xz * p.refraction_strength,
                               float2(0.001), float2(0.999));
    float3 refracted = scene_color.sample(scene_sampler, refract_uv).rgb;

    // Sample main depth at the *refracted* pixel so the depth match the
    // pixel we just sampled - guards against the refraction sample bending
    // into a foreground edge that's actually closer than the water.
    uint2 ref_pixel = uint2(refract_uv * viewport);
    ref_pixel = min(ref_pixel, uint2(viewport) - uint2(1));
    float scene_depth01 = scene_depth.read(ref_pixel);

    // Linear distance from camera to the scene point under the water vs.
    // to the water surface itself.
    float2 ndc_xy = float2(frag_uv.x * 2.0 - 1.0,
                            -(frag_uv.y * 2.0 - 1.0));
    float scene_dist = view_linear_depth(ndc_xy.x, ndc_xy.y, scene_depth01, v);
    float water_dist = distance(in.world_pos, v.camera_pos.xyz);
    float water_depth = max(scene_dist - water_dist, 0.0);

    // Tint: linear blend up to the deep colour over `depth_falloff` metres.
    float depth_t = 1.0 - exp(-water_depth / max(p.depth_falloff, 1e-3));
    float3 tinted = mix(p.shallow_colour.rgb, p.deep_colour.rgb, depth_t);
    float3 absorbed = mix(refracted * p.shallow_colour.rgb, tinted, depth_t);

    // Foam at intersection edges: a soft band where the seabed is just
    // below the surface.
    float foam_t = saturate(1.0 - water_depth / max(p.foam_width, 1e-3));
    float foam = foam_t * foam_t * p.foam_intensity;
    absorbed = mix(absorbed, float3(1.0), foam);

    // Reflection along the reflected view dir. When the planar reflection is
    // active (`planar.x > 0.5`: the scene re-rendered mirrored across the water
    // plane) sample it projectively at this fragment's screen UV, perturbed by
    // the wave normal's XZ for ripple distortion - a sharp, scene-correct
    // reflection. Otherwise fall back to the local box-projected probe set (the
    // scene capture), else the IBL prefilter cube, else - with no environment map
    // - a vertical sky-tint gradient so the surface still has a reflective signal.
    float3 r = reflect(-view_dir, normal);
    float mip = clamp(p.roughness, 0.0, 1.0) * max(p.prefilter_mip_count - 1.0, 0.0);
    float3 reflected;
    if (p.planar.x > 0.5) {
        float2 planar_uv = clamp(frag_uv + normal.xz * p.planar.y,
                                 float2(0.001), float2(0.999));
        reflected = planar_reflection.sample(scene_sampler, planar_uv).rgb;
    } else if (probe_set.count > 0u) {
        reflected = probe_set_specular(probe_set, probes, in.world_pos, r, mip, cube_sampler);
    } else if (p.prefilter_mip_count > 0.5) {
        reflected = prefilter.sample(cube_sampler, r, level(mip)).rgb;
    } else {
        // Hand-tuned sky fallback: bluer overhead, paler at the horizon.
        float horizon = saturate(r.y * 0.5 + 0.5);
        reflected = mix(float3(0.55, 0.62, 0.7), float3(0.25, 0.45, 0.7), horizon);
    }

    // Fresnel mix. Schlick approximation with a tunable power; F0 of 0.02
    // is a sane water-vs-air value, the exponent shapes the falloff.
    float n_dot_v = saturate(dot(normal, view_dir));
    float fresnel = 0.02 + (1.0 - 0.02)
                  * pow(1.0 - n_dot_v, max(p.fresnel_power, 0.001));

    float3 colour = mix(absorbed, reflected, fresnel);

    // Straight-alpha blend; the pipeline handles SRC_ALPHA + ONE_MINUS_SRC_ALPHA.
    return float4(colour, 1.0);
}
