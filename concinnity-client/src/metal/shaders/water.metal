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

fragment float4 water_fragment(
    WaterVtxOut                       in           [[stage_in]],
    constant WaterView               &v            [[buffer(5)]],
    constant WaterParams             &p            [[buffer(6)]],
    texture2d<float, access::sample>  scene_color  [[texture(0)]],
    depth2d<float>                    scene_depth  [[texture(1)]],
    texturecube<float, access::sample> prefilter   [[texture(2)]],
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

    // Reflection: sample the IBL prefilter cube at the reflected view dir.
    // When no environment map is bound, prefilter_mip_count is 0 and we
    // fall back to a vertical sky-tint gradient so the surface still has
    // some reflective signal instead of going dark at the horizon.
    float3 r = reflect(-view_dir, normal);
    float3 reflected;
    if (p.prefilter_mip_count > 0.5) {
        float mip = clamp(p.roughness, 0.0, 1.0)
                  * (p.prefilter_mip_count - 1.0);
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
