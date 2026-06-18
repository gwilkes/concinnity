// src/metal/shaders/raymarch_helpers.metal
//
// Engine-shipped header for raymarched SDF volumes. Prepended to the
// user's fragment shader at compile time; provides the type layouts the
// raymarch pass binds, the Inigo Quilez SDF primitive library, the cone-
// stepping marcher, and the PBR helpers user shaders call from `shade`.

#include <metal_stdlib>
using namespace metal;

// Uniform layouts - MUST stay in sync with concinnity-client/src/metal/raymarch.rs
// (Rust-side `repr(C)` structs).

struct RaymarchView {
    float4x4 vp;
    float4x4 inv_vp;
    packed_float3 cam_pos;
    float _pad0;
    float2 viewport;
    float time;
    /// Mip count of the bound IBL prefilter cube. 0 = "no
    /// EnvironmentMap bound" → IBL helpers fall back to the hand-tuned
    /// hemispheric ambient.
    float prefilter_mip_count;
};

// Fixed-size parameter block the user shader interprets. 32 scalar
// floats - access via `params.vals[i]` (i in 0..32). 4-byte aligned in
// MSL, byte-identical to Rust `[f32; 32]`.
struct SdfParams {
    float vals[32];
};

// Per-point sample a volumetric SdfVolume's `sampleVolume` returns. The
// volumetric template integrates these front-to-back (Beer-Lambert):
//   * `density`    - extinction coefficient at the point (>= 0).
//   * `scattering` - single-scatter albedo, multiplied by sun radiance.
//   * `emission`   - self-emitted radiance added regardless of lighting.
struct VolumeSample {
    float density;
    float3 scattering;
    float3 emission;
};

// Per-volume uniforms. Matches RaymarchVolumeUniforms in raymarch.rs.
struct SdfVolumeUniforms {
    packed_float3 centre;
    float _pad0;
    packed_float3 extent;
    float _pad1;
    float cone_ratio;
    float max_distance;
    int   max_steps;
    int   receive_shadows;
    SdfParams params;
};

struct DirLight {
    packed_float3 direction;
    float intensity;
    packed_float3 color;
    float _pad;
};

struct PointLight {
    packed_float3 position;
    float range;
    packed_float3 color;
    float intensity;
};

// Matches LightUniforms in src/gfx/render_types.rs. The same layout the
// Main pass uses, so raymarched surfaces light up identically to
// rasterised geometry under the same scene lights.
struct RaymarchLights {
    DirLight directional[4];
    PointLight point[8];
    int num_directional;
    int num_point;
    float _pad0;
    float _pad1;
};

// Cascaded shadow map view-projections + per-cascade splits. Matches
// `ShadowUniforms` in `src/gfx/render_types.rs` - same layout the Main
// pass binds at fragment buffer(5), so the raymarch helpers can run
// the exact same cascade selection + PCF kernel the rasterised path
// uses.
constant constexpr int RAYMARCH_NUM_SHADOW_CASCADES = 4;
struct RaymarchShadowUniforms {
    float4x4 light_vps[RAYMARCH_NUM_SHADOW_CASCADES];
    float    cascade_splits[RAYMARCH_NUM_SHADOW_CASCADES];
};

// Per-point material the user's `shade` returns. The template runs PBR
// on this so the look stays consistent with the Main pass. `transmitted`
// is an additive contribution the template adds after the PBR sun + IBL
// terms - opaque user shaders leave it zero; refractive shaders (water,
// glass) set it to the scene-tap colour they want to show through. The
// engine doesn't auto-attenuate it by Fresnel; the user shader is
// expected to bake any Fresnel / extinction / opacity blend into the
// value before returning.
struct SdfSurface {
    float3 albedo;
    float roughness;
    float metallic;
    float3 emissive;
    float3 transmitted;
};

// IQ primitive library - https://iquilezles.org/articles/distfunctions/
// Keep the set small and well-known; users compose them inside `map`.

inline float sdSphere(float3 p, float r) {
    return length(p) - r;
}

inline float sdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

inline float sdRoundBox(float3 p, float3 b, float r) {
    float3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

inline float sdTorus(float3 p, float2 t) {
    float2 q = float2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

inline float sdCapsule(float3 p, float3 a, float3 b, float r) {
    float3 pa = p - a;
    float3 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

inline float sdPlane(float3 p, float3 n, float h) {
    return dot(p, n) + h;
}

inline float opSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

inline float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / max(k, 1e-6), 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

inline float opSmoothIntersection(float a, float b, float k) {
    float h = clamp(0.5 - 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

// User-provided functions - forward declarations. The user's shader
// (sandwiched between this header and the template) MUST define both.

float map(float3 p, constant SdfParams& params, float time);
// `frag_uv` is the [0, 1] screen-space UV of the current pixel (Metal's
// y-down convention - matches `scene_color.sample` directly).
// `scene_color` is the pre-raymarch HDR scene snapshot the template
// blitted at the start of the pass (RGBA16F, linear-light); user
// shaders that don't refract can ignore it. `scene_samp` is a linear-
// clamp filter sampler.
SdfSurface shade(float3 p, float3 normal,
                 constant SdfParams& params, float time,
                 float2 frag_uv,
                 texture2d<float> scene_color,
                 sampler scene_samp);

// Engine helpers the template + user shaders can call.

// 4-tap central-difference gradient. Normalised. `eps` should be small
// enough that the linearisation is accurate but big enough that the SDF
// doesn't return zero on both sides (~ 0.001 in world units works well
// for the IQ primitives).
inline float3 sdfNormal(float3 p, constant SdfParams& params,
                        float time, float eps) {
    float3 ex = float3(eps, 0.0, 0.0);
    float3 ey = float3(0.0, eps, 0.0);
    float3 ez = float3(0.0, 0.0, eps);
    return normalize(float3(
        map(p + ex, params, time) - map(p - ex, params, time),
        map(p + ey, params, time) - map(p - ey, params, time),
        map(p + ez, params, time) - map(p - ez, params, time)
    ));
}

struct RayHit {
    float t;
    bool hit;
    int steps;
};

// Cone-stepping sphere trace. Marches from `t_start` along `dir` until
// the SDF returns < `surface_eps`, or until `t` exceeds `t_max`, or
// until the per-volume step cap fires. `cone_ratio` is the Lipschitz
// reciprocal (≤ 1 for 1-Lipschitz SDFs; the IQ library is 1-Lipschitz).
inline RayHit coneRaymarch(float3 origin, float3 dir,
                            float t_start, float t_max,
                            constant SdfVolumeUniforms& vol,
                            float time) {
    RayHit r;
    r.t = t_start;
    r.hit = false;
    r.steps = 0;
    float t = t_start;
    int cap = min(vol.max_steps, 256);
    float ratio = max(vol.cone_ratio, 0.01);
    const float surface_eps = 0.001;
    for (int i = 0; i < cap; ++i) {
        if (t >= t_max) break;
        float3 p = origin + dir * t;
        float d = map(p, vol.params, time);
        r.steps = i + 1;
        if (abs(d) < surface_eps) {
            r.t = t;
            r.hit = true;
            return r;
        }
        t += max(abs(d) * ratio, 0.001);
    }
    return r;
}

// Sample the pre-raymarch HDR scene with a normal-perturbed screen UV
// for refraction. The perturbation is the world-space normal's XZ tilt
// scaled by `strength` (typical 0.02–0.10 for a water surface);
// stronger values bend the screen sample more aggressively. The
// returned colour is linear-light RGB - combine with the per-channel
// `rwWaterExtinction`-style attenuation in the user shader before
// writing it into `SdfSurface.transmitted`.
inline float3 sampleSceneRefracted(float2 frag_uv, float3 normal,
                                   float strength,
                                   texture2d<float> scene_color,
                                   sampler scene_samp) {
    float2 refract_uv = clamp(frag_uv + normal.xz * strength, 0.0, 1.0);
    return scene_color.sample(scene_samp, refract_uv).rgb;
}

// Slab ray-box intersection. Returns `(t_enter, t_exit)`. When the box
// is missed entirely, `t_exit < max(0, t_enter)`.
inline float2 rayBox(float3 ro, float3 rd, float3 box_min, float3 box_max) {
    float3 inv = 1.0 / rd;
    float3 t0 = (box_min - ro) * inv;
    float3 t1 = (box_max - ro) * inv;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return float2(t_enter, t_exit);
}

// PBR sun helper. Cook-Torrance GGX + Smith G + Schlick F, matching the
// Main pass's math so raymarched + rasterised geometry agree under the
// same lights. `shadow` is in [0, 1] (1 = fully lit).
inline float3 shadePbrSun(SdfSurface s, float3 normal, float3 viewDir,
                           DirLight sun, float shadow) {
    float3 L = normalize(float3(sun.direction));
    float3 H = normalize(viewDir + L);
    float NdotL = max(0.0, dot(normal, L));
    float NdotV = max(1e-3, dot(normal, viewDir));
    float NdotH = max(0.0, dot(normal, H));
    float VdotH = max(0.0, dot(viewDir, H));

    float a = max(s.roughness * s.roughness, 1e-3);
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom);

    float k = (s.roughness + 1.0) * (s.roughness + 1.0) / 8.0;
    float G = (NdotL / (NdotL * (1.0 - k) + k))
            * (NdotV / (NdotV * (1.0 - k) + k));

    float3 F0 = mix(float3(0.04), s.albedo, s.metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    float3 spec = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-3);
    float3 diff = (1.0 - F) * (1.0 - s.metallic) * s.albedo / 3.14159265;
    float3 light = float3(sun.color) * sun.intensity * shadow;
    return (diff + spec) * light * NdotL;
}

// Hand-tuned hemispheric ambient fallback. Used when no
// `EnvironmentMap` is bound (`view.prefilter_mip_count == 0`); the
// IBL helper below is what runs on a real world.
inline float3 shadeAmbient(SdfSurface s, float3 normal) {
    float3 sky = float3(0.45, 0.52, 0.62);
    float3 ground = float3(0.07, 0.06, 0.05);
    float t = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
    float3 hemi = mix(ground, sky, t);
    return s.albedo * hemi * 0.35 + s.emissive;
}

// CSM cascade-shadow PCF - mirrors `shadow_factor_cascaded` in
// `src/build/shaders/default.metal`. Identical math so raymarched
// surfaces receive shadows that match rasterised geometry exactly.

inline float raymarchHashRotation(float2 p) {
    float h = fract(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

inline float sampleSunShadow(
    float3 world_pos,
    float view_depth,
    float2 screen_xy,
    constant RaymarchShadowUniforms& shadow,
    depth2d_array<float> shadow_map,
    sampler shadow_samp
) {
    // Cascade selection: smallest index whose far split exceeds this
    // fragment's view-space depth. Beyond the last cascade = fully lit.
    int cascade = RAYMARCH_NUM_SHADOW_CASCADES;
    if (view_depth < shadow.cascade_splits[0])      cascade = 0;
    else if (view_depth < shadow.cascade_splits[1]) cascade = 1;
    else if (view_depth < shadow.cascade_splits[2]) cascade = 2;
    else if (view_depth < shadow.cascade_splits[3]) cascade = 3;
    if (cascade >= RAYMARCH_NUM_SHADOW_CASCADES) return 1.0;

    float4 light_clip = shadow.light_vps[cascade] * float4(world_pos, 1.0);
    float3 ndc = light_clip.xyz / max(light_clip.w, 1e-6);
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if (any(uv < 0.0f) || any(uv > 1.0f) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    // Per-cascade depth bias proportional to texel size (texel grows
    // for distant cascades, so bias scales accordingly). Matches the
    // Main pass's bias schedule.
    float bias = 0.0015 * (1.0 + float(cascade) * 0.7);
    float ref = ndc.z - bias;

    // Per-pixel rotation breaks the 5×5 PCF kernel's banding.
    float angle = raymarchHashRotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    float2 tex_size = float2(1.0) /
        float2(shadow_map.get_width(), shadow_map.get_height());

    float sum = 0.0;
    constexpr int RADIUS = 2;
    constexpr float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            float2 off = float2(dx, dy);
            float2 rot = float2(off.x * ca - off.y * sa,
                                off.x * sa + off.y * ca);
            float2 sample_uv = uv + rot * tex_size;
            sum += shadow_map.sample_compare(shadow_samp, sample_uv,
                                             cascade, ref);
        }
    }
    return sum / SAMPLES;
}

// IBL - mirrors the ambient term in `default.metal::shade_surface`.
// Karis split-sum approximation for specular reflection; irradiance cube
// for diffuse. The same `env_brdf_approx` fit + Fresnel Schlick the Main
// pass uses, so raymarched surfaces light up identically under the same
// EnvironmentMap.

inline float2 raymarchEnvBrdfApprox(float NdV, float rough) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4( 1.0,  0.0425,  1.040, -0.040);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

inline float3 raymarchFresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) *
        pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Real PBR IBL ambient. Samples the engine's irradiance cube for
// diffuse + the prefilter cube (mip = roughness × mip_count) for
// specular, combined via the Karis split-sum. Falls back to the
// hand-tuned hemispheric ambient when `prefilter_mip_count <= 0` (no
// EnvironmentMap bound) - the helper's call site checks the gate.
inline float3 shadeAmbientIbl(
    SdfSurface s,
    float3 normal,
    float3 view_dir,
    float prefilter_mip_count,
    texturecube<float> irradiance_cube,
    texturecube<float> prefilter_cube,
    sampler cube_samp
) {
    if (prefilter_mip_count <= 0.5) {
        return shadeAmbient(s, normal);
    }
    float NdV = max(dot(normal, view_dir), 0.0);
    float3 F0 = mix(float3(0.04), s.albedo, s.metallic);
    float3 F_ibl = raymarchFresnelSchlick(NdV, F0);
    float3 kd_ibl = (1.0 - F_ibl) * (1.0 - s.metallic);

    float3 irradiance = irradiance_cube.sample(cube_samp, normal).rgb;
    float3 diffuse_ibl = kd_ibl * s.albedo * irradiance / 3.14159265;

    float3 R = reflect(-view_dir, normal);
    float lod = s.roughness * (prefilter_mip_count - 1.0);
    float3 prefiltered =
        prefilter_cube.sample(cube_samp, R, level(lod)).rgb;
    float2 ab = raymarchEnvBrdfApprox(NdV, s.roughness);
    float3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

    return diffuse_ibl + specular_ibl + s.emissive;
}
