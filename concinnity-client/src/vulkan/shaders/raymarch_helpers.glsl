#version 450

// src/vulkan/shaders/raymarch_helpers.glsl
//
// Engine-shipped GLSL header for raymarched SDF volumes on Vulkan. Prepended
// to the user's fragment shader (then the template) at compile time; provides
// the descriptor-bound uniform layouts, the Inigo Quilez SDF primitive
// library, the cone-stepping marcher, and the PBR / IBL / shadow helpers user
// shaders call from `shade`. GLSL port of
// src/directx/shaders/raymarch_helpers.hlsl: the uniform byte layouts of
// RaymarchView / SdfVolumeUniforms / SdfSurface stay identical so the Rust
// repr(C) structs round-trip across all three backends.
//
// Bindings (must match vulkan/raymarch.rs):
//   set 0 b0  RaymarchView UBO        set 1 b0  SdfVolumeUniforms UBO
//   set 0 b1  RaymarchLights UBO      set 0 b2  RaymarchShadow UBO
//   set 0 b3  shadow_map (sampler2DArrayShadow)
//   set 0 b4  irradiance_cube (samplerCube)
//   set 0 b5  prefilter_cube  (samplerCube)
//   set 0 b6  scene_color     (sampler2D, pre-raymarch HDR snapshot)

layout(std140, set = 0, binding = 0) uniform RaymarchViewBlock {
    mat4 view_vp;
    mat4 view_inv_vp;
    vec3 view_cam_pos;
    float view_pad0;
    vec2 view_viewport;
    float view_time;
    float view_prefilter_mip_count;
} rmview;

struct SdfParams { vec4 vals[8]; };

// Fetch a single param slot (0..31) via the packed vec4 array.
float sdfParamValue(SdfParams p, uint i) { return p.vals[i / 4u][i % 4u]; }

layout(std140, set = 1, binding = 0) uniform SdfVolumeBlock {
    vec3 vol_centre;
    float vol_pad0;
    vec3 vol_extent;
    float vol_pad1;
    float vol_cone_ratio;
    float vol_max_distance;
    int vol_max_steps;
    int vol_receive_shadows;
    SdfParams vol_params;
} vol;

// std140 LightUniforms: two vec4s per light (matches the main pass + the Rust
// struct: direction[3]+intensity, color[3]+pad).
struct DirLight  { vec4 dir_i; vec4 col; };
struct PointLight { vec4 pos_r; vec4 col_i; };

layout(std140, set = 0, binding = 1) uniform RaymarchLightsBlock {
    DirLight dir[4];
    PointLight pt[8];
    int num_dir;
    int num_pt;
    float lpad0;
    float lpad1;
} lights;

// Number of CSM cascades -- must match NUM_SHADOW_CASCADES in gfx::render_types.
const int RAYMARCH_NUM_SHADOW_CASCADES = 4;

layout(std140, set = 0, binding = 2) uniform RaymarchShadowBlock {
    mat4 light_vps[4];
    vec4 cascade_splits;
} shadow_uni;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadow_map;
layout(set = 0, binding = 4) uniform samplerCube irradiance_cube;
layout(set = 0, binding = 5) uniform samplerCube prefilter_cube;
layout(set = 0, binding = 6) uniform sampler2D scene_color;

// Per-point material the user's `shade` returns. `transmitted` is an additive
// contribution the template adds after PBR -- opaque shaders leave it zero,
// refractive shaders set it to the scene-tap colour they want to show through.
struct SdfSurface {
    vec3 albedo;
    float roughness;
    float metallic;
    vec3 emissive;
    vec3 transmitted;
};

// IQ primitive library -- https://iquilezles.org/articles/distfunctions/

float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float sdPlane(vec3 p, vec3 n, float h) { return dot(p, n) + h; }

float opSmoothUnion(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / max(k, 1e-6), 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float opSmoothIntersection(float a, float b, float k) {
    float h = clamp(0.5 - 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
    return mix(b, a, h) + k * h * (1.0 - h);
}

// User-provided functions -- forward declarations. The user shader (sandwiched
// between this header and the template) defines the functions matching its mode:
//   surface volumes (default) -> `map` + `shade`
//   volumetric volumes        -> `sampleVolume`
// glslang prunes the unreachable forward-declared functions (and the engine
// helpers that call them) from each entry point, so a volumetric author doesn't
// need stub `map` / `shade` and a surface author doesn't need a stub
// `sampleVolume`. Mirrors the FXC / Metal DCE the DirectX + Metal helpers rely
// on.
float map(vec3 p, SdfParams params, float time);
SdfSurface shade(vec3 p, vec3 normal, SdfParams params, float time, vec2 frag_uv);

// Per-point participating-media sample a volumetric user shader returns. The
// volumetric template integrates these front-to-back (Beer-Lambert):
//   * `density`    -- extinction coefficient at the point (>= 0); 0 = empty.
//   * `scattering` -- single-scatter albedo, multiplied by sun radiance.
//   * `emission`   -- self-emitted radiance added regardless of lighting.
// Layout matches the Metal / HLSL `VolumeSample`.
struct VolumeSample {
    float density;
    vec3 scattering;
    vec3 emission;
};

VolumeSample sampleVolume(vec3 p, SdfParams params, float time);

// 4-tap central-difference gradient. Normalised.
vec3 sdfNormal(vec3 p, SdfParams params, float time, float eps) {
    vec3 ex = vec3(eps, 0.0, 0.0);
    vec3 ey = vec3(0.0, eps, 0.0);
    vec3 ez = vec3(0.0, 0.0, eps);
    return normalize(vec3(
        map(p + ex, params, time) - map(p - ex, params, time),
        map(p + ey, params, time) - map(p - ey, params, time),
        map(p + ez, params, time) - map(p - ez, params, time)));
}

struct RayHit {
    float t;
    bool hit;
    int steps;
};

// Cone-stepping sphere trace. Mirrors the DirectX / Metal helper.
RayHit coneRaymarch(vec3 origin, vec3 dir, float t_start, float t_max, float time) {
    RayHit r;
    r.t = t_start;
    r.hit = false;
    r.steps = 0;
    float t = t_start;
    int cap = min(vol.vol_max_steps, 256);
    float ratio = max(vol.vol_cone_ratio, 0.01);
    const float surface_eps = 0.001;
    for (int i = 0; i < cap; ++i) {
        if (t >= t_max) break;
        vec3 p = origin + dir * t;
        float d = map(p, vol.vol_params, time);
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

// Sample the pre-raymarch scene with a normal-perturbed screen UV. `scene_color`
// is the post-AutoExposure, pre-raymarch HDR snapshot the encoder copies at the
// head of the pass. Mirrors the DirectX helper.
vec3 sampleSceneRefracted(vec2 frag_uv, vec3 normal, float strength) {
    vec2 refract_uv = clamp(frag_uv + normal.xz * strength, 0.0, 1.0);
    return textureLod(scene_color, refract_uv, 0.0).rgb;
}

// Slab ray-box intersection.
vec2 rayBox(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max) {
    vec3 inv = 1.0 / rd;
    vec3 t0 = (box_min - ro) * inv;
    vec3 t1 = (box_max - ro) * inv;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float t_enter = max(max(tmin.x, tmin.y), tmin.z);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2(t_enter, t_exit);
}

// PBR sun helper. Identical math to the Main pass.
vec3 shadePbrSun(SdfSurface s, vec3 normal, vec3 viewDir, DirLight sun, float shadow) {
    vec3 L = normalize(sun.dir_i.xyz);
    vec3 H = normalize(viewDir + L);
    float NdotL = max(0.0, dot(normal, L));
    float NdotV = max(1e-3, dot(normal, viewDir));
    float NdotH = max(0.0, dot(normal, H));
    float VdotH = max(0.0, dot(viewDir, H));

    float a = max(s.roughness * s.roughness, 1e-3);
    float a2 = a * a;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (3.14159265 * denom * denom);

    float k = (s.roughness + 1.0) * (s.roughness + 1.0) / 8.0;
    float G = (NdotL / (NdotL * (1.0 - k) + k)) * (NdotV / (NdotV * (1.0 - k) + k));

    vec3 F0 = mix(vec3(0.04), s.albedo, s.metallic);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

    vec3 spec = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-3);
    vec3 diff = (1.0 - F) * (1.0 - s.metallic) * s.albedo / 3.14159265;
    vec3 light = sun.col.xyz * sun.dir_i.w * shadow;
    return (diff + spec) * light * NdotL;
}

// Hand-tuned hemispheric ambient fallback (no EnvironmentMap bound).
vec3 shadeAmbient(SdfSurface s, vec3 normal) {
    vec3 sky = vec3(0.45, 0.52, 0.62);
    vec3 ground = vec3(0.07, 0.06, 0.05);
    float t = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
    vec3 hemi = mix(ground, sky, t);
    return s.albedo * hemi * 0.35 + s.emissive;
}

float raymarchHashRotation(vec2 p) {
    float h = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

// CSM cascade-shadow PCF -- mirrors `shadow_factor_cascaded` in main.frag so
// raymarched surfaces receive shadows that match rasterised geometry exactly.
float sampleSunShadow(vec3 world_pos, float view_depth, vec2 screen_xy) {
    int cascade = 4;
    if (view_depth < shadow_uni.cascade_splits[0]) cascade = 0;
    else if (view_depth < shadow_uni.cascade_splits[1]) cascade = 1;
    else if (view_depth < shadow_uni.cascade_splits[2]) cascade = 2;
    else if (view_depth < shadow_uni.cascade_splits[3]) cascade = 3;
    if (cascade >= 4) return 1.0;

    vec4 lc = shadow_uni.light_vps[cascade] * vec4(world_pos, 1.0);
    vec3 ndc = lc.xyz / max(lc.w, 1e-6);
    vec2 uv = vec2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    float bias = 0.0015 * (1.0 + float(cascade) * 0.7);
    float ref = ndc.z - bias;

    float angle = raymarchHashRotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    vec2 tex_size = 1.0 / vec2(textureSize(shadow_map, 0).xy);

    float sum = 0.0;
    const int RADIUS = 2;
    const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
            vec2 off = vec2(float(dx), float(dy));
            vec2 rot = vec2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
            vec2 sample_uv = uv + rot * tex_size;
            sum += texture(shadow_map, vec4(sample_uv, float(cascade), ref));
        }
    }
    return sum / SAMPLES;
}

// IBL ambient -- mirrors the main pass ambient term.
vec2 raymarchEnvBrdfApprox(float NdV, float rough) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.040, -0.040);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

vec3 raymarchFresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 shadeAmbientIbl(SdfSurface s, vec3 normal, vec3 view_dir, float prefilter_mip_count) {
    if (prefilter_mip_count <= 0.5) {
        return shadeAmbient(s, normal);
    }
    float NdV = max(dot(normal, view_dir), 0.0);
    vec3 F0 = mix(vec3(0.04), s.albedo, s.metallic);
    vec3 F_ibl = raymarchFresnelSchlick(NdV, F0);
    vec3 kd_ibl = (1.0 - F_ibl) * (1.0 - s.metallic);

    vec3 irradiance = textureLod(irradiance_cube, normal, 0.0).rgb;
    vec3 diffuse_ibl = kd_ibl * s.albedo * irradiance / 3.14159265;

    vec3 R = reflect(-view_dir, normal);
    float lod = s.roughness * (prefilter_mip_count - 1.0);
    vec3 prefiltered = textureLod(prefilter_cube, R, lod).rgb;
    vec2 ab = raymarchEnvBrdfApprox(NdV, s.roughness);
    vec3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

    return diffuse_ibl + specular_ibl + s.emissive;
}
