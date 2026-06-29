// src/directx/shaders/raymarch_helpers.hlsl
//
// Engine-shipped HLSL header for raymarched SDF volumes on D3D12.
// Prepended to the user's fragment shader at compile time; provides the
// type layouts the raymarch pass binds, the Inigo Quilez SDF primitive
// library, the cone-stepping marcher, and the PBR helpers user shaders
// call from `shade`.
//
// HLSL port of src/metal/shaders/raymarch_helpers.metal - the byte
// layouts of `RaymarchView` / `SdfVolumeUniforms` / `SdfSurface` / the
// light + shadow uniform blocks all stay identical to the Metal helpers
// so the Rust-side `RaymarchView` / `RaymarchVolumeUniforms` repr(C)
// structs round-trip across both backends.

#pragma pack_matrix(column_major)

// Uniform layouts - MUST stay in sync with
// concinnity-client/src/directx/raymarch.rs (Rust-side `repr(C)` structs) and
// the matching Metal layouts.

cbuffer RaymarchView : register(b0) {
  float4x4 view_vp;
  float4x4 view_inv_vp;
  float3 view_cam_pos;
  float view_pad0;
  float2 view_viewport;
  float view_time;
  float view_prefilter_mip_count;
};

struct SdfParams {
  float4
      vals[8];  // 32 floats packed as 8 vec4s (HLSL packs scalars in vec4 rows)
};

// Fetch a single param slot (0..31) via the packed vec4 array.
float sdfParamValue(SdfParams p, uint i) { return p.vals[i / 4u][i % 4u]; }

cbuffer SdfVolumeUniforms : register(b1) {
  float3 vol_centre;
  float vol_pad0;
  float3 vol_extent;
  float vol_pad1;
  float vol_cone_ratio;
  float vol_max_distance;
  int vol_max_steps;
  int vol_receive_shadows;
  SdfParams vol_params;
};

struct DirLight {
  float3 direction;
  float intensity;
  float3 color;
  float pad;
};

struct PointLight {
  float3 position;
  float range;
  float3 color;
  float intensity;
};

cbuffer RaymarchLights : register(b2) {
  DirLight light_directional[4];
  PointLight light_point[8];
  int light_num_directional;
  int light_num_point;
  float light_pad0;
  float light_pad1;
};

// Number of CSM cascades - must match `NUM_SHADOW_CASCADES` in
// gfx::render_types and the matching Metal constant.
static const int RAYMARCH_NUM_SHADOW_CASCADES = 4;

cbuffer RaymarchShadowUniforms : register(b3) {
  float4x4 shadow_light_vps[4];
  float4 shadow_cascade_splits;
  uint shadow_active_cascades;
};

// Per-point material the user's `shade` returns. Same layout as the
// Metal `SdfSurface` struct. `transmitted` is an additive contribution
// the template adds after PBR - opaque shaders leave it zero, refractive
// shaders set it to the scene-tap colour they want to show through.
struct SdfSurface {
  float3 albedo;
  float roughness;
  float metallic;
  float3 emissive;
  float3 transmitted;
};

// IQ primitive library - https://iquilezles.org/articles/distfunctions/

float sdSphere(float3 p, float r) { return length(p) - r; }

float sdBox(float3 p, float3 b) {
  float3 q = abs(p) - b;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundBox(float3 p, float3 b, float r) {
  float3 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdTorus(float3 p, float2 t) {
  float2 q = float2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}

float sdCapsule(float3 p, float3 a, float3 b, float r) {
  float3 pa = p - a;
  float3 ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h) - r;
}

float sdPlane(float3 p, float3 n, float h) { return dot(p, n) + h; }

float opSmoothUnion(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
  return lerp(b, a, h) - k * h * (1.0 - h);
}

float opSmoothSubtraction(float d1, float d2, float k) {
  float h = clamp(0.5 - 0.5 * (d2 + d1) / max(k, 1e-6), 0.0, 1.0);
  return lerp(d2, -d1, h) + k * h * (1.0 - h);
}

float opSmoothIntersection(float a, float b, float k) {
  float h = clamp(0.5 - 0.5 * (b - a) / max(k, 1e-6), 0.0, 1.0);
  return lerp(b, a, h) + k * h * (1.0 - h);
}

// User-provided functions - forward declarations. The user's shader
// (sandwiched between this header and the template) MUST define the
// functions matching its mode:
//   surface volumes (default) → `map` + `shade`
//   volumetric volumes        → `sampleVolume`
// The unused forward decls / engine helpers are DCE'd by FXC, so a
// volumetric author doesn't need a stub `map` / `shade` and vice versa.

float map(float3 p, SdfParams params, float time);
SdfSurface shade(float3 p, float3 normal, SdfParams params, float time,
                 float2 frag_uv, Texture2D<float4> scene_color,
                 SamplerState scene_samp);

// Per-point participating-media sample returned by a volumetric user
// shader. `density` is the extinction coefficient (sigma_t) in 1/metre;
// 0 = empty. `scattering` is the RGB single-scattering coefficient (the
// fraction of sun radiance that scatters along the view ray at this
// point; pre-multiplied by albedo). `emission` is self-emitted radiance.
struct VolumeSample {
  float density;
  float3 scattering;
  float3 emission;
};

VolumeSample sampleVolume(float3 p, SdfParams params, float time);

// Engine helpers the template + user shaders can call.

// 4-tap central-difference gradient. Normalised.
float3 sdfNormal(float3 p, SdfParams params, float time, float eps) {
  float3 ex = float3(eps, 0.0, 0.0);
  float3 ey = float3(0.0, eps, 0.0);
  float3 ez = float3(0.0, 0.0, eps);
  return normalize(
      float3(map(p + ex, params, time) - map(p - ex, params, time),
             map(p + ey, params, time) - map(p - ey, params, time),
             map(p + ez, params, time) - map(p - ez, params, time)));
}

struct RayHit {
  float t;
  bool hit;
  int steps;
};

// Cone-stepping sphere trace. Mirrors the Metal helper.
RayHit coneRaymarch(float3 origin, float3 dir, float t_start, float t_max,
                    float time) {
  RayHit r;
  r.t = t_start;
  r.hit = false;
  r.steps = 0;
  float t = t_start;
  int cap = min(vol_max_steps, 256);
  float ratio = max(vol_cone_ratio, 0.01);
  const float surface_eps = 0.001;
  for (int i = 0; i < cap; ++i) {
    if (t >= t_max) break;
    float3 p = origin + dir * t;
    float d = map(p, vol_params, time);
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

// Sample the pre-raymarch scene with a normal-perturbed screen UV.
// `scene_color` is bound to the `hdr_resolve_copy` snapshot the raymarch
// encoder takes at the head of the pass - i.e. the post-AutoExposure,
// pre-raymarch HDR scene. The perturbation is the world-space normal's
// XZ tilt scaled by `strength` (typical 0.02–0.10 for a water surface).
// Combine with any per-channel extinction the user shader wants before
// writing into `SdfSurface.transmitted`. Mirrors the Metal helper.
float3 sampleSceneRefracted(float2 frag_uv, float3 normal, float strength,
                            Texture2D<float4> scene_color,
                            SamplerState scene_samp) {
  float2 refract_uv = clamp(frag_uv + normal.xz * strength, 0.0, 1.0);
  return scene_color.SampleLevel(scene_samp, refract_uv, 0.0).rgb;
}

// Slab ray-box intersection.
float2 rayBox(float3 ro, float3 rd, float3 box_min, float3 box_max) {
  float3 inv = 1.0 / rd;
  float3 t0 = (box_min - ro) * inv;
  float3 t1 = (box_max - ro) * inv;
  float3 tmin = min(t0, t1);
  float3 tmax = max(t0, t1);
  float t_enter = max(max(tmin.x, tmin.y), tmin.z);
  float t_exit = min(min(tmax.x, tmax.y), tmax.z);
  return float2(t_enter, t_exit);
}

// PBR sun helper. Identical math to the Main pass.
float3 shadePbrSun(SdfSurface s, float3 normal, float3 viewDir, DirLight sun,
                   float shadow) {
  float3 L = normalize(sun.direction);
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
  float G =
      (NdotL / (NdotL * (1.0 - k) + k)) * (NdotV / (NdotV * (1.0 - k) + k));

  float3 F0 = lerp(float3(0.04, 0.04, 0.04), s.albedo, s.metallic);
  float3 F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);

  float3 spec = (D * G * F) / max(4.0 * NdotL * NdotV, 1e-3);
  float3 diff = (1.0 - F) * (1.0 - s.metallic) * s.albedo / 3.14159265;
  float3 light = sun.color * sun.intensity * shadow;
  return (diff + spec) * light * NdotL;
}

// Hand-tuned hemispheric ambient fallback. Used when no EnvironmentMap
// is bound (`view.prefilter_mip_count == 0`).
float3 shadeAmbient(SdfSurface s, float3 normal) {
  float3 sky = float3(0.45, 0.52, 0.62);
  float3 ground = float3(0.07, 0.06, 0.05);
  float t = clamp(0.5 + 0.5 * normal.y, 0.0, 1.0);
  float3 hemi = lerp(ground, sky, t);
  return s.albedo * hemi * 0.35 + s.emissive;
}

// CSM cascade-shadow PCF - mirrors `shadow_factor_cascaded` in the Metal
// helpers. Identical math so raymarched surfaces receive shadows that
// match rasterised geometry exactly.

float raymarchHashRotation(float2 p) {
  float h = frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
  return h * 6.2831853;
}

float sampleSunShadow(float3 world_pos, float view_depth, float2 screen_xy,
                      Texture2DArray<float> shadow_map,
                      SamplerComparisonState shadow_samp) {
  // Single return path - see `shadeAmbientIbl` for the X4000 rationale.
  float result = 1.0;
  int cascade = RAYMARCH_NUM_SHADOW_CASCADES;
  if (view_depth < shadow_cascade_splits.x)
    cascade = 0;
  else if (view_depth < shadow_cascade_splits.y)
    cascade = 1;
  else if (view_depth < shadow_cascade_splits.z)
    cascade = 2;
  else if (view_depth < shadow_cascade_splits.w)
    cascade = 3;

  if (cascade < (int)shadow_active_cascades) {
    float4 light_clip = mul(shadow_light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = light_clip.xyz / max(light_clip.w, 1e-6);
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if (!(any(uv < 0.0) || any(uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)) {
      float bias = 0.0015 * (1.0 + float(cascade) * 0.7);
      float ref = ndc.z - bias;

      float angle = raymarchHashRotation(screen_xy);
      float ca = cos(angle);
      float sa = sin(angle);

      uint w = 1, h = 1, n_layers = 1, n_mips = 1;
      shadow_map.GetDimensions(0, w, h, n_layers, n_mips);
      float2 tex_size = float2(1.0 / float(w), 1.0 / float(h));

      float sum = 0.0;
      const int RADIUS = 2;
      const float SAMPLES = float((2 * RADIUS + 1) * (2 * RADIUS + 1));
      for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {
          float2 off = float2(dx, dy);
          float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
          float2 sample_uv = uv + rot * tex_size;
          sum += shadow_map.SampleCmpLevelZero(
              shadow_samp, float3(sample_uv, float(cascade)), ref);
        }
      }
      result = sum / SAMPLES;
    }
  }
  return result;
}

// IBL - mirrors the ambient term in src/build/shaders/default_frag.hlsl.

float2 raymarchEnvBrdfApprox(float NdV, float rough) {
  const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
  const float4 c1 = float4(1.0, 0.0425, 1.040, -0.040);
  float4 r = rough * c0 + c1;
  float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
  return float2(-1.04, 1.04) * a004 + r.zw;
}

float3 raymarchFresnelSchlick(float cosTheta, float3 F0) {
  return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float3 shadeAmbientIbl(SdfSurface s, float3 normal, float3 view_dir,
                       float prefilter_mip_count,
                       TextureCube<float4> irradiance_cube,
                       TextureCube<float4> prefilter_cube,
                       SamplerState cube_samp) {
  // Single return path - FXC's X4000 ("potentially uninitialised
  // variable") fires on this function with the early-return style
  // because the intermediate-code generator treats the function's
  // return register as potentially-unwritten across the branch.
  float3 result;
  if (prefilter_mip_count <= 0.5) {
    result = shadeAmbient(s, normal);
  } else {
    float NdV = max(dot(normal, view_dir), 0.0);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), s.albedo, s.metallic);
    float3 F_ibl = raymarchFresnelSchlick(NdV, F0);
    float3 kd_ibl = (1.0 - F_ibl) * (1.0 - s.metallic);

    float3 irradiance = irradiance_cube.SampleLevel(cube_samp, normal, 0.0).rgb;
    float3 diffuse_ibl = kd_ibl * s.albedo * irradiance / 3.14159265;

    float3 R = reflect(-view_dir, normal);
    float lod = s.roughness * (prefilter_mip_count - 1.0);
    float3 prefiltered = prefilter_cube.SampleLevel(cube_samp, R, lod).rgb;
    float2 ab = raymarchEnvBrdfApprox(NdV, s.roughness);
    float3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

    result = diffuse_ibl + specular_ibl + s.emissive;
  }
  return result;
}
