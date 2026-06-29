#version 450

layout(location = 0) in vec3 frag_world_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_tangent;
layout(location = 3) in vec3 frag_bitangent;
layout(location = 4) in vec2 frag_uv;
layout(location = 5) in float frag_view_depth;
layout(location = 6) in vec3 frag_color;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform ViewBlock {
    mat4  vp;
    mat4  view_mat;
    float elapsed;
    float _pad0;
    float cam_x; float cam_y; float cam_z;
    // prefilter_mip_count = number of mip levels in the IBL prefilter cube. 0 = IBL off.
    float prefilter_mip_count; float _ep0; float _ep1;
} view;

// std140 LightUniforms: two vec4s per light keeps each light at 32 bytes,
// matching the Rust struct (direction[3]+intensity+color[3]+_pad = 32 bytes).
struct DirLight  { vec4 dir_i; vec4 col; };   // dir_i.xyz=dir, .w=intensity; col.xyz=color
struct PointLight{ vec4 pos_r; vec4 col_i; }; // pos_r.xyz=pos, .w=range;   col_i.xyz=color, .w=intensity

layout(std140, set = 0, binding = 1) uniform LightBlock {
    DirLight   dir[4];
    PointLight pt[8];
    int num_dir;
    int num_pt;
    float _lpad0; float _lpad1;
} lights;

layout(std140, set = 0, binding = 2) uniform ShadowBlock {
    mat4 light_vps[4];
    vec4 cascade_splits;
    // Live cascade count (1..4); slots at or beyond it are unrendered, so the
    // selection + blend below must not reach them.
    uint active_cascades;
} shadow_uni;

// 4-layer array shadow map (one slice per cascade), sampled with depth-compare
// in `shadow_factor_cascaded` below.
layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadow_map;

// IBL: irradiance + prefilter cubes, scene-wide. Bound to a 1x1 fallback
// when no EnvironmentMap is present so the shader sampler is always valid.
layout(set = 0, binding = 4) uniform samplerCube irradiance_cube;
layout(set = 0, binding = 5) uniform samplerCube prefilter_cube;

// Blurred SSAO occlusion (1×1 white when SSAO is disabled). Modulates the
// indirect ambient/IBL term only; direct lighting is unaffected.
layout(set = 0, binding = 6) uniform sampler2D ssao_tex;

// Per-object textures
layout(set = 1, binding = 0) uniform sampler2D albedo_tex;
layout(set = 1, binding = 1) uniform sampler2D normal_tex;

layout(push_constant) uniform PushBlock {
    mat4  model;
    float roughness;
    float metallic;
    float _mpad0; float _mpad1;
    vec3  tint;
    float _mpad2;
    vec3  emissive;
    float _mpad3;
} push;

const float PI = 3.14159265359;

// Procedural sky palette; falls back when no EnvironmentMap is bound.
const vec3 SKY_ZENITH  = vec3(0.110, 0.322, 0.726);
const vec3 SKY_HORIZON = vec3(0.765, 0.863, 0.941);

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float NdH2 = NdH * NdH;
    float denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

float geometry_schlick_ggx(float NdV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Karis 2014 analytic fit of the GGX directional-albedo BRDF LUT. Returns the
// (scale, bias) pair such that single-scatter spec albedo for a given F0 is
// approximately F0 * scale + bias. Used for direct-light energy compensation
// here and, later, for IBL specular when env maps land.
vec2 env_brdf_approx(float NdV, float rough) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Geometric specular antialiasing (Kaplanyan et al. 2016, as in Filament):
// widen the NDF by the screen-space variance of the shading normal so an
// undersampled high-frequency normal map at a distance does not alias into
// specular fireflies. A no-op where the normal is smooth (close up), so the
// surface detail is preserved.
float specular_aa_roughness(vec3 N, float perceptual_roughness) {
    const float VARIANCE  = 0.25;
    const float THRESHOLD = 0.18;
    vec3 dndx = dFdx(N);
    vec3 dndy = dFdy(N);
    float variance = VARIANCE * (dot(dndx, dndx) + dot(dndy, dndy));
    float alpha = perceptual_roughness * perceptual_roughness;
    float kernel = min(2.0 * variance, THRESHOLD);
    float filtered_alpha2 = clamp(alpha * alpha + kernel, 0.0, 1.0);
    return sqrt(sqrt(filtered_alpha2));
}

// Hash a 2D pixel coord to a rotation angle in [0, 2*pi).
float hash_rotation(vec2 p) {
    float h = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

// 5x5 hash-rotated PCF of a single cascade via sampler2DArrayShadow. Returns
// the shadow factor in [0, 1] (1.0 fully lit), or 1.0 when the fragment lies
// outside this cascade's light frustum. Mirrors `sample_cascade_pcf` in
// default.metal.
float sample_cascade_pcf(int cascade, vec3 world_pos, vec2 screen_xy) {
    vec4 lc = shadow_uni.light_vps[cascade] * vec4(world_pos, 1.0);
    vec3 ndc = lc.xyz / lc.w;
    // Flip Y: the shadow pass renders with a negative-height viewport, so the
    // sampled UV must mirror Y to match (same as Metal / DirectX).
    vec2 uv = vec2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    // Depth bias as a world-space offset along the light: in NDC that is the
    // world offset over the cascade depth range, i.e. world_bias * length(VP
    // row2 xyz) (the ortho z scale). csm.rs extends each cascade's near plane to
    // capture tall casters, decoupling the depth range from the XY radius, so
    // deriving the base from row2 keeps a given world offset constant per
    // cascade. The (1 + cascade * 2) factor then grows that offset with cascade
    // index: a distant cascade covers more world per shadow texel, so a flat
    // bias under-biases the far cascades and leaves self-shadow acne that steps
    // at each cascade boundary (a faint line that sweeps with the camera). The
    // raymarch/fog shadow taps scale bias the same way; Metal applies the
    // per-cascade term in the shadow-pass rasterizer (metal/draw/shadow.rs).
    vec3 vp_row2 = vec3(shadow_uni.light_vps[cascade][0][2],
                        shadow_uni.light_vps[cascade][1][2],
                        shadow_uni.light_vps[cascade][2][2]);
    float bias = 0.03 * (1.0 + float(cascade) * 2.0) * length(vp_row2);
    float ref = ndc.z - bias;

    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    vec2 tex_size = 1.0 / vec2(textureSize(shadow_map, 0).xy);

    float sum = 0.0;
    const int RADIUS = 2; // 5x5
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

// Cascade-aware PCF with cross-cascade blending. Selects the cascade whose far
// split exceeds the fragment's view-space depth, then blends into the next
// cascade across a band at the far edge of that cascade's depth range. Each
// cascade places the shadow edge slightly differently (its own texel grid +
// resolution), and the split boundary sits a fixed distance ahead of the
// camera, so under a hard switch that boundary sweeps across the world as the
// camera moves and the shadow edge appears to glide. Blending the factor over
// the band turns the jump into a smooth, world-anchored transition. Mirrors
// `shadow_factor_cascaded` in default.metal.
float shadow_factor_cascaded(vec3 world_pos, float view_depth, vec2 screen_xy) {
    int cascade = 4; // sentinel: "beyond last cascade"
    if      (view_depth < shadow_uni.cascade_splits[0]) cascade = 0;
    else if (view_depth < shadow_uni.cascade_splits[1]) cascade = 1;
    else if (view_depth < shadow_uni.cascade_splits[2]) cascade = 2;
    else if (view_depth < shadow_uni.cascade_splits[3]) cascade = 3;
    if (cascade >= int(shadow_uni.active_cascades)) return 1.0;

    float shade = sample_cascade_pcf(cascade, world_pos, screen_xy);

    if (cascade + 1 < int(shadow_uni.active_cascades)) {
        float split_far  = shadow_uni.cascade_splits[cascade];
        float split_near = (cascade == 0) ? 0.0 : shadow_uni.cascade_splits[cascade - 1];
        float band = (split_far - split_near) * 0.15;
        float t = (view_depth - (split_far - band)) / max(band, 1e-4);
        if (t > 0.0) {
            float next = sample_cascade_pcf(cascade + 1, world_pos, screen_xy);
            shade = mix(shade, next, clamp(t, 0.0, 1.0));
        }
    }
    return shade;
}

void main() {
    vec3 cam_pos = vec3(view.cam_x, view.cam_y, view.cam_z);
    bool ibl_enabled = view.prefilter_mip_count > 0.5;

    // Skybox pass: blue channel sentinel > 1.5 (matches Metal/DX).
    if (frag_color.b > 1.5) {
        vec3 view_dir = normalize(frag_world_pos - cam_pos);
        vec3 sky;
        if (ibl_enabled) {
            sky = textureLod(prefilter_cube, view_dir, 0.0).rgb;
        } else {
            float t = max(0.0, view_dir.y);
            sky = mix(SKY_HORIZON, SKY_ZENITH, t);
        }
        out_color = vec4(sky, 1.0);
        return;
    }

    // Sample albedo and apply vertex color + tint.
    vec4 albedo_samp = texture(albedo_tex, frag_uv);
    vec3 albedo = albedo_samp.rgb * frag_color * push.tint;

    // Reconstruct normal from normal map.
    vec3 norm_samp = texture(normal_tex, frag_uv).rgb * 2.0 - 1.0;
    mat3 TBN = mat3(
        normalize(frag_tangent),
        normalize(frag_bitangent),
        normalize(frag_normal)
    );
    vec3 N = normalize(TBN * norm_samp);

    // Geometric specular antialiasing on the normal map. Minification aliasing
    // is handled by the texture's mip chain (trilinear + anisotropic sampling);
    // this widens the specular NDF for residual sub-pixel normal variance.
    float roughness = specular_aa_roughness(N, push.roughness);

    vec3 V   = normalize(cam_pos - frag_world_pos);
    float NdV = max(dot(N, V), 0.0);

    // PBR base reflectance.
    vec3 F0 = mix(vec3(0.04), albedo, push.metallic);

    // Cascade-aware PCF shadow factor for the first directional light.
    float shadow = shadow_factor_cascaded(frag_world_pos, frag_view_depth, gl_FragCoord.xy);

    // Energy-conserving multi-scatter compensation (Fdez-Aguera / Filament).
    // Karis BRDF approximation gives the single-scatter directional albedo Eo;
    // 1 + F0 * (1/Eo - 1) restores the energy that GGX masking-shadowing drops.
    // View-only; reused across every direct light.
    vec2 ab        = env_brdf_approx(NdV, roughness);
    float ess      = ab.x + ab.y;
    vec3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

    // Accumulate light contributions.
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < lights.num_dir; i++) {
        vec3 L = normalize(lights.dir[i].dir_i.xyz);
        float intensity = lights.dir[i].dir_i.w;
        vec3 radiance = lights.dir[i].col.xyz * intensity;

        vec3 H = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        vec3  F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        vec3 kd = (1.0 - F) * (1.0 - push.metallic);
        vec3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        vec3 diff = kd * albedo / PI;

        // Apply shadow only to the first directional light.
        float s = (i == 0) ? shadow : 1.0;
        Lo += (diff + spec) * radiance * NdL * s;
    }

    for (int i = 0; i < lights.num_pt; i++) {
        vec3  pos_w   = lights.pt[i].pos_r.xyz;
        float range   = lights.pt[i].pos_r.w;
        vec3  col     = lights.pt[i].col_i.xyz;
        float intens  = lights.pt[i].col_i.w;

        vec3  L    = normalize(pos_w - frag_world_pos);
        float dist = length(pos_w - frag_world_pos);
        float atten = clamp(1.0 - (dist / range), 0.0, 1.0);
        atten *= atten;
        vec3 radiance = col * intens * atten;

        vec3  H   = normalize(V + L);
        float NdL = max(dot(N, L), 0.0);

        float D = distribution_ggx(N, H, roughness);
        float G = geometry_smith(N, V, L, roughness);
        vec3  F = fresnel_schlick(max(dot(H, V), 0.0), F0);

        vec3 kd   = (1.0 - F) * (1.0 - push.metallic);
        vec3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        vec3 diff = kd * albedo / PI;
        Lo += (diff + spec) * radiance * NdL;
    }

    // Ambient: IBL when bound, otherwise legacy flat placeholder.
    vec3 ambient;
    if (ibl_enabled) {
        vec3 F_ibl       = fresnel_schlick(NdV, F0);
        vec3 kd_ibl      = (1.0 - F_ibl) * (1.0 - push.metallic);
        vec3 irradiance  = texture(irradiance_cube, N).rgb;
        vec3 diffuse_ibl = kd_ibl * albedo * irradiance / PI;

        vec3 R = reflect(-V, N);
        // texture() with a bias (not textureLod) so the reflection vector's
        // screen-space footprint widens the mip at grazing or distant angles.
        // A forced LOD defeats minification filtering and aliases the
        // environment into sparkle on near mirrors; flat close-up pixels have a
        // near-zero footprint, so they keep the plain roughness mip.
        float lod = roughness * (view.prefilter_mip_count - 1.0);
        vec3 prefiltered  = texture(prefilter_cube, R, lod).rgb;
        vec3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

        ambient = diffuse_ibl + specular_ibl;
    } else {
        ambient = vec3(0.03) * albedo;
    }

    // SSAO modulates the indirect (ambient / IBL) term only - direct lighting
    // is unaffected. A 1×1 white SRV is bound when SSAO is disabled, so this
    // samples a constant 1.0 then.
    vec2 ssao_size = vec2(textureSize(ssao_tex, 0));
    vec2 ssao_uv   = gl_FragCoord.xy / ssao_size;
    ambient *= texture(ssao_tex, ssao_uv).r;

    vec3 color = ambient + Lo + push.emissive;

    // Write linear-light HDR. The composite pass owns the ACES tonemap +
    // gamma 2.2 encode (see COMPOSITE_FRAG_GLSL); doing it here would double
    // tonemap. Matches Metal's default.metal, which also writes raw HDR.
    out_color = vec4(color, albedo_samp.a);
}
