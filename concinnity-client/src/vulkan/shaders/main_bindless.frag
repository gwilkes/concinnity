#version 450

layout(location = 0) in vec3 frag_world_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_tangent;
layout(location = 3) in vec3 frag_bitangent;
layout(location = 4) in vec2 frag_uv;
layout(location = 5) in float frag_view_depth;
layout(location = 6) in vec3 frag_color;
layout(location = 7) flat in uint frag_object_id;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform ViewBlock {
    mat4  vp;
    mat4  view_mat;
    float elapsed;
    // 1.0 when an SSR / RT reflection composite owns the sharp specular this frame
    // (fade the glossy-dielectric forward probe specular); 0.0 keeps it all.
    float reflections_enabled;
    float cam_x; float cam_y; float cam_z;
    float prefilter_mip_count; float _ep0; float _ep1;
} view;

// Surfaces rougher than this get no SSR / RT reflection; the forward fade ramps in
// below it. Matches the resolve gloss gate (SSR_ROUGH_CUT / RT_ROUGH_CUT); keep in
// sync (the shared-prelude single-sourcing is a deferred cleanup).
const float REFLECTION_ROUGHNESS_CUT = 0.6;

struct DirLight  { vec4 dir_i; vec4 col; };
struct PointLight{ vec4 pos_r; vec4 col_i; };

layout(std140, set = 0, binding = 1) uniform LightBlock {
    DirLight   dir[4];
    PointLight pt[8];
    int num_dir;
    int num_pt;
    // Indirect-ambient multiplier (PostProcessConfig.ambient_intensity); 1.0 is
    // a no-op. First trailing pad word, matching the Rust LightUniforms layout.
    float ambient_intensity; float _lpad1;
} lights;

layout(std140, set = 0, binding = 2) uniform ShadowBlock {
    mat4 light_vps[4];
    vec4 cascade_splits;
} shadow_uni;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadow_map;
layout(set = 0, binding = 4) uniform samplerCube irradiance_cube;
layout(set = 0, binding = 5) uniform samplerCube prefilter_cube;
// Blurred SSAO occlusion (1×1 white when SSAO is disabled).
layout(set = 0, binding = 6) uniform sampler2D ssao_tex;

// Reflection-probe sampling: the ProbeSet UBO (binding 7), the probe cube array
// (binding 8), and the box-parallax partition-of-unity helpers, substituted in
// from probe_common.glsl at compile time (shaderc has no #include). With the set
// empty (count 0) the forward specular below keeps the sky prefilter path.
{PROBE_COMMON}

// Layout must match the #[repr(C)] GpuObjectData in gfx::render_types.
struct GpuObjectData {
    mat4  model;
    vec3  tint;      float roughness;
    vec3  emissive;  float metallic;
    uint  albedo_index;
    uint  normal_index;
    float macro_variation;
    float terrain_blend;
    vec3  bb_min;    float cull_distance;
    vec3  bb_max;    float secondary_blend_sharpness;
    uint  albedo_secondary_index;
    uint  normal_secondary_index;
    uint  emissive_map_index;
    uint  orm_map_index;
};

layout(std430, set = 1, binding = 0) readonly buffer ObjectBlock {
    GpuObjectData objects[];
} obj_buf;

// Bindless texture pool: [albedo textures..] ++ [normal maps..]. The object
// record's albedo_index / normal_index address it directly.
layout(set = 1, binding = 1) uniform sampler2D tex_pool[{POOL_SIZE}];

const float PI = 3.14159265359;

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
    int cascade = 4;
    if      (view_depth < shadow_uni.cascade_splits[0]) cascade = 0;
    else if (view_depth < shadow_uni.cascade_splits[1]) cascade = 1;
    else if (view_depth < shadow_uni.cascade_splits[2]) cascade = 2;
    else if (view_depth < shadow_uni.cascade_splits[3]) cascade = 3;
    if (cascade >= 4) return 1.0;

    float shade = sample_cascade_pcf(cascade, world_pos, screen_xy);

    if (cascade + 1 < 4) {
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
    GpuObjectData od = obj_buf.objects[frag_object_id];
    float roughness = od.roughness;
    float metallic  = od.metallic;
    vec3  tint      = od.tint;
    vec3  emissive  = od.emissive;

    vec3 cam_pos = vec3(view.cam_x, view.cam_y, view.cam_z);
    bool ibl_enabled = view.prefilter_mip_count > 0.5;

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

    vec4 albedo_samp = texture(tex_pool[od.albedo_index], frag_uv);
    vec3 albedo = albedo_samp.rgb * frag_color * tint;

    // Per-material emissive texture carries the colour (the scalar factor is a
    // uniform strength when a map is bound). Slot 0 is the "no map" sentinel.
    if (od.emissive_map_index != 0u) {
        emissive *= texture(tex_pool[od.emissive_map_index], frag_uv).rgb;
    }

    vec3 norm_samp = texture(tex_pool[od.normal_index], frag_uv).rgb * 2.0 - 1.0;
    mat3 TBN = mat3(
        normalize(frag_tangent),
        normalize(frag_bitangent),
        normalize(frag_normal)
    );
    vec3 N = normalize(TBN * norm_samp);

    // Geometric specular antialiasing on the normal map. Minification aliasing
    // is handled by the texture's mip chain (trilinear + anisotropic sampling);
    // this widens the specular NDF for residual sub-pixel normal variance.
    roughness = specular_aa_roughness(N, roughness);

    vec3 V   = normalize(cam_pos - frag_world_pos);
    float NdV = max(dot(N, V), 0.0);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    float shadow = shadow_factor_cascaded(frag_world_pos, frag_view_depth, gl_FragCoord.xy);

    vec2 ab        = env_brdf_approx(NdV, roughness);
    float ess      = ab.x + ab.y;
    vec3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

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

        vec3 kd = (1.0 - F) * (1.0 - metallic);
        vec3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        vec3 diff = kd * albedo / PI;

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

        vec3 kd   = (1.0 - F) * (1.0 - metallic);
        vec3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        vec3 diff = kd * albedo / PI;
        Lo += (diff + spec) * radiance * NdL;
    }

    vec3 ambient;
    if (ibl_enabled) {
        vec3 F_ibl       = fresnel_schlick(NdV, F0);
        vec3 kd_ibl      = (1.0 - F_ibl) * (1.0 - metallic);
        vec3 irradiance  = texture(irradiance_cube, N).rgb;
        vec3 diffuse_ibl = kd_ibl * albedo * irradiance / PI;

        vec3 R = reflect(-V, N);
        // texture() with a bias (not textureLod) so the reflection vector's
        // screen-space footprint widens the mip at grazing or distant angles.
        // A forced LOD defeats minification filtering and aliases the
        // environment into sparkle on near mirrors; flat close-up pixels have a
        // near-zero footprint, so they keep the plain roughness mip.
        float lod = roughness * (view.prefilter_mip_count - 1.0);
        // Local reflection probes when any are baked (box-parallax partition of
        // unity), else the imported environment prefilter cube. With no probe
        // baked `probe_set.count` is 0, so this is the sky path unchanged.
        vec3 prefiltered  = (probe_set.count > 0u)
            ? probe_set_specular(frag_world_pos, R, lod)
            : texture(prefilter_cube, R, lod).rgb;
        vec3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

        // When an SSR / RT reflection composite owns the sharp specular for glossy
        // surfaces this frame, fade the forward probe specular for glossy
        // dielectrics so the two do not double-count. Metals keep their full
        // albedo-tinted forward specular (the resolve adds only a faint dielectric
        // term), and surfaces rougher than the cut (which the resolve skips) keep
        // theirs too.
        if (view.reflections_enabled > 0.5) {
            float fade = smoothstep(REFLECTION_ROUGHNESS_CUT * 0.7, REFLECTION_ROUGHNESS_CUT, roughness);
            specular_ibl *= mix(1.0, fade, 1.0 - metallic);
        }

        ambient = diffuse_ibl + specular_ibl;
    } else {
        ambient = vec3(0.03) * albedo;
    }

    // Authored indirect-fill multiplier (PostProcessConfig.ambient_intensity);
    // 1.0 is a no-op. Lifts shadow fill without touching sun-lit surfaces.
    ambient *= lights.ambient_intensity;

    // SSAO modulates the indirect (ambient / IBL) term only - direct lighting
    // is unaffected. A 1×1 white SRV is bound when SSAO is disabled, so this
    // samples a constant 1.0 then.
    vec2 ssao_size = vec2(textureSize(ssao_tex, 0));
    vec2 ssao_uv   = gl_FragCoord.xy / ssao_size;
    ambient *= texture(ssao_tex, ssao_uv).r;

    vec3 color = ambient + Lo + emissive;

    out_color = vec4(color, albedo_samp.a);
}
