// Default D3D12 fragment (pixel) shader for Concinnity scenes.
//
// Textured geometry with Cook-Torrance GGX PBR, tangent-space normal mapping,
// 4-cascade soft PCF shadows, IBL ambient (when bound), and a skybox pass.
//
// Root signature layout (must match directx/pipeline.rs):
//   b0 PushConstants    : model mat4 + material (28 DWORDs)
//   b1 ViewBlock        : vp mat4, view_mat, elapsed, cam xyz, prefilter_mip_count
//   b2 LightBlock       : up to 4 directional + 8 point lights
//   b3 ShadowBlock      : light_vps[4] + cascade_splits
//   t0 shadow_map       : depth array SRV (Texture2DArray, comparison sampler s0)
//   t1 albedo_tex       : RGBA albedo
//   t2 normal_tex       : tangent-space normal map
//   t5 irradiance_cube  : IBL irradiance TextureCube
//   t6 prefilter_cube   : IBL prefiltered radiance TextureCube (with mips)
//   s0 shadow sampler   (comparison, LessEqual)
//   s1 linear repeat sampler
//   s2 cube_sampler     (linear, clamp-to-edge)

#pragma pack_matrix(column_major)

#define NUM_SHADOW_CASCADES 4

cbuffer PushConstants : register(b0)
{
    float4x4 model;
    float roughness;
    float metallic;
    float _mpad0;
    float _mpad1;
    float3 tint;
    float _mpad2;
    float3 emissive;
    float _mpad3;
}

cbuffer ViewBlock : register(b1)
{
    float4x4 vp;
    float4x4 view_mat;
    float elapsed;
    float _pad0;
    float cam_x;
    float cam_y;
    float cam_z;
    // Number of mip levels in the bound IBL prefilter cubemap. 0 = IBL off.
    float prefilter_mip_count;
    float _ep0;
    float _ep1;
}

cbuffer ShadowBlock : register(b3)
{
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    float4   cascade_splits;
}

struct DirLight   { float4 dir_i;  float4 col;   };
struct PointLight { float4 pos_r;  float4 col_i; };

cbuffer LightBlock : register(b2)
{
    DirLight   dir[4];
    PointLight pt[8];
    int num_dir;
    int num_pt;
    // Indirect-ambient multiplier (PostProcessConfig.ambient_intensity); 1.0 is
    // a no-op. First trailing pad word, so the cbuffer still matches the Rust
    // LightUniforms layout.
    float ambient_intensity;
    float _lpad1;
}

Texture2DArray<float>  shadow_map      : register(t0);
SamplerComparisonState shadow_sampler  : register(s0);
Texture2D              albedo_tex      : register(t1);
SamplerState           linear_sampler  : register(s1);
Texture2D              normal_tex      : register(t2);
// Blurred SSAO occlusion (1x1 white when SSAO is disabled).
Texture2D              ssao_tex        : register(t4);
TextureCube            irradiance_cube : register(t5);
TextureCube            prefilter_cube  : register(t6);
SamplerState           cube_sampler    : register(s2);

struct PsIn
{
    float4 sv_pos     : SV_POSITION;
    float3 world_pos  : TEXCOORD0;
    float3 normal     : TEXCOORD1;
    float3 tangent    : TEXCOORD2;
    float3 bitangent  : TEXCOORD3;
    float2 uv         : TEXCOORD4;
    float  view_depth : TEXCOORD5;
    float3 color      : TEXCOORD6;
};

static const float PI = 3.14159265359;

// Sky gradient matching the procedural sky texture generator.
static const float3 SKY_ZENITH  = float3(0.110, 0.322, 0.726);
static const float3 SKY_HORIZON = float3(0.765, 0.863, 0.941);

float distribution_ggx(float3 N, float3 H, float rough)
{
    float a  = rough * rough;
    float a2 = a * a;
    float NdH  = max(dot(N, H), 0.0);
    float NdH2 = NdH * NdH;
    float denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

float geometry_schlick_ggx(float NdV, float rough)
{
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    return NdV / (NdV * (1.0 - k) + k);
}

float geometry_smith(float3 N, float3 V, float3 L, float rough)
{
    float NdV = max(dot(N, V), 0.0);
    float NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, rough) * geometry_schlick_ggx(NdL, rough);
}

float3 fresnel_schlick(float cosTheta, float3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Karis 2014 analytic fit of the GGX directional-albedo BRDF LUT. Returns the
// (scale, bias) pair such that single-scatter spec albedo for a given F0 is
// approximately F0 * scale + bias. Used for direct-light energy compensation
// here and, later, for IBL specular when env maps land.
float2 env_brdf_approx(float NdV, float rough)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4( 1.0,  0.0425,  1.040, -0.040);
    float4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

float hash_rotation(float2 p)
{
    float h = frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453);
    return h * 6.2831853;
}

// 5x5 hash-rotated PCF of a single cascade. Returns the shadow factor in
// [0, 1] (1.0 fully lit), or 1.0 when the fragment lies outside this cascade's
// light frustum. Mirrors `sample_cascade_pcf` in default.metal.
float sample_cascade_pcf(uint cascade, float3 world_pos, float2 screen_xy)
{
    float4 lc = mul(light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = lc.xyz / lc.w;
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    float depth = ndc.z;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ||
        depth < 0.0 || depth > 1.0)
        return 1.0;

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
    float3 vp_row2 = light_vps[cascade][2].xyz;
    float bias = 0.03 * (1.0 + (float)cascade * 2.0) * length(vp_row2);
    float ref  = depth - bias;

    float angle = hash_rotation(screen_xy);
    float ca = cos(angle);
    float sa = sin(angle);

    uint w, h, elems, mips;
    shadow_map.GetDimensions(0, w, h, elems, mips);
    float2 tex_size = 1.0 / float2((float)w, (float)h);

    float sum = 0.0;
    [unroll] for (int dy = -2; dy <= 2; dy++)
    [unroll] for (int dx = -2; dx <= 2; dx++)
    {
        float2 off = float2(dx, dy);
        float2 rot = float2(off.x * ca - off.y * sa, off.x * sa + off.y * ca);
        float2 sample_uv = uv + rot * tex_size;
        sum += shadow_map.SampleCmpLevelZero(
            shadow_sampler, float3(sample_uv, (float)cascade), ref);
    }
    return sum / 25.0;
}

// Cascade-aware soft PCF with cross-cascade blending. Picks the smallest
// cascade whose view-space far depth exceeds this fragment's view depth, then
// blends into the next cascade across a band at the far edge of that cascade's
// depth range. Each cascade places the shadow edge slightly differently (its
// own texel grid + resolution), and the split boundary sits a fixed distance
// ahead of the camera, so under a hard switch that boundary sweeps across the
// world as the camera moves and the shadow edge appears to glide. Blending the
// factor over the band turns the jump into a smooth, world-anchored transition.
// Mirrors `shadow_factor_cascaded` in default.metal.
float shadow_factor_cascaded(float3 world_pos, float view_depth, float2 screen_xy)
{
    uint cascade = NUM_SHADOW_CASCADES;
    if      (view_depth < cascade_splits[0]) cascade = 0;
    else if (view_depth < cascade_splits[1]) cascade = 1;
    else if (view_depth < cascade_splits[2]) cascade = 2;
    else if (view_depth < cascade_splits[3]) cascade = 3;
    if (cascade >= NUM_SHADOW_CASCADES) return 1.0;

    float shade = sample_cascade_pcf(cascade, world_pos, screen_xy);

    if (cascade + 1 < NUM_SHADOW_CASCADES)
    {
        uint  prev       = (cascade == 0) ? 0 : cascade - 1;
        float split_far  = cascade_splits[cascade];
        float split_near = (cascade == 0) ? 0.0 : cascade_splits[prev];
        float band = (split_far - split_near) * 0.15;
        float t = (view_depth - (split_far - band)) / max(band, 1e-4);
        if (t > 0.0)
        {
            float next = sample_cascade_pcf(cascade + 1, world_pos, screen_xy);
            shade = lerp(shade, next, saturate(t));
        }
    }
    return shade;
}

float4 main(PsIn p) : SV_TARGET
{
    float3 cam_pos = float3(cam_x, cam_y, cam_z);
    bool ibl_enabled = prefilter_mip_count > 0.5;

    // Skybox pass: blue channel sentinel > 1.5.
    if (p.color.b > 1.5) {
        float3 view_dir = normalize(p.world_pos - cam_pos);
        if (ibl_enabled) {
            return float4(prefilter_cube.SampleLevel(cube_sampler, view_dir, 0.0).rgb, 1.0);
        }
        float t = max(0.0, view_dir.y);
        return float4(lerp(SKY_HORIZON, SKY_ZENITH, t), 1.0);
    }

    float4 albedo_samp = albedo_tex.Sample(linear_sampler, p.uv);
    float3 albedo = albedo_samp.rgb * p.color * tint;

    float3 norm_samp = normal_tex.Sample(linear_sampler, p.uv).rgb * 2.0 - 1.0;
    float3x3 TBN = float3x3(
        normalize(p.tangent),
        normalize(p.bitangent),
        normalize(p.normal)
    );
    float3 N = normalize(mul(norm_samp, TBN));

    float3 V   = normalize(cam_pos - p.world_pos);
    float  NdV = max(dot(N, V), 0.0);
    float3 F0  = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);

    float shadow = shadow_factor_cascaded(p.world_pos, p.view_depth, p.sv_pos.xy);

    // Energy-conserving multi-scatter compensation (Fdez-Aguera / Filament).
    // Karis BRDF approximation gives the single-scatter directional albedo Eo;
    // 1 + F0 * (1/Eo - 1) restores the energy that GGX masking-shadowing drops.
    // View-only; reused across every direct light.
    float2 ab        = env_brdf_approx(NdV, roughness);
    float  ess       = ab.x + ab.y;
    float3 energy_ms = 1.0 + F0 * (1.0 / max(ess, 0.001) - 1.0);

    float3 Lo = float3(0.0, 0.0, 0.0);

    for (int i = 0; i < num_dir; i++)
    {
        float3 L        = normalize(dir[i].dir_i.xyz);
        float  intens   = dir[i].dir_i.w;
        float3 radiance = dir[i].col.xyz * intens;
        float3 H   = normalize(V + L);
        float  NdL = max(dot(N, L), 0.0);
        float  D   = distribution_ggx(N, H, roughness);
        float  G   = geometry_smith(N, V, L, roughness);
        float3 F   = fresnel_schlick(max(dot(H, V), 0.0), F0);
        float3 kd  = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;
        float  s    = (i == 0) ? shadow : 1.0;
        Lo += (diff + spec) * radiance * NdL * s;
    }

    for (int j = 0; j < num_pt; j++)
    {
        float3 pos_w  = pt[j].pos_r.xyz;
        float  range  = pt[j].pos_r.w;
        float3 col    = pt[j].col_i.xyz;
        float  intens = pt[j].col_i.w;
        float3 L      = normalize(pos_w - p.world_pos);
        float  dist   = length(pos_w - p.world_pos);
        float  atten  = clamp(1.0 - (dist / range), 0.0, 1.0);
        atten *= atten;
        float3 radiance = col * intens * atten;
        float3 H   = normalize(V + L);
        float  NdL = max(dot(N, L), 0.0);
        float  D   = distribution_ggx(N, H, roughness);
        float  G   = geometry_smith(N, V, L, roughness);
        float3 F   = fresnel_schlick(max(dot(H, V), 0.0), F0);
        float3 kd  = (1.0 - F) * (1.0 - metallic);
        float3 spec = (D * G * F) / max(4.0 * NdV * NdL, 0.001) * energy_ms;
        float3 diff = kd * albedo / PI;
        Lo += (diff + spec) * radiance * NdL;
    }

    float3 ambient;
    if (ibl_enabled) {
        float3 F_ibl       = fresnel_schlick(NdV, F0);
        float3 kd_ibl      = (1.0 - F_ibl) * (1.0 - metallic);
        float3 irradiance  = irradiance_cube.Sample(cube_sampler, N).rgb;
        float3 diffuse_ibl = kd_ibl * albedo * irradiance / PI;

        float3 R = reflect(-V, N);
        // SampleBias (not SampleLevel) so the reflection vector's screen-space
        // footprint widens the mip at grazing or distant angles. A fixed level
        // defeats minification filtering and aliases the environment into
        // sparkle on near mirrors; flat close-up pixels have a near-zero
        // footprint, so they keep the plain roughness mip.
        float  lod = roughness * (prefilter_mip_count - 1.0);
        float3 prefiltered  = prefilter_cube.SampleBias(cube_sampler, R, lod).rgb;
        float3 specular_ibl = prefiltered * (F0 * ab.x + ab.y);

        ambient = diffuse_ibl + specular_ibl;
    } else {
        // Soft blue-tinted sky bounce so worlds without IBL aren't near-black.
        ambient = float3(0.35, 0.4, 0.5) * 0.4 * albedo;
    }

    // Authored indirect-fill multiplier (PostProcessConfig.ambient_intensity);
    // 1.0 is a no-op. Lifts shadow fill without touching sun-lit surfaces.
    ambient *= ambient_intensity;

    // Screen-space ambient occlusion modulates the indirect (ambient / IBL)
    // term only - direct lighting is unaffected. A 1x1 white SRV is bound
    // when SSAO is disabled, so this samples a constant 1.0 then.
    uint sw, sh;
    ssao_tex.GetDimensions(sw, sh);
    float2 ssao_uv = p.sv_pos.xy / float2(float(sw), float(sh));
    ambient *= ssao_tex.Sample(linear_sampler, ssao_uv).r;

    // Linear-light HDR output. The ACES tonemap + gamma encode + FXAA run in
    // the off-screen composite pass (see directx/pipeline.rs COMPOSITE_FRAG_HLSL).
    float3 color = ambient + Lo + emissive;
    return float4(color, albedo_samp.a);
}
