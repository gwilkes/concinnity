#pragma pack_matrix(column_major)

// SSR resolve fragment shader: a fullscreen ray-march over the pre-pass
// G-buffer + roughness texture, compositing the reflected scene colour over
// `scene`. Mirrors ssr_resolve_fragment in src/metal/shaders/ssr.metal.

cbuffer SsrParams : register(b0)
{
    float    intensity;
    float    max_distance;
    float    tan_half_fov_y;
    float    aspect;
    float    stride;
    float    thickness;
    // IBL prefilter cubemap mip count; 0 means no EnvironmentMap is bound and
    // the cube fallback is skipped (missed rays keep the base shading).
    float    prefilter_mip_count;
    float    _pad;
    // Camera-to-world transform; its 3x3 turns the view-space reflection ray into
    // the world-space direction the prefilter cubemap is sampled with. (The
    // translation column carries the camera position for the Metal reflection-probe
    // miss fallback; this backend uses only the 3x3.)
    float4x4 inv_view;
}

Texture2D    scene     : register(t0);
Texture2D    gbuffer   : register(t1);
Texture2D    rough_tex : register(t2);
TextureCube  prefilter : register(t3);
SamplerState smp       : register(s0);
SamplerState cube_smp  : register(s1);

// `cube_sampler` (s2), the probe cube array (t7..), and the `ProbeBlock` cbuffer
// (b4) are declared in probe_common.hlsl, concatenated ahead of this shader (the
// DX HLSL path has no #include handler). On a missed ray they let the resolve
// fall back to the local reflection probe instead of the foreign sky cube.

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

static const int   SSR_MAX_STEPS = 48;
static const int   SSR_REFINE    = 5;
// Surfaces rougher than `REFLECTION_ROUGHNESS_CUT` get no SSR; glossiness ramps in
// below it. The cut is the shared `static const` injected by the host
// (pipeline.rs::reflection_cut_prelude), so it matches the RT resolve + composite.

// Dielectric base reflectance (water, glass, polished stone) for the Fresnel.
static const float SSR_F0        = 0.04;
// UV margin over which a hit near the screen border fades out.
static const float SSR_EDGE_FADE = 0.12;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// The inverse of ssr_project; matches ssao_view_pos in the SSAO kernel.
float3 ssr_view_pos(float2 uv, float depth, float tan_y, float asp)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * asp, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
float2 ssr_project(float3 q, float tan_y, float asp)
{
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (tan_y * asp), q.y * inv / tan_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

float4 main(VsOut p) : SV_TARGET
{
    float3 base  = scene.Sample(smp, p.uv).rgb;
    float4 c     = gbuffer.Sample(smp, p.uv);
    float  depth = c.a;
    if (depth <= 0.0) return float4(base, 0.0);     // background / sky: weight 0

    float roughness = rough_tex.Sample(smp, p.uv).r;
    // Glossy surfaces reflect sharply; rough ones get nothing. A non-reflecting
    // pixel writes weight 0 so the composite keeps the scene there.
    float gloss = saturate((REFLECTION_ROUGHNESS_CUT - roughness) / REFLECTION_ROUGHNESS_CUT);
    if (gloss <= 0.0) return float4(base, 0.0);

    float3 N = normalize(c.xyz);
    float3 P = ssr_view_pos(p.uv, depth, tan_half_fov_y, aspect);
    float3 V = normalize(-P);                       // P in view space, camera at origin
    float3 R = reflect(-V, N);                      // reflected ray direction

    // Environment fallback for a missed (or screen-edge) ray, in the reflected
    // direction at a roughness-keyed mip so a rougher surface reflects a blurrier
    // environment (matching the main pass). With a baked reflection probe this is
    // the local scene capture (box-projected + blended across covering probes),
    // the same source the forward IBL specular term uses, rather than the foreign
    // sky HDR; otherwise it is the IBL prefilter cube. With no EnvironmentMap bound
    // there is nothing to fall back to, so missed rays keep the base shading.
    bool   ibl = prefilter_mip_count > 0.5;
    float3 env = base;
    if (ibl)
    {
        float3 r_world = mul((float3x3)inv_view, R);
        float  lod     = roughness * (prefilter_mip_count - 1.0);
        if (probes.count > 0u)
        {
            // The full inv_view (its translation column carries the camera
            // position) lifts the view-space surface point P to world space, which
            // the probe box-projection needs.
            float3 world_pos = mul(inv_view, float4(P, 1.0)).xyz;
            env = probe_set_specular(probes, world_pos, r_world, lod);
        }
        else
        {
            env = prefilter.SampleLevel(cube_smp, r_world, lod).rgb;
        }
    }

    float3 step_v = R * stride;
    float3 q = P;
    bool   hit = false;
    float2 hit_uv = p.uv;
    int    steps_taken = SSR_MAX_STEPS;
    [loop] for (int i = 0; i < SSR_MAX_STEPS; i++)
    {
        q += step_v;
        if (q.z >= 0.0) { steps_taken = i; break; } // crossed the camera plane
        float2 uv = ssr_project(q, tan_half_fov_y, aspect);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
        {
            steps_taken = i;
            break;
        }
        float scene_depth = gbuffer.Sample(smp, uv).a;
        if (scene_depth <= 0.0) continue;           // sky here - keep marching
        float diff = (-q.z) - scene_depth;          // > 0: ray is behind the surface
        if (diff > 0.0 && diff < thickness)
        {
            // Binary-search refine between the last two samples.
            float3 lo = q - step_v;
            float3 hi = q;
            [unroll] for (int r = 0; r < SSR_REFINE; r++)
            {
                float3 mid = (lo + hi) * 0.5;
                float2 muv = ssr_project(mid, tan_half_fov_y, aspect);
                float  sd  = gbuffer.Sample(smp, muv).a;
                if (sd > 0.0 && (-mid.z) - sd > 0.0) hi = mid; else lo = mid;
            }
            hit_uv = ssr_project(hi, tan_half_fov_y, aspect);
            hit = true;
            steps_taken = i;
            break;
        }
    }

    // The reflected colour: a single sharp screen-space tap (the reflection
    // composite blurs it by roughness), or the environment cube when the ray
    // missed. A hit near the screen border or at the end of its march fades toward
    // the environment rather than snapping flat to the base shading.
    float3 reflected;
    if (hit)
    {
        float3 hit_color = scene.Sample(smp, hit_uv).rgb;
        float2 e = smoothstep(0.0, SSR_EDGE_FADE, hit_uv)
                 * smoothstep(0.0, SSR_EDGE_FADE, 1.0 - hit_uv);
        float edge = e.x * e.y;
        float march = float(steps_taken) / float(SSR_MAX_STEPS);
        float dist_fade = 1.0 - smoothstep(0.7, 1.0, march);
        reflected = lerp(env, hit_color, edge * dist_fade);
    }
    else
    {
        reflected = env;
    }

    float ndv     = saturate(dot(N, V));
    float fresnel = SSR_F0 + (1.0 - SSR_F0) * pow(1.0 - ndv, 5.0);
    float w = saturate(fresnel * gloss * intensity);
    // Reflected radiance (.rgb) + composite weight (.a). The reflection composite
    // blurs this by surface roughness and blends it over the scene; the resolve no
    // longer composites inline.
    return float4(reflected, w);
}
