// GTAO horizon-search kernel. Reads the SSAO pre-pass G-buffer (view normal +
// linear view depth) and writes per-pixel visibility into an R8 occlusion
// target. Translated 1:1 from src/metal/shaders/ssao.metal::ssao_fragment.

cbuffer SsaoParams : register(b0)
{
    float radius;
    float intensity;
    float tan_half_fov_y;
    float aspect;
}

Texture2D    gbuffer    : register(t0);
SamplerState smp        : register(s0);

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

static const int   SSAO_SLICES  = 3;
static const int   SSAO_STEPS   = 6;
static const float SSAO_PI      = 3.14159265359;
static const float SSAO_HALF_PI = 1.57079632679;
// Cap on the kernel's UV footprint so geometry right in front of the camera
// does not blow the search radius out to most of the screen.
static const float SSAO_MAX_UV  = 0.2;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
float3 ssao_view_pos(float2 uv, float depth, float tan_half_y, float aspect_v)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_half_y * aspect_v, ndc.y * tan_half_y, -1.0) * depth;
}

float main(VsOut p) : SV_TARGET
{
    float4 c = gbuffer.Sample(smp, p.uv);
    float depth = c.a;
    if (depth <= 0.0) return 1.0;          // background - no geometry, fully lit

    float3 N = normalize(c.xyz);
    float3 P = ssao_view_pos(p.uv, depth, tan_half_fov_y, aspect);
    float3 V = normalize(-P);              // P is in view space; camera is origin

    float radius_uv = radius / max(2.0 * tan_half_fov_y * depth, 1e-4);
    radius_uv = min(radius_uv, SSAO_MAX_UV);

    // Interleaved gradient noise: per-pixel slice rotation + step jitter.
    float ign = frac(52.9829189 *
        frac(dot(p.sv_pos.xy, float2(0.06711056, 0.00583715))));

    float visibility = 0.0;
    [unroll] for (int s = 0; s < SSAO_SLICES; s++)
    {
        float ang = (float(s) + ign) * (SSAO_PI / float(SSAO_SLICES));
        float2 dir = float2(cos(ang), sin(ang));

        // Slice plane: spanned by V and the screen direction lifted to view
        // space. The projected normal and both horizons are measured inside it.
        float3 dir_vs   = normalize(float3(dir, 0.0));
        float3 plane_n  = normalize(cross(dir_vs, V));
        float3 proj_n   = N - plane_n * dot(N, plane_n);
        float  proj_len = length(proj_n);
        if (proj_len < 1e-4)
        {
            continue;
        }
        float3 tangent = cross(plane_n, V);
        float  n = atan2(dot(proj_n, tangent), dot(proj_n, V));

        // Horizon search: march both screen directions.
        float cos_plus  = -1.0;
        float cos_minus = -1.0;
        [unroll] for (int step = 1; step <= SSAO_STEPS; step++)
        {
            float t = (float(step) - 0.5 + ign) / float(SSAO_STEPS);
            float2 off = dir * radius_uv * t;

            float2 uvp = p.uv + off;
            float dp = gbuffer.Sample(smp, uvp).a;
            if (dp > 0.0)
            {
                float3 sp = ssao_view_pos(uvp, dp, tan_half_fov_y, aspect) - P;
                float  lp = length(sp);
                float  fo = saturate(1.0 - lp / max(radius, 1e-4));
                cos_plus = lerp(cos_plus, max(cos_plus, dot(sp / max(lp, 1e-5), V)), fo);
            }
            float2 uvm = p.uv - off;
            float dm = gbuffer.Sample(smp, uvm).a;
            if (dm > 0.0)
            {
                float3 sm = ssao_view_pos(uvm, dm, tan_half_fov_y, aspect) - P;
                float  lm = length(sm);
                float  fo = saturate(1.0 - lm / max(radius, 1e-4));
                cos_minus = lerp(cos_minus, max(cos_minus, dot(sm / max(lm, 1e-5), V)), fo);
            }
        }

        // Horizon angles, clamped into the hemisphere around the projected
        // normal, then the GTAO cosine-weighted arc integral for the slice.
        float h1 = -acos(clamp(cos_minus, -1.0, 1.0));
        float h2 =  acos(clamp(cos_plus,  -1.0, 1.0));
        h1 = n + max(h1 - n, -SSAO_HALF_PI);
        h2 = n + min(h2 - n,  SSAO_HALF_PI);
        float sin_n = sin(n);
        float cos_n = cos(n);
        float a1 = 0.25 * (-cos(2.0 * h1 - n) + cos_n + 2.0 * h1 * sin_n);
        float a2 = 0.25 * (-cos(2.0 * h2 - n) + cos_n + 2.0 * h2 * sin_n);
        visibility += proj_len * (a1 + a2);
    }

    visibility = saturate(visibility / float(SSAO_SLICES));
    // `intensity` sharpens the contact darkening; 1.0 is the integrated amount.
    return pow(visibility, max(intensity, 0.0));
}
