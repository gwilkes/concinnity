#include <metal_stdlib>
using namespace metal;

struct SsaoVtxOut {
    float4 position [[position]];
    float2 uv;
};

// Fullscreen triangle generated from vertex_id 0..2 - no vertex buffer.
vertex SsaoVtxOut ssao_fullscreen_vertex(uint vid [[vertex_id]]) {
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    SsaoVtxOut out;
    out.position = float4(pos, 0.0, 1.0);
    out.uv = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return out;
}

// buffer(0): SSAO tunables. Layout matches render_types::SsaoParams.
struct SsaoParams {
    float radius;
    float intensity;
    float tan_half_fov_y;
    float aspect;
};

constant int   SSAO_SLICES   = 3;
constant int   SSAO_STEPS    = 6;
constant float SSAO_PI       = 3.14159265359;
constant float SSAO_HALF_PI  = 1.57079632679;
// Cap on the kernel's UV footprint so geometry right in front of the camera
// does not blow the search radius out to most of the screen.
constant float SSAO_MAX_UV   = 0.2;

// Rebuild a view-space position from a UV and its linear (view-space) depth.
static float3 ssao_view_pos(float2 uv, float depth, float tan_half_y, float aspect) {
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_half_y * aspect, ndc.y * tan_half_y, -1.0) * depth;
}

fragment float ssao_fragment(
    SsaoVtxOut in            [[stage_in]],
    constant SsaoParams &p   [[buffer(0)]],
    texture2d<float> gbuffer [[texture(0)]],
    sampler smp              [[sampler(0)]]
) {
    float4 c = gbuffer.sample(smp, in.uv);
    float depth = c.a;
    if (depth <= 0.0) return 1.0;          // background - no geometry, fully lit

    float3 N = normalize(c.xyz);
    float3 P = ssao_view_pos(in.uv, depth, p.tan_half_fov_y, p.aspect);
    float3 V = normalize(-P);              // P is in view space; camera is origin

    // UV-space radius of the world-space search radius at this depth. The
    // viewport spans 2*tan_half_fov*depth view units vertically.
    float radius_uv = p.radius / max(2.0 * p.tan_half_fov_y * depth, 1e-4);
    radius_uv = min(radius_uv, SSAO_MAX_UV);

    // Interleaved gradient noise: a per-pixel slice rotation + step jitter that
    // trades banding for high-frequency noise the blur pass then cleans up.
    float ign = fract(52.9829189 *
        fract(dot(in.position.xy, float2(0.06711056, 0.00583715))));

    float visibility = 0.0;
    for (int s = 0; s < SSAO_SLICES; s++) {
        float ang = (float(s) + ign) * (SSAO_PI / float(SSAO_SLICES));
        float2 dir = float2(cos(ang), sin(ang));

        // Slice plane: spanned by V and the screen direction lifted to view
        // space. The projected surface normal and both horizons are measured
        // inside this plane.
        float3 dir_vs   = normalize(float3(dir, 0.0));
        float3 plane_n  = normalize(cross(dir_vs, V));
        float3 proj_n   = N - plane_n * dot(N, plane_n);
        float  proj_len = length(proj_n);
        if (proj_len < 1e-4) {
            continue;
        }
        float3 tangent = cross(plane_n, V);
        float  n = atan2(dot(proj_n, tangent), dot(proj_n, V));

        // Horizon search: march both screen directions, keeping the widest
        // horizon cosine, distance-attenuated so far occluders fade out.
        float cos_plus  = -1.0;
        float cos_minus = -1.0;
        for (int step = 1; step <= SSAO_STEPS; step++) {
            float t = (float(step) - 0.5 + ign) / float(SSAO_STEPS);
            float2 off = dir * radius_uv * t;

            float2 uvp = in.uv + off;
            float dp = gbuffer.sample(smp, uvp).a;
            if (dp > 0.0) {
                float3 sp = ssao_view_pos(uvp, dp, p.tan_half_fov_y, p.aspect) - P;
                float  lp = length(sp);
                float  fo = saturate(1.0 - lp / max(p.radius, 1e-4));
                cos_plus = mix(cos_plus, max(cos_plus, dot(sp / max(lp, 1e-5), V)), fo);
            }
            float2 uvm = in.uv - off;
            float dm = gbuffer.sample(smp, uvm).a;
            if (dm > 0.0) {
                float3 sm = ssao_view_pos(uvm, dm, p.tan_half_fov_y, p.aspect) - P;
                float  lm = length(sm);
                float  fo = saturate(1.0 - lm / max(p.radius, 1e-4));
                cos_minus = mix(cos_minus, max(cos_minus, dot(sm / max(lm, 1e-5), V)), fo);
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
    return pow(visibility, max(p.intensity, 0.0));
}

// Depth-aware 5x5 box blur. Weighting each tap by view-depth similarity keeps
// the noisy GTAO output from bleeding occlusion across silhouette edges.
fragment float ssao_blur_fragment(
    SsaoVtxOut in            [[stage_in]],
    texture2d<float> ao      [[texture(0)]],
    texture2d<float> gbuffer [[texture(1)]],
    sampler smp              [[sampler(0)]]
) {
    float2 texel = 1.0 / float2(ao.get_width(), ao.get_height());
    float center_depth = gbuffer.sample(smp, in.uv).a;
    if (center_depth <= 0.0) {
        return 1.0;
    }
    float sum = 0.0;
    float wsum = 0.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float2 uv = in.uv + float2(float(x), float(y)) * texel;
            float d = gbuffer.sample(smp, uv).a;
            // Depth-similarity weight; background taps (d<=0) drop out.
            float w = (d > 0.0)
                ? exp(-abs(d - center_depth) * 8.0 / max(center_depth, 1e-3))
                : 0.0;
            sum  += ao.sample(smp, uv).r * w;
            wsum += w;
        }
    }
    return (wsum > 1e-4) ? (sum / wsum) : ao.sample(smp, in.uv).r;
}
