// Screen-space global illumination for the D3D12 backend: a refinement of SSR.
// Two fullscreen passes share this file (one vertex entry, two fragment
// entries), mirroring src/metal/shaders/ssgi.metal:
//
//   * ssgi_gather_frag:    per pixel, cone of cosine-weighted hemisphere rays
//                          marched against the SSR pre-pass G-buffer,
//                          accumulating the lit scene colour at each on-screen
//                          hit into the off-screen `gi` target.
//   * ssgi_composite_frag: a depth-aware blur of that noisy `gi` target, which
//                          the pipeline then additively blends (ONE / ONE) into
//                          `hdr_resolve` so the near-field indirect bounce
//                          layers on top of the IBL ambient.
//
// The view-space reconstruction + projection match ssr_resolve_frag.hlsl
// (ssr_view_pos / ssr_project) byte-for-byte, so the gather agrees with the
// G-buffer the SSR pre-pass wrote (rgb = unit view normal, a = -view_z).

// b0: SSGI tunables. Layout matches gfx::render_types::SsgiParams (32 bytes).
cbuffer SsgiParams : register(b0)
{
    float intensity;
    float max_distance;
    float tan_half_fov_y;
    float aspect;
    float stride;
    float thickness;
    // Hemisphere rays per pixel + ray-march samples per ray, read as int loop
    // bounds below (carried as f32 to keep the 32-byte layout). Mirrors
    // ssgi.metal, which reads int(p.rays) / int(p.steps).
    float rays;
    float steps;
}

// t0: the lit scene radiance (gather) or the noisy gather output (composite).
// t1: the SSR pre-pass G-buffer (rgb = view normal, a = linear view depth).
Texture2D    scene_or_gi : register(t0);
Texture2D    gbuffer     : register(t1);
SamplerState smp         : register(s0);

// Origin offset along the surface normal (x stride) so a ray doesn't
// immediately self-intersect the surface it starts on.
static const float SSGI_NORMAL_BIAS = 0.5;
static const float SSGI_PI = 3.14159265359;
// Depth-aware blur footprint (composite pass): a (2R+1)^2 box weighted by depth
// similarity, so the indirect term denoises without bleeding across silhouettes.
static const int   SSGI_BLUR_RADIUS = 2;

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

// Fullscreen triangle generated from SV_VertexID 0..2, no vertex buffer.
// Matches ssr_fullscreen_vert.hlsl's UV convention.
VsOut ssgi_fullscreen_vert(uint vid : SV_VertexID)
{
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    VsOut o;
    o.sv_pos = float4(pos, 0.0, 1.0);
    o.uv     = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return o;
}

// Rebuild a view-space position from a UV and its linear (view-space) depth.
// Matches ssr_view_pos / ssao_view_pos.
float3 ssgi_view_pos(float2 uv, float depth, float tan_y, float asp)
{
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return float3(ndc.x * tan_y * asp, ndc.y * tan_y, -1.0) * depth;
}

// Project a view-space point (z < 0, in front of the camera) to a screen UV.
float2 ssgi_project(float3 q, float tan_y, float asp)
{
    float inv = 1.0 / max(-q.z, 1e-4);
    float2 ndc = float2(q.x * inv / (tan_y * asp), q.y * inv / tan_y);
    return float2(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
}

// Interleaved gradient noise: a cheap per-pixel hash in [0, 1). Decorrelates the
// hemisphere sampling spatially; the depth-aware blur + TAA clean up the
// residual high-frequency noise.
float ssgi_ign(float2 p)
{
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

// Van der Corput radical inverse (base 2), for the low-discrepancy ray set.
float ssgi_vdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 2^32
}

// Gather pass: per pixel, cast `rays` cosine-weighted hemisphere rays around
// the surface normal, screen-march each against the SSR pre-pass G-buffer, and
// accumulate the lit scene colour at each on-screen hit. Misses contribute
// nothing (the IBL ambient already covers the off-screen / sky term). The
// cosine-weighted importance sampling folds the cos theta / pdf factor away, so
// the estimate of the (albedo-free) indirect irradiance is just the mean hit
// radiance.
float4 ssgi_gather_frag(VsOut p) : SV_TARGET
{
    float4 c     = gbuffer.Sample(smp, p.uv);
    float  depth = c.a;
    if (depth <= 0.0) return float4(0.0, 0.0, 0.0, 1.0); // background / sky

    float3 N = normalize(c.xyz);
    float3 P = ssgi_view_pos(p.uv, depth, tan_half_fov_y, aspect);

    // Orthonormal basis around the view-space normal.
    float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 T  = normalize(cross(up, N));
    float3 B  = cross(N, T);

    float  jitter = ssgi_ign(p.sv_pos.xy);
    float3 origin = P + N * (stride * SSGI_NORMAL_BIAS);

    // Hemisphere rays + march samples per ray, read from the UBO (the cbuffer
    // `rays` / `steps` floors at 1). Mirrors ssgi.metal:102-103.
    int ray_count  = max(1, (int)rays);
    int step_count = max(1, (int)steps);

    float3 indirect = float3(0.0, 0.0, 0.0);
    [loop] for (int i = 0; i < ray_count; i++)
    {
        // Stratified cosine-weighted hemisphere sample, jittered per pixel.
        float u1 = (float(i) + jitter) / float(ray_count);
        float u2 = frac(ssgi_vdc(uint(i + 1)) + jitter);
        float r   = sqrt(u1);
        float phi = 2.0 * SSGI_PI * u2;
        float3 d_t = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        float3 d   = normalize(T * d_t.x + B * d_t.y + N * d_t.z);

        float3 step_v = d * stride;
        float3 q = origin;
        [loop] for (int s = 0; s < step_count; s++)
        {
            q += step_v;
            if (q.z >= 0.0) break;                       // crossed the camera plane
            float2 uv = ssgi_project(q, tan_half_fov_y, aspect);
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
            float scene_depth = gbuffer.Sample(smp, uv).a;
            if (scene_depth <= 0.0) continue;            // sky here -- keep marching
            float diff = (-q.z) - scene_depth;           // > 0: ray is behind the surface
            if (diff > 0.0 && diff < thickness)
            {
                indirect += scene_or_gi.Sample(smp, uv).rgb; // bounced radiance
                break;
            }
        }
    }

    indirect *= (1.0 / float(ray_count));
    return float4(indirect, 1.0);
}

// Composite pass: a depth-aware box blur over the noisy gather output that the
// pipeline then additively blends (ONE / ONE) into the scene, scaled by the
// authored intensity. Background pixels emit zero so the sky is untouched.
float4 ssgi_composite_frag(VsOut p) : SV_TARGET
{
    float center_depth = gbuffer.Sample(smp, p.uv).a;
    if (center_depth <= 0.0) return float4(0.0, 0.0, 0.0, 1.0);

    uint gw, gh;
    scene_or_gi.GetDimensions(gw, gh);
    float2 texel = float2(1.0 / float(max(gw, 1u)), 1.0 / float(max(gh, 1u)));

    float3 sum = float3(0.0, 0.0, 0.0);
    float  wsum = 0.0;
    [loop] for (int dy = -SSGI_BLUR_RADIUS; dy <= SSGI_BLUR_RADIUS; dy++)
    {
        for (int dx = -SSGI_BLUR_RADIUS; dx <= SSGI_BLUR_RADIUS; dx++)
        {
            float2 uv = p.uv + float2(float(dx), float(dy)) * texel;
            float d = gbuffer.Sample(smp, uv).a;
            if (d <= 0.0) continue;                      // skip background taps
            // Depth-similarity weight: taps on a different surface fall off
            // sharply so the indirect term doesn't bleed across edges.
            float dd = abs(d - center_depth);
            float w = exp2(-dd * 8.0);
            sum  += scene_or_gi.Sample(smp, uv).rgb * w;
            wsum += w;
        }
    }
    float3 gi = wsum > 0.0 ? sum / wsum : scene_or_gi.Sample(smp, p.uv).rgb;
    return float4(gi * intensity, 1.0);
}
