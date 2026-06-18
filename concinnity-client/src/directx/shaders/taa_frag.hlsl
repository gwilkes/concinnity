Texture2D    scene_tex      : register(t0);
Texture2D    velocity_tex   : register(t1);
Texture2D    history_tex    : register(t2);
SamplerState linear_sampler : register(s0);

// 0 on the first frame - history is then ignored.
cbuffer TaaBlock : register(b0)
{
    float history_valid;
}

struct PsIn
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

// History blend weight. 0.9 keeps 90% of the accumulated history each frame -
// roughly a 10-frame exponential moving average once converged.
static const float TAA_BLEND = 0.9;
// Standard deviations of the neighbourhood the history is allowed to span.
static const float TAA_VARIANCE_GAMMA = 1.0;
// Finite ceiling every sampled colour is clamped to (largest finite half).
static const float TAA_HDR_CLAMP = 65504.0;

// Scrub a non-finite sample: NaN -> 0, +Inf -> the HDR ceiling, negative -> 0.
float3 taa_sanitize(float3 c)
{
    c = isnan(c) ? float3(0.0, 0.0, 0.0) : c;
    return clamp(c, 0.0, TAA_HDR_CLAMP);
}

// RGB <-> YCoCg. The clip box is built in YCoCg so its luma axis aligns with
// perceived error - a tighter, better-oriented box than an RGB AABB.
float3 rgb_to_ycocg(float3 c)
{
    return float3(
         0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
         0.5  * c.r            - 0.5  * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
float3 ycocg_to_rgb(float3 c)
{
    float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}

// Clip the history sample to the neighbourhood box along the line toward the
// box centre (Karis 2014) - preserves colour direction, so hue shifts less.
float3 clip_to_aabb(float3 bmin, float3 bmax, float3 hist)
{
    float3 center = 0.5 * (bmax + bmin);
    float3 extent = 0.5 * (bmax - bmin) + 1e-5;
    float3 v = hist - center;
    float3 a = abs(v) / extent;
    float ma = max(a.x, max(a.y, a.z));
    return (ma > 1.0) ? (center + v / ma) : hist;
}

float4 main(PsIn p) : SV_TARGET
{
    float2 uv = p.uv;
    uint w, h;
    scene_tex.GetDimensions(w, h);
    float2 texel = 1.0 / float2((float)w, (float)h);
    float3 cur = taa_sanitize(scene_tex.SampleLevel(linear_sampler, uv, 0.0).rgb);

    // 3x3 neighbourhood statistics in YCoCg; the reprojected history is clipped
    // to mean +/- gamma*stddev.
    float3 m1 = float3(0.0, 0.0, 0.0);
    float3 m2 = float3(0.0, 0.0, 0.0);
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            float3 s = taa_sanitize(
                scene_tex.SampleLevel(linear_sampler, uv + float2(dx, dy) * texel, 0.0).rgb);
            float3 c = rgb_to_ycocg(s);
            m1 += c;
            m2 += c * c;
        }
    }
    float3 mean  = m1 / 9.0;
    float3 sigma = sqrt(max(m2 / 9.0 - mean * mean, float3(0.0, 0.0, 0.0)));
    float3 bmin  = mean - TAA_VARIANCE_GAMMA * sigma;
    float3 bmax  = mean + TAA_VARIANCE_GAMMA * sigma;

    // The velocity pre-pass stored each surface's screen-space motion as the
    // offset that maps a current-frame UV onto its previous-frame UV.
    float2 motion  = velocity_tex.SampleLevel(linear_sampler, uv, 0.0).rg;
    float2 prev_uv = uv + motion;
    bool on_screen = all(prev_uv >= 0.0) && all(prev_uv <= 1.0);

    float3 hist = rgb_to_ycocg(
        taa_sanitize(history_tex.SampleLevel(linear_sampler, prev_uv, 0.0).rgb));
    hist = clip_to_aabb(bmin, bmax, hist);

    // Accumulate only when there is valid, on-screen history; otherwise the
    // current frame passes straight through (first frame, off-screen).
    float alpha = (history_valid > 0.5 && on_screen) ? TAA_BLEND : 0.0;
    return float4(lerp(cur, ycocg_to_rgb(hist), alpha), 1.0);
}
