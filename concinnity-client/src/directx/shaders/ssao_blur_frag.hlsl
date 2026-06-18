// Depth-aware 5x5 box blur of the raw GTAO occlusion. Weighting each tap by
// view-depth similarity keeps the noisy kernel output from bleeding occlusion
// across silhouette edges. Mirrors ssao_blur_fragment in
// src/metal/shaders/ssao.metal.

Texture2D    ao         : register(t0);
Texture2D    gbuffer    : register(t1);
SamplerState smp        : register(s0);

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

float main(VsOut p) : SV_TARGET
{
    uint w, h;
    ao.GetDimensions(w, h);
    float2 texel = 1.0 / float2(float(w), float(h));
    float center_depth = gbuffer.Sample(smp, p.uv).a;
    if (center_depth <= 0.0)
    {
        return 1.0;
    }
    float sum = 0.0;
    float wsum = 0.0;
    [unroll] for (int y = -2; y <= 2; y++)
    [unroll] for (int x = -2; x <= 2; x++)
    {
        float2 uv = p.uv + float2(float(x), float(y)) * texel;
        float d = gbuffer.Sample(smp, uv).a;
        float wd = (d > 0.0)
            ? exp(-abs(d - center_depth) * 8.0 / max(center_depth, 1e-3))
            : 0.0;
        sum  += ao.Sample(smp, uv).r * wd;
        wsum += wd;
    }
    return (wsum > 1e-4) ? (sum / wsum) : ao.Sample(smp, p.uv).r;
}
