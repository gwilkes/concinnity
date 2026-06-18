Texture2D    src  : register(t0);
SamplerState samp : register(s0);

struct PsIn
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

// 13-tap downsample (Jimenez/CoD). With `karis` set, the taps are grouped into
// five overlapping 2x2 boxes and luma-weighted so a single firefly does not
// dominate - used only on the first (prefilter) downsample.
float bloom_luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }
float karis_weight(float3 c) { return 1.0 / (1.0 + bloom_luma(c)); }

float3 downsample_13(Texture2D src, SamplerState samp, float2 uv, float2 texel, bool karis)
{
    float3 a = src.SampleLevel(samp, uv + texel * float2(-2.0, -2.0), 0.0).rgb;
    float3 b = src.SampleLevel(samp, uv + texel * float2( 0.0, -2.0), 0.0).rgb;
    float3 c = src.SampleLevel(samp, uv + texel * float2( 2.0, -2.0), 0.0).rgb;
    float3 d = src.SampleLevel(samp, uv + texel * float2(-2.0,  0.0), 0.0).rgb;
    float3 e = src.SampleLevel(samp, uv + texel * float2( 0.0,  0.0), 0.0).rgb;
    float3 f = src.SampleLevel(samp, uv + texel * float2( 2.0,  0.0), 0.0).rgb;
    float3 g = src.SampleLevel(samp, uv + texel * float2(-2.0,  2.0), 0.0).rgb;
    float3 h = src.SampleLevel(samp, uv + texel * float2( 0.0,  2.0), 0.0).rgb;
    float3 i = src.SampleLevel(samp, uv + texel * float2( 2.0,  2.0), 0.0).rgb;
    float3 j = src.SampleLevel(samp, uv + texel * float2(-1.0, -1.0), 0.0).rgb;
    float3 k = src.SampleLevel(samp, uv + texel * float2( 1.0, -1.0), 0.0).rgb;
    float3 l = src.SampleLevel(samp, uv + texel * float2(-1.0,  1.0), 0.0).rgb;
    float3 m = src.SampleLevel(samp, uv + texel * float2( 1.0,  1.0), 0.0).rgb;

    if (karis)
    {
        float3 g0 = (a + b + d + e) * (0.125 / 4.0);
        float3 g1 = (b + c + e + f) * (0.125 / 4.0);
        float3 g2 = (d + e + g + h) * (0.125 / 4.0);
        float3 g3 = (e + f + h + i) * (0.125 / 4.0);
        float3 g4 = (j + k + l + m) * (0.5 / 4.0);
        g0 *= karis_weight(g0);
        g1 *= karis_weight(g1);
        g2 *= karis_weight(g2);
        g3 *= karis_weight(g3);
        g4 *= karis_weight(g4);
        return g0 + g1 + g2 + g3 + g4;
    }

    float3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
    return result;
}

float4 main(PsIn p) : SV_TARGET
{
    uint w, h;
    src.GetDimensions(w, h);
    float2 texel = 1.0 / float2((float)w, (float)h);
    return float4(downsample_13(src, samp, p.uv, texel, false), 1.0);
}
