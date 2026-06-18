Texture2D    src  : register(t0);
SamplerState samp : register(s0);

struct PsIn
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

// 9-tap tent upsample filter (weights 1 2 1 / 2 4 2 / 1 2 1, /16).
float3 upsample_tent(Texture2D s, SamplerState samp, float2 uv, float2 texel)
{
    float3 sum = float3(0.0, 0.0, 0.0);
    sum += s.SampleLevel(samp, uv + texel * float2(-1.0, -1.0), 0.0).rgb * 1.0;
    sum += s.SampleLevel(samp, uv + texel * float2( 0.0, -1.0), 0.0).rgb * 2.0;
    sum += s.SampleLevel(samp, uv + texel * float2( 1.0, -1.0), 0.0).rgb * 1.0;
    sum += s.SampleLevel(samp, uv + texel * float2(-1.0,  0.0), 0.0).rgb * 2.0;
    sum += s.SampleLevel(samp, uv + texel * float2( 0.0,  0.0), 0.0).rgb * 4.0;
    sum += s.SampleLevel(samp, uv + texel * float2( 1.0,  0.0), 0.0).rgb * 2.0;
    sum += s.SampleLevel(samp, uv + texel * float2(-1.0,  1.0), 0.0).rgb * 1.0;
    sum += s.SampleLevel(samp, uv + texel * float2( 0.0,  1.0), 0.0).rgb * 2.0;
    sum += s.SampleLevel(samp, uv + texel * float2( 1.0,  1.0), 0.0).rgb * 1.0;
    return sum * (1.0 / 16.0);
}

float4 main(PsIn p) : SV_TARGET
{
    uint w, h;
    src.GetDimensions(w, h);
    float2 texel = 1.0 / float2((float)w, (float)h);
    return float4(upsample_tent(src, samp, p.uv, texel), 1.0);
}
