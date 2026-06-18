// Particle render pipeline - fragment shader. Mirrors `particle_fragment` in
// src/metal/shaders/particle.metal. Samples the emitter's albedo texture,
// multiplies it by the age-interpolated gradient colour the vertex stage
// emitted, and lets the blend state composite the result into the resolved
// HDR target.

#pragma pack_matrix(column_major)

struct PsIn
{
    float4 sv_pos       : SV_POSITION;
    float2 uv           : TEXCOORD0;
    float4 color        : TEXCOORD1;
    float  discard_flag : TEXCOORD2;
};

Texture2D<float4> albedo : register(t1);
SamplerState      samp   : register(s0);

float4 main(PsIn i) : SV_Target0
{
    if (i.discard_flag > 0.5)
    {
        discard;
    }
    float2 uv = float2(i.uv.x, 1.0 - i.uv.y);
    float4 sampled = albedo.Sample(samp, uv);
    float4 c;
    c.rgb = sampled.rgb * i.color.rgb;
    c.a   = sampled.a   * i.color.a;
    return c;
}
