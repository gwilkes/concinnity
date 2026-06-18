cbuffer TextPush : register(b0)
{
    float win_width;
    float win_height;
    float _pad0;
    float _pad1;
}

struct VsIn
{
    float2 pos   : POSITION;
    float2 uv    : TEXCOORD0;
    float3 color : COLOR;
};

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
    float3 color  : TEXCOORD1;
};

VsOut main(VsIn v)
{
    VsOut o;
    float nx =  (v.pos.x / win_width)  * 2.0 - 1.0;
    float ny = -((v.pos.y / win_height) * 2.0 - 1.0);
    o.sv_pos = float4(nx, ny, 0.0, 1.0);
    o.uv     = v.uv;
    o.color  = v.color;
    return o;
}
