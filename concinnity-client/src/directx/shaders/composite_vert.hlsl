struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

VsOut main(uint vid : SV_VertexID)
{
    VsOut o;
    // Fullscreen triangle from vertex ids 0..2 - no vertex buffer.
    float2 pos = float2((vid == 2) ? 3.0 : -1.0,
                        (vid == 1) ? 3.0 : -1.0);
    o.sv_pos = float4(pos, 0.0, 1.0);
    // D3D texture origin is top-left, so flip Y when mapping clip space to UV.
    o.uv = float2((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return o;
}
