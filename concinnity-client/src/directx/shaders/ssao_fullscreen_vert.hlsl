// Fullscreen-triangle vertex shader for the SSAO kernel and blur passes.
// Mirrors ssao_fullscreen_vertex in src/metal/shaders/ssao.metal; the UV
// flip matches the D3D top-left texture origin so the kernel samples the
// G-buffer at pixel coordinates that line up with the main pass.

struct VsOut
{
    float4 sv_pos : SV_POSITION;
    float2 uv     : TEXCOORD0;
};

VsOut main(uint vid : SV_VertexID)
{
    float2 pos = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
    VsOut o;
    o.sv_pos = float4(pos, 0.0, 1.0);
    o.uv     = float2((pos.x + 1.0) * 0.5, 1.0 - (pos.y + 1.0) * 0.5);
    return o;
}
