// Volumetric fog pass - fullscreen-triangle vertex shader. Mirrors
// fog_vertex in src/metal/shaders/fog.metal. The three vertices form a single
// triangle that covers the entire screen so the pixel shader runs once per
// framebuffer pixel; no vertex buffer is needed.

struct VsOut
{
    float4 sv_pos : SV_POSITION;
};

VsOut main(uint vid : SV_VertexID)
{
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2( 3.0, -1.0),
        float2(-1.0,  3.0)
    };
    VsOut o;
    o.sv_pos = float4(positions[vid], 0.0, 1.0);
    return o;
}
