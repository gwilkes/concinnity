#include <metal_stdlib>
using namespace metal;

struct TextUniforms { float win_width; float win_height; float pad0; float pad1; };

struct TextVtxIn {
    float2 pos   [[attribute(0)]];
    float2 uv    [[attribute(1)]];
    float3 color [[attribute(2)]];
};

struct TextVtxOut {
    float4 position [[position]];
    float2 uv;
    float3 color;
};

vertex TextVtxOut text_vertex_main(
    TextVtxIn in [[stage_in]],
    constant TextUniforms& uni [[buffer(0)]]
) {
    TextVtxOut out;
    out.position = float4(
        (in.pos.x / uni.win_width)  * 2.0 - 1.0,
        1.0 - (in.pos.y / uni.win_height) * 2.0,
        0.0, 1.0);
    out.uv    = in.uv;
    out.color = in.color;
    return out;
}

fragment float4 text_fragment_main(
    TextVtxOut in [[stage_in]],
    texture2d<float> atlas [[texture(0)]],
    sampler smp [[sampler(0)]]
) {
    // A negative u marks a solid background-box vertex (a TextLabel.background
    // quad): emit the colour directly, alpha carried through in v, no atlas
    // sample.
    if (in.uv.x < 0.0) {
        return float4(in.color, in.uv.y);
    }
    // Atlas stores a signed distance field: 0.5 = edge, >0.5 = inside, <0.5 = outside.
    // fwidth gives the screen-space derivative of d, so the smoothstep spans exactly
    // one screen pixel regardless of text scale or display density.
    float d = atlas.sample(smp, in.uv).r;
    float aa = fwidth(d);
    float a = smoothstep(0.5 - aa, 0.5 + aa, d);
    return float4(in.color, a);
}
