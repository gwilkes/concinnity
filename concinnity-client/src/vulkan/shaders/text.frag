#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec3 frag_color;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D atlas;

void main() {
    // A negative u marks a solid background-box vertex (a TextLabel.background
    // quad emitted by gfx::text::build_text_calls): emit the colour directly,
    // alpha carried through in v, no atlas sample. Mirrors the DirectX / Metal
    // text fragment shaders.
    if (frag_uv.x < 0.0) {
        out_color = vec4(frag_color, frag_uv.y);
        return;
    }
    // Atlas stores a signed distance field: 0.5 = edge, >0.5 = inside,
    // <0.5 = outside. fwidth gives the screen-space derivative of d, so the
    // smoothstep spans exactly one screen pixel - the glyph stays crisp at any
    // text scale or display density. Sampling d directly as alpha (as this
    // shader once did) ramps the whole distance field and reads as blurry.
    float d = texture(atlas, frag_uv).r;
    float aa = fwidth(d);
    float a = smoothstep(0.5 - aa, 0.5 + aa, d);
    out_color = vec4(frag_color, a);
}
