#version 450
layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 0) uniform sampler2D src;

// 9-tap tent upsample filter (weights 1 2 1 / 2 4 2 / 1 2 1, /16). The result
// is additively blended onto the destination mip by the pipeline blend state.
vec3 upsample_tent(vec2 uv, vec2 texel) {
    vec3 sum = vec3(0.0);
    sum += texture(src, uv + texel * vec2(-1.0, -1.0)).rgb * 1.0;
    sum += texture(src, uv + texel * vec2( 0.0, -1.0)).rgb * 2.0;
    sum += texture(src, uv + texel * vec2( 1.0, -1.0)).rgb * 1.0;
    sum += texture(src, uv + texel * vec2(-1.0,  0.0)).rgb * 2.0;
    sum += texture(src, uv + texel * vec2( 0.0,  0.0)).rgb * 4.0;
    sum += texture(src, uv + texel * vec2( 1.0,  0.0)).rgb * 2.0;
    sum += texture(src, uv + texel * vec2(-1.0,  1.0)).rgb * 1.0;
    sum += texture(src, uv + texel * vec2( 0.0,  1.0)).rgb * 2.0;
    sum += texture(src, uv + texel * vec2( 1.0,  1.0)).rgb * 1.0;
    return sum * (1.0 / 16.0);
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(src, 0));
    out_color = vec4(upsample_tent(frag_uv, texel), 1.0);
}
