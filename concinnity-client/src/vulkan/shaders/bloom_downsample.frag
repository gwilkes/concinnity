#version 450
layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;
layout(set = 0, binding = 0) uniform sampler2D src;

// 13-tap downsample (Jimenez/CoD). With `karis` set, the taps are grouped into
// five overlapping 2x2 boxes and luma-weighted so a single firefly does not
// dominate - used only on the first (prefilter) downsample.
float bloom_luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float karis_weight(vec3 c) { return 1.0 / (1.0 + bloom_luma(c)); }

vec3 downsample_13(sampler2D src, vec2 uv, vec2 texel, bool karis) {
    vec3 a = texture(src, uv + texel * vec2(-2.0, -2.0)).rgb;
    vec3 b = texture(src, uv + texel * vec2( 0.0, -2.0)).rgb;
    vec3 c = texture(src, uv + texel * vec2( 2.0, -2.0)).rgb;
    vec3 d = texture(src, uv + texel * vec2(-2.0,  0.0)).rgb;
    vec3 e = texture(src, uv + texel * vec2( 0.0,  0.0)).rgb;
    vec3 f = texture(src, uv + texel * vec2( 2.0,  0.0)).rgb;
    vec3 g = texture(src, uv + texel * vec2(-2.0,  2.0)).rgb;
    vec3 h = texture(src, uv + texel * vec2( 0.0,  2.0)).rgb;
    vec3 i = texture(src, uv + texel * vec2( 2.0,  2.0)).rgb;
    vec3 j = texture(src, uv + texel * vec2(-1.0, -1.0)).rgb;
    vec3 k = texture(src, uv + texel * vec2( 1.0, -1.0)).rgb;
    vec3 l = texture(src, uv + texel * vec2(-1.0,  1.0)).rgb;
    vec3 m = texture(src, uv + texel * vec2( 1.0,  1.0)).rgb;

    if (karis) {
        vec3 g0 = (a + b + d + e) * (0.125 / 4.0);
        vec3 g1 = (b + c + e + f) * (0.125 / 4.0);
        vec3 g2 = (d + e + g + h) * (0.125 / 4.0);
        vec3 g3 = (e + f + h + i) * (0.125 / 4.0);
        vec3 g4 = (j + k + l + m) * (0.5 / 4.0);
        g0 *= karis_weight(g0);
        g1 *= karis_weight(g1);
        g2 *= karis_weight(g2);
        g3 *= karis_weight(g3);
        g4 *= karis_weight(g4);
        return g0 + g1 + g2 + g3 + g4;
    }

    vec3 result = e * 0.125;
    result += (a + c + g + i) * 0.03125;
    result += (b + d + f + h) * 0.0625;
    result += (j + k + l + m) * 0.125;
    return result;
}

void main() {
    vec2 texel = 1.0 / vec2(textureSize(src, 0));
    out_color = vec4(downsample_13(src, frag_uv, texel, false), 1.0);
}
