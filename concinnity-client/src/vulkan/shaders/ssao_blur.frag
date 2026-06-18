#version 450

// Depth-aware 5x5 box blur of the raw GTAO occlusion. Weighting each tap by
// view-depth similarity keeps the noisy kernel output from bleeding occlusion
// across silhouette edges. Mirrors ssao_blur_frag.hlsl.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out float out_ao;

layout(set = 0, binding = 0) uniform sampler2D ao_raw;
layout(set = 0, binding = 1) uniform sampler2D gbuffer;

void main() {
    vec2 tex_size = vec2(textureSize(ao_raw, 0));
    vec2 texel    = 1.0 / tex_size;
    float center_depth = texture(gbuffer, frag_uv).a;
    if (center_depth <= 0.0) {
        out_ao = 1.0;
        return;
    }
    float sum  = 0.0;
    float wsum = 0.0;
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2 uv = frag_uv + vec2(float(x), float(y)) * texel;
            float d = texture(gbuffer, uv).a;
            float wd = (d > 0.0)
                ? exp(-abs(d - center_depth) * 8.0 / max(center_depth, 1e-3))
                : 0.0;
            sum  += texture(ao_raw, uv).r * wd;
            wsum += wd;
        }
    }
    out_ao = (wsum > 1e-4) ? (sum / wsum) : texture(ao_raw, frag_uv).r;
}
