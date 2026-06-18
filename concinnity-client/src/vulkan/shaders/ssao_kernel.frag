#version 450

// GTAO horizon-search kernel. Reads the SSAO pre-pass G-buffer (view normal +
// linear view depth) and writes per-pixel visibility into an R8 occlusion
// target. Translated 1:1 from src/directx/shaders/ssao_kernel_frag.hlsl,
// which in turn mirrors src/metal/shaders/ssao.metal::ssao_fragment.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out float out_ao;

layout(push_constant) uniform SsaoParamsBlock {
    float radius;
    float intensity;
    float tan_half_fov_y;
    float aspect;
} params;

layout(set = 0, binding = 0) uniform sampler2D gbuffer;

const int   SSAO_SLICES  = 3;
const int   SSAO_STEPS   = 6;
const float SSAO_PI      = 3.14159265359;
const float SSAO_HALF_PI = 1.57079632679;
const float SSAO_MAX_UV  = 0.2;

vec3 ssao_view_pos(vec2 uv, float depth, float tan_half_y, float aspect_v) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return vec3(ndc.x * tan_half_y * aspect_v, ndc.y * tan_half_y, -1.0) * depth;
}

void main() {
    vec4 c = texture(gbuffer, frag_uv);
    float depth = c.a;
    if (depth <= 0.0) {
        out_ao = 1.0;
        return;
    }

    vec3 N = normalize(c.xyz);
    vec3 P = ssao_view_pos(frag_uv, depth, params.tan_half_fov_y, params.aspect);
    vec3 V = normalize(-P);

    float radius_uv = params.radius / max(2.0 * params.tan_half_fov_y * depth, 1e-4);
    radius_uv = min(radius_uv, SSAO_MAX_UV);

    // Interleaved gradient noise: per-pixel slice rotation + step jitter.
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));

    float visibility = 0.0;
    for (int s = 0; s < SSAO_SLICES; s++) {
        float ang = (float(s) + ign) * (SSAO_PI / float(SSAO_SLICES));
        vec2 dir = vec2(cos(ang), sin(ang));

        vec3 dir_vs  = normalize(vec3(dir, 0.0));
        vec3 plane_n = normalize(cross(dir_vs, V));
        vec3 proj_n  = N - plane_n * dot(N, plane_n);
        float proj_len = length(proj_n);
        if (proj_len < 1e-4) {
            continue;
        }
        vec3 tangent = cross(plane_n, V);
        float n = atan(dot(proj_n, tangent), dot(proj_n, V));

        float cos_plus  = -1.0;
        float cos_minus = -1.0;
        for (int step = 1; step <= SSAO_STEPS; step++) {
            float t = (float(step) - 0.5 + ign) / float(SSAO_STEPS);
            vec2 off = dir * radius_uv * t;

            vec2 uvp = frag_uv + off;
            float dp = texture(gbuffer, uvp).a;
            if (dp > 0.0) {
                vec3 sp = ssao_view_pos(uvp, dp, params.tan_half_fov_y, params.aspect) - P;
                float lp = length(sp);
                float fo = clamp(1.0 - lp / max(params.radius, 1e-4), 0.0, 1.0);
                cos_plus = mix(cos_plus, max(cos_plus, dot(sp / max(lp, 1e-5), V)), fo);
            }
            vec2 uvm = frag_uv - off;
            float dm = texture(gbuffer, uvm).a;
            if (dm > 0.0) {
                vec3 sm = ssao_view_pos(uvm, dm, params.tan_half_fov_y, params.aspect) - P;
                float lm = length(sm);
                float fo = clamp(1.0 - lm / max(params.radius, 1e-4), 0.0, 1.0);
                cos_minus = mix(cos_minus, max(cos_minus, dot(sm / max(lm, 1e-5), V)), fo);
            }
        }

        float h1 = -acos(clamp(cos_minus, -1.0, 1.0));
        float h2 =  acos(clamp(cos_plus,  -1.0, 1.0));
        h1 = n + max(h1 - n, -SSAO_HALF_PI);
        h2 = n + min(h2 - n,  SSAO_HALF_PI);
        float sin_n = sin(n);
        float cos_n = cos(n);
        float a1 = 0.25 * (-cos(2.0 * h1 - n) + cos_n + 2.0 * h1 * sin_n);
        float a2 = 0.25 * (-cos(2.0 * h2 - n) + cos_n + 2.0 * h2 * sin_n);
        visibility += proj_len * (a1 + a2);
    }

    visibility = clamp(visibility / float(SSAO_SLICES), 0.0, 1.0);
    out_ao = pow(visibility, max(params.intensity, 0.0));
}
