#version 450

// Reflection composite, pass 2 (full resolution): per pixel lerps the FULL-RES
// sharp reflection against the upsampled half-res blur by roughness, then
// composites the result over the scene by the blended weight. A near-mirror
// (roughness ~0) reads the sharp full-res tap so it stays razor-sharp; a rough
// surface reads the cheap upsampled blur, which is low-frequency anyway. Ports
// src/metal/shaders/reflection_composite.metal::reflection_composite_fragment.

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

// reflection: the full-res resolve target (rgb = radiance, a = weight). scene: the
// base HDR scene. gbuffer: normal+depth, with .a = linear depth. rough_tex: the
// G-buffer roughness. blur: the reduced-resolution blur from pass 1.
layout(set = 0, binding = 0) uniform sampler2D reflection;
layout(set = 0, binding = 1) uniform sampler2D scene;
layout(set = 0, binding = 2) uniform sampler2D gbuffer;
layout(set = 0, binding = 3) uniform sampler2D rough_tex;
layout(set = 0, binding = 4) uniform sampler2D blur_tex;

// Matches the resolve gloss gate; keep in sync with ssr_resolve.frag /
// rt_reflections.frag / reflection_blur.frag.
const float REFLECTION_ROUGHNESS_CUT = 0.6;

void main() {
    vec3  base  = texture(scene, frag_uv).rgb;
    float depth = texture(gbuffer, frag_uv).a;
    vec4  c     = texture(reflection, frag_uv);
    // Background, or a pixel that does not reflect (weight 0): keep the scene. A
    // tiny weight is still honoured so the composite is exact at the edges.
    if (depth <= 0.0 || c.a <= 0.0) {
        out_color = vec4(mix(base, c.rgb, c.a), 1.0);
        return;
    }

    // 0 at a mirror -> sharp full-res tap; 1 at the cut -> cheap upsampled blur.
    float t = clamp(texture(rough_tex, frag_uv).r / REFLECTION_ROUGHNESS_CUT, 0.0, 1.0);
    vec4  b = texture(blur_tex, frag_uv);

    vec3  reflected = mix(c.rgb, b.rgb, t);
    float weight    = mix(c.a, b.a, t);
    out_color = vec4(mix(base, reflected, weight), 1.0);
}
