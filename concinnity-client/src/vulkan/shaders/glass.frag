#version 450

// Glass panel pass - fragment shader. Mirrors ps_main in
// src/directx/shaders/glass.hlsl and glass_fragment in
// src/metal/shaders/glass.metal. Runs in the PassId::Transparent slot after
// SSR resolve and before TAA.
//
// The fragment shader discards where nearer opaque geometry occludes the pane
// (manual depth test against the main pass depth, since the transparent pass
// binds no depth attachment), refracts the pre-transparent scene snapshot,
// tints it, and adds a Schlick-Fresnel rim. The pipeline straight-alpha blends
// the result (SRC_ALPHA / ONE_MINUS_SRC_ALPHA).
//
// `USE_MSAA` is injected by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth sampler type matches the resource's sample count.

layout(std140, set = 0, binding = 0) uniform TransparentViewBlock {
    mat4  vp;
    mat4  inv_vp;
    vec4  camera_pos;
    vec2  viewport;
    float time;
    float _pad;
} view;

layout(std140, set = 1, binding = 0) uniform GlassParamsBlock {
    vec4  centre;  // world-space pane centre, .w unused
    vec4  normal;  // unit pane normal (facing direction), .w unused
    vec4  tint;    // colour multiplied into the refracted scene, .w unused
    float opacity;
    float refraction_strength;
    float fresnel_power;
    float _pad1;
} params;

// Pre-transparent scene snapshot (single-sample HDR) sampled for refraction.
layout(set = 0, binding = 1) uniform sampler2D scene_color;
// Main-pass depth, for the manual occlusion test. Matched to the resource's
// sample count via USE_MSAA.
#if USE_MSAA
layout(set = 0, binding = 2) uniform sampler2DMS scene_depth;
#else
layout(set = 0, binding = 2) uniform sampler2D scene_depth;
#endif

layout(location = 0) in vec3 world_pos;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 view_dir = normalize(view.camera_pos.xyz - world_pos);
    // Two-sided: orient the normal toward the viewer so a pane lit from behind
    // still Fresnels correctly.
    vec3 n = normalize(params.normal.xyz);
    if (dot(n, view_dir) < 0.0) {
        n = -n;
    }

    vec2 vp_dim = max(view.viewport, vec2(1.0, 1.0));
    vec2 frag_uv = gl_FragCoord.xy / vp_dim;

    // Manual depth occlusion: discard where the scene depth at this pixel is
    // nearer than the pane (the pass has no hardware depth test). Vulkan depth
    // is [0, 1] with 0 = near, matching gl_FragCoord.z, so a smaller stored
    // value means opaque geometry sits in front of the pane. The main pass
    // rasterised this depth under the same negative-height viewport, so
    // gl_FragCoord lines up with the stored texel (matches the decal pass).
    ivec2 pixel = min(ivec2(gl_FragCoord.xy), ivec2(vp_dim) - ivec2(1, 1));
    float scene_self_depth = texelFetch(scene_depth, pixel, 0).r;
    if (scene_self_depth < gl_FragCoord.z) {
        discard;
    }

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    vec2 refract_uv = clamp(frag_uv + n.xy * params.refraction_strength,
                            vec2(0.001), vec2(0.999));
    vec3 refracted = texture(scene_color, refract_uv).rgb * params.tint.rgb;

    // Schlick-Fresnel rim: brighter + more opaque at grazing angles.
    float n_dot_v = clamp(dot(n, view_dir), 0.0, 1.0);
    float fresnel = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));

    vec3 rim = vec3(1.0);
    vec3 colour = mix(refracted, rim, fresnel * 0.5);
    float alpha = clamp(mix(params.opacity, 1.0, fresnel), 0.0, 1.0);

    out_color = vec4(colour, alpha);
}
