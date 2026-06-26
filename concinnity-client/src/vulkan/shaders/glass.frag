#version 450

// Glass panel pass - fragment shader. Mirrors ps_main in
// src/directx/shaders/glass.hlsl and glass_fragment in
// src/metal/shaders/glass.metal. Runs in the PassId::Transparent slot after
// SSR resolve and before TAA.
//
// The fragment shader discards where nearer opaque geometry occludes the pane
// (manual depth test against the main pass depth, since the transparent pass
// binds no depth attachment), refracts the pre-transparent scene snapshot,
// tints it, and reflects the local box-projected probe set (or the sky prefilter
// cube, or a white rim) along the mirror direction, blending the two by a Schlick
// Fresnel term. The pipeline straight-alpha blends the result
// (SRC_ALPHA / ONE_MINUS_SRC_ALPHA).
//
// `USE_MSAA` is injected by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth sampler type matches the resource's sample count.

layout(std140, set = 0, binding = 0) uniform TransparentViewBlock {
    mat4  vp;
    mat4  inv_vp;
    vec4  camera_pos;
    vec2  viewport;
    float time;
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound (the reflection
    // then keeps the white rim where no probe covers and no env cube exists).
    float prefilter_mip_count;
} view;

layout(std140, set = 1, binding = 0) uniform GlassParamsBlock {
    vec4  centre;  // world-space pane centre, .w unused
    vec4  normal;  // unit pane normal (facing direction), .w unused
    vec4  tint;    // colour multiplied into the refracted scene, .w unused
    float opacity;
    float refraction_strength;
    float fresnel_power;
    // 1.0 when this pane has a planar reflection slot: sample the sharp mirror
    // render (set 1 binding 1) projectively instead of the box-projected probe.
    float planar;
} params;

// This pane's planar reflection target (the scene re-rendered mirrored across the
// pane plane), bound per pane. Sampled projectively when `planar > 0.5`; a pane
// with no planar slot binds the scene snapshot here as a valid stand-in and never
// samples it (the shader gates on the flag).
layout(set = 1, binding = 1) uniform sampler2D planar_reflection;

// Pre-transparent scene snapshot (single-sample HDR) sampled for refraction.
layout(set = 0, binding = 1) uniform sampler2D scene_color;
// Main-pass depth, for the manual occlusion test. Matched to the resource's
// sample count via USE_MSAA.
#if USE_MSAA
layout(set = 0, binding = 2) uniform sampler2DMS scene_depth;
#else
layout(set = 0, binding = 2) uniform sampler2D scene_depth;
#endif

// Sky IBL prefilter cube: the reflection fallback where no probe covers the pane.
// Rides the per-frame global set (bound as set 2 here), the same set probe_common
// puts the probe set + cube array on.
layout(set = 2, binding = 5) uniform samplerCube prefilter_cube;

// The reflection-probe set (binding 7) + cube array (binding 8) + box-parallax
// sampling, substituted from probe_common.glsl ({PROBE_DESC_SET} = 2). Lets glass
// reflect the same local scene capture the forward IBL specular and the SSR/RT
// miss fallback use, instead of only the foreign sky cube.
{PROBE_COMMON}

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

    // Reflection along the mirror direction. When the planar reflection is active
    // (`planar > 0.5`: the scene re-rendered mirrored across this pane's plane)
    // sample it projectively at this fragment's own screen UV -- a flat pane is a
    // perfect mirror, so the mirrored render lands exactly under the reflector and
    // needs no distortion. Otherwise prefer the local box-projected probe set (the
    // scene capture) when the world has baked probes, else the sky prefilter cube,
    // else a white rim so a probe-less, env-less world still reads as glass. A pane
    // is smooth, so every path is sharp (mip 0).
    vec3 R = reflect(-view_dir, n);
    vec3 reflection;
    if (params.planar > 0.5) {
        reflection = texture(planar_reflection, frag_uv).rgb;
    } else if (probe_set.count > 0u) {
        reflection = probe_set_specular(world_pos, R, 0.0);
    } else if (view.prefilter_mip_count > 0.5) {
        reflection = textureLod(prefilter_cube, R, 0.0).rgb;
    } else {
        reflection = vec3(1.0);
    }

    // Schlick Fresnel (F0 = 0.04 dielectric) drives the reflection/refraction
    // blend: ~4% head-on, rising to a full mirror at grazing. `fresnel_power`
    // stays the author's grazing-rim shaping control for the opacity ramp.
    float n_dot_v = clamp(dot(n, view_dir), 0.0, 1.0);
    float rim = pow(1.0 - n_dot_v, max(params.fresnel_power, 1e-3));
    float refl_weight = clamp(0.04 + 0.96 * rim, 0.0, 1.0);
    vec3 colour = mix(refracted, reflection, refl_weight);
    float alpha = clamp(mix(params.opacity, 1.0, rim), 0.0, 1.0);

    out_color = vec4(colour, alpha);
}
