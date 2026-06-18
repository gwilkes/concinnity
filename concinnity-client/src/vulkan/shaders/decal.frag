#version 450

// Projected decal pass - fragment shader. Mirrors
// src/directx/shaders/decal_frag.hlsl and src/metal/shaders/decal.metal.
// Reconstructs the world-space sample point at each rasterised pixel from
// the main pass's depth attachment, transforms it back into decal-local
// space, clips against the unit box [-0.5, 0.5]^3, and stamps the decal
// texture × tint into the resolved HDR target. The pipeline's blend
// state alpha-composites the result on top of the scene.
//
// `USE_MSAA` is injected by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth sampler type matches the underlying resource.

layout(std140, set = 0, binding = 0) uniform DecalViewBlock {
    mat4  vp;
    mat4  inv_vp;
    vec2  viewport;
    vec2  _pad;
} view;

layout(std140, set = 0, binding = 1) uniform DecalParamsBlock {
    mat4 model;
    mat4 inv_model;
    vec4 tint;
    vec4 fade;   // .x = fade_pow
} params;

#if USE_MSAA
layout(set = 0, binding = 2) uniform sampler2DMS scene_depth;
#else
layout(set = 0, binding = 2) uniform sampler2D scene_depth;
#endif

// Per-decal albedo lives in its own set so swapping decals only rebinds
// set 1, not the per-frame set 0.
layout(set = 1, binding = 0) uniform sampler2D decal_tex;

layout(location = 0) out vec4 out_color;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(view.viewport.x) || pixel.y >= int(view.viewport.y))
    {
        discard;
    }
    // Sample 0 of the MSAA depth (or the single-sample depth) is the
    // cleared / "no geometry" sentinel when the main pass left the pixel
    // empty. A value of exactly 1.0 means nothing to project onto.
    float depth = texelFetch(scene_depth, pixel, 0).r;
    if (depth >= 1.0) {
        discard;
    }

    // Reconstruct world-space at this pixel via the inverse VP. The pass
    // uses the same negative-height viewport as the main pass, so
    // gl_FragCoord.y = 0 sits at the top of the framebuffer - same Y
    // convention as DirectX, hence the same flip here.
    vec2 ndc_xy = (gl_FragCoord.xy / view.viewport) * 2.0 - 1.0;
    ndc_xy.y    = -ndc_xy.y;
    vec4 clip   = vec4(ndc_xy, depth, 1.0);
    vec4 world  = view.inv_vp * clip;
    world      /= world.w;

    // Decal-local clip against the unit box.
    vec4 local = params.inv_model * world;
    vec3 ab    = abs(local.xyz);
    if (ab.x > 0.5 || ab.y > 0.5 || ab.z > 0.5) {
        discard;
    }

    // Soft fade along the projection axis (local +Y) so the stamp doesn't
    // show a hard band where the surface tilts away from the projection
    // plane. Alpha rolls off as |local.y| approaches 0.5.
    float fade = clamp(1.0 - (ab.y * 2.0), 0.0, 1.0);
    fade       = pow(fade, max(params.fade.x, 1.0));

    // Sample the decal texture on local X-Z; UV in [0, 1] with V=0 at top
    // to match the engine's other textures.
    vec2 uv = local.xz + 0.5;
    uv.y    = 1.0 - uv.y;
    vec4 tex = texture(decal_tex, uv);

    out_color.rgb = tex.rgb * params.tint.rgb;
    out_color.a   = tex.a * params.tint.a * fade;
}
