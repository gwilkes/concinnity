#version 450

// Volumetric fog fullscreen sampler. Mirrors fog_fragment in
// src/metal/shaders/fog.metal and src/directx/shaders/fog_frag.hlsl. The
// per-frame fog_froxel.comp compute kernel populates a screen-aligned 3D
// RGBA16F volume of (scattered_rgb, 1 - T) across the view frustum; this pass
// samples that volume by (screen_uv, view_z) and emits the pair so the
// pipeline's `over` blend resolves to `final = scene * T + scattered`.
//
// `USE_MSAA` is injected by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth sampler declaration matches the underlying
// resource's sample count.

layout(std140, set = 0, binding = 0) uniform FogBlock {
    mat4  inv_vp;            // offset   0
    vec4  color;             // offset  64
    vec3  cam_pos;           // offset  80
    float _pad0;             // offset  92
    vec3  sun_dir;           // offset  96
    float _pad1;             // offset 108
    vec3  sun_color;         // offset 112
    float _pad2;             // offset 124
    float density;           // offset 128
    float height_falloff;    // offset 132
    float height_reference;  // offset 136
    float max_distance;      // offset 140
    float phase_g;           // offset 144
    float ambient;           // offset 148
    vec2  viewport;          // offset 152
    float inv_max_distance;  // offset 160
    float _pad3a;            // offset 164
    float _pad3b;            // offset 168
    float _pad3c;            // offset 172
} fog;                       // total 176 B, matches gfx::render_types::FogParams

layout(std140, set = 0, binding = 2) uniform FogFroxelBlock {
    mat4  view;              // offset  0 (world -> view)
    uvec3 froxel_dims;       // offset 64
    uint  _pad_align;        // offset 76
    float z_near;            // offset 80
    float z_far;             // offset 84
    vec2  _pad_ff;           // offset 88
} froxel;                    // total 96 B, matches gfx::render_types::FogFroxelParams

#if USE_MSAA
layout(set = 0, binding = 1) uniform sampler2DMS scene_depth;
#else
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
#endif

// Froxel volume populated by fog_froxel.comp. Sampled trilinearly: the stored
// (scattered, 1 - T) is camera->slice integrated, so the sample IS the blend
// pair.
layout(set = 0, binding = 3) uniform sampler3D fog_volume;

layout(location = 0) out vec4 out_color;

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(fog.viewport.x) || pixel.y >= int(fog.viewport.y))
    {
        discard;
    }
    float depth = texelFetch(scene_depth, pixel, 0).r;

    // Screen UV + Y-flipped NDC (the main pass writes depth with a negative-
    // height viewport, so framebuffer-Y = 0 is clip-Y = +1).
    vec2 uv     = gl_FragCoord.xy / fog.viewport;
    vec2 ndc_xy = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    // Reconstruct view-space depth at the pixel. depth == 1.0 (skybox / never-
    // written) maps to the far edge of the volume so the sky takes fog
    // integrated across the whole volume.
    float view_z;
    if (depth < 1.0) {
        vec4 clip  = vec4(ndc_xy, depth, 1.0);
        vec4 world = fog.inv_vp * clip;
        world /= world.w;
        view_z = -(froxel.view * vec4(world.xyz, 1.0)).z;
    } else {
        view_z = froxel.z_far;
    }

    // Map view_z onto a normalised volume W. Clamp so the skybox + anything
    // past the volume's far edge sample the fully-integrated last slice.
    float z01 = clamp((view_z - froxel.z_near) / max(froxel.z_far - froxel.z_near, 1e-4),
                      0.0, 1.0);

    // Trilinear sample. The volume already stores camera->slice integrated
    // (scattered, 1 - T), so the sample IS the output blend pair.
    out_color = texture(fog_volume, vec3(uv, z01));
}
