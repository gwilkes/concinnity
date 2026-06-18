#pragma pack_matrix(column_major)

// Volumetric fog fullscreen sampler. Mirrors fog_fragment in
// src/metal/shaders/fog.metal. The per-frame fog_froxel.hlsl compute kernel
// populates a screen-aligned 3D `RGBA16Float` volume of `(scattered_rgb,
// 1 - T)` across the view frustum; this pass samples that volume by
// `(screen_uv, view_z)` and emits the pair so the pipeline's `over` blend
// resolves to `final = scene * T + scattered`.
//
// `USE_MSAA` is defined by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth SRV declaration and load call match the underlying
// resource's sample count.

cbuffer FogParams : register(b0)
{
    float4x4 inv_vp;             // offset   0
    float4   color;              // offset  64
    float3   cam_pos;            // offset  80
    float    _pad0;              // offset  92
    float3   sun_dir;            // offset  96
    float    _pad1;              // offset 108
    float3   sun_color;          // offset 112
    float    _pad2;              // offset 124
    float    density;            // offset 128
    float    height_falloff;     // offset 132
    float    height_reference;   // offset 136
    float    max_distance;       // offset 140
    float    phase_g;            // offset 144
    float    ambient;            // offset 148
    float2   viewport;           // offset 152
    float    inv_max_distance;   // offset 160
    float    _pad3a;             // offset 164
    float    _pad3b;             // offset 168
    float    _pad3c;             // offset 172
}                                // 176 B - matches gfx::render_types::FogParams

cbuffer FogFroxelParams : register(b1)
{
    float4x4 view;               // offset  0 (world → view)
    uint3    froxel_dims;        // offset 64
    uint     _pad_align;         // offset 76
    float    z_near;             // offset 80
    float    z_far;              // offset 84
    float2   _pad_ff;            // offset 88
}                                // 96 B - matches gfx::render_types::FogFroxelParams

#if USE_MSAA
Texture2DMS<float> scene_depth : register(t0);
#else
Texture2D<float>   scene_depth : register(t0);
#endif
Texture3D<float4>  volume      : register(t1);

SamplerState volume_samp : register(s0);

struct VsOut
{
    float4 sv_pos : SV_POSITION;
};

float4 main(VsOut p) : SV_TARGET
{
    int2 pixel = int2(p.sv_pos.xy);
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(viewport.x) || pixel.y >= int(viewport.y))
    {
        discard;
    }
#if USE_MSAA
    float depth = scene_depth.Load(pixel, 0);
#else
    float depth = scene_depth.Load(int3(pixel, 0));
#endif

    // D3D has y=0 at the top of the framebuffer but +y is up in NDC, so flip y.
    float2 uv     = p.sv_pos.xy / viewport;
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    // Reconstruct view-space depth at the pixel. depth == 1.0 (skybox / never-
    // written) maps to the far edge of the volume so the sky takes fog
    // integrated across the whole volume.
    float view_z;
    if (depth < 1.0)
    {
        float4 clip  = float4(ndc_xy, depth, 1.0);
        float4 world = mul(inv_vp, clip);
        world /= world.w;
        view_z = -mul(view, float4(world.xyz, 1.0)).z;
    }
    else
    {
        view_z = z_far;
    }

    // Map view_z onto a normalised volume W. Clamp so the skybox + anything
    // past the volume's far edge sample the fully-integrated last slice.
    float z01 = saturate((view_z - z_near) / max(z_far - z_near, 1e-4));

    // Trilinear sample. The volume already stores camera→slice integrated
    // (scattered, 1 - T), so the sample IS the output blend pair.
    float4 s = volume.SampleLevel(volume_samp, float3(uv, z01), 0);
    return s;
}
