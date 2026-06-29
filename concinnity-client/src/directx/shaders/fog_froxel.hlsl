#pragma pack_matrix(column_major)

// Volumetric-fog froxel-volume compute kernel. Mirrors fog_froxel_kernel in
// src/metal/shaders/fog.metal. Each frame the kernel populates a screen-
// aligned 3D `RGBA16Float` volume of `(scattered_rgb, 1 - T)` across the
// view frustum; the fullscreen `Fog` fragment shader samples the volume by
// `(screen_uv, view_z)` instead of marching per pixel.
//
// One thread per (x, y) tile of the volume. The kernel walks the Z slices
// from front to back, accumulating per-slab scatter + transmittance with a
// CSM shadow tap per slab. Slab integration uses Beer-Lambert + an analytic
// energy-conserving in-scatter term; the result of slice N is the camera→
// slice-N transmittance + integrated scatter, so the fragment shader's
// trilinear sample IS the output blend pair.

#define NUM_SHADOW_CASCADES 4

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
    float4x4 view;               // offset  0
    uint3    froxel_dims;        // offset 64
    uint     _pad_align;         // offset 76
    float    z_near;             // offset 80
    float    z_far;              // offset 84
    float2   _pad_ff;            // offset 88
}                                // 96 B - matches gfx::render_types::FogFroxelParams

cbuffer ShadowUniforms : register(b2)
{
    float4x4 light_vps[NUM_SHADOW_CASCADES];
    float4   cascade_splits;
    uint     active_cascades;
}

Texture2DArray<float> shadow_map : register(t0);
RWTexture3D<float4>   volume     : register(u0);

// Comparison sampler for the shadow map. Static in the root signature so we
// don't need a sampler heap bind on the compute pass.
SamplerComparisonState shadow_samp : register(s0);

float henyey_greenstein(float cos_theta, float g)
{
    float g2    = g * g;
    float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * 3.14159265358979 * pow(max(denom, 1e-5), 1.5));
}

// Cascade-aware single-sample shadow tap. No PCF - the trilinear sample at
// the fragment-shader stage smooths the result. Returns 1.0 (fully lit)
// outside any cascade, matching the main shader's fall-through.
float fog_shadow_factor(float3 world_pos, float view_depth)
{
    uint cascade = NUM_SHADOW_CASCADES;
    if      (view_depth < cascade_splits.x) cascade = 0;
    else if (view_depth < cascade_splits.y) cascade = 1;
    else if (view_depth < cascade_splits.z) cascade = 2;
    else if (view_depth < cascade_splits.w) cascade = 3;
    if (cascade >= active_cascades) return 1.0;

    float4 light_clip = mul(light_vps[cascade], float4(world_pos, 1.0));
    float3 ndc = light_clip.xyz / light_clip.w;
    // D3D NDC y is up; flip into [0,1] UV that matches the depth attachment.
    float2 uv = float2(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (any(uv < 0.0) || any(uv > 1.0) || ndc.z < 0.0 || ndc.z > 1.0)
    {
        return 1.0;
    }
    float bias = 0.0015 * (1.0 + (float)cascade * 0.7);
    float ref = ndc.z - bias;
    return shadow_map.SampleCmpLevelZero(
        shadow_samp,
        float3(uv, (float)cascade),
        ref);
}

// Reconstruct the world-space position at a froxel centre. `z_slice` is a
// floating-point slab index - we sample at the slab centre with an
// interleaved-gradient-noise jitter (see the kernel).
float3 froxel_to_world(uint x, uint y, float z_slice)
{
    float2 uv = (float2((float)x + 0.5, (float)y + 0.5))
              / float2((float)froxel_dims.x, (float)froxel_dims.y);
    float2 ndc_xy = float2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));

    float view_z = lerp(z_near, z_far,
                        (z_slice + 0.5) / (float)froxel_dims.z);

    float4 clip_far  = float4(ndc_xy, 1.0, 1.0);
    float4 world_far = mul(inv_vp, clip_far);
    world_far /= world_far.w;
    float3 ray = normalize(world_far.xyz - cam_pos);

    // Projection of `ray` onto the view-forward axis. `view` is the
    // world→view matrix in standard row-vector form (`mul(view, world_pos)`
    // gives view space), so row 2 is the view-z basis expressed in world
    // coordinates; negated gives view-forward (positive view depth is -z).
    // HLSL's `m[i]` returns the i-th row regardless of storage layout, so
    // `view[2].xyz` is the right read. (Metal's MSL `m[col][row]` would
    // build the same vector via `(m[0][2], m[1][2], m[2][2])`; the literal
    // index-pair transcription doesn't carry over.)
    float3 view_fwd = -view[2].xyz;
    float  forward  = max(dot(ray, view_fwd), 1e-4);
    float  t        = view_z / forward;

    return cam_pos + ray * t;
}

[numthreads(8, 8, 1)]
void main(uint2 tid : SV_DispatchThreadID)
{
    if (tid.x >= froxel_dims.x || tid.y >= froxel_dims.y)
    {
        return;
    }

    // Interleaved gradient noise: a per-(x, y) tile offset so neighbouring
    // froxel columns sample density / shadows at slightly different Z. The
    // trilinear sample at fragment-shader time + TAA smear the noise into
    // smooth gradients.
    float2 tile_xy = float2((float)tid.x, (float)tid.y);
    float  ign     = frac(52.9829189 *
                          frac(dot(tile_xy, float2(0.06711056, 0.00583715))));

    // Ray direction at the (x, y, 0) froxel - within a column the direction
    // is approximately constant across Z (small-FOV approximation), so we
    // sample it once and reuse for the phase term.
    float3 col_world = froxel_to_world(tid.x, tid.y, 0.0);
    float3 ray_dir   = normalize(col_world - cam_pos);
    float  cos_theta = dot(ray_dir, normalize(sun_dir));
    float  phase     = henyey_greenstein(cos_theta, phase_g);

    float3 sun_inscatter_unshadowed = sun_color * phase * color.rgb;
    float3 ambient_inscatter        = color.rgb * ambient;

    float total_z  = z_far - z_near;
    float step_len = total_z / (float)froxel_dims.z;

    float3 accumulated   = float3(0.0, 0.0, 0.0);
    float  transmittance = 1.0;

    for (uint z = 0; z < froxel_dims.z; ++z)
    {
        // Jittered slab centre. The slab integral stays exact in the
        // constant-density limit since `tau` uses the full slab width;
        // only the sample point shifts.
        float  z_jittered = (float)z + ign - 0.5;
        float3 pos        = froxel_to_world(tid.x, tid.y, z_jittered);

        // Exponential height falloff (matches the inline ray-march path).
        float h             = pos.y - height_reference;
        float local_density = density * exp(-max(h, -50.0) * height_falloff);

        // CSM shadow tap.
        float slab_view_z = lerp(z_near, z_far,
                                 (z_jittered + 0.5) / (float)froxel_dims.z);
        float shad        = fog_shadow_factor(pos, slab_view_z);

        // Per-slab Beer-Lambert + analytic energy-conserving in-scatter.
        float  tau        = local_density * step_len;
        float  slab_T     = exp(-tau);
        float  slab_alpha = 1.0 - slab_T;
        float3 inscatter  = sun_inscatter_unshadowed * shad + ambient_inscatter;
        accumulated  += transmittance * slab_alpha * inscatter;
        transmittance *= slab_T;

        // Write the running pair into this slice. Each slice carries the
        // camera→slice integral so a sample at any depth gives the right
        // value without an extra accumulation pass.
        float4 stored = float4(accumulated, 1.0 - transmittance);
        volume[uint3(tid.x, tid.y, z)] = stored;

        // Early-out once the medium is almost opaque. Fill the remaining
        // slices with the saturated value; any sample past this point
        // produces the same result.
        if (transmittance < 0.005)
        {
            transmittance = 0.0;
            for (uint zz = z + 1; zz < froxel_dims.z; ++zz)
            {
                volume[uint3(tid.x, tid.y, zz)] = stored;
            }
            break;
        }
    }
}
