#pragma pack_matrix(column_major)

// Projected decal pass - fragment shader. Mirrors decal_fragment in
// src/metal/shaders/decal.metal. Reconstructs the world-space sample point
// at each rasterised pixel from the main pass's depth attachment, transforms
// it back into decal-local space, clips against the unit box [-0.5, 0.5]^3,
// and stamps the decal texture × tint into the resolved HDR target. The
// pipeline's blend state alpha-composites the result on top of the scene.
//
// `USE_MSAA` is defined by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth SRV declaration and sampler call match the
// underlying resource.

cbuffer DecalView : register(b0)
{
    float4x4 vp;
    float4x4 inv_vp;
    float2   viewport;
    float2   _pad;
}

cbuffer DecalParams : register(b1)
{
    float4x4 model;
    float4x4 inv_model;
    float4   tint;
    float    fade_pow;
    float    _p0;
    float    _p1;
    float    _p2;
}

#if USE_MSAA
Texture2DMS<float> scene_depth : register(t0);
#else
Texture2D<float>   scene_depth : register(t0);
#endif

Texture2D    decal_tex : register(t1);
SamplerState samp      : register(s0);

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
    // Sample 0 of the MSAA depth (or the single-sample depth) is the cleared
    // / "no geometry" sentinel when the main pass left the pixel empty. A
    // value of exactly 1.0 means nothing to project onto. Sub-1.0 = real
    // geometry; the skybox sentinel pin keeps the sky from receiving decals.
#if USE_MSAA
    float depth = scene_depth.Load(pixel, 0);
#else
    float depth = scene_depth.Load(int3(pixel, 0));
#endif
    if (depth >= 1.0)
    {
        discard;
    }

    // Reconstruct the world-space point at this pixel via the inverse VP.
    // SV_Position's xy is the pixel-centre coordinate; divide by viewport to
    // get [0, 1], then map to NDC [-1, 1]. D3D has y=0 at the top of the
    // framebuffer but +y is up in NDC, so flip y to match the Metal port.
    float2 ndc_xy = (p.sv_pos.xy / viewport) * 2.0 - 1.0;
    ndc_xy.y = -ndc_xy.y;
    float4 clip  = float4(ndc_xy, depth, 1.0);
    float4 world = mul(inv_vp, clip);
    world /= world.w;

    // Decal-local clip against the unit box.
    float4 local = mul(inv_model, world);
    float3 ab    = abs(local.xyz);
    if (ab.x > 0.5 || ab.y > 0.5 || ab.z > 0.5)
    {
        discard;
    }

    // Soft fade along the projection axis (local +Y) so the stamp doesn't
    // show a hard band where the surface tilts away from the projection
    // plane. Alpha rolls off as |local.y| approaches 0.5.
    float fade = saturate(1.0 - (ab.y * 2.0));
    fade = pow(fade, max(fade_pow, 1.0));

    // Sample the decal texture on local X-Z; UV in [0, 1] with V=0 at top to
    // match the engine's other textures.
    float2 uv = local.xz + 0.5;
    uv.y      = 1.0 - uv.y;
    float4 tex = decal_tex.Sample(samp, uv);

    float4 c;
    c.rgb = tex.rgb * tint.rgb;
    c.a   = tex.a * tint.a * fade;
    return c;
}
