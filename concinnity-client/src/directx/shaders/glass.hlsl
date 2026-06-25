#pragma pack_matrix(column_major)

// Glass panel pass. Mirrors glass_vertex / glass_fragment in
// src/metal/shaders/glass.metal. Runs in the PassId::Transparent slot after
// SSR resolve and before TAA. The quad vertices are pre-transformed into world
// space at build time (see geometry::glass_quad), so the vertex shader only
// projects them.
//
// The fragment shader discards where nearer opaque geometry occludes the pane
// (manual depth test against the main pass depth, since the transparent pass
// binds no depth attachment), refracts the pre-transparent scene snapshot,
// tints it, and adds a Schlick-Fresnel rim. The pipeline straight-alpha blends
// the result (SRC_ALPHA / ONE_MINUS_SRC_ALPHA).
//
// `USE_MSAA` is defined by the host (1 when the main pass uses MSAA, 0
// otherwise) so the depth SRV declaration matches the resource's sample count.

cbuffer TransparentView : register(b0)
{
    float4x4 vp;          // world -> clip (jittered when TAA is on)
    float4x4 inv_vp;      // clip -> world
    float4   camera_pos;  // world-space camera, .w unused
    float2   viewport;    // attachment dimensions in pixels
    float    time;        // seconds since startup
    // Mips in the sky prefilter cube; 0 = no EnvironmentMap bound (the reflection
    // then keeps the white rim where no probe covers). DX puts this per-frame
    // value in TransparentView (not GlassParams, where Metal does) because the
    // per-panel GlassParams CBV is static; it is only a "has env" gate for glass.
    float    prefilter_mip_count;
}

cbuffer GlassParams : register(b1)
{
    float4 centre;  // world-space pane centre, .w unused
    float4 normal;  // unit pane normal (facing direction), .w unused
    float4 tint;    // colour multiplied into the refracted scene, .w unused
    float  opacity;
    float  refraction_strength;
    float  fresnel_power;
    // 1.0 when this pane has a planar reflection slot: sample the sharp mirror
    // render (t3) projectively instead of the box-projected probe cube.
    float  planar;
}

Texture2D<float4> scene_color : register(t0);
#if USE_MSAA
Texture2DMS<float> scene_depth : register(t1);
#else
Texture2D<float>   scene_depth : register(t1);
#endif
// Sky IBL prefilter cube: the reflection fallback where no probe covers the pane.
TextureCube<float4> prefilter_cube : register(t2);
// This pane's planar reflection resolve (the scene re-rendered mirrored across the
// pane plane), bound per pane. Sampled projectively when `planar > 0.5`.
Texture2D<float4> planar_reflection : register(t3);
SamplerState post_samp : register(s0);

// The probe cube array (t7), the `ProbeBlock` cbuffer (b4), and `cube_sampler`
// (s2) are declared in probe_common.hlsl, concatenated ahead of this shader (no
// #include handler on DX). They let glass sample the LOCAL box-projected scene
// capture instead of only the foreign sky cube -- the same source the forward IBL
// specular and the SSR/RT miss fallback use.

struct VsIn
{
    float3 pos     : POSITION;
    float3 normal  : NORMAL;
    float3 tangent : TANGENT;
    float3 color   : COLOR;
    float2 uv      : TEXCOORD;
};

struct VsOut
{
    float4 sv_pos    : SV_POSITION;
    float3 world_pos : TEXCOORD0;
};

VsOut vs_main(VsIn input)
{
    VsOut output;
    // Quad vertices are pre-transformed into world space at build time.
    output.world_pos = input.pos;
    output.sv_pos = mul(vp, float4(input.pos, 1.0));
    return output;
}

float4 ps_main(VsOut input) : SV_TARGET
{
    float3 view_dir = normalize(camera_pos.xyz - input.world_pos);
    // Two-sided: orient the normal toward the viewer so a pane lit from behind
    // still Fresnels correctly.
    float3 n = normalize(normal.xyz);
    if (dot(n, view_dir) < 0.0)
    {
        n = -n;
    }

    float2 vp_dim = max(viewport, float2(1.0, 1.0));
    float2 frag_uv = float2(input.sv_pos.x / vp_dim.x, input.sv_pos.y / vp_dim.y);

    // Manual depth occlusion: discard where the scene depth at this pixel is
    // nearer than the pane (the pass has no hardware depth test). D3D depth is
    // [0, 1] with 0 = near, matching SV_Position.z, so a smaller stored value
    // means opaque geometry sits in front of the pane.
    int2 pixel = min(int2(input.sv_pos.xy), int2(vp_dim) - int2(1, 1));
#if USE_MSAA
    float scene_self_depth = scene_depth.Load(pixel, 0);
#else
    float scene_self_depth = scene_depth.Load(int3(pixel, 0));
#endif
    if (scene_self_depth < input.sv_pos.z)
    {
        discard;
    }

    // Refraction: perturb the screen lookup by the pane normal's screen-plane
    // component so the background bends across the pane.
    float2 refract_uv = clamp(frag_uv + n.xy * refraction_strength,
                              float2(0.001, 0.001), float2(0.999, 0.999));
    float3 refracted = scene_color.Sample(post_samp, refract_uv).rgb * tint.rgb;

    // Reflection along the mirror direction. When the planar reflection is active
    // (`planar > 0.5`: the scene re-rendered mirrored across this pane's plane)
    // sample it projectively at this fragment's own screen UV -- a flat pane is a
    // perfect mirror, so the mirrored render lands exactly under the reflector and
    // needs no distortion. Otherwise prefer the local box-projected probe set (the
    // scene capture) when the world has baked probes, else the sky prefilter cube,
    // else a white rim so a probe-less, env-less world still reads as glass. A pane
    // is smooth, so every path is sharp (mip 0).
    float3 R = reflect(-view_dir, n);
    float3 reflection;
    if (planar > 0.5)
    {
        reflection = planar_reflection.Sample(post_samp, frag_uv).rgb;
    }
    else if (probes.count > 0u)
    {
        reflection = probe_set_specular(probes, input.world_pos, R, 0.0);
    }
    else if (prefilter_mip_count > 0.5)
    {
        reflection = prefilter_cube.SampleLevel(cube_sampler, R, 0.0).rgb;
    }
    else
    {
        reflection = float3(1.0, 1.0, 1.0);
    }

    // Schlick Fresnel (F0 = 0.04 dielectric) drives the reflection/refraction
    // blend: ~4% head-on, rising to a full mirror at grazing. `fresnel_power`
    // stays the author's grazing-rim shaping control for the opacity ramp.
    float n_dot_v = saturate(dot(n, view_dir));
    float rim = pow(1.0 - n_dot_v, max(fresnel_power, 1e-3));
    float refl_weight = saturate(0.04 + 0.96 * rim);
    float3 colour = lerp(refracted, reflection, refl_weight);
    float alpha = saturate(lerp(opacity, 1.0, rim));

    return float4(colour, alpha);
}
