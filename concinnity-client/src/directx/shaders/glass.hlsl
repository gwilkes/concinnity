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
    float    _pad;
}

cbuffer GlassParams : register(b1)
{
    float4 centre;  // world-space pane centre, .w unused
    float4 normal;  // unit pane normal (facing direction), .w unused
    float4 tint;    // colour multiplied into the refracted scene, .w unused
    float  opacity;
    float  refraction_strength;
    float  fresnel_power;
    float  _pad1;
}

Texture2D<float4> scene_color : register(t0);
#if USE_MSAA
Texture2DMS<float> scene_depth : register(t1);
#else
Texture2D<float>   scene_depth : register(t1);
#endif
SamplerState post_samp : register(s0);

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

    // Schlick-Fresnel rim: brighter + more opaque at grazing angles.
    float n_dot_v = saturate(dot(n, view_dir));
    float fresnel = pow(1.0 - n_dot_v, max(fresnel_power, 1e-3));

    float3 rim = float3(1.0, 1.0, 1.0);
    float3 colour = lerp(refracted, rim, fresnel * 0.5);
    float alpha = saturate(lerp(opacity, 1.0, fresnel));

    return float4(colour, alpha);
}
