#pragma pack_matrix(column_major)

struct GpuObjectData
{
    row_major float4x4 model;
    float3 tint;
    float roughness;
    float3 emissive;
    float metallic;
    uint albedo_index;
    uint normal_index;
    float macro_variation;
    float terrain_blend;
    float3 bb_min;
    float cull_distance;
    float3 bb_max;
    float secondary_blend_sharpness;
    uint albedo_secondary_index;
    uint normal_secondary_index;
    uint _pad2;
    uint _pad3;
};

struct GpuDrawArgs
{
    uint index_count;
    uint index_offset;
    uint base_vertex;
    uint flags;
};

// One ExecuteIndirect command: the b0 object-id root constant followed by
// D3D12_DRAW_INDEXED_ARGUMENTS. 24 bytes; matches the command signature.
struct IndirectCommand
{
    uint object_id;
    uint index_count;
    uint instance_count;
    uint start_index;
    int  base_vertex;
    uint start_instance;
};

cbuffer CullParams : register(b0)
{
    float4 planes[6];
    float3 cam_pos;
    uint object_count;
    // Previous frame's view-projection. Used to project the AABB into the
    // depth space that the Hi-Z pyramid was built from. Same column-major
    // packing as ViewBlock.vp in the main pass.
    float4x4 prev_view_proj;
    // Hi-Z dimensions (mip 0, in texels) and how many mip levels live in
    // the bound Texture2D. `hiz_enabled = 0` skips the Hi-Z test entirely -
    // set on the first frame, before any Hi-Z has been built.
    float2 hiz_size;
    uint hiz_mip_count;
    uint hiz_enabled;
}

StructuredBuffer<GpuObjectData> objects   : register(t0);
StructuredBuffer<GpuDrawArgs>   draw_args : register(t1);
Texture2D<float>                hiz_tex   : register(t2);
RWStructuredBuffer<IndirectCommand> commands : register(u0);
// Per-object outcome of phase-1 cull, written for two-pass occlusion. The
// phase-2 kernel (`main_phase2`) reads it to decide which objects to re-test
// against the rebuilt Hi-Z. Always bound + written so phase 2 sees valid data;
// under single-pass occlusion the values are simply ignored. Mirrors
// metal/shaders/cull.metal.
RWStructuredBuffer<uint>            cull_status : register(u1);

#define DRAW_ENABLED  1u
#define DRAW_CULLABLE 2u

// `cull_status` values. STATUS_HIZ_CANDIDATE is the only outcome phase 2
// re-tests; the others are settled by phase 1.
#define STATUS_DRAWN         0u // visible in phase 1 -> never re-tested
#define STATUS_HIZ_CANDIDATE 1u // Hi-Z-occluded in phase 1 -> phase-2 candidate
#define STATUS_CULLED        2u // frustum/distance/disabled -> never re-tested

// AABB entirely behind any plane -> outside the frustum. Negation of
// gfx::frustum::Frustum::intersects_aabb (the p-vertex test).
bool frustum_culled(float3 bb_min, float3 bb_max)
{
    [unroll] for (uint i = 0u; i < 6u; ++i)
    {
        float3 n = planes[i].xyz;
        float3 farthest = float3(
            n.x >= 0.0 ? bb_max.x : bb_min.x,
            n.y >= 0.0 ? bb_max.y : bb_min.y,
            n.z >= 0.0 ? bb_max.z : bb_min.z);
        if (dot(n, farthest) + planes[i].w < 0.0)
        {
            return true;
        }
    }
    return false;
}

// Squared distance from the camera to the closest point on the AABB; 0 when
// the camera is inside. Mirrors gfx::frustum::aabb_distance_sq.
float aabb_distance_sq(float3 bb_min, float3 bb_max)
{
    float3 d = max(max(bb_min - cam_pos, cam_pos - bb_max), float3(0.0, 0.0, 0.0));
    return dot(d, d);
}

// Project the eight corners of the AABB through `prev_view_proj` and reduce to
// a screen-space rect (NDC.xy in [-1, 1]) plus the AABB's closest NDC depth.
// Returns false if any corner ended up behind the camera (w <= 0) - in that
// case we conservatively treat the AABB as potentially visible and skip the
// Hi-Z test.
bool project_aabb(
    float3 bb_min,
    float3 bb_max,
    out float2 ndc_min,
    out float2 ndc_max,
    out float min_depth)
{
    ndc_min = float2( 1.0,  1.0);
    ndc_max = float2(-1.0, -1.0);
    min_depth = 1.0;
    [unroll] for (uint i = 0u; i < 8u; ++i)
    {
        float3 corner = float3(
            (i & 1u) ? bb_max.x : bb_min.x,
            (i & 2u) ? bb_max.y : bb_min.y,
            (i & 4u) ? bb_max.z : bb_min.z);
        float4 clip = mul(prev_view_proj, float4(corner, 1.0));
        if (clip.w <= 0.0)
        {
            return false;
        }
        float3 ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);
        min_depth = min(min_depth, ndc.z);
    }
    return true;
}

// Test whether the AABB is fully occluded by the Hi-Z pyramid (built from the
// previous frame's depth). Returns true to cull. Conservative - any uncertain
// case returns false (keep the object alive).
bool hiz_occluded(float3 bb_min, float3 bb_max)
{
    float2 ndc_min, ndc_max;
    float aabb_min_depth;
    if (!project_aabb(bb_min, bb_max, ndc_min, ndc_max, aabb_min_depth))
    {
        return false;
    }
    // Clip to NDC bounds. If the AABB extends outside the viewport on both
    // sides of an axis, the frustum check above would already have rejected
    // it; here we just clamp so the UV math stays sane.
    ndc_min = max(ndc_min, float2(-1.0, -1.0));
    ndc_max = min(ndc_max, float2( 1.0,  1.0));
    if (any(ndc_min > ndc_max))
    {
        return false;
    }
    // Standard depth: nearest point of the AABB at NDC.z near 0; anything in
    // [0, 1] is potentially in front of the far plane and worth testing.
    // Behind-near or behind-far means we conservatively keep the AABB.
    if (aabb_min_depth < 0.0 || aabb_min_depth > 1.0)
    {
        return false;
    }
    // Map NDC -> UV (y flips because NDC y is up, UV v is down).
    float2 uv_min = float2(ndc_min.x * 0.5 + 0.5, 0.5 - ndc_max.y * 0.5);
    float2 uv_max = float2(ndc_max.x * 0.5 + 0.5, 0.5 - ndc_min.y * 0.5);
    // Size of the rect at mip 0, in texels.
    float2 size_tex = (uv_max - uv_min) * hiz_size;
    float max_dim = max(size_tex.x, size_tex.y);
    // Pick the mip whose texels are roughly twice the rect size - guarantees a
    // 2x2 footprint covers the rect, matching the standard Hi-Z 4-tap pattern.
    int mip = (int)ceil(log2(max(max_dim, 1.0)));
    mip = clamp(mip, 0, (int)hiz_mip_count - 1);
    // Convert the rect's UV corners into integer texel coords at the picked
    // mip, sample the four corner taps, take the max.
    float2 mip_dim = max(hiz_size / float(1u << (uint)mip), float2(1.0, 1.0));
    int2 lo = int2(floor(uv_min * mip_dim));
    int2 hi = int2(floor(uv_max * mip_dim));
    int2 max_xy = int2(mip_dim) - int2(1, 1);
    lo = clamp(lo, int2(0, 0), max_xy);
    hi = clamp(hi, int2(0, 0), max_xy);
    float d0 = hiz_tex.Load(int3(lo.x, lo.y, mip));
    float d1 = hiz_tex.Load(int3(hi.x, lo.y, mip));
    float d2 = hiz_tex.Load(int3(lo.x, hi.y, mip));
    float d3 = hiz_tex.Load(int3(hi.x, hi.y, mip));
    float occluder_depth = max(max(d0, d1), max(d2, d3));
    // If the AABB's closest projected depth is strictly behind the farthest
    // previously-rasterised surface in this region, the whole AABB is hidden.
    return aabb_min_depth > occluder_depth;
}

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= object_count)
    {
        return;
    }
    GpuDrawArgs a = draw_args[i];

    IndirectCommand cmd;
    cmd.object_id = i;
    cmd.index_count = a.index_count;
    cmd.instance_count = 1u;
    cmd.start_index = a.index_offset;
    cmd.base_vertex = int(a.base_vertex);
    cmd.start_instance = 0u;

    // Record the cull outcome for two-pass occlusion. A Hi-Z cull is the only
    // outcome phase 2 reconsiders against the rebuilt pyramid; everything else
    // is settled here.
    uint status = STATUS_DRAWN;
    if ((a.flags & DRAW_ENABLED) == 0u)
    {
        cmd.instance_count = 0u;
        status = STATUS_CULLED;
    }
    else if (a.flags & DRAW_CULLABLE)
    {
        GpuObjectData obj = objects[i];
        if (frustum_culled(obj.bb_min, obj.bb_max))
        {
            cmd.instance_count = 0u;
            status = STATUS_CULLED;
        }
        else if (obj.cull_distance > 0.0
            && aabb_distance_sq(obj.bb_min, obj.bb_max)
                > obj.cull_distance * obj.cull_distance)
        {
            cmd.instance_count = 0u;
            status = STATUS_CULLED;
        }
        else if (hiz_enabled != 0u && hiz_occluded(obj.bb_min, obj.bb_max))
        {
            cmd.instance_count = 0u;
            status = STATUS_HIZ_CANDIDATE;
        }
    }
    commands[i] = cmd;
    cull_status[i] = status;
}

// GPU-driven shadow cull. One thread per record tests its AABB against the
// cascade's light frustum (the `planes` are this cascade's light-VP planes) and
// writes a draw / no-op into the cascade's region of the shadow indirect buffer.
// Light-frustum only -- deliberately NO Hi-Z (sun cascades have no light-space
// depth pyramid; `hiz_enabled` is ignored here) and NO per-object distance cull:
// the cascade light frustum already bounds the shadow draw distance via its
// extents, and the per-object view `cull_distance` is a view-LOD-fade concept
// that must not silence shadows (the legacy CPU shadow pass drew every caster,
// including off-screen casters beyond cull_distance that still cast into view).
// `cull_status` (u1) is bound but never written (single-pass, no phase 2).
[numthreads(64, 1, 1)]
void main_shadow(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= object_count)
    {
        return;
    }
    GpuDrawArgs a = draw_args[i];

    IndirectCommand cmd;
    cmd.object_id = i;
    cmd.index_count = a.index_count;
    cmd.instance_count = 1u;
    cmd.start_index = a.index_offset;
    cmd.base_vertex = int(a.base_vertex);
    cmd.start_instance = 0u;

    if ((a.flags & DRAW_ENABLED) == 0u)
    {
        cmd.instance_count = 0u;
    }
    else if (a.flags & DRAW_CULLABLE)
    {
        GpuObjectData obj = objects[i];
        if (frustum_culled(obj.bb_min, obj.bb_max))
        {
            cmd.instance_count = 0u;
        }
    }
    commands[i] = cmd;
}

// Phase-2 cull for two-pass occlusion. Runs after the Hi-Z pyramid has been
// rebuilt from this frame's phase-1 depth. Re-tests only the objects phase 1
// marked STATUS_HIZ_CANDIDATE against the fresh pyramid (projected through this
// frame's view-projection, carried in `prev_view_proj` exactly as phase 1 used
// the previous frame's), and emits a draw into the phase-2 command buffer for
// any that turn out visible. Everything else is reset to an instance_count-0
// no-op. Mirrors metal/shaders/cull.metal::cull_encode_phase2.
[numthreads(64, 1, 1)]
void main_phase2(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= object_count)
    {
        return;
    }
    GpuDrawArgs a = draw_args[i];

    IndirectCommand cmd;
    cmd.object_id = i;
    cmd.index_count = a.index_count;
    cmd.instance_count = 1u;
    cmd.start_index = a.index_offset;
    cmd.base_vertex = int(a.base_vertex);
    cmd.start_instance = 0u;

    if (cull_status[i] != STATUS_HIZ_CANDIDATE)
    {
        // Drawn or frustum/distance/disabled in phase 1: phase 1 settled it.
        cmd.instance_count = 0u;
    }
    else
    {
        GpuObjectData obj = objects[i];
        // Re-test against the rebuilt pyramid. A candidate still occluded by
        // this frame's actual depth stays culled; one now visible is redrawn.
        if (hiz_enabled != 0u && hiz_occluded(obj.bb_min, obj.bb_max))
        {
            cmd.instance_count = 0u;
        }
    }
    commands[i] = cmd;
}
