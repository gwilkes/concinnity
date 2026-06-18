#include <metal_stdlib>
#include <metal_command_buffer>
using namespace metal;

// Mirrors gfx::render_types::GpuObjectData; only the cull-bounds fields are
// read here, but the full layout must match so indexing lines up.
struct GpuObjectData {
    float4x4      model;
    packed_float3 tint;
    float         roughness;
    packed_float3 emissive;
    float         metallic;
    uint          albedo_index;
    uint          normal_index;
    float         macro_variation;
    float         terrain_blend;
    packed_float3 bb_min;
    float         cull_distance;
    packed_float3 bb_max;
    float         secondary_blend_sharpness;
    uint          albedo_secondary_index;
    uint          normal_secondary_index;
    uint          _pad2;
    uint          _pad3;
};

// Mirrors gfx::render_types::GpuDrawArgs.
struct GpuDrawArgs {
    uint index_count;
    uint index_offset;
    uint base_vertex;
    uint flags;
};

constant uint DRAW_ENABLED  = 1u;
constant uint DRAW_CULLABLE = 2u;

// Per-object outcome of phase-1 cull, written into the `cull_status` buffer
// for two-pass occlusion. `cull_encode_phase2` reads it to decide which
// objects to re-test against the rebuilt Hi-Z. Bytes mirror the values the
// Rust side expects; STATUS_HIZ_CANDIDATE is the only one phase 2 re-tests.
constant uint STATUS_DRAWN         = 0u; // visible in phase 1 → never re-tested
constant uint STATUS_HIZ_CANDIDATE = 1u; // Hi-Z-occluded in phase 1 → phase-2 candidate
constant uint STATUS_CULLED        = 2u; // frustum/distance/disabled → never re-tested

// Per-frame cull inputs. The six frustum planes are extracted CPU-side
// (Gribb-Hartmann, already normalised); xyz = plane normal, w = plane d.
// Layout must match `metal::uniforms::CullUniforms` (208 bytes).
struct CullUniforms {
    float4        planes[6];
    packed_float3 cam_pos;
    uint          object_count;
    // Previous frame's un-jittered view-projection. Projects each AABB into
    // the depth space the Hi-Z pyramid was built from (applied as `M * v`,
    // matching the engine's other VP uniforms).
    float4x4      prev_view_proj;
    // Hi-Z mip-0 dimensions (in texels) and how many mip levels live in the
    // bound texture. `hiz_enabled = 0` skips the Hi-Z test entirely, set on
    // the first frame and immediately after a resize, before a valid pyramid
    // exists.
    float2        hiz_size;
    uint          hiz_mip_count;
    uint          hiz_enabled;
    // Index into the unified cull list where the skinned records begin
    // (= static + instances). Records at or past this index draw the deformed
    // skinned geometry through the u16 `skinned_index_buf`; earlier records use
    // the static u32 `index_buf`. Equals `object_count` when no skinned mesh is
    // folded, so the skinned branch is then never taken. (Metal bakes the index
    // buffer into each indirect command, so unlike DX/VK the kernel must pick it
    // per record rather than the encoder binding it per draw range.)
    uint          skinned_base;
    // Command-slot base offset for the GPU-driven shadow cull. The shadow ICB
    // holds NUM_SHADOW_CASCADES * object_count command slots; cascade `c`'s
    // dispatch writes its survivors at `cascade_base + tid` (cascade_base =
    // c * object_count). The main cull (cull_encode / cull_encode_phase2)
    // leaves it 0 and writes at `tid`, so the shared layout is untouched there.
    uint          cascade_base;
};

// The kernel reaches the indirect command buffer through an argument buffer.
struct ICBContainer {
    command_buffer icb [[id(0)]];
};

// AABB entirely behind any plane -> outside the frustum. Negation of
// gfx::frustum::Frustum::intersects_aabb (the p-vertex test).
bool frustum_culled(float3 bb_min, float3 bb_max, constant float4 *planes) {
    for (uint i = 0u; i < 6u; ++i) {
        float3 n = planes[i].xyz;
        float3 farthest = select(bb_min, bb_max, n >= 0.0);
        if (dot(n, farthest) + planes[i].w < 0.0) {
            return true;
        }
    }
    return false;
}

// Squared distance from the camera to the closest point on the AABB; 0 when
// the camera is inside. Mirrors gfx::frustum::aabb_distance_sq.
float aabb_distance_sq(float3 cam, float3 bb_min, float3 bb_max) {
    float3 d = max(max(bb_min - cam, cam - bb_max), float3(0.0));
    return dot(d, d);
}

// Project the eight corners of the AABB through `vp` and reduce to a
// screen-space rect (NDC.xy in [-1, 1]) plus the AABB's closest NDC depth.
// Returns false if any corner ended up behind the camera (w <= 0); in that
// case we conservatively treat the AABB as potentially visible and skip the
// Hi-Z test. Mirrors directx/shaders/cull.hlsl::project_aabb.
bool project_aabb(
    float3 bb_min,
    float3 bb_max,
    float4x4 vp,
    thread float2 &ndc_min,
    thread float2 &ndc_max,
    thread float &min_depth
) {
    ndc_min = float2( 1.0,  1.0);
    ndc_max = float2(-1.0, -1.0);
    min_depth = 1.0;
    for (uint i = 0u; i < 8u; ++i) {
        float3 corner = float3(
            (i & 1u) ? bb_max.x : bb_min.x,
            (i & 2u) ? bb_max.y : bb_min.y,
            (i & 4u) ? bb_max.z : bb_min.z);
        float4 clip = vp * float4(corner, 1.0);
        if (clip.w <= 0.0) {
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
// previous frame's depth). Returns true to cull. Conservative: any uncertain
// case returns false (keep the object alive). Mirrors
// directx/shaders/cull.hlsl::hiz_occluded.
bool hiz_occluded(
    float3 bb_min,
    float3 bb_max,
    constant CullUniforms &cull,
    texture2d<float, access::read> hiz_tex
) {
    float2 ndc_min, ndc_max;
    float aabb_min_depth;
    if (!project_aabb(bb_min, bb_max, cull.prev_view_proj, ndc_min, ndc_max, aabb_min_depth)) {
        return false;
    }
    // Clip to NDC bounds. If the AABB extends past the viewport on both sides
    // of an axis the frustum check above would already have rejected it; here
    // we just clamp so the UV math stays sane.
    ndc_min = max(ndc_min, float2(-1.0, -1.0));
    ndc_max = min(ndc_max, float2( 1.0,  1.0));
    if (any(ndc_min > ndc_max)) {
        return false;
    }
    // Standard depth: nearest point of the AABB at NDC.z near 0. Behind-near
    // or behind-far means we conservatively keep the AABB.
    if (aabb_min_depth < 0.0 || aabb_min_depth > 1.0) {
        return false;
    }
    // Map NDC -> UV (y flips because NDC y is up, UV v is down).
    float2 uv_min = float2(ndc_min.x * 0.5 + 0.5, 0.5 - ndc_max.y * 0.5);
    float2 uv_max = float2(ndc_max.x * 0.5 + 0.5, 0.5 - ndc_min.y * 0.5);
    // Size of the rect at mip 0, in texels.
    float2 size_tex = (uv_max - uv_min) * cull.hiz_size;
    float max_dim = max(size_tex.x, size_tex.y);
    // Pick the mip whose texels are roughly the rect size, guarantees a 2x2
    // footprint covers the rect, matching the standard Hi-Z 4-tap pattern.
    int mip = (int)ceil(log2(max(max_dim, 1.0)));
    mip = clamp(mip, 0, (int)cull.hiz_mip_count - 1);
    // Convert the rect's UV corners into integer texel coords at the picked
    // mip, sample the four corner taps, take the max.
    float2 mip_dim = max(cull.hiz_size / float(1u << (uint)mip), float2(1.0, 1.0));
    int2 lo = int2(floor(uv_min * mip_dim));
    int2 hi = int2(floor(uv_max * mip_dim));
    int2 max_xy = int2(mip_dim) - int2(1, 1);
    lo = clamp(lo, int2(0, 0), max_xy);
    hi = clamp(hi, int2(0, 0), max_xy);
    float d0 = hiz_tex.read(uint2(lo.x, lo.y), (uint)mip).r;
    float d1 = hiz_tex.read(uint2(hi.x, lo.y), (uint)mip).r;
    float d2 = hiz_tex.read(uint2(lo.x, hi.y), (uint)mip).r;
    float d3 = hiz_tex.read(uint2(hi.x, hi.y), (uint)mip).r;
    float occluder_depth = max(max(d0, d1), max(d2, d3));
    // If the AABB's closest projected depth is strictly behind the farthest
    // previously-rasterised surface in this region, the whole AABB is hidden.
    return aabb_min_depth > occluder_depth;
}

// One thread per draw object. Survivors encode an indexed draw at their own
// command slot; everything else resets its slot to a no-op. base_instance
// carries the object id into the vertex/fragment shaders' [[base_instance]].
//
// `cull_status` records each object's outcome for two-pass occlusion: STATUS_*
// per the constants above. It is always bound (a small per-object buffer) and
// always written so `cull_encode_phase2` reads valid data; under single-pass
// occlusion the values are simply ignored.
kernel void cull_encode(
    constant GpuObjectData *objects     [[buffer(0)]],
    constant GpuDrawArgs   *draw_args   [[buffer(1)]],
    constant CullUniforms  &cull        [[buffer(2)]],
    const device uint      *index_buf   [[buffer(3)]],
    device ICBContainer    *icb_c       [[buffer(4)]],
    device uint            *cull_status [[buffer(5)]],
    const device ushort    *skinned_index_buf [[buffer(6)]],
    texture2d<float, access::read> hiz_tex [[texture(0)]],
    uint                    tid         [[thread_position_in_grid]]
) {
    if (tid >= cull.object_count) {
        return;
    }
    render_command cmd(icb_c->icb, tid);
    GpuDrawArgs a = draw_args[tid];

    if ((a.flags & DRAW_ENABLED) == 0u) {
        cmd.reset();
        cull_status[tid] = STATUS_CULLED;
        return;
    }
    if (a.flags & DRAW_CULLABLE) {
        GpuObjectData obj = objects[tid];
        if (frustum_culled(obj.bb_min, obj.bb_max, cull.planes)) {
            cmd.reset();
            cull_status[tid] = STATUS_CULLED;
            return;
        }
        if (obj.cull_distance > 0.0) {
            float dsq = aabb_distance_sq(cull.cam_pos, obj.bb_min, obj.bb_max);
            if (dsq > obj.cull_distance * obj.cull_distance) {
                cmd.reset();
                cull_status[tid] = STATUS_CULLED;
                return;
            }
        }
        // Hi-Z occlusion: cull when the AABB is fully behind the previous
        // frame's depth pyramid. Skipped on the first frame / after a resize
        // (`hiz_enabled = 0`), where no valid pyramid exists yet. A Hi-Z cull
        // here is the only outcome two-pass phase 2 reconsiders.
        if (cull.hiz_enabled != 0u && hiz_occluded(obj.bb_min, obj.bb_max, cull, hiz_tex)) {
            cmd.reset();
            cull_status[tid] = STATUS_HIZ_CANDIDATE;
            return;
        }
    }
    // Skinned records (tid >= skinned_base) draw the compute-deformed geometry
    // through the u16 skinned index buffer; everything else uses the static u32
    // index buffer. The index buffer is part of the indirect command on Metal,
    // so it is selected here rather than bound per draw range like DX/VK.
    if (tid >= cull.skinned_base) {
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    skinned_index_buf + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    tid);
    } else {
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    index_buf + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    tid);
    }
    cull_status[tid] = STATUS_DRAWN;
}

// Phase-2 cull for two-pass occlusion. Runs after the Hi-Z pyramid has been
// rebuilt from this frame's phase-1 depth. Re-tests only the objects phase 1
// marked STATUS_HIZ_CANDIDATE against the fresh pyramid (projected through
// this frame's view-projection, carried in `cull.prev_view_proj` exactly as
// phase 1 used the previous frame's), and encodes a draw into the phase-2 ICB
// for any that turn out visible. Everything else resets its slot. Objects that
// were drawn or frustum/distance-culled in phase 1 are skipped; phase 1
// already settled them. `cull.hiz_enabled` is expected to be 1 here (the
// rebuild always precedes this dispatch), but the guard keeps the kernel safe
// if it is ever dispatched without a valid pyramid (all candidates then redraw,
// which is conservative).
kernel void cull_encode_phase2(
    constant GpuObjectData *objects     [[buffer(0)]],
    constant GpuDrawArgs   *draw_args   [[buffer(1)]],
    constant CullUniforms  &cull        [[buffer(2)]],
    const device uint      *index_buf   [[buffer(3)]],
    device ICBContainer    *icb_c       [[buffer(4)]],
    device uint            *cull_status [[buffer(5)]],
    const device ushort    *skinned_index_buf [[buffer(6)]],
    texture2d<float, access::read> hiz_tex [[texture(0)]],
    uint                    tid         [[thread_position_in_grid]]
) {
    if (tid >= cull.object_count) {
        return;
    }
    render_command cmd(icb_c->icb, tid);
    if (cull_status[tid] != STATUS_HIZ_CANDIDATE) {
        cmd.reset();
        return;
    }
    GpuObjectData obj = objects[tid];
    // Re-test against the rebuilt pyramid. A candidate still occluded by this
    // frame's actual depth stays culled; one that is now visible is redrawn.
    if (cull.hiz_enabled != 0u && hiz_occluded(obj.bb_min, obj.bb_max, cull, hiz_tex)) {
        cmd.reset();
        return;
    }
    GpuDrawArgs a = draw_args[tid];
    if (tid >= cull.skinned_base) {
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    skinned_index_buf + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    tid);
    } else {
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    index_buf + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    tid);
    }
}

// GPU-driven cascaded-shadow cull. One thread per record, dispatched
// once per re-rendered cascade with that cascade's LIGHT frustum in
// `cull.planes` and `cull.cascade_base = cascade_idx * object_count`. Survivors
// encode a depth-only indexed draw at slot `cascade_base + tid`; the depth-only
// bindless shadow pipeline then issues that cascade's slice of the ICB.
//
// Frustum-ONLY: no Hi-Z (sun cascades have no light-space depth pyramid) and no
// per-object distance cull. The cascade light frustum already bounds the shadow
// draw distance, and the per-object view `cull_distance` is a view-LOD-fade
// concept that must not silence a shadow (you do not want shadows popping as
// objects LOD-fade). Non-cullable records (chunks / runtime clones, DRAW_CULLABLE
// clear) draw into every cascade, exactly as the legacy CPU shadow loop did. No
// `cull_status` is written: shadow runs single-pass, and the shared status
// buffer belongs to the main two-pass occlusion path. The `skinned_base`
// index-buffer branch is identical to `cull_encode` (the shadow ICB must bake
// the u16 skinned IB for the deformed tail just like the main ICB).
kernel void cull_encode_shadow(
    constant GpuObjectData *objects     [[buffer(0)]],
    constant GpuDrawArgs   *draw_args   [[buffer(1)]],
    constant CullUniforms  &cull        [[buffer(2)]],
    const device uint      *index_buf   [[buffer(3)]],
    device ICBContainer    *icb_c       [[buffer(4)]],
    const device ushort    *skinned_index_buf [[buffer(6)]],
    uint                    tid         [[thread_position_in_grid]]
) {
    if (tid >= cull.object_count) {
        return;
    }
    render_command cmd(icb_c->icb, cull.cascade_base + tid);
    GpuDrawArgs a = draw_args[tid];

    if ((a.flags & DRAW_ENABLED) == 0u) {
        cmd.reset();
        return;
    }
    if (a.flags & DRAW_CULLABLE) {
        GpuObjectData obj = objects[tid];
        if (frustum_culled(obj.bb_min, obj.bb_max, cull.planes)) {
            cmd.reset();
            return;
        }
    }
    if (tid >= cull.skinned_base) {
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    skinned_index_buf + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    tid);
    } else {
        cmd.draw_indexed_primitives(primitive_type::triangle,
                                    a.index_count,
                                    index_buf + a.index_offset,
                                    1u,
                                    a.base_vertex,
                                    tid);
    }
}
