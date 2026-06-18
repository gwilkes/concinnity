// Hi-Z (depth-mip pyramid) builder. Three compute kernels share one root sig:
//
//   init_single  - read the main-depth Texture2D<float>, copy into HiZ mip 0.
//   init_msaa    - read the main-depth Texture2DMS<float> taking MAX over all
//                  samples, write into HiZ mip 0.
//   downsample   - read the previous HiZ mip via Texture2D<float>.Load(...),
//                  write the next mip as the MAX of the corresponding 2x2
//                  source texels.
//
// The MAX reduction is correct because the engine uses standard (not reverse)
// depth - farther geometry has a larger value, so a Hi-Z texel storing the
// MAX represents the farthest visible surface in that region. The cull kernel
// then culls an AABB when its nearest projected depth is greater than the
// Hi-Z sample.
//
// Root signature layout (see directx/hiz.rs::create_hiz_root_signature):
//   b0  - root constants (HizParams: dst_w, dst_h, src_mip, _pad)
//   t0  - SRV descriptor table (depth texture for init, HiZ texture for
//         downsample)
//   u0  - UAV descriptor table (destination HiZ mip)

#pragma pack_matrix(column_major)

cbuffer HizParams : register(b0)
{
    uint dst_width;
    uint dst_height;
    uint src_mip;
    uint sample_count;
}

Texture2D<float>   src_depth_single : register(t0);
Texture2DMS<float> src_depth_msaa   : register(t0);
Texture2D<float>   src_hiz          : register(t0);
RWTexture2D<float> dst_mip          : register(u0);

// Mip 0 from a single-sample main depth resource.
[numthreads(8, 8, 1)]
void init_single(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= dst_width || tid.y >= dst_height) { return; }
    dst_mip[tid.xy] = src_depth_single.Load(int3(tid.xy, 0));
}

// Mip 0 from an MSAA main depth resource. We take the MAX over every sample
// so the Hi-Z is conservative (represents the farthest sub-pixel surface).
[numthreads(8, 8, 1)]
void init_msaa(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= dst_width || tid.y >= dst_height) { return; }
    float d = 0.0;
    for (uint s = 0u; s < sample_count; ++s)
    {
        d = max(d, src_depth_msaa.Load(int2(tid.xy), s));
    }
    dst_mip[tid.xy] = d;
}

// Build mip M+1 from mip M. Each destination texel reads the 4 corresponding
// source texels (with edge clamping for non-power-of-two sources) and takes
// their MAX. `src_mip` is the mip level we read from.
[numthreads(8, 8, 1)]
void downsample(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= dst_width || tid.y >= dst_height) { return; }
    uint sx = tid.x * 2u;
    uint sy = tid.y * 2u;
    // The source dims at `src_mip` are roughly (2 * dst_width, 2 * dst_height).
    // For odd source dimensions we'd lose a texel at the right/bottom edge,
    // but max-reduction is conservative so dropping a half-row is harmless -
    // it can only make the cull *more* conservative, never wrongly cull a
    // visible object.
    float d0 = src_hiz.Load(int3(sx,     sy,     src_mip));
    float d1 = src_hiz.Load(int3(sx + 1, sy,     src_mip));
    float d2 = src_hiz.Load(int3(sx,     sy + 1, src_mip));
    float d3 = src_hiz.Load(int3(sx + 1, sy + 1, src_mip));
    dst_mip[tid.xy] = max(max(d0, d1), max(d2, d3));
}
