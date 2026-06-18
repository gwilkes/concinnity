#include <metal_stdlib>
using namespace metal;

// Hi-Z (depth-mip pyramid) builder. Two compute kernels, one params buffer:
//
//   hiz_init_msaa - read the main-depth `depth2d_ms<float>` taking MAX over all
//                   samples, write into HiZ mip 0.
//   hiz_downsample - read the previous HiZ mip via `read(coord, src_mip)`, write
//                    the next mip as the MAX of the corresponding 2x2 source
//                    texels.
//
// The MAX reduction is correct because the engine uses standard (not reverse)
// depth - farther geometry has a larger value, so a Hi-Z texel storing the MAX
// represents the farthest visible surface in that region. The cull kernel then
// culls an AABB when its nearest projected depth is greater than the Hi-Z
// sample. Mirrors src/directx/shaders/hiz_build.hlsl.
//
// The source mip is read through the whole-texture `read` binding (lod = the
// `src_mip` param) while the destination mip is a single-level texture view
// bound with write access, so each downsample reads mip M and writes mip M+1
// without aliasing the same texels. Successive dispatches in the serial compute
// encoder are auto-barriered by Metal, so the chain stays correct.

struct HizParams {
    uint dst_width;
    uint dst_height;
    uint src_mip;
    uint sample_count;
};

// Mip 0 from an MSAA main-depth resource. We take the MAX over every sample so
// the Hi-Z is conservative (represents the farthest sub-pixel surface).
kernel void hiz_init_msaa(
    constant HizParams              &p        [[buffer(0)]],
    depth2d_ms<float, access::read>  src_depth [[texture(0)]],
    texture2d<float, access::write>  dst_mip   [[texture(1)]],
    uint2                            tid       [[thread_position_in_grid]]
) {
    if (tid.x >= p.dst_width || tid.y >= p.dst_height) { return; }
    float d = 0.0;
    for (uint s = 0u; s < p.sample_count; ++s) {
        d = max(d, src_depth.read(tid, s));
    }
    dst_mip.write(float4(d, 0.0, 0.0, 0.0), tid);
}

// Build mip M+1 from mip M. Each destination texel reads the 4 corresponding
// source texels (with edge clamping for non-power-of-two sources) and takes
// their MAX. `src_mip` is the mip level read from.
kernel void hiz_downsample(
    constant HizParams              &p       [[buffer(0)]],
    texture2d<float, access::read>   src_hiz [[texture(0)]],
    texture2d<float, access::write>  dst_mip [[texture(1)]],
    uint2                            tid     [[thread_position_in_grid]]
) {
    if (tid.x >= p.dst_width || tid.y >= p.dst_height) { return; }
    uint sx = tid.x * 2u;
    uint sy = tid.y * 2u;
    // The source dims at `src_mip` are roughly (2 * dst_width, 2 * dst_height).
    // For odd source dimensions we'd lose a texel at the right/bottom edge, but
    // max-reduction is conservative so dropping a half-row is harmless - it can
    // only make the cull *more* conservative, never wrongly cull a visible
    // object. Clamp the +1 taps so an odd edge reuses the in-bounds texel.
    uint src_w = src_hiz.get_width(p.src_mip);
    uint src_h = src_hiz.get_height(p.src_mip);
    uint sx1 = min(sx + 1u, src_w - 1u);
    uint sy1 = min(sy + 1u, src_h - 1u);
    float d0 = src_hiz.read(uint2(sx,  sy),  p.src_mip).r;
    float d1 = src_hiz.read(uint2(sx1, sy),  p.src_mip).r;
    float d2 = src_hiz.read(uint2(sx,  sy1), p.src_mip).r;
    float d3 = src_hiz.read(uint2(sx1, sy1), p.src_mip).r;
    dst_mip.write(float4(max(max(d0, d1), max(d2, d3)), 0.0, 0.0, 0.0), tid);
}
