// src/gfx/mipmap.rs
//
// Backend-agnostic mip-chain generation for streamed RGBA8 textures. Each
// backend's texture upload calls `generate_mip_chain` and uploads every level,
// so albedo and normal maps minify through a proper trilinear chain instead of
// aliasing from a single mip-0 sample at a distance.
//
// Levels are produced by a 2x2 box filter in stored (RGBA8) space, halving each
// axis with floor division so every level's dimensions match the GPU mip
// convention `max(1, base >> level)`. That keeps the CPU chain in lockstep with
// the image's allocated mip levels on all three backends.

// One level of a mip chain: dimensions plus tightly packed RGBA8 pixels
// (`width * height * 4` bytes, no row padding).
pub(crate) struct MipLevel {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

// Number of mip levels for a `width` x `height` texture: the full chain down to
// 1x1, i.e. floor(log2(max(w, h))) + 1.
pub(crate) fn mip_level_count(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height).max(1);
    32 - max_dim.leading_zeros()
}

// Build the full mip chain for a `width` x `height` RGBA8 image. Level 0 is the
// input copied verbatim; each subsequent level halves both axes (floored, min 1)
// and box-filters the level above it. `rgba8` must hold at least
// `width * height * 4` bytes (the backend uploads validate this before calling).
pub(crate) fn generate_mip_chain(width: u32, height: u32, rgba8: &[u8]) -> Vec<MipLevel> {
    let count = mip_level_count(width, height);
    let base_len = width as usize * height as usize * 4;
    let mut levels: Vec<MipLevel> = Vec::with_capacity(count as usize);
    levels.push(MipLevel {
        width,
        height,
        pixels: rgba8[..base_len].to_vec(),
    });
    for _ in 1..count {
        let prev = levels.last().unwrap();
        let dw = (prev.width / 2).max(1);
        let dh = (prev.height / 2).max(1);
        let mut pixels = vec![0u8; dw as usize * dh as usize * 4];
        downsample_box(prev, dw, dh, &mut pixels);
        levels.push(MipLevel {
            width: dw,
            height: dh,
            pixels,
        });
    }
    levels
}

// Average each destination texel from the corresponding 2x2 block of `src`,
// clamping source indices at the edge (so an odd source dimension reuses its
// last row/column rather than reading out of bounds).
fn downsample_box(src: &MipLevel, dw: u32, dh: u32, dst: &mut [u8]) {
    let sw = src.width as usize;
    let sh = src.height as usize;
    for y in 0..dh as usize {
        let sy0 = (2 * y).min(sh - 1);
        let sy1 = (2 * y + 1).min(sh - 1);
        for x in 0..dw as usize {
            let sx0 = (2 * x).min(sw - 1);
            let sx1 = (2 * x + 1).min(sw - 1);
            let i00 = (sy0 * sw + sx0) * 4;
            let i01 = (sy0 * sw + sx1) * 4;
            let i10 = (sy1 * sw + sx0) * 4;
            let i11 = (sy1 * sw + sx1) * 4;
            let d = (y * dw as usize + x) * 4;
            for c in 0..4 {
                let sum = src.pixels[i00 + c] as u32
                    + src.pixels[i01 + c] as u32
                    + src.pixels[i10 + c] as u32
                    + src.pixels[i11 + c] as u32;
                // +2 rounds to nearest before the divide by 4.
                dst[d + c] = ((sum + 2) / 4) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_count_matches_floor_log2_plus_one() {
        assert_eq!(mip_level_count(1, 1), 1);
        assert_eq!(mip_level_count(2, 2), 2);
        assert_eq!(mip_level_count(256, 256), 9);
        assert_eq!(mip_level_count(512, 512), 10);
        // Non-square / non-power-of-two key off the larger axis.
        assert_eq!(mip_level_count(640, 384), 10); // 640 -> floor(log2)=9, +1
        assert_eq!(mip_level_count(1, 8), 4); // 8 -> 3, +1
    }

    #[test]
    fn chain_dimensions_halve_to_one() {
        let px = vec![0u8; 4 * 4 * 4];
        let chain = generate_mip_chain(4, 4, &px);
        let dims: Vec<(u32, u32)> = chain.iter().map(|m| (m.width, m.height)).collect();
        assert_eq!(dims, vec![(4, 4), (2, 2), (1, 1)]);
        assert_eq!(chain.len() as u32, mip_level_count(4, 4));
    }

    #[test]
    fn non_square_chain_floors_each_axis_independently() {
        let px = vec![0u8; 4 * 2 * 4];
        let chain = generate_mip_chain(4, 2, &px);
        let dims: Vec<(u32, u32)> = chain.iter().map(|m| (m.width, m.height)).collect();
        // Width halves to 1 in two steps; height bottoms out at 1 and stays.
        assert_eq!(dims, vec![(4, 2), (2, 1), (1, 1)]);
    }

    #[test]
    fn two_by_two_averages_to_single_texel() {
        // Four grey texels 0, 4, 8, 12 -> mean 6 (rounded).
        let px = vec![
            0, 0, 0, 0, // (0,0)
            4, 4, 4, 4, // (1,0)
            8, 8, 8, 8, // (0,1)
            12, 12, 12, 12, // (1,1)
        ];
        let chain = generate_mip_chain(2, 2, &px);
        assert_eq!(chain.len(), 2);
        let mip1 = &chain[1];
        assert_eq!((mip1.width, mip1.height), (1, 1));
        assert_eq!(mip1.pixels, vec![6, 6, 6, 6]);
    }

    #[test]
    fn rounds_to_nearest() {
        // 0, 0, 0, 1 -> mean 0.25 -> rounds to 0; 0,1,1,1 -> 0.75 -> rounds to 1.
        let dark = vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 1, 1, 1, 255];
        let c = generate_mip_chain(2, 2, &dark);
        assert_eq!(&c[1].pixels[0..3], &[0, 0, 0]);

        let bright = vec![0, 0, 0, 255, 1, 1, 1, 255, 1, 1, 1, 255, 1, 1, 1, 255];
        let c = generate_mip_chain(2, 2, &bright);
        assert_eq!(&c[1].pixels[0..3], &[1, 1, 1]);
    }
}
