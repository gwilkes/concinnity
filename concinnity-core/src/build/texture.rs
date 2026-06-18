// src/build/texture.rs
//
// Texture payload format helpers shared between the runtime and the build
// crate. The file -> pixels decoders (PNG / JPEG / DDS / TGA / glb-embedded
// images) live in `concinnity_cook::texture`; this module keeps only what a
// running engine needs with no image-decode dependencies: turning a compiled
// payload back into pixels (`deserialise`) and the box-filter `downscale_rgba`
// the build pipeline uses to cap oversized source maps.
//
// Payload format (little-endian):
//   u32  width
//   u32  height
//   width * height * 4 bytes   RGBA, one byte per channel, row-major

// Deserialise a packed payload back into (width, height, RGBA pixel bytes).
//
// Called by GraphicsSystem at runtime to recover texture dimensions before
// uploading to the GPU.
pub fn deserialise(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    if bytes.len() < 8 {
        return Err(format!(
            "texture payload too short: {} bytes (need at least 8 for header)",
            bytes.len()
        ));
    }
    let width = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let expected = 8 + (width as usize) * (height as usize) * 4;
    if bytes.len() < expected {
        return Err(format!(
            "texture payload too short for {}x{}: need {} bytes, got {}",
            width,
            height,
            expected,
            bytes.len()
        ));
    }
    Ok((width, height, bytes[8..expected].to_vec()))
}

// Box-filter an RGBA image down so its longest edge is at most `max_size`. A
// `max_size` of 0 (or an image already within budget) returns the input
// unchanged. Used to keep oversized source maps (4K+ DDS) from exploding the
// compiled blob, which stores raw RGBA8.
pub fn downscale_rgba(
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    max_size: u32,
) -> (u32, u32, Vec<u8>) {
    if max_size == 0 || (width <= max_size && height <= max_size) {
        return (width, height, pixels);
    }
    let scale = (width.max(height) as f32 / max_size as f32).ceil() as u32;
    let scale = scale.max(2);
    let dst_w = (width / scale).max(1);
    let dst_h = (height / scale).max(1);

    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for sy in 0..scale {
                let src_y = dy * scale + sy;
                if src_y >= height {
                    break;
                }
                for sx in 0..scale {
                    let src_x = dx * scale + sx;
                    if src_x >= width {
                        break;
                    }
                    let si = ((src_y * width + src_x) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += pixels[si + c] as u32;
                    }
                    n += 1;
                }
            }
            let di = ((dy * dst_w + dx) * 4) as usize;
            for c in 0..4 {
                out[di + c] = acc[c].checked_div(n).unwrap_or(0) as u8;
            }
        }
    }
    (dst_w, dst_h, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_rgba_noop_within_budget() {
        let px = vec![1u8; 8 * 8 * 4];
        let (w, h, out) = downscale_rgba(8, 8, px.clone(), 16);
        assert_eq!((w, h), (8, 8));
        assert_eq!(out, px);
    }

    #[test]
    fn downscale_rgba_halves_oversized() {
        // 8x8 solid mid-grey -> capped at 4 -> 4x4, value preserved by averaging.
        let px = vec![128u8; 8 * 8 * 4];
        let (w, h, out) = downscale_rgba(8, 8, px, 4);
        assert_eq!((w, h), (4, 4));
        assert_eq!(out.len(), 4 * 4 * 4);
        assert!(out.iter().all(|&v| v == 128));
    }

    #[test]
    fn downscale_rgba_zero_max_is_noop() {
        let px = vec![7u8; 4 * 4 * 4];
        let (w, h, out) = downscale_rgba(4, 4, px.clone(), 0);
        assert_eq!((w, h), (4, 4));
        assert_eq!(out, px);
    }
}
