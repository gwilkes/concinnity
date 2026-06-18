// src/build/bcn.rs
//
// Block-compression decoders for the three DDS formats the asset set ships:
// BC1 (DXT1), BC3 (DXT5), and BC5 (ATI2, two-channel). Each decodes a compressed
// buffer into a tightly packed RGBA8 image, row-major, top row first.
//
// BC5 holds only two channels (red, green). It is used for tangent-space normal
// maps, so the blue channel is reconstructed as the unit-length Z and written
// back in [0, 1] encoding, matching the RGB normal maps the shader already reads.

// Expand a 16-bit 565 colour to RGB888.
fn rgb565(c: u16) -> [u8; 3] {
    let r = ((c >> 11) & 0x1F) as u32;
    let g = ((c >> 5) & 0x3F) as u32;
    let b = (c & 0x1F) as u32;
    [
        ((r * 255 + 15) / 31) as u8,
        ((g * 255 + 31) / 63) as u8,
        ((b * 255 + 15) / 31) as u8,
    ]
}

// Decode one BC1 colour block (8 bytes) into 16 RGBA pixels in row-major order.
// When `opaque` is false the 1-bit punch-through alpha mode is honoured (used by
// standalone BC1); inside BC3 the colour block is always the 4-colour opaque mode.
fn decode_bc1_block(block: &[u8], opaque: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let a = rgb565(c0);
    let b = rgb565(c1);

    let mut pal = [[0u8; 4]; 4];
    pal[0] = [a[0], a[1], a[2], 255];
    pal[1] = [b[0], b[1], b[2], 255];
    if opaque || c0 > c1 {
        for k in 0..3 {
            pal[2][k] = ((2 * a[k] as u32 + b[k] as u32) / 3) as u8;
            pal[3][k] = ((a[k] as u32 + 2 * b[k] as u32) / 3) as u8;
        }
        pal[2][3] = 255;
        pal[3][3] = 255;
    } else {
        for k in 0..3 {
            pal[2][k] = ((a[k] as u32 + b[k] as u32) / 2) as u8;
        }
        pal[2][3] = 255;
        pal[3] = [0, 0, 0, 0];
    }

    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let mut out = [[0u8; 4]; 16];
    for (i, px) in out.iter_mut().enumerate() {
        let idx = ((bits >> (2 * i)) & 0x3) as usize;
        *px = pal[idx];
    }
    out
}

// Decode one BC4-style single-channel block (8 bytes) into 16 values. Shared by
// the BC3 alpha block and both halves of a BC5 block.
fn decode_bc4_block(block: &[u8]) -> [u8; 16] {
    let e0 = block[0];
    let e1 = block[1];
    let mut pal = [0u8; 8];
    pal[0] = e0;
    pal[1] = e1;
    if e0 > e1 {
        for i in 0..6 {
            pal[2 + i] = (((6 - i as u32) * e0 as u32 + (1 + i as u32) * e1 as u32) / 7) as u8;
        }
    } else {
        for i in 0..4 {
            pal[2 + i] = (((4 - i as u32) * e0 as u32 + (1 + i as u32) * e1 as u32) / 5) as u8;
        }
        pal[6] = 0;
        pal[7] = 255;
    }

    let bits = (block[2] as u64)
        | (block[3] as u64) << 8
        | (block[4] as u64) << 16
        | (block[5] as u64) << 24
        | (block[6] as u64) << 32
        | (block[7] as u64) << 40;
    let mut out = [0u8; 16];
    for (i, v) in out.iter_mut().enumerate() {
        let idx = ((bits >> (3 * i)) & 0x7) as usize;
        *v = pal[idx];
    }
    out
}

// Walk every 4x4 block left-to-right, top-to-bottom, writing decoded pixels into
// a width*height RGBA8 image and clipping blocks that overhang the edges.
fn assemble<F>(
    data: &[u8],
    width: u32,
    height: u32,
    block_bytes: usize,
    mut decode_block: F,
) -> Result<Vec<u8>, String>
where
    F: FnMut(&[u8]) -> [[u8; 4]; 16],
{
    let bx = width.div_ceil(4);
    let by = height.div_ceil(4);
    let needed = bx as usize * by as usize * block_bytes;
    if data.len() < needed {
        return Err(format!(
            "block-compressed data too short: have {}, need {} for {}x{}",
            data.len(),
            needed,
            width,
            height
        ));
    }

    let mut out = vec![0u8; (width as usize) * (height as usize) * 4];
    for byi in 0..by {
        for bxi in 0..bx {
            let off = (byi as usize * bx as usize + bxi as usize) * block_bytes;
            let px = decode_block(&data[off..off + block_bytes]);
            for ry in 0..4u32 {
                for rx in 0..4u32 {
                    let x = bxi * 4 + rx;
                    let y = byi * 4 + ry;
                    if x < width && y < height {
                        let pi = (ry * 4 + rx) as usize;
                        let oi = ((y * width + x) * 4) as usize;
                        out[oi..oi + 4].copy_from_slice(&px[pi]);
                    }
                }
            }
        }
    }
    Ok(out)
}

// Decode a BC1 (DXT1) buffer into RGBA8. Honours 1-bit punch-through alpha.
pub fn decode_bc1(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    assemble(data, width, height, 8, |block| {
        decode_bc1_block(block, false)
    })
}

// Decode a BC3 (DXT5) buffer into RGBA8: an 8-byte alpha block followed by an
// 8-byte opaque BC1 colour block per 4x4 tile.
pub fn decode_bc3(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    assemble(data, width, height, 16, |block| {
        let alpha = decode_bc4_block(&block[0..8]);
        let mut rgba = decode_bc1_block(&block[8..16], true);
        for (i, px) in rgba.iter_mut().enumerate() {
            px[3] = alpha[i];
        }
        rgba
    })
}

// Decode a BC5 (ATI2) two-channel buffer into RGBA8, reconstructing the blue
// channel as the unit-length normal Z so tangent-space normal maps read
// correctly. Red block first, green block second.
pub fn decode_bc5(data: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    assemble(data, width, height, 16, |block| {
        let red = decode_bc4_block(&block[0..8]);
        let green = decode_bc4_block(&block[8..16]);
        let mut rgba = [[0u8; 4]; 16];
        for i in 0..16 {
            let nx = red[i] as f32 / 255.0 * 2.0 - 1.0;
            let ny = green[i] as f32 / 255.0 * 2.0 - 1.0;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            let bz = ((nz * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            rgba[i] = [red[i], green[i], bz, 255];
        }
        rgba
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A BC1 block with c0 > c1 and all indices 0 yields a solid colour0 fill.
    #[test]
    fn bc1_solid_color0() {
        // color0 = pure red 565 (0xF800), color1 = 0x0000, indices all 0.
        let block = [0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let px = decode_bc1(&block, 4, 4).unwrap();
        for chunk in px.chunks(4) {
            assert_eq!(chunk, &[255, 0, 0, 255]);
        }
    }

    // c0 <= c1 with index 3 selects the transparent-black slot (punch-through).
    #[test]
    fn bc1_punchthrough_alpha() {
        // color0 = 0x0000, color1 = 0xFFFF (so c0 <= c1). All indices = 3 ->
        // every two-bit field is 0b11 -> bytes 0xFF.
        let block = [0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let px = decode_bc1(&block, 4, 4).unwrap();
        for chunk in px.chunks(4) {
            assert_eq!(
                chunk,
                &[0, 0, 0, 0],
                "index-3 in 4-colour-or-less mode is transparent"
            );
        }
    }

    // BC3 alpha endpoints with a0 > a1 and index 0 -> alpha == a0 everywhere.
    #[test]
    fn bc3_alpha_endpoint() {
        let mut block = [0u8; 16];
        block[0] = 200; // a0
        block[1] = 10; // a1 (a0 > a1)
        // alpha indices all 0 (bytes 2..8 = 0) -> alpha 200
        // colour block: color0 = white 565, indices 0 -> white
        block[8] = 0xFF;
        block[9] = 0xFF;
        let px = decode_bc3(&block, 4, 4).unwrap();
        for chunk in px.chunks(4) {
            assert_eq!(chunk[3], 200);
            assert_eq!(&chunk[0..3], &[255, 255, 255]);
        }
    }

    // BC5 with red=green=128 (~0.0 in [-1,1]) reconstructs blue ~ full Z (1.0).
    #[test]
    fn bc5_reconstructs_flat_normal() {
        let mut block = [0u8; 16];
        // red block: both endpoints 128, indices 0 -> red 128 everywhere
        block[0] = 128;
        block[1] = 128;
        // green block: both endpoints 128 -> green 128 everywhere
        block[8] = 128;
        block[9] = 128;
        let px = decode_bc5(&block, 4, 4).unwrap();
        for chunk in px.chunks(4) {
            assert_eq!(chunk[0], 128);
            assert_eq!(chunk[1], 128);
            // nx = ny ~ 0 -> nz ~ 1 -> blue ~ 255
            assert!(chunk[2] >= 253, "expected near-255 blue, got {}", chunk[2]);
            assert_eq!(chunk[3], 255);
        }
    }

    #[test]
    fn rejects_short_buffer() {
        let block = [0u8; 4];
        assert!(decode_bc1(&block, 4, 4).is_err());
    }

    // Non-multiple-of-4 dimensions clip cleanly to the requested size.
    #[test]
    fn handles_non_block_aligned_size() {
        let block = [0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        let px = decode_bc1(&block, 3, 2).unwrap();
        assert_eq!(px.len(), 3 * 2 * 4);
        assert_eq!(&px[0..4], &[255, 0, 0, 255]);
    }
}
