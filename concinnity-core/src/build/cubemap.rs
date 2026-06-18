// src/build/cubemap.rs
//
// Compiles a CubemapTexture component's args into the binary payload that the
// renderer reads at runtime. A cubemap is six square HDR faces stored as
// RGBA32F in face-major order (face 0 → face 5, each face row-major top-down).
//
// Source format: equirectangular Radiance HDR (.hdr / RGBE). The
// equirect is resampled at build time into six cube faces using bilinear
// interpolation in HDR space.
//
// Payload format (little-endian):
//   u32  magic     = b"CUBE" = 0x45425543
//   u32  face_size
//   u32  mip_count = 1
//   u32  format_id = 0  (RGBA32F)
//   6 * face_size * face_size * 4 * 4 bytes  raw RGBA32F, face-major
//
// Face order matches the standard cube convention used by Metal / Vulkan / DX:
//   0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z

pub const CUBE_PAYLOAD_MAGIC: u32 = u32::from_le_bytes(*b"CUBE");
pub const CUBE_FORMAT_RGBA32F: u32 = 0;
pub const CUBE_PAYLOAD_HEADER_BYTES: usize = 16;

// Deserialise a cubemap payload back into (face_size, RGBA32F bytes for 6 faces).
// The byte slice returned is borrowed from the input; callers can reinterpret
// it as `&[f32]` after a length check.
#[allow(dead_code)]
pub fn deserialise(bytes: &[u8]) -> Result<(u32, &[u8]), String> {
    if bytes.len() < CUBE_PAYLOAD_HEADER_BYTES {
        return Err(format!(
            "cubemap payload too short: {} bytes (need at least {} for header)",
            bytes.len(),
            CUBE_PAYLOAD_HEADER_BYTES
        ));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != CUBE_PAYLOAD_MAGIC {
        return Err(format!(
            "cubemap payload magic 0x{:08x} does not match expected 0x{:08x}",
            magic, CUBE_PAYLOAD_MAGIC
        ));
    }
    let face_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let mip_count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let format_id = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    if mip_count != 1 {
        return Err(format!(
            "cubemap payload mip_count {} unsupported (only single-mip supported today)",
            mip_count
        ));
    }
    if format_id != CUBE_FORMAT_RGBA32F {
        return Err(format!(
            "cubemap payload format_id {} unsupported (only RGBA32F supported today)",
            format_id
        ));
    }
    let face_bytes = (face_size as usize) * (face_size as usize) * 4 * 4;
    let expected = CUBE_PAYLOAD_HEADER_BYTES + 6 * face_bytes;
    if bytes.len() < expected {
        return Err(format!(
            "cubemap payload too short for face_size {}: need {} bytes, got {}",
            face_size,
            expected,
            bytes.len()
        ));
    }
    Ok((face_size, &bytes[CUBE_PAYLOAD_HEADER_BYTES..expected]))
}

// HDR (Radiance RGBE) decoder

// Linear-light HDR image. Pixels are row-major top-down RGB triples.
#[derive(Debug)]
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

// Decode a Radiance .hdr blob. Supports both the run-length-encoded
// scanline format and the older raw 4-byte-per-pixel layout.
pub fn decode_hdr(bytes: &[u8]) -> Result<HdrImage, String> {
    // Header: ASCII lines terminated by `\n`, ending with an empty line and
    // then a resolution line of the form `-Y <h> +X <w>` (or sign variants).
    let mut cursor = 0usize;
    let magic = read_line(bytes, &mut cursor)?;
    if !(magic.starts_with("#?RADIANCE") || magic.starts_with("#?RGBE")) {
        return Err(format!("missing Radiance magic header, got {:?}", magic));
    }
    let mut format_seen = false;
    loop {
        let line = read_line(bytes, &mut cursor)?;
        if line.is_empty() {
            break;
        }
        if let Some(rest) = line.strip_prefix("FORMAT=") {
            format_seen = true;
            if rest.trim() != "32-bit_rle_rgbe" {
                return Err(format!("unsupported Radiance FORMAT {:?}", rest));
            }
        }
    }
    if !format_seen {
        return Err("Radiance header missing FORMAT line".into());
    }

    let res_line = read_line(bytes, &mut cursor)?;
    let (width, height, flip_y, flip_x) = parse_resolution(&res_line)?;
    let mut pixels = vec![[0.0f32, 0.0, 0.0]; (width as usize) * (height as usize)];

    let mut scanline = vec![0u8; (width as usize) * 4];
    for scan_y in 0..height as usize {
        read_scanline(bytes, &mut cursor, &mut scanline, width)?;
        let dst_y = if flip_y {
            height as usize - 1 - scan_y
        } else {
            scan_y
        };
        for x in 0..width as usize {
            let src_x = if flip_x { width as usize - 1 - x } else { x };
            let off = src_x * 4;
            let r = scanline[off];
            let g = scanline[off + 1];
            let b = scanline[off + 2];
            let e = scanline[off + 3];
            pixels[dst_y * width as usize + x] = rgbe_to_float(r, g, b, e);
        }
    }
    Ok(HdrImage {
        width,
        height,
        pixels,
    })
}

fn read_line(bytes: &[u8], cursor: &mut usize) -> Result<String, String> {
    let start = *cursor;
    while *cursor < bytes.len() && bytes[*cursor] != b'\n' {
        *cursor += 1;
    }
    if *cursor >= bytes.len() {
        return Err("unexpected end of HDR header".into());
    }
    let line = std::str::from_utf8(&bytes[start..*cursor])
        .map_err(|_| "non-UTF8 in HDR header".to_string())?
        .trim_end_matches('\r')
        .to_string();
    *cursor += 1;
    Ok(line)
}

fn parse_resolution(line: &str) -> Result<(u32, u32, bool, bool), String> {
    // Examples: "-Y 512 +X 1024", "+Y 512 +X 1024"
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() != 4 {
        return Err(format!("malformed Radiance resolution line {:?}", line));
    }
    let parse_axis = |tag: &str, val: &str| -> Result<u32, String> {
        if tag.len() != 2 {
            return Err(format!("bad axis tag {:?}", tag));
        }
        val.parse::<u32>()
            .map_err(|e| format!("bad axis value {:?}: {}", val, e))
    };
    // Y always comes first per the Radiance spec for the supported orientations.
    let (y_tag, y_val, x_tag, x_val) = (parts[0], parts[1], parts[2], parts[3]);
    if !(y_tag.ends_with('Y') && x_tag.ends_with('X')) {
        return Err(format!("unsupported Radiance orientation {:?}", line));
    }
    let height = parse_axis(y_tag, y_val)?;
    let width = parse_axis(x_tag, x_val)?;
    // -Y: scanlines start at the top (the common case); flip_y = false here.
    // +Y: scanlines start at the bottom; flip during decode.
    let flip_y = y_tag.starts_with('+');
    // +X: pixels left-to-right (common); -X reverses.
    let flip_x = x_tag.starts_with('-');
    Ok((width, height, flip_y, flip_x))
}

fn read_scanline(
    bytes: &[u8],
    cursor: &mut usize,
    out: &mut [u8],
    width: u32,
) -> Result<(), String> {
    if (width as usize) * 4 != out.len() {
        return Err("scanline buffer size mismatch".into());
    }
    if *cursor + 4 > bytes.len() {
        return Err("unexpected end of HDR pixel data".into());
    }
    // The new-style RLE scanline header is `0x02 0x02 (width_hi) (width_lo)`
    // with the high bit of width_hi clear. Anything else means an old-format
    // scanline (raw 4-byte pixels): fall back to that path.
    let rle_marker =
        bytes[*cursor] == 0x02 && bytes[*cursor + 1] == 0x02 && (bytes[*cursor + 2] & 0x80) == 0;
    let rle_width = if rle_marker {
        ((bytes[*cursor + 2] as u32) << 8) | bytes[*cursor + 3] as u32
    } else {
        0
    };
    if rle_marker && rle_width == width && (8..=0x7fff).contains(&width) {
        *cursor += 4;
        for ch in 0..4usize {
            // RLE-decode one channel into the out buffer at offset `ch`.
            let mut written = 0usize;
            while written < width as usize {
                if *cursor >= bytes.len() {
                    return Err("RLE scanline truncated".into());
                }
                let lead = bytes[*cursor];
                *cursor += 1;
                if lead > 128 {
                    let run = (lead - 128) as usize;
                    if written + run > width as usize {
                        return Err("RLE run overruns scanline".into());
                    }
                    if *cursor >= bytes.len() {
                        return Err("RLE run byte missing".into());
                    }
                    let val = bytes[*cursor];
                    *cursor += 1;
                    for _ in 0..run {
                        out[written * 4 + ch] = val;
                        written += 1;
                    }
                } else {
                    let run = lead as usize;
                    if written + run > width as usize {
                        return Err("RLE literal overruns scanline".into());
                    }
                    if *cursor + run > bytes.len() {
                        return Err("RLE literal truncated".into());
                    }
                    for i in 0..run {
                        out[(written + i) * 4 + ch] = bytes[*cursor + i];
                    }
                    *cursor += run;
                    written += run;
                }
            }
        }
        Ok(())
    } else {
        // Old-format scanline: raw 4-byte pixels, no RLE.
        let needed = (width as usize) * 4;
        if *cursor + needed > bytes.len() {
            return Err("raw scanline truncated".into());
        }
        out.copy_from_slice(&bytes[*cursor..*cursor + needed]);
        *cursor += needed;
        Ok(())
    }
}

fn rgbe_to_float(r: u8, g: u8, b: u8, e: u8) -> [f32; 3] {
    if e == 0 {
        return [0.0, 0.0, 0.0];
    }
    // f = ((mantissa + 0.5) / 256) * 2^(e - 128). We use the standard ldexp
    // formulation: scale = 2^(e - 128 - 8).
    let scale = (2.0f32).powi(e as i32 - 128 - 8);
    [
        (r as f32 + 0.5) * scale,
        (g as f32 + 0.5) * scale,
        (b as f32 + 0.5) * scale,
    ]
}

// Equirectangular to cubemap resampling

// Resample an equirectangular HDR image into six square cube faces of
// `face_size` pixels. Output is RGBA32F (alpha = 1.0) row-major top-down,
// matching the Metal / Vulkan / DX cube convention.
pub fn equirect_to_cube(hdr: &HdrImage, face_size: u32) -> [Vec<f32>; 6] {
    let f = face_size as usize;
    let mut faces: [Vec<f32>; 6] = std::array::from_fn(|_| vec![0.0; f * f * 4]);
    for (face, face_buf) in faces.iter_mut().enumerate() {
        for y in 0..f {
            for x in 0..f {
                // Map pixel center to NDC [-1, 1].
                let u = (x as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                let v = (y as f32 + 0.5) / face_size as f32 * 2.0 - 1.0;
                let dir = face_uv_to_dir(face, u, v);
                let sample = sample_equirect(hdr, dir);
                let off = (y * f + x) * 4;
                face_buf[off] = sample[0];
                face_buf[off + 1] = sample[1];
                face_buf[off + 2] = sample[2];
                face_buf[off + 3] = 1.0;
            }
        }
    }
    faces
}

// Convert a face index + face UV in NDC [-1, 1] to a world-space direction.
// Face order: 0:+X, 1:-X, 2:+Y, 3:-Y, 4:+Z, 5:-Z.
fn face_uv_to_dir(face: usize, u: f32, v: f32) -> [f32; 3] {
    let d = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        5 => [-u, -v, -1.0],
        _ => unreachable!("invalid cube face index {}", face),
    };
    normalize3(d)
}

fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn sample_equirect(hdr: &HdrImage, dir: [f32; 3]) -> [f32; 3] {
    let phi = dir[2].atan2(dir[0]); // [-π, π]
    let theta = dir[1].clamp(-1.0, 1.0).acos(); // [0, π]
    let u = phi / (2.0 * std::f32::consts::PI) + 0.5;
    let v = theta / std::f32::consts::PI;
    let fx = u * hdr.width as f32 - 0.5;
    let fy = v * hdr.height as f32 - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    let dx = fx - x0 as f32;
    let dy = fy - y0 as f32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let w00 = (1.0 - dx) * (1.0 - dy);
    let w10 = dx * (1.0 - dy);
    let w01 = (1.0 - dx) * dy;
    let w11 = dx * dy;
    let p00 = fetch_wrap(hdr, x0, y0);
    let p10 = fetch_wrap(hdr, x1, y0);
    let p01 = fetch_wrap(hdr, x0, y1);
    let p11 = fetch_wrap(hdr, x1, y1);
    [
        p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
        p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
        p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
    ]
}

fn fetch_wrap(hdr: &HdrImage, x: i32, y: i32) -> [f32; 3] {
    // Horizontal wrap (longitude), vertical clamp (latitude poles).
    let w = hdr.width as i32;
    let h = hdr.height as i32;
    let xw = x.rem_euclid(w);
    let yc = y.clamp(0, h - 1);
    hdr.pixels[(yc * w + xw) as usize]
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_rgbe_one_pixel(r: f32, g: f32, b: f32) -> [u8; 4] {
        let maxv = r.max(g).max(b);
        if maxv < 1e-32 {
            return [0, 0, 0, 0];
        }
        let (mantissa, exp) = frexp_f32(maxv);
        let scale = (mantissa * 256.0) / maxv;
        [
            (r * scale) as u8,
            (g * scale) as u8,
            (b * scale) as u8,
            (exp + 128) as u8,
        ]
    }

    // Manual frexp for tests: std::f32 doesn't expose it stably.
    fn frexp_f32(x: f32) -> (f32, i32) {
        if x == 0.0 {
            return (0.0, 0);
        }
        let bits = x.to_bits();
        let raw_exp = ((bits >> 23) & 0xff) as i32;
        let exp = raw_exp - 126;
        let mantissa_bits = (bits & 0x7f_ffff) | (126 << 23);
        let mantissa = f32::from_bits(mantissa_bits);
        (mantissa, exp)
    }

    fn raw_hdr_blob(width: u32, height: u32, pixels: &[[u8; 4]]) -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"#?RADIANCE\n");
        blob.extend_from_slice(b"FORMAT=32-bit_rle_rgbe\n\n");
        blob.extend_from_slice(format!("-Y {} +X {}\n", height, width).as_bytes());
        for p in pixels {
            blob.extend_from_slice(p);
        }
        blob
    }

    #[test]
    fn decode_hdr_solid_color_old_format() {
        let pixel = synth_rgbe_one_pixel(1.0, 0.5, 0.25);
        let pixels: Vec<[u8; 4]> = std::iter::repeat_n(pixel, 4 * 2).collect();
        let blob = raw_hdr_blob(4, 2, &pixels);
        let img = decode_hdr(&blob).expect("decode");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 2);
        // Rough tolerance: RGBE is lossy.
        for p in &img.pixels {
            assert!((p[0] - 1.0).abs() < 0.02, "R was {}", p[0]);
            assert!((p[1] - 0.5).abs() < 0.02, "G was {}", p[1]);
            assert!((p[2] - 0.25).abs() < 0.02, "B was {}", p[2]);
        }
    }

    #[test]
    fn decode_hdr_rejects_bad_magic() {
        let blob = b"#?NOTHDR\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\x00\x00\x00\x00".to_vec();
        let err = decode_hdr(&blob).unwrap_err();
        assert!(err.contains("magic"), "got: {}", err);
    }

    #[test]
    fn equirect_solid_color_produces_solid_cube() {
        let pixel = [0.8f32, 0.4, 0.1];
        let hdr = HdrImage {
            width: 32,
            height: 16,
            pixels: vec![pixel; 32 * 16],
        };
        let faces = equirect_to_cube(&hdr, 16);
        for (idx, face) in faces.iter().enumerate() {
            assert_eq!(face.len(), 16 * 16 * 4);
            for px in face.chunks_exact(4) {
                assert!((px[0] - pixel[0]).abs() < 1e-4, "face {} R", idx);
                assert!((px[1] - pixel[1]).abs() < 1e-4, "face {} G", idx);
                assert!((px[2] - pixel[2]).abs() < 1e-4, "face {} B", idx);
                assert!((px[3] - 1.0).abs() < 1e-6, "face {} A", idx);
            }
        }
    }

    #[test]
    fn equirect_red_seam_lights_only_the_minus_x_face() {
        // Paint a four-pixel-wide red band on the equirect straddling the
        // longitude = ±π seam (columns {30, 31, 0, 1} for a 32-wide image).
        // The -X face is centered on that longitude; +X is on the opposite
        // side and should see almost no red.
        let mut pixels = vec![[0.0f32; 3]; 32 * 16];
        for y in 0..16 {
            for &x in &[30usize, 31, 0, 1] {
                pixels[y * 32 + x] = [10.0, 0.0, 0.0];
            }
        }
        let hdr = HdrImage {
            width: 32,
            height: 16,
            pixels,
        };
        let faces = equirect_to_cube(&hdr, 16);
        let mean_red = |face: &[f32]| -> f32 {
            let n = face.len() / 4;
            face.chunks_exact(4).map(|p| p[0]).sum::<f32>() / n as f32
        };
        let plus_x = mean_red(&faces[0]);
        let minus_x = mean_red(&faces[1]);
        assert!(
            minus_x > 5.0 * plus_x.max(0.001),
            "-X mean red ({}) should dwarf +X mean red ({})",
            minus_x,
            plus_x
        );
    }
}
