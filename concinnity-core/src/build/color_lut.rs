// src/build/color_lut.rs
//
// Compiles a ColorLut component's args into the binary payload the renderer
// uploads as a 3D colour-grading LUT. The runtime samples this LUT in the
// composite (post-process) pass with the display-referred sRGB colour as the
// texture coordinate, blending the graded result by `PostProcessConfig`'s
// `lut_strength`.
//
// Two source formats are supported, picked by file extension:
//   - `.cube`  Adobe Cube LUT (plain text): a `LUT_3D_SIZE n` line followed by
//              n*n*n "r g b" float triplets with red varying fastest.
//   - `.png`   A horizontal slice strip: an (n*n)-by-n image of n square
//              slices. Blue selects the slice, red is the column within a
//              slice, green is the row.
//
// Payload format (little-endian):
//   u32  magic     = b"LUT3" = 0x3354554c
//   u32  size      LUT edge length n; the runtime texture is n x n x n
//   u32  format_id = 0  (RGBA8)
//   n*n*n*4 bytes  raw RGBA8, x(red) fastest, then y(green), then z(blue)
//
// The texel order matches both the Metal 3D-texture upload layout and the
// `.cube` data-line order, so the `.cube` path appends triplets verbatim.

pub const LUT_PAYLOAD_MAGIC: u32 = u32::from_le_bytes(*b"LUT3");
pub const LUT_FORMAT_RGBA8: u32 = 0;
pub const LUT_PAYLOAD_HEADER_BYTES: usize = 12;

// Accepted LUT edge lengths. The lower bound keeps trilinear interpolation
// meaningful; the upper bound caps the payload (128³ RGBA8 is 8 MiB).
const MIN_LUT_SIZE: u32 = 2;
const MAX_LUT_SIZE: u32 = 128;

// Classify a `ColorLut` source path by extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutFormat {
    Cube,
    Png,
}

pub fn classify_source(source: &str) -> Result<LutFormat, String> {
    let lower = source.to_ascii_lowercase();
    if lower.ends_with(".cube") {
        Ok(LutFormat::Cube)
    } else if lower.ends_with(".png") {
        Ok(LutFormat::Png)
    } else {
        Err(format!(
            "ColorLut source '{}' must be a .cube or .png file",
            source
        ))
    }
}

// Resolve a `ColorLut` source string into the actual file path on disk. The
// runtime hot-reload watcher needs the resolved path so it can subscribe to
// the correct parent directory; bare filenames are otherwise unfindable
// after the build pipeline runs.
pub fn resolve_source_path(source: &str) -> String {
    resolve_lut_source(source)
}

// Resolve a ColorLut source string into a filesystem path. Bare filenames are
// searched recursively under the assets directory (the same lookup used by
// `EnvironmentMap` and `ShaderStage`); anything with a directory component is
// returned unchanged so absolute / relative paths still work.
pub fn resolve_lut_source(source: &str) -> String {
    let p = std::path::Path::new(source);
    let is_bare = p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true);
    if !is_bare {
        return source.to_string();
    }
    if let Some(path) = crate::world::preset::find_in_assets(source) {
        return path;
    }
    crate::paths::assets_dir()
        .join(source)
        .to_string_lossy()
        .into_owned()
}

// Validate a LUT edge length against the accepted range. Shared by the
// runtime [`deserialise`], the `.cube` parser, and the build crate's PNG-strip
// parser so all three reject the same out-of-range sizes.
pub fn validate_size(size: u32) -> Result<(), String> {
    if !(MIN_LUT_SIZE..=MAX_LUT_SIZE).contains(&size) {
        return Err(format!(
            "ColorLut size {} out of range ({}..={})",
            size, MIN_LUT_SIZE, MAX_LUT_SIZE
        ));
    }
    Ok(())
}

// .cube parsing

// Parse an Adobe Cube LUT into `(size, RGBA8 data)`. Data lines list red,
// green, blue floats in `[0, 1]` with red varying fastest: exactly the
// payload texel order, so triplets are converted and appended verbatim.
pub fn parse_cube(text: &str) -> Result<(u32, Vec<u8>), String> {
    let mut size: Option<u32> = None;
    let mut data: Vec<u8> = Vec::new();

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
            let n: u32 = rest
                .trim()
                .parse()
                .map_err(|e| format!("bad LUT_3D_SIZE value {:?}: {}", rest.trim(), e))?;
            validate_size(n)?;
            size = Some(n);
            continue;
        }
        if line.starts_with("LUT_1D_SIZE") {
            return Err("ColorLut: 1D .cube LUTs are unsupported (need LUT_3D_SIZE)".into());
        }
        // A data line begins with a numeric token. Metadata keywords (TITLE,
        // DOMAIN_MIN, DOMAIN_MAX, ...) start with a letter and are skipped.
        let first = line.as_bytes()[0];
        if !(first.is_ascii_digit() || first == b'+' || first == b'-' || first == b'.') {
            continue;
        }
        let mut it = line.split_whitespace();
        let mut next_channel = |label: &str| -> Result<u8, String> {
            let tok = it
                .next()
                .ok_or_else(|| format!("ColorLut .cube data line missing {} channel", label))?;
            let v: f32 = tok
                .parse()
                .map_err(|e| format!("bad {} value {:?} in .cube: {}", label, tok, e))?;
            Ok((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        };
        let r = next_channel("red")?;
        let g = next_channel("green")?;
        let b = next_channel("blue")?;
        data.extend_from_slice(&[r, g, b, 255]);
    }

    let size = size.ok_or("ColorLut .cube is missing a LUT_3D_SIZE line")?;
    let expected = (size as usize).pow(3) * 4;
    if data.len() != expected {
        return Err(format!(
            "ColorLut .cube has {} entries, expected {} for size {}",
            data.len() / 4,
            expected / 4,
            size
        ));
    }
    Ok((size, data))
}

// The `.png` slice-strip decode (`parse_png_strip` / `load_png_rgba8`) and the
// file-reading `decode_source` live in `concinnity_cook::color_lut`; this
// module keeps the no-dependency `.cube` text parse, the format classifier, the
// payload (de)serialisers, and the shared size validator.

// (de)serialisation

pub fn serialise(size: u32, data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(LUT_PAYLOAD_HEADER_BYTES + data.len());
    buf.extend_from_slice(&LUT_PAYLOAD_MAGIC.to_le_bytes());
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&LUT_FORMAT_RGBA8.to_le_bytes());
    buf.extend_from_slice(data);
    buf
}

// Deserialise a LUT payload back into `(size, RGBA8 bytes)`. The byte slice
// is borrowed from the input. Called by the Metal, Vulkan, and DirectX
// backends at upload time.
pub fn deserialise(bytes: &[u8]) -> Result<(u32, &[u8]), String> {
    if bytes.len() < LUT_PAYLOAD_HEADER_BYTES {
        return Err(format!(
            "ColorLut payload too short: {} bytes (need at least {} for header)",
            bytes.len(),
            LUT_PAYLOAD_HEADER_BYTES
        ));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != LUT_PAYLOAD_MAGIC {
        return Err(format!(
            "ColorLut payload magic 0x{:08x} does not match expected 0x{:08x}",
            magic, LUT_PAYLOAD_MAGIC
        ));
    }
    let size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let format_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if format_id != LUT_FORMAT_RGBA8 {
        return Err(format!(
            "ColorLut payload format_id {} unsupported (only RGBA8)",
            format_id
        ));
    }
    validate_size(size)?;
    let expected = LUT_PAYLOAD_HEADER_BYTES + (size as usize).pow(3) * 4;
    if bytes.len() < expected {
        return Err(format!(
            "ColorLut payload too short for size {}: need {} bytes, got {}",
            size,
            expected,
            bytes.len()
        ));
    }
    Ok((size, &bytes[LUT_PAYLOAD_HEADER_BYTES..expected]))
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // Build the identity .cube text of edge length `n`: each entry maps a
    // coordinate straight back to itself.
    fn identity_cube(n: u32) -> String {
        let mut s = format!("TITLE \"identity\"\nLUT_3D_SIZE {}\n", n);
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let d = (n - 1) as f32;
                    s.push_str(&format!(
                        "{} {} {}\n",
                        r as f32 / d,
                        g as f32 / d,
                        b as f32 / d
                    ));
                }
            }
        }
        s
    }

    #[test]
    fn classify_rejects_unknown_extension() {
        assert!(classify_source("grade.exr").is_err());
        assert_eq!(classify_source("grade.cube").unwrap(), LutFormat::Cube);
        assert_eq!(classify_source("grade.PNG").unwrap(), LutFormat::Png);
    }

    #[test]
    fn parse_cube_identity_round_trips() {
        let (size, data) = parse_cube(&identity_cube(4)).expect("parse");
        assert_eq!(size, 4);
        assert_eq!(data.len(), 4 * 4 * 4 * 4);
        // Texel (r=3,g=0,b=0) is the red corner: index r-fastest = 3.
        assert_eq!(&data[3 * 4..3 * 4 + 4], &[255, 0, 0, 255]);
        // Texel (r=0,g=0,b=3) is the blue corner: index = 3*16.
        assert_eq!(&data[3 * 16 * 4..3 * 16 * 4 + 4], &[0, 0, 255, 255]);
    }

    #[test]
    fn parse_cube_skips_metadata_and_comments() {
        let text = "# a comment\nTITLE \"x\"\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n".to_string()
            + &identity_cube(2)["TITLE \"identity\"\n".len()..];
        let (size, data) = parse_cube(&text).expect("parse");
        assert_eq!(size, 2);
        assert_eq!(data.len(), 2 * 2 * 2 * 4);
    }

    #[test]
    fn parse_cube_rejects_wrong_entry_count() {
        let text = "LUT_3D_SIZE 2\n0 0 0\n1 1 1\n";
        let err = parse_cube(text).unwrap_err();
        assert!(err.contains("entries"), "got: {}", err);
    }

    #[test]
    fn parse_cube_rejects_missing_size() {
        let err = parse_cube("0 0 0\n").unwrap_err();
        assert!(err.contains("LUT_3D_SIZE"), "got: {}", err);
    }

    #[test]
    fn payload_round_trip_via_deserialise() {
        let (size, data) = parse_cube(&identity_cube(3)).expect("parse");
        let blob = serialise(size, &data);
        let (got_size, got) = deserialise(&blob).expect("deserialise");
        assert_eq!(got_size, 3);
        assert_eq!(got, data.as_slice());
    }

    #[test]
    fn deserialise_rejects_bad_magic() {
        let mut blob = serialise(2, &[0u8; 2 * 2 * 2 * 4]);
        blob[0] ^= 0xff;
        assert!(deserialise(&blob).unwrap_err().contains("magic"));
    }
}
