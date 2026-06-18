// src/color_lut.rs
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

// The no-dependency `.cube` parse, the classifier, the (de)serialisers, and the
// size validator stay in concinnity-core; the `.png` slice-strip decode below
// lives here in the build crate alongside the `png` crate.
use concinnity_core::build::color_lut::{
    LutFormat, classify_source, parse_cube, resolve_lut_source, serialise, validate_size,
};

// Validate that args specify a `ColorLut` source with a supported extension.
pub fn validate_color_lut_args(args: &serde_json::Value) -> Result<(), String> {
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if source.is_empty() {
        return Err("ColorLut requires a `source` path".into());
    }
    classify_source(source)?;
    Ok(())
}

// Compile a `ColorLut` component's JSON args into a packed binary payload.
pub fn compile_color_lut_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    validate_color_lut_args(args)?;
    let source = args.get("source").and_then(|v| v.as_str()).unwrap();
    let format = classify_source(source)?;
    let resolved = resolve_lut_source(source);

    let (size, data) = match format {
        LutFormat::Cube => {
            let text = std::fs::read_to_string(&resolved)
                .map_err(|e| format!("failed to read LUT source '{}': {}", resolved, e))?;
            parse_cube(&text)?
        }
        LutFormat::Png => parse_png_strip(&resolved)?,
    };
    Ok(serialise(size, &data))
}

// Decode a ColorLut source path the same way `compile_color_lut_payload` does
// at build time. Dispatches between `.cube` text parse and PNG-strip parse
// based on the resolved file extension. Exposed for the runtime asset
// hot-reload path (`cn debug` only); production reads the compiled payload via
// `concinnity_core::build::color_lut::deserialise` instead. The `source`
// argument is the raw string from the asset declaration; this function applies
// the same asset-dir resolution the build pipeline uses.
pub fn decode_source(source: &str) -> Result<(u32, Vec<u8>), String> {
    let format = classify_source(source)?;
    let resolved = resolve_lut_source(source);
    match format {
        LutFormat::Cube => {
            let text = std::fs::read_to_string(&resolved)
                .map_err(|e| format!("failed to read LUT source '{}': {}", resolved, e))?;
            parse_cube(&text)
        }
        LutFormat::Png => parse_png_strip(&resolved),
    }
}

// .png slice-strip parsing

// Parse a horizontal LUT slice strip PNG into `(size, RGBA8 data)`. The image
// is `(n*n)` wide by `n` tall: n square slices laid left-to-right where blue
// selects the slice, red is the column within a slice and green is the row.
pub fn parse_png_strip(path: &str) -> Result<(u32, Vec<u8>), String> {
    let (width, height, pixels) = load_png_rgba8(path)?;
    let size = height;
    validate_size(size)?;
    if width != size * size {
        return Err(format!(
            "ColorLut strip '{}' is {}x{}; a size-{} strip must be {}x{}",
            path,
            width,
            height,
            size,
            size * size,
            size
        ));
    }
    let n = size as usize;
    let mut data = Vec::with_capacity(n * n * n * 4);
    // Emit in red-fastest, then green, then blue order to match the payload.
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let px = b * n + r;
                let src = (g * width as usize + px) * 4;
                data.extend_from_slice(&pixels[src..src + 4]);
            }
        }
    }
    Ok((size, data))
}

// Decode a PNG into (width, height, RGBA8 pixels). Only RGB / RGBA colour
// types are accepted: a LUT strip is always full colour.
fn load_png_rgba8(path: &str) -> Result<(u32, u32, Vec<u8>), String> {
    use png::ColorType;
    let file = std::fs::File::open(path)
        .map_err(|e| format!("failed to open LUT source '{}': {}", path, e))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("failed to read PNG info for '{}': {}", path, e))?;
    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or("failed to compute PNG output buffer size")?
    ];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("failed to decode PNG frame for '{}': {}", path, e))?;
    let raw = &buf[..info.buffer_size()];
    let pixels = match info.color_type {
        ColorType::Rgba => raw.to_vec(),
        ColorType::Rgb => {
            let mut out = Vec::with_capacity(info.width as usize * info.height as usize * 4);
            for chunk in raw.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        other => {
            return Err(format!(
                "ColorLut strip '{}' has unsupported PNG colour type {:?} (need RGB/RGBA)",
                path, other
            ));
        }
    };
    Ok((info.width, info.height, pixels))
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_args_requires_supported_source() {
        assert!(validate_color_lut_args(&serde_json::json!({})).is_err());
        assert!(validate_color_lut_args(&serde_json::json!({"source": "g.tga"})).is_err());
        validate_color_lut_args(&serde_json::json!({"source": "g.cube"})).expect("ok");
    }
}
