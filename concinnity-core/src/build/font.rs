// src/build/font.rs
//
// Build-time font compilation: reads a TTF file, rasterises all printable ASCII
// glyphs using fontdue, packs them into a power-of-two RGBA atlas as a signed
// distance field (SDF), and serialises the result as a blob payload consumed by
// GraphicsSystem at runtime.
//
// Each atlas texel stores a normalised SDF value in [0, 1] where 0.5 = the glyph
// outline. Values > 0.5 are inside; values < 0.5 are outside. The fragment shader
// uses smoothstep + fwidth to reconstruct crisp, scale-independent alpha.

// Per-glyph metrics stored in the compiled payload.
#[derive(Debug, Clone, Copy)]
pub struct GlyphMetrics {
    pub char_code: u32,
    pub atlas_x: u16,
    pub atlas_y: u16,
    pub atlas_w: u16,
    pub atlas_h: u16,
    pub advance_px: f32,
    pub bearing_x: f32,
    pub bearing_y: f32,
}

// Decoded font payload: atlas width, atlas height, supersample factor, RGBA
// atlas pixels, and per-glyph metrics. Aliased so the decode signature stays
// readable and under clippy's type-complexity bar.
pub type DecodedFont = (u32, u32, u32, Vec<u8>, Vec<GlyphMetrics>);

// Deserialise a font payload back into atlas dimensions, the atlas supersample
// factor, RGBA pixels, and metrics.
pub fn deserialise(bytes: &[u8]) -> Result<DecodedFont, String> {
    if bytes.len() < 12 {
        return Err("font payload too short".into());
    }
    let atlas_w = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let atlas_h = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let supersample = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let pixel_bytes = (atlas_w * atlas_h * 4) as usize;
    let rgba_end = 12 + pixel_bytes;
    if bytes.len() < rgba_end + 4 {
        return Err("font payload truncated before glyph count".into());
    }
    let rgba = bytes[12..rgba_end].to_vec();
    let glyph_count = u32::from_le_bytes(bytes[rgba_end..rgba_end + 4].try_into().unwrap());
    const GLYPH_STRIDE: usize = 4 + 2 + 2 + 2 + 2 + 4 + 4 + 4; // 24 bytes
    let expected = rgba_end + 4 + glyph_count as usize * GLYPH_STRIDE;
    if bytes.len() < expected {
        return Err(format!(
            "font payload truncated: need {} bytes, have {}",
            expected,
            bytes.len()
        ));
    }
    let mut metrics = Vec::with_capacity(glyph_count as usize);
    let mut cursor = rgba_end + 4;
    for _ in 0..glyph_count {
        let char_code = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
        let atlas_x = u16::from_le_bytes(bytes[cursor + 4..cursor + 6].try_into().unwrap());
        let atlas_y = u16::from_le_bytes(bytes[cursor + 6..cursor + 8].try_into().unwrap());
        let atlas_w = u16::from_le_bytes(bytes[cursor + 8..cursor + 10].try_into().unwrap());
        let atlas_h = u16::from_le_bytes(bytes[cursor + 10..cursor + 12].try_into().unwrap());
        let advance_px = f32::from_le_bytes(bytes[cursor + 12..cursor + 16].try_into().unwrap());
        let bearing_x = f32::from_le_bytes(bytes[cursor + 16..cursor + 20].try_into().unwrap());
        let bearing_y = f32::from_le_bytes(bytes[cursor + 20..cursor + 24].try_into().unwrap());
        metrics.push(GlyphMetrics {
            char_code,
            atlas_x,
            atlas_y,
            atlas_w,
            atlas_h,
            advance_px,
            bearing_x,
            bearing_y,
        });
        cursor += GLYPH_STRIDE;
    }
    Ok((atlas_w, atlas_h, supersample, rgba, metrics))
}
