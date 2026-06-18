// src/texture.rs
//
// Compiles a Texture component's args into the binary payload that
// GraphicsSystem reads at runtime, and decodes file-backed texture sources
// (PNG / JPEG / DDS / TGA / glb-embedded images) into RGBA pixels for both the
// build pipeline and the `cn debug` asset hot-reload path.
//
// Payload format (little-endian):
//   u32  width
//   u32  height
//   width * height * 4 bytes   RGBA, one byte per channel, row-major
//
// File-backed textures are decoded by extension: PNG and JPEG files on disk,
// or an image embedded in a `.glb`. Procedural generators produce pixel data
// directly and require no file I/O.
//
// Adding a new procedural generator
// 1. Add a branch to the match in compile_texture_payload().
// 2. Write a private generate_* function that returns (u32, u32, Vec<u8>)
//    for (width, height, RGBA pixels).
// 3. No other files need to change.

// The pre-compiled payload `deserialise` and the resolution-cap `downscale_rgba`
// stay in concinnity-core (no image-decode deps); the file -> pixels decoders
// below live here in the build crate alongside the png / jpeg / gltf crates.
use concinnity_core::assets::Texture;
use concinnity_core::build::texture::downscale_rgba;

// Validate the texture generator name in args without generating pixel data.
pub fn validate_texture_generator(args: &serde_json::Value) -> Result<(), String> {
    let generator = args.get("generator").and_then(|v| v.as_str()).unwrap_or("");
    match generator {
        "checker" | "brick" | "concrete" | "grass" | "sky" | "wood" | "tile" | "metal"
        | "terrain" | "stone" | "plaster" | "" => Ok(()),
        other => Err(format!("unknown texture generator '{other}'")),
    }
}

// Compile a Texture component's JSON args into a packed binary payload.
pub fn compile_texture_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let tex: Texture = serde_json::from_value(args.clone())
        .map_err(|e| format!("Texture: invalid args: {}", e))?;

    let (width, height, pixels) = match tex.generator.as_str() {
        "checker" => generate_checker(tex.resolution),
        "brick" => generate_brick(tex.resolution),
        "concrete" => generate_concrete(tex.resolution),
        "grass" => generate_grass(tex.resolution),
        "sky" => generate_sky(tex.resolution),
        "wood" => generate_wood(tex.resolution),
        "tile" => generate_tile(tex.resolution),
        "metal" => generate_metal(tex.resolution),
        "terrain" => generate_terrain(tex.resolution),
        "stone" => generate_stone(tex.resolution),
        "plaster" => generate_plaster(tex.resolution),
        "" => {
            if tex.source.is_empty() {
                return Err("file-backed Texture requires a `source` path".to_string());
            }
            let (w, h, px) = decode_file_source(&tex.source, tex.image_index)?;
            downscale_rgba(w, h, px, tex.max_size)
        }
        other => return Err(format!("unknown texture generator '{other}'")),
    };

    Ok(serialise(width, height, &pixels))
}

fn serialise(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8 + pixels.len());
    buf.extend_from_slice(&width.to_le_bytes());
    buf.extend_from_slice(&height.to_le_bytes());
    buf.extend_from_slice(pixels);
    buf
}

// File-backed source decode
//
// Turns a PNG / JPEG / DDS / TGA file on disk, or an image embedded in a
// `.glb`, into (width, height, RGBA8 pixels). `compile_texture_payload` uses
// this for the file-backed branch; the `cn debug` asset hot-reload path calls
// `decode_source` directly to re-decode a changed file. Both share one dispatch
// so a texture that builds also hot-reloads.

// Decode a file-backed texture source the same way the build pipeline does at
// build time. Dispatches between a PNG / JPEG on disk and an image embedded in
// a `.glb` based on the source extension. Exposed for the runtime asset
// hot-reload path (`cn debug` only); production never calls this, it reads the
// compiled payload via `concinnity_core::build::texture::deserialise` instead.
pub fn decode_source(source: &str, image_index: u32) -> Result<(u32, u32, Vec<u8>), String> {
    decode_file_source(source, image_index)
}

// Single source of truth for decoding a file-backed texture source into
// (width, height, RGBA pixels). Dispatches on the lower-cased extension: an
// image embedded in a `.glb`, a `.jpg` / `.jpeg`, a `.dds`, a `.tga`, or a PNG
// (the default). Both the build-time compiler and the runtime [`decode_source`]
// delegate here, so the two paths always accept the same set of formats: a
// texture that builds also hot-reloads.
pub fn decode_file_source(source: &str, image_index: u32) -> Result<(u32, u32, Vec<u8>), String> {
    let lower = source.to_lowercase();
    if lower.ends_with(".glb") {
        load_glb_image(source, image_index)
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        load_jpg(source)
    } else if lower.ends_with(".dds") {
        load_dds(source)
    } else if lower.ends_with(".tga") {
        load_tga(source)
    } else {
        load_png(source)
    }
}

// Decode a block-compressed DDS (DXT1/DXT5/ATI2) from disk into RGBA.
fn load_dds(source: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(source)
        .map_err(|e| format!("failed to open DDS texture '{}': {}", source, e))?;
    concinnity_core::build::dds::decode_dds(&bytes).map_err(|e| format!("'{}': {}", source, e))
}

// Decode a Targa (.tga) image from disk into RGBA.
fn load_tga(source: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(source)
        .map_err(|e| format!("failed to open TGA texture '{}': {}", source, e))?;
    concinnity_core::build::tga::decode_tga(&bytes).map_err(|e| format!("'{}': {}", source, e))
}

// Decode a PNG file from disk into (width, height, RGBA pixels).
fn load_png(source: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let bytes = std::fs::read(source)
        .map_err(|e| format!("failed to open texture source '{}': {}", source, e))?;
    decode_png_bytes(&bytes).map_err(|e| format!("'{}': {}", source, e))
}

// Decode a JPEG file to RGBA. The `jpeg-decoder` crate (already on the
// dep tree for glb image fallback) handles baseline + progressive JPEGs
// in RGB / Grayscale / CMYK colour modes; we normalise to RGBA so the
// pipeline downstream of this function doesn't need a per-format branch.
// Used for PolyHaven-style PBR sets where the diffuse + roughness
// channels ship as `.jpg`.
fn load_jpg(source: &str) -> Result<(u32, u32, Vec<u8>), String> {
    use jpeg_decoder::PixelFormat;
    let bytes = std::fs::read(source)
        .map_err(|e| format!("failed to open JPEG texture '{}': {}", source, e))?;
    let mut decoder = jpeg_decoder::Decoder::new(std::io::Cursor::new(bytes));
    let raw = decoder
        .decode()
        .map_err(|e| format!("'{}': failed to decode JPEG: {}", source, e))?;
    let info = decoder
        .info()
        .ok_or_else(|| format!("'{}': JPEG has no info after decode", source))?;
    let w = info.width as u32;
    let h = info.height as u32;
    let pixels = match info.pixel_format {
        PixelFormat::RGB24 => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for chunk in raw.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        PixelFormat::L8 => {
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for &v in &raw {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        PixelFormat::L16 => {
            // 16-bit luminance, downsample to 8-bit so the RGBA8 output
            // upload path stays unchanged. Two bytes per source pixel,
            // little-endian.
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for chunk in raw.chunks_exact(2) {
                let v = u16::from_le_bytes([chunk[0], chunk[1]]);
                let v8 = (v >> 8) as u8;
                out.extend_from_slice(&[v8, v8, v8, 255]);
            }
            out
        }
        PixelFormat::CMYK32 => {
            // Approximate CMYK->RGB via the standard `R = (1-C)*(1-K)` etc.
            // Good enough for the rare PolyHaven asset that ships CMYK.
            let mut out = Vec::with_capacity(w as usize * h as usize * 4);
            for chunk in raw.chunks_exact(4) {
                let c = chunk[0] as f32 / 255.0;
                let m = chunk[1] as f32 / 255.0;
                let y = chunk[2] as f32 / 255.0;
                let k = chunk[3] as f32 / 255.0;
                let r = ((1.0 - c) * (1.0 - k) * 255.0) as u8;
                let g = ((1.0 - m) * (1.0 - k) * 255.0) as u8;
                let b = ((1.0 - y) * (1.0 - k) * 255.0) as u8;
                out.extend_from_slice(&[r, g, b, 255]);
            }
            out
        }
    };
    Ok((w, h, pixels))
}

// Decode an in-memory PNG byte buffer into (width, height, RGBA pixels). Used
// for both file-on-disk PNGs and PNG image buffers embedded in a .glb.
fn decode_png_bytes(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use png::ColorType;

    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|e| format!("failed to read PNG info: {}", e))?;

    let mut img_data = vec![
        0u8;
        reader
            .output_buffer_size()
            .ok_or("failed to compute PNG output buffer size")?
    ];
    let info = reader
        .next_frame(&mut img_data)
        .map_err(|e| format!("failed to decode PNG frame: {}", e))?;

    let width = info.width;
    let height = info.height;
    let raw = &img_data[..info.buffer_size()];

    // normalise all color types to RGBA
    let pixels = match info.color_type {
        ColorType::Rgba => raw.to_vec(),
        ColorType::Rgb => {
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            for chunk in raw.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        ColorType::GrayscaleAlpha => {
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            for chunk in raw.chunks_exact(2) {
                out.extend_from_slice(&[chunk[0], chunk[0], chunk[0], chunk[1]]);
            }
            out
        }
        ColorType::Grayscale => {
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            for &v in raw {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        other => {
            return Err(format!(
                "unsupported PNG color type {:?}; convert to RGBA or RGB first",
                other
            ));
        }
    };

    Ok((width, height, pixels))
}

// Decode an in-memory JPEG byte buffer into (width, height, RGBA pixels).
// JPEG has no alpha channel, so every output pixel ends up fully opaque.
fn decode_jpeg_bytes(bytes: &[u8]) -> Result<(u32, u32, Vec<u8>), String> {
    use jpeg_decoder::{Decoder, PixelFormat};

    let mut decoder = Decoder::new(std::io::Cursor::new(bytes));
    let raw = decoder
        .decode()
        .map_err(|e| format!("failed to decode JPEG: {}", e))?;
    let info = decoder
        .info()
        .ok_or("JPEG decode succeeded but produced no metadata")?;
    let width = info.width as u32;
    let height = info.height as u32;

    let pixels = match info.pixel_format {
        PixelFormat::RGB24 => {
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            for chunk in raw.chunks_exact(3) {
                out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            out
        }
        PixelFormat::L8 => {
            let mut out = Vec::with_capacity(width as usize * height as usize * 4);
            for &v in &raw {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        other => {
            return Err(format!(
                "unsupported JPEG pixel format {:?}; use RGB or grayscale",
                other
            ));
        }
    };
    Ok((width, height, pixels))
}

// Extract the indexed embedded image from a .glb. The image's bytes can live
// either in a `bufferView` (typical) or as a data URI on `Source::Uri` (rare
// for GLBs but legal). The MIME type selects the decoder.
fn load_glb_image(source: &str, image_index: u32) -> Result<(u32, u32, Vec<u8>), String> {
    let doc = crate::glb::parse_glb(source)?;
    decode_glb_image_from_doc(&doc, source, image_index)
}

// Pulled out of [`load_glb_image`] so the caller can amortise `parse_glb`
// across every texture / mesh / skinned mesh that shares a `.glb` in a
// single asset hot-reload pass: the worker thread keeps a
// `HashMap<String, gltf::Gltf>` and calls this entry point per texture
// instead of re-parsing the file 43+ times.
pub fn decode_glb_image_from_doc(
    doc: &gltf::Gltf,
    source: &str,
    image_index: u32,
) -> Result<(u32, u32, Vec<u8>), String> {
    let blob = doc.blob.as_deref();

    let image = doc
        .document
        .images()
        .nth(image_index as usize)
        .ok_or_else(|| format!("'{}': image_index {} is out of range", source, image_index))?;

    let (bytes, mime_type): (Vec<u8>, String) = match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            let buf = view.buffer();
            let backing = match buf.source() {
                gltf::buffer::Source::Bin => blob.ok_or_else(|| {
                    format!(
                        "'{}': image {} references the binary chunk but the GLB has no blob",
                        source, image_index
                    )
                })?,
                gltf::buffer::Source::Uri(_) => {
                    return Err(format!(
                        "'{}': image {} lives in an external buffer; only embedded \
                         GLB binary chunks are supported",
                        source, image_index
                    ));
                }
            };
            let start = view.offset();
            let end = start + view.length();
            if end > backing.len() {
                return Err(format!(
                    "'{}': image {} bufferView [{}, {}) exceeds blob size {}",
                    source,
                    image_index,
                    start,
                    end,
                    backing.len()
                ));
            }
            (backing[start..end].to_vec(), mime_type.to_string())
        }
        gltf::image::Source::Uri { .. } => {
            return Err(format!(
                "'{}': image {} uses an external URI; only embedded images in the \
                 GLB binary chunk are supported",
                source, image_index
            ));
        }
    };

    match mime_type.as_str() {
        "image/png" => decode_png_bytes(&bytes)
            .map_err(|e| format!("'{}': image {}: {}", source, image_index, e)),
        "image/jpeg" => decode_jpeg_bytes(&bytes)
            .map_err(|e| format!("'{}': image {}: {}", source, image_index, e)),
        other => Err(format!(
            "'{}': image {} has unsupported MIME type '{}'; only image/png and \
             image/jpeg are handled",
            source, image_index, other
        )),
    }
}

// 8x8 grey/white checkerboard tiled across the requested resolution.
// Cell size scales so the pattern stays visually consistent regardless of
// resolution.
fn generate_checker(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(8);
    let cell = (size / 8).max(1);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let cx = x / cell;
            let cy = y / cell;
            let v: u8 = if (cx + cy).is_multiple_of(2) { 200 } else { 80 };
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    (size, size, pixels)
}

// Running-bond brick pattern: warm terracotta bricks separated by light grey
// mortar lines.  Every other row is offset by half a brick width.
fn generate_brick(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(8);
    // brick proportions: roughly 3:1 width-to-height ratio per brick
    let brick_h = (size / 8).max(1);
    let brick_w = (size / 4).max(1);
    let mortar = (brick_h / 4).max(1);

    let brick_color = [178u8, 90, 62, 255]; // terracotta
    let mortar_color = [200u8, 195, 188, 255]; // off-white

    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        let row = y / brick_h;
        let local_y = y % brick_h;
        let is_mortar_row = local_y < mortar;
        // every other row of bricks is offset by half a brick width
        let offset = if row % 2 == 1 { brick_w / 2 } else { 0 };

        for x in 0..size {
            let shifted_x = (x + size - offset) % size;
            let local_x = shifted_x % brick_w;
            let is_mortar_col = local_x < mortar;

            let color = if is_mortar_row || is_mortar_col {
                mortar_color
            } else {
                // small per-brick brightness variation for visual variety
                let brick_idx = (shifted_x / brick_w + row * 4) % 5;
                let delta = (brick_idx * 8) as i16;
                let r = (brick_color[0] as i16 + delta - 16).clamp(0, 255) as u8;
                let g = (brick_color[1] as i16 + delta / 2 - 8).clamp(0, 255) as u8;
                [r, g, brick_color[2], 255]
            };
            pixels.extend_from_slice(&color);
        }
    }
    (size, size, pixels)
}

// Low-frequency smooth noise approximating a bare concrete surface.
// Uses a deterministic hash so the output is identical across builds.
fn generate_concrete(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(8);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let v = smooth_noise(x, y, size);
            // concrete sits in the 140-190 grey range
            let g = 140u8.saturating_add((v >> 2) as u8).min(190);
            pixels.extend_from_slice(&[g, g, g, 255]);
        }
    }
    (size, size, pixels)
}

// Natural grass: short blade-like strokes of varied green over a dark soil base.
//
// Two layers:
//   1. A soil base using low-frequency noise in earthy brown-green tones.
//   2. Blade strokes: thin near-vertical bright green lines placed at
//      deterministic positions, varying in height and hue between blades.
//
// No RNG is used; all variation comes from lcg_hash so the output is identical
// across builds.
fn generate_grass(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = vec![0u8; (size * size * 4) as usize];

    // base soil layer: low-frequency noise in earthy green-brown tones
    for y in 0..size {
        for x in 0..size {
            let v = smooth_noise(x, y, size);
            // bias toward green-brown; darker than the blades to read as ground
            let r = 38u8.saturating_add((v >> 3) as u8);
            let g = 52u8.saturating_add((v >> 2) as u8);
            let b = 18u8.saturating_add((v >> 4) as u8);
            let idx = ((y * size + x) * 4) as usize;
            pixels[idx] = r;
            pixels[idx + 1] = g;
            pixels[idx + 2] = b;
            pixels[idx + 3] = 255;
        }
    }

    // blade layer: one blade per cell in a grid, offset and height varied by hash
    let cell = (size / 32).max(2);
    let num_cells = size / cell;

    for cy in 0..num_cells {
        for cx in 0..num_cells {
            // hash the cell position to derive blade properties
            let h1 = lcg_hash(cx.wrapping_mul(1619).wrapping_add(cy.wrapping_mul(31337)));
            let h2 = lcg_hash(h1);
            let h3 = lcg_hash(h2);

            // horizontal position within the cell (0..cell range)
            let bx = cx * cell + (h1 & 0xFF) % cell;
            // blade height: 40-90% of cell height
            let blade_h = cell / 2 + (h2 & 0xFF) % (cell / 2).max(1);
            // hue variation: shift red and green channels slightly per blade
            let r_shift = (h3 & 0x0F) as i16 - 8;
            let g_shift = ((h3 >> 4) & 0x1F) as i16;

            let base_y = (cy + 1) * cell; // bottom of the blade row
            for dy in 0..blade_h {
                let y = base_y.saturating_sub(dy);
                if y >= size || bx >= size {
                    continue;
                }
                // brightness increases toward the blade tip
                let t = dy as f32 / blade_h as f32;
                let bright = (0.6 + t * 0.4).min(1.0);
                let r = ((40.0 + r_shift as f32) * bright).clamp(0.0, 255.0) as u8;
                let g = ((100.0 + g_shift as f32) * bright).clamp(0.0, 255.0) as u8;
                let b = (20.0 * bright).clamp(0.0, 255.0) as u8;
                let idx = ((y * size + bx) * 4) as usize;
                pixels[idx] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }
    }

    (size, size, pixels)
}

// Sky gradient: a vertical band from deep azure at the zenith to a
// warm pale blue-white at the horizon.
//
// The texture is 4 pixels wide and `resolution` pixels tall. Metal maps
// V = 0 to the first row stored in memory and V = 1 to the last row, with
// V = 0 at the top of the image. Row 0 is therefore the zenith colour (deep
// azure) and the last row is the horizon colour (pale blue-white). The skybox
// mesh sets V = 0 on wall top edges (zenith) and V = 1 on wall bottom edges
// (horizon), so the gradient renders correctly from ground level to overhead.
//
// Width 4 is sufficient because the gradient varies only in V; any
// horizontal filtering artefacts are invisible on a featureless sky.
fn generate_sky(resolution: u32) -> (u32, u32, Vec<u8>) {
    let height = resolution.max(64);
    let width = 4u32;
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);

    // zenith colour: deep azure
    let zenith = [28u8, 82u8, 185u8];
    // mid-sky colour: clear cornflower blue
    let mid = [100u8, 160u8, 220u8];
    // horizon colour: pale blue-white
    let horizon = [195u8, 220u8, 240u8];

    // row 0 = V = 0 = zenith (dark azure, overhead);
    // last row = V = 1 = horizon (pale blue-white, near ground).
    for row in 0..height {
        // t = 1 at zenith (row 0), 0 at horizon (last row)
        let t = 1.0 - row as f32 / (height - 1) as f32;

        let (r, g, b) = if t > 0.5 {
            // zenith -> mid-sky
            let s = (t - 0.5) * 2.0;
            (
                lerp_u8(mid[0], zenith[0], s),
                lerp_u8(mid[1], zenith[1], s),
                lerp_u8(mid[2], zenith[2], s),
            )
        } else {
            // mid-sky -> horizon
            let s = t * 2.0;
            // faint warm haze near the horizon
            let warm = ((1.0 - s) * (1.0 - s) * 18.0) as u8;
            (
                lerp_u8(horizon[0], mid[0], s).saturating_add(warm / 2),
                lerp_u8(horizon[1], mid[1], s).saturating_add(warm / 4),
                lerp_u8(horizon[2], mid[2], s),
            )
        };

        for _ in 0..width {
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    (width, height, pixels)
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

// Bilinear smooth noise: interpolates between four deterministic lattice hashes.
fn smooth_noise(x: u32, y: u32, size: u32) -> u32 {
    let scale = (size / 8).max(1);
    let gx = x / scale;
    let gy = y / scale;
    let fx = (x % scale) as f32 / scale as f32;
    let fy = (y % scale) as f32 / scale as f32;

    let g = |lx: u32, ly: u32| -> f32 {
        let h = lcg_hash(lx.wrapping_mul(1619).wrapping_add(ly.wrapping_mul(31337)));
        (h & 0xFF) as f32
    };

    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let top = lerp(g(gx, gy), g(gx + 1, gy), fx);
    let bot = lerp(g(gx, gy + 1), g(gx + 1, gy + 1), fx);
    lerp(top, bot, fy) as u32
}

// Horizontal wood grain: warm brown base with alternating lighter/darker bands
// driven by a sine wave plus low-frequency noise for a natural feel.
fn generate_wood(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            // Grain runs horizontally; band frequency scales with resolution.
            let freq = size as f32 / 32.0;
            let noise_val = smooth_noise(x, y, size) as f32 / 255.0; // [0, 1]
            let grain = ((y as f32 / size as f32 * freq + noise_val * 0.4) * std::f32::consts::TAU)
                .sin()
                * 0.5
                + 0.5;

            // Base brown range: [130, 80, 40] to [200, 140, 80]
            let r = (130.0 + grain * 70.0) as u8;
            let g = (80.0 + grain * 60.0) as u8;
            let b = (40.0 + grain * 40.0) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }
    (size, size, pixels)
}

// Ceramic tile: square light-grey tiles separated by narrow dark grout lines.
fn generate_tile(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);

    // Tile covers ~90 % of each cell; the remaining ~10 % is grout.
    let tile_count = 4u32; // tiles per axis within the texture
    let cell = size / tile_count;
    let grout = (cell / 10).max(1);

    let tile_color = [210u8, 210, 215, 255];
    let grout_color = [100u8, 100, 105, 255];

    for y in 0..size {
        for x in 0..size {
            let cx = x % cell;
            let cy = y % cell;
            let is_grout = cx < grout || cy < grout;
            if is_grout {
                pixels.extend_from_slice(&grout_color);
            } else {
                // Slight per-tile brightness variation using the tile index as a seed.
                let tx = x / cell;
                let ty = y / cell;
                let seed = lcg_hash(tx.wrapping_mul(17).wrapping_add(ty.wrapping_mul(31)));
                let var = (seed & 0x0F) as i32 - 8; // ±8 brightness offset
                let r = (tile_color[0] as i32 + var).clamp(0, 255) as u8;
                let g = (tile_color[1] as i32 + var).clamp(0, 255) as u8;
                let b = (tile_color[2] as i32 + var).clamp(0, 255) as u8;
                pixels.extend_from_slice(&[r, g, b, 255]);
            }
        }
    }
    (size, size, pixels)
}

// Brushed metal: cool grey base with horizontal brush-stroke bands plus
// a fine vertical highlight stripe to simulate directional polishing.
fn generate_metal(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        // Coarse horizontal band: low-frequency noise along Y.
        let band = smooth_noise(0, y, size) as f32 / 255.0; // [0, 1]

        for x in 0..size {
            // Fine vertical highlight: a narrow specular stripe.
            let highlight_u = (x as f32 / size as f32 - 0.5).abs();
            let highlight = (1.0 - highlight_u * 6.0).clamp(0.0, 1.0) * 0.15;

            // Base: cool grey [165, 172, 178] modulated by band noise.
            let base = 165.0 + band * 30.0 + highlight * 60.0;
            let r = (base * 0.97).clamp(0.0, 255.0) as u8;
            let g = (base).clamp(0.0, 255.0) as u8;
            let b = (base * 1.04).clamp(0.0, 255.0) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }
    (size, size, pixels)
}

// Natural terrain ground: earthy soil and rock blended via smooth noise.
//
// The terrain mesh assigns UV = [world_x, world_z], so this texture tiles
// across the surface at roughly one repeat per metre. Two noise octaves produce
// broad colour zones (sand → earthy brown → rocky grey) with finer variation
// layered on top for ground-level detail.
fn generate_terrain(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            // Low-frequency zone: broad patches of sand/soil/rock.
            let zone = smooth_noise(x, y, size) as f32 / 255.0; // [0, 1]
            // High-frequency detail: fine grain variation.
            let detail = smooth_noise(
                x.wrapping_mul(3).wrapping_add(137),
                y.wrapping_mul(3).wrapping_add(211),
                size,
            ) as f32
                / 255.0;

            // Blend between three ground types by zone value:
            //   0.0-0.35  sandy soil  [185, 155, 100]
            //   0.35-0.65 earthy brown [130, 100, 65]
            //   0.65-1.0  rocky grey  [120, 115, 108]
            let (base_r, base_g, base_b) = if zone < 0.35 {
                let t = zone / 0.35;
                (
                    lerp_u8(185, 130, t),
                    lerp_u8(155, 100, t),
                    lerp_u8(100, 65, t),
                )
            } else {
                let t = (zone - 0.35) / 0.65;
                (
                    lerp_u8(130, 120, t),
                    lerp_u8(100, 115, t),
                    lerp_u8(65, 108, t),
                )
            };

            // Apply fine detail as a brightness modulation (±15 levels).
            let d = (detail * 30.0) as i32 - 15;
            let r = (base_r as i32 + d).clamp(0, 255) as u8;
            let g = (base_g as i32 + d / 2).clamp(0, 255) as u8;
            let b = (base_b as i32 + d / 3).clamp(0, 255) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    (size, size, pixels)
}

// Rough-cut stone: dark grey base with natural variation and faint horizontal
// stratification lines, suitable for cave walls, dungeon floors, or fortress
// exteriors. Colour sits in the 60-120 grey range with a cool blue-grey cast.
fn generate_stone(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            // Base noise: mid-frequency variation in cool grey tones.
            let base = smooth_noise(x, y, size) as f32 / 255.0;
            // Fine detail: higher-frequency grain.
            let detail = smooth_noise(
                x.wrapping_mul(5).wrapping_add(73),
                y.wrapping_mul(5).wrapping_add(149),
                size,
            ) as f32
                / 255.0;
            // Horizontal stratification: faint banding every ~1/16 of texture height.
            let strat_period = (size / 16).max(4);
            let strat = smooth_noise(x / 4, y / strat_period, size / strat_period) as f32 / 255.0;

            // Combine layers: base controls overall tone, detail adds grain,
            // strat adds subtle horizontal banding.
            let t = base * 0.6 + detail * 0.25 + strat * 0.15;

            // Cool grey-blue stone: [60, 65, 75] to [115, 118, 125]
            let r = (60.0 + t * 55.0) as u8;
            let g = (65.0 + t * 53.0) as u8;
            let b = (75.0 + t * 50.0) as u8;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }

    (size, size, pixels)
}

// Smooth painted plaster: near-white base with very low-frequency noise that
// simulates slight surface irregularities from hand-application. Suitable for
// interior room walls, ceilings, and any neutral indoor surface. Colour sits
// in the 200-240 range, slightly warm (cream-white rather than pure white).
fn generate_plaster(resolution: u32) -> (u32, u32, Vec<u8>) {
    let size = resolution.max(64);
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);

    for y in 0..size {
        for x in 0..size {
            // Very low-frequency noise for subtle surface undulation.
            let scale = (size / 4).max(1);
            let gx = x / scale;
            let gy = y / scale;
            let fx = (x % scale) as f32 / scale as f32;
            let fy = (y % scale) as f32 / scale as f32;
            let g = |lx: u32, ly: u32| -> f32 {
                let h = lcg_hash(lx.wrapping_mul(1619).wrapping_add(ly.wrapping_mul(31337)));
                (h & 0xFF) as f32 / 255.0
            };
            let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
            let coarse = lerp(
                lerp(g(gx, gy), g(gx + 1, gy), fx),
                lerp(g(gx, gy + 1), g(gx + 1, gy + 1), fx),
                fy,
            );

            // Fine grain: very subtle texture.
            let fine = smooth_noise(
                x.wrapping_mul(7).wrapping_add(211),
                y.wrapping_mul(7).wrapping_add(83),
                size,
            ) as f32
                / 255.0;

            let t = coarse * 0.7 + fine * 0.3;

            // Cream-white: [210, 205, 195] to [240, 238, 230]
            let r = (210.0 + t * 30.0) as u8;
            let g_ch = (205.0 + t * 33.0) as u8;
            let b = (195.0 + t * 35.0) as u8;
            pixels.extend_from_slice(&[r, g_ch, b, 255]);
        }
    }

    (size, size, pixels)
}

fn lcg_hash(mut v: u32) -> u32 {
    v = v.wrapping_mul(1664525).wrapping_add(1013904223);
    v ^= v >> 16;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // The reload decoder dispatches by extension, so a missing file's error
    // text reveals which loader it routed to. These lock in parity with the
    // build-time path and guard the regression where `.jpg` reloads fell
    // through to the PNG decoder and failed with "Invalid PNG signature".

    #[test]
    fn decode_source_routes_jpg_to_jpeg_loader() {
        let err = decode_source("zzz_missing_texture.jpg", 0).unwrap_err();
        assert!(
            err.contains("failed to open JPEG texture"),
            "expected .jpg to route to the JPEG loader, got: {err}"
        );
    }

    #[test]
    fn decode_source_routes_jpeg_extension_to_jpeg_loader() {
        let err = decode_source("zzz_missing_texture.jpeg", 0).unwrap_err();
        assert!(
            err.contains("failed to open JPEG texture"),
            "expected .jpeg to route to the JPEG loader, got: {err}"
        );
    }

    #[test]
    fn decode_source_routes_uppercase_jpg_to_jpeg_loader() {
        // Extension matching is case-insensitive (build path lower-cases too).
        let err = decode_source("zzz_missing_texture.JPG", 0).unwrap_err();
        assert!(
            err.contains("failed to open JPEG texture"),
            "expected .JPG to route to the JPEG loader, got: {err}"
        );
    }

    #[test]
    fn decode_source_routes_png_to_png_loader() {
        let err = decode_source("zzz_missing_texture.png", 0).unwrap_err();
        assert!(
            err.contains("failed to open texture source"),
            "expected .png to route to the PNG loader, got: {err}"
        );
    }

    #[test]
    fn decode_source_routes_dds_to_dds_loader() {
        let err = decode_source("zzz_missing_texture.dds", 0).unwrap_err();
        assert!(
            err.contains("failed to open DDS texture"),
            "expected .dds to route to the DDS loader, got: {err}"
        );
    }

    #[test]
    fn decode_source_routes_tga_to_tga_loader() {
        // Mixed-case extension still routes correctly (dispatch lower-cases).
        let err = decode_source("zzz_missing_texture.TGA", 0).unwrap_err();
        assert!(
            err.contains("failed to open TGA texture"),
            "expected .TGA to route to the TGA loader, got: {err}"
        );
    }

    #[test]
    fn compile_texture_payload_checker_succeeds() {
        let args = serde_json::json!({"generator": "checker", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn compile_texture_payload_wood_succeeds() {
        let args = serde_json::json!({"generator": "wood", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn compile_texture_payload_tile_succeeds() {
        let args = serde_json::json!({"generator": "tile", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn compile_texture_payload_metal_succeeds() {
        let args = serde_json::json!({"generator": "metal", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn compile_texture_payload_terrain_succeeds() {
        let args = serde_json::json!({"generator": "terrain", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn compile_texture_payload_unknown_generator_errors() {
        let args = serde_json::json!({"generator": "nonexistent"});
        assert!(compile_texture_payload(&args).is_err());
    }

    #[test]
    fn validate_texture_generator_known_generators_ok() {
        for name in &[
            "checker", "brick", "concrete", "grass", "sky", "wood", "tile", "metal", "terrain",
            "stone", "plaster", "",
        ] {
            let args = serde_json::json!({"generator": name});
            assert!(
                validate_texture_generator(&args).is_ok(),
                "expected ok for generator '{name}'"
            );
        }
    }

    #[test]
    fn validate_texture_generator_unknown_errors() {
        let args = serde_json::json!({"generator": "unknown_generator"});
        let err = validate_texture_generator(&args).unwrap_err();
        assert!(
            err.contains("unknown_generator"),
            "expected error mentioning 'unknown_generator', got: {err}"
        );
    }

    #[test]
    fn validate_texture_generator_missing_field_treated_as_file_backed() {
        let args = serde_json::json!({"source": "tex.png"});
        assert!(validate_texture_generator(&args).is_ok());
    }

    #[test]
    fn generate_wood_output_dimensions_match_resolution() {
        let (w, h, px) = generate_wood(128);
        assert_eq!(w, 128);
        assert_eq!(h, 128);
        assert_eq!(px.len(), (128 * 128 * 4) as usize);
    }

    #[test]
    fn generate_tile_grout_pixels_are_darker() {
        // The top-left pixel of a tile grid is always grout.
        let (_, _, px) = generate_tile(64);
        let grout_r = px[0] as u32;
        // Tile base is 210; grout is 100, grout should be notably darker.
        assert!(grout_r < 150, "expected grout pixel, got r={}", grout_r);
    }

    #[test]
    fn generate_metal_pixel_values_in_valid_range() {
        let (_, _, px) = generate_metal(64);
        for chunk in px.chunks(4) {
            // All channels should be plausible grey values, not black or white.
            assert!(chunk[0] > 50 && chunk[0] < 250);
        }
    }

    #[test]
    fn compile_texture_payload_stone_succeeds() {
        let args = serde_json::json!({"generator": "stone", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn compile_texture_payload_plaster_succeeds() {
        let args = serde_json::json!({"generator": "plaster", "resolution": 64});
        assert!(compile_texture_payload(&args).is_ok());
    }

    #[test]
    fn generate_stone_is_dark_grey() {
        let (_, _, px) = generate_stone(64);
        let avg_r = px.chunks(4).map(|c| c[0] as u32).sum::<u32>() / (64 * 64);
        // stone should be noticeably darker than white (avg R in 60-130 range)
        assert!(
            avg_r > 55 && avg_r < 135,
            "expected dark grey, avg_r={avg_r}"
        );
    }

    #[test]
    fn generate_plaster_is_near_white() {
        let (_, _, px) = generate_plaster(64);
        let avg_r = px.chunks(4).map(|c| c[0] as u32).sum::<u32>() / (64 * 64);
        // plaster should be light (avg R > 210)
        assert!(avg_r > 200, "expected near-white, avg_r={avg_r}");
    }
}
