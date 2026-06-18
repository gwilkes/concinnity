// src/font.rs
//
// Build-time font compilation: reads a TTF file, rasterises all printable ASCII
// glyphs using fontdue, packs them into a power-of-two RGBA atlas as a signed
// distance field (SDF), and serialises the result as a blob payload consumed by
// GraphicsSystem at runtime.
//
// Each atlas texel stores a normalised SDF value in [0, 1] where 0.5 = the glyph
// outline. Values > 0.5 are inside; values < 0.5 are outside. The fragment shader
// uses smoothstep + fwidth to reconstruct crisp, scale-independent alpha.

use concinnity_core::assets::Font;
use concinnity_core::build::font::GlyphMetrics;

// Pixels of distance gradient on each side of the glyph edge stored in the atlas
// (in low-resolution / atlas pixels).
const SDF_SPREAD: f32 = 4.0;

// Glyphs are rasterised at this multiple of the requested size, the SDF is
// computed at that high resolution, and then box-filtered down to the atlas
// resolution. Oversampling avoids the staircase artefacts that come from
// thresholding a low-resolution coverage bitmap; the box filter averages
// OVERSAMPLE² high-res samples per atlas texel, so curves stay smooth.
const OVERSAMPLE: u32 = 8;

// Atlas supersampling: the final atlas stores each glyph at this multiple of the
// requested size, while the positional metrics (advance, bearing) stay in
// requested-size units, so on-screen layout is unchanged but the atlas carries
// SUPERSAMPLE times more texels per displayed pixel. HUD chips draw the font
// minified (small scale), so a 1x atlas undersamples and aliases; the extra
// texels let the renderer's trilinear mip chain supersample the glyph down
// cleanly. It also shrinks the SDF spread in screen terms (SDF_SPREAD is fixed
// in atlas texels), so thin strokes hold their contrast at small sizes instead
// of fading out of the antialiasing band.
const SUPERSAMPLE: u32 = 2;

// If 8× still isn't enough, the next lever isn't more oversample
// (diminishing returns and atlas memory growth), but switching the EDT to
// consume fontdue's antialiased coverage values directly instead of
// binary-thresholding them. That's the Gustavson 2012 anti-aliased EDT, and
// it places the implicit surface at sub-pixel positions derived from coverage.

// The engine's bundled default font, shipped in the binary (no external file).
// `BUILTIN_FONT_FILE` is its source filename: companion injection derives the
// auto-injected Font asset's name from it, so a generated default font is named
// exactly as `cn add` would name the same file. Keep it in sync with the
// `include_bytes!` path below.
pub const BUILTIN_FONT_FILE: &str = "Questrial-Regular.ttf";
const BUILTIN_FONT_BYTES: &[u8] = include_bytes!("fonts/Questrial-Regular.ttf");

// Serialise font atlas + metrics into the binary blob payload format.
//
// When `path` is empty or absent the engine's bundled default font is used
// instead of reading from disk, so no external file is required.
pub fn compile_font_payload(args: &serde_json::Value) -> Result<Vec<u8>, String> {
    let font: Font =
        serde_json::from_value(args.clone()).map_err(|e| format!("Font: invalid args: {}", e))?;
    let path = font.path.as_str();
    let logical_size_px = font.size_px as f32;
    // The whole pipeline (rasterise, SDF, pack) runs at the supersampled size, so
    // the atlas and its texel-space metrics come out SUPERSAMPLE times larger.
    // Positional metrics are divided back to `logical_size_px` units at emit time.
    let size_px = logical_size_px * SUPERSAMPLE as f32;

    let ttf_bytes: Vec<u8> = if path.is_empty() {
        BUILTIN_FONT_BYTES.to_vec()
    } else {
        std::fs::read(path).map_err(|e| format!("Font: could not read '{}': {}", path, e))?
    };
    let source = if path.is_empty() { "<built-in>" } else { path };

    let settings = fontdue::FontSettings {
        scale: size_px * OVERSAMPLE as f32,
        ..Default::default()
    };
    let font = fontdue::Font::from_bytes(ttf_bytes.as_slice(), settings)
        .map_err(|e| format!("Font: failed to parse '{}': {}", source, e))?;

    // Rasterise every printable ASCII character (32-126) at OVERSAMPLE × the
    // target size. The SDF is computed at this high resolution and box-filtered
    // back down to atlas resolution; the resulting low-res field captures
    // sub-pixel edge positions that a same-resolution threshold would lose.
    let chars: Vec<char> = (32u8..=126u8).map(|b| b as char).collect();
    let rast_size = size_px * OVERSAMPLE as f32;

    let mut bitmaps: Vec<(char, Vec<u8>, fontdue::Metrics)> = Vec::new();

    for &ch in &chars {
        let (metrics, bitmap) = font.rasterize(ch, rast_size);
        bitmaps.push((ch, bitmap, metrics));
    }

    // Atlas layout is planned in low-res (final) pixels and scaled up by
    // OVERSAMPLE for the working high-res atlas, so atlas dimensions and
    // every glyph position downsample cleanly to integer low-res coordinates.
    const PAD_LO: u16 = 4;
    let pad_hi = PAD_LO * OVERSAMPLE as u16;
    let oversample_u16 = OVERSAMPLE as u16;

    // Per-glyph high-res sizes, rounded up to a multiple of OVERSAMPLE so each
    // glyph cell aligns to the low-res grid.
    let glyph_dims_hi: Vec<(u16, u16)> = bitmaps
        .iter()
        .map(|(_, _, m)| {
            (
                round_up_to(m.width as u16, oversample_u16),
                round_up_to(m.height as u16, oversample_u16),
            )
        })
        .collect();

    let max_glyph_w_hi = glyph_dims_hi.iter().map(|(w, _)| *w).max().unwrap_or(0) + pad_hi * 2;
    let max_glyph_h_hi = glyph_dims_hi.iter().map(|(_, h)| *h).max().unwrap_or(0) + pad_hi * 2;

    let glyph_count = bitmaps.len() as u16;
    // Compute the atlas layout in u32: at larger font sizes the high-res glyph
    // stride times the glyph count overflows u16 (a debug-only multiply panic),
    // even though the packed atlas width is then clamped to <= 2048 logical px.
    let ideal_w_hi = (max_glyph_w_hi as u32 + pad_hi as u32) * glyph_count as u32;
    let atlas_w_hi = next_pow2(ideal_w_hi.max(64)).min(2048 * OVERSAMPLE) as u16;
    let glyphs_per_row = (atlas_w_hi / (max_glyph_w_hi + pad_hi)).max(1);
    let rows = glyph_count.div_ceil(glyphs_per_row);
    let atlas_h_hi =
        next_pow2((max_glyph_h_hi as u32 + pad_hi as u32) * rows as u32 + pad_hi as u32) as u16;

    let atlas_w_hi = atlas_w_hi as u32;
    let atlas_h_hi = atlas_h_hi as u32;
    debug_assert_eq!(atlas_w_hi % OVERSAMPLE, 0);
    debug_assert_eq!(atlas_h_hi % OVERSAMPLE, 0);

    // Each glyph is processed in its own cell buffer rather than a shared
    // high-res atlas. This keeps the EDT and box-filter working on a small
    // region (~cell_w×cell_h pixels) instead of the full atlas (~33M pixels),
    // which makes a large difference in unoptimised (debug) builds.
    let cell_w_hi = max_glyph_w_hi as u32; // includes 2×pad_hi on each axis
    let cell_h_hi = max_glyph_h_hi as u32;
    let cell_w_lo = cell_w_hi / OVERSAMPLE;
    let cell_h_lo = cell_h_hi / OVERSAMPLE;
    let cell_n = (cell_w_hi * cell_h_hi) as usize;
    let block_count = OVERSAMPLE * OVERSAMPLE;
    let oversample_f = OVERSAMPLE as f32;
    let sdf_spread = SDF_SPREAD * oversample_f;

    // Final low-res atlas.
    let atlas_w = atlas_w_hi / OVERSAMPLE;
    let atlas_h = atlas_h_hi / OVERSAMPLE;
    let mut atlas = vec![0u8; (atlas_w * atlas_h * 4) as usize];

    // Reusable per-glyph buffers, allocated once, cleared each iteration.
    let mut cell_hi = vec![0u8; cell_n * 4];
    let mut inside_dist2 = vec![0.0f32; cell_n];
    let mut outside_dist2 = vec![0.0f32; cell_n];

    // EDT scratch, sized for the largest cell dimension; reused across calls.
    let max_cell_dim = cell_w_hi.max(cell_h_hi) as usize;
    let mut edt_v = vec![0usize; max_cell_dim];
    let mut edt_z = vec![0.0f32; max_cell_dim + 1];
    let mut edt_row_tmp = vec![0.0f32; cell_w_hi as usize];
    let mut edt_col_src = vec![0.0f32; cell_h_hi as usize];
    let mut edt_col_dst = vec![0.0f32; cell_h_hi as usize];

    let mut metrics_out: Vec<GlyphMetrics> = Vec::new();

    for (i, (ch, bitmap, metrics)) in bitmaps.iter().enumerate() {
        let col = (i as u16) % glyphs_per_row;
        let row = (i as u16) / glyphs_per_row;
        let ax_hi = pad_hi + col * (max_glyph_w_hi + pad_hi);
        let ay_hi = pad_hi + row * (max_glyph_h_hi + pad_hi);

        let gw_raw = metrics.width as u16;
        let gh_raw = metrics.height as u16;

        // Fill cell_hi: zero it, then place glyph coverage at (pad_hi, pad_hi).
        cell_hi.fill(0);
        for py in 0..gh_raw {
            for px in 0..gw_raw {
                let src = (py as usize) * (gw_raw as usize) + px as usize;
                let dst = ((pad_hi as u32 + py as u32) * cell_w_hi + pad_hi as u32 + px as u32)
                    as usize
                    * 4;
                cell_hi[dst] = bitmap[src];
            }
        }

        // Compute per-glyph SDF using pre-allocated scratch.
        cell_coverage_to_sdf(
            &mut cell_hi,
            cell_w_hi as usize,
            cell_h_hi as usize,
            sdf_spread,
            &mut inside_dist2,
            &mut outside_dist2,
            &mut edt_v,
            &mut edt_z,
            &mut edt_row_tmp,
            &mut edt_col_src,
            &mut edt_col_dst,
        );

        // Box-filter the high-res cell into the final low-res atlas.
        // Cell origin in the low-res atlas matches the original layout.
        let cell_ax_lo = col as u32 * (cell_w_lo + PAD_LO as u32);
        let cell_ay_lo = row as u32 * (cell_h_lo + PAD_LO as u32);
        for ly in 0..cell_h_lo {
            for lx in 0..cell_w_lo {
                let mut sum = [0u32; 1];
                for dy in 0..OVERSAMPLE {
                    for dx in 0..OVERSAMPLE {
                        let hx = lx * OVERSAMPLE + dx;
                        let hy = ly * OVERSAMPLE + dy;
                        // All four channels are identical after SDF conversion;
                        // sample only the R channel and replicate on write.
                        let hi_idx = ((hy * cell_w_hi + hx) * 4) as usize;
                        sum[0] += cell_hi[hi_idx] as u32;
                    }
                }
                let ax = cell_ax_lo + lx;
                let ay = cell_ay_lo + ly;
                let lo_idx = ((ay * atlas_w + ax) * 4) as usize;
                let v = (sum[0] / block_count) as u8;
                atlas[lo_idx] = v;
                atlas[lo_idx + 1] = v;
                atlas[lo_idx + 2] = v;
                atlas[lo_idx + 3] = v;
            }
        }

        // Glyph bounding box rounded up to the low-res grid.
        let gw_hi = glyph_dims_hi[i].0;
        let gh_hi = glyph_dims_hi[i].1;

        // atlas_* stay in (supersampled) atlas texels so the UV math addresses
        // the real texture; the positional fields divide by SUPERSAMPLE too so
        // they land in logical requested-size units and on-screen layout is
        // unchanged.
        let ss_f = SUPERSAMPLE as f32;
        metrics_out.push(GlyphMetrics {
            char_code: *ch as u32,
            atlas_x: ax_hi / oversample_u16,
            atlas_y: ay_hi / oversample_u16,
            atlas_w: gw_hi / oversample_u16,
            atlas_h: gh_hi / oversample_u16,
            advance_px: metrics.advance_width / oversample_f / ss_f,
            bearing_x: metrics.xmin as f32 / oversample_f / ss_f,
            bearing_y: (metrics.ymin as f32 + gh_raw as f32) / oversample_f / ss_f,
        });
    }

    serialise(atlas_w, atlas_h, SUPERSAMPLE, &atlas, &metrics_out)
}

fn round_up_to(n: u16, mult: u16) -> u16 {
    n.div_ceil(mult) * mult
}

// 1-D squared Euclidean distance transform (Felzenszwalb-Huttenlocher).
// `f` is either 0.0 (foreground) or a large value (background).
// `d` receives the squared distance to the nearest foreground sample.
// `v` and `z` are caller-supplied scratch buffers of length >= n and >= n+1.
fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    debug_assert_eq!(n, d.len());
    debug_assert!(v.len() >= n);
    debug_assert!(z.len() > n);
    if n == 0 {
        return;
    }

    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    let mut k = 0usize;

    for q in 1..n {
        loop {
            let r = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[r] + (r * r) as f32))
                / (2.0 * q as f32 - 2.0 * r as f32);
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            // z[0] = -INF so s > z[0] is always true; k==0 branch never reached
            if k == 0 {
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for (q, dq) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let r = v[k];
        let diff = q as f32 - r as f32;
        *dq = diff * diff + f[r];
    }
}

// 2-D squared Euclidean distance transform via two separable 1-D passes.
// `out` must be pre-initialised by the caller: 0.0 for foreground, INF for background.
// All scratch slices are caller-provided to avoid per-call allocation.
#[allow(clippy::too_many_arguments)] // caller-provided scratch buffers, kept separate to avoid per-call allocation
fn edt_2d(
    w: usize,
    h: usize,
    out: &mut [f32],
    v: &mut [usize],
    z: &mut [f32],
    row_tmp: &mut [f32],
    col_src: &mut [f32],
    col_dst: &mut [f32],
) {
    // Row pass
    for y in 0..h {
        edt_1d(
            &out[y * w..(y + 1) * w],
            row_tmp,
            &mut v[..w],
            &mut z[..w + 1],
        );
        out[y * w..(y + 1) * w].copy_from_slice(&row_tmp[..w]);
    }

    // Column pass
    for x in 0..w {
        for y in 0..h {
            col_src[y] = out[y * w + x];
        }
        edt_1d(col_src, col_dst, &mut v[..h], &mut z[..h + 1]);
        for y in 0..h {
            out[y * w + x] = col_dst[y];
        }
    }
}

// Convert the R channel of an RGBA cell buffer from raw coverage (0-255) to SDF.
// All buffers are caller-provided so no heap allocation happens per call.
// After conversion every channel holds the normalised distance in [0, 255]:
//   128 ≈ glyph outline, >128 = inside, <128 = outside.
#[allow(clippy::too_many_arguments)] // caller-provided scratch buffers, kept separate to avoid per-call allocation
fn cell_coverage_to_sdf(
    cell: &mut [u8],
    w: usize,
    h: usize,
    spread: f32,
    inside_dist2: &mut [f32],
    outside_dist2: &mut [f32],
    edt_v: &mut [usize],
    edt_z: &mut [f32],
    row_tmp: &mut [f32],
    col_src: &mut [f32],
    col_dst: &mut [f32],
) {
    const INF: f32 = 1e9;
    let n = w * h;

    // Initialise EDT grids directly from coverage, skipping the bool_buf pass.
    for i in 0..n {
        let fg = cell[i * 4] > 127;
        inside_dist2[i] = if fg { 0.0 } else { INF };
        outside_dist2[i] = if fg { INF } else { 0.0 };
    }

    edt_2d(
        w,
        h,
        &mut inside_dist2[..n],
        edt_v,
        edt_z,
        row_tmp,
        col_src,
        col_dst,
    );
    edt_2d(
        w,
        h,
        &mut outside_dist2[..n],
        edt_v,
        edt_z,
        row_tmp,
        col_src,
        col_dst,
    );

    let spread2 = spread * spread;
    for i in 0..n {
        // For every pixel exactly one of inside_dist2/outside_dist2 is 0.0
        // (foreground pixels have inside_dist2=0; background have outside_dist2=0).
        // Avoid one sqrt unconditionally, and both sqrts for clamped pixels.
        let v = if inside_dist2[i] == 0.0 {
            let d2 = outside_dist2[i];
            if d2 >= spread2 {
                255
            } else {
                (255.0 * (0.5 + 0.5 * d2.sqrt() / spread)).round() as u8
            }
        } else {
            let d2 = inside_dist2[i];
            if d2 >= spread2 {
                0
            } else {
                (255.0 * (0.5 - 0.5 * d2.sqrt() / spread).max(0.0)).round() as u8
            }
        };
        cell[i * 4] = v;
        cell[i * 4 + 1] = v;
        cell[i * 4 + 2] = v;
        cell[i * 4 + 3] = v;
    }
}

fn serialise(
    atlas_w: u32,
    atlas_h: u32,
    supersample: u32,
    rgba: &[u8],
    metrics: &[GlyphMetrics],
) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    out.extend_from_slice(&atlas_w.to_le_bytes());
    out.extend_from_slice(&atlas_h.to_le_bytes());
    out.extend_from_slice(&supersample.to_le_bytes());
    out.extend_from_slice(rgba);
    out.extend_from_slice(&(metrics.len() as u32).to_le_bytes());
    for m in metrics {
        out.extend_from_slice(&m.char_code.to_le_bytes());
        out.extend_from_slice(&m.atlas_x.to_le_bytes());
        out.extend_from_slice(&m.atlas_y.to_le_bytes());
        out.extend_from_slice(&m.atlas_w.to_le_bytes());
        out.extend_from_slice(&m.atlas_h.to_le_bytes());
        out.extend_from_slice(&m.advance_px.to_le_bytes());
        out.extend_from_slice(&m.bearing_x.to_le_bytes());
        out.extend_from_slice(&m.bearing_y.to_le_bytes());
    }
    Ok(out)
}

fn next_pow2(n: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use concinnity_core::build::font::deserialise;

    #[test]
    fn round_trip_serialise() {
        let metrics = vec![
            GlyphMetrics {
                char_code: b'A' as u32,
                atlas_x: 1,
                atlas_y: 1,
                atlas_w: 10,
                atlas_h: 12,
                advance_px: 11.5,
                bearing_x: 0.0,
                bearing_y: 12.0,
            },
            GlyphMetrics {
                char_code: b' ' as u32,
                atlas_x: 15,
                atlas_y: 1,
                atlas_w: 0,
                atlas_h: 0,
                advance_px: 6.0,
                bearing_x: 0.0,
                bearing_y: 0.0,
            },
        ];
        let rgba = vec![128u8; 64 * 64 * 4]; // 64x64 atlas
        let payload = serialise(64, 64, 2, &rgba, &metrics).unwrap();
        let (w, h, supersample, out_rgba, out_metrics) = deserialise(&payload).unwrap();
        assert_eq!(w, 64);
        assert_eq!(h, 64);
        assert_eq!(supersample, 2);
        assert_eq!(out_rgba, rgba);
        assert_eq!(out_metrics.len(), 2);
        assert_eq!(out_metrics[0].char_code, b'A' as u32);
        assert!((out_metrics[0].advance_px - 11.5).abs() < 1e-5);
        assert_eq!(out_metrics[1].char_code, b' ' as u32);
    }

    // An empty or absent `path` compiles the bundled default font at the default
    // 48px size. Before the atlas layout sizes were widened to u32, the high-res
    // glyph stride times the glyph count overflowed u16 and panicked in debug
    // builds at this size.
    #[test]
    fn builtin_font_compiles_at_default_size() {
        for args in [
            serde_json::json!({ "size_px": 48 }),
            serde_json::json!({ "path": "", "size_px": 48 }),
        ] {
            let payload = compile_font_payload(&args).expect("compile bundled font at 48px");
            let (w, h, _supersample, rgba, metrics) = deserialise(&payload).unwrap();
            assert!(w > 0 && h > 0, "atlas has non-zero dimensions");
            assert!(!rgba.is_empty());
            assert!(!metrics.is_empty());
        }
    }
}
