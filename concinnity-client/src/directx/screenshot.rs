// src/directx/screenshot.rs
//
// Headless frame capture for the D3D12 backend. The `cn debug` WS server's
// `screenshot` command routes here (via `RenderBackend::screenshot`) to copy
// the most recently presented swapchain back-buffer into a readback buffer and
// encode it to a PNG on disk. This is the on-GPU verification path the renderer
// otherwise leaves to a human eyeballing the live window: a headless smoke can
// now assert on actual pixels. Mirrors src/vulkan/screenshot.rs.
//
// Capture is synchronous: it idles the GPU (so the last-presented buffer is
// stable and no in-flight command list still references it), copies the
// last-presented back-buffer (still in `PRESENT`) into a `READBACK` buffer on a
// one-shot DIRECT command list, restores the buffer to `PRESENT`, then maps +
// de-pads + decodes + PNG-encodes on the CPU. The readback buffer is sized from
// `GetCopyableFootprints` (D3D12 aligns each row to
// `D3D12_TEXTURE_DATA_PITCH_ALIGNMENT` = 256), so the per-row de-pad below
// strips that padding back to a tight RGBA8 image. A swapchain rebuild clears
// `last_present_index`, so a capture in the brief window before the next
// present returns a clean error rather than reading an unrendered buffer.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::hdr_output::HdrEncoding;

use super::context::DxContext;
use super::texture::{create_buffer, one_shot_submit, transition_barrier};

impl DxContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous; `#[allow(dead_code)]` because it
    // is reached only through the `RenderBackend` vtable (bin-only `cn debug`).
    #[allow(dead_code)]
    pub fn capture_screenshot(&mut self, path: &str) -> Result<String, String> {
        let Some(back_idx) = self.last_present_index else {
            return Err("screenshot: no frame has been presented yet".into());
        };
        let back_buffer = self
            .back_buffers
            .get(back_idx)
            .ok_or("screenshot: stale back-buffer index")?
            .clone();
        let width = self.output_width;
        let height = self.output_height;
        if width == 0 || height == 0 {
            return Err("screenshot: zero-sized swapchain".into());
        }

        // The GPU must be idle: the last-presented buffer is then stable and no
        // in-flight command list still references the resources we touch.
        self.wait_idle();

        // Describe the back-buffer so `GetCopyableFootprints` can hand back the
        // placed-footprint layout (row pitch, total padded size) the copy needs.
        let tex_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: self.swap_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut row_count: u32 = 0;
        let mut row_size: u64 = 0;
        let mut total_size: u64 = 0;
        unsafe {
            self.device.GetCopyableFootprints(
                &tex_desc,
                0,
                1,
                0,
                Some(&mut layout),
                Some(&mut row_count),
                Some(&mut row_size),
                Some(&mut total_size),
            );
        }

        // Host-readable buffer sized for the padded footprint. READBACK heap
        // resources start in COPY_DEST and never need a barrier.
        let readback = create_buffer(
            &self.device,
            total_size,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        // Copy the presented back-buffer into the readback buffer, bracketing
        // with PRESENT <-> COPY_SOURCE barriers so the buffer is left exactly as
        // the next present expects it. `one_shot_submit` fence-waits internally.
        // `pResource` is borrowed (no AddRef) via `transmute_copy`: the field is
        // a `ManuallyDrop`, so a `clone()` here would never be released and would
        // leak one reference per screenshot. Leaking a back-buffer reference in
        // particular would later block `ResizeBuffers`. `readback` / `back_buffer`
        // outlive the synchronous `one_shot_submit` below, so the raw pointers
        // stay valid.
        let dst_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let src_loc = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&back_buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let to_src = transition_barrier(
                &back_buffer,
                D3D12_RESOURCE_STATE_PRESENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[to_src]);
            cmd.CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, None);
            let to_present = transition_barrier(
                &back_buffer,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_PRESENT,
            );
            cmd.ResourceBarrier(&[to_present]);
        })?;

        // Map + de-pad + decode, then always unmap. The readback rows are padded
        // to `layout.Footprint.RowPitch`; `row_size` is the tight byte width.
        let row_pitch = layout.Footprint.RowPitch as usize;
        let tight_row = row_size as usize;
        let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
        unsafe { readback.Map(0, None, Some(&mut map_ptr)) }
            .map_err(|e| format!("screenshot: map readback: {e}"))?;
        // SAFETY: the buffer is `total_size` bytes long and the copy completed
        // (one_shot_submit waits its fence). Read each row's tight span out of
        // the padded footprint into a contiguous source image.
        let mut packed = vec![0u8; tight_row * height as usize];
        for row in 0..height as usize {
            let src = unsafe { (map_ptr as *const u8).add(row * row_pitch) };
            // SAFETY: `tight_row` bytes are valid within each padded row.
            let src_slice = unsafe { std::slice::from_raw_parts(src, tight_row) };
            packed[row * tight_row..(row + 1) * tight_row].copy_from_slice(src_slice);
        }
        unsafe { readback.Unmap(0, None) };

        let rgba = decode_to_rgba8(&packed, self.swap_format, self.hdr_encoding);
        encode_png(path, width, height, &rgba)?;
        Ok(path.to_string())
    }
}

// Bytes per texel for the swapchain colour formats this backend can present.
// The DX swapchain only ever resolves to `B8G8R8A8_UNORM` for SDR or
// `R16G16B16A16_FLOAT` for the HDR (scRGB-linear / PQ-float) path; see
// `init/window.rs`. Unknown formats default to 4, the common 32-bit-texel case.
// The capture path sizes its readback from `GetCopyableFootprints` (which also
// folds in the 256-byte row alignment), so this helper only documents +
// asserts the format-to-texel-size mapping under test.
#[allow(dead_code)]
fn swapchain_bytes_per_pixel(format: DXGI_FORMAT) -> u32 {
    match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT => 8,
        _ => 4,
    }
}

// Convert the tightly-packed readback bytes to opaque RGBA8, decoding per the
// swapchain format. The alpha channel is forced to 255 so the saved PNG is
// opaque regardless of the composited alpha. `encoding` is the resolved HDR
// encoding (None on the SDR path) and only matters for the float swapchain.
fn decode_to_rgba8(raw: &[u8], format: DXGI_FORMAT, encoding: Option<HdrEncoding>) -> Vec<u8> {
    match format {
        DXGI_FORMAT_R16G16B16A16_FLOAT => decode_rgba16f(raw, encoding),
        _ => decode_8bit(raw, format),
    }
}

// 8-bit-per-channel swapchain formats. The DX SDR swapchain is BGRA8 on
// Windows; RGBA8 is handled too for completeness.
fn decode_8bit(raw: &[u8], format: DXGI_FORMAT) -> Vec<u8> {
    let bgra = matches!(
        format,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
    );
    let mut out = Vec::with_capacity(raw.len());
    for px in raw.chunks_exact(4) {
        if bgra {
            out.extend_from_slice(&[px[2], px[1], px[0], 255]);
        } else {
            out.extend_from_slice(&[px[0], px[1], px[2], 255]);
        }
    }
    out
}

// `R16G16B16A16_FLOAT` HDR swapchain (8 B/px, four halfs RGBA). On the
// scRGB-linear path the stored values are linear extended-range (1.0 = SDR
// white), so apply the sRGB OETF to get a valid (non-tonemapped) image. On the
// PQ path the stored values are PQ code values already in [0, 1]; pass them
// through clamped. The PQ capture is not display-ready, but it must still be a
// valid PNG rather than a device-lost crash. Mirrors the Vulkan path.
fn decode_rgba16f(raw: &[u8], encoding: Option<HdrEncoding>) -> Vec<u8> {
    let scrgb = !matches!(encoding, Some(HdrEncoding::Pq));
    let mut out = Vec::with_capacity(raw.len() / 2);
    for px in raw.chunks_exact(8) {
        let r = f16_to_f32(u16::from_le_bytes([px[0], px[1]]));
        let g = f16_to_f32(u16::from_le_bytes([px[2], px[3]]));
        let b = f16_to_f32(u16::from_le_bytes([px[4], px[5]]));
        if scrgb {
            out.extend_from_slice(&[
                linear_to_srgb8(r),
                linear_to_srgb8(g),
                linear_to_srgb8(b),
                255,
            ]);
        } else {
            out.extend_from_slice(&[unorm_to_u8(r), unorm_to_u8(g), unorm_to_u8(b), 255]);
        }
    }
    out
}

// Decode an IEEE 754 half (binary16) to f32. Handles zero, subnormals,
// normals, and inf/NaN. `pub(super)` so the reflection-probe readback
// (`directx/probe.rs`) can decode its RGBA16Float cube faces.
pub(super) fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    let val = match exp {
        0 => mant * 2f32.powi(-24),
        0x1f => {
            if mant == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mant / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    sign * val
}

// sRGB OETF (linear -> display), clamped and quantised to 8-bit. NaN maps to 0.
fn linear_to_srgb8(c: f32) -> u8 {
    if c.is_nan() {
        return 0;
    }
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    unorm_to_u8(s)
}

// Quantise a [0, 1] value to 8-bit with rounding. NaN maps to 0.
fn unorm_to_u8(c: f32) -> u8 {
    if c.is_nan() {
        return 0;
    }
    (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

// Write RGBA8 pixel data to a PNG file.
fn encode_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("screenshot: create {path}: {e}"))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("screenshot: png header: {e}"))?;
    writer
        .write_image_data(rgba)
        .map_err(|e| format!("screenshot: png data: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_pixel_matches_swapchain_formats() {
        // The SDR swapchain is 4 B/px; the float HDR swapchain is 8.
        assert_eq!(swapchain_bytes_per_pixel(DXGI_FORMAT_B8G8R8A8_UNORM), 4);
        assert_eq!(swapchain_bytes_per_pixel(DXGI_FORMAT_R8G8B8A8_UNORM), 4);
        assert_eq!(swapchain_bytes_per_pixel(DXGI_FORMAT_R16G16B16A16_FLOAT), 8);
    }

    #[test]
    fn f16_round_trips_reference_values() {
        assert_eq!(f16_to_f32(0x0000), 0.0); // +0
        assert_eq!(f16_to_f32(0x3c00), 1.0); // 1.0
        assert_eq!(f16_to_f32(0x3800), 0.5); // 0.5
        assert_eq!(f16_to_f32(0x4000), 2.0); // 2.0
        assert_eq!(f16_to_f32(0xbc00), -1.0); // -1.0
        assert!(f16_to_f32(0x7c00).is_infinite()); // +inf
        assert!(f16_to_f32(0x7e00).is_nan()); // NaN
    }

    #[test]
    fn bgra8_is_swizzled_and_made_opaque() {
        // One BGRA pixel (B=10, G=20, R=30, A=40) -> RGBA (30, 20, 10, 255).
        let raw = [10u8, 20, 30, 40];
        let out = decode_to_rgba8(&raw, DXGI_FORMAT_B8G8R8A8_UNORM, None);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn bgra8_srgb_is_also_swizzled() {
        let raw = [10u8, 20, 30, 40];
        let out = decode_to_rgba8(&raw, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, None);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn rgba8_passes_through_with_forced_alpha() {
        let raw = [30u8, 20, 10, 40];
        let out = decode_to_rgba8(&raw, DXGI_FORMAT_R8G8B8A8_UNORM, None);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn scrgb_float_applies_srgb_oetf() {
        // Linear 1.0 -> sRGB 255, 0.0 -> 0, 0.5 -> ~188 (1.055*0.5^(1/2.4)-0.055).
        let mut raw = Vec::new();
        for h in [0x3c00u16, 0x3800, 0x0000, 0x3c00] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let out = decode_to_rgba8(
            &raw,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            Some(HdrEncoding::ExtendedLinear),
        );
        assert_eq!(out[0], 255); // r = linear 1.0
        assert!((out[1] as i32 - 188).abs() <= 1); // g = linear 0.5
        assert_eq!(out[2], 0); // b = linear 0.0
        assert_eq!(out[3], 255); // forced opaque
    }

    #[test]
    fn scrgb_float_clamps_out_of_range() {
        // Extended-range > 1.0 and negative clamp to white / black.
        let mut raw = Vec::new();
        for h in [0x4000u16, 0xbc00, 0x0000, 0x3c00] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let out = decode_to_rgba8(
            &raw,
            DXGI_FORMAT_R16G16B16A16_FLOAT,
            Some(HdrEncoding::ExtendedLinear),
        );
        assert_eq!(out[0], 255); // r = 2.0 clamps high
        assert_eq!(out[1], 0); // g = -1.0 clamps low
    }

    #[test]
    fn pq_float_passes_code_values_through() {
        // PQ code values are already in [0, 1]; no sRGB OETF, just quantise.
        let mut raw = Vec::new();
        for h in [0x3c00u16, 0x3800, 0x0000, 0x3c00] {
            raw.extend_from_slice(&h.to_le_bytes());
        }
        let out = decode_to_rgba8(&raw, DXGI_FORMAT_R16G16B16A16_FLOAT, Some(HdrEncoding::Pq));
        assert_eq!(out[0], 255); // 1.0
        assert_eq!(out[1], 128); // 0.5 -> round(127.5)
        assert_eq!(out[2], 0); // 0.0
        assert_eq!(out[3], 255);
    }
}
