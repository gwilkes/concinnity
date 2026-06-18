// src/metal/screenshot.rs
//
// Headless frame capture for the Metal backend. The `cn debug` WS server's
// `screenshot` command routes here (via `RenderBackend::screenshot`) to copy
// the most recently presented drawable's colour texture into a host-readable
// texture and encode it to a PNG on disk. This is the on-GPU verification path
// the renderer otherwise leaves to a human eyeballing the live window: a
// headless smoke can now assert on actual pixels. Mirrors
// src/directx/screenshot.rs / src/vulkan/screenshot.rs.
//
// Metal has no persistent swapchain image array to read back the way D3D12 /
// Vulkan do; `CAMetalDrawable`s are transient. So `draw_frame` retains the last
// presented drawable's texture in `last_present_texture` (only under
// `hot_reload`, the path that also switches the MTKView's `framebufferOnly`
// off so the drawable can be a blit source). Capture is synchronous: a one-shot
// blit copies that texture into a `StorageModeShared` staging texture, waits,
// then `getBytes` + decode + PNG-encode on the CPU. The blit's own command
// buffer commits after every frame command buffer on the same queue, so
// same-queue FIFO order guarantees the drawable is fully rendered before the
// copy reads it. The decode follows the swapchain format (4-byte SDR `BGRA8` or
// 8-byte HDR `RGBA16Float`), not a fixed texel size.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLCommandQueue as _, MTLDevice as _, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize,
    MTLStorageMode, MTLTexture as _, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage,
};

use crate::gfx::hdr_output::HdrEncoding;

use super::context::MtlContext;

impl MtlContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous; `#[allow(dead_code)]` because it
    // is reached only through the `RenderBackend` vtable (bin-only `cn debug`).
    #[allow(dead_code)]
    pub(in crate::metal) fn capture_screenshot(&mut self, path: &str) -> Result<String, String> {
        // `None` both before the first present and in production (capture is a
        // `cn debug`-only feature; see `last_present_texture`). The retained
        // texture keeps the drawable's colour surface alive for the read-back.
        let src = self
            .last_present_texture
            .clone()
            .ok_or("screenshot: no frame has been presented yet (capture is cn debug only)")?;
        let width = src.width();
        let height = src.height();
        if width == 0 || height == 0 {
            return Err("screenshot: zero-sized drawable".into());
        }

        // Host-readable staging texture matching the drawable's format. The
        // drawable's own storage mode is driver-chosen (often not host-visible),
        // so blit into a `StorageModeShared` texture we control and `getBytes`
        // from that. `ShaderRead` is the default usage and is enough for a blit
        // destination.
        let desc = MTLTextureDescriptor::new();
        unsafe {
            desc.setTextureType(MTLTextureType::Type2D);
            desc.setPixelFormat(self.swap_pixel_format);
            desc.setWidth(width);
            desc.setHeight(height);
            desc.setUsage(MTLTextureUsage::ShaderRead);
            desc.setStorageMode(MTLStorageMode::Shared);
        }
        let staging = self
            .device
            .newTextureWithDescriptor(&desc)
            .ok_or("screenshot: failed to create staging texture")?;

        // One-shot blit: drawable colour -> staging. Committed after every
        // frame command buffer on the shared queue, so FIFO order has the
        // composite pass (which wrote the drawable) complete first; the
        // `waitUntilCompleted` then guarantees the copy is done before the read.
        let cmd_buf = self
            .command_queue
            .commandBuffer()
            .ok_or("screenshot: failed to get command buffer")?;
        let blit = cmd_buf
            .blitCommandEncoder()
            .ok_or("screenshot: failed to get blit encoder")?;
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &src,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width, height, depth: 1 },
                &staging,
                0,
                0,
                MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read the staging texture back tightly (no row padding) and decode.
        let bytes_per_pixel = swapchain_bytes_per_pixel(self.swap_pixel_format) as usize;
        let bytes_per_row = width * bytes_per_pixel;
        let mut raw = vec![0u8; bytes_per_row * height];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        // SAFETY: `raw` is `bytes_per_row * height` bytes long (exactly the tight
        // footprint requested), the staging texture is `StorageModeShared` and
        // the blit completed (`waitUntilCompleted` above), so the copy is valid.
        unsafe {
            staging.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                std::ptr::NonNull::new(raw.as_mut_ptr() as *mut std::ffi::c_void)
                    .ok_or("screenshot: null readback pointer")?,
                bytes_per_row,
                region,
                0,
            );
        }

        let rgba = decode_to_rgba8(&raw, self.swap_pixel_format, self.hdr_encoding);
        encode_png(path, width as u32, height as u32, &rgba)?;
        Ok(path.to_string())
    }
}

// Bytes per texel for the swapchain colour formats this backend can present.
// The MTKView only ever presents `BGRA8Unorm` for SDR or `RGBA16Float` for the
// HDR EDR path (see `metal/init/window.rs::swap_pixel_format`). Unknown formats
// default to 4, the common 32-bit-texel case.
fn swapchain_bytes_per_pixel(format: MTLPixelFormat) -> u32 {
    match format {
        MTLPixelFormat::RGBA16Float => 8,
        _ => 4,
    }
}

// Convert the tightly-packed read-back bytes to opaque RGBA8, decoding per the
// swapchain format. The alpha channel is forced to 255 so the saved PNG is
// opaque regardless of the composited alpha. `encoding` is the resolved HDR
// encoding (None on the SDR path) and only matters for the float swapchain.
fn decode_to_rgba8(raw: &[u8], format: MTLPixelFormat, encoding: Option<HdrEncoding>) -> Vec<u8> {
    match format {
        MTLPixelFormat::RGBA16Float => decode_rgba16f(raw, encoding),
        _ => decode_8bit(raw, format),
    }
}

// 8-bit-per-channel swapchain formats. The Metal SDR swapchain is `BGRA8Unorm`;
// `RGBA8Unorm` is handled too for completeness.
fn decode_8bit(raw: &[u8], format: MTLPixelFormat) -> Vec<u8> {
    let bgra = matches!(
        format,
        MTLPixelFormat::BGRA8Unorm | MTLPixelFormat::BGRA8Unorm_sRGB
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

// `RGBA16Float` HDR swapchain (8 B/px, four halfs RGBA). On the scRGB-linear
// path the stored values are linear extended-range (1.0 = SDR white), so apply
// the sRGB OETF to get a valid (non-tonemapped) image. On the PQ path the
// stored values are PQ code values already in [0, 1]; pass them through clamped.
// The PQ capture is not display-ready, but it must still be a valid PNG rather
// than a crash. Mirrors the DX/Vulkan paths.
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
// normals, and inf/NaN.
fn f16_to_f32(h: u16) -> f32 {
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
        assert_eq!(swapchain_bytes_per_pixel(MTLPixelFormat::BGRA8Unorm), 4);
        assert_eq!(swapchain_bytes_per_pixel(MTLPixelFormat::RGBA8Unorm), 4);
        assert_eq!(swapchain_bytes_per_pixel(MTLPixelFormat::RGBA16Float), 8);
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
        let out = decode_to_rgba8(&raw, MTLPixelFormat::BGRA8Unorm, None);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn bgra8_srgb_is_also_swizzled() {
        let raw = [10u8, 20, 30, 40];
        let out = decode_to_rgba8(&raw, MTLPixelFormat::BGRA8Unorm_sRGB, None);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn rgba8_passes_through_with_forced_alpha() {
        let raw = [30u8, 20, 10, 40];
        let out = decode_to_rgba8(&raw, MTLPixelFormat::RGBA8Unorm, None);
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
            MTLPixelFormat::RGBA16Float,
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
            MTLPixelFormat::RGBA16Float,
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
        let out = decode_to_rgba8(&raw, MTLPixelFormat::RGBA16Float, Some(HdrEncoding::Pq));
        assert_eq!(out[0], 255); // 1.0
        assert_eq!(out[1], 128); // 0.5 -> round(127.5)
        assert_eq!(out[2], 0); // 0.0
        assert_eq!(out[3], 255);
    }
}
