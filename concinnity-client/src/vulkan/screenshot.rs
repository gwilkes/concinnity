// src/vulkan/screenshot.rs
//
// Headless frame capture for the Vulkan backend. The `cn debug` WS server's
// `screenshot` command routes here (via `RenderBackend::screenshot`) to copy
// the most recently presented swapchain image into a host-visible buffer and
// encode it to a PNG on disk. This is the on-GPU verification path the renderer
// otherwise leaves to a human eyeballing the live window: a headless smoke can
// now assert on actual pixels.
//
// The swapchain images are created with `TRANSFER_SRC` usage (see
// `swapchain.rs`) so the presented image can be copied. Capture is synchronous:
// it idles the device, copies the last-presented image (still in
// `PRESENT_SRC_KHR`) into the buffer, restores the image to `PRESENT_SRC_KHR`,
// then maps + decodes + PNG-encodes on the CPU. The read-back buffer and the
// per-pixel decode both follow the swapchain format (4-byte SDR `BGRA8` or
// 8-byte HDR `RGBA16F`), not a fixed texel size. A swapchain rebuild clears
// `last_present_index`, so a capture in the brief window before the next present
// returns a clean error rather than reading an unrendered image.

use ash::vk;

use super::context::VkContext;
use super::texture::{create_buffer, one_shot_submit};
use crate::gfx::hdr_output::{HdrEncoding, HdrOutputMode};

impl VkContext {
    // Capture the last presented frame to a PNG at `path`. Returns the path on
    // success. Distinct name from the `RenderBackend::screenshot` trait method
    // so the backend forwarder is unambiguous; `#[allow(dead_code)]` because it
    // is reached only through the `RenderBackend` vtable (bin-only `cn debug`).
    #[allow(dead_code)]
    pub(in crate::vulkan) fn capture_screenshot(&mut self, path: &str) -> Result<String, String> {
        let Some(image_index) = self.last_present_index else {
            return Err("screenshot: no frame has been presented yet".into());
        };
        let src_image = *self
            .swapchain_images
            .get(image_index as usize)
            .ok_or("screenshot: stale swapchain image index")?;
        let width = self.swapchain_extent.width;
        let height = self.swapchain_extent.height;
        if width == 0 || height == 0 {
            return Err("screenshot: zero-sized swapchain".into());
        }

        // The GPU must be idle: the last-presented image is then stable and no
        // in-flight command buffer still references the resources we touch.
        unsafe { self.device.device_wait_idle() }
            .map_err(|e| format!("screenshot: wait idle: {e}"))?;

        // Host-visible readback buffer, tightly packed at the swapchain
        // format's texel size. The SDR swapchain is `BGRA8_UNORM` (4 B/px), but
        // the HDR swapchain is `R16G16B16A16_SFLOAT` (8 B/px); sizing this for a
        // fixed 4 B/px overflows the `vkCmdCopyImageToBuffer` on the HDR path and
        // loses the device, so derive it from the actual format.
        let bytes_per_pixel = swapchain_bytes_per_pixel(self.swapchain_format) as u64;
        let byte_size = (width as u64) * (height as u64) * bytes_per_pixel;
        let (buffer, memory) = create_buffer(
            &self.instance,
            &self.device,
            self.physical_device,
            byte_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // Copy the presented image into the buffer, bracketing with
        // PRESENT_SRC <-> TRANSFER_SRC barriers so the image is left exactly as
        // present expects it for the next acquire.
        let device = self.device.clone();
        let copied = one_shot_submit(
            &device,
            self.commands.command_pool,
            self.graphics_queue,
            |cmd| {
                let to_src = image_barrier(
                    src_image,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_READ,
                );
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                let to_present = image_barrier(
                    src_image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::empty(),
                );
                unsafe {
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_src],
                    );
                    device.cmd_copy_image_to_buffer(
                        cmd,
                        src_image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        buffer,
                        std::slice::from_ref(&region),
                    );
                    device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[to_present],
                    );
                }
            },
        );

        // Map + swizzle + encode, then always free the buffer.
        let result = copied.and_then(|()| {
            let ptr =
                unsafe { device.map_memory(memory, 0, byte_size, vk::MemoryMapFlags::empty()) }
                    .map_err(|e| format!("screenshot: map readback: {e}"))?
                    as *const u8;
            // SAFETY: the buffer is HOST_COHERENT and `byte_size` bytes long; the
            // copy above completed (one_shot_submit waits its fence).
            let raw = unsafe { std::slice::from_raw_parts(ptr, byte_size as usize) };
            // The HDR float swapchain needs the encoding to decode for display:
            // scRGB-linear gets the sRGB OETF, PQ-encoded code values pass
            // through (not display-correct, but a valid PNG rather than a crash).
            let encoding = match self.hdr_mode {
                HdrOutputMode::Hdr { encoding, .. } => Some(encoding),
                HdrOutputMode::Sdr => None,
            };
            let rgba = decode_to_rgba8(raw, self.swapchain_format, encoding);
            unsafe { device.unmap_memory(memory) };
            encode_png(path, width, height, &rgba)
        });
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        result.map(|()| path.to_string())
    }
}

// A whole-image colour barrier on a swapchain image, used to flip between
// PRESENT_SRC and TRANSFER_SRC for the readback copy.
fn image_barrier(
    image: vk::Image,
    old: vk::ImageLayout,
    new: vk::ImageLayout,
    src: vk::AccessFlags,
    dst: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_access_mask(src)
        .dst_access_mask(dst)
        .old_layout(old)
        .new_layout(new)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

// Bytes per texel for the swapchain colour formats this backend can present.
// The swapchain only ever resolves to one of these (see `create_swapchain_inner`
// in swapchain.rs): `BGRA8_UNORM` for SDR, `R16G16B16A16_SFLOAT` for the scRGB /
// PQ-float HDR path, or `A2B10G10R10_UNORM_PACK32` for the packed PQ fallback.
// Unknown formats default to 4, the common 32-bit-texel case.
fn swapchain_bytes_per_pixel(format: vk::Format) -> u32 {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => 4,
    }
}

// Convert the raw readback bytes to tightly-packed opaque RGBA8, decoding per
// the swapchain format. The alpha channel is forced to 255 so the saved PNG is
// opaque regardless of the composited alpha. `encoding` is the resolved HDR
// encoding (None on the SDR path) and only matters for the float swapchain.
fn decode_to_rgba8(raw: &[u8], format: vk::Format, encoding: Option<HdrEncoding>) -> Vec<u8> {
    match format {
        vk::Format::R16G16B16A16_SFLOAT => decode_rgba16f(raw, encoding),
        vk::Format::A2B10G10R10_UNORM_PACK32 => decode_a2b10g10r10(raw),
        _ => decode_8bit(raw, format),
    }
}

// 8-bit-per-channel swapchain formats. Almost always BGRA8 on Windows; RGBA8 is
// handled too.
fn decode_8bit(raw: &[u8], format: vk::Format) -> Vec<u8> {
    let bgra = matches!(
        format,
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_SNORM
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

// `R16G16B16A16_SFLOAT` HDR swapchain (8 B/px, four halfs RGBA). On the
// scRGB-linear path the stored values are linear extended-range (1.0 = SDR
// white), so apply the sRGB OETF to get a valid (non-tonemapped) image. On the
// PQ path the stored values are PQ code values already in [0, 1]; pass them
// through clamped.
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

// `A2B10G10R10_UNORM_PACK32` PQ fallback swapchain (4 B/px, one little-endian
// uint32 per texel: R in bits [9:0], G [19:10], B [29:20], A [31:30]). The
// values are PQ code values, so this is not display-ready, but it is a valid PNG.
fn decode_a2b10g10r10(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len());
    for px in raw.chunks_exact(4) {
        let v = u32::from_le_bytes([px[0], px[1], px[2], px[3]]);
        let r = v & 0x3ff;
        let g = (v >> 10) & 0x3ff;
        let b = (v >> 20) & 0x3ff;
        out.extend_from_slice(&[u10_to_u8(r), u10_to_u8(g), u10_to_u8(b), 255]);
    }
    out
}

// Decode an IEEE 754 half (binary16) to f32. Handles zero, subnormals,
// normals, and inf/NaN. Shared with the reflection-probe cube readback
// (`probe.rs`), which decodes its `R16G16B16A16_SFLOAT` faces the same way.
pub(in crate::vulkan) fn f16_to_f32(h: u16) -> f32 {
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

// Scale a 10-bit unsigned value (0..=1023) to 8-bit with rounding.
fn u10_to_u8(v: u32) -> u8 {
    ((v * 255 + 511) / 1023) as u8
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
        // SDR + the packed PQ fallback are 4 B/px; the float HDR swapchain is 8.
        assert_eq!(swapchain_bytes_per_pixel(vk::Format::B8G8R8A8_UNORM), 4);
        assert_eq!(swapchain_bytes_per_pixel(vk::Format::R8G8B8A8_UNORM), 4);
        assert_eq!(
            swapchain_bytes_per_pixel(vk::Format::A2B10G10R10_UNORM_PACK32),
            4
        );
        assert_eq!(
            swapchain_bytes_per_pixel(vk::Format::R16G16B16A16_SFLOAT),
            8
        );
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
        let out = decode_to_rgba8(&raw, vk::Format::B8G8R8A8_UNORM, None);
        assert_eq!(out, vec![30, 20, 10, 255]);
    }

    #[test]
    fn rgba8_passes_through_with_forced_alpha() {
        let raw = [30u8, 20, 10, 40];
        let out = decode_to_rgba8(&raw, vk::Format::R8G8B8A8_UNORM, None);
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
            vk::Format::R16G16B16A16_SFLOAT,
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
            vk::Format::R16G16B16A16_SFLOAT,
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
        let out = decode_to_rgba8(&raw, vk::Format::R16G16B16A16_SFLOAT, Some(HdrEncoding::Pq));
        assert_eq!(out[0], 255); // 1.0
        assert_eq!(out[1], 128); // 0.5 -> round(127.5)
        assert_eq!(out[2], 0); // 0.0
        assert_eq!(out[3], 255);
    }

    #[test]
    fn a2b10g10r10_unpacks_channels() {
        // R=1023, G=0, B=1023, A=3 packed little-endian.
        let v: u32 = 1023 | (1023 << 20) | (3 << 30);
        let out = decode_to_rgba8(&v.to_le_bytes(), vk::Format::A2B10G10R10_UNORM_PACK32, None);
        assert_eq!(out, vec![255, 0, 255, 255]);
    }
}
