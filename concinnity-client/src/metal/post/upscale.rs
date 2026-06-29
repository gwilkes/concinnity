// src/metal/post/upscale.rs
//
// MetalFX temporal upscaling: the engine renders the 3D scene at a fraction
// of drawable size and this pass reconstructs a drawable-resolution image
// the bloom + composite stack consumes. Lives in `metal/post/` next to TAA
// + bloom + SSAO + SSR; the scaler descriptor, the output texture, and the
// per-frame encoder all live together so the effect is a single unit.
//
// The scaler does temporal accumulation itself, so the existing TAA pass is
// bypassed while upscaling is on (`PostProcessConfig.aa_mode` is ignored). The
// existing velocity pre-pass still runs: the scaler consumes its motion
// vectors. The existing projection jitter still runs: the scaler consumes
// its sub-pixel offset.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice as _, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
    MTLTextureType, MTLTextureUsage,
};
use objc2_metal_fx::{MTLFXTemporalScaler, MTLFXTemporalScalerBase, MTLFXTemporalScalerDescriptor};

use crate::metal::context::MtlContext;

// All MetalFX-temporal-upscaling state grouped into one feature unit: the
// scaler instance, the input/output scale ratio, the per-frame projection
// jitter, and the history-reset flag. `scaler` is `Some` only when the world
// requested temporal upscaling AND the GPU supports it; otherwise `scale` is
// `1.0` (render-res == output-res) and the rest are inert.
pub(crate) struct UpscaleState {
    // The MetalFX scaler. `Some` only when upscaling is active.
    pub scaler: Option<MetalFXUpscaler>,
    // Per-axis input-to-output ratio. `1.0` when the scaler is absent. Kept on
    // the context so a swapchain resize knows what fraction to render at.
    pub scale: f32,
    // Pixel-space jitter offset the projection applied this frame. Written on
    // the main thread before fan-out, read by `encode_upscale` on a worker;
    // packed atomically so the worker snapshots it without a mutex.
    pub jitter: UpscaleJitter,
    // Whether the scaler should discard its temporal history on the next
    // encode. Raised after a scaler rebuild (resize / startup); cleared by
    // `encode_upscale` after honouring it.
    pub reset_pending: std::sync::atomic::AtomicBool,
}

// MetalFX temporal upscaler. Owns the descriptor-bound scaler instance plus
// the output texture the bloom + composite stack consumes (it must be at
// output resolution, GPU-private, with `ShaderRead | RenderTarget` usage
// per Apple's compatibility requirements).
//
// The scaler is sized once at construction. A window resize or a runtime
// scale change rebuilds the whole struct via [`MetalFXUpscaler::new`]; the
// old one drops and Metal retires its GPU storage on the next idle.
pub(crate) struct MetalFXUpscaler {
    // The MetalFX scaler instance. Its `colorTexture` / `depthTexture` /
    // `motionTexture` / `outputTexture` / `jitterOffset` / `reset` properties
    // are set per frame by `MtlContext::encode_upscale` before
    // `encodeToCommandBuffer:`.
    pub(crate) scaler: Retained<ProtocolObject<dyn MTLFXTemporalScaler>>,
    // The output texture the scaler writes into and the post stack reads
    // from. Always sized at `(output_width, output_height)`. Bloom +
    // Composite see this as `scene_color` when upscaling is on.
    pub(crate) output: Retained<ProtocolObject<dyn MTLTexture>>,
    // Render-resolution width (= input width fed to the scaler).
    pub(crate) input_width: u32,
    // Render-resolution height.
    pub(crate) input_height: u32,
    // Drawable-resolution width (= scaler output width).
    pub(crate) output_width: u32,
    // Drawable-resolution height.
    pub(crate) output_height: u32,
}

// Does the active device support MetalFX temporal scaling at all? Used at
// init to log a clear "MetalFX not supported on this GPU; rendering at
// native resolution" warning and silently fall back rather than fail.
pub(crate) fn temporal_scaler_supported(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> bool {
    unsafe { MTLFXTemporalScalerDescriptor::supportsDevice(device) }
}

impl MetalFXUpscaler {
    // Build a fresh upscaler at the given output size and per-axis scale.
    // `scale` is clamped to the device's supported range; the resolved
    // input dimensions are stored on the struct so the caller knows what
    // resolution to actually render at.
    pub(crate) fn new(
        device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
        output_width: u32,
        output_height: u32,
        scale: f32,
    ) -> Result<Self, String> {
        // Apple's `inputContentMin/MaxScale` reports the OUTPUT-over-INPUT
        // ratio: a 1.0 minimum means "the smallest upscale is no upscale";
        // a 3.0 maximum means "the largest upscale is 3× per axis". Our
        // user-facing `scale` is the inverse (input/output, ≤ 1.0), so flip
        // it to the device's units, clamp, and flip back.
        let min_ratio = unsafe {
            MTLFXTemporalScalerDescriptor::supportedInputContentMinScaleForDevice(device)
        };
        let max_ratio = unsafe {
            MTLFXTemporalScalerDescriptor::supportedInputContentMaxScaleForDevice(device)
        };
        let requested_ratio = if scale > 0.0 { 1.0 / scale } else { 1.0 };
        let clamped_ratio = requested_ratio.clamp(min_ratio.max(1.0), max_ratio.max(1.0));
        let scale = 1.0 / clamped_ratio;
        let input_width = ((output_width as f32) * scale).max(1.0) as u32;
        let input_height = ((output_height as f32) * scale).max(1.0) as u32;

        let descriptor = unsafe { MTLFXTemporalScalerDescriptor::new() };
        unsafe {
            // Color: matches the engine's pre-TAA scene format (RGBA16Float
            // single-sample, written by the SSR resolve or the HDR resolve
            // depending on configuration).
            descriptor.setColorTextureFormat(MTLPixelFormat::RGBA16Float);
            // Depth: the velocity pre-pass writes a single-sample Depth32Float
            // depth buffer at render-res that doubles as the scaler depth
            // input. (The main pass's MSAA depth is not directly usable.)
            descriptor.setDepthTextureFormat(MTLPixelFormat::Depth32Float);
            // Motion: the velocity pre-pass writes RG16Float UV-space motion
            // vectors at render-res. The scaler interprets them in input
            // pixel coords once `motionVectorScale` rescales the UV delta.
            descriptor.setMotionTextureFormat(MTLPixelFormat::RG16Float);
            // Output: drawable-res RGBA16Float so the existing post stack
            // (bloom, composite) can read it without an additional format
            // conversion. Composite handles ACES + LUT + FXAA when SDR;
            // skips them on the HDR path.
            descriptor.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
            descriptor.setInputWidth(input_width as usize);
            descriptor.setInputHeight(input_height as usize);
            descriptor.setOutputWidth(output_width as usize);
            descriptor.setOutputHeight(output_height as usize);
            // We do not author a pre-exposed colour buffer (the input is
            // linear HDR fresh from the SSR / HDR resolve), so leave
            // auto-exposure off and let MetalFX use its built-in heuristic.
            descriptor.setAutoExposureEnabled(false);
        }

        let scaler = unsafe { descriptor.newTemporalScalerWithDevice(device) }
            .ok_or_else(|| "MetalFX: failed to create temporal scaler".to_string())?;

        // The scaler enforces a minimum texture-usage set on the output
        // texture; query it and union with the bloom + composite read
        // requirement so the post stack can sample the result.
        let required_output_usage = unsafe { scaler.outputTextureUsage() };
        let output_desc = MTLTextureDescriptor::new();
        unsafe {
            output_desc.setTextureType(MTLTextureType::Type2D);
            output_desc.setPixelFormat(MTLPixelFormat::RGBA16Float);
            output_desc.setWidth(output_width.max(1) as usize);
            output_desc.setHeight(output_height.max(1) as usize);
            output_desc.setStorageMode(MTLStorageMode::Private);
            output_desc.setUsage(MTLTextureUsage(
                required_output_usage.0
                    | MTLTextureUsage::ShaderRead.0
                    | MTLTextureUsage::RenderTarget.0,
            ));
        }
        let output = device
            .newTextureWithDescriptor(&output_desc)
            .ok_or("MetalFX: failed to create upscaler output texture")?;

        Ok(MetalFXUpscaler {
            scaler,
            output,
            input_width,
            input_height,
            output_width,
            output_height,
        })
    }
}

impl MtlContext {
    // Encode the MetalFX temporal upscale: feed it the pre-TAA scene
    // (post-SSR, post-Fog, post-particles), the velocity pre-pass's
    // motion vectors + depth, and the sub-pixel jitter offset applied to
    // this frame's projection. Outputs into `upscaler.output`, which the
    // bloom + composite passes then read as `scene_color`.
    //
    // `scene_pre_taa` is whatever pre-TAA texture the graph routed in for
    // this frame (SSR resolve output when SSR is on; otherwise the main
    // pass's `hdr_resolve`). The scaler needs its `depthTexture` to be
    // the depth that produced `scene_pre_taa`: we use the single-sample
    // depth the velocity pre-pass already writes at render resolution,
    // which is rasterised from the same geometry the main pass shaded.
    //
    // Reset is requested whenever the scaler was just rebuilt (resize or
    // first frame); after the encode the flag is cleared via the atomic
    // stash that `draw_frame` mutates on the main thread between frames.
    pub(in crate::metal) fn encode_upscale(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        scene_pre_taa: &Retained<ProtocolObject<dyn MTLTexture>>,
    ) -> Result<u32, String> {
        let upscaler = self
            .upscale
            .scaler
            .as_ref()
            .ok_or("Upscale enabled but upscaler missing")?;
        let gbuf = self
            .gbuffer
            .targets
            .as_ref()
            .ok_or("Upscale enabled but G-buffer targets missing")?;

        unsafe {
            upscaler
                .scaler
                .setColorTexture(Some(scene_pre_taa.as_ref()));
            upscaler.scaler.setDepthTexture(Some(gbuf.depth.as_ref()));
            upscaler
                .scaler
                .setMotionTexture(Some(gbuf.velocity.as_ref()));
            upscaler
                .scaler
                .setOutputTexture(Some(upscaler.output.as_ref()));

            // Motion vectors are stored as `prev_uv - cur_uv` in UV space
            // (RG16Float). The scaler expects motion in input-pixel coords,
            // so the per-axis scale is the input texture extent.
            upscaler
                .scaler
                .setMotionVectorScaleX(upscaler.input_width as f32);
            upscaler
                .scaler
                .setMotionVectorScaleY(upscaler.input_height as f32);

            // The jitter the projection matrix applied this frame in input
            // pixel space (each Halton sample is in [-0.5, 0.5] pixels).
            // `draw_frame` stashes it on the context before fan-out.
            let [jx, jy] = self
                .upscale
                .jitter
                .load(std::sync::atomic::Ordering::Relaxed);
            upscaler.scaler.setJitterOffsetX(jx);
            upscaler.scaler.setJitterOffsetY(jy);

            // Reverse-Z is not in use (the depth state writes 0=near,
            // 1=far), so leave depth-reversed off.
            upscaler.scaler.setDepthReversed(false);

            // First-frame-after-rebuild discards history. The flag is
            // owned by an atomic on the context; `draw_frame` raises it
            // on the main thread between frames whenever the scaler was
            // rebuilt.
            let reset = self
                .upscale
                .reset_pending
                .swap(false, std::sync::atomic::Ordering::AcqRel);
            upscaler.scaler.setReset(reset);
        }

        // MetalFX's encode is a single dispatch that doesn't go through a render
        // or compute encoder we own, so the standard `attach_render` /
        // `attach_compute` plumbing doesn't apply. The pass-timing slot for
        // Upscale stays at zero until Apple exposes a sampling hook; the chip
        // simply omits the row.

        unsafe {
            upscaler.scaler.encodeToCommandBuffer(cmd_buf);
        }
        Ok(0)
    }
}

// Per-axis upscale-jitter holder. Two `f32` slots packed atomically so the
// main thread (which computes Halton samples) and the worker thread (which
// reads them in `encode_upscale`) can synchronise without a mutex. Bit-cast
// through `u64` since `AtomicU64` is widely available on the targets
// MetalFX runs on (macOS 13+, aarch64 / x86_64).
#[derive(Default)]
pub(crate) struct UpscaleJitter(std::sync::atomic::AtomicU64);

impl UpscaleJitter {
    pub(crate) fn store(&self, jx: f32, jy: f32, ordering: std::sync::atomic::Ordering) {
        let packed = (jx.to_bits() as u64) | ((jy.to_bits() as u64) << 32);
        self.0.store(packed, ordering);
    }

    pub(crate) fn load(&self, ordering: std::sync::atomic::Ordering) -> [f32; 2] {
        let packed = self.0.load(ordering);
        let jx = f32::from_bits(packed as u32);
        let jy = f32::from_bits((packed >> 32) as u32);
        [jx, jy]
    }
}
