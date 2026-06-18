// src/metal/auto_exposure.rs
//
// Auto-exposure (EV adaptation) on Metal: a per-frame CPU readback of the
// previous frame's average log-luminance, an EMA step that updates the adapted
// EV, and the histogram build + average compute dispatches that produce next
// frame's average. The compute passes are encoded after the main HDR resolve
// (where `hdr_resolve` carries this frame's scene colour) and read CPU-side at
// the top of the next frame, so there is one frame of latency between the
// scene's actual luminance and the exposure applied to it: invisible at
// human-scale eye-adaptation rates.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer as _, MTLCommandBuffer as _, MTLComputeCommandEncoder as _, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLDevice as _, MTLLibrary as _, MTLSize, MTLTexture as _,
};

use super::context::*;
use super::pipeline::{ns_str, shader_source};
use super::scoped_encoder::ScopedEncoder;
use super::uniforms::*;
use crate::gfx::auto_exposure::{AutoExposureSettings, AutoExposureState};

// All auto-exposure (EV adaptation) state grouped into one feature unit: the
// resolved tunables, the EMA-tracked adapted EV, the authored bias, the
// histogram-build + average compute pipelines, their shared/readback buffers,
// and the per-frame timing bookkeeping. All `Some` only when the world's
// `PostProcessConfig` turns auto-exposure on; otherwise the static-exposure
// path drives `post_process.exposure` directly and these stay `None`.
pub(crate) struct AutoExposureGpu {
    pub settings: Option<AutoExposureSettings>,
    // EMA-tracked adapted EV, updated each frame from the previous frame's
    // GPU-measured average log-luminance.
    pub state: Option<AutoExposureState>,
    // Authored `exposure_ev` carried through as an additive bias (in stops)
    // on the adapted EV. 0.0 when auto-exposure is off.
    pub bias_ev: f32,
    pub pipelines: Option<AutoExposurePipelines>,
    // 256-bin global histogram the build kernel accumulates into (shared
    // storage so the average kernel can read + clear it in one pass).
    pub histogram: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // One-float readback buffer the average kernel writes the count-weighted
    // average log-luminance into; CPU reads it at the top of the next frame.
    pub output: Option<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>,
    // Last frame's `elapsed`, used to derive a frame `dt` for the EMA step.
    pub last_elapsed: f32,
}

impl MtlContext {
    // Build the per-frame compute params for the auto-exposure kernels. The
    // log-luminance range and the precomputed `bins / range` scale match the
    // `gfx::auto_exposure::LUM_LOG2_*` constants exactly.
    fn auto_exposure_params(&self) -> AutoExposureParams {
        use crate::gfx::auto_exposure::{HISTOGRAM_BINS, LUM_LOG2_MAX, LUM_LOG2_MIN};
        let range = LUM_LOG2_MAX - LUM_LOG2_MIN;
        AutoExposureParams {
            lum_log2_min: LUM_LOG2_MIN,
            lum_log2_range: range,
            lum_to_bin_scale: HISTOGRAM_BINS as f32 / range,
            _pad: 0.0,
        }
    }

    // Step the auto-exposure EMA from the previous frame's GPU measurement,
    // then push the new exposure multiplier into `self.post_process.exposure`.
    // A no-op when auto-exposure is disabled: the static authored EV then
    // drives `exposure` unchanged.
    //
    // `elapsed` is the total elapsed seconds since startup, the same value
    // `draw_frame` already receives. We diff against the previous call's
    // elapsed to derive a frame `dt`; on the first frame `dt` is 0 so the
    // EMA snaps to the initial state (midpoint of the clamp range).
    pub(super) fn update_auto_exposure(&mut self, elapsed: f32) {
        let Some(settings) = self.auto_exposure.settings.as_ref().copied() else {
            return;
        };
        let Some(state) = self.auto_exposure.state.as_mut() else {
            return;
        };
        let Some(output_buf) = self.auto_exposure.output.as_ref() else {
            return;
        };

        // Read the previous frame's average log-luminance from the shared
        // output buffer. Shared storage on Apple silicon means the CPU sees
        // the GPU's most recent write without an explicit sync; in the worst
        // case it sees a stale value, which the EMA smooths over.
        let avg_log_lum = unsafe {
            let ptr = output_buf.contents().as_ptr() as *const f32;
            ptr.read()
        };
        let avg_log_lum = if avg_log_lum.is_finite() {
            avg_log_lum
        } else {
            crate::gfx::auto_exposure::LUM_LOG2_MIN
        };

        let dt = (elapsed - self.auto_exposure.last_elapsed).max(0.0);
        self.auto_exposure.last_elapsed = elapsed;

        let adapted_ev = state.update(avg_log_lum, self.auto_exposure.bias_ev, &settings, dt);
        // `self.post_process.exposure` is the linear multiplier the post pass
        // and bloom prefilter consume; it already folds in the authored
        // exposure_ev when auto-exposure is off, so we only overwrite it here
        // when the GPU path owns the value. `state.update` already folds the
        // bias into `adapted_ev`'s target; re-adding it would double the bias.
        self.post_process.exposure = adapted_ev.exp2();
    }

    // Encode the auto-exposure histogram passes against `hdr_resolve`. The
    // build kernel runs one thread per HDR pixel; the average kernel runs
    // one threadgroup of 256 threads that reduces the histogram, clears it
    // for the next frame, and writes the average log-luminance to the
    // shared output buffer the CPU will read at the top of the next frame.
    // A no-op when auto-exposure is disabled.
    // pub(in crate::metal) so the render-graph executor in
    // metal/graph_exec.rs can dispatch this pass from a CompiledGraph.
    pub(in crate::metal) fn encode_auto_exposure(
        &self,
        cmd_buf: &ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    ) -> Result<u32, String> {
        let (Some(pipelines), Some(histogram), Some(output)) = (
            self.auto_exposure.pipelines.as_ref(),
            self.auto_exposure.histogram.as_ref(),
            self.auto_exposure.output.as_ref(),
        ) else {
            return Ok(0);
        };

        let params = self.auto_exposure_params();
        let hdr_tex: &ProtocolObject<dyn objc2_metal::MTLTexture> =
            self.hdr_targets.hdr_resolve.as_ref();
        let tex_w = hdr_tex.width();
        let tex_h = hdr_tex.height();
        if tex_w == 0 || tex_h == 0 {
            return Ok(0);
        }

        let ae_desc = MTLComputePassDescriptor::new();
        if let Some(t) = &self.pass_timing {
            t.attach_compute(&ae_desc, super::pass_timing::PassId::AutoExposure);
        }
        let enc = ScopedEncoder::new(
            cmd_buf
                .computeCommandEncoderWithDescriptor(&ae_desc)
                .ok_or("failed to get auto-exposure compute encoder")?,
            "auto-exposure",
        );

        // Build kernel: 16x16 threadgroups, one thread per HDR pixel.
        enc.setComputePipelineState(&pipelines.build);
        unsafe {
            enc.setTexture_atIndex(Some(hdr_tex), 0);
            enc.setBuffer_offset_atIndex(Some(histogram), 0, 0);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&params).cast(),
                std::mem::size_of::<AutoExposureParams>(),
                1,
            );
        }
        let tg = MTLSize {
            width: 16,
            height: 16,
            depth: 1,
        };
        let grid = MTLSize {
            width: tex_w,
            height: tex_h,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);

        // Average kernel: one threadgroup of 256 threads reduces the
        // histogram and clears it for the next frame. The output buffer at
        // buffer(1) receives the count-weighted average log-luminance.
        enc.setComputePipelineState(&pipelines.average);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(histogram), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::from(&params).cast(),
                std::mem::size_of::<AutoExposureParams>(),
                2,
            );
        }
        let avg_grid = MTLSize {
            width: crate::gfx::auto_exposure::HISTOGRAM_BINS,
            height: 1,
            depth: 1,
        };
        let avg_tg = MTLSize {
            width: crate::gfx::auto_exposure::HISTOGRAM_BINS,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(avg_grid, avg_tg);

        Ok(0)
    }
}

// Pair of compute pipelines driving the auto-exposure histogram path: the
// build kernel (one thread per HDR-resolve pixel) that accumulates a 256-bin
// log-luminance histogram, and the average kernel (one threadgroup of 256
// threads) that reduces the histogram to a single average log-luminance value
// and clears it for the next frame.
pub(super) struct AutoExposurePipelines {
    pub build: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub average: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// Build the auto-exposure compute pipelines. Compiles `auto_exposure.metal`
// once and pulls both kernel entry points from it. Returned only when the
// world's `PostProcessConfig` opts into auto-exposure; otherwise the histogram
// pass is skipped entirely.
pub(super) fn build_auto_exposure_pipelines(
    device: &ProtocolObject<dyn objc2_metal::MTLDevice>,
    hot_reload: bool,
) -> Result<AutoExposurePipelines, String> {
    let msl = shader_source(hot_reload, "auto_exposure.metal");
    let options = objc2_metal::MTLCompileOptions::new();
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(msl.as_ref()), Some(&options))
        .map_err(|e| format!("auto-exposure shader compile error: {:?}", e))?;
    let build_fn = library
        .newFunctionWithName(&ns_str("histogram_build"))
        .ok_or("histogram_build not found in auto_exposure library")?;
    let average_fn = library
        .newFunctionWithName(&ns_str("histogram_average"))
        .ok_or("histogram_average not found in auto_exposure library")?;
    let build = device
        .newComputePipelineStateWithFunction_error(&build_fn)
        .map_err(|e| format!("failed to create histogram_build pipeline: {:?}", e))?;
    let average = device
        .newComputePipelineStateWithFunction_error(&average_fn)
        .map_err(|e| format!("failed to create histogram_average pipeline: {:?}", e))?;
    Ok(AutoExposurePipelines { build, average })
}
