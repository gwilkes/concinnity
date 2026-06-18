// src/gfx/profile.rs
//
// Per-frame profiling data. Backend-agnostic: `World::step` records each
// system's CPU step time here, the active render backend writes its draw /
// GPU stats here, and `StatHud` reads it back to drive the on-screen HUD.
// The debug server's `profile` command also reports it for headless
// verification.

// Maximum number of per-pass GPU timings tracked by [`RenderStats`]. Must be
// at least the client render graph's `PassId` count (`PASS_COUNT`), which the
// per-pass timing loop iterates; sized with headroom so unused slots carry the
// `""` sentinel name and a zero microsecond reading.
pub const MAX_PASS_TIMINGS: usize = 32;

// One per-pass GPU timing measurement: a stable pass name and the GPU
// microseconds spent in that pass during the most recently completed frame.
// Empty-string entries are unused slots, not real passes.
pub type PassTiming = (&'static str, u32);

// Per-frame render-backend statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderStats {
    // CPU-issued geometry draw calls this frame: the shadow, main, and
    // composite + text passes. The optional screen-space-effect passes (SSR,
    // SSAO, TAA, bloom) issue a fixed handful of fullscreen draws and are not
    // counted here -- `gpu_frame_us` still covers their GPU time.
    pub draw_calls: u32,
    // Renderable objects in the scene this frame: static draw objects, every
    // instanced-cluster instance, and skinned meshes.
    pub objects: u32,
    // GPU execution time of the most recently completed frame, in
    // microseconds. Reported one or more frames late: the GPU timestamps are
    // only known once that frame's command buffer completion handler fires.
    pub gpu_frame_us: u32,
    // Bytes of GPU memory currently allocated by the render device. On
    // unified-memory hardware (Apple Silicon) this is the device's share of
    // system memory rather than dedicated VRAM.
    pub vram_bytes: u64,
    // Per-pass GPU microseconds for the most recently completed frame.
    // Filled by the active backend only when its GPU supports timestamp
    // sampling; otherwise every slot stays at the default
    // `("", 0)`. Slot order is backend-defined and stable for the process
    // lifetime.
    pub pass_times_us: [PassTiming; MAX_PASS_TIMINGS],
    // Current adapted exposure value (EV) from the auto-exposure EMA.
    // `None` when the world did not opt in to auto-exposure (or the active
    // backend has not yet wired the readout). The `StatHud` overlay reads
    // this to render the on-screen EV chip; downstream consumers can map
    // it to an exposure multiplier via `2^ev`.
    pub auto_exposure_ev: Option<f32>,
    // Active display's reported maximum extended-range colour-component
    // multiplier when the renderer is on the HDR path. `Some(2.0)` on a
    // typical HDR400 panel, `Some(8.0+)` on HDR1000-class panels; `None`
    // on SDR (because the world disabled HDR, the platform fell back, or
    // the active backend has not wired the readout). The `StatHud` overlay
    // reads this to render the on-screen `EDR` chip; the value is also
    // the linear scaling factor between SDR reference white and the
    // panel's peak brightness, so a colour-grading consumer can interpret
    // it directly.
    pub max_edr: Option<f32>,
}

impl Default for RenderStats {
    fn default() -> Self {
        Self {
            draw_calls: 0,
            objects: 0,
            gpu_frame_us: 0,
            vram_bytes: 0,
            pass_times_us: [("", 0); MAX_PASS_TIMINGS],
            auto_exposure_ev: None,
            max_edr: None,
        }
    }
}

// Timing collected for one frame and read back by the profiler overlay.
//
// The system timings are double-buffered: a frame accumulates into
// `current`, and `begin_frame` rotates the just-finished frame into `last`
// so a reader always sees a complete frame rather than a partial one.
#[derive(Debug, Default)]
pub struct FrameProfile {
    // System CPU step times from the last fully completed frame, in step
    // order: `(system name, microseconds)`.
    last: Vec<(&'static str, u32)>,
    // Accumulator for the frame currently in progress.
    current: Vec<(&'static str, u32)>,
    // Render-backend stats for the most recent drawn frame. Left at the
    // default when no graphics backend is running.
    pub render: RenderStats,
}

impl FrameProfile {
    // Rotate the system-timing buffers at the start of a frame: the frame
    // that just finished becomes the readable snapshot and the accumulator
    // is cleared for the new frame.
    pub fn begin_frame(&mut self) {
        std::mem::swap(&mut self.last, &mut self.current);
        self.current.clear();
    }

    // Record one system's CPU step time for the in-progress frame.
    pub fn record_system(&mut self, name: &'static str, micros: u32) {
        self.current.push((name, micros));
    }

    // System step times from the last fully completed frame, in step order.
    // Read by the runtime debug server's `profile` command (a binary-only
    // module), so the lib build sees no caller.
    #[allow(dead_code)]
    pub fn system_timings(&self) -> &[(&'static str, u32)] {
        &self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timings_empty_until_first_rotation() {
        let mut p = FrameProfile::default();
        p.record_system("A", 100);
        // Nothing is readable until begin_frame rotates the accumulator.
        assert!(p.system_timings().is_empty());
        p.begin_frame();
        assert_eq!(p.system_timings(), &[("A", 100)]);
    }

    #[test]
    fn begin_frame_rotates_and_clears() {
        let mut p = FrameProfile::default();
        p.record_system("A", 10);
        p.record_system("B", 20);
        p.begin_frame();
        assert_eq!(p.system_timings(), &[("A", 10), ("B", 20)]);
        // The next frame records fresh values; the snapshot only updates on
        // the following begin_frame.
        p.record_system("A", 99);
        assert_eq!(p.system_timings(), &[("A", 10), ("B", 20)]);
        p.begin_frame();
        assert_eq!(p.system_timings(), &[("A", 99)]);
    }

    #[test]
    fn render_stats_default_is_zero() {
        let p = FrameProfile::default();
        assert_eq!(p.render, RenderStats::default());
        assert_eq!(p.render.draw_calls, 0);
    }
}
