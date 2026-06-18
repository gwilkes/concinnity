// src/hud/stat_hud.rs
//
// Performance-HUD overlay behavior. An internal system (not a declarable
// asset): `World::start` constructs one from the world's `StatHud` component
// and it writes live engine stats into that component's `TextLabel` chips.

use crate::assets::{Camera3D, FrameInput, StatHud, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::profile::PassTiming;
use std::time::Instant;

// How often the chip text is rebuilt, in seconds. The frame rate is averaged
// over this window so the number is readable rather than flickering.
const EMIT_INTERVAL_SECS: f32 = 0.5;

// How many per-pass entries the passes chip lists. Picked to fit comfortably
// under the FPS / VRAM / EV / EDR strip on a typical 720p+ HUD; passes past
// this count are dropped from the chip (still visible via the debug WS
// `profile.passes` reply).
const PASSES_CHIP_TOP_N: usize = 6;

// Build the frame-rate chip text from a frame count over an elapsed window.
fn fps_text(frames: u32, elapsed_secs: f32) -> String {
    let fps = if elapsed_secs > 0.0 {
        frames as f32 / elapsed_secs
    } else {
        0.0
    };
    format!("FPS {fps:.0}")
}

// Build the GPU-memory chip text from a byte count.
fn vram_text(bytes: u64) -> String {
    format!("VRAM {} MB", bytes / (1024 * 1024))
}

// Build the exposure-value chip text from the current auto-exposure EMA
// reading, or `None` when auto-exposure is not active for this world.
fn ev_text(ev: Option<f32>) -> String {
    match ev {
        Some(v) if v.is_finite() => format!("EV {v:+.2}"),
        _ => String::new(),
    }
}

// Build the EDR-headroom chip text from the active panel's maximum
// extended-range multiplier. `None` on SDR (the chip then stays empty so
// the HUD strip has no orphan reading on a non-HDR display).
fn edr_text(max_edr: Option<f32>) -> String {
    match max_edr {
        Some(v) if v.is_finite() && v > 0.0 => format!("EDR x{v:.1}"),
        _ => String::new(),
    }
}

// Build the cursor-position chip text from the latest window-space cursor
// coordinates (origin top-left). Rounded to whole pixels: sub-pixel jitter is
// not useful on a debug readout. The reading is only meaningful when the
// cursor is not captured (free-look worlds leave it stale); the chip still
// renders so a screenshot carries a reference point.
fn mouse_text(x: f32, y: f32) -> String {
    format!("MOUSE {x:.0}, {y:.0}")
}

// Build the camera-pose chip text from the live `Camera3D` pose, or blank when
// the world has no camera. The values are exactly what the debug `camera-set`
// command consumes, so a screenshot of this chip carries the arguments to
// reproduce the shot.
fn camera_text(pose: Option<([f32; 3], f32, f32)>) -> String {
    match pose {
        Some((p, yaw, pitch)) => {
            format!(
                "CAM {:.2} {:.2} {:.2}\nyaw {yaw:.3} pitch {pitch:.3}",
                p[0], p[1], p[2]
            )
        }
        None => String::new(),
    }
}

// Build the per-pass timing chip text from the active backend's
// `RenderStats.pass_times_us` array. Picks the top
// [`PASSES_CHIP_TOP_N`] entries by GPU microseconds, descending, one per
// line as `name µs`. Returns an empty string when every slot is at the
// default `("", 0)` (the chip then renders nothing, DX/Vulkan keep it
// blank until their per-pass timing pools land).
//
// **Apple-GPU caveat:** the GPU overlaps fragment work across encoders,
// so summing these values exceeds `gpu_frame_us`. Display them as
// per-pass attributions, not as components of a total. The chip's
// "PASSES" header is meant to make that obvious at a glance: there is
// no row labelled "total".
fn passes_text(slots: &[PassTiming]) -> String {
    let mut entries: Vec<(&'static str, u32)> = slots
        .iter()
        .copied()
        .filter(|(name, micros)| !name.is_empty() && *micros > 0)
        .collect();
    if entries.is_empty() {
        return String::new();
    }
    // Sort descending by µs; stable so equal-time passes keep their
    // PassId-order (alphabetical-ish across a typical frame).
    entries.sort_by_key(|e| std::cmp::Reverse(e.1));
    entries.truncate(PASSES_CHIP_TOP_N);
    let mut out = String::from("PASSES");
    for (name, micros) in entries {
        out.push('\n');
        out.push_str(name);
        out.push(' ');
        // Format compactly. < 1 ms reads as raw µs (e.g. "120 us"); ≥ 1 ms
        // reads as a single-decimal millisecond figure ("1.2 ms"). Keeps
        // the column width steady across the typical 50 µs to 5 ms band.
        if micros < 1000 {
            out.push_str(&format!("{micros} us"));
        } else {
            out.push_str(&format!("{:.1} ms", micros as f32 / 1000.0));
        }
    }
    out
}

// Draws a compact performance HUD: an `FPS` chip, a `VRAM` chip, an `EV`
// chip (when auto-exposure is on), and an `EDR` chip (when the renderer is
// on the HDR display path), each written into its own `TextLabel`. Give
// the labels a `background` colour for the boxed look. The HUD is toggled
// on and off at runtime with **F1**.
//
// The detailed per-system timing breakdown is not drawn on screen: it is
// reported by the runtime debug server's `profile` command instead.
//
// `VRAM` is the render device's current allocation; on Apple Silicon's
// unified memory that is its share of system RAM. Filled on Metal
// (`MTLDevice.currentAllocatedSize`) and DirectX
// (`IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL).CurrentUsage`); Vulkan
// still reads `0 MB`.
//
// `EV` is the adapted exposure value the auto-exposure EMA settled on for
// the most recent frame (the multiplier the post-process stack uses is
// `2^ev`). Filled on every backend (Metal, DirectX, Vulkan) when the world's
// `PostProcessConfig` enables auto-exposure; the chip stays blank otherwise so
// a world without auto-exposure has no orphan reading.
//
// `EDR` is the active panel's maximum extended-range colour-component
// multiplier (e.g. `EDR x2.0` on an HDR400 panel, `x8.0` on HDR1000).
// Filled on Metal when `PostProcessConfig.hdr_display = true` AND the
// platform reports an EDR headroom above SDR reference white; the chip
// stays blank on SDR or when the HDR request fell back.
//
// `PASSES` is a multi-line chip listing the top six render-graph passes
// of the last completed frame in descending GPU-microsecond order
// (e.g. `main 1.4 ms`, `shadow 380 us`). Filled on Metal when the device
// exposes `MTLCommonCounterSetTimestamp`; the chip stays blank on
// DirectX and Vulkan until their per-pass timing pools land. **The
// values are per-pass attributions, not components of `gpu_frame_us`**:
// the Apple GPU overlaps fragment work across encoders, so summing
// these consistently exceeds the whole-frame timer. RenderDoc /
// Instruments behave the same way; treat each row as standalone.
//
// ```jsonl
// {"type":"Font","name":"hud_font","args":{"size_px":20}}
// {"type":"TextLabel","name":"fps_chip","args":{"font":"hud_font","x":10,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"vram_chip","args":{"font":"hud_font","x":92,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"ev_chip","args":{"font":"hud_font","x":192,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"edr_chip","args":{"font":"hud_font","x":272,"y":10,"scale":0.7,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"TextLabel","name":"passes_chip","args":{"font":"hud_font","x":10,"y":36,"scale":0.6,"color":[1,1,1],"background":[0,0.22,0.08,0.85],"padding":5}}
// {"type":"StatHud","name":"hud","args":{"fps_label":"fps_chip","vram_label":"vram_chip","ev_label":"ev_chip","edr_label":"edr_chip","passes_label":"passes_chip"}}
// ```
#[derive(Debug)]
pub struct StatHudSystem {
    fps_label: Option<AssetId>,
    vram_label: Option<AssetId>,
    ev_label: Option<AssetId>,
    edr_label: Option<AssetId>,
    passes_label: Option<AssetId>,
    mouse_label: Option<AssetId>,
    camera_label: Option<AssetId>,
    // Whether the HUD is currently shown. Toggled by F1.
    visible: bool,
    // Start of the current averaging window.
    last_emit: Instant,
    // Frames counted since `last_emit`.
    frames: u32,
    // Most recent GPU-memory sample, bytes.
    vram_bytes: u64,
    // Most recent auto-exposure EV; `None` when auto-exposure is not active
    // for this world (the chip is then blanked).
    ev: Option<f32>,
    // Most recent panel EDR multiplier; `None` on the SDR path (the chip
    // is then blanked).
    max_edr: Option<f32>,
    // Most recent per-pass GPU microseconds (a snapshot of
    // `RenderStats.pass_times_us`); empty when the active backend has
    // no per-pass timing wired and the chip therefore stays blank.
    pass_times: Vec<PassTiming>,
    // Most recent cursor position (window pixels) from `FrameInput`.
    mouse_pos: (f32, f32),
    // Most recent live camera pose (position, yaw, pitch); `None` until a
    // `Camera3D` is seen (a world with no camera leaves the chip blank).
    camera_pose: Option<([f32; 3], f32, f32)>,
}

impl StatHudSystem {
    // Build the HUD from a world's `StatHud` request component.
    pub fn new(config: StatHud) -> Self {
        Self {
            fps_label: config.fps_label,
            vram_label: config.vram_label,
            ev_label: config.ev_label,
            edr_label: config.edr_label,
            passes_label: config.passes_label,
            mouse_label: config.mouse_label,
            camera_label: config.camera_label,
            visible: true,
            last_emit: Instant::now(),
            frames: 0,
            vram_bytes: 0,
            ev: None,
            max_edr: None,
            pass_times: Vec::new(),
            mouse_pos: (0.0, 0.0),
            camera_pose: None,
        }
    }

    // Write `text` into the TextLabel with the given id, if it exists.
    fn write_chip(ctx: &mut PipelineContext, id: Option<AssetId>, text: String) {
        let Some(id) = id else {
            return;
        };
        for label in ctx.query_mut::<TextLabel>() {
            if label.asset_id == id {
                label.content = text;
                return;
            }
        }
    }
}

impl System for StatHudSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // F1 toggles the HUD. FrameInput is queried (not drained) so
        // Camera3DSystem still consumes it later this frame; the same snapshot
        // carries the cursor position for the mouse chip.
        let frame_input = ctx.query::<FrameInput>().next();
        let toggled = frame_input.is_some_and(|input| input.hud_toggle);
        if let Some(input) = frame_input {
            self.mouse_pos = (input.mouse_x, input.mouse_y);
        }
        if toggled {
            self.visible = !self.visible;
            self.frames = 0;
            self.last_emit = Instant::now();
        }

        if !self.visible {
            // Blank chips read as empty content -> the renderer draws neither
            // text nor the background box, so the HUD fully disappears.
            Self::write_chip(ctx, self.fps_label, String::new());
            Self::write_chip(ctx, self.vram_label, String::new());
            Self::write_chip(ctx, self.ev_label, String::new());
            Self::write_chip(ctx, self.edr_label, String::new());
            Self::write_chip(ctx, self.passes_label, String::new());
            Self::write_chip(ctx, self.mouse_label, String::new());
            Self::write_chip(ctx, self.camera_label, String::new());
            return StepResult::Continue;
        }

        self.frames += 1;
        self.vram_bytes = ctx.profile.render.vram_bytes;
        self.ev = ctx.profile.render.auto_exposure_ev;
        self.max_edr = ctx.profile.render.max_edr;
        // Snapshot the per-pass slots once per frame so the chip refresh
        // works off a single stable sample instead of racing each
        // half-second emit against the latest atomic readback.
        self.pass_times.clear();
        self.pass_times
            .extend_from_slice(&ctx.profile.render.pass_times_us);
        // Live camera pose for the camera chip. Read here (before
        // Camera3DSystem steps) so a screenshot carries a settled reference
        // pose; one component read, cheap to refresh every frame.
        self.camera_pose = ctx
            .query::<Camera3D>()
            .next()
            .map(|c| (c.position, c.yaw, c.pitch));

        let now = Instant::now();
        let elapsed = now.duration_since(self.last_emit).as_secs_f32();
        if elapsed >= EMIT_INTERVAL_SECS {
            Self::write_chip(ctx, self.fps_label, fps_text(self.frames, elapsed));
            Self::write_chip(ctx, self.vram_label, vram_text(self.vram_bytes));
            Self::write_chip(ctx, self.ev_label, ev_text(self.ev));
            Self::write_chip(ctx, self.edr_label, edr_text(self.max_edr));
            Self::write_chip(ctx, self.passes_label, passes_text(&self.pass_times));
            Self::write_chip(
                ctx,
                self.mouse_label,
                mouse_text(self.mouse_pos.0, self.mouse_pos.1),
            );
            Self::write_chip(ctx, self.camera_label, camera_text(self.camera_pose));
            self.frames = 0;
            self.last_emit = now;
        }
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_text_averages_frames_over_window() {
        // 60 frames in 1.0 s -> 60 FPS.
        assert_eq!(fps_text(60, 1.0), "FPS 60");
        // 75 frames in 0.5 s -> 150 FPS.
        assert_eq!(fps_text(75, 0.5), "FPS 150");
    }

    #[test]
    fn fps_text_handles_zero_window() {
        assert_eq!(fps_text(0, 0.0), "FPS 0");
    }

    #[test]
    fn vram_text_reports_whole_megabytes() {
        assert_eq!(vram_text(0), "VRAM 0 MB");
        assert_eq!(vram_text(512 * 1024 * 1024), "VRAM 512 MB");
        // Truncates to whole MB.
        assert_eq!(vram_text(1024 * 1024 + 1), "VRAM 1 MB");
    }

    #[test]
    fn ev_text_formats_signed_value_with_two_decimals() {
        // Positive bias keeps the leading "+" so the sign is unambiguous on
        // screen.
        assert_eq!(ev_text(Some(1.25)), "EV +1.25");
        assert_eq!(ev_text(Some(-0.5)), "EV -0.50");
        assert_eq!(ev_text(Some(0.0)), "EV +0.00");
    }

    #[test]
    fn ev_text_blanks_when_auto_exposure_off() {
        // `None` means the world did not opt in to auto-exposure; the chip
        // stays empty so the HUD doesn't show a stale or misleading value.
        assert_eq!(ev_text(None), "");
    }

    #[test]
    fn ev_text_blanks_on_non_finite_values() {
        // A backend bug could leave the EV at NaN or infinity; render blank
        // rather than print "NaN" in the HUD.
        assert_eq!(ev_text(Some(f32::NAN)), "");
        assert_eq!(ev_text(Some(f32::INFINITY)), "");
    }

    #[test]
    fn edr_text_formats_multiplier_with_one_decimal() {
        // HDR400-class panels typically report 2.0; HDR1000-class 8.0+.
        // One decimal place keeps the chip narrow and unambiguous.
        assert_eq!(edr_text(Some(2.0)), "EDR x2.0");
        assert_eq!(edr_text(Some(8.5)), "EDR x8.5");
    }

    #[test]
    fn edr_text_blanks_when_sdr() {
        // `None` from the renderer means the SDR path is active: the chip
        // should not show an "EDR x1.0" reading because there is no HDR
        // headroom in play.
        assert_eq!(edr_text(None), "");
    }

    #[test]
    fn edr_text_blanks_on_invalid_values() {
        // Defensive: a backend reporting non-finite or non-positive max_edr
        // should not surface as "EDR xNaN" / "EDR x-1.0" on screen.
        assert_eq!(edr_text(Some(f32::NAN)), "");
        assert_eq!(edr_text(Some(f32::INFINITY)), "");
        assert_eq!(edr_text(Some(0.0)), "");
        assert_eq!(edr_text(Some(-1.0)), "");
    }

    #[test]
    fn passes_text_blanks_on_all_zero_slots() {
        // Every slot at the default ("", 0) → DX/Vulkan baseline → chip
        // stays empty so the HUD doesn't render an orphan "PASSES" header.
        let slots = vec![("", 0u32); 8];
        assert_eq!(passes_text(&slots), "");
    }

    #[test]
    fn passes_text_lists_top_entries_descending() {
        // Five recorded passes, three of them above the truncation cutoff
        // for a Top-N=6 chip. Order is by µs descending, regardless of the
        // input slot order.
        let slots = vec![
            ("shadow", 380),
            ("", 0),
            ("main", 1400),
            ("ssao_kernel", 120),
            ("composite", 60),
            ("ssr_resolve", 800),
            ("", 0),
        ];
        let out = passes_text(&slots);
        // Header first, then six lines for the recorded entries (well
        // under PASSES_CHIP_TOP_N so no truncation).
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "PASSES");
        assert_eq!(lines[1], "main 1.4 ms");
        assert_eq!(lines[2], "ssr_resolve 800 us");
        assert_eq!(lines[3], "shadow 380 us");
        assert_eq!(lines[4], "ssao_kernel 120 us");
        assert_eq!(lines[5], "composite 60 us");
        assert_eq!(lines.len(), 6);
    }

    #[test]
    fn passes_text_truncates_to_top_n() {
        // Eight non-empty slots, all distinct microsecond values → the
        // chip keeps only the top PASSES_CHIP_TOP_N (= 6) and drops the
        // smallest two.
        let slots: Vec<PassTiming> = vec![
            ("a", 80),
            ("b", 70),
            ("c", 60),
            ("d", 50),
            ("e", 40),
            ("f", 30),
            ("g", 20),
            ("h", 10),
        ];
        let out = passes_text(&slots);
        // Header line + exactly PASSES_CHIP_TOP_N entries.
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 1 + PASSES_CHIP_TOP_N);
        // The two smallest passes are dropped: never appear in the chip.
        assert!(!out.contains("g "));
        assert!(!out.contains("h "));
    }

    #[test]
    fn passes_text_formats_microseconds_below_one_ms() {
        // < 1000 µs reads as a bare integer; ≥ 1000 µs reads as a
        // single-decimal millisecond figure.
        let slots = vec![
            ("a", 999u32),
            ("b", 1000u32),
            ("c", 1499u32),
            ("d", 1500u32),
        ];
        let out = passes_text(&slots);
        assert!(out.contains("a 999 us"));
        assert!(out.contains("b 1.0 ms"));
        assert!(out.contains("c 1.5 ms"));
        assert!(out.contains("d 1.5 ms"));
    }

    #[test]
    fn mouse_text_rounds_to_whole_pixels() {
        assert_eq!(mouse_text(0.0, 0.0), "MOUSE 0, 0");
        // Sub-pixel coordinates round to the nearest whole pixel.
        assert_eq!(mouse_text(640.4, 360.6), "MOUSE 640, 361");
    }

    #[test]
    fn camera_text_formats_pose_in_camera_set_form() {
        // Position to two decimals, yaw/pitch to three, on two lines so the
        // chip reads back as the camera-set arguments.
        let out = camera_text(Some(([3.0, 1.6, 20.0], 1.2, -0.1)));
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "CAM 3.00 1.60 20.00");
        assert_eq!(lines[1], "yaw 1.200 pitch -0.100");
    }

    #[test]
    fn camera_text_blanks_without_camera() {
        // No Camera3D in the world: the chip stays empty rather than showing a
        // stale or zeroed pose.
        assert_eq!(camera_text(None), "");
    }

    // A StatHud component spawns the internal HUD system.
    #[test]
    fn stat_hud_component_spawns_internal_system() {
        use crate::assets::StatHud;
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(StatHud::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["StatHud"]);
    }

    #[test]
    fn no_stat_hud_no_system() {
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.start().unwrap();
        assert!(world.systems().is_empty());
    }
}
