// src/hud/debug_hud.rs
//
// Developer debug-HUD overlay behavior. An internal system (not a declarable
// asset): `World::start` constructs one from the world's `DebugHud` component
// and it writes diagnostic readouts into that component's `TextLabel` chips.
//
// The chips are toggled together with F1 (hidden by default) and anchored to
// the top-right of the window by `GraphicsSystem` (it owns the font metrics and
// live window size needed to right-align and stack them).

use crate::assets::{Camera3D, DebugHud, FrameInput, TextLabel};
use crate::ecs::asset_id::AssetId;
use crate::ecs::{PipelineContext, StepResult, System};
use crate::gfx::profile::PassTiming;

// How many per-pass entries the passes chip lists. Picked to fit comfortably
// in the top-right debug column; passes past this count are dropped from the
// chip (still visible via the debug WS `profile.passes` reply).
const PASSES_CHIP_TOP_N: usize = 6;

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

/// Draws the developer debug HUD: a multi-line `PASSES` chip listing the top
/// render-graph passes of the last frame, a `MOUSE` chip with the cursor
/// position, and a `CAM` chip with the live camera pose. The chips are anchored
/// to the top-right of the window (stacked cursor, passes, camera) and toggled
/// together with **F1**; the HUD starts hidden.
///
/// `PASSES` lists the top six passes in descending GPU-microsecond order
/// (e.g. `main 1.4 ms`, `shadow 380 us`). Filled on Metal when the device
/// exposes `MTLCommonCounterSetTimestamp`; blank on DirectX / Vulkan until
/// their per-pass timing pools land. **The values are per-pass attributions,
/// not components of `gpu_frame_us`**: the Apple GPU overlaps fragment work
/// across encoders, so summing them exceeds the whole-frame timer.
#[derive(Debug)]
pub struct DebugHudSystem {
    passes_label: Option<AssetId>,
    mouse_label: Option<AssetId>,
    camera_label: Option<AssetId>,
    // Whether the HUD is currently shown. Toggled by F1; hidden by default.
    visible: bool,
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

impl DebugHudSystem {
    // Build the debug HUD from a world's `DebugHud` request component.
    pub fn new(config: DebugHud) -> Self {
        Self {
            passes_label: config.passes_label,
            mouse_label: config.mouse_label,
            camera_label: config.camera_label,
            visible: false,
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

impl System for DebugHudSystem {
    fn step(&mut self, ctx: &mut PipelineContext) -> StepResult {
        // F1 toggles the debug HUD. The per-frame input snapshot is read from
        // the FrameInput resource GraphicsSystem publishes earlier in the frame;
        // it also carries the cursor position for the mouse chip.
        let frame_input = ctx.resource::<FrameInput>();
        let toggled = frame_input.is_some_and(|input| input.hud_toggle);
        if let Some(input) = frame_input {
            self.mouse_pos = (input.mouse_x, input.mouse_y);
        }
        if toggled {
            self.visible = !self.visible;
        }

        if !self.visible {
            // Blank chips read as empty content -> the renderer draws neither
            // text nor the background box, so the HUD fully disappears.
            Self::write_chip(ctx, self.passes_label, String::new());
            Self::write_chip(ctx, self.mouse_label, String::new());
            Self::write_chip(ctx, self.camera_label, String::new());
            return StepResult::Continue;
        }

        // Snapshot the per-pass slots once per frame so the chip refresh works
        // off a single stable sample instead of racing the latest readback.
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

        Self::write_chip(ctx, self.passes_label, passes_text(&self.pass_times));
        Self::write_chip(
            ctx,
            self.mouse_label,
            mouse_text(self.mouse_pos.0, self.mouse_pos.1),
        );
        Self::write_chip(ctx, self.camera_label, camera_text(self.camera_pose));
        StepResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // A DebugHud component spawns the internal debug-HUD system.
    #[test]
    fn debug_hud_component_spawns_internal_system() {
        use crate::ecs::World;

        let mut world = World::new_empty();
        world.add_component(DebugHud::default());
        world.start().unwrap();
        let names: Vec<&str> = world.systems().iter().map(|s| s.name()).collect();
        assert_eq!(names, ["DebugHud"]);
    }
}
