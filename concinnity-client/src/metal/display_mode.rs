// src/metal/display_mode.rs
//
// Display-mode enumeration and fullscreen mode switching via CoreGraphics.
// `enumerate` lists the modes (pixel resolution + refresh rate) the window's
// display supports, feeding the Resolution settings row. While the window is
// in native fullscreen, `FullscreenDisplayMode` holds the display to the
// user's chosen mode with `CGDisplaySetDisplayMode` and restores the desktop's
// original mode when the window leaves fullscreen (including OS-driven exits:
// the reconcile runs once per frame off the delegate-tracked flag) or the
// context is dropped. Outside fullscreen the choice is only remembered; the
// windowed resize path is unaffected.

use objc2_app_kit::{NSScreen, NSWindow};
use objc2_core_foundation::{CFDictionary, CFRetained, Type, kCFBooleanTrue};
use objc2_core_graphics::{
    CGDirectDisplayID, CGDisplayCopyAllDisplayModes, CGDisplayCopyDisplayMode, CGDisplayMode,
    CGDisplaySetDisplayMode, CGError, CGMainDisplayID, kCGDisplayShowDuplicateLowResolutionModes,
};
use objc2_foundation::{NSNumber, ns_string};

use crate::gfx::display_mode::DisplayMode;

// The id of the display `window` sits on, falling back to the main display
// (also the embedded-mode answer, where no engine window exists).
fn display_id(window: Option<&NSWindow>) -> CGDirectDisplayID {
    window
        .and_then(|w| w.screen())
        .and_then(|s| screen_display_id(&s))
        .unwrap_or_else(|| CGMainDisplayID())
}

// NSScreen's display id, read from its device description ("NSScreenNumber").
fn screen_display_id(screen: &NSScreen) -> Option<CGDirectDisplayID> {
    let desc = screen.deviceDescription();
    let obj = desc.objectForKey(ns_string!("NSScreenNumber"))?;
    let num = obj.downcast_ref::<NSNumber>()?;
    Some(num.unsignedIntValue())
}

// The modes the window's display supports, as raw (pixel width, pixel height,
// refresh Hz) values; the caller dedups + sorts. Duplicate low-resolution
// (HiDPI) modes are requested so the common game resolutions appear on a
// Retina panel; modes CoreGraphics marks unusable for a desktop GUI (e.g.
// stretched) are skipped. Empty when CoreGraphics returns nothing.
pub(super) fn enumerate(window: Option<&NSWindow>) -> Vec<DisplayMode> {
    let display = display_id(window);
    let mut out = Vec::new();
    for mode in copy_all_modes(display) {
        if !CGDisplayMode::is_usable_for_desktop_gui(Some(&mode)) {
            continue;
        }
        if let Some(dm) = display_mode_of(&mode) {
            out.push(dm);
        }
    }
    out
}

// The mode the window's display is currently running.
pub(super) fn current(window: Option<&NSWindow>) -> Option<DisplayMode> {
    let mode = CGDisplayCopyDisplayMode(display_id(window))?;
    display_mode_of(&mode)
}

// Every CGDisplayMode of `display` (including HiDPI duplicates), retained.
fn copy_all_modes(display: CGDirectDisplayID) -> Vec<CFRetained<CGDisplayMode>> {
    // Ask for the duplicate low-resolution modes too; without the option the
    // scaled (HiDPI) resolutions a game typically offers are omitted.
    let options = unsafe {
        kCFBooleanTrue.map(|yes| {
            CFDictionary::from_slices(&[kCGDisplayShowDuplicateLowResolutionModes], &[yes])
        })
    };
    let array =
        unsafe { CGDisplayCopyAllDisplayModes(display, options.as_deref().map(|d| d.as_opaque())) };
    let Some(array) = array else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for i in 0..array.count() {
        // The array's values are CGDisplayModes (documented by
        // CGDisplayCopyAllDisplayModes) and `i` is in bounds, so the raw
        // accessor's requirements hold; retaining detaches the reference from
        // the array's lifetime.
        let ptr = unsafe { array.value_at_index(i) };
        if ptr.is_null() {
            continue;
        }
        let mode: &CGDisplayMode = unsafe { &*ptr.cast() };
        out.push(mode.retain());
    }
    out
}

// The (pixel width, pixel height, rounded refresh Hz) of a CGDisplayMode, or
// `None` for a degenerate mode. A refresh of 0 stays 0 (unknown; some built-in
// panels report none).
fn display_mode_of(mode: &CGDisplayMode) -> Option<DisplayMode> {
    let width = CGDisplayMode::pixel_width(Some(mode)) as u32;
    let height = CGDisplayMode::pixel_height(Some(mode)) as u32;
    if width == 0 || height == 0 {
        return None;
    }
    let refresh_hz = CGDisplayMode::refresh_rate(Some(mode)).round().max(0.0) as u32;
    Some(DisplayMode {
        width,
        height,
        refresh_hz,
    })
}

// The CGDisplayMode of `display` matching `want`. An exact (resolution, rate)
// match wins; a `want` with an unknown rate (0) takes the matching resolution's
// highest rate. `None` when the display has no mode at that resolution (e.g. a
// stale persisted choice from another monitor).
fn find_native_mode(
    display: CGDirectDisplayID,
    want: DisplayMode,
) -> Option<CFRetained<CGDisplayMode>> {
    let mut best: Option<(u32, CFRetained<CGDisplayMode>)> = None;
    for mode in copy_all_modes(display) {
        let Some(dm) = display_mode_of(&mode) else {
            continue;
        };
        if dm.width != want.width || dm.height != want.height {
            continue;
        }
        if want.refresh_hz != 0 && dm.refresh_hz == want.refresh_hz {
            return Some(mode);
        }
        if best.as_ref().is_none_or(|(hz, _)| dm.refresh_hz > *hz) {
            best = Some((dm.refresh_hz, mode));
        }
    }
    // With a known wanted rate but no exact rate match, the resolution's
    // nearest-available (highest) rate still honours the resolution choice.
    best.map(|(_, mode)| mode)
}

// Holds the display to the user's chosen mode while the window is in native
// fullscreen. Owned by MtlContext; `reconcile` runs once per frame from the
// delegate-tracked fullscreen flag, so entry (after the animation starts),
// menu-driven exit, and OS-driven exit (green traffic-light button, Mission
// Control) all converge on the right display state.
pub(super) struct FullscreenDisplayMode {
    // The user's chosen mode, or `None` to leave the display alone.
    desired: Option<DisplayMode>,
    // The desktop mode captured before the first switch, restored on exit.
    original: Option<CFRetained<CGDisplayMode>>,
    // The display being held, resolved from the window when first switching
    // (a fullscreen window cannot change displays mid-hold).
    display: Option<CGDirectDisplayID>,
    // The mode currently applied, so the per-frame reconcile is a no-op
    // comparison while nothing changes.
    applied: Option<DisplayMode>,
}

impl std::fmt::Debug for FullscreenDisplayMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullscreenDisplayMode")
            .field("desired", &self.desired)
            .field("applied", &self.applied)
            .field("holds_original", &self.original.is_some())
            .finish()
    }
}

impl FullscreenDisplayMode {
    pub(super) fn new() -> Self {
        Self {
            desired: None,
            original: None,
            display: None,
            applied: None,
        }
    }

    // Remember the mode to hold while fullscreen. Applied (or re-applied) by
    // the next reconcile; outside fullscreen it only updates the memory.
    pub(super) fn set_desired(&mut self, mode: DisplayMode) {
        self.desired = Some(mode);
    }

    // Converge the display on the current fullscreen state: switch to the
    // desired mode while fullscreen, restore the desktop mode otherwise.
    // Cheap when nothing changed (two field compares).
    pub(super) fn reconcile(&mut self, window: Option<&NSWindow>, fullscreen: bool) {
        if !fullscreen {
            self.restore();
            return;
        }
        let Some(desired) = self.desired else {
            return;
        };
        if self.applied == Some(desired) {
            return;
        }
        let display = *self.display.get_or_insert_with(|| display_id(window));
        let Some(mode) = find_native_mode(display, desired) else {
            // No mode at that resolution on this display (a stale persisted
            // choice): leave the display alone and stop retrying.
            tracing::warn!(
                "display has no {}x{} mode; keeping the current mode",
                desired.width,
                desired.height
            );
            self.desired = None;
            return;
        };
        if self.original.is_none() {
            self.original = CGDisplayCopyDisplayMode(display);
        }
        let err = unsafe { CGDisplaySetDisplayMode(display, Some(&mode), None) };
        if err == CGError::Success {
            self.applied = Some(desired);
        } else {
            tracing::warn!(
                "CGDisplaySetDisplayMode({}x{}@{}Hz) failed: {:?}",
                desired.width,
                desired.height,
                desired.refresh_hz,
                err
            );
            self.desired = None;
        }
    }

    // Put the display back on the desktop mode captured before the first
    // switch. Idempotent; also called from MtlContext::drop so quitting while
    // fullscreen never strands the desktop on the game's mode.
    pub(super) fn restore(&mut self) {
        let Some(original) = self.original.take() else {
            return;
        };
        let Some(display) = self.display else {
            return;
        };
        let err = unsafe { CGDisplaySetDisplayMode(display, Some(&original), None) };
        if err != CGError::Success {
            tracing::warn!("restoring the desktop display mode failed: {:?}", err);
        }
        self.applied = None;
    }
}

impl Drop for FullscreenDisplayMode {
    fn drop(&mut self) {
        self.restore();
    }
}
