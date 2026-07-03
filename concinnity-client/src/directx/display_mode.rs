// src/directx/display_mode.rs
//
// Display-mode enumeration and fullscreen mode switching via the Win32
// ChangeDisplaySettings family. `enumerate` lists the modes (resolution +
// refresh rate) of the monitor the window sits on, feeding the Resolution
// settings row. While the window is in (borderless) fullscreen,
// `FullscreenDisplayMode` holds the monitor to the user's chosen mode with
// `ChangeDisplaySettingsExW` and restores the monitor's original mode when the
// window leaves fullscreen or the context is dropped (`CDS_FULLSCREEN` marks
// the switch temporary, so the OS also restores it if the process dies).
// Outside fullscreen the choice is only remembered. Mirrors
// `metal/display_mode.rs`.

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Gdi::{
    CDS_FULLSCREEN, ChangeDisplaySettingsExW, DEVMODEW, DISP_CHANGE_SUCCESSFUL,
    ENUM_CURRENT_SETTINGS, ENUM_DISPLAY_SETTINGS_MODE, EnumDisplaySettingsW, GetMonitorInfoW,
    MONITOR_DEFAULTTONEAREST, MONITORINFOEXW, MonitorFromWindow,
};
use windows::core::PCWSTR;

use crate::gfx::display_mode::{DisplayMode, best_native_index};

// The GDI device name (e.g. "\\.\DISPLAY1") of the monitor `hwnd` is mostly
// on, the key the EnumDisplaySettings/ChangeDisplaySettings calls address a
// monitor by. None if the monitor-info query fails.
fn device_name(hwnd: HWND) -> Option<[u16; 32]> {
    let mon = unsafe { MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST) };
    let mut info = MONITORINFOEXW::default();
    // GetMonitorInfoW fills the extended struct (device name included) when
    // cbSize says the buffer is the EXW size; the two structs share a prefix.
    info.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
    unsafe { GetMonitorInfoW(mon, &mut info.monitorInfo) }
        .as_bool()
        .then_some(info.szDevice)
}

// The DEVMODE at `index` of `device` (or ENUM_CURRENT_SETTINGS for the mode
// the monitor is running), or None past the end of the mode list.
fn enum_mode(device: &[u16; 32], index: ENUM_DISPLAY_SETTINGS_MODE) -> Option<DEVMODEW> {
    let mut dm = DEVMODEW {
        dmSize: std::mem::size_of::<DEVMODEW>() as u16,
        ..Default::default()
    };
    unsafe { EnumDisplaySettingsW(PCWSTR(device.as_ptr()), index, &mut dm) }
        .as_bool()
        .then_some(dm)
}

// The (width, height, refresh Hz) of a DEVMODE, or None for a degenerate or
// legacy low-color mode. Modern drivers list only 32-bit modes; skipping any
// lower depth keeps the row (and the apply below) off 16-bit variants. A
// frequency of 0 or 1 means hardware default (unknown) and maps to 0.
fn display_mode_of(dm: &DEVMODEW) -> Option<DisplayMode> {
    if dm.dmPelsWidth == 0 || dm.dmPelsHeight == 0 || dm.dmBitsPerPel < 32 {
        return None;
    }
    let refresh_hz = if dm.dmDisplayFrequency <= 1 {
        0
    } else {
        dm.dmDisplayFrequency
    };
    Some(DisplayMode {
        width: dm.dmPelsWidth,
        height: dm.dmPelsHeight,
        refresh_hz,
    })
}

// Every DEVMODE of `device` with its shaped DisplayMode, in enumeration order.
fn all_modes(device: &[u16; 32]) -> Vec<(DisplayMode, DEVMODEW)> {
    let mut out = Vec::new();
    let mut i = 0u32;
    while let Some(dm) = enum_mode(device, ENUM_DISPLAY_SETTINGS_MODE(i)) {
        i += 1;
        if let Some(mode) = display_mode_of(&dm) {
            out.push((mode, dm));
        }
    }
    out
}

// The modes the window's monitor supports, as raw (width, height, refresh Hz)
// values; the caller dedups + sorts. Empty when the monitor query fails.
pub(super) fn enumerate(hwnd: HWND) -> Vec<DisplayMode> {
    let Some(device) = device_name(hwnd) else {
        return Vec::new();
    };
    all_modes(&device).into_iter().map(|(m, _)| m).collect()
}

// The mode the window's monitor is currently running.
pub(super) fn current(hwnd: HWND) -> Option<DisplayMode> {
    let device = device_name(hwnd)?;
    display_mode_of(&enum_mode(&device, ENUM_CURRENT_SETTINGS)?)
}

// Holds the monitor on the user's chosen mode while the window is in
// (borderless) fullscreen. Owned by DxContext; `reconcile` runs once per frame
// from `window_closed` (after the message pump), so a mode chosen in any
// window mode, a later fullscreen entry, and a menu-driven exit all converge
// on the right monitor state. Cheap when nothing changed (two field compares).
pub(super) struct FullscreenDisplayMode {
    // The user's chosen mode, or None to leave the monitor alone.
    desired: Option<DisplayMode>,
    // The monitor mode captured before the first switch, restored on exit.
    original: Option<DEVMODEW>,
    // The monitor being held, resolved from the window when first switching
    // (the covering fullscreen window cannot change monitors mid-hold).
    device: Option<[u16; 32]>,
    // The mode currently applied, so the per-frame reconcile is a no-op
    // comparison while nothing changes.
    applied: Option<DisplayMode>,
}

impl FullscreenDisplayMode {
    pub(super) fn new() -> Self {
        Self {
            desired: None,
            original: None,
            device: None,
            applied: None,
        }
    }

    // Remember the mode to hold while fullscreen. Applied (or re-applied) by
    // the next reconcile; outside fullscreen it only updates the memory.
    pub(super) fn set_desired(&mut self, mode: DisplayMode) {
        self.desired = Some(mode);
    }

    // Converge the monitor on the current fullscreen state: switch to the
    // desired mode while fullscreen, restore the original mode otherwise.
    // Returns true when the monitor mode changed, so the caller can re-cover
    // the monitor's new bounds with the window.
    pub(super) fn reconcile(&mut self, hwnd: HWND, fullscreen: bool) -> bool {
        if !fullscreen {
            return self.restore();
        }
        let Some(desired) = self.desired else {
            return false;
        };
        if self.applied == Some(desired) {
            return false;
        }
        if self.device.is_none() {
            self.device = device_name(hwnd);
        }
        let Some(device) = self.device else {
            return false;
        };
        let modes = all_modes(&device);
        let shaped: Vec<DisplayMode> = modes.iter().map(|(m, _)| *m).collect();
        let Some(idx) = best_native_index(&shaped, desired) else {
            // No mode at that resolution on this monitor (a stale persisted
            // choice): leave the monitor alone and stop retrying.
            tracing::warn!(
                "display has no {}x{} mode; keeping the current mode",
                desired.width,
                desired.height
            );
            self.desired = None;
            return false;
        };
        if self.original.is_none() {
            self.original = enum_mode(&device, ENUM_CURRENT_SETTINGS);
        }
        let devmode = &modes[idx].1;
        let err = unsafe {
            ChangeDisplaySettingsExW(
                PCWSTR(device.as_ptr()),
                Some(std::ptr::from_ref(devmode)),
                None,
                CDS_FULLSCREEN,
                None,
            )
        };
        if err == DISP_CHANGE_SUCCESSFUL {
            self.applied = Some(desired);
            true
        } else {
            tracing::warn!(
                "ChangeDisplaySettingsExW({}x{}@{}Hz) failed: {:?}",
                desired.width,
                desired.height,
                desired.refresh_hz,
                err
            );
            self.desired = None;
            false
        }
    }

    // Put the monitor back on the mode captured before the first switch.
    // Idempotent; also runs on drop so quitting while fullscreen never strands
    // the desktop on the game's mode. Returns true when a restore happened.
    pub(super) fn restore(&mut self) -> bool {
        let Some(original) = self.original.take() else {
            return false;
        };
        let Some(device) = self.device else {
            return false;
        };
        let err = unsafe {
            ChangeDisplaySettingsExW(
                PCWSTR(device.as_ptr()),
                Some(std::ptr::from_ref(&original)),
                None,
                CDS_FULLSCREEN,
                None,
            )
        };
        if err != DISP_CHANGE_SUCCESSFUL {
            tracing::warn!("restoring the desktop display mode failed: {:?}", err);
        }
        self.applied = None;
        true
    }
}

impl Drop for FullscreenDisplayMode {
    fn drop(&mut self) {
        self.restore();
    }
}
