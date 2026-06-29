#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2_app_kit::{
    NSApplication, NSCursor, NSEventMask, NSEventModifierFlags, NSEventType, NSWindow,
    NSWindowButton, NSWindowStyleMask, NSWindowTitleVisibility,
};
use objc2_foundation::{NSDate, NSPoint, NSSize};

use crate::assets::{Key, WindowMode};
use crate::gfx::keymap::KeyMap;

use super::context::MtlContext;

// The previously-duplicated InputState collapsed into the shared
// crate::gfx::input::RenderInput; this alias keeps the historical name.
pub use crate::gfx::input::RenderInput as InputState;

// Persistent key state tracked across frames. Key booleans are set on KeyDown
// and cleared on KeyUp; they are never reset between frames so that held keys
// remain active even when no repeat event arrives (avoiding the OS key-repeat
// delay gap). Mouse deltas are accumulated here and cleared by take_input().
// Pulse fields are set on KeyDown and cleared after one take_input() call so
// callers see exactly one true frame per press.
#[derive(Default)]
pub(super) struct KeyState {
    pub(super) forward: bool,
    pub(super) backward: bool,
    pub(super) left: bool,
    pub(super) right: bool,
    pub(super) sprint: bool,
    pub(super) interact_pulse: bool,
    pub(super) jump_pulse: bool,
    pub(super) mouse_dx: f32,
    pub(super) mouse_dy: f32,
    // Accumulated vertical scroll-wheel delta since the last take_input();
    // cleared by take_input() like the mouse deltas. Used by scrollable UI.
    pub(super) scroll_delta: f32,
    // Absolute cursor position in window-content pixels (origin top-left).
    pub(super) mouse_x: f32,
    pub(super) mouse_y: f32,
    // Pulse: set on left-mouse-down when cursor is free; cleared by take_input().
    pub(super) left_click_pulse: bool,
    // Held: set on left-mouse-down and cleared on left-mouse-up (cursor free).
    // Unlike the pulse it persists across frames, so a UI drag (slider) can
    // track the cursor for the whole press. NOT cleared by take_input().
    pub(super) left_button_down: bool,
    // Pulse: set on F1 key-down; cleared by take_input().
    pub(super) hud_toggle_pulse: bool,
    // Pulse: set on Escape key-down when the cursor is not captured;
    // cleared by take_input(). When the cursor is captured Escape continues
    // to call release_cursor() instead.
    pub(super) escape_pulse: bool,
    // Pulse: the canonical key pressed since the last take_input(), for the
    // settings menu's rebind capture. Set on any KeyDown with a known mapping
    // (and on the Shift rising edge); cleared by take_input(). Not gated by
    // capture / menu state so a rebind row can read it while a menu is open.
    pub(super) captured_key: Option<Key>,
    // Whether Shift is currently held, tracked from FlagsChanged so the rising
    // edge can fire `captured_key` and drive any action bound to Shift (Shift is
    // a pure modifier on macOS: it generates FlagsChanged, not KeyDown/KeyUp).
    pub(super) shift_down: bool,
    // Set by capture_cursor(); the next mouse-motion event after capture
    // has its delta discarded so queued pre-capture events (which were
    // produced before CGAssociateMouseAndMouseCursorPosition(0) took
    // effect, often during init) can't snap the camera.
    pub(super) discard_next_motion: bool,
}

unsafe extern "C" {
    // Moves the OS cursor without generating a mouse-moved event.
    fn CGWarpMouseCursorPosition(new_cursor_position: NSPoint) -> i32;
    // When connected=false (0), decouples cursor position from hardware mouse
    // movement so deltaX/deltaY in NSEvents are pure hardware deltas with no
    // warp feedback. Part of CoreGraphics (CGRemoteOperation.h).
    fn CGAssociateMouseAndMouseCursorPosition(connected: i32) -> i32;
}

// Restore a standard titled window's chrome after leaving borderless mode.
fn restore_titlebar(window: &NSWindow) {
    window.setTitlebarAppearsTransparent(false);
    window.setTitleVisibility(NSWindowTitleVisibility::Visible);
    set_window_buttons_hidden(window, false);
}

// Show or hide the close / minimize / zoom traffic-light buttons.
fn set_window_buttons_hidden(window: &NSWindow, hidden: bool) {
    for kind in [
        NSWindowButton::CloseButton,
        NSWindowButton::MiniaturizeButton,
        NSWindowButton::ZoomButton,
    ] {
        if let Some(button) = window.standardWindowButton(kind) {
            button.setHidden(hidden);
        }
    }
}

impl MtlContext {
    // The NSWindow currently hosting the renderer. In windowed mode this is
    // the NSWindow we created; in embedded mode (preview tab, or the
    // play-in-view path where Swift owns the window) it is the MTKView's
    // host. Returns None only when the MTKView isn't yet in a window
    // (transient: during init the parent hasn't been set yet).
    pub(super) fn host_window(&self) -> Option<Retained<NSWindow>> {
        if let Some(ref w) = self.window {
            return Some(w.clone());
        }
        self.mtk_view.window()
    }

    // Hide the cursor and begin accumulating relative mouse deltas. No-op
    // for the preview tab (pump_events=false), where the cursor must remain
    // usable for the SwiftUI tab bar and sidebar controls. Also a no-op when
    // no host window is yet attached.
    pub fn capture_cursor(&mut self) {
        if !self.pump_events {
            return;
        }
        let Some(window) = self.host_window() else {
            return;
        };
        NSCursor::hide();
        // Decouple cursor position from movement so deltaX/deltaY are pure
        // hardware deltas. Without this, CGWarpMouseCursorPosition generates
        // a spurious event with the warp distance as its delta, causing a
        // camera snap on first mouse move.
        unsafe { CGAssociateMouseAndMouseCursorPosition(0) };
        // Warp once to centre so the cursor is in a known position.
        if let Some(screen) = window.screen() {
            let frame = window.frame();
            let centre = NSPoint::new(
                frame.origin.x + frame.size.width * 0.5,
                screen.frame().size.height - (frame.origin.y + frame.size.height * 0.5),
            );
            unsafe { CGWarpMouseCursorPosition(centre) };
        }
        // Drop any deltas already accumulated before capture, and arm a
        // one-shot discard so the first motion event pumped after capture
        // (which may have been queued during init, before the OS settled
        // into raw-delta mode) doesn't snap the camera.
        self.keys.mouse_dx = 0.0;
        self.keys.mouse_dy = 0.0;
        self.keys.discard_next_motion = true;
        self.cursor_captured = true;
        self.recapture_on_click = false;
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered: NSCursor hide/unhide are
    // ref-counted, so we only toggle on a state change. No-op for the preview
    // tab (pump_events=false), which must keep the system cursor usable.
    pub fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        if !self.pump_events || hidden == self.ui_cursor_hidden {
            return;
        }
        self.ui_cursor_hidden = hidden;
        if hidden {
            NSCursor::hide();
        } else {
            NSCursor::unhide();
        }
    }

    // Show the cursor and stop accumulating mouse deltas.
    pub fn release_cursor(&mut self) {
        if !self.cursor_captured {
            return;
        }
        self.cursor_captured = false;
        self.recapture_on_click = true;
        unsafe { CGAssociateMouseAndMouseCursorPosition(1) };
        NSCursor::unhide();
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`.
    pub fn set_menu_mode(&mut self, on: bool) {
        self.menu_mode = on;
    }

    // Edge-triggered capture: capture for camera control, release while a menu
    // is open. GraphicsSystem calls this each frame in menu mode.
    pub fn set_camera_capture(&mut self, capture: bool) {
        if capture == self.cursor_captured {
            return;
        }
        if capture {
            self.capture_cursor();
        } else {
            self.release_cursor();
        }
    }

    // Turn display sync (vsync) on or off at runtime via the backing
    // CAMetalLayer. Setting displaySyncEnabled is an idempotent property write
    // (no swapchain rebuild on Metal), so a redundant call is cheap.
    pub fn set_vsync(&mut self, on: bool) {
        super::init::set_display_sync(&self.mtk_view, on);
    }

    // Switch the engine-created window between windowed / borderless /
    // fullscreen. Only `self.window` is touched: in embedded mode (the preview
    // tab or a Swift-owned window) this is a no-op so we never restyle a host
    // window. The change flows through the per-frame drawableSize() resize
    // detection, so no render targets are rebuilt here.
    pub fn set_window_mode(&mut self, mode: WindowMode) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let standard = NSWindowStyleMask::Titled
            | NSWindowStyleMask::Closable
            | NSWindowStyleMask::Miniaturizable
            | NSWindowStyleMask::Resizable;
        // Read the fullscreen state from the flag the NSWindowDelegate keeps in
        // sync (it flips at the start of the animation via
        // windowWillEnter/ExitFullScreen). This does not lag the way the
        // style-mask bit does, so stepping the Window Mode row faster than the
        // ~1s native-fullscreen animation no longer toggles the wrong way.
        let is_fullscreen = self.fullscreen.load(std::sync::atomic::Ordering::Relaxed);
        // Record the intended fullscreen state synchronously so a second step
        // issued before the delegate callback lands still decides correctly;
        // the delegate's did-callbacks reconcile this with reality at the end
        // of the transition (and capture OS-driven toggles like the green
        // traffic-light button).
        self.fullscreen.store(
            matches!(mode, WindowMode::Fullscreen),
            std::sync::atomic::Ordering::Relaxed,
        );
        match mode {
            WindowMode::Windowed => {
                if is_fullscreen {
                    window.toggleFullScreen(None);
                }
                window.setStyleMask(standard);
                restore_titlebar(window);
            }
            WindowMode::Borderless => {
                if is_fullscreen {
                    window.toggleFullScreen(None);
                }
                // Keep the window key-window-eligible (a pure Borderless,
                // non-panel window cannot become key, which kills keyboard
                // input): a Titled + full-size-content window with a
                // transparent, hidden title bar and hidden traffic-light
                // buttons reads as borderless but still receives key events.
                window.setStyleMask(
                    NSWindowStyleMask::Titled
                        | NSWindowStyleMask::Closable
                        | NSWindowStyleMask::Resizable
                        | NSWindowStyleMask::FullSizeContentView,
                );
                window.setTitlebarAppearsTransparent(true);
                window.setTitleVisibility(NSWindowTitleVisibility::Hidden);
                set_window_buttons_hidden(window, true);
                // Borderless covers the window's current display.
                if let Some(screen) = window.screen() {
                    window.setFrame_display(screen.frame(), true);
                }
            }
            WindowMode::Fullscreen => {
                // Native fullscreen animates from a standard titled window.
                window.setStyleMask(standard);
                restore_titlebar(window);
                if !is_fullscreen {
                    window.toggleFullScreen(None);
                }
            }
        }
        // Re-acquire key + front so keyboard input keeps flowing after a restyle.
        window.makeKeyAndOrderFront(None);
    }

    // Resize the engine-created window's content area (windowed mode only).
    // No-op in embedded mode or while in native fullscreen.
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        // Resizing the content area is meaningless while in native fullscreen;
        // read the delegate-tracked flag (not the lagging style-mask bit).
        if self.fullscreen.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let Some(window) = self.window.as_ref() else {
            return;
        };
        window.setContentSize(NSSize::new(width as f64, height as f64));
    }

    // Replace the live post-process parameters. They are pushed to the bloom
    // prefilter + composite shaders every frame (see draw/composite.rs), so a
    // change takes effect on the next draw with no allocation or pipeline
    // rebuild. Auto-exposure, when on, overwrites `exposure` each frame from
    // the adapted EV, so a static exposure change is only visible with
    // auto-exposure off.
    pub fn update_post_process(&mut self, params: crate::gfx::render_types::PostProcessParams) {
        self.post_process = params;
    }

    // Set the live ambient (IBL) light scale. `ambient_intensity` lives in
    // `LightUniforms`, which the main lighting pass uploads every frame, so the
    // change takes effect on the next draw with no allocation. It is not
    // re-derived per frame (unlike auto-exposure's `exposure`), so the value
    // stands until changed again.
    pub fn set_ambient_intensity(&mut self, value: f32) {
        self.light_uniforms.ambient_intensity = value;
    }

    // Set the live shadow cascade re-render cadence. The scheduler reads
    // `shadow_update` at the start of each shadow pass, so a change takes effect
    // on the next draw. Every cascade is already primed, so switching policy never
    // leaves a slice unsampled (priming is one-shot per cascade, not per policy).
    pub fn set_shadow_update(&mut self, update: crate::assets::ShadowUpdate) {
        self.shadow_update = update;
    }

    // Snapshot the current input state for this frame.
    // Key booleans reflect what is held right now; mouse deltas are cleared
    // after being read so they don't accumulate across frames.
    // `interact` and `jump` are true for exactly one frame per key press then cleared.
    pub fn take_input(&mut self) -> InputState {
        let snapshot = InputState {
            forward: self.keys.forward,
            backward: self.keys.backward,
            left: self.keys.left,
            right: self.keys.right,
            sprint: self.keys.sprint,
            interact: self.keys.interact_pulse,
            jump: self.keys.jump_pulse,
            mouse_dx: self.keys.mouse_dx,
            mouse_dy: self.keys.mouse_dy,
            scroll_delta: self.keys.scroll_delta,
            mouse_x: self.keys.mouse_x,
            mouse_y: self.keys.mouse_y,
            left_click: self.keys.left_click_pulse,
            // Held state: read but not cleared here (cleared on LeftMouseUp).
            left_button_down: self.keys.left_button_down,
            hud_toggle: self.keys.hud_toggle_pulse,
            escape: self.keys.escape_pulse,
            captured_key: self.keys.captured_key,
        };
        self.keys.interact_pulse = false;
        self.keys.jump_pulse = false;
        self.keys.mouse_dx = 0.0;
        self.keys.mouse_dy = 0.0;
        self.keys.scroll_delta = 0.0;
        self.keys.left_click_pulse = false;
        self.keys.hud_toggle_pulse = false;
        self.keys.escape_pulse = false;
        self.keys.captured_key = None;
        snapshot
    }

    // Dequeue all pending NSEvents and update input state. Sets window_closed=true
    // on a window-will-close application event. Key events update the persistent
    // key state; mouse moved events accumulate deltas if the cursor is captured.
    pub(super) fn pump_ns_events(&mut self, mtm: objc2::MainThreadMarker) {
        let ns_app = NSApplication::sharedApplication(mtm);
        loop {
            let event = ns_app.nextEventMatchingMask_untilDate_inMode_dequeue(
                NSEventMask::Any,
                Some(&NSDate::distantPast()),
                objc2_foundation::ns_string!("kCFRunLoopDefaultMode"),
                true,
            );
            let event = match event {
                Some(e) => e,
                None => break,
            };

            match event.r#type() {
                NSEventType::KeyDown => self.handle_key(&event, true),
                NSEventType::KeyUp => self.handle_key(&event, false),
                NSEventType::FlagsChanged => {
                    // Fires immediately when a modifier key is pressed or
                    // released, independent of any other key event. Shift is a
                    // pure modifier on macOS (no KeyDown/KeyUp), so it is decoded
                    // here: drive any action bound to Shift (sprint by default)
                    // and fire the rebind-capture pulse on its rising edge.
                    let shift = event.modifierFlags().contains(NSEventModifierFlags::Shift);
                    let edge_down = shift && !self.keys.shift_down;
                    self.keys.shift_down = shift;
                    if edge_down {
                        self.keys.captured_key = Some(Key::Shift);
                    }
                    self.apply_binding(Key::Shift, shift, edge_down);
                }
                NSEventType::MouseMoved | NSEventType::LeftMouseDragged => {
                    if self.cursor_captured {
                        // CGAssociateMouseAndMouseCursorPosition(false) is active while captured,
                        // so deltaX/deltaY are pure hardware deltas with no warp
                        // feedback. No per-event warp needed.
                        if self.keys.discard_next_motion {
                            self.keys.discard_next_motion = false;
                        } else {
                            self.keys.mouse_dx += event.deltaX() as f32;
                            self.keys.mouse_dy += event.deltaY() as f32;
                        }
                    } else {
                        // Track absolute position for UI hit-testing.
                        // locationInWindow() origin is bottom-left of the content area (AppKit
                        // convention); flip Y so (0,0) is top-left, matching TextLabel coords.
                        let loc = event.locationInWindow();
                        let h = self
                            .host_window()
                            .map(|w| w.contentRectForFrameRect(w.frame()).size.height as f32)
                            .unwrap_or(720.0);
                        self.keys.mouse_x = loc.x as f32;
                        self.keys.mouse_y = h - loc.y as f32;
                    }
                }
                NSEventType::LeftMouseDown => {
                    if !self.cursor_captured {
                        // In menu mode a click fires a UI action; capture is
                        // driven by the active menu, not by clicking.
                        if !self.menu_mode
                            && self.recapture_on_click
                            && self.in_content_area(&event)
                        {
                            self.capture_cursor();
                        } else {
                            self.keys.left_click_pulse = true;
                            self.keys.left_button_down = true;
                        }
                    }
                    ns_app.sendEvent(&event);
                }
                NSEventType::LeftMouseUp => {
                    // End any held-button state (drag release). Always cleared,
                    // even if the down began while captured, so the flag can
                    // never stick across a capture transition.
                    self.keys.left_button_down = false;
                    ns_app.sendEvent(&event);
                }
                NSEventType::ScrollWheel => {
                    // Accumulate the wheel delta for scrollable UI while the
                    // cursor is free. scrollingDeltaY is positive when scrolling
                    // up (away from the user); negate so positive moves a panel's
                    // content up (matching FrameInput.scroll_delta's convention).
                    if !self.cursor_captured {
                        self.keys.scroll_delta -= event.scrollingDeltaY() as f32;
                    }
                    ns_app.sendEvent(&event);
                }
                NSEventType::ApplicationDefined => {
                    self.window_closed = true;
                }
                _ => {
                    ns_app.sendEvent(&event);
                }
            }
        }
    }

    // Returns true when the event's click position is inside the window's
    // content area (below the title bar). Title-bar clicks (traffic lights,
    // drag area) return false so they don't trigger cursor recapture.
    fn in_content_area(&self, event: &objc2_app_kit::NSEvent) -> bool {
        let Some(window) = self.host_window() else {
            return false;
        };
        let content_h = window.contentRectForFrameRect(window.frame()).size.height;
        let loc = event.locationInWindow();
        loc.y >= 0.0 && loc.y < content_h
    }

    // Replace the runtime movement key map. `handle_key` decodes events through
    // it, so a settings-menu rebind takes effect on the next key event.
    pub fn set_keymap(&mut self, keymap: &KeyMap) {
        self.keymap = *keymap;
    }

    // Apply a key transition to whichever gameplay actions are bound to `key`.
    // `down` is the held state (movement / sprint follow it); `fire_pulse` fires
    // the one-shot actions (jump / interact). For a keyboard event the press
    // edge is the KeyDown, so both come from `pressed`; for the Shift modifier
    // the pulse fires only on the rising edge (FlagsChanged can re-fire while
    // Shift stays held if another modifier changes).
    fn apply_binding(&mut self, key: Key, down: bool, fire_pulse: bool) {
        let km = self.keymap;
        if km.forward == key {
            self.keys.forward = down;
        }
        if km.backward == key {
            self.keys.backward = down;
        }
        if km.left == key {
            self.keys.left = down;
        }
        if km.right == key {
            self.keys.right = down;
        }
        if km.sprint == key {
            self.keys.sprint = down;
        }
        if fire_pulse {
            if km.jump == key {
                self.keys.jump_pulse = true;
            }
            if km.interact == key {
                self.keys.interact_pulse = true;
            }
        }
    }

    // Update the persistent key state from a key event. Escape and F1 are fixed
    // (not rebindable); every other key is decoded to a canonical `Key` and
    // routed through the runtime key map. Sprint's default (Shift) is a pure
    // modifier and is handled in the FlagsChanged arm, not here.
    fn handle_key(&mut self, event: &objc2_app_kit::NSEvent, pressed: bool) {
        let kc = event.keyCode();
        // Fixed keys.
        match kc {
            53 if pressed => {
                // Escape. In menu mode (a MainMenu over a captured camera) it
                // always pulses so UiInputSystem can toggle the menu and
                // GraphicsSystem drives capture from there. Otherwise: a
                // captured-cursor world releases the cursor (the safe exit), and
                // a free-cursor world pulses for UiInputSystem.
                if self.menu_mode || !self.cursor_captured {
                    self.keys.escape_pulse = true;
                } else {
                    self.release_cursor();
                }
            }
            122 if pressed => self.keys.hud_toggle_pulse = true, // F1: stat HUD.
            _ => {}
        }
        // Rebindable keys, decoded through the runtime key map.
        if let Some(key) = key_from_mac(kc) {
            if pressed {
                self.keys.captured_key = Some(key);
            }
            self.apply_binding(key, pressed, pressed);
        }
    }
}

// Map a macOS virtual key code to a canonical `Key`, or `None` for a key the
// engine does not bind (modifiers other than Shift, function keys, Escape, etc.).
// The codes are hardware-independent (the same on every Mac keyboard). Shift is
// deliberately absent: it arrives via FlagsChanged, not a key code.
fn key_from_mac(kc: u16) -> Option<Key> {
    Some(match kc {
        0 => Key::A,
        11 => Key::B,
        8 => Key::C,
        2 => Key::D,
        14 => Key::E,
        3 => Key::F,
        5 => Key::G,
        4 => Key::H,
        34 => Key::I,
        38 => Key::J,
        40 => Key::K,
        37 => Key::L,
        46 => Key::M,
        45 => Key::N,
        31 => Key::O,
        35 => Key::P,
        12 => Key::Q,
        15 => Key::R,
        1 => Key::S,
        17 => Key::T,
        32 => Key::U,
        9 => Key::V,
        13 => Key::W,
        7 => Key::X,
        16 => Key::Y,
        6 => Key::Z,
        29 => Key::Num0,
        18 => Key::Num1,
        19 => Key::Num2,
        20 => Key::Num3,
        21 => Key::Num4,
        23 => Key::Num5,
        22 => Key::Num6,
        26 => Key::Num7,
        28 => Key::Num8,
        25 => Key::Num9,
        49 => Key::Space,
        48 => Key::Tab,
        36 => Key::Enter,
        123 => Key::Left,
        124 => Key::Right,
        125 => Key::Down,
        126 => Key::Up,
        27 => Key::Minus,
        24 => Key::Equals,
        33 => Key::LeftBracket,
        30 => Key::RightBracket,
        42 => Key::Backslash,
        41 => Key::Semicolon,
        39 => Key::Quote,
        43 => Key::Comma,
        47 => Key::Period,
        44 => Key::Slash,
        50 => Key::Backtick,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_from_mac_covers_the_defaults() {
        // The default bindings must decode, so a fresh world keeps moving.
        assert_eq!(key_from_mac(13), Some(Key::W));
        assert_eq!(key_from_mac(0), Some(Key::A));
        assert_eq!(key_from_mac(1), Some(Key::S));
        assert_eq!(key_from_mac(2), Some(Key::D));
        assert_eq!(key_from_mac(49), Some(Key::Space));
        assert_eq!(key_from_mac(14), Some(Key::E));
        // Escape / F1 stay fixed (no canonical mapping).
        assert_eq!(key_from_mac(53), None);
        assert_eq!(key_from_mac(122), None);
    }
}
