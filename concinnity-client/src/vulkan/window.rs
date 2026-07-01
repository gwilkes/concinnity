// src/vulkan/window.rs
//
// GLFW window and input for the Vulkan backend.
//
// Input design mirrors metal.rs: events accumulate into InputState between
// poll() calls; GraphicsSystem drains the state each step via take_input()
// and deposits it as a FrameInput component for Camera3DSystem to consume.
//
// Cursor capture is enabled by GraphicsSystem::init() when a Camera3D
// component is present. GLFW's CursorDisabled mode delivers raw relative
// deltas directly via CursorPos events, so no manual warping is needed.

use crate::assets::{Key, WindowMode};
use crate::gfx::keymap::KeyMap;

// Win32 cursor clipping for the fullscreen menu confine. GLFW has no
// clip-to-window mode for a FREE cursor, so on Windows we reach through the
// window's HWND and use `ClipCursor` directly -- a hard OS boundary the cursor
// physically cannot cross -- exactly like the DirectX backend, instead of the
// per-poll warp-back (which visibly overshoots + fights on a multi-monitor edge).
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::{HWND, POINT, RECT};
#[cfg(target_os = "windows")]
use windows::Win32::Graphics::Gdi::ClientToScreen;
#[cfg(target_os = "windows")]
use windows::Win32::UI::WindowsAndMessaging::{ClipCursor, GetClientRect, GetForegroundWindow};

// Accumulated input state between poll() calls.
// Snapshotted by GraphicsSystem each step via take_input(); held-key state
// persists across the call, momentary/accumulated fields reset.
#[derive(Default, Clone, Copy)]
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub sprint: bool,
    // True for exactly one frame per E key press.
    pub interact: bool,
    // True for exactly one frame per Space key press.
    pub jump: bool,
    // Accumulated mouse delta since last take_input() call.
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    // Absolute cursor position in window pixels (origin top-left).
    // Only meaningful when the cursor is not captured.
    pub mouse_x: f32,
    pub mouse_y: f32,
    // True for exactly one frame when the left mouse button is pressed
    // while the cursor is not captured.
    pub left_click: bool,
    // True while the left mouse button is held with the cursor free (a UI
    // drag, e.g. a settings Slider handle). Set on Button1 press, cleared on
    // release; unlike `left_click` it persists across take_input(). Mirrors the
    // Metal / DirectX `left_button_down` signal.
    pub left_button_down: bool,
    // Accumulated vertical scroll-wheel delta since the last take_input(), in
    // scroll_delta units (GLFW Scroll notches converted via
    // `wheel_notches_to_scroll_delta`). Reset each take_input().
    pub scroll_delta: f32,
    // True for exactly one frame when F1 is pressed. Drives the StatHud
    // system's HUD toggle so the in-engine profiler overlay can be
    // flipped at runtime.
    pub hud_toggle: bool,
    // True for exactly one frame when Escape is pressed while the cursor
    // is not captured. When captured, Escape releases the cursor instead
    // (and this flag stays false), matching Metal / DirectX.
    pub escape: bool,
    // The canonical key pressed since the last take_input(), for the
    // settings-menu rebind capture, or `None`. A one-shot, surfaced regardless
    // of capture / menu state. Mirrors Metal / DirectX.
    pub captured_key: Option<Key>,
}

// Owns the GLFW library handle, the window, and the event receiver.
//
// Created once by GraphicsSystem during init(); polled every step().
// All GLFW calls must happen on the thread that created this struct -- the
// world loop guarantees single-threaded system execution.
pub struct GlfwWindow {
    pub glfw: glfw::Glfw,
    pub window: glfw::PWindow,
    events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    // last cursor position, used to compute deltas when not in raw mode
    last_cursor: Option<(f64, f64)>,
    input: InputState,
    cursor_captured: bool,
    // Whether the OS cursor is hidden for an in-engine UI cursor (e.g. a
    // MainMenu) while not captured. GLFW's cursor mode is a single enum, so the
    // effective mode is computed from both this and `cursor_captured` (see
    // `apply_cursor_mode`); tracked so `set_ui_cursor_hidden` only re-applies on
    // a transition.
    ui_cursor_hidden: bool,
    // A togglable menu coexists with a captured camera (a MainMenu over a
    // Camera3D world). When set, Escape routes to the ECS instead of releasing
    // the cursor inline; GraphicsSystem drives capture from the active menu.
    menu_mode: bool,
    // The current window mode. Tracked so the per-frame cursor confinement knows
    // whether to confine (Fullscreen) or hide the in-engine arrow on leave
    // (Windowed / Borderless). Seeded from the creation mode and kept in sync by
    // `set_window_mode`.
    window_mode: WindowMode,
    // Whether the real cursor has left the window content area while the cursor
    // is free (windowed / borderless). Recomputed each poll by
    // `update_ui_cursor_confinement`; the renderer hides the in-engine cursor
    // when set. False while captured or in fullscreen (which confines instead).
    cursor_outside_window: bool,
    // Whether the per-poll confinement currently holds a Win32 `ClipCursor` clip
    // for a fullscreen menu. Tracked so the clip is released exactly once when the
    // confining condition ends (mode change or capture engaging, which hands the
    // clip to GLFW's Disabled mode). Windows-only: other platforms fall back to
    // the per-poll cursor clamp, which holds no OS state.
    #[cfg(target_os = "windows")]
    menu_clip_active: bool,
    // The runtime movement key map. The key event arm decodes through it instead
    // of hardcoded keys, so a settings-menu rebind takes effect immediately.
    // Defaults to W/S/A/D/Shift/Space/E. (GLFW delivers Shift as Left/Right Shift
    // key events, so it is just another key here -- no separate modifier path.)
    keymap: KeyMap,
}

// GlfwWindow is only ever used on the thread that created it.
unsafe impl Send for GlfwWindow {}

// The window's client area in screen coordinates (top-left origin, y down), or
// None if the rect query fails. The `ClipCursor` bounds for the fullscreen menu
// confine (mirrors the DirectX `client_screen_rect`).
#[cfg(target_os = "windows")]
fn client_screen_rect(hwnd: HWND) -> Option<RECT> {
    let mut rect = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut rect) }.is_err() {
        return None;
    }
    let mut tl = POINT {
        x: rect.left,
        y: rect.top,
    };
    let mut br = POINT {
        x: rect.right,
        y: rect.bottom,
    };
    unsafe {
        if ClientToScreen(hwnd, &mut tl).as_bool() && ClientToScreen(hwnd, &mut br).as_bool() {
            Some(RECT {
                left: tl.x,
                top: tl.y,
                right: br.x,
                bottom: br.y,
            })
        } else {
            None
        }
    }
}

// Resolve the GLFW cursor mode from the two independent intents. A captured
// camera locks + hides the cursor (Disabled, raw deltas) and takes precedence;
// otherwise an in-engine UI cursor hides it but keeps it freely positioned
// (Hidden); with neither, the OS cursor is shown (Normal).
fn resolve_cursor_mode(captured: bool, ui_cursor_hidden: bool) -> glfw::CursorMode {
    if captured {
        glfw::CursorMode::Disabled
    } else if ui_cursor_hidden {
        glfw::CursorMode::Hidden
    } else {
        glfw::CursorMode::Normal
    }
}

// Map a cursor position from GLFW window (screen) coordinates into framebuffer
// pixels. GLFW reports the cursor in window coordinates, but the overlay UI is
// hit-tested in framebuffer pixels (`VkContext::logical_size` returns the
// swapchain extent). `win`/`fb` are the window and framebuffer sizes: equal on
// Windows and unscaled X11, so the scale is 1.0 and this is a no-op; on a scaled
// surface (hi-DPI Wayland) the framebuffer is larger than the window by the
// content scale, so the cursor must be multiplied up to land on the right
// overlay region. A zero window dimension (minimised / mid-resize) falls back to
// a 1.0 scale rather than dividing by zero.
fn scale_cursor_to_framebuffer(x: f32, y: f32, win: (i32, i32), fb: (i32, i32)) -> (f32, f32) {
    let sx = if win.0 > 0 {
        fb.0 as f32 / win.0 as f32
    } else {
        1.0
    };
    let sy = if win.1 > 0 {
        fb.1 as f32 / win.1 as f32
    } else {
        1.0
    };
    (x * sx, y * sy)
}

// Apply a key transition to whichever gameplay actions are bound to `key`.
// `pressed` is the held state (movement / sprint follow it) and, on a press,
// fires the one-shot actions (jump / interact). Mirrors the Metal / DirectX
// `apply_binding`.
fn apply_binding(input: &mut InputState, km: KeyMap, key: Key, pressed: bool) {
    if km.forward == key {
        input.forward = pressed;
    }
    if km.backward == key {
        input.backward = pressed;
    }
    if km.left == key {
        input.left = pressed;
    }
    if km.right == key {
        input.right = pressed;
    }
    if km.sprint == key {
        input.sprint = pressed;
    }
    if pressed {
        if km.jump == key {
            input.jump = true;
        }
        if km.interact == key {
            input.interact = true;
        }
    }
}

// Map a GLFW key to a canonical `Key`, or `None` for a key the engine does not
// bind (function keys, Escape, Ctrl/Alt, keypad, etc.). Left/Right Shift both map
// to `Key::Shift`: GLFW delivers them as ordinary key events.
fn key_from_glfw(key: glfw::Key) -> Option<Key> {
    use glfw::Key as G;
    Some(match key {
        G::A => Key::A,
        G::B => Key::B,
        G::C => Key::C,
        G::D => Key::D,
        G::E => Key::E,
        G::F => Key::F,
        G::G => Key::G,
        G::H => Key::H,
        G::I => Key::I,
        G::J => Key::J,
        G::K => Key::K,
        G::L => Key::L,
        G::M => Key::M,
        G::N => Key::N,
        G::O => Key::O,
        G::P => Key::P,
        G::Q => Key::Q,
        G::R => Key::R,
        G::S => Key::S,
        G::T => Key::T,
        G::U => Key::U,
        G::V => Key::V,
        G::W => Key::W,
        G::X => Key::X,
        G::Y => Key::Y,
        G::Z => Key::Z,
        G::Num0 => Key::Num0,
        G::Num1 => Key::Num1,
        G::Num2 => Key::Num2,
        G::Num3 => Key::Num3,
        G::Num4 => Key::Num4,
        G::Num5 => Key::Num5,
        G::Num6 => Key::Num6,
        G::Num7 => Key::Num7,
        G::Num8 => Key::Num8,
        G::Num9 => Key::Num9,
        G::Space => Key::Space,
        G::Tab => Key::Tab,
        G::Enter => Key::Enter,
        G::LeftShift | G::RightShift => Key::Shift,
        G::Up => Key::Up,
        G::Down => Key::Down,
        G::Left => Key::Left,
        G::Right => Key::Right,
        G::Minus => Key::Minus,
        G::Equal => Key::Equals,
        G::LeftBracket => Key::LeftBracket,
        G::RightBracket => Key::RightBracket,
        G::Backslash => Key::Backslash,
        G::Semicolon => Key::Semicolon,
        G::Apostrophe => Key::Quote,
        G::Comma => Key::Comma,
        G::Period => Key::Period,
        G::Slash => Key::Slash,
        G::GraveAccent => Key::Backtick,
        _ => return None,
    })
}

impl GlfwWindow {
    // create a new glfw window with no opengl context (vulkan surface mode)
    pub fn new(
        title: &str,
        width: u32,
        height: u32,
        mode: &WindowMode,
        resizable: bool,
    ) -> Result<Self, String> {
        let mut glfw = glfw::init(glfw::fail_on_errors).map_err(|e| format!("glfw init: {e}"))?;

        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(resizable));

        let (mut window, events) = match mode {
            WindowMode::Windowed => glfw
                .create_window(width, height, title, glfw::WindowMode::Windowed)
                .ok_or_else(|| "Failed to create GLFW window (windowed)".to_string())?,

            WindowMode::Fullscreen => glfw.with_primary_monitor(|glfw, monitor| {
                let monitor = monitor.ok_or("No primary monitor")?;
                glfw.create_window(width, height, title, glfw::WindowMode::FullScreen(monitor))
                    .ok_or_else(|| "Failed to create GLFW window (fullscreen)".to_string())
            })?,

            WindowMode::Borderless => glfw.with_primary_monitor(|glfw, monitor| {
                let monitor = monitor.ok_or("No primary monitor")?;
                let vid_mode = monitor
                    .get_video_mode()
                    .ok_or("Could not query primary monitor video mode")?;
                glfw.window_hint(glfw::WindowHint::Decorated(false));
                glfw.create_window(
                    vid_mode.width,
                    vid_mode.height,
                    title,
                    glfw::WindowMode::Windowed, // borderless = undecorated windowed
                )
                .ok_or_else(|| "Failed to create GLFW window (borderless)".to_string())
            })?,
        };

        window.set_close_polling(true);
        window.set_key_polling(true);
        window.set_cursor_pos_polling(true);
        // Mouse-button events drive UI clicks (e.g. a MainMenu HitRegion).
        // Without this GLFW never queues Button events, so the `MouseButton`
        // arm in `poll()` never runs and clicks are silently dropped.
        window.set_mouse_button_polling(true);
        // Scroll events drive scrollable UI (e.g. the settings panel). Without
        // this GLFW never queues Scroll events, so the `Scroll` arm in `poll()`
        // never runs and the wheel is silently dropped.
        window.set_scroll_polling(true);
        window.set_framebuffer_size_polling(true);

        Ok(Self {
            glfw,
            window,
            events,
            last_cursor: None,
            input: InputState::default(),
            cursor_captured: false,
            ui_cursor_hidden: false,
            menu_mode: false,
            window_mode: *mode,
            cursor_outside_window: false,
            #[cfg(target_os = "windows")]
            menu_clip_active: false,
            keymap: KeyMap::default(),
        })
    }

    // Push the cursor mode resolved from the two independent intents onto the
    // window. Centralised because GLFW exposes one mode enum where Metal /
    // DirectX keep two independent ref-counts.
    fn apply_cursor_mode(&mut self) {
        self.window.set_cursor_mode(resolve_cursor_mode(
            self.cursor_captured,
            self.ui_cursor_hidden,
        ));
    }

    // Hide the cursor and begin delivering relative mouse deltas via CursorPos
    // events. Should be called once after the window is shown, when a
    // Camera3D component is present.
    pub fn capture_cursor(&mut self) {
        self.cursor_captured = true;
        self.apply_cursor_mode();
        // enable raw mouse motion if the platform supports it -- bypasses
        // pointer acceleration for more direct 1:1 feel
        if self.glfw.supports_raw_motion() {
            self.window.set_raw_mouse_motion(true);
        }
        self.last_cursor = None;
    }

    // Show the cursor and stop accumulating relative deltas; symmetric with
    // `capture_cursor`. Driven by `set_camera_capture` in menu mode.
    pub fn release_cursor(&mut self) {
        if !self.cursor_captured {
            return;
        }
        self.cursor_captured = false;
        self.apply_cursor_mode();
    }

    // Hide or show the OS cursor for an in-engine UI cursor (e.g. a MainMenu),
    // without engaging camera capture. Edge-triggered: re-applies the combined
    // cursor mode only on a transition.
    pub fn set_ui_cursor_hidden(&mut self, hidden: bool) {
        if hidden == self.ui_cursor_hidden {
            return;
        }
        self.ui_cursor_hidden = hidden;
        self.apply_cursor_mode();
    }

    // A togglable menu coexists with a captured camera; see
    // `RenderBackend::set_menu_mode`. The poll loop reads this to route Escape
    // to the ECS instead of releasing the cursor inline.
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

    // Whether the real cursor has left the window so the renderer should stop
    // drawing the in-engine UI cursor (windowed / borderless). Recomputed each
    // `poll`; false while captured or in fullscreen (which confines instead).
    pub fn cursor_outside_window(&self) -> bool {
        self.cursor_outside_window
    }

    // Per-poll bookkeeping for an in-engine UI cursor (a menu), mirroring the
    // Metal `update_ui_cursor_confinement`: report whether the real cursor has
    // left the window content area so the renderer can stop drawing the in-engine
    // cursor in windowed / borderless modes, and confine the cursor to the window
    // while in fullscreen so it cannot stray onto another display. A no-op while
    // the cursor is captured (a gameplay camera owns the pointer, in GLFW's
    // Disabled mode).
    fn update_ui_cursor_confinement(&mut self) {
        if self.cursor_captured {
            // GLFW's Disabled mode owns the OS cursor (and, on Windows, its own
            // clip); relinquish our menu-clip flag without releasing so we never
            // fight it (releasing would undo the capture clip).
            #[cfg(target_os = "windows")]
            {
                self.menu_clip_active = false;
            }
            self.cursor_outside_window = false;
            return;
        }
        if matches!(self.window_mode, WindowMode::Fullscreen) {
            // Confine the cursor to the fullscreen window so a menu pointer cannot
            // wander onto another monitor.
            self.confine_fullscreen();
            self.cursor_outside_window = false;
            return;
        }
        // Windowed / borderless: the in-engine cursor shows only while the real
        // cursor is over the content area. GLFW_HOVERED is the OS-tracked signal
        // for that, so no manual bounds test is needed. Drop any fullscreen menu
        // clip first (the mode may have just changed out of fullscreen).
        #[cfg(target_os = "windows")]
        self.release_menu_clip();
        self.cursor_outside_window = !self.window.is_hovered();
    }

    // Hard-confine the cursor to the fullscreen window. On Windows this is a Win32
    // `ClipCursor` on the window's HWND -- a continuous OS boundary the cursor
    // cannot cross, gated on foreground so a backgrounded window never steals the
    // cursor from the foreground app (mirrors the DirectX backend). The clip is
    // released once via `release_menu_clip` when the condition ends; capture's
    // GLFW-Disabled clip is left alone (the captured branch returns early).
    #[cfg(target_os = "windows")]
    fn confine_fullscreen(&mut self) {
        let hwnd = HWND(self.window.get_win32_window());
        if unsafe { GetForegroundWindow() } == hwnd {
            if let Some(rect) = client_screen_rect(hwnd) {
                let _ = unsafe { ClipCursor(Some(&rect)) };
                self.menu_clip_active = true;
            }
        } else {
            // Windows already dropped our clip on deactivation; clear the flag
            // without issuing ClipCursor(None), which from the background could
            // stomp the foreground app's own clip.
            self.menu_clip_active = false;
        }
    }

    // Non-Windows fallback: no clip-to-window mode exists for a free cursor, so
    // warp the cursor back to the content bounds each poll (best-effort; produces
    // a slight snap-back at a multi-monitor edge). Gated on input focus: GLFW's
    // set_cursor_pos no-ops without focus anyway. A single-display fullscreen is
    // already OS-confined, so the clamp never fires there.
    #[cfg(not(target_os = "windows"))]
    fn confine_fullscreen(&mut self) {
        let (w, h) = self.window.get_size();
        if self.window.is_focused() && w > 0 && h > 0 {
            let (cx, cy) = self.window.get_cursor_pos();
            let clamped_x = cx.clamp(0.0, (w - 1) as f64);
            let clamped_y = cy.clamp(0.0, (h - 1) as f64);
            if clamped_x != cx || clamped_y != cy {
                self.window.set_cursor_pos(clamped_x, clamped_y);
            }
        }
    }

    // Release the fullscreen-menu `ClipCursor` clip if we hold one. Edge-triggered
    // so it never frees a clip we do not own (capture's GLFW-Disabled clip).
    #[cfg(target_os = "windows")]
    fn release_menu_clip(&mut self) {
        if self.menu_clip_active {
            self.menu_clip_active = false;
            let _ = unsafe { ClipCursor(None) };
        }
    }

    // NOTE: the window mode/size methods below mirror the Metal implementation
    // for cross-backend parity but were written on macOS and have NOT been built
    // or run on Linux/Windows. Verify the glfw crate API (`set_monitor`,
    // `set_decorated`, `set_size`) and surface survival across a monitor change
    // (Wayland may invalidate the Vulkan surface -- the present path already
    // rebuilds the swapchain on ERROR_OUT_OF_DATE_KHR, which should cover it).
    //
    // Switch windowed / borderless / fullscreen. The framebuffer-size change
    // makes the next present return OUT_OF_DATE, which rebuilds the swapchain.
    pub fn set_window_mode(&mut self, mode: WindowMode) {
        // Record the mode so the per-frame cursor confinement can tell fullscreen
        // (confine) from windowed / borderless (hide the arrow on leave).
        self.window_mode = mode;
        // Disjoint field borrows so the with_primary_monitor closure can drive
        // the window while glfw is borrowed for the monitor lookup.
        let glfw = &mut self.glfw;
        let window = &mut self.window;
        match mode {
            WindowMode::Windowed => {
                window.set_decorated(true);
                let (x, y) = window.get_pos();
                let (w, h) = window.get_size();
                window.set_monitor(
                    glfw::WindowMode::Windowed,
                    x.max(0),
                    y.max(0),
                    w.max(640) as u32,
                    h.max(480) as u32,
                    None,
                );
            }
            WindowMode::Borderless => {
                // Undecorated windowed at the monitor's video-mode size.
                window.set_decorated(false);
                glfw.with_primary_monitor(|_, monitor| {
                    if let Some(m) = monitor
                        && let Some(vid) = m.get_video_mode()
                    {
                        window.set_monitor(
                            glfw::WindowMode::Windowed,
                            0,
                            0,
                            vid.width,
                            vid.height,
                            None,
                        );
                    }
                });
            }
            WindowMode::Fullscreen => {
                window.set_decorated(true);
                glfw.with_primary_monitor(|_, monitor| {
                    if let Some(m) = monitor
                        && let Some(vid) = m.get_video_mode()
                    {
                        window.set_monitor(
                            glfw::WindowMode::FullScreen(m),
                            0,
                            0,
                            vid.width,
                            vid.height,
                            Some(vid.refresh_rate),
                        );
                    }
                });
            }
        }
    }

    // Resize the window (windowed mode only; GraphicsSystem gates this). The
    // framebuffer-size change triggers a swapchain rebuild via OUT_OF_DATE.
    pub fn set_window_size(&mut self, width: u32, height: u32) {
        self.window.set_size(width as i32, height as i32);
    }

    // Replace the runtime movement key map. `poll` decodes key events through
    // it, so a settings-menu rebind takes effect immediately.
    pub fn set_keymap(&mut self, keymap: &KeyMap) {
        self.keymap = *keymap;
    }

    // Drain all pending GLFW events, update input state, and return true if
    // the window should close. Key state is tracked as a running bitmask;
    // cursor deltas are accumulated so no delta is lost between poll calls.
    pub fn poll(&mut self) -> bool {
        self.glfw.poll_events();
        let mut should_close = self.window.should_close();

        for (_, event) in glfw::flush_messages(&self.events) {
            match event {
                glfw::WindowEvent::Close => {
                    should_close = true;
                }
                glfw::WindowEvent::Key(glfw::Key::Escape, _, glfw::Action::Press, _) => {
                    // In menu mode (a MainMenu over a captured camera) Escape
                    // always pulses so UiInputSystem can toggle the menu and
                    // GraphicsSystem drives capture from there. Otherwise a
                    // captured-cursor world releases the cursor (matching
                    // Metal / DirectX) and a free-cursor world pulses for
                    // UiInputSystem. The release is a direct field write rather
                    // than `release_cursor()` because `self.events` is borrowed
                    // by this loop; this branch is reached only with no UI
                    // cursor, so plain Normal is the correct combined mode.
                    if self.menu_mode || !self.cursor_captured {
                        self.input.escape = true;
                    } else {
                        self.window.set_cursor_mode(glfw::CursorMode::Normal);
                        self.cursor_captured = false;
                    }
                }
                glfw::WindowEvent::Key(glfw::Key::F1, _, glfw::Action::Press, _) => {
                    // F1 toggles the in-engine profiler HUD. Pulse-only
                    // (cleared by `take_input`).
                    self.input.hud_toggle = true;
                }
                glfw::WindowEvent::Key(key, _, action, _) => {
                    // Decode through the runtime key map (GLFW delivers Shift as
                    // Left/Right Shift key events, so it is handled like any other
                    // key -- no separate modifier path, matching DirectX).
                    if action != glfw::Action::Repeat
                        && let Some(canon) = key_from_glfw(key)
                    {
                        let pressed = action == glfw::Action::Press;
                        if pressed {
                            self.input.captured_key = Some(canon);
                        }
                        apply_binding(&mut self.input, self.keymap, canon, pressed);
                    }
                }
                glfw::WindowEvent::CursorPos(x, y) => {
                    if self.cursor_captured {
                        if let Some((lx, ly)) = self.last_cursor {
                            self.input.mouse_dx += (x - lx) as f32;
                            self.input.mouse_dy += (y - ly) as f32;
                        }
                        self.last_cursor = Some((x, y));
                    } else {
                        // GLFW CursorPos has origin top-left with Y increasing
                        // downward -- matches TextLabel coords directly. It
                        // arrives in window (screen) coordinates; map it into
                        // framebuffer pixels so it lines up with the overlay
                        // hit-testing space (logical_size = swapchain extent).
                        // No-op where window coords equal framebuffer pixels
                        // (Windows / unscaled X11); scales up on hi-DPI Wayland.
                        let (mx, my) = scale_cursor_to_framebuffer(
                            x as f32,
                            y as f32,
                            self.window.get_size(),
                            self.window.get_framebuffer_size(),
                        );
                        self.input.mouse_x = mx;
                        self.input.mouse_y = my;
                    }
                }
                glfw::WindowEvent::MouseButton(
                    glfw::MouseButton::Button1,
                    glfw::Action::Press,
                    _,
                ) => {
                    if !self.cursor_captured {
                        self.input.left_click = true;
                        // Begin a held-button (UI drag) gesture.
                        self.input.left_button_down = true;
                    }
                }
                glfw::WindowEvent::MouseButton(
                    glfw::MouseButton::Button1,
                    glfw::Action::Release,
                    _,
                ) => {
                    // End any held-button (drag) gesture. Always cleared, even if
                    // the press began while captured, so the flag can never stick
                    // across a capture transition. Mirrors metal / directx.
                    self.input.left_button_down = false;
                }
                glfw::WindowEvent::Scroll(_, yoffset) if !self.cursor_captured => {
                    // Accumulate the wheel delta for scrollable UI while the
                    // cursor is free. GLFW yoffset is in notches, positive when
                    // rotated away from the user; convert to a scroll_delta
                    // increment (matching the Metal sign convention).
                    self.input.scroll_delta +=
                        crate::gfx::input::wheel_notches_to_scroll_delta(yoffset as f32);
                }
                _ => {}
            }
        }

        // After draining this poll's events, refresh the in-engine cursor's
        // window-exit / fullscreen-confinement state (mirrors the tail of Metal's
        // `pump_ns_events`). GraphicsSystem reads `cursor_outside_window` later
        // this same frame.
        self.update_ui_cursor_confinement();

        should_close
    }

    // Return a snapshot of the current input state. Held-key flags
    // (forward/backward/left/right/sprint) and the absolute cursor position
    // persist -- they only change on a GLFW Key/CursorPos event, and GLFW
    // sends no events for a key that is simply held down (the first repeat
    // event lags the press by ~0.5 s). Resetting them here, as a blanket
    // `mem::take` once did, dropped held movement between events and made
    // the camera stutter for that gap. Only the momentary one-shot inputs
    // (interact/jump/left_click) and the per-call accumulated mouse delta
    // are cleared.
    pub fn take_input(&mut self) -> InputState {
        let snapshot = self.input;
        self.input.interact = false;
        self.input.jump = false;
        self.input.left_click = false;
        self.input.hud_toggle = false;
        self.input.escape = false;
        self.input.captured_key = None;
        self.input.mouse_dx = 0.0;
        self.input.mouse_dy = 0.0;
        // Accumulated like the mouse delta; the held-button flag persists until
        // its release event.
        self.input.scroll_delta = 0.0;
        snapshot
    }

    // create a vulkan surface for this window
    // returns the raw `VkSurfaceKHR` handle as a `u64`
    pub fn create_surface(&mut self, instance_handle: usize) -> Result<usize, String> {
        let mut raw_surface: usize = 0;
        let result = unsafe {
            self.window.create_window_surface(
                instance_handle as *mut _,
                std::ptr::null(),
                &mut raw_surface as *mut usize as *mut *mut _,
            )
        };
        if result != 0 {
            Err(format!(
                "glfwCreateWindowSurface failed: VkResult({result})"
            ))
        } else {
            Ok(raw_surface)
        }
    }

    // vulkan instance extensions required for surface creation on this platform
    pub fn required_instance_extensions(&self) -> Vec<String> {
        self.glfw
            .get_required_instance_extensions()
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_mode_prefers_capture_then_ui_then_normal() {
        // Capture wins regardless of the UI-cursor intent (camera control needs
        // the locked, raw-delta Disabled mode).
        assert_eq!(resolve_cursor_mode(true, false), glfw::CursorMode::Disabled);
        assert_eq!(resolve_cursor_mode(true, true), glfw::CursorMode::Disabled);
        // Not captured but a UI cursor is shown: hide the OS cursor while
        // keeping it freely positioned.
        assert_eq!(resolve_cursor_mode(false, true), glfw::CursorMode::Hidden);
        // Neither: the OS cursor is visible.
        assert_eq!(resolve_cursor_mode(false, false), glfw::CursorMode::Normal);
    }

    #[test]
    fn cursor_scale_is_noop_when_window_equals_framebuffer() {
        // Windows / unscaled X11: window coords already equal framebuffer pixels.
        let p = scale_cursor_to_framebuffer(640.0, 360.0, (1280, 720), (1280, 720));
        assert_eq!(p, (640.0, 360.0));
    }

    #[test]
    fn cursor_scales_up_on_hidpi_surface() {
        // hi-DPI Wayland at 2x: a 1280x720 window backs a 2560x1440 framebuffer,
        // so a cursor at the window centre maps to the framebuffer centre.
        let p = scale_cursor_to_framebuffer(640.0, 360.0, (1280, 720), (2560, 1440));
        assert_eq!(p, (1280.0, 720.0));
    }

    #[test]
    fn cursor_scale_guards_zero_window_size() {
        // A zero window dimension (minimised / mid-resize) must not divide by
        // zero; fall back to a 1.0 scale on that axis.
        let p = scale_cursor_to_framebuffer(10.0, 20.0, (0, 0), (1280, 720));
        assert_eq!(p, (10.0, 20.0));
    }
}
