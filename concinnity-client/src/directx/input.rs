// Input state for the DirectX backend.
// Win32 keyboard/mouse events are processed in the window message loop inside
// context.rs; this struct is the snapshot consumed by GraphicsSystem each tick.

use crate::assets::Key;
use crate::gfx::keymap::KeyMap;
use windows::Win32::UI::Input::KeyboardAndMouse::*;

// The previously-duplicated InputState collapsed into the shared
// crate::gfx::input::RenderInput; this alias keeps the historical name.
pub use crate::gfx::input::RenderInput as InputState;

// Per-key pressed state tracked across Win32 WM_KEYDOWN / WM_KEYUP messages.
#[derive(Default)]
pub(super) struct KeyState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub sprint: bool,
    // One-shot flags: set on down, cleared after take_input() reads them.
    pub interact_pending: bool,
    pub jump_pending: bool,
    // One-shot: set on F1-down, cleared by `take`. Drives the `StatHud`
    // system's F1 toggle so the in-engine profiler overlay can be flipped
    // at runtime.
    pub hud_toggle_pending: bool,
    // One-shot: set on Escape-down when the cursor is *not* captured.
    // (When the cursor is captured the wnd_proc routes Escape through
    // `do_release_cursor` instead, matching the Metal backend.)
    pub escape_pending: bool,
    // One-shot: the canonical key pressed since the last `take`, for the
    // settings-menu rebind capture. Set on any mapped key-down; reset by `take`.
    pub captured_key: Option<Key>,
    // The runtime movement key map. `on_key_down` / `on_key_up` decode events
    // through it instead of hardcoded keys, so a settings-menu rebind takes
    // effect immediately. Defaults to W/S/A/D/Shift/Space/E. (Windows delivers
    // Shift as an ordinary WM_KEYDOWN, so it is just another key here -- no
    // separate modifier path is needed, unlike macOS.)
    pub keymap: KeyMap,
}

impl KeyState {
    // Replace the runtime movement key map.
    pub(super) fn set_keymap(&mut self, keymap: &KeyMap) {
        self.keymap = *keymap;
    }

    // Apply a key transition to whichever gameplay actions are bound to `key`.
    // `down` is the held state (movement / sprint follow it); `fire_pulse`
    // fires the one-shot actions (jump / interact) on a press.
    fn apply_binding(&mut self, key: Key, down: bool, fire_pulse: bool) {
        let km = self.keymap;
        if km.forward == key {
            self.forward = down;
        }
        if km.backward == key {
            self.backward = down;
        }
        if km.left == key {
            self.left = down;
        }
        if km.right == key {
            self.right = down;
        }
        if km.sprint == key {
            self.sprint = down;
        }
        if fire_pulse {
            if km.jump == key {
                self.jump_pending = true;
            }
            if km.interact == key {
                self.interact_pending = true;
            }
        }
    }

    // Update held/pending flags from a WM_KEYDOWN message. F1 stays fixed (the
    // stat-HUD toggle); every other key routes through the key map.
    pub(super) fn on_key_down(&mut self, vk: VIRTUAL_KEY) {
        if vk == VK_F1 {
            self.hud_toggle_pending = true;
        }
        if let Some(key) = key_from_vk(vk) {
            self.captured_key = Some(key);
            self.apply_binding(key, true, true);
        }
    }

    // Note an Escape press while the cursor is *not* captured. The wnd_proc
    // keeps swallowing Escape into `do_release_cursor` while captured, so
    // this is only called for the "menu / UI" case; mirrors the Metal
    // `escape_pulse` rule.
    pub(super) fn on_escape_uncaptured(&mut self) {
        self.escape_pending = true;
    }

    // Update held flags from a WM_KEYUP message.
    pub(super) fn on_key_up(&mut self, vk: VIRTUAL_KEY) {
        if let Some(key) = key_from_vk(vk) {
            self.apply_binding(key, false, false);
        }
    }

    // Drain into an InputState snapshot, resetting one-shot flags. The mouse
    // fields (deltas, position, click, held-button, scroll) are owned by
    // `WindowState` and passed in; the keyboard one-shots tracked here are reset.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn take(
        &mut self,
        mouse_dx: f32,
        mouse_dy: f32,
        mouse_x: f32,
        mouse_y: f32,
        left_click: bool,
        left_button_down: bool,
        scroll_delta: f32,
    ) -> InputState {
        let s = InputState {
            forward: self.forward,
            backward: self.backward,
            left: self.left,
            right: self.right,
            sprint: self.sprint,
            interact: self.interact_pending,
            jump: self.jump_pending,
            mouse_dx,
            mouse_dy,
            scroll_delta,
            mouse_x,
            mouse_y,
            left_click,
            left_button_down,
            hud_toggle: self.hud_toggle_pending,
            escape: self.escape_pending,
            captured_key: self.captured_key,
        };
        self.interact_pending = false;
        self.jump_pending = false;
        self.hud_toggle_pending = false;
        self.escape_pending = false;
        self.captured_key = None;
        s
    }
}

// Translate a WM_KEYDOWN/WM_KEYUP wParam into a VIRTUAL_KEY.
pub(super) fn vk_from_wparam(wparam: usize) -> VIRTUAL_KEY {
    VIRTUAL_KEY(wparam as u16)
}

// Map a Win32 virtual key to a canonical `Key`, or `None` for a key the engine
// does not bind (function keys, Escape, Ctrl/Alt, etc.). Shift is mapped: unlike
// macOS, Windows delivers it as an ordinary key-down.
fn key_from_vk(vk: VIRTUAL_KEY) -> Option<Key> {
    Some(match vk {
        VK_A => Key::A,
        VK_B => Key::B,
        VK_C => Key::C,
        VK_D => Key::D,
        VK_E => Key::E,
        VK_F => Key::F,
        VK_G => Key::G,
        VK_H => Key::H,
        VK_I => Key::I,
        VK_J => Key::J,
        VK_K => Key::K,
        VK_L => Key::L,
        VK_M => Key::M,
        VK_N => Key::N,
        VK_O => Key::O,
        VK_P => Key::P,
        VK_Q => Key::Q,
        VK_R => Key::R,
        VK_S => Key::S,
        VK_T => Key::T,
        VK_U => Key::U,
        VK_V => Key::V,
        VK_W => Key::W,
        VK_X => Key::X,
        VK_Y => Key::Y,
        VK_Z => Key::Z,
        VK_0 => Key::Num0,
        VK_1 => Key::Num1,
        VK_2 => Key::Num2,
        VK_3 => Key::Num3,
        VK_4 => Key::Num4,
        VK_5 => Key::Num5,
        VK_6 => Key::Num6,
        VK_7 => Key::Num7,
        VK_8 => Key::Num8,
        VK_9 => Key::Num9,
        VK_SPACE => Key::Space,
        VK_TAB => Key::Tab,
        VK_RETURN => Key::Enter,
        VK_SHIFT => Key::Shift,
        VK_LEFT => Key::Left,
        VK_RIGHT => Key::Right,
        VK_UP => Key::Up,
        VK_DOWN => Key::Down,
        VK_OEM_MINUS => Key::Minus,
        VK_OEM_PLUS => Key::Equals,
        VK_OEM_4 => Key::LeftBracket,
        VK_OEM_6 => Key::RightBracket,
        VK_OEM_5 => Key::Backslash,
        VK_OEM_1 => Key::Semicolon,
        VK_OEM_7 => Key::Quote,
        VK_OEM_COMMA => Key::Comma,
        VK_OEM_PERIOD => Key::Period,
        VK_OEM_2 => Key::Slash,
        VK_OEM_3 => Key::Backtick,
        _ => return None,
    })
}
