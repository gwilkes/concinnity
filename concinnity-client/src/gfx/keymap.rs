// src/gfx/keymap.rs
//
// The runtime, rebindable key map for the gameplay movement keys. Each backend
// decodes physical keys into the same semantic booleans (forward, jump, ...);
// this map says which canonical Key drives each action, so the settings menu can
// remap them at runtime. The map is canonical (backend-agnostic Key values); a
// backend resolves it to its own native key codes when it is pushed via
// `RenderBackend::set_keymap`.

use crate::assets::Key;
use serde::{Deserialize, Serialize};

// A rebindable gameplay action. The four movement directions, sprint, jump, and
// interact. Pause (Escape) is deliberately not here: it carries cursor-release /
// menu semantics that are fixed per-backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Bindable {
    Forward,
    Backward,
    Left,
    Right,
    Sprint,
    Jump,
    Interact,
}

impl Bindable {
    // Every rebindable action, in Controls-tab row order.
    pub const ALL: [Bindable; 7] = [
        Bindable::Forward,
        Bindable::Backward,
        Bindable::Left,
        Bindable::Right,
        Bindable::Sprint,
        Bindable::Jump,
        Bindable::Interact,
    ];

    // The settings key string used in `setting:<key>:rebind` actions and the
    // engine settings registry.
    pub fn setting_key(self) -> &'static str {
        match self {
            Bindable::Forward => "key_forward",
            Bindable::Backward => "key_backward",
            Bindable::Left => "key_left",
            Bindable::Right => "key_right",
            Bindable::Sprint => "key_sprint",
            Bindable::Jump => "key_jump",
            Bindable::Interact => "key_interact",
        }
    }

    // The action for a settings key string, or `None` if it is not a rebind key.
    pub fn from_setting_key(key: &str) -> Option<Bindable> {
        Bindable::ALL.into_iter().find(|b| b.setting_key() == key)
    }
}

// The canonical action -> key map. Persisted in `ControlsSettings` and pushed to
// the active backend. Each field is `#[serde(default)]` so adding an action in a
// future build never invalidates an existing settings file (a missing field
// falls back to its default rather than failing the whole load).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyMap {
    #[serde(default = "def_forward")]
    pub forward: Key,
    #[serde(default = "def_backward")]
    pub backward: Key,
    #[serde(default = "def_left")]
    pub left: Key,
    #[serde(default = "def_right")]
    pub right: Key,
    #[serde(default = "def_sprint")]
    pub sprint: Key,
    #[serde(default = "def_jump")]
    pub jump: Key,
    #[serde(default = "def_interact")]
    pub interact: Key,
}

impl KeyMap {
    // The default bindings: the keys that were hardcoded before rebinding.
    pub const DEFAULT: KeyMap = KeyMap {
        forward: Key::W,
        backward: Key::S,
        left: Key::A,
        right: Key::D,
        sprint: Key::Shift,
        jump: Key::Space,
        interact: Key::E,
    };

    // The key currently bound to an action.
    pub fn get(self, action: Bindable) -> Key {
        match action {
            Bindable::Forward => self.forward,
            Bindable::Backward => self.backward,
            Bindable::Left => self.left,
            Bindable::Right => self.right,
            Bindable::Sprint => self.sprint,
            Bindable::Jump => self.jump,
            Bindable::Interact => self.interact,
        }
    }

    // Bind an action to a key directly (no conflict handling).
    pub fn set(&mut self, action: Bindable, key: Key) {
        match action {
            Bindable::Forward => self.forward = key,
            Bindable::Backward => self.backward = key,
            Bindable::Left => self.left = key,
            Bindable::Right => self.right = key,
            Bindable::Sprint => self.sprint = key,
            Bindable::Jump => self.jump = key,
            Bindable::Interact => self.interact = key,
        }
    }

    // The action a key is bound to, or `None` if unbound. The map keeps each key
    // bound to at most one action (the invariant `rebind` maintains), so this is
    // the unique holder.
    pub fn action_for_key(self, key: Key) -> Option<Bindable> {
        Bindable::ALL.into_iter().find(|&b| self.get(b) == key)
    }

    // Bind `action` to `new_key`, swapping with whichever action already holds
    // `new_key` so every action stays bound. Rebinding an action to its own key
    // is a no-op.
    pub fn rebind(&mut self, action: Bindable, new_key: Key) {
        let old_key = self.get(action);
        if old_key == new_key {
            return;
        }
        if let Some(other) = self.action_for_key(new_key)
            && other != action
        {
            self.set(other, old_key);
        }
        self.set(action, new_key);
    }
}

impl Default for KeyMap {
    fn default() -> Self {
        Self::DEFAULT
    }
}

fn def_forward() -> Key {
    KeyMap::DEFAULT.forward
}
fn def_backward() -> Key {
    KeyMap::DEFAULT.backward
}
fn def_left() -> Key {
    KeyMap::DEFAULT.left
}
fn def_right() -> Key {
    KeyMap::DEFAULT.right
}
fn def_sprint() -> Key {
    KeyMap::DEFAULT.sprint
}
fn def_jump() -> Key {
    KeyMap::DEFAULT.jump
}
fn def_interact() -> Key {
    KeyMap::DEFAULT.interact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_wasd_shift_space_e() {
        let m = KeyMap::default();
        assert_eq!(m.forward, Key::W);
        assert_eq!(m.backward, Key::S);
        assert_eq!(m.left, Key::A);
        assert_eq!(m.right, Key::D);
        assert_eq!(m.sprint, Key::Shift);
        assert_eq!(m.jump, Key::Space);
        assert_eq!(m.interact, Key::E);
    }

    #[test]
    fn setting_key_round_trips() {
        for b in Bindable::ALL {
            assert_eq!(Bindable::from_setting_key(b.setting_key()), Some(b));
        }
        assert_eq!(Bindable::from_setting_key("vsync"), None);
        assert_eq!(Bindable::from_setting_key("key_nope"), None);
    }

    #[test]
    fn get_set_round_trip() {
        let mut m = KeyMap::default();
        m.set(Bindable::Forward, Key::Up);
        assert_eq!(m.get(Bindable::Forward), Key::Up);
    }

    #[test]
    fn action_for_key_finds_the_holder() {
        let m = KeyMap::default();
        assert_eq!(m.action_for_key(Key::W), Some(Bindable::Forward));
        assert_eq!(m.action_for_key(Key::Space), Some(Bindable::Jump));
        // A key bound to nothing.
        assert_eq!(m.action_for_key(Key::Q), None);
    }

    #[test]
    fn rebind_to_free_key_just_sets_it() {
        let mut m = KeyMap::default();
        m.rebind(Bindable::Forward, Key::Q);
        assert_eq!(m.forward, Key::Q);
        // The others are untouched.
        assert_eq!(m.backward, Key::S);
    }

    #[test]
    fn rebind_to_own_key_is_a_noop() {
        let mut m = KeyMap::default();
        m.rebind(Bindable::Forward, Key::W);
        assert_eq!(m, KeyMap::default());
    }

    #[test]
    fn rebind_to_occupied_key_swaps() {
        // Bind Forward to S, which Backward holds: they swap, so Backward
        // inherits Forward's old key (W) and every action stays bound.
        let mut m = KeyMap::default();
        m.rebind(Bindable::Forward, Key::S);
        assert_eq!(m.forward, Key::S);
        assert_eq!(m.backward, Key::W);
        // No key is bound twice.
        for b in Bindable::ALL {
            assert_eq!(m.action_for_key(m.get(b)), Some(b));
        }
    }

    #[test]
    fn cbor_round_trip_and_missing_field_defaults() {
        // A full map survives a CBOR round trip.
        let m = KeyMap {
            forward: Key::Up,
            ..KeyMap::default()
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&m, &mut bytes).unwrap();
        let back: KeyMap = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, m);

        // A map written without one field (an older build) still loads, the
        // missing field falling back to its default rather than failing.
        #[derive(Serialize)]
        struct Partial {
            forward: Key,
            backward: Key,
            left: Key,
            right: Key,
            sprint: Key,
            jump: Key,
            // `interact` omitted.
        }
        let partial = Partial {
            forward: Key::Up,
            backward: Key::S,
            left: Key::A,
            right: Key::D,
            sprint: Key::Shift,
            jump: Key::Space,
        };
        let mut bytes = Vec::new();
        ciborium::into_writer(&partial, &mut bytes).unwrap();
        let loaded: KeyMap = ciborium::from_reader(&bytes[..]).unwrap();
        assert_eq!(loaded.forward, Key::Up);
        assert_eq!(loaded.interact, Key::E);
    }
}
