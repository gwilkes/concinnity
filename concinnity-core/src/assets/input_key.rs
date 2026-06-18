// src/assets/input_key.rs

/// A canonical, backend-agnostic keyboard key.
///
/// Each rendering backend maps its native key codes (macOS NSEvent key codes,
/// Windows virtual keys, GLFW keys) to and from this enum, so a key binding can
/// be stored and shown the same way everywhere. Unit variants serialize to
/// their name, so a persisted binding survives a build.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Key {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    Space,
    Tab,
    Enter,
    Shift,
    Control,
    Alt,
    Up,
    Down,
    Left,
    Right,
    Minus,
    Equals,
    LeftBracket,
    RightBracket,
    Backslash,
    Semicolon,
    Quote,
    Comma,
    Period,
    Slash,
    Backtick,
}

impl Key {
    /// A short label for the settings menu (e.g. `"W"`, `"Space"`, `"Shift"`).
    pub fn display_name(self) -> &'static str {
        match self {
            Key::A => "A",
            Key::B => "B",
            Key::C => "C",
            Key::D => "D",
            Key::E => "E",
            Key::F => "F",
            Key::G => "G",
            Key::H => "H",
            Key::I => "I",
            Key::J => "J",
            Key::K => "K",
            Key::L => "L",
            Key::M => "M",
            Key::N => "N",
            Key::O => "O",
            Key::P => "P",
            Key::Q => "Q",
            Key::R => "R",
            Key::S => "S",
            Key::T => "T",
            Key::U => "U",
            Key::V => "V",
            Key::W => "W",
            Key::X => "X",
            Key::Y => "Y",
            Key::Z => "Z",
            Key::Num0 => "0",
            Key::Num1 => "1",
            Key::Num2 => "2",
            Key::Num3 => "3",
            Key::Num4 => "4",
            Key::Num5 => "5",
            Key::Num6 => "6",
            Key::Num7 => "7",
            Key::Num8 => "8",
            Key::Num9 => "9",
            Key::Space => "Space",
            Key::Tab => "Tab",
            Key::Enter => "Enter",
            Key::Shift => "Shift",
            Key::Control => "Ctrl",
            Key::Alt => "Alt",
            Key::Up => "Up",
            Key::Down => "Down",
            Key::Left => "Left",
            Key::Right => "Right",
            Key::Minus => "-",
            Key::Equals => "=",
            Key::LeftBracket => "[",
            Key::RightBracket => "]",
            Key::Backslash => "\\",
            Key::Semicolon => ";",
            Key::Quote => "'",
            Key::Comma => ",",
            Key::Period => ".",
            Key::Slash => "/",
            Key::Backtick => "`",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_variant_name() {
        // A unit variant serializes to its name, so a persisted binding is
        // readable and stable across builds.
        let json = serde_json::to_string(&Key::W).unwrap();
        assert_eq!(json, "\"W\"");
        let back: Key = serde_json::from_str(&json).unwrap();
        assert_eq!(back, Key::W);
    }

    #[test]
    fn display_names_are_short() {
        assert_eq!(Key::W.display_name(), "W");
        assert_eq!(Key::Space.display_name(), "Space");
        assert_eq!(Key::Shift.display_name(), "Shift");
        assert_eq!(Key::Num1.display_name(), "1");
    }
}
