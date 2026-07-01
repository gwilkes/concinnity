// Project state root: where the engine's `.concinnity/` directory is anchored.
//
// Everything the engine writes for a project lives under `.concinnity/`: the
// compiled blob (`data/`), the payload cache (`cache/`), fetched source assets
// (`assets/`), mutable runtime config (`config/`), and named worlds
// (`worlds/`). By default that tree is addressed relative to the current
// working directory, so `.concinnity/` sits wherever a command runs. That is
// the historical behavior and is unchanged when no root is installed.
//
// A host that must change the working directory for an unrelated reason (an
// example that chdirs so its world's relative asset paths resolve against the
// example directory) would otherwise drag `.concinnity/` along with it. Such a
// host captures the invocation directory and installs it here before it
// chdirs, so state stays put while content resolution follows the working
// directory.
//
// Resolution order, highest precedence first:
//   1. a root installed via `set_root`
//   2. the `CN_HOME` environment variable
//   3. none: paths stay relative to the current directory (the default)

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

// Name of the state directory joined onto the resolved root.
pub const STATE_DIR: &str = ".concinnity";

// Environment variable that anchors the state root when no root is installed.
pub const HOME_ENV: &str = "CN_HOME";

fn installed_root() -> &'static Mutex<Option<PathBuf>> {
    static ROOT: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    ROOT.get_or_init(|| Mutex::new(None))
}

// Anchor `.concinnity/` to `dir` for the rest of the process, taking precedence
// over `CN_HOME` and the working-directory default. A host that chdirs for
// content resolution installs the invocation directory here before it chdirs.
pub fn set_root<P: Into<PathBuf>>(dir: P) {
    *installed_root().lock().unwrap() = Some(dir.into());
}

// Remove an installed root, restoring environment/working-directory resolution.
pub fn clear_root() {
    *installed_root().lock().unwrap() = None;
}

// The resolved root, or `None` when paths should stay relative to the cwd.
fn root() -> Option<PathBuf> {
    installed_root()
        .lock()
        .unwrap()
        .clone()
        .or_else(|| std::env::var_os(HOME_ENV).map(PathBuf::from))
}

// The `.concinnity` state directory: `<root>/.concinnity` when a root is
// installed, otherwise the relative `.concinnity` (resolved against the cwd).
pub fn state_dir() -> PathBuf {
    anchor(root().as_deref(), Path::new(STATE_DIR))
}

// Join `rel` onto `root`, or return it unchanged (relative) when `root` is
// `None`. Split out so the anchoring rule is unit-testable without touching the
// process-global root.
fn anchor(root: Option<&Path>, rel: &Path) -> PathBuf {
    match root {
        Some(r) => r.join(rel),
        None => rel.to_path_buf(),
    }
}

pub fn assets_dir() -> PathBuf {
    state_dir().join("assets")
}

pub fn data_dir() -> PathBuf {
    state_dir().join("data")
}

pub fn config_dir() -> PathBuf {
    state_dir().join("config")
}

pub fn worlds_dir() -> PathBuf {
    state_dir().join("worlds")
}

pub fn cache_dir() -> PathBuf {
    state_dir().join("cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchor_without_root_stays_relative() {
        let p = anchor(None, Path::new(STATE_DIR));
        assert_eq!(p, Path::new(STATE_DIR));
        assert!(p.is_relative());
    }

    #[test]
    fn anchor_with_root_is_under_it() {
        let root = Path::new("/proj/game");
        let p = anchor(Some(root), Path::new(STATE_DIR));
        assert!(p.starts_with(root));
        assert_eq!(p, root.join(STATE_DIR));
        assert_eq!(p.file_name().unwrap(), STATE_DIR);
    }

    #[test]
    fn subdirs_hang_off_the_state_dir() {
        // The layout under `.concinnity/` is stable regardless of the anchor.
        // Exercised through the pure `anchor` helper so the assertion does not
        // depend on the process-global root.
        let base = anchor(Some(Path::new("/proj")), Path::new(STATE_DIR));
        for sub in ["assets", "data", "config", "worlds", "cache"] {
            assert_eq!(base.join(sub), Path::new("/proj").join(STATE_DIR).join(sub));
        }
    }

    // Exercises the process-global root end to end. Safe because no other test
    // in this crate reads the global (the blob tests use explicit temp paths).
    #[test]
    fn installed_root_redirects_every_state_dir() {
        // The installed root takes precedence over CN_HOME and the cwd default,
        // so these assertions hold regardless of the test environment.
        let root = Path::new("/tmp/anchor-probe");
        set_root(root);
        let expected = root.join(STATE_DIR);
        assert_eq!(state_dir(), expected);
        assert_eq!(cache_dir(), expected.join("cache"));
        assert_eq!(data_dir(), expected.join("data"));
        assert_eq!(config_dir(), expected.join("config"));
        assert_eq!(assets_dir(), expected.join("assets"));
        assert_eq!(worlds_dir(), expected.join("worlds"));

        // Clearing restores cwd-relative resolution, unless CN_HOME is set.
        clear_root();
        if std::env::var_os(HOME_ENV).is_none() {
            assert_eq!(cache_dir(), Path::new(STATE_DIR).join("cache"));
        }
    }
}
