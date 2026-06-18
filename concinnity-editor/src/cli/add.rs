// src/cli/add.rs: discovery wrapper around crate::app::add::add_to_path
//
// The CLI binary handles world-path discovery: try the standard
// `.concinnity/worlds/` location first, then fall back to `world.jsonl` in
// cwd. When the fallback is hit and the target is a 3D scene (.glb),
// `add_to_path` scaffolds a fresh world at that location.

use crate::app::add::add_to_path;
use crate::world::{WORLD_JSONL, find_world_jsonl};

pub fn add(name: Option<&str>, target: &str, template: Option<&str>) -> std::io::Result<()> {
    let world_path = match find_world_jsonl(None) {
        Ok(p) => p,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => WORLD_JSONL.to_string(),
        Err(e) => return Err(e),
    };
    add_to_path(&world_path, name, target, template)
}
