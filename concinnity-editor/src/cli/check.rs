// src/cli/check.rs: discovery wrapper around crate::app::check::check_at_path
//
// `cn test` accepts an optional --file path. When the path is missing or
// doesn't exist on disk, fall back to discovery via find_world_jsonl.

use crate::app::check::check_at_path;
use crate::world::find_world_jsonl;

pub fn check(json_path: &str) -> std::io::Result<()> {
    let resolved;
    let json_path = if !std::path::Path::new(json_path).exists() {
        resolved = find_world_jsonl(None)?;
        resolved.as_str()
    } else {
        json_path
    };
    check_at_path(json_path)
}
