use crate::world::find_world_jsonl;
use concinnity_cook::build_from_path;

// Compile a world to binary blobs and write world-lock.json.
// Entry point for the `cn build` CLI subcommand.
//
// `world` is an optional world name resolved against .concinnity/worlds/; None
// selects the most recently modified world there, falling back to world.jsonl.
//
// If `server` and `user` are both provided, any source files missing from
// .concinnity/assets/ are fetched from the server before compiling.
pub fn build(json_path: Option<&str>) -> std::io::Result<()> {
    let resolved;
    let json_path = match json_path {
        Some(p) if std::path::Path::new(p).exists() => p,
        _ => {
            resolved = find_world_jsonl(None)?;
            resolved.as_str()
        }
    };
    build_from_path(json_path)
}
