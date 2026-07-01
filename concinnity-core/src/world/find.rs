pub const WORLD_JSONL: &str = "world.jsonl";
// Locate a world JSONL file.
//
// If `name` is given, returns `.concinnity/worlds/<name>.jsonl` when it exists.
// If `name` is None, returns the most recently modified `.jsonl` in
// `.concinnity/worlds/`. Falls back to `world.jsonl` in the current directory
// and then walks up parent directories for backward compatibility with
// development workflows that pre-date the `.concinnity/` layout.
pub fn find_world_jsonl(name: Option<&str>) -> std::io::Result<String> {
    let worlds_dir = crate::paths::worlds_dir();

    if let Some(n) = name {
        let path = worlds_dir.join(format!("{}.jsonl", n));
        if path.exists() {
            return Ok(path.to_string_lossy().into_owned());
        }
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("world '{}' not found at {}", n, path.display()),
        ));
    }

    // No name given: pick the most recently modified world in .concinnity/worlds/.
    if worlds_dir.is_dir() {
        let mut best: Option<(std::time::SystemTime, std::path::PathBuf)> = None;
        if let Ok(entries) = std::fs::read_dir(&worlds_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                    continue;
                }
                if let Ok(meta) = std::fs::metadata(&path) {
                    let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
                    if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                        best = Some((mtime, path));
                    }
                }
            }
        }
        if let Some((_, path)) = best {
            return Ok(path.to_string_lossy().into_owned());
        }
    }

    // Fall back to world.jsonl in cwd or any parent directory.
    let mut dir = std::env::current_dir()?;
    loop {
        let candidate = dir.join(WORLD_JSONL);
        if candidate.exists() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "no world found: run `cn fetch-world` or create `{}`",
                        WORLD_JSONL,
                    ),
                ));
            }
        }
    }
}
