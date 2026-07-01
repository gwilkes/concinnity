// src/world/preset.rs
// Preset file loading utilities shared by all expansion modules.

// Recursively search .concinnity/assets/ for a file matching the given bare filename.
pub fn find_in_assets(filename: &str) -> Option<String> {
    fn walk(dir: &std::path::Path, filename: &str) -> Option<std::path::PathBuf> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return None;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(filename) {
                return Some(path);
            }
            if path.is_dir()
                && let Some(found) = walk(&path, filename)
            {
                return Some(found);
            }
        }
        None
    }
    walk(&crate::paths::assets_dir(), filename).map(|p| p.to_string_lossy().into_owned())
}

fn find_preset_path(filename: &str, subdir: &str) -> Option<String> {
    if let Some(p) = find_in_assets(filename) {
        return Some(p);
    }
    let path = crate::paths::assets_dir().join(subdir).join(filename);
    if path.exists() {
        return Some(path.to_string_lossy().into_owned());
    }
    None
}

// Load a JSON object from assets/<subdir>/<name>.json.
pub fn load_preset_obj(name: &str, subdir: &str) -> serde_json::Value {
    let filename = format!("{}.json", name);
    let path = find_preset_path(&filename, subdir);
    let Some(path) = path else {
        return serde_json::Value::Null;
    };
    let Ok(content) = std::fs::read_to_string(&path) else {
        return serde_json::Value::Null;
    };
    serde_json::from_str::<serde_json::Value>(&content).unwrap_or(serde_json::Value::Null)
}
