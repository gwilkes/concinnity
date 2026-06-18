pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    if let Some(entries) = args.get("scenes").and_then(|v| v.as_array()) {
        if entries.is_empty() {
            return Err(format!(
                "Asset '{}': SceneReel 'scenes' list is empty",
                name
            ));
        }
        for (i, entry) in entries.iter().enumerate() {
            if entry.as_str().map(|s| s.is_empty()).unwrap_or(true) {
                return Err(format!(
                    "Asset '{}': SceneReel 'scenes[{}]' must be a non-empty scene name string",
                    name, i
                ));
            }
        }
    }
    Ok(())
}
