// src/cli/explain.rs
// Print one asset's effective entry from the expanded world: the full JSONL
// line as the build sees it, pasteable into world.jsonl verbatim. This is the
// override path for injected defaults and expanded assets, which have no line
// in the authored file to copy from.

use super::list::{provenance, resolve_world_path};

pub fn explain(name: &str, json_path: Option<&str>) -> std::io::Result<()> {
    let json_path = resolve_world_path(json_path)?;
    let content = std::fs::read_to_string(&json_path)?;

    let loaded = concinnity_cook::prepare_world(&content)
        .map_err(|errs| concinnity_cook::check::report_validation_errors(&errs))?;

    let Some(asset) = loaded.assets.iter().find(|a| a.name == name) else {
        let mut close: Vec<&str> = loaded
            .assets
            .iter()
            .map(|a| a.name.as_str())
            .filter(|n| n.contains(name))
            .take(5)
            .collect();
        close.sort_unstable();
        let hint = if close.is_empty() {
            String::new()
        } else {
            format!("; close matches: {}", close.join(", "))
        };
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("no asset named '{}' in the expanded world{}", name, hint),
        ));
    };

    let line = serde_json::json!({
        "name": asset.name,
        "type": asset.asset_type,
        "args": asset.args,
    });

    println!("// {}", provenance(&loaded, &asset.name));
    println!("{}", serde_json::to_string(&line)?);
    Ok(())
}
