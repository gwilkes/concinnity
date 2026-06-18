// src/world.rs
// world.jsonl I/O: parsing, serialization, and file utilities.

pub mod preset;

mod find;

pub use find::{WORLD_JSONL, find_world_jsonl};

// Local project state directories created on first fetch.

pub const CONCINNITY_ASSETS_DIR: &str = ".concinnity/assets";
pub const CONCINNITY_DATA_DIR: &str = ".concinnity/data";
// Mutable counterpart to CONCINNITY_DATA_DIR: build output is regenerated and
// immutable, this holds runtime state the user changes (e.g. settings-menu
// choices) and is never written by a build. A sibling of `data`, so it shares
// the same project-root anchor in both the CLI (cwd) and the editor FFI (the
// cwd guard), with no extra path plumbing.
pub const CONCINNITY_CONFIG_DIR: &str = ".concinnity/config";

// An asset entry after $include resolution and type parsing
#[derive(Clone)]
pub struct WorldJsonlAsset {
    pub name: String,
    pub asset_type: String,
    pub args: serde_json::Value,
}

impl WorldJsonlAsset {
    // Build a typed entry from a raw JSON asset object. `name` and `type` are
    // expected to be present; a missing field degrades to an empty string
    // rather than failing.
    pub fn from_value(v: &serde_json::Value) -> Self {
        WorldJsonlAsset {
            name: v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string(),
            asset_type: v
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            args: v
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
        }
    }
}

// Parse a world.jsonl string into a flat list of raw asset objects.
//
// Each non-blank, non-comment line must be a valid JSON object. The order
// of entries is preserved. Returns an error on the first malformed line.
pub fn parse_world_jsonl(content: &str) -> Result<Vec<serde_json::Value>, serde_json::Error> {
    let mut assets = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        assets.push(value);
    }
    Ok(assets)
}

// Serialize a list of asset objects back to world.jsonl format.
//
// Each entry is written as a compact single-line JSON object followed by a
// newline. The result is a valid world.jsonl file.
pub fn write_world_jsonl(assets: &[serde_json::Value]) -> serde_json::Result<String> {
    let mut out = String::new();
    for asset in assets {
        out.push_str(&serde_json::to_string(asset)?);
        out.push('\n');
    }
    Ok(out)
}

// Read src_path, apply a fallible mutation to the asset list, and write
// the result to dst_path. src and dst may be the same path or different.
pub fn patch_world_jsonl_to<F>(src_path: &str, dst_path: &str, f: F) -> std::io::Result<()>
where
    F: FnOnce(&mut Vec<serde_json::Value>) -> std::io::Result<()>,
{
    let content = std::fs::read_to_string(src_path).map_err(|e| {
        tracing::error!("Could not read {}: {}", src_path, e);
        e
    })?;

    let mut assets = parse_world_jsonl(&content).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse {}: {}", src_path, e),
        )
    })?;

    f(&mut assets)?;

    let out = write_world_jsonl(&assets).map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(dst_path, out)
}

// Read world.jsonl at json_path, mutate the asset list in-place, write back
pub fn patch_world_jsonl<F>(json_path: &str, f: F) -> std::io::Result<()>
where
    F: FnOnce(&mut Vec<serde_json::Value>),
{
    patch_world_jsonl_to(json_path, json_path, |assets| {
        f(assets);
        Ok(())
    })
}

// Read asset names from world.jsonl without a full parse, for error messages
pub fn known_names(json_path: &str) -> std::io::Result<Vec<String>> {
    let content = std::fs::read_to_string(json_path)?;
    let assets = parse_world_jsonl(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(assets
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_world_jsonl_empty_string_returns_empty() {
        let assets = parse_world_jsonl("").unwrap();
        assert!(assets.is_empty());
    }

    #[test]
    fn parse_world_jsonl_skips_blank_and_comment_lines() {
        let content = "\n  \n// this is a comment\n";
        let assets = parse_world_jsonl(content).unwrap();
        assert!(assets.is_empty());
    }

    #[test]
    fn parse_world_jsonl_returns_entries_in_order() {
        let content = r#"{"name":"a","type":"Logger"}
{"name":"b","type":"Window"}
"#;
        let assets = parse_world_jsonl(content).unwrap();
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0]["name"], "a");
        assert_eq!(assets[1]["name"], "b");
    }

    #[test]
    fn parse_world_jsonl_errors_on_invalid_json() {
        let result = parse_world_jsonl("{not valid json}");
        assert!(result.is_err());
    }

    #[test]
    fn write_world_jsonl_one_line_per_entry() {
        let assets = vec![
            serde_json::json!({"name": "a", "type": "Logger"}),
            serde_json::json!({"name": "b", "type": "Window"}),
        ];
        let out = write_world_jsonl(&assets).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"a\""));
        assert!(lines[1].contains("\"b\""));
    }

    #[test]
    fn write_world_jsonl_round_trips_through_parse() {
        let assets = vec![serde_json::json!({"name": "x", "type": "Logger", "args": {}})];
        let out = write_world_jsonl(&assets).unwrap();
        let parsed = parse_world_jsonl(&out).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["name"], "x");
    }

    #[test]
    fn patch_world_jsonl_to_applies_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("world.jsonl");
        let dst = dir.path().join("out.jsonl");
        std::fs::write(&src, "{\"name\":\"a\",\"type\":\"Logger\"}\n").unwrap();

        patch_world_jsonl_to(src.to_str().unwrap(), dst.to_str().unwrap(), |assets| {
            assets.push(serde_json::json!({"name":"b","type":"Window"}));
            Ok(())
        })
        .unwrap();

        let content = std::fs::read_to_string(&dst).unwrap();
        let parsed = parse_world_jsonl(&content).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[1]["name"], "b");
    }

    #[test]
    fn patch_world_jsonl_to_propagates_mutation_error() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("world.jsonl");
        std::fs::write(&src, "{\"name\":\"a\",\"type\":\"Logger\"}\n").unwrap();

        let result =
            patch_world_jsonl_to(src.to_str().unwrap(), src.to_str().unwrap(), |_assets| {
                Err(std::io::Error::other("boom"))
            });
        assert!(result.is_err());
    }

    #[test]
    fn known_names_extracts_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(
            &path,
            "{\"name\":\"a\",\"type\":\"Logger\"}\n{\"name\":\"b\",\"type\":\"Window\"}\n",
        )
        .unwrap();
        let names = known_names(path.to_str().unwrap()).unwrap();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn known_names_empty_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("world.jsonl");
        std::fs::write(&path, "").unwrap();
        let names = known_names(path.to_str().unwrap()).unwrap();
        assert!(names.is_empty());
    }
}
