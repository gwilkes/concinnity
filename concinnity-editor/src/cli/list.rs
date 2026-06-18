// src/cli/list.rs
use crate::ecs::ComponentType;
use crate::world::{find_world_jsonl, parse_world_jsonl};
use concinnity_cook::world::resolve_includes;

pub fn list(json_path: Option<&str>) -> std::io::Result<()> {
    let resolved;
    let json_path = match json_path {
        Some(p) if std::path::Path::new(p).exists() => p,
        _ => {
            resolved = find_world_jsonl(None)?;
            resolved.as_str()
        }
    };

    let content = std::fs::read_to_string(json_path).map_err(|e| {
        tracing::error!("Could not read {}: {}", json_path, e);
        e
    })?;

    let assets = parse_world_jsonl(&content).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to parse {}: {}", json_path, e),
        )
    })?;

    let raw = resolve_includes(assets)?;

    if raw.is_empty() {
        println!("{} has no assets.", json_path);
        return Ok(());
    }

    // collect rows, then print aligned
    struct Row {
        name: String,
        type_str: String,
        origin: String,
        payload: String,
    }

    let rows: Vec<Row> = raw
        .iter()
        .map(|v| {
            let name = v
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("(unnamed)")
                .to_string();
            let type_str = v
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("?")
                .to_string();

            let (origin, payload) = if let Ok(ct) = ComponentType::parse(&type_str) {
                let r = ct.registration();
                (format!("{:?}", r.origin), format!("{:?}", r.payload))
            } else {
                ("?".to_string(), "?".to_string())
            };

            Row {
                name,
                type_str,
                origin,
                payload,
            }
        })
        .collect();

    let w_name = rows.iter().map(|r| r.name.len()).max().unwrap_or(4).max(4);
    let w_type = rows
        .iter()
        .map(|r| r.type_str.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let w_origin = rows
        .iter()
        .map(|r| r.origin.len())
        .max()
        .unwrap_or(6)
        .max(6);

    println!(
        "{:<w_name$}  {:<w_type$}  {:<w_origin$}  PAYLOAD",
        "NAME",
        "TYPE",
        "ORIGIN",
        w_name = w_name,
        w_type = w_type,
        w_origin = w_origin,
    );
    println!("{}", "-".repeat(w_name + w_type + w_origin + 16));

    for r in &rows {
        println!(
            "{:<w_name$}  {:<w_type$}  {:<w_origin$}  {}",
            r.name,
            r.type_str,
            r.origin,
            r.payload,
            w_name = w_name,
            w_type = w_type,
            w_origin = w_origin,
        );
    }

    println!("\n{} asset(s) in {}", rows.len(), json_path);
    Ok(())
}
