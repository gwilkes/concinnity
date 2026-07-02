// src/cli/new.rs

use crate::world::WORLD_JSONL;
use concinnity_cook::build_from_path;

// Default starter world file. Everything else a running world needs (window,
// renderer, debug HUD) is injected at build time and recorded in
// world-lock.json; `cn list --expanded` shows the effective world.
const INIT_WORLD_JSONL: &str = r#"{"name":"hello_world","type":"TextLabel","args":{"content":"Hello, world!"}}
"#;

pub fn new(path: &str) -> std::io::Result<()> {
    if std::path::Path::new(path).exists() {
        // allow creating a project in a pre-existing empty directory,
        // but refuse if it already has a world.jsonl
        let world = std::path::Path::new(path).join(WORLD_JSONL);
        if world.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("'{}' already contains a {}", path, WORLD_JSONL),
            ));
        }
    }
    std::fs::create_dir_all(path)?;
    println!("Created directory '{}'", path);
    init_in_dir(path)
}

pub fn init() -> std::io::Result<()> {
    init_in_dir(".")
}

// Write the starter world.jsonl into `dir` and run an initial build
fn init_in_dir(dir: &str) -> std::io::Result<()> {
    let world_path = std::path::Path::new(dir).join(WORLD_JSONL);

    if world_path.exists() {
        println!("{} already exists, skipping init", world_path.display());
        return Ok(());
    }

    std::fs::write(&world_path, INIT_WORLD_JSONL)?;
    println!("Created {}", world_path.display());

    let world_path_str = world_path.to_str().unwrap_or(WORLD_JSONL);
    build_from_path(world_path_str)
}
