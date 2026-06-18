// src/app/check.rs
// Validate a world JSONL without producing blob files.
//
// Runs the validation front half of the build pipeline (load, expand, and
// the semantic checks in `crate::check`) and reports the outcome. Used by
// `cn test`, the FFI `cn_check_world` entry, and the infra agentic loop.

// Read `world_path`, run validation, and report results. Returns Ok if every
// asset passes; otherwise an error whose Display contains a human-readable
// summary of every failure (one per asset).
pub fn check_at_path(world_path: &str) -> std::io::Result<()> {
    let content = std::fs::read_to_string(world_path)?;
    check_from_str(&content, world_path)
}

// Run validation against an in-memory world JSONL string. `label` is the
// origin used in messages (typically the source path).
pub fn check_from_str(content: &str, label: &str) -> std::io::Result<()> {
    match concinnity_cook::prepare_world(content) {
        Ok(loaded) => {
            println!("ok: {} asset(s) passed in {}", loaded.assets.len(), label);
            Ok(())
        }
        Err(errors) => Err(concinnity_cook::check::report_validation_errors(&errors)),
    }
}
