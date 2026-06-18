// src/app/pending.rs
//
// In-memory staging area for the active scene.
//
// While the app is connected to the server via `cn debug --websocket`, the
// agentic loop issues add / rm / load commands that modify this pending list.
// No disk write or build is triggered until `save` is called.
//
// On first access the list is seeded from the current world.jsonl so that
// the LLM can extend or modify an existing scene incrementally.

// Driven only by the binary's debug WebSocket channel; unreferenced in the
// FFI lib build.
#![allow(dead_code)]

use crate::world::{find_world_jsonl, parse_world_jsonl, write_world_jsonl};
use serde_json::Value;
use std::sync::Mutex;

static PENDING: Mutex<Option<Vec<Value>>> = Mutex::new(None);

fn ensure_initialized(guard: &mut std::sync::MutexGuard<Option<Vec<Value>>>) {
    if guard.is_none() {
        let assets = find_world_jsonl(None)
            .ok()
            .and_then(|path| std::fs::read_to_string(path).ok())
            .and_then(|content| parse_world_jsonl(&content).ok())
            .unwrap_or_default();
        **guard = Some(assets);
    }
}

pub fn add(asset_type: String, name: Option<String>, args: Option<Value>) -> Result<(), String> {
    let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    ensure_initialized(&mut guard);
    let list = guard.as_mut().unwrap();

    let name = name.unwrap_or_else(|| asset_type.to_lowercase());

    if list
        .iter()
        .any(|a| a.get("name").and_then(|v| v.as_str()) == Some(&name))
    {
        return Err(format!(
            "asset '{}' already exists in pending scene; rm it first",
            name
        ));
    }

    let args = args.unwrap_or_else(|| Value::Object(Default::default()));
    list.push(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": args,
    }));
    tracing::debug!("pending add: {} ({})", name, asset_type);
    Ok(())
}

pub fn rm(name: &str) -> Result<(), String> {
    let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    ensure_initialized(&mut guard);
    let list = guard.as_mut().unwrap();

    let before = list.len();
    list.retain(|a| a.get("name").and_then(|v| v.as_str()) != Some(name));

    if list.len() == before {
        Err(format!("no asset named '{}' in pending scene", name))
    } else {
        tracing::debug!("pending rm: {}", name);
        Ok(())
    }
}

pub fn load(assets: Vec<Value>) {
    let mut guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    *guard = Some(assets);
    tracing::debug!("pending load: {} assets", guard.as_ref().unwrap().len());
}

pub fn save() -> std::io::Result<()> {
    let guard = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    let list = match guard.as_ref() {
        Some(l) => l,
        None => return Ok(()),
    };

    // Use existing world.jsonl path or fall back to cwd
    let world_path = find_world_jsonl(None).unwrap_or_else(|_| "world.jsonl".to_string());

    let content = write_world_jsonl(list).map_err(|e| std::io::Error::other(e.to_string()))?;

    std::fs::write(&world_path, &content)?;
    tracing::info!(
        "saved pending scene ({} assets) to {}",
        list.len(),
        world_path
    );
    Ok(())
}
