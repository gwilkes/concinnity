// src/app/add.rs
// Add an asset to a world JSONL and rebuild.
//
// The CLI and FFI both funnel through `add_to_path`, which:
//   - bootstraps a missing world from a `.glb` / `.txt` / `.md` target (a
//     text target becomes a `TextLabel` whose content is the file body); the
//     renderer stack itself is injected at build time from the entries'
//     companions, so no scaffold lines are written,
//   - appends a named content template's entries (`--template showcase`) when
//     one is requested for a `.glb` landing in a renderer-less world,
//   - resolves `target` as a file path, a known asset type name, or inline
//     JSON, building one or more asset entries,
//   - patches the world JSONL atomically (via a tmp file) and reruns the
//     build pipeline so blobs and the lock file stay in sync. Only the
//     requested entries are written; injected companions and engine defaults
//     stay build-time only (see world-lock.json).

use crate::ecs::ComponentType;
use crate::ecs::asset_api::{AssetRequest, create_asset_def};
use crate::world::{WORLD_JSONL, patch_world_jsonl_to};
use concinnity_cook::build_from_path;

// Add an asset to `world_path` and rebuild. See module docs.
//
// `template` selects a named scaffold preset when scaffolding fires
// (target is `.glb`, world has no renderer trigger). `None` uses the
// default scaffold; `Some("showcase")` uses the polished showcase template.
// Unknown names error out before touching the world file.
pub fn add_to_path(
    world_path: &str,
    name: Option<&str>,
    target: &str,
    template: Option<&str>,
) -> std::io::Result<()> {
    let scaffold = scaffold_to_inject(world_path, target, template)?;
    ensure_world_file_exists(world_path)?;

    let mut entries = resolve_add_target(target)?;

    match (name, entries.len()) {
        (Some(n), 1) => {
            entries[0]["name"] = serde_json::Value::String(n.to_string());
        }
        (Some(n), _) => {
            // multi-entry (e.g. a .metal file with both vertex and fragment stages):
            // use the supplied name as a prefix, keeping the existing _vert/_frag suffix
            for entry in &mut entries {
                let existing = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let suffix = existing
                    .rsplit_once('_')
                    .map(|(_, s)| format!("_{s}"))
                    .unwrap_or_default();
                entry["name"] = serde_json::Value::String(format!("{n}{suffix}"));
            }
        }
        (None, _) => {}
    }

    let entry_names: Vec<String> = entries
        .iter()
        .map(|e| {
            e.get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "resolved asset entry has no `name` field",
                    )
                })
                .map(str::to_string)
        })
        .collect::<Result<_, _>>()?;

    let tmp_path = format!("{}.tmp", world_path);

    patch_world_jsonl_to(world_path, &tmp_path, |assets| {
        // Re-adding a text file (or any other single-entry TextLabel target)
        // refreshes the existing same-name TextLabel's `content` in place
        // instead of erroring. Drag-dropping the same `.txt` twice (or
        // editing its body and re-adding) should "just work". Other args
        // (font, x/y, color, scale, centered, ...) are left alone so any
        // hand edits to the TextLabel survive the refresh.
        if let Some(refreshed) = try_refresh_text_label(assets, &entries) {
            tracing::info!("refreshed TextLabel '{}' content", refreshed);
            return Ok(());
        }

        for entry_name in &entry_names {
            if let Some(existing) = assets
                .iter()
                .find(|a| a.get("name").and_then(|v| v.as_str()) == Some(entry_name.as_str()))
            {
                let existing_type = existing.get("type").and_then(|v| v.as_str()).unwrap_or("?");
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!(
                        "an asset named '{}' (type: {}) already exists in {}; \
                         remove it first with `concinnity rm {}`",
                        entry_name, existing_type, WORLD_JSONL, entry_name
                    ),
                ));
            }
        }
        // Append template entries first so the new asset's own systems (e.g.
        // a glTF's Camera3D) run alongside the template's setup. Any template
        // entry whose name already exists in the world is skipped to avoid
        // clobbering user-authored assets that happen to share the name.
        for entry in scaffold {
            let n = entry.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if !assets
                .iter()
                .any(|a| a.get("name").and_then(|v| v.as_str()) == Some(n))
            {
                assets.push(entry);
            }
        }
        assets.extend(entries);
        Ok(())
    })?;

    match build_from_path(&tmp_path) {
        Ok(()) => std::fs::rename(&tmp_path, world_path).inspect_err(|_e| {
            let _ = std::fs::remove_file(&tmp_path);
        }),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp_path);
            Err(e)
        }
    }
}

// If `entries` is a single TextLabel whose name matches an existing
// TextLabel in `assets`, overwrite only the existing entry's `args.content`
// from the new entry and return the name. Returns `None` otherwise: the
// caller falls back to the normal "append, error on duplicate" flow.
//
// Scope is intentionally narrow:
//   - Single-entry only, so a `.glb` re-add (which fans into many entries)
//     doesn't accidentally clobber one of the existing materials/meshes.
//   - Same name AND same type, so a TextLabel never overwrites an
//     unrelated asset that happens to share a name.
//   - Only `content` is copied across; the user's edits to font, x/y,
//     color, scale, centered, background, padding, visible, view all stay.
fn try_refresh_text_label(
    assets: &mut [serde_json::Value],
    entries: &[serde_json::Value],
) -> Option<String> {
    if entries.len() != 1 {
        return None;
    }
    let new = &entries[0];
    if new.get("type").and_then(|v| v.as_str()) != Some("TextLabel") {
        return None;
    }
    let new_name = new.get("name").and_then(|v| v.as_str())?.to_string();
    let new_content = new.get("args").and_then(|a| a.get("content")).cloned()?;

    let existing = assets.iter_mut().find(|a| {
        a.get("name").and_then(|v| v.as_str()) == Some(new_name.as_str())
            && a.get("type").and_then(|v| v.as_str()) == Some("TextLabel")
    })?;
    let args = existing.get_mut("args")?.as_object_mut()?;
    args.insert("content".to_string(), new_content);
    Some(new_name)
}

// Decide which scaffold entries the patch closure should inject. Returns
// empty when no scaffolding is needed: the target doesn't bootstrap a world,
// the world already has a renderer-trigger asset (GraphicsSystem /
// GraphicsConfig / TextLabel / Window), or the target type provides its own
// renderer trigger (e.g. a text target's TextLabel). Errors when the world
// file is missing and the target can't bootstrap one: only `.glb`, `.txt`,
// and `.md` are allowed to create a world from nothing.
//
// `template` picks which scaffold flavour to inject. Unknown names error
// out before any file I/O so the user sees the typo immediately. An unknown
// template is treated as an error even when scaffolding wouldn't fire
// (e.g. existing world with a renderer trigger), since silently ignoring
// `--template foo` would mask the typo. `--template` is GLB-only: passing
// it with a text target is also rejected so the typo doesn't survive.
fn scaffold_to_inject(
    world_path: &str,
    target: &str,
    template: Option<&str>,
) -> std::io::Result<Vec<serde_json::Value>> {
    let template_entries = resolve_template(template)?;

    let world_exists = std::path::Path::new(world_path).exists();
    let bootstrap = target_bootstrap_kind(target);

    if !world_exists && matches!(bootstrap, BootstrapKind::None) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no world found at '{}': create one with `cn fetch-world` or `cn new`",
                world_path
            ),
        ));
    }

    if template_entries.is_some() && !matches!(bootstrap, BootstrapKind::Scene) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--template only applies to 3D scene targets (.glb)",
        ));
    }

    match bootstrap {
        BootstrapKind::None | BootstrapKind::Text => Ok(Vec::new()),
        BootstrapKind::Scene => {
            if world_exists && has_renderer_trigger(world_path)? {
                return Ok(Vec::new());
            }
            // No default scaffold: the renderer stack is injected at build
            // time from the scene's own assets. Only an explicitly requested
            // template writes extra entries.
            Ok(template_entries.unwrap_or_default())
        }
    }
}

// Map a `--template <name>` value to its scaffold entries. Returns:
//   - `Ok(None)`              when no template was requested (use default scaffold)
//   - `Ok(Some(entries))`     when the named template is known
//   - `Err(InvalidInput)`     when the name is unrecognised (typo → fail fast)
fn resolve_template(template: Option<&str>) -> std::io::Result<Option<Vec<serde_json::Value>>> {
    match template {
        None => Ok(None),
        Some("showcase") => Ok(Some(super::sources::glb::template_showcase())),
        Some(other) => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("unknown template '{other}'; available: showcase"),
        )),
    }
}

// Asset types that either declare the renderer or trigger its companion
// injection (see `world::companion`). If any of these are already present
// the world will start the GraphicsSystem on its own and no scaffold is
// needed.
const RENDERER_TRIGGER_TYPES: &[&str] = &["GraphicsConfig", "TextLabel", "Window"];

// Whether the JSONL file at `world_path` already contains a renderer trigger.
// Malformed lines are skipped silently: the regular load path will surface
// any parse problems with full diagnostics.
fn has_renderer_trigger(world_path: &str) -> std::io::Result<bool> {
    let content = std::fs::read_to_string(world_path)?;
    Ok(jsonl_has_renderer_trigger(&content))
}

// Pure-string variant of `has_renderer_trigger`, exposed for unit tests.
fn jsonl_has_renderer_trigger(content: &str) -> bool {
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(t) = value.get("type").and_then(|v| v.as_str())
            && RENDERER_TRIGGER_TYPES.contains(&t)
        {
            return true;
        }
    }
    false
}

// Make sure `world_path` (and its parent directory) exists so
// `patch_world_jsonl_to` can read from it. Creates an empty file when missing;
// no-op when the file already exists.
fn ensure_world_file_exists(world_path: &str) -> std::io::Result<()> {
    if std::path::Path::new(world_path).exists() {
        return Ok(());
    }
    if let Some(parent) = std::path::Path::new(world_path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(world_path, "")?;
    tracing::info!("created empty world file at {}", world_path);
    Ok(())
}

// How a target relates to renderer scaffolding. Targets that can bootstrap a
// world from nothing fall into `Scene` (needs the GLB scaffold injected
// alongside) or `Text` (the TextLabel that gets emitted is itself a renderer
// trigger and its companions inject the rest of the stack). Everything else
// is `None`: adding a shader or font into a missing world is rejected.
pub(crate) enum BootstrapKind {
    None,
    Scene,
    Text,
}

pub(crate) fn target_bootstrap_kind(target: &str) -> BootstrapKind {
    let ext = std::path::Path::new(target)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("glb") | Some("fbx") => BootstrapKind::Scene,
        Some("txt") | Some("md") => BootstrapKind::Text,
        _ => BootstrapKind::None,
    }
}

fn is_path_like(s: &str) -> bool {
    if s.contains('/') || s.contains('\\') {
        return true;
    }
    if s.starts_with('.') || s.starts_with('~') {
        return true;
    }
    // has a dot but the full string isn't a known type name
    if s.contains('.') {
        return ComponentType::parse(s).is_err();
    }
    false
}

fn validated_entry(
    name: &str,
    asset_type: &str,
    args: serde_json::Value,
) -> std::io::Result<serde_json::Value> {
    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args: Some(args),
    };
    let def = create_asset_def(&req)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let resolved_args: serde_json::Value = serde_json::from_slice(&def.args_bytes)
        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));

    Ok(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": resolved_args,
    }))
}

// Build a SceneImport entry for a 3D scene file. SceneImport is a build-time
// (BuildOnly) asset, so it can't go through `validated_entry` /
// `create_asset_def` (which only build External components); materialize its
// default args from the registration and set the source path.
fn scene_import_entry(name: &str, source: &str) -> std::io::Result<serde_json::Value> {
    let reg = ComponentType::parse("SceneImport")
        .map_err(|_| std::io::Error::other("SceneImport asset type is unavailable"))?
        .registration();
    let mut args = reg
        .default_args
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
    if let serde_json::Value::Object(map) = &mut args {
        map.insert(
            "source".to_string(),
            serde_json::Value::String(source.to_string()),
        );
    }
    Ok(serde_json::json!({
        "name": name,
        "type": "SceneImport",
        "args": args,
    }))
}

fn entry_from_path(path_str: &str) -> std::io::Result<Vec<serde_json::Value>> {
    let path = std::path::Path::new(path_str);

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    // stem without extension, dots replaced with underscores (shared with
    // companion injection so a generated default asset is named identically)
    let stem = concinnity_cook::world::asset_name_from_path(path_str);

    // full filename with dots replaced with underscores (used for most types)
    let base_name = if ext == "json" {
        stem.clone()
    } else {
        path.file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.replace('.', "_"))
            .unwrap_or_else(|| path_str.to_string())
    };

    if ext == "json" {
        return entry_from_json_file(path, &base_name).map(|e| vec![e]);
    }

    match ext.as_str() {
        // Dedicated-extension GLSL shaders: stage is unambiguous from the extension
        "vert" => Ok(vec![validated_entry(
            &base_name,
            "ShaderStage",
            serde_json::json!({ "kind": "vertex", "source": path_str }),
        )?]),
        "frag" => Ok(vec![validated_entry(
            &base_name,
            "ShaderStage",
            serde_json::json!({ "kind": "fragment", "source": path_str }),
        )?]),

        // GLSL: infer stage from filename stem
        "glsl" => {
            let kind = if stem.contains("frag") || stem.contains("fragment") {
                "fragment"
            } else {
                "vertex"
            };
            Ok(vec![validated_entry(
                &base_name,
                "ShaderStage",
                serde_json::json!({ "kind": kind, "source": path_str }),
            )?])
        }

        // Metal: parse source to detect which stages are present
        "metal" => {
            let source = read_source_file(path_str)?;
            shader_entries_from_stages(path_str, &stem, "metal", &detect_metal_stages(&source))
        }

        // WGSL: parse source to detect which stages are present
        "wgsl" => {
            let source = read_source_file(path_str)?;
            shader_entries_from_stages(path_str, &stem, "wgsl", &detect_wgsl_stages(&source))
        }

        // Fonts: stem only, no extension suffix needed, font names won't conflict with shaders
        "ttf" | "otf" => Ok(vec![validated_entry(
            &stem,
            "Font",
            serde_json::json!({ "path": path_str }),
        )?]),

        // LLM weights
        "gguf" => Ok(vec![validated_entry(
            &base_name,
            "LLM",
            serde_json::json!({ "lib_path": "", "model_path": path_str }),
        )?]),

        // File-backed assets: path is stored as-is; build compiles the blob
        "obj" | "mtl" | "png" | "jpg" | "jpeg" | "bmp" | "tga" | "gif" => {
            Ok(vec![validated_entry(
                &base_name,
                "File",
                serde_json::json!({ "path": path_str, "kind": ext.as_str() }),
            )?])
        }

        // Text files become a TextLabel carrying the file contents. The label
        // is its own renderer trigger (see RENDERER_TRIGGER_TYPES) and companion
        // injection adds GraphicsSystem + a default Font, so a fresh
        // `cn add notes.txt` lands a renderable world without a scaffold.
        //
        // `centered: true` is set explicitly here to match `cn init`'s output.
        // (Companion injection's default-font pass auto-centers labels that omit
        // `centered`, but the validated_entry round-trip materializes the full
        // default struct so the auto-center path doesn't fire.)
        "txt" | "md" => Ok(vec![validated_entry(
            &stem,
            "TextLabel",
            serde_json::json!({
                "content": read_text_content(path_str)?,
                "centered": true,
            }),
        )?]),

        // 3D scene files: one SceneImport line. The build expands it into
        // Textures / Materials / Meshes / Models / Props at compile time, so
        // world.jsonl stays compact (see concinnity_core::build::import).
        "glb" | "fbx" => Ok(vec![scene_import_entry(&stem, path_str)?]),

        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "unknown extension '.{}'; pass a type name instead \
                 (e.g. `concinnity add Logger`) or edit {} directly",
                other, WORLD_JSONL
            ),
        )),
    }
}

// Build ShaderStage entries for each detected stage of a multi-stage source file.
// Names follow the pattern {stem}_{ext}_{stage_abbrev}, e.g. "default_metal_vert".
fn shader_entries_from_stages(
    path_str: &str,
    stem: &str,
    ext: &str,
    stages: &[&str],
) -> std::io::Result<Vec<serde_json::Value>> {
    stages
        .iter()
        .map(|&stage| {
            let name = format!("{}_{}_{}", stem, ext, stage_abbrev(stage));
            validated_entry(
                &name,
                "ShaderStage",
                serde_json::json!({ "kind": stage, "source": path_str }),
            )
        })
        .collect()
}

fn stage_abbrev(stage: &str) -> &str {
    match stage {
        "vertex" => "vert",
        "fragment" => "frag",
        other => other,
    }
}

// Detect Metal pipeline stages from source text.
// Metal uses `vertex` / `fragment` as function-qualifier keywords at the start of declarations.
// Returns a non-empty list; defaults to ["vertex"] when no qualifiers are found.
pub(crate) fn detect_metal_stages(source: &str) -> Vec<&'static str> {
    let has_vertex = source.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("vertex ") || t.starts_with("vertex\t")
    });
    let has_fragment = source.lines().any(|l| {
        let t = l.trim_start();
        t.starts_with("fragment ") || t.starts_with("fragment\t")
    });
    stages_from_flags(has_vertex, has_fragment)
}

// Detect WGSL pipeline stages from source text.
// WGSL uses `@vertex` / `@fragment` attribute decorators.
// Returns a non-empty list; defaults to ["vertex"] when no attributes are found.
pub(crate) fn detect_wgsl_stages(source: &str) -> Vec<&'static str> {
    stages_from_flags(source.contains("@vertex"), source.contains("@fragment"))
}

fn stages_from_flags(has_vertex: bool, has_fragment: bool) -> Vec<&'static str> {
    let mut stages = Vec::new();
    if has_vertex {
        stages.push("vertex");
    }
    if has_fragment {
        stages.push("fragment");
    }
    if stages.is_empty() {
        stages.push("vertex");
    }
    stages
}

fn read_source_file(path_str: &str) -> std::io::Result<String> {
    std::fs::read_to_string(path_str)
        .map_err(|e| std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e)))
}

// Cap text-label contents at a size that makes sense for a HUD overlay. A
// TextLabel isn't a document viewer; multi-MB drops would silently bloat
// world.jsonl and the per-frame layout pass.
const TEXT_LABEL_MAX_BYTES: usize = 64 * 1024;

// Read a `.txt` / `.md` file for use as TextLabel `content`. Strips a single
// trailing `\n` (and a preceding `\r`, in case the file is CRLF) so labels
// don't render with a stray blank line at the bottom, but otherwise preserves
// internal whitespace and newlines verbatim. Files larger than
// `TEXT_LABEL_MAX_BYTES` are rejected up front rather than truncated, so the
// user knows the contents weren't silently clipped.
fn read_text_content(path_str: &str) -> std::io::Result<String> {
    let metadata = std::fs::metadata(path_str).map_err(|e| {
        std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e))
    })?;
    if metadata.len() as usize > TEXT_LABEL_MAX_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "'{}' is {} bytes; TextLabel content is capped at {} bytes: \
                 trim the file or add it as a `File` asset via inline JSON instead",
                path_str,
                metadata.len(),
                TEXT_LABEL_MAX_BYTES
            ),
        ));
    }
    let mut content = std::fs::read_to_string(path_str).map_err(|e| {
        std::io::Error::new(e.kind(), format!("could not read '{}': {}", path_str, e))
    })?;
    if content.ends_with('\n') {
        content.pop();
        if content.ends_with('\r') {
            content.pop();
        }
    }
    Ok(content)
}

fn entry_from_json_file(
    path: &std::path::Path,
    stem_name: &str,
) -> std::io::Result<serde_json::Value> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        std::io::Error::new(
            e.kind(),
            format!("could not read '{}': {}", path.display(), e),
        )
    })?;
    let json: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("could not parse '{}': {}", path.display(), e),
        )
    })?;

    let asset_type = json.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("'{}' has no `type` field", path.display()),
        )
    })?;

    if asset_type.to_lowercase().replace('_', "") == "buildconfig" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "BuildConfig cannot be added via `concinnity add`; edit {} directly",
                WORLD_JSONL
            ),
        ));
    }

    let args = json
        .get("args")
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));

    let name = json
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or(stem_name)
        .to_string();

    validated_entry(&name, asset_type, args)
}

fn entry_from_inline_json(raw: &str) -> std::io::Result<serde_json::Value> {
    let json: serde_json::Value = serde_json::from_str(raw).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("could not parse inline JSON: {}", e),
        )
    })?;

    if !json.is_object() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "inline JSON must be an object (e.g. '{\"type\": \"Window\"}')",
        ));
    }

    let asset_type = json.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "inline JSON must contain a `type` field",
        )
    })?;

    if asset_type.to_lowercase().replace('_', "") == "buildconfig" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "BuildConfig cannot be added via `concinnity add`; edit {} directly",
                WORLD_JSONL
            ),
        ));
    }

    let name = json
        .get("name")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| asset_type.to_lowercase());

    let args = json.get("args").cloned();

    let req = AssetRequest {
        asset_type: asset_type.to_string(),
        args,
    };
    let def = create_asset_def(&req)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let resolved_args: serde_json::Value = serde_json::from_slice(&def.args_bytes)
        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));

    Ok(serde_json::json!({
        "name": name,
        "type": asset_type,
        "args": resolved_args,
    }))
}

fn resolve_add_target(target: &str) -> std::io::Result<Vec<serde_json::Value>> {
    if is_path_like(target) {
        return entry_from_path(target);
    }

    if ComponentType::parse(target).is_ok() {
        return entry_from_type_name(target).map(|e| vec![e]);
    }

    let as_path = std::path::Path::new(target);
    if as_path.exists() && as_path.is_file() {
        let name = as_path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.replace('.', "_"))
            .unwrap_or_else(|| target.to_string());
        return entry_from_json_file(as_path, &name).map(|e| vec![e]);
    }

    if target.trim_start().starts_with('{') {
        return entry_from_inline_json(target).map(|e| vec![e]);
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidInput,
        format!(
            "could not resolve '{}' as any of:\n  \
             - a file path (no such file found)\n  \
             - a known asset type (use `concinnity list` to see available types)\n  \
             - an inline JSON object (must start with '{{' and contain a `type` field)",
            target
        ),
    ))
}

fn entry_from_type_name(type_str: &str) -> std::io::Result<serde_json::Value> {
    if type_str.to_lowercase().replace('_', "") == "buildconfig" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "BuildConfig cannot be added via `concinnity add`; edit {} directly",
                WORLD_JSONL
            ),
        ));
    }

    let req = AssetRequest {
        asset_type: type_str.to_string(),
        args: None,
    };
    let def = create_asset_def(&req)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
    let args: serde_json::Value = serde_json::from_slice(&def.args_bytes)
        .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));

    let name = type_str.to_lowercase();

    Ok(serde_json::json!({
        "name": name,
        "type": type_str,
        "args": args,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // detect_metal_stages

    #[test]
    fn metal_both_stages() {
        let src = "vertex VertexOut vert_main() {}\nfragment float4 frag_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex", "fragment"]);
    }

    #[test]
    fn metal_vertex_only() {
        let src = "vertex VertexOut vert_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex"]);
    }

    #[test]
    fn metal_fragment_only() {
        let src = "fragment float4 frag_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["fragment"]);
    }

    #[test]
    fn metal_no_qualifiers_defaults_to_vertex() {
        let src = "// helper only\nfloat4 helper() { return float4(1.0); }";
        assert_eq!(detect_metal_stages(src), vec!["vertex"]);
    }

    #[test]
    fn metal_tab_separated_qualifier() {
        let src = "vertex\tVertexOut vert_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex"]);
    }

    #[test]
    fn metal_indented_qualifier_still_detected() {
        // Metal qualifiers can appear with leading whitespace (e.g. inside a namespace-like block)
        let src = "  vertex VertexOut vert_main() {}\n  fragment float4 frag_main() {}";
        assert_eq!(detect_metal_stages(src), vec!["vertex", "fragment"]);
    }

    // detect_wgsl_stages

    #[test]
    fn wgsl_both_stages() {
        let src =
            "@vertex\nfn vs() -> VertexOutput {}\n@fragment\nfn fs() -> @location(0) vec4<f32> {}";
        assert_eq!(detect_wgsl_stages(src), vec!["vertex", "fragment"]);
    }

    #[test]
    fn wgsl_vertex_only() {
        let src = "@vertex fn vs() {}";
        assert_eq!(detect_wgsl_stages(src), vec!["vertex"]);
    }

    #[test]
    fn wgsl_no_attributes_defaults_to_vertex() {
        let src = "fn helper() -> f32 { return 1.0; }";
        assert_eq!(detect_wgsl_stages(src), vec!["vertex"]);
    }

    // stage_abbrev + name format

    #[test]
    fn stage_abbrev_vertex() {
        assert_eq!(stage_abbrev("vertex"), "vert");
    }

    #[test]
    fn stage_abbrev_fragment() {
        assert_eq!(stage_abbrev("fragment"), "frag");
    }

    #[test]
    fn stage_abbrev_passthrough() {
        assert_eq!(stage_abbrev("shadow"), "shadow");
    }

    #[test]
    fn shader_entry_names_metal_both_stages() {
        let stages = detect_metal_stages("vertex VertexOut v() {}\nfragment float4 f() {}");
        let names: Vec<String> = stages
            .iter()
            .map(|&s| format!("{}_{}_{}", "default", "metal", stage_abbrev(s)))
            .collect();
        assert_eq!(names, vec!["default_metal_vert", "default_metal_frag"]);
    }

    #[test]
    fn shader_entry_names_wgsl_both_stages() {
        let stages = detect_wgsl_stages("@vertex fn v() {}\n@fragment fn f() {}");
        let names: Vec<String> = stages
            .iter()
            .map(|&s| format!("{}_{}_{}", "scene", "wgsl", stage_abbrev(s)))
            .collect();
        assert_eq!(names, vec!["scene_wgsl_vert", "scene_wgsl_frag"]);
    }

    // font stem naming

    #[test]
    fn font_name_uses_stem_only() {
        let path = std::path::Path::new("fonts/JetBrainsMono-Regular.ttf");
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.replace('.', "_"))
            .unwrap_or_default();
        // stem should not include the .ttf extension
        assert_eq!(stem, "JetBrainsMono-Regular");
        assert!(!stem.contains("ttf"));
    }

    // unknown extension

    #[test]
    fn unknown_extension_errors() {
        // Use a path that won't exist on disk so it goes through entry_from_path
        let result = entry_from_path("assets/shader.xyz");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains(".xyz"));
    }

    // stages_from_flags edge cases

    #[test]
    fn stages_from_flags_neither() {
        assert_eq!(stages_from_flags(false, false), vec!["vertex"]);
    }

    #[test]
    fn stages_from_flags_both() {
        assert_eq!(stages_from_flags(true, true), vec!["vertex", "fragment"]);
    }

    #[test]
    fn stages_from_flags_fragment_only() {
        assert_eq!(stages_from_flags(false, true), vec!["fragment"]);
    }

    // target_bootstrap_kind

    #[test]
    fn target_bootstrap_kind_glb_is_scene() {
        assert!(matches!(
            target_bootstrap_kind("models/scene.glb"),
            BootstrapKind::Scene
        ));
        assert!(matches!(
            target_bootstrap_kind("scene.GLB"),
            BootstrapKind::Scene
        ));
    }

    #[test]
    fn target_bootstrap_kind_text_is_text() {
        assert!(matches!(
            target_bootstrap_kind("notes.txt"),
            BootstrapKind::Text
        ));
        assert!(matches!(
            target_bootstrap_kind("README.MD"),
            BootstrapKind::Text
        ));
    }

    #[test]
    fn target_bootstrap_kind_other_is_none() {
        assert!(matches!(
            target_bootstrap_kind("Logger"),
            BootstrapKind::None
        ));
        assert!(matches!(
            target_bootstrap_kind("shader.vert"),
            BootstrapKind::None
        ));
        assert!(matches!(
            target_bootstrap_kind("font.ttf"),
            BootstrapKind::None
        ));
    }

    // jsonl_has_renderer_trigger

    #[test]
    fn renderer_trigger_matches_graphics_config() {
        let jsonl = r#"{"name":"gc","type":"GraphicsConfig","args":{}}"#;
        assert!(jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_matches_text_label() {
        let jsonl = r#"{"name":"lbl","type":"TextLabel","args":{}}"#;
        assert!(jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_matches_window() {
        let jsonl = r#"{"name":"win","type":"Window","args":{}}"#;
        assert!(jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_absent_for_render_data_only_world() {
        // Exactly the shape that produced the no-window bug: textures /
        // materials / meshes / models / camera but no renderer-trigger.
        let jsonl = concat!(
            r#"{"name":"tex","type":"Texture","args":{}}"#,
            "\n",
            r#"{"name":"mat","type":"Material","args":{}}"#,
            "\n",
            r#"{"name":"mesh","type":"Mesh","args":{}}"#,
            "\n",
            r#"{"name":"model","type":"Model","args":{}}"#,
            "\n",
            r#"{"name":"prop","type":"Prop","args":{}}"#,
            "\n",
            r#"{"name":"cam","type":"Camera3D","args":{}}"#,
            "\n",
        );
        assert!(!jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_skips_malformed_lines() {
        let jsonl = concat!(
            "garbage line\n",
            r#"{"name":"tex","type":"Texture","args":{}}"#,
            "\n",
        );
        assert!(!jsonl_has_renderer_trigger(jsonl));
    }

    #[test]
    fn renderer_trigger_handles_empty_jsonl() {
        assert!(!jsonl_has_renderer_trigger(""));
    }

    // scaffold_to_inject

    #[test]
    fn scaffold_to_inject_writes_nothing_without_a_template() {
        // Existing world with renderer-less assets + a .glb target: the
        // renderer stack is injected at build time, so no lines are written.
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let world = dir.join("world.jsonl");
        std::fs::write(
            &world,
            concat!(
                r#"{"name":"tex","type":"Texture","args":{}}"#,
                "\n",
                r#"{"name":"cam","type":"Camera3D","args":{}}"#,
                "\n",
            ),
        )
        .unwrap();

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", None).unwrap();
        assert!(scaffold.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scaffold_to_inject_returns_empty_when_world_has_graphics_config() {
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let world = dir.join("world.jsonl");
        std::fs::write(
            &world,
            r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
        )
        .unwrap();

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", None).unwrap();
        assert!(scaffold.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scaffold_to_inject_allows_missing_world_with_glb() {
        // No file present, target is `.glb`: the caller is expected to create
        // the file; the renderer stack comes from build-time injection, so no
        // template entries are needed.
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        let world = dir.join("world.jsonl");

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", None).unwrap();
        assert!(scaffold.is_empty());
    }

    #[test]
    fn scaffold_to_inject_errors_for_missing_world_with_non_scene_target() {
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        let world = dir.join("world.jsonl");

        let err = scaffold_to_inject(world.to_str().unwrap(), "Logger", None)
            .expect_err("missing world + non-scene target must error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn scaffold_to_inject_returns_empty_for_non_scene_target_into_existing_world() {
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let world = dir.join("world.jsonl");
        std::fs::write(&world, "").unwrap();

        // Non-scene target shouldn't trigger scaffolding even when the world
        // has no renderer: we don't try to guess intent for shaders/fonts.
        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "Logger", None).unwrap();
        assert!(scaffold.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // template dispatch

    #[test]
    fn scaffold_to_inject_uses_showcase_template_when_named() {
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        let world = dir.join("world.jsonl");

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", Some("showcase"))
            .expect("showcase template should apply for renderer-less glb add");
        assert_eq!(
            scaffold.len(),
            super::super::sources::glb::template_showcase().len(),
            "expected the showcase template entries"
        );
        assert!(!scaffold.is_empty());
    }

    #[test]
    fn scaffold_to_inject_errors_on_unknown_template() {
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        let world = dir.join("world.jsonl");

        let err = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", Some("nope"))
            .expect_err("unknown template name must surface as InvalidInput");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("nope"),
            "error should name the bad template: {err}"
        );
    }

    #[test]
    fn scaffold_to_inject_rejects_unknown_template_even_when_no_scaffold_would_fire() {
        // An unknown template name is a typo, full stop. We don't want the
        // user to silently get nothing when they intended to ask for a
        // template.
        let dir =
            std::env::temp_dir().join(format!("cn_add_test_{}_{}", std::process::id(), line!()));
        std::fs::create_dir_all(&dir).unwrap();
        let world = dir.join("world.jsonl");
        // Existing world with a renderer trigger: scaffolding wouldn't fire.
        std::fs::write(
            &world,
            r#"{"name":"gfx","type":"GraphicsConfig","args":{}}"#,
        )
        .unwrap();

        let err = scaffold_to_inject(world.to_str().unwrap(), "scene.glb", Some("nope"))
            .expect_err("unknown template should fail fast regardless of scaffold path");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let _ = std::fs::remove_dir_all(&dir);
    }

    // text-file targets

    fn text_test_dir(line: u32) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("cn_add_text_test_{}_{}", std::process::id(), line));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn text_file_becomes_text_label_with_content() {
        let dir = text_test_dir(line!());
        let path = dir.join("greeting.txt");
        std::fs::write(&path, "Hello, world!").unwrap();

        let entries = entry_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry["type"], "TextLabel");
        assert_eq!(entry["name"], "greeting");
        assert_eq!(entry["args"]["content"], "Hello, world!");
        // Mirrors `cn init`: short labels render centered by default.
        assert_eq!(entry["args"]["centered"], serde_json::json!(true));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_file_strips_single_trailing_newline() {
        let dir = text_test_dir(line!());
        let lf = dir.join("lf.txt");
        std::fs::write(&lf, "line one\nline two\n").unwrap();
        let crlf = dir.join("crlf.md");
        std::fs::write(&crlf, "line one\r\nline two\r\n").unwrap();

        let lf_entry = &entry_from_path(lf.to_str().unwrap()).unwrap()[0];
        assert_eq!(lf_entry["args"]["content"], "line one\nline two");
        let crlf_entry = &entry_from_path(crlf.to_str().unwrap()).unwrap()[0];
        assert_eq!(crlf_entry["args"]["content"], "line one\r\nline two");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn text_file_over_size_cap_errors() {
        let dir = text_test_dir(line!());
        let path = dir.join("huge.txt");
        let blob = "a".repeat(TEXT_LABEL_MAX_BYTES + 1);
        std::fs::write(&path, blob).unwrap();

        let err = entry_from_path(path.to_str().unwrap())
            .expect_err("oversized text file must fail rather than silently truncate");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("capped"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scaffold_to_inject_empty_for_missing_world_with_text_target() {
        // Text targets bootstrap a missing world via the TextLabel itself:
        // no separate scaffold needed, and the missing file must not error.
        let dir = text_test_dir(line!());
        let world = dir.join("world.jsonl");

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "notes.txt", None).unwrap();
        assert!(scaffold.is_empty(), "text target should emit no scaffold");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn scaffold_to_inject_empty_for_renderer_less_world_with_text_target() {
        let dir = text_test_dir(line!());
        let world = dir.join("world.jsonl");
        std::fs::write(&world, r#"{"name":"tex","type":"Texture","args":{}}"#).unwrap();

        let scaffold = scaffold_to_inject(world.to_str().unwrap(), "notes.md", None).unwrap();
        assert!(
            scaffold.is_empty(),
            "text target never injects GLB scaffold"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // try_refresh_text_label

    #[test]
    fn try_refresh_text_label_updates_content_in_place() {
        // Existing TextLabel was hand-edited: y, color, scale, centered all
        // diverged from defaults. A refresh should touch only `content`.
        let mut assets = vec![serde_json::json!({
            "name": "greeting",
            "type": "TextLabel",
            "args": {
                "content": "Old text",
                "font": "Questrial-Regular",
                "x": 10.0,
                "y": 200.0,
                "color": [0.2, 0.8, 0.4],
                "scale": 1.5,
                "centered": false,
                "background": [0.0, 0.0, 0.0, 0.0],
                "padding": 0.0,
                "visible": true,
                "view": null,
            }
        })];
        let entries = vec![serde_json::json!({
            "name": "greeting",
            "type": "TextLabel",
            "args": {
                "content": "New text",
                "centered": true,
            }
        })];

        let refreshed = try_refresh_text_label(&mut assets, &entries);
        assert_eq!(refreshed.as_deref(), Some("greeting"));
        let args = assets[0]["args"].as_object().unwrap();
        assert_eq!(args["content"], "New text");
        // Hand edits survive: only `content` was copied across.
        assert_eq!(args["y"], 200.0);
        assert_eq!(args["color"], serde_json::json!([0.2, 0.8, 0.4]));
        assert_eq!(args["scale"], 1.5);
        assert_eq!(args["centered"], false);
    }

    #[test]
    fn try_refresh_text_label_skips_non_textlabel_new_entry() {
        let mut assets = vec![serde_json::json!({
            "name": "thing",
            "type": "TextLabel",
            "args": {"content": "old"}
        })];
        let entries = vec![serde_json::json!({
            "name": "thing",
            "type": "Font",
            "args": {"path": "x.ttf"}
        })];
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_refresh_text_label_skips_when_existing_is_different_type() {
        let mut assets = vec![serde_json::json!({
            "name": "thing",
            "type": "Font",
            "args": {"path": "x.ttf"}
        })];
        let entries = vec![serde_json::json!({
            "name": "thing",
            "type": "TextLabel",
            "args": {"content": "hi"}
        })];
        // Same name, different existing type: refresh shouldn't fire; caller
        // falls through to the duplicate-name error.
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_refresh_text_label_skips_when_name_misses() {
        let mut assets = vec![serde_json::json!({
            "name": "greeting",
            "type": "TextLabel",
            "args": {"content": "old"}
        })];
        let entries = vec![serde_json::json!({
            "name": "caption",
            "type": "TextLabel",
            "args": {"content": "new"}
        })];
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn try_refresh_text_label_skips_multi_entry() {
        // A fan-out target (e.g. a metal shader producing vert+frag) must
        // never trigger the in-place refresh path.
        let mut assets = vec![serde_json::json!({
            "name": "label",
            "type": "TextLabel",
            "args": {"content": "old"}
        })];
        let entries = vec![
            serde_json::json!({"name": "label", "type": "TextLabel", "args": {"content": "new"}}),
            serde_json::json!({"name": "other", "type": "TextLabel", "args": {"content": "other"}}),
        ];
        assert!(try_refresh_text_label(&mut assets, &entries).is_none());
    }

    #[test]
    fn scaffold_to_inject_rejects_template_with_text_target() {
        let dir = text_test_dir(line!());
        let world = dir.join("world.jsonl");

        let err = scaffold_to_inject(world.to_str().unwrap(), "notes.txt", Some("showcase"))
            .expect_err("--template should only apply to scene targets");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
