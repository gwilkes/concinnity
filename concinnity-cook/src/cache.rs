// Content-addressed cache for compiled asset payloads.
//
// Some assets are expensive to compile -- the EnvironmentMap IBL convolution
// alone is hundreds of millions of float ops per build. The compiled payload
// is, however, a deterministic function of a small set of inputs: the cache
// format version, the component discriminant, the asset's args JSON, and the
// contents of any source files the args reference. This module hashes those
// inputs into a key and stores the compiled bytes under `.concinnity/cache/`.
// A later build that produces the same key reuses the cached payload instead
// of recompiling.
//
// Every operation here is best-effort: a cache miss, a read error, or a write
// error all fall back to a normal compile, so the cache can never break or
// corrupt a build.

use crate::asset::BuildCtx;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

// SHA-256 a source file's contents, memoized by path within the process. A
// single build can reference one large source file from hundreds of assets
// (e.g. every Mesh imported from one `.fbx`), and hashing it once per asset
// dominates the build; the memo reads + hashes each unique file once. Keyed by
// (mtime, len) so a file edited between in-process rebuilds (the `cn debug`
// hot-reload path) is re-hashed rather than served stale.
fn file_content_hash(path: &str) -> Option<[u8; 32]> {
    // path -> (mtime_nanos, len, content hash)
    type HashMemo = Mutex<HashMap<String, (u64, u64, [u8; 32])>>;
    static MEMO: OnceLock<HashMemo> = OnceLock::new();
    let meta = std::fs::metadata(path).ok()?;
    let len = meta.len();
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&(m, l, h)) = memo.lock().unwrap().get(path)
        && m == mtime
        && l == len
    {
        return Some(h);
    }
    let bytes = std::fs::read(path).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash: [u8; 32] = hasher.finalize().into();
    memo.lock()
        .unwrap()
        .insert(path.to_string(), (mtime, len, hash));
    Some(hash)
}

// Bump this whenever a compile path's output changes without a corresponding
// change to asset args -- e.g. a convolution algorithm tweak, a payload format
// revision, or a change to a default sample count. A bump changes every key
// and so invalidates every existing cache entry.
//
// 4: font payload gained a supersample factor in its header (build::font).
// 5: EnvironmentMap default irradiance_face_size changed 32 -> 8
//    (build::environment_map), so worlds that omit it bake a different cube.
//    (The counter was later reset to 1 with the postcard/blob migration.)
// 2: EnvironmentMap glossy reflection mips gained a firefly clamp
//    (prefilter_clamp, default 12); worlds that omit the arg still bake dimmer
//    hot texels, so every cached envmap must rebake (build::environment_map).
const CACHE_FORMAT_VERSION: u32 = 2;

const CACHE_DIR: &str = ".concinnity/cache";

// Compute the cache key for one compiled asset. The key folds in the cache
// format version, the active backend's shader platform, the component
// discriminant, the args JSON, and a content hash of every source file the
// args reference (see `referenced_files`). `extra_source_files` adds further
// on-disk paths the asset's `BuildAsset::compile_payload` reads that the
// generic JSON-string walk would miss (e.g. an `SdfVolume` fragment shader
// resolved from the source-tree `assets/` directory). The platform is part of
// the key because `BuildAsset::compile_payload` can short-circuit differently
// per backend (e.g. a ShaderStage with no `glsl` source emits empty bytes only
// under the Vulkan backend, which then compiles its inline GLSL), so an entry
// compiled on one platform must not be returned to a build on another.
pub fn payload_key(
    discriminant: u8,
    args: &serde_json::Value,
    ctx: &BuildCtx<'_>,
    extra_source_files: &[String],
) -> String {
    let mut files = referenced_files(args, ctx);
    for path in extra_source_files {
        if let Some(h) = file_content_hash(path) {
            files.push((path.clone(), h));
        }
    }
    let base = key_from_parts(discriminant, args, &files);
    format!(
        "{}-{}",
        concinnity_core::build::Platform::current().key(),
        base
    )
}

// Bump when the SceneImport expansion output shape changes (a new generated
// asset field, a renamed arg, a different naming scheme) so existing cached
// entry lists are invalidated. v2: glass materials are detected (by FBX
// transparency / name) and emitted smooth + translucent.
const EXPAND_FORMAT_VERSION: u32 = 2;

// Cache key for a SceneImport expansion. The generated asset-entry list is a
// deterministic function of the source file's contents, the import options,
// and the expansion format version, so editing the source file or changing an
// option busts the entry. Unlike `payload_key` this is platform-independent:
// the entries are plain JSON with no per-backend branching. `load` / `store`
// are shared with the payload cache (same `.concinnity/cache/` directory); the
// `expand-` prefix keeps the two key spaces visibly distinct.
pub fn expand_key(source: &str, args: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    hasher.update(EXPAND_FORMAT_VERSION.to_le_bytes());
    if let Some(h) = file_content_hash(source) {
        hasher.update(h);
    }
    let args_bytes = serde_json::to_vec(args).unwrap_or_default();
    hasher.update((args_bytes.len() as u64).to_le_bytes());
    hasher.update(&args_bytes);
    format!("expand-{:x}", hasher.finalize())
}

// Read a cached payload for `key`, if one is present.
pub fn load(key: &str) -> Option<Vec<u8>> {
    // Disabled under `cargo test` so the suite neither creates stray cache
    // directories nor lets a stale entry mask a change to a compile path.
    if cfg!(test) {
        return None;
    }
    std::fs::read(Path::new(CACHE_DIR).join(key)).ok()
}

// Store a compiled payload under `key`. Best-effort: any error is ignored.
// The bytes are written to a temp file and renamed into place so a concurrent
// reader never observes a half-written entry.
pub fn store(key: &str, bytes: &[u8]) {
    if cfg!(test) {
        return;
    }
    store_in(Path::new(CACHE_DIR), key, bytes);
}

// Hash the fixed parts of a key. Split out from `payload_key` so tests can
// supply file contents directly without touching the filesystem.
fn key_from_parts(
    discriminant: u8,
    args: &serde_json::Value,
    files: &[(String, [u8; 32])],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_FORMAT_VERSION.to_le_bytes());
    hasher.update([discriminant]);

    let args_bytes = serde_json::to_vec(args).unwrap_or_default();
    hasher.update((args_bytes.len() as u64).to_le_bytes());
    hasher.update(&args_bytes);

    // Sort so the key does not depend on JSON traversal order.
    let mut files = files.to_vec();
    files.sort();
    for (path, content_hash) in &files {
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update(content_hash);
    }
    format!("{:x}", hasher.finalize())
}

// Collect (path, content-hash) for every source file the args reference.
// Walks the args JSON for string leaves and resolves each one to a file using
// the same lookup rules the asset compilers use: a bare filename is searched
// under `.concinnity/assets/`, a relative or absolute path is used directly,
// and `artifacts_dir` is consulted when set. Strings that do not resolve to a
// file (asset names, generator keywords, colors) contribute nothing.
//
// A string that names a built-in shader is a special case: the source is
// embedded in the binary rather than living at a filesystem path, and built-ins
// always win over a disk copy at compile time (see shader::read_shader_source).
// Such a string is hashed from its embedded source under a `builtin:` key so
// that editing a shipped shader and rebuilding the binary busts the cache.
fn referenced_files(args: &serde_json::Value, ctx: &BuildCtx<'_>) -> Vec<(String, [u8; 32])> {
    let mut strings = Vec::new();
    collect_strings(args, &mut strings);

    let mut out = Vec::new();
    for s in strings {
        if let Some(src) = concinnity_core::build::shader::builtin_shader_source(&s) {
            let bare = Path::new(&s)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&s);
            let mut h = Sha256::new();
            h.update(src.as_bytes());
            out.push((format!("builtin:{bare}"), h.finalize().into()));
            continue;
        }
        let Some(path) = resolve_source(&s, ctx) else {
            continue;
        };
        if let Some(h) = file_content_hash(&path) {
            out.push((path, h));
        }
    }
    out
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|e| collect_strings(e, out)),
        serde_json::Value::Object(m) => m.values().for_each(|e| collect_strings(e, out)),
        _ => {}
    }
}

// Resolve a single args string to an existing file path, or None. Only strings
// that look like filenames (have an extension or a path separator) are probed,
// so the common case of short keyword args costs nothing.
fn resolve_source(s: &str, ctx: &BuildCtx<'_>) -> Option<String> {
    let looks_like_file = s.contains('/') || s.contains('\\') || Path::new(s).extension().is_some();
    if !looks_like_file {
        return None;
    }
    // Direct path (absolute, or relative to the build working directory).
    if Path::new(s).is_file() {
        return Some(s.to_string());
    }
    // Bare filename searched recursively under .concinnity/assets/.
    if let Some(p) = crate::world::preset::find_in_assets(s) {
        return Some(p);
    }
    // Account artifact directory, when the build supplied one.
    if let Some(dir) = ctx.artifacts_dir {
        let p = format!("{dir}/{s}");
        if Path::new(&p).is_file() {
            return Some(p);
        }
    }
    None
}

fn store_in(dir: &Path, key: &str, bytes: &[u8]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = dir.join(format!("{key}.{}.tmp", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() {
        let _ = std::fs::rename(&tmp, dir.join(key));
    }
}

#[cfg(test)]
fn load_in(dir: &Path, key: &str) -> Option<Vec<u8>> {
    std::fs::read(dir.join(key)).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ctx() -> BuildCtx<'static> {
        BuildCtx {
            name: "test",
            artifacts_dir: None,
            all_assets: &[],
        }
    }

    #[test]
    fn key_is_stable_for_same_inputs() {
        let a = json!({"generator": "box", "half_extents": [1, 2, 3]});
        assert_eq!(key_from_parts(7, &a, &[]), key_from_parts(7, &a, &[]));
    }

    #[test]
    fn key_changes_with_args_discriminant_and_files() {
        let a = json!({"generator": "box"});
        let b = json!({"generator": "sphere"});
        let base = key_from_parts(1, &a, &[]);
        assert_ne!(base, key_from_parts(1, &b, &[]), "args must affect the key");
        assert_ne!(
            base,
            key_from_parts(2, &a, &[]),
            "discriminant must affect the key"
        );
        assert_ne!(
            base,
            key_from_parts(1, &a, &[("x.hdr".into(), [9u8; 32])]),
            "a referenced file must affect the key"
        );
    }

    #[test]
    fn key_ignores_referenced_file_order() {
        let a = json!({});
        let f1 = ("a.hdr".to_string(), [1u8; 32]);
        let f2 = ("b.hdr".to_string(), [2u8; 32]);
        assert_eq!(
            key_from_parts(0, &a, &[f1.clone(), f2.clone()]),
            key_from_parts(0, &a, &[f2, f1]),
        );
    }

    #[test]
    fn key_tracks_referenced_file_contents() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("env.hdr");
        std::fs::write(&file, b"first").unwrap();
        let args = json!({ "source": file.to_str().unwrap() });

        let before = payload_key(3, &args, &ctx(), &[]);
        std::fs::write(&file, b"second").unwrap();
        let after = payload_key(3, &args, &ctx(), &[]);
        assert_ne!(
            before, after,
            "key must change when a referenced file changes"
        );
    }

    #[test]
    fn key_tracks_extra_source_file_contents() {
        // Files whose paths the generic JSON-string walk can't resolve (e.g.
        // an SdfVolume `fragment_shader` resolved through the source-tree
        // `assets/` dir) must still bust the cache when their contents change.
        // The asset's `BuildAsset::source_files` override hands those paths to
        // `payload_key` via `extra_source_files`.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("shader.metal");
        std::fs::write(&file, b"void shade() {}").unwrap();
        let path = file.to_str().unwrap().to_string();
        // Args reference the file by a bare token the cache cannot resolve on
        // its own (no extension, no separator), so only `extra_source_files`
        // can contribute the content hash.
        let args = json!({ "fragment_shader": "chrome" });

        let before = payload_key(11, &args, &ctx(), std::slice::from_ref(&path));
        std::fs::write(&file, b"void shade(float) {}").unwrap();
        let after = payload_key(11, &args, &ctx(), std::slice::from_ref(&path));
        assert_ne!(
            before, after,
            "an extra source file's contents must affect the key"
        );
    }

    #[test]
    fn key_ignores_unreadable_extra_source_file() {
        // A path that does not exist is silently dropped (best-effort, matching
        // the rest of the cache layer). A missing file produces the same key
        // as no extra files at all.
        let args = json!({ "fragment_shader": "chrome" });
        let missing = "/definitely/not/a/real/path.metal".to_string();
        assert_eq!(
            payload_key(11, &args, &ctx(), &[]),
            payload_key(11, &args, &ctx(), std::slice::from_ref(&missing)),
        );
    }

    #[test]
    fn non_file_strings_are_not_resolved() {
        // "box" has neither an extension nor a separator -> never probed.
        assert!(referenced_files(&json!({"generator": "box"}), &ctx()).is_empty());
    }

    #[test]
    fn builtin_shader_content_is_folded_into_key() {
        use concinnity_core::build::shader::builtin_shader_source;

        // A built-in shader referenced by bare filename has no filesystem path,
        // but its embedded source must still contribute to the key.
        let args = json!({ "sources": { "metal": "default.metal" } });
        let files = referenced_files(&args, &ctx());

        let src = builtin_shader_source("default.metal").expect("default.metal is built in");
        let mut h = Sha256::new();
        h.update(src.as_bytes());
        let expected: [u8; 32] = h.finalize().into();
        assert_eq!(
            files,
            vec![("builtin:default.metal".to_string(), expected)],
            "a built-in shader reference must contribute its embedded source hash",
        );

        // The key is a function of that hash, so any edit to the shader source
        // changes the key. A perturbed hash stands in for an edited shader.
        let real_key = key_from_parts(5, &args, &files);
        let edited = vec![("builtin:default.metal".to_string(), [0u8; 32])];
        assert_ne!(
            real_key,
            key_from_parts(5, &args, &edited),
            "editing a built-in shader source must change the key",
        );
    }

    #[test]
    fn builtin_shader_directory_prefix_resolves_to_bare_key() {
        // Built-ins match by bare filename, so a leading directory must not
        // produce a distinct key entry.
        let bare = referenced_files(&json!("default.metal"), &ctx());
        let prefixed = referenced_files(&json!("default_shader/default.metal"), &ctx());
        assert_eq!(bare, prefixed);
        assert_eq!(bare.len(), 1);
    }

    #[test]
    fn expand_key_tracks_source_contents_and_args() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("scene.fbx");
        std::fs::write(&file, b"first").unwrap();
        let src = file.to_str().unwrap();
        let args = json!({ "prefix": "scn", "texture_max_size": 512 });

        let base = expand_key(src, &args);
        // Stable for identical inputs.
        assert_eq!(base, expand_key(src, &args));
        // Changing an option busts the key.
        assert_ne!(
            base,
            expand_key(src, &json!({ "prefix": "scn", "texture_max_size": 256 }))
        );
        // Editing the source file busts the key.
        std::fs::write(&file, b"second").unwrap();
        assert_ne!(base, expand_key(src, &args));
        // Expansion keys are namespaced apart from payload keys.
        assert!(base.starts_with("expand-"));
    }

    #[test]
    fn store_then_load_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        store_in(dir.path(), "abc123", b"payload bytes");
        assert_eq!(
            load_in(dir.path(), "abc123").as_deref(),
            Some(&b"payload bytes"[..])
        );
        assert_eq!(load_in(dir.path(), "missing"), None);
    }
}
