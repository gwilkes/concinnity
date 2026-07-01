// src/assets/shader_stage.rs

// **CRITICAL: `packed_float3` in light structs**: In MSL constant buffers `float3`
// has size=16, but Rust `[f32; 3]` (what the engine sends) has size=12. If you
// declare `DirectionalLightData` or `PointLightData` with plain `float3`, the color
// field will read as zeros (black light) and `num_directional` will read garbage,
// causing ambient-only rendering. Always use `packed_float3` for vector fields in
// these structs:

use crate::ecs::{AssetOrigin, AssetPayload, Component, PayloadLocator};

/// Which stage in the render pipeline this shader drives.
///
/// `VertexInstanced` is the GPU-instanced sibling of `Vertex`, reading per-
/// instance model matrices instead of a per-draw transform. Required for any
/// world containing [InstancedProp](#instancedprop) components; otherwise
/// unused.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ShaderKind {
    #[default]
    Vertex,
    Fragment,
    #[serde(rename = "vertex_instanced", alias = "vertexinstanced")]
    VertexInstanced,
}

impl ShaderKind {
    /// The compile kind string expected by ShaderCompileArgs.
    pub fn compile_kind(&self) -> &'static str {
        match self {
            ShaderKind::Vertex | ShaderKind::VertexInstanced => "vertex",
            ShaderKind::Fragment => "fragment",
        }
    }
}

/// Declares a compiled shader stage.
///
/// **Vertex and fragment stages are required for anything to render.** The
/// shadow pass is engine-internal (no `ShaderStage` of its own); enable or
/// size it with `shadow_map_size` in [GraphicsConfig](#graphicsconfig).
///
/// Provide either `source` (single platform) or `sources` (multi-platform). When both are
/// present, `sources` takes priority for the current platform.
///
/// **Platform keys:** `"metal"` (macOS), `"hlsl"` (Windows), `"glsl"` (Linux/Vulkan).
///
/// **Bundled shaders:**
///
/// - `"default.metal"` / `"default_vert.hlsl"` / `"default_frag.hlsl"`: standard diffuse/specular lighting.
///
/// The engine-internal shadow map covers a ±20 m world-space region centred at
/// the origin with 80 m depth. For larger scenes, increase `shadow_map_size` in
/// [GraphicsConfig](#graphicsconfig) to maintain resolution.
///
/// ```jsonl
/// // Multi-platform standard scene:
/// {"name":"vert","type":"ShaderStage","args":{"kind":"vertex","sources":{"metal":"default.metal","hlsl":"default_vert.hlsl"}}}
/// {"name":"frag","type":"ShaderStage","args":{"kind":"fragment","sources":{"metal":"default.metal","hlsl":"default_frag.hlsl"}}}
///
/// // Single-platform (macOS only):
/// {"name":"vert","type":"ShaderStage","args":{"kind":"vertex","source":"default.metal"}}
/// ```
///
/// **Custom shader vertex layout**: the engine always supplies vertices with 5
/// attributes at a fixed 56-byte stride. Any custom `.metal` shader **must** declare
/// `struct Vertex` exactly as shown below: wrong attribute indices cause tangent
/// data to be read as vertex colour, producing red/green/blue geometry:
///
/// ```metal
/// struct Vertex {
///     float3 pos     [[attribute(0)]];  // offset  0
///     float3 normal  [[attribute(1)]];  // offset 12
///     float3 tangent [[attribute(2)]];  // offset 24
///     float3 color   [[attribute(3)]];  // offset 36
///     float2 uv      [[attribute(4)]];  // offset 48
/// };
/// ```
///
/// Buffer and texture bindings that must match:
///
/// ```metal
/// struct DirectionalLightData {
///     packed_float3 direction;
///     float         intensity;
///     packed_float3 color;
///     float         _pad;
/// };
///
/// struct PointLightData {
///     packed_float3 position;
///     float         range;
///     packed_float3 color;
///     float         intensity;
/// };
///
/// struct ShadowUniforms {
///     float4x4 light_vp;
/// };
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShaderStage {
    /// Which stage this shader drives.
    pub kind: ShaderKind,
    /// Single-platform source path; used when `sources` is absent or lacks the current platform key.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
    /// Per-platform source paths keyed by `"metal"`, `"hlsl"`, or `"glsl"`. Takes priority over `source`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<std::collections::HashMap<String, String>>,
    /// Injected at load time from BlobAssetDef::payload.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for ShaderStage {
    fn default() -> Self {
        let mut sources = std::collections::HashMap::new();
        sources.insert("metal".to_string(), "default.metal".to_string());
        sources.insert("hlsl".to_string(), "default_vert.hlsl".to_string());
        Self {
            kind: ShaderKind::Vertex,
            source: String::new(),
            sources: Some(sources),
            locator: None,
        }
    }
}

impl ShaderStage {
    /// Resolves the source filename for the current build platform from this
    /// stage's declared `source` / `sources`. Mirrors the build-time
    /// `SourceBacked::source_path` selection so the hot-reload subsystem picks
    /// the same per-platform source the build path read at compile time.
    /// Returns `None` when no current-platform source is declared (e.g. a
    /// stage that only declares `glsl` running on the Metal backend, which
    /// loads the embedded GLSL fallback at init and has no on-disk file to
    /// hot-reload).
    pub fn current_platform_source(&self) -> Option<String> {
        use crate::build::SourceBacked;
        let args = serde_json::to_value(self).ok()?;
        <Self as SourceBacked>::source_path(&args, crate::build::Platform::current())
    }
}

/// Apply the same bare-filename fallback the build pipeline runs when
/// compiling a `ShaderStage`'s shader source to a raw
/// shader source string: bare filenames are searched in
/// `.concinnity/assets/` (recursively), then fall back to a direct path
/// under the same directory. Paths that already contain a directory
/// component are returned unchanged. Used by the asset hot-reload subsystem
/// to convert a `ShaderStage`'s declared source into the on-disk path the
/// watcher subscribes to + the runtime recompile reads.
pub fn resolve_runtime_source_path(raw: &str) -> String {
    let p = std::path::Path::new(raw);
    if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
        if let Some(path) = crate::world::preset::find_in_assets(raw) {
            return path;
        }
        return crate::paths::assets_dir()
            .join(raw)
            .to_string_lossy()
            .into_owned();
    }
    raw.to_string()
}

impl Component for ShaderStage {
    const NAME: &'static str = "ShaderStage";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn to_args(&self) -> Self {
        self.clone()
    }
    fn from_args(args: Self) -> Self {
        args
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }
}

// Resolve a raw per-platform source string to the on-disk path the build
// will read. A bare filename is looked up recursively under
// `.concinnity/assets/` first, then under `<artifacts_dir>` when set, then
// directly under `.concinnity/assets/<raw>`. A path with a directory
// component is used verbatim. Mirrors the resolution `compile_payload`
// applies; built-in shaders short-circuit upstream and never reach this.
pub fn resolve_source_path_for(raw: &str, ctx: &crate::build::BuildCtx<'_>) -> String {
    let p = std::path::Path::new(raw);
    if p.parent().map(|d| d.as_os_str().is_empty()).unwrap_or(true) {
        if let Some(path) = crate::world::preset::find_in_assets(raw) {
            return path;
        }
        if let Some(dir) = ctx.artifacts_dir {
            let artifact_path = format!("{dir}/{raw}");
            if std::path::Path::new(&artifact_path).exists() {
                return artifact_path;
            }
        }
        return crate::paths::assets_dir()
            .join(raw)
            .to_string_lossy()
            .into_owned();
    }
    raw.to_string()
}

/// Validate ShaderStage args without compiling.
pub fn check(args: &serde_json::Value) -> Result<(), String> {
    if resolve_source_from_args(args).is_none() {
        // On Linux/Vulkan, missing sources are non-fatal: the runtime falls
        // back to built-in GLSL. See `compile_payload` for the matching carve-out.
        if platform_key() == "glsl" {
            return Ok(());
        }
        return Err(format!(
            "ShaderStage requires a `source` or a `sources` entry for platform \"{}\"",
            platform_key()
        ));
    }
    Ok(())
}

/// Returns the platform key used to look up entries in the `sources` map.
pub fn platform_key() -> &'static str {
    crate::build::Platform::current().key()
}

/// Resolves the shader source filename for the current platform from raw asset args.
///
/// Convenience wrapper over the `SourceBacked` impl that resolves against
/// `Platform::current()`. New code should prefer
/// `<ShaderStage as SourceBacked>::source_path(args, platform)` directly.
pub fn resolve_source_from_args(args: &serde_json::Value) -> Option<String> {
    use crate::build::SourceBacked;
    <ShaderStage as SourceBacked>::source_path(args, crate::build::Platform::current())
}

// True when this stage declares at least one source and every declared source
// (in `sources` and `source`) is an engine built-in shader. The bundled
// default shader set declares only `metal` + `hlsl` built-ins and no `glsl`,
// so on the Vulkan/GLSL backend it resolves to no source and renders via the
// backend's inline GLSL by design -- not a user mistake. A custom stage that
// merely forgot its `glsl` variant has at least one non-built-in source and is
// not covered, so the missing-source path still flags it.
pub fn declares_only_builtin_sources(args: &serde_json::Value) -> bool {
    use crate::build::shader::builtin_shader_source;
    let mut saw_any = false;
    let mut check = |name: &str| {
        if name.is_empty() {
            return true;
        }
        saw_any = true;
        builtin_shader_source(name).is_some()
    };
    if let Some(obj) = args.get("sources").and_then(|v| v.as_object()) {
        for v in obj.values() {
            if let Some(s) = v.as_str()
                && !check(s)
            {
                return false;
            }
        }
    }
    if let Some(s) = args.get("source").and_then(|v| v.as_str())
        && !check(s)
    {
        return false;
    }
    saw_any
}

impl crate::build::SourceBacked for ShaderStage {
    fn source_path(args: &serde_json::Value, platform: crate::build::Platform) -> Option<String> {
        if let Some(obj) = args.get("sources").and_then(|v| v.as_object())
            && let Some(src) = obj.get(platform.key()).and_then(|v| v.as_str())
        {
            return Some(src.to_string());
        }
        let src = args
            .get("source")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())?;
        let ext = std::path::Path::new(src)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        if platform.accepts_ext(ext) {
            Some(src.to_string())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod builtin_source_tests {
    use super::declares_only_builtin_sources;

    #[test]
    fn bundled_default_set_is_all_builtin() {
        // The vertex + fragment defaults the GraphicsConfig companion injects
        // declare only built-in metal/hlsl sources, so the GLSL fallback is
        // expected and must not be flagged.
        let vert = serde_json::json!({
            "kind": "vertex",
            "sources": {"metal": "default.metal", "hlsl": "default_vert.hlsl"}
        });
        let frag = serde_json::json!({
            "kind": "fragment",
            "sources": {"metal": "default.metal", "hlsl": "default_frag.hlsl"}
        });
        assert!(declares_only_builtin_sources(&vert));
        assert!(declares_only_builtin_sources(&frag));
    }

    #[test]
    fn custom_source_is_not_builtin() {
        // A custom stage that forgot its glsl variant has a non-built-in
        // source and stays flagged.
        let mixed = serde_json::json!({
            "kind": "fragment",
            "sources": {"metal": "default.metal", "hlsl": "my_custom.hlsl"}
        });
        let custom = serde_json::json!({"kind": "vertex", "source": "my_custom.metal"});
        assert!(!declares_only_builtin_sources(&mixed));
        assert!(!declares_only_builtin_sources(&custom));
    }

    #[test]
    fn no_declared_source_is_not_builtin() {
        // A stage declaring nothing is malformed, not an engine default.
        assert!(!declares_only_builtin_sources(
            &serde_json::json!({"kind": "vertex"})
        ));
        assert!(!declares_only_builtin_sources(
            &serde_json::json!({"kind": "vertex", "source": ""})
        ));
    }
}
