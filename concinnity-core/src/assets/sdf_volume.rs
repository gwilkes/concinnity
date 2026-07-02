// src/assets/sdf_volume.rs
//
// A raymarched signed-distance-field volume. Authors a world-space
// bounding box plus a user-written fragment shader (containing the SDF
// `map` and per-point `shade` functions). At init the Metal backend
// builds a per-volume render pipeline that sphere-traces the SDF inside
// the box; hits write opaque colour into `hdr_resolve` and update the
// main depth attachment so the raymarched surface composites with
// rasterised geometry naturally.
//
// The user writes one `.metal` file that defines two functions:
//
// ```metal
// float map(float3 p, constant SdfParams& params, float time);
// SdfSurface shade(float3 p, float3 normal,
//                  constant SdfParams& params, float time);
// ```
//
// The engine prepends a header (`raymarch_helpers.metal`: IQ primitive
// library, `sdfNormal`, `coneRaymarch`, PBR helpers) and appends a
// template (`raymarch_template.metal`: vertex + `fragment_main` that
// reconstructs the ray, samples main depth for early-out, calls the
// user's `map` + `shade`, applies PBR + shadow, writes colour + depth).
// The wrapped source compiles at runtime via
// `newLibraryWithSource_options_error`, matching how the water / fog /
// decal / particle passes load their own MSL.
//
// The build pipeline (`compile_payload`) reads the user's source file
// and packs the raw bytes as this volume's payload, so production
// `cn run` worlds don't need the .metal file on disk at runtime: the
// bytes ride in the blob.

use crate::ecs::asset_id::AssetId;
use crate::ecs::{AssetOrigin, AssetPayload, CompanionSpec, Component, PayloadLocator};

/// Per-volume parameter slots packed into a single fixed-size uniform
/// block. The user shader casts the bound buffer to its own typed
/// struct; the engine just transports the bytes. Sized to comfortably
/// fit a flow-water shader (flow speed, wave coefficients, deep + shallow
/// colours, foam params, ...) without forcing schema design.
pub const SDF_PARAMS_LEN: usize = 32;

/// Hard cap on the per-volume cone-march step count. Matches the
/// runtime kernel's loop bound; values above this are clamped.
pub const SDF_MAX_STEPS_CEILING: u32 = 256;

/// Lower bound on the per-volume cone-march step count. Below this the
/// march doesn't have enough budget to converge on anything interesting.
pub const SDF_MAX_STEPS_FLOOR: u32 = 8;

/// A raymarched signed-distance-field volume. It occupies a world-space
/// bounding box; a user-authored fragment shader sphere-traces an SDF inside
/// the box, composites correctly with the surrounding scene through the depth
/// buffer, and shades hits with the engine's lighting helpers.
///
/// The fragment shader is selected per backend: a `fragment_shaders` map keyed
/// by `"metal"` / `"hlsl"` / `"glsl"` lets one volume target multiple backends,
/// and the build only requires the entry for the backend it is building for. A
/// single `fragment_shader` path is the fallback when no map entry matches.
///
/// ```jsonl
/// {"name":"chrome_blob","type":"SdfVolume","args":{
///   "centre":[0.0, 2.0, -4.0],
///   "extent":[2.0, 2.0, 2.0],
///   "fragment_shaders":{"metal":"shaders/chrome_blob.metal",
///                       "hlsl":"shaders/chrome_blob.hlsl"},
///   "max_gradient":1.0,
///   "max_steps":64,
///   "max_distance":12.0,
///   "params":[0.95, 0.85, 0.55, 0.08, 1.0, 0.0, 0.0, 0.0,
///             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
///             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
///             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
/// }}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SdfVolume {
    /// Asset identity; injected via `inject_name`. Not part of `args`.
    #[serde(skip)]
    pub asset_id: AssetId,
    /// World-space centre of the bounding box.
    pub centre: [f32; 3],
    /// XYZ half-widths of the bounding box. The raymarch is clipped to the box,
    /// so the SDF only has to be well-defined inside this region.
    pub extent: [f32; 3],
    /// Single-platform fragment shader source path (e.g.
    /// `"shaders/chrome_blob.metal"`), resolved relative to the project's
    /// `assets/` at build time. Used when `fragment_shaders` has no entry for
    /// the building backend; the file extension must match the backend
    /// (`.metal` / `.hlsl`). The file defines the SDF's `map` and `shade`
    /// functions.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub fragment_shader: String,
    /// Per-backend fragment shader source paths keyed by `"metal"`, `"hlsl"`,
    /// or `"glsl"`. Takes priority over `fragment_shader`, letting one volume
    /// target multiple backends from a single declaration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fragment_shaders: Option<std::collections::HashMap<String, String>>,
    /// Worst-case gradient of the SDF, used to size the cone-march step. `1.0`
    /// is correct for any well-formed SDF; higher values shorten the step but
    /// stay safe. Must be > 0.
    pub max_gradient: f32,
    /// Maximum cone-march steps per pixel. Clamped to `[8, 256]`.
    pub max_steps: u32,
    /// Maximum march distance in metres. Must be ≥ 0.1.
    pub max_distance: f32,
    /// Generic parameter block passed to the shader as a uniform buffer; the
    /// shader interprets it however it likes. Up to 32 values.
    pub params: [f32; SDF_PARAMS_LEN],
    /// When true, the volume casts shadows onto the surrounding scene. Disable
    /// for translucent / volumetric effects that shouldn't block light.
    pub cast_shadows: bool,
    /// When true (the default), the volume is shadowed by the scene. Set to
    /// false for unlit / always-bright effects (energy fields, etc.).
    pub receive_shadows: bool,
    /// When true, the volume renders as a participating medium (clouds, smoke,
    /// fog blobs, energy fields) instead of an opaque surface. The shader must
    /// define `sampleVolume(p, params, time)` returning per-point density,
    /// scattering colour, and emission instead of `map` / `shade`. Volumetrics
    /// never cast shadows (`cast_shadows` is forced off). The medium fills the
    /// whole bounding box, so don't overlap it with geometry it should render
    /// behind.
    pub volumetric: bool,
    /// When false the volume is skipped each frame.
    pub visible: bool,
    /// Injected at load time from the blob def. Carries the user
    /// shader source bytes packed at build time.
    #[serde(skip)]
    pub locator: Option<PayloadLocator>,
}

impl Default for SdfVolume {
    fn default() -> Self {
        Self {
            asset_id: AssetId::default(),
            centre: [0.0, 0.0, 0.0],
            extent: [1.0, 1.0, 1.0],
            fragment_shader: String::new(),
            fragment_shaders: None,
            max_gradient: 1.0,
            max_steps: 64,
            max_distance: 30.0,
            params: [0.0; SDF_PARAMS_LEN],
            cast_shadows: false,
            receive_shadows: true,
            volumetric: false,
            visible: true,
            locator: None,
        }
    }
}

#[allow(dead_code)] // Consumed by the backend raymarch passes (client crate); unused within core.
impl SdfVolume {
    /// Effective cone-march step ratio derived from the Lipschitz
    /// constant. A 1-Lipschitz SDF (gradient ≤ 1) cone-marches at
    /// ratio 1; larger gradients shorten the step proportionally.
    pub fn cone_ratio(&self) -> f32 {
        1.0 / self.max_gradient.max(f32::EPSILON)
    }
}

impl SdfVolume {
    /// Resolve the fragment shader source path for the current build backend
    /// from this volume's `fragment_shaders` map (preferred) or its
    /// `fragment_shader` fallback. Mirrors the build-time
    /// `SourceBacked::source_path` selection. Returns `None` when no source
    /// matches the current backend (e.g. a Metal-only volume inspected on a
    /// DirectX build).
    pub fn current_platform_source(&self) -> Option<String> {
        use crate::build::SourceBacked;
        let args = serde_json::to_value(self).ok()?;
        <Self as SourceBacked>::source_path(&args, crate::build::Platform::current())
    }
}

impl Component for SdfVolume {
    const NAME: &'static str = "SdfVolume";
    const ORIGIN: AssetOrigin = AssetOrigin::External;
    const PAYLOAD: AssetPayload = AssetPayload::Compiled;
    type Args = Self;

    fn from_args(mut args: Self) -> Self {
        // Extents must be positive: a zero or negative extent would
        // produce an inside-out bounding box no fragment ever enters.
        for axis in args.extent.iter_mut() {
            if !axis.is_finite() || *axis <= 0.0 {
                *axis = 1.0;
            }
        }
        if !args.max_gradient.is_finite() || args.max_gradient <= 0.0 {
            args.max_gradient = 1.0;
        }
        args.max_steps = args
            .max_steps
            .clamp(SDF_MAX_STEPS_FLOOR, SDF_MAX_STEPS_CEILING);
        if !args.max_distance.is_finite() || args.max_distance < 0.1 {
            args.max_distance = 0.1;
        }
        // Volumetrics are translucent: they don't write depth, so the
        // shadow pass has no surface to project. Force the flag off
        // rather than silently building an unusable shadow PSO.
        if args.volumetric {
            args.cast_shadows = false;
        }
        // Collapse the per-backend `fragment_shaders` map down to the single
        // `fragment_shader` for the current backend so the runtime sees a
        // concrete current-backend source path regardless of how the volume
        // was authored. In particular the DirectX raymarch pass filters
        // volumes by this path's extension; keeping it populated lets that
        // filter work for map-authored volumes without a backend-specific
        // change. No-op when the map has no entry for this backend.
        if let Some(src) = args.current_platform_source() {
            args.fragment_shader = src;
        }
        args
    }

    fn to_args(&self) -> Self {
        self.clone()
    }

    fn inject_name(&mut self, id: AssetId) {
        self.asset_id = id;
    }

    fn inject_locator(&mut self, locator: PayloadLocator) {
        self.locator = Some(locator);
    }

    fn companions(_args: &serde_json::Value, _world: &[serde_json::Value]) -> Vec<CompanionSpec> {
        vec![CompanionSpec {
            name: "GraphicsConfig",
            asset_type: "GraphicsConfig",
            args: serde_json::json!({}),
        }]
    }
}

// Resolve a raw `fragment_shader` arg to an on-disk path, picking the first
// candidate that exists. Resolution order:
//   1. `.concinnity/assets/<raw>`: runtime-fetched cache (the production
//      location once a world has been built and `cn run` fetches its
//      dependencies).
//   2. `.concinnity/assets/<bare>` recursive search: same bare-filename
//      match `ShaderStage` does.
//   3. `<artifacts_dir>/<raw>`: LLM-written artifact under
//      `data/artifacts/<account_id>/`, matching the existing ShaderStage path.
//   4. `assets/<raw>`: source-tree convenience for `cn debug` run from
//      `concinnity-client/` against shaders authored in the repo's `assets/`
//      directory.
//   5. `<raw>` as-is: relative-to-cwd fallback (matches how other asset
//      `source` fields handle e.g. `"../concinnity-infra/assets/..."`).
// Returns `None` when nothing exists; `compile_payload` falls back to the raw
// path in that case so the read error surfaces with a useful message.
pub fn resolve_source_path(raw: &str, ctx: &crate::build::BuildCtx<'_>) -> Option<String> {
    let raw_path = std::path::Path::new(raw);
    let mut candidates: Vec<String> = Vec::new();
    if raw_path.is_absolute() {
        candidates.push(raw.to_string());
    } else {
        candidates.push(
            crate::paths::assets_dir()
                .join(raw)
                .to_string_lossy()
                .into_owned(),
        );
        if raw_path
            .parent()
            .map(|d| d.as_os_str().is_empty())
            .unwrap_or(true)
            && let Some(found) = crate::world::preset::find_in_assets(raw)
        {
            candidates.push(found);
        }
        if let Some(dir) = ctx.artifacts_dir {
            candidates.push(format!("{dir}/{raw}"));
        }
        candidates.push(format!("assets/{raw}"));
        candidates.push(raw.to_string());
    }
    candidates
        .into_iter()
        .find(|p| std::path::Path::new(p).exists())
}

impl crate::build::SourceBacked for SdfVolume {
    fn source_path(args: &serde_json::Value, platform: crate::build::Platform) -> Option<String> {
        // Prefer the per-backend map entry for this platform.
        if let Some(obj) = args.get("fragment_shaders").and_then(|v| v.as_object())
            && let Some(src) = obj
                .get(platform.key())
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
        {
            return Some(src.to_string());
        }
        // Fall back to the single path, but only when its extension matches
        // this backend: a `.hlsl` path is not a source the Metal build needs.
        let src = args
            .get("fragment_shader")
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

/// Resolve the raw fragment shader source declared for the current build
/// backend, applying the `fragment_shaders` map / `fragment_shader` fallback.
pub fn current_platform_source_arg(args: &serde_json::Value) -> Option<String> {
    use crate::build::SourceBacked;
    <SdfVolume as SourceBacked>::source_path(args, crate::build::Platform::current())
}

/// Blob indices that hold an `SdfVolume` fragment-shader payload.
///
/// The graphics-system init drains `SdfVolume`s and reads their payload
/// bytes via the locator. The release sweep earlier in the same init
/// frees every blob whose contents have already been consumed, but
/// because the SDF drain runs *after* that sweep, any blob holding only
/// an SDF payload would be freed before being read. (When the world
/// has other small assets, the SDF shader bytes typically share a blob
/// with a kept asset and survive by accident; a world whose SDF shader
/// ends up alone in its blob exposes the bug as "SdfVolume payload
/// FileIo, skipping" with no surface drawn.) This helper lets the
/// release sweep keep SDF blobs resident, matching the
/// `audio_clip_blob_indices` pattern.
pub fn sdf_volume_blob_indices(
    ctx: &crate::ecs::PipelineContext,
) -> std::collections::HashSet<u32> {
    ctx.query::<SdfVolume>()
        .filter_map(|v| v.locator.as_ref().map(|l| l.blob_index))
        .collect()
}

/// Resolve the fragment shader source path the runtime should watch /
/// re-read for hot-reload. Mirrors `ShaderStage::resolve_runtime_source_path`
/// so the asset-hot-reload subsystem can subscribe to changes under
/// `concinnity-client/assets/shaders/` for every live SdfVolume. Unused
/// today.
#[allow(dead_code)]
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

/// Validate `SdfVolume` args without compiling. Called by the check pass.
pub fn check(args: &serde_json::Value) -> Result<(), String> {
    if current_platform_source_arg(args).is_none() {
        let platform_key = crate::build::Platform::current().key();
        return Err(format!(
            "SdfVolume requires a `fragment_shader` or a `fragment_shaders` \
             entry for backend \"{platform_key}\" (a path to a shader file \
             declaring map + shade)"
        ));
    }
    if let Some(params) = args.get("params").and_then(|v| v.as_array())
        && params.len() > SDF_PARAMS_LEN
    {
        return Err(format!(
            "SdfVolume `params` is {} entries; max is {} \
                 (extra entries would be ignored)",
            params.len(),
            SDF_PARAMS_LEN
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// File extension matching the backend these tests compile against, so a
    /// single `fragment_shader` path resolves as current-platform-compatible
    /// on Metal, DirectX, and Vulkan alike.
    fn platform_ext() -> &'static str {
        crate::build::Platform::current().key()
    }

    #[test]
    fn defaults_are_sensible() {
        let v = SdfVolume::default();
        assert_eq!(v.centre, [0.0, 0.0, 0.0]);
        assert_eq!(v.extent, [1.0, 1.0, 1.0]);
        assert_eq!(v.max_gradient, 1.0);
        assert_eq!(v.max_steps, 64);
        assert_eq!(v.max_distance, 30.0);
        assert!(v.receive_shadows);
        assert!(!v.cast_shadows);
        assert!(v.visible);
        assert_eq!(v.params.len(), SDF_PARAMS_LEN);
        assert_eq!(v.cone_ratio(), 1.0);
    }

    #[test]
    fn from_args_clamps_steps() {
        let mut a = SdfVolume {
            max_steps: 1,
            ..Default::default()
        };
        let clamped = SdfVolume::from_args(a.clone());
        assert_eq!(clamped.max_steps, SDF_MAX_STEPS_FLOOR);

        a.max_steps = 9999;
        let clamped = SdfVolume::from_args(a);
        assert_eq!(clamped.max_steps, SDF_MAX_STEPS_CEILING);
    }

    #[test]
    fn from_args_repairs_bad_extent() {
        let a = SdfVolume {
            extent: [0.0, -1.0, f32::NAN],
            ..Default::default()
        };
        let fixed = SdfVolume::from_args(a);
        assert_eq!(fixed.extent, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn from_args_repairs_bad_gradient_and_distance() {
        let a = SdfVolume {
            max_gradient: -0.5,
            max_distance: f32::NAN,
            ..Default::default()
        };
        let fixed = SdfVolume::from_args(a);
        assert_eq!(fixed.max_gradient, 1.0);
        assert_eq!(fixed.max_distance, 0.1);
    }

    #[test]
    fn cone_ratio_inverts_gradient() {
        let v = SdfVolume {
            max_gradient: 2.0,
            ..Default::default()
        };
        assert!((v.cone_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn check_requires_fragment_shader() {
        let args = serde_json::json!({});
        assert!(check(&args).is_err());

        let args = serde_json::json!({"fragment_shader": ""});
        assert!(check(&args).is_err());

        let args =
            serde_json::json!({"fragment_shader": format!("shaders/blob.{}", platform_ext())});
        assert!(check(&args).is_ok());
    }

    #[test]
    fn check_rejects_oversized_params() {
        let mut params = vec![0.0; SDF_PARAMS_LEN + 1];
        params[0] = 1.0;
        let args = serde_json::json!({
            "fragment_shader": format!("shaders/blob.{}", platform_ext()),
            "params": params,
        });
        assert!(check(&args).is_err());
    }

    #[test]
    fn check_accepts_short_params() {
        // Less than SDF_PARAMS_LEN is fine: the rest defaults to 0.
        let args = serde_json::json!({
            "fragment_shader": format!("shaders/blob.{}", platform_ext()),
            "params": [1.0, 2.0, 3.0],
        });
        assert!(check(&args).is_ok());
    }

    #[test]
    fn check_rejects_source_for_other_backend_only() {
        // A single path whose extension targets a different backend is "no
        // source for this platform": the build needs a current-backend
        // shader, so validation fails rather than trying to read it.
        let other_ext = match platform_ext() {
            "metal" => "hlsl",
            _ => "metal",
        };
        let args = serde_json::json!({ "fragment_shader": format!("shaders/blob.{other_ext}") });
        assert!(check(&args).is_err());
    }

    #[test]
    fn check_accepts_sources_map_with_current_backend() {
        // A per-backend map that includes the current backend validates even
        // when it also lists other backends the build won't compile here.
        let args = serde_json::json!({
            "fragment_shaders": {
                "metal": "shaders/blob.metal",
                "hlsl": "shaders/blob.hlsl",
                "glsl": "shaders/blob.glsl",
            }
        });
        assert!(check(&args).is_ok());
    }

    #[test]
    fn check_rejects_sources_map_without_current_backend() {
        // A map lacking the current backend's entry has nothing to build here.
        let other_ext = match platform_ext() {
            "metal" => "hlsl",
            _ => "metal",
        };
        let args = serde_json::json!({
            "fragment_shaders": { other_ext: format!("shaders/blob.{other_ext}") }
        });
        assert!(check(&args).is_err());
    }

    #[test]
    fn source_path_prefers_map_over_single() {
        use crate::build::{Platform, SourceBacked};
        let args = serde_json::json!({
            "fragment_shader": "shaders/single.metal",
            "fragment_shaders": { "metal": "shaders/from_map.metal" },
        });
        assert_eq!(
            <SdfVolume as SourceBacked>::source_path(&args, Platform::Metal).as_deref(),
            Some("shaders/from_map.metal")
        );
    }

    #[test]
    fn from_args_collapses_map_to_current_backend() {
        // The runtime struct should carry the current backend's path in
        // `fragment_shader` so the DirectX path-extension filter still works
        // for map-authored volumes.
        // Include every backend so the collapse resolves regardless of which
        // backend this test build targets (metal / hlsl / glsl).
        let mut map = std::collections::HashMap::new();
        map.insert("metal".to_string(), "shaders/blob.metal".to_string());
        map.insert("hlsl".to_string(), "shaders/blob.hlsl".to_string());
        map.insert("glsl".to_string(), "shaders/blob.glsl".to_string());
        let a = SdfVolume {
            fragment_shaders: Some(map),
            ..Default::default()
        };
        let resolved = SdfVolume::from_args(a);
        assert_eq!(
            resolved.fragment_shader,
            format!("shaders/blob.{}", platform_ext())
        );
    }

    #[test]
    fn volumetric_forces_cast_shadows_off() {
        let a = SdfVolume {
            volumetric: true,
            cast_shadows: true,
            ..Default::default()
        };
        let fixed = SdfVolume::from_args(a);
        assert!(fixed.volumetric);
        assert!(
            !fixed.cast_shadows,
            "volumetric SDFs are translucent and must not cast hard shadows"
        );
    }

    #[test]
    fn volumetric_default_is_off() {
        let v = SdfVolume::default();
        assert!(!v.volumetric);
    }

    #[test]
    fn roundtrip_through_args() {
        let mut v = SdfVolume {
            centre: [1.0, 2.0, 3.0],
            extent: [4.0, 5.0, 6.0],
            fragment_shader: "shaders/foo.metal".to_string(),
            ..Default::default()
        };
        v.params[7] = 0.42;
        let json = serde_json::to_value(v.to_args()).expect("serialises");
        let back: SdfVolume = serde_json::from_value(json).expect("deserialises");
        let back = SdfVolume::from_args(back);
        assert_eq!(back.centre, [1.0, 2.0, 3.0]);
        assert_eq!(back.extent, [4.0, 5.0, 6.0]);
        assert_eq!(back.fragment_shader, "shaders/foo.metal");
        assert_eq!(back.params[7], 0.42);
    }
}
