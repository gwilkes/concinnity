// src/assets/scene_import.rs

use crate::ecs::{AssetOrigin, Component};

/// Imports a 3D scene file as a single declaration.
///
/// One `SceneImport` stands in for the whole asset graph a scene file
/// describes: its [Texture](#texture)s, [Material](#material)s,
/// [Mesh](#mesh)es, [Model](#model)s, and [Prop](#prop)s. The build expands the
/// import into those concrete assets, so `world.jsonl` stays small and
/// human-editable while the full graph lives in the lock file and compiled
/// blob. Geometry and texture pixels are never inlined into `world.jsonl`.
///
/// Supported `source` formats: `.fbx` and `.glb`.
///
/// **Generated names** are prefixed with the import's own asset `name`
/// (`<name>_mat_0`, `<name>_prim_0`, `<name>_model_0`, ...), so they never
/// clash with hand-authored assets. Because they only appear in the lock file
/// and blob, you never reference them by hand.
///
/// **Camera:** the import frames a [Camera3D](#camera3d) to the scene's bounds
/// so a freshly imported scene is immediately viewable. It is suppressed when
/// the world already declares a `Camera3D` (yours wins) or when `emit_camera`
/// is set to `false`.
///
/// ```jsonl
/// {"name":"bistro","type":"SceneImport","args":{"source":"assets/Bistro/BistroExterior.fbx","texture_max_size":512}}
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SceneImport {
    /// Path to the scene file, relative to the project root. `.fbx` or `.glb`.
    pub source: String,
    /// Ceiling on the longest edge of each imported texture, in pixels. Large
    /// source maps (2K-4K) are box-filtered down so the compiled scene, which
    /// stores uncompressed pixels, stays within a sane memory budget. `0` keeps
    /// each texture at its source resolution.
    pub texture_max_size: u32,
    /// Emissive factor applied to a material that carries an emissive map. Scene
    /// files often ship a zero emissive factor that would cancel the map, so a
    /// textured emissive gets this punchy factor instead.
    pub emissive_map_strength: f32,
    /// Whether to emit a [Camera3D](#camera3d) framed to the scene's bounds.
    /// Suppressed automatically when the world already declares a `Camera3D`.
    pub emit_camera: bool,
}

impl Default for SceneImport {
    fn default() -> Self {
        Self {
            source: String::new(),
            texture_max_size: 512,
            emissive_map_strength: 3.0,
            emit_camera: true,
        }
    }
}

impl Component for SceneImport {
    const NAME: &'static str = "SceneImport";
    const ORIGIN: AssetOrigin = AssetOrigin::BuildOnly;
    type Args = Self;

    fn from_args(args: Self) -> Self {
        args
    }
    fn to_args(&self) -> Self {
        self.clone()
    }
}
