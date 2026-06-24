// src/build/shader.rs
//
// Dispatches to the correct backend compiler based on the source file extension.
//   .metal -> xcrun metal + xcrun metallib -> raw .metallib bytes (macOS only)
//   all others -> glslc or shaderc -> SPIR-V bytes
//
// Built-in shader sources are embedded into the binary at compile time so the
// runtime never depends on `assets/` for them. Any caller-supplied source path
// whose bare filename matches a built-in name resolves to the embedded bytes.
// Source is handed straight to the platform compiler (over stdin or in
// memory); no shader source file is written to disk.

// Engine-shipped shader source files embedded at compile time. The bare
// filename is the public identifier used in world.jsonl (e.g. "default.metal").
const BUILTIN_DEFAULT_METAL: &str = include_str!("shaders/default.metal");
// `pub` so the client's DirectX backend can embed the same default HLSL it
// would otherwise have to `include_str!` across the crate boundary.
pub const BUILTIN_DEFAULT_VERT_HLSL: &str = include_str!("shaders/default_vert.hlsl");
pub const BUILTIN_DEFAULT_FRAG_HLSL: &str = include_str!("shaders/default_frag.hlsl");
pub const BUILTIN_DEFAULT_VERT_INSTANCED_HLSL: &str =
    include_str!("shaders/default_vert_instanced.hlsl");
pub const BUILTIN_SHADOW_MAP_VERT_HLSL: &str = include_str!("shaders/shadow_map_vert.hlsl");

// Returns the embedded source for a bare built-in shader filename. The match
// is by bare filename only (any leading directory is stripped first), so
// `"default.metal"` and `"default_shader/default.metal"` both resolve.
pub fn builtin_shader_source(filename: &str) -> Option<&'static str> {
    let bare = std::path::Path::new(filename)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(filename);
    Some(match bare {
        "default.metal" => BUILTIN_DEFAULT_METAL,
        "default_vert.hlsl" => BUILTIN_DEFAULT_VERT_HLSL,
        "default_frag.hlsl" => BUILTIN_DEFAULT_FRAG_HLSL,
        "default_vert_instanced.hlsl" => BUILTIN_DEFAULT_VERT_INSTANCED_HLSL,
        "shadow_map_vert.hlsl" => BUILTIN_SHADOW_MAP_VERT_HLSL,
        _ => return None,
    })
}

#[cfg(test)]
mod builtin_registry_tests {
    use super::builtin_shader_source;

    #[test]
    fn shipped_builtins_are_registered() {
        // Every shader shipped under src/build/shaders/ must resolve to
        // non-empty source through `builtin_shader_source`. A new built-in
        // default/outdoor/shadow shader that is dropped in but never added to
        // the match would otherwise fall through to a disk read at build time
        // (and fail far from the cause). This is the core-side twin of the
        // Metal client's `shipped_shaders_are_registered`.
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/build/shaders");
        let mut checked = 0usize;
        for entry in std::fs::read_dir(dir).expect("read builtin shaders dir") {
            let file_name = entry.expect("dir entry").file_name();
            let name = file_name.to_str().expect("utf8 shader filename");
            if !(name.ends_with(".metal") || name.ends_with(".hlsl")) {
                continue;
            }
            let src = builtin_shader_source(name).unwrap_or_else(|| {
                panic!("{name} is shipped but not registered in builtin_shader_source")
            });
            assert!(
                !src.trim().is_empty(),
                "{name}: registered but empty source"
            );
            checked += 1;
        }
        assert!(checked > 0, "no built-in shaders found under {dir}");
    }

    #[test]
    fn default_metal_reflection_cut_matches_canonical() {
        // default.metal is compiled offline and baked, so it keeps its own
        // `constant float REFL_RESOLVE_CUT` instead of the runtime-injected
        // shared constant the resolve shaders use. Lock it to the canonical
        // value so the forward double-count fade can never drift from the
        // SSR / RT resolve gates. Expects a clean `= <value>;` declaration.
        let src = builtin_shader_source("default.metal").expect("default.metal builtin");
        let decl = src
            .lines()
            .find(|l| l.contains("constant float REFL_RESOLVE_CUT"))
            .expect("REFL_RESOLVE_CUT declaration in default.metal");
        let value: f32 = decl
            .split(';')
            .next()
            .and_then(|head| head.split('=').nth(1))
            .map(str::trim)
            .and_then(|s| s.parse().ok())
            .expect("parse REFL_RESOLVE_CUT value from default.metal");
        assert_eq!(
            value,
            crate::gfx::ssr::REFLECTION_ROUGHNESS_CUT,
            "default.metal REFL_RESOLVE_CUT must equal REFLECTION_ROUGHNESS_CUT"
        );
    }
}
