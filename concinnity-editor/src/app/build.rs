// src/app/build.rs: CLI build command + shared build orchestration

#[allow(unused_imports)]
pub use concinnity_cook::{
    PipelineResult, build_compiled, build_pipeline_from_str, validate_asset, validate_world_jsonl,
};

use crate::ecs::{ComponentAsset, World};
use concinnity_cook::world::LoadedWorld;

// Load, validate, and (when server credentials are present) fetch the missing
// source files for a world. The returned LoadedWorld has passed the full
// validation front half and is ready for concinnity_cook::build_compiled.
//
// This is the shared front half of every build path: the CLI build, the
// interpreted `run`, and the Swift FFI build/preview entry points all funnel
// through here so validation and asset fetching behave identically.
// Driven by the binary's CLI build and interpreted run; unreferenced in the
// FFI lib build, which prepares through concinnity_cook directly.
#[allow(dead_code)]
pub(crate) fn prepare(content: &str) -> std::io::Result<LoadedWorld> {
    // Install the render backend's shader-layout validator before any shader
    // compiles, so a user shader that mis-declares an engine buffer struct fails
    // the build with a clear message instead of faulting the GPU at run time.
    // One call here covers the CLI build, `run`, and the FFI entry points.
    ensure_shader_layout_validator();

    let loaded = concinnity_cook::prepare_world(content)
        .map_err(|errs| concinnity_cook::check::report_validation_errors(&errs))?;

    Ok(loaded)
}

// Register the backend's shader-layout validator with the core build pipeline.
// Only the Metal backend ships one today; other backends leave the hook
// unregistered and build exactly as before.
#[cfg(backend_metal)]
#[allow(dead_code)]
fn ensure_shader_layout_validator() {
    crate::shader_reflect::register_shader_layout_validator();
}

#[cfg(not(backend_metal))]
#[allow(dead_code)]
fn ensure_shader_layout_validator() {}

// Compile a prepared world and assemble it into an in-memory World, ready to
// run without touching any blob files on disk.
pub(crate) fn world_from_loaded(loaded: LoadedWorld) -> std::io::Result<World> {
    let result = build_compiled(loaded.assets, None)?;

    let payload_sections: Vec<Option<Vec<u8>>> = result.payloads.into_iter().map(Some).collect();
    let mut world = World::new(crate::blob::BlobData::new(payload_sections));
    for def in &result.defs {
        let mut component = ComponentAsset::from_def(def).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Asset construction failed: {:?}", e),
            )
        })?;
        if let Some(locator) = &def.payload {
            component.inject_locator(locator.clone());
        }
        world.add(component);
    }
    Ok(world)
}
