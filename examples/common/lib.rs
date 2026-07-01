// examples/common/lib.rs
//
// Shared host glue for the runnable examples. An example reads a world.jsonl,
// compiles it in memory with the asset pipeline (concinnity-cook), and plays
// it through the runtime renderer (concinnity-client). This crate owns that
// compile-and-run path so each example's main.rs is just its own preflight
// (locating the world, fetching assets) plus a call into here.
//
// world.jsonl cannot be played directly: the runtime reads compiled blobs, not
// source declarations. The compile step (validate, expand the build-time
// macros, rasterize fonts, decode source files) lives in concinnity-cook. The
// editor crate (CLI, debug server, FFI) is deliberately not a dependency.

use concinnity_client::app::state::App;
use concinnity_client::blob::BlobData;
use concinnity_client::ecs::{ComponentAsset, World};
use concinnity_cook::world::LoadedWorld;
use concinnity_cook::{build_compiled, check::report_validation_errors, prepare_world};

// Install the runtime's tracing subscriber. Call once at the top of main so the
// compile step's logs are formatted. Re-exported from the runtime crate.
pub use concinnity_client::app::run::init_logging;

// Project state-root anchor. An example that chdirs so its world's relative
// asset paths resolve calls `paths::set_root(invocation_dir)` first, so the
// `.concinnity/` cache and config land where the command was run, not in the
// example's directory.
pub use concinnity_cook::paths;

// Compile world.jsonl into a runnable World entirely in memory: validate and
// expand the declarations, compile each asset's payload, then assemble the
// components into a World backed by the compiled blob. Source-backed assets are
// cached under `.concinnity/cache/` (relative to the current directory), so a
// second run with unchanged sources skips the expensive decode/compile.
pub fn compile_world(content: &str) -> std::io::Result<World> {
    let loaded: LoadedWorld =
        prepare_world(content).map_err(|errs| report_validation_errors(&errs))?;

    let result = build_compiled(loaded.assets, None)?;

    let payload_sections: Vec<Option<Vec<u8>>> = result.payloads.into_iter().map(Some).collect();
    let mut world = World::new(BlobData::new(payload_sections));

    for def in &result.defs {
        let mut component = ComponentAsset::from_def(def).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("asset construction failed: {e:?}"),
            )
        })?;
        if let Some(locator) = &def.payload {
            component.inject_locator(locator.clone());
        }
        world.add(component);
    }

    Ok(world)
}

// Play a compiled world on the runtime's render loop until the window closes,
// a system stops the world, or CTRL+C is received.
pub fn run(world: World) -> std::io::Result<()> {
    let mut app = App::new();
    *app.world_mut() = world;
    concinnity_client::app::run::start_runtime(app)
}
