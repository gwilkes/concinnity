// src/main.rs
//
// The `concinnity` dev CLI binary. Dispatches build / add / rm / list / test /
// run / debug. The runtime (`cn run`) is delegated to the runtime crate; every
// other command drives the compile pipeline + the in-process debug server.

// Bridge: re-export the runtime crate's modules under crate::* so the editor
// code moved out of the runtime crate keeps its historical `crate::<module>`
// import paths. Mirrors lib.rs (a binary is a separate crate root).
#[cfg(backend_dx)]
#[allow(unused_imports)]
pub(crate) use concinnity_client::directx;
#[cfg(backend_metal)]
#[allow(unused_imports)]
pub(crate) use concinnity_client::metal;
#[cfg(backend_vk)]
#[allow(unused_imports)]
pub(crate) use concinnity_client::vulkan;
#[allow(unused_imports)]
pub(crate) use concinnity_client::{assets, blob, config, ecs, gfx, jobs};
#[allow(unused_imports)]
pub(crate) use concinnity_core::{build, geometry, result, world};

// Editor-owned modules (moved out of the runtime crate).
mod app;
mod cli;
mod debug;
// C-ABI module compiled into the binary's tree so that helpers it consumes
// aren't flagged as dead code here. The exported symbols are unused from the
// CLI but harmless.
#[cfg(target_os = "macos")]
mod ffi;
#[cfg(backend_metal)]
mod shader_reflect;
// Animation clip hot-reload decode (driven by the debug server).
mod anim_reload;

// Microsoft Agility SDK opt-in.
//
// Windows' system `d3d12.dll` reads these two symbols from the host
// EXE's PE export table at process start: when both are present and
// the named SDK path resolves to a directory containing `D3D12Core.dll`,
// it loads that copy in place of the OS-bundled (older) D3D12 runtime.
// Modern FidelityFX FSR3 (and any other feature requiring a recent
// D3D12 capability bit) needs the Agility SDK; without these exports
// `ffxCreateContext` throws a C++ exception that aborts the process.
//
// The companion `build.rs` setup copies `D3D12Core.dll` +
// `d3d12SDKLayers.dll` from the NuGet package into
// `target/{profile}/D3D12/` so this relative path resolves. Setting
// the `D3D12_SDK_VERSION` value here to match the NuGet package
// version is critical; the directory name is `microsoft.direct3d.d3d12.1.<VER>.<PATCH>`.
//
// `#[used]` forces the linker to keep the symbols around even though
// nothing in Rust references them; `#[no_mangle]` keeps the exact
// case-sensitive name `d3d12.dll` looks up.
#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKVersion: u32 = 619;

#[cfg(backend_dx)]
#[unsafe(no_mangle)]
#[used]
pub static D3D12SDKPath: &[u8; 9] = b".\\D3D12\\\0";

use clap::{Parser, Subcommand};

const BANNER: &str = r#"
   ______                                
  / ____/___  ____  ___________  ____  __________  __
 / /   / __ \/ __ \/ ___/ / __ \/ __ \/ /_  __/ / / /
/ /___/ /_/ / / / / /__/ / / / / / / / / / / / /_/ /
\____/\____/_/ /_/\___/_/_/ /_/_/ /_/_/ /_/  \__, /
                                            /____/"#;

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a new app in the current directory
    #[command(name = "init")]
    Init,

    /// Create a new app in a new directory
    #[command(name = "new")]
    New(NewArgs),

    /// Build a world from .concinnity/worlds/ into binary blobs
    #[command(name = "build")]
    Build(BuildArgs),

    /// Run a compiled world
    //
    // Production path: no debug server and no WebSocket command channel.
    // A shipped run is neither remotely inspectable nor remotely driven:
    // use `cn debug` for that.
    #[command(name = "run")]
    Run(RunArgs),

    /// Run interpreted directly from a world jsonl file
    //
    // Compiles the world in memory (no prior `cn build` needed) and stands
    // up the localhost debug server.
    // This is the development run, and the path the agentic loop / Swift UI
    // use when they need to read or drive runtime state over a WebSocket.
    #[command(name = "debug")]
    Debug(DebugArgs),

    /// Add an asset to the active world
    //
    // TARGET can be:
    //   - A file path  (shaders/pbr.vert, models/scene.obj)
    //     Type is inferred from the file extension or the JSON `type` field.
    //   - A type name  (Logger, LLM, HttpServer, VulkanRenderer, ...)
    //     Asset is created with the type's registered default args.
    #[command(name = "add")]
    Add(AddArgs),

    /// Remove an asset from the active world by its unique name
    //
    // NAME is the value of the `name` field in world.jsonl
    // (e.g. "my_llm", "pbr_vert", "tool_agent").
    #[command(name = "rm")]
    Rm(RmArgs),

    /// List all declared assets
    #[command(name = "list")]
    List(ListArgs),

    /// Validate a world without building
    #[command(name = "test")]
    Test(TestArgs),
}

#[derive(Parser, Debug)]
#[command(name = "concinnity")]
#[command(about = BANNER, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, clap::Args)]
pub struct DebugArgs {
    // Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    // Connect to the server's WebSocket command channel.
    // Value is the ws:// or wss:// URL of the endpoint,
    // e.g. ws://127.0.0.1:8080/v1/ws
    #[arg(long)]
    pub websocket: Option<String>,

    // Account ID to authenticate with when connecting over WebSocket.
    // Must be non-empty, <= 128 chars, and not prefixed with "guest:".
    // Required when --websocket is set.
    #[arg(long)]
    pub ws_user: Option<String>,

    // Base HTTP URL of the infra server used to fetch missing asset files.
    // Defaults to the value in the client config (~/.config/concinnity/config.json).
    #[arg(long)]
    pub server: Option<String>,

    // Account ID for asset fetching authentication.
    // Defaults to the value in the client config.
    #[arg(long)]
    pub user: Option<String>,

    // Port for the localhost runtime debug server (default 8777).
    #[arg(long)]
    pub debug_port: Option<u16>,

    // Enable graphics API validation. Omitted defaults to on for debug builds
    // and off for release. See `RunArgs::validation`.
    #[arg(long)]
    pub validation: Option<bool>,
}

#[derive(Debug, clap::Args)]
pub struct RunArgs {
    // Enable graphics API validation: the DirectX / Vulkan debug layers, or on
    // macOS the Metal API-validation layer (the process re-execs once with
    // `MTL_DEBUG_LAYER` set, since Metal cannot toggle it from inside a running
    // process). Omitted defaults to on for debug builds and off for release;
    // pass `--validation false` to force it off in a debug build. The heavier
    // Metal shader validation is not enabled by this flag; set
    // `MTL_SHADER_VALIDATION=1` in the environment for that.
    #[arg(long)]
    pub validation: Option<bool>,
}

#[derive(Debug, clap::Args)]
pub struct AddArgs {
    // File path (shaders/pbr.vert) or asset type name (Logger, LLM, ...)
    pub target: String,

    // Override the asset name written into world.jsonl.
    // If omitted, the name is derived from the filename (including extension).
    #[arg(short, long)]
    pub name: Option<String>,

    // Named scaffold preset used when bootstrapping a fresh GLB world.
    // Currently only "showcase" (adds bloom/IBL/fog on top of the base scaffold).
    // Ignored when scaffolding doesn't fire.
    #[arg(short = 't', long)]
    pub template: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct RmArgs {
    // The `name` field of the asset to remove
    pub name: String,
}

#[derive(Debug, clap::Args)]
pub struct TestArgs {
    // Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct ListArgs {
    // Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Debug, clap::Args)]
pub struct NewArgs {
    // Directory to create the project in
    pub path: String,
}

#[derive(Debug, clap::Args)]
pub struct BuildArgs {
    // Path to a world JSONL file (default: discover from .concinnity/worlds/)
    #[arg(short = 'f', long)]
    pub file: Option<String>,

    // Base HTTP URL of the infra server used to fetch missing asset files.
    // Defaults to the value in the client config (~/.config/concinnity/config.json).
    #[arg(long)]
    pub server: Option<String>,

    // Account ID for asset fetching authentication.
    // Defaults to the value in the client config.
    #[arg(long)]
    pub user: Option<String>,
}

// When a render command requests graphics validation on macOS, relaunch the
// process with Metal's API-validation layer (`MTL_DEBUG_LAYER`) set in the
// environment, then return into the replacement image. Metal reads that
// variable during early framework initialisation, so it cannot be toggled from
// a process that has already touched Metal -- and `std::env::set_var` is
// unsound once worker threads exist (the frameworks call `getenv` off-thread).
// Re-exec sidesteps both: the child starts with the variable present from PID
// birth. DirectX / Vulkan take the request through `dev_flags` and need no
// relaunch, so this is a macOS-only concern.
//
// The heavier `MTL_SHADER_VALIDATION` is deliberately left off: it is far more
// expensive and its memory footprint climbs over a long run, so it stays an
// explicit manual opt-in rather than riding a flag that defaults on in debug
// builds.
#[cfg(target_os = "macos")]
fn reexec_with_metal_validation(cli: &Cli) {
    use std::os::unix::process::CommandExt;

    // Only the rendering commands create a Metal context.
    let requested = match &cli.command {
        Commands::Run(args) => args.validation,
        Commands::Debug(args) => args.validation,
        _ => return,
    };
    if !requested.unwrap_or(cfg!(debug_assertions)) {
        return;
    }
    // The relaunched child inherits the variable, so the guard is
    // self-terminating: it stops the second pass from re-execing again.
    if std::env::var_os("MTL_DEBUG_LAYER").is_some() {
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("validation: cannot locate current executable to re-exec: {e}");
            return;
        }
    };
    // `exec` replaces this image in place (no lingering parent process) and only
    // returns on failure. On failure we fall through and run without Metal
    // validation rather than aborting the user's session.
    let err = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env("MTL_DEBUG_LAYER", "1")
        .exec();
    eprintln!("validation: failed to re-exec with Metal validation enabled: {err}");
}

#[cfg(not(target_os = "macos"))]
fn reexec_with_metal_validation(_cli: &Cli) {}

fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    // Must run before any thread spawns or the Metal framework initialises.
    reexec_with_metal_validation(&cli);

    match &cli.command {
        Commands::Init => cli::init(),
        Commands::New(args) => cli::new(&args.path),
        Commands::Build(args) => cli::build(args.file.as_deref()),
        Commands::Run(args) => {
            app::dev_flags::set_validation(args.validation);
            concinnity_client::app::run()
        }
        Commands::Debug(args) => {
            app::dev_flags::set_enabled(true);
            app::dev_flags::set_validation(args.validation);
            let port = args.debug_port.unwrap_or(8777);
            let debug_hook: Box<dyn app::DebugHook> = match debug::DebugServer::start(port) {
                Ok(srv) => Box::new(srv),
                Err(e) => {
                    eprintln!("error: could not start debug server: {e}");
                    return Err(e);
                }
            };
            app::run_interpreted(
                args.file.as_deref(),
                args.websocket.clone(),
                args.ws_user.clone(),
                Some(debug_hook),
            )
        }
        Commands::Add(args) => {
            cli::add(args.name.as_deref(), &args.target, args.template.as_deref())
        }
        Commands::Rm(args) => cli::rm(&args.name),
        Commands::List(args) => cli::list(args.file.as_deref()),
        Commands::Test(args) => {
            let path = args.file.as_deref().unwrap_or("");
            cli::check(path)
        }
    }
}
