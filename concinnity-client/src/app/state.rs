// src/app/state.rs
use crate::blob;
use crate::ecs::{StepResult, World};
use crate::result::CnResult;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppStatus {
    Created,
    Started,
    #[allow(dead_code)]
    Stopped,
}

#[derive(Debug)]
pub struct App {
    status: AppStatus,
    world: World,
    shutdown: CancellationToken,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self {
            status: AppStatus::Created,
            world: World::new_empty(),
            shutdown: CancellationToken::new(),
        }
    }

    pub fn new_with_token(shutdown: CancellationToken) -> Self {
        Self {
            status: AppStatus::Created,
            world: World::new_empty(),
            shutdown,
        }
    }

    // load assets and blob payload data from the primary blob and
    // populate the world. Replaces any previously loaded world
    pub fn load_blob(&mut self) -> Result<(), CnResult> {
        let (assets, blob_data) = blob::load()?;

        let mut world = World::new(blob_data);
        for asset in assets {
            world.add(asset);
        }
        self.world = world;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    // clone of the root cancellation token. Pass this to systems or the
    // ctrl+c handler so they all share a single cancellation source
    pub fn shutdown_token(&self) -> CancellationToken {
        self.shutdown.clone()
    }

    pub fn start(&mut self) -> Result<(), CnResult> {
        if self.status != AppStatus::Created {
            tracing::error!("App must be in Created state to start");
            return Err(CnResult::InvalidState);
        }
        self.world.start()?;
        self.status = AppStatus::Started;
        Ok(())
    }

    // Replace the current world and reset to Created so start() can be called again.
    // Used to load a new scene at runtime.
    #[allow(dead_code)]
    pub fn load_world(&mut self, world: World) {
        self.world = world;
        self.status = AppStatus::Created;
    }

    // single world step, for callers that drive their own outer loop
    // (e.g. run_loop_macos in crate::app::run, which interleaves CFRunLoop pumps)
    pub fn world_step(&mut self) -> StepResult {
        self.world.step()
    }
}
