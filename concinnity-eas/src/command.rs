// concinnity-eas/src/command.rs
//
// Deferred structural-change buffer. Systems record spawn / despawn / edit
// intents without touching shared storage, so recording is parallelism-safe;
// the intents are applied later at a sync point under exclusive access, when
// reordering a column is safe.
//
// Because this crate is closed over no concrete component set, a typed
// component insert is expressed here as a boxed closure over the engine's World
// (the `Run` command); the engine supplies those closures from its
// ComponentStorage. Despawn is routed through the `CommandTarget` trait the
// World implements, so cascading a despawn to every column stays an engine
// concern. Fresh entity ids are reserved lock-free via `Entities::reserve`, so
// a spawned handle is usable in the same frame it is recorded.

use crate::entity::{Entities, Entity};

type WorldFn<W> = Box<dyn FnOnce(&mut W) + Send>;

// What the World must provide for a command queue to apply against it. The
// engine's World implements this once the EAS World lands; the `Run` closures
// receive `&mut W` directly and need nothing from the trait.
pub trait CommandTarget {
    fn despawn_entity(&mut self, entity: Entity);
}

pub enum Command<W> {
    Despawn(Entity),
    Run(WorldFn<W>),
}

#[derive(Default)]
pub struct CommandQueue<W> {
    commands: Vec<Command<W>>,
}

impl<W> CommandQueue<W> {
    pub fn new() -> CommandQueue<W> {
        CommandQueue {
            commands: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    // Borrow the queue for recording, pairing it with the entity allocator so
    // `spawn` can reserve fresh ids.
    pub fn recorder<'a>(&'a mut self, entities: &'a Entities) -> Commands<'a, W> {
        Commands {
            entities,
            queue: self,
        }
    }

    fn push(&mut self, command: Command<W>) {
        self.commands.push(command);
    }

    // Apply every recorded command in record order against the World, draining
    // the queue. Record order (not completion order) keeps application
    // deterministic.
    pub fn apply(&mut self, world: &mut W)
    where
        W: CommandTarget,
    {
        for command in self.commands.drain(..) {
            match command {
                Command::Despawn(entity) => world.despawn_entity(entity),
                Command::Run(f) => f(world),
            }
        }
    }
}

pub struct Commands<'a, W> {
    entities: &'a Entities,
    queue: &'a mut CommandQueue<W>,
}

impl<W> Commands<'_, W> {
    // Reserve a fresh entity id, usable immediately. Components are populated by
    // queuing a `run` closure that inserts them.
    pub fn spawn(&mut self) -> Entity {
        self.entities.reserve()
    }

    // Queue a despawn, applied at the next sync point.
    pub fn despawn(&mut self, entity: Entity) {
        self.queue.push(Command::Despawn(entity));
    }

    // Queue an arbitrary World mutation, applied at the next sync point. This is
    // the closed-world escape hatch the engine uses for typed component
    // insert and remove.
    pub fn run(&mut self, f: impl FnOnce(&mut W) + Send + 'static) {
        self.queue.push(Command::Run(Box::new(f)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct MockWorld {
        despawned: Vec<Entity>,
        log: Vec<u32>,
    }

    impl CommandTarget for MockWorld {
        fn despawn_entity(&mut self, entity: Entity) {
            self.despawned.push(entity);
        }
    }

    #[test]
    fn spawn_reserves_a_usable_handle() {
        let entities = Entities::new();
        let mut queue: CommandQueue<MockWorld> = CommandQueue::new();
        let mut commands = queue.recorder(&entities);
        let a = commands.spawn();
        let b = commands.spawn();
        assert_ne!(a.index(), b.index());
    }

    #[test]
    fn commands_apply_in_record_order() {
        let entities = Entities::new();
        let mut queue: CommandQueue<MockWorld> = CommandQueue::new();
        let target = entities.reserve();
        {
            let mut commands = queue.recorder(&entities);
            commands.run(|w: &mut MockWorld| w.log.push(1));
            commands.despawn(target);
            commands.run(|w: &mut MockWorld| w.log.push(2));
        }
        assert_eq!(queue.len(), 3);

        let mut world = MockWorld::default();
        queue.apply(&mut world);
        assert_eq!(world.log, vec![1, 2]);
        assert_eq!(world.despawned, vec![target]);
        // Applying drains the queue.
        assert!(queue.is_empty());
    }
}
