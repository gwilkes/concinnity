// concinnity-eas/src/lib.rs
//
// Concinnity EAS (Entity-Asset-System): the engine's closed-world ECS
// mechanism, in its own crate so it carries no engine domain types. It provides
// the generic primitives only: entities, typed storage columns, change ticks,
// component masks and a join index, resources, events, the deferred command
// buffer, and system access sets for conflict-free scheduling. The concrete
// component
// set is registered by concinnity-core through the define_components! macro;
// nothing here knows about meshes, blobs, or rendering.
//
// Closed-world by design: EAS stores only the component types the project
// defines, registered at compile time. There is no TypeId-keyed type erasure
// and no open-world insert of arbitrary external types.

mod access;
mod column;
mod command;
mod entity;
mod event;
mod join;
#[cfg(test)]
mod join_bench;
mod mask;
mod resource;
mod sparse;
mod tick;

pub use access::Access;
pub use column::{Column, StorageKind};
pub use command::{Command, CommandQueue, CommandTarget, Commands};
pub use entity::{Entities, Entity};
pub use event::{EventCursor, Events};
pub use join::JoinIndex;
pub use mask::{ComponentId, ComponentMask};
pub use resource::Resources;
pub use sparse::SparseColumn;
pub use tick::{MAX_CHANGE_AGE, Tick};
