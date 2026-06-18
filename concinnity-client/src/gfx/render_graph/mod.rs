// src/gfx/render_graph/mod.rs
//
// Backend-agnostic render graph. Types, builder, and compile pass with
// unit tests; the per-backend executors live alongside each backend
// (`metal/graph_exec.rs`, `vulkan/graph_exec.rs`, `directx/graph_exec.rs`)
// and consume the `CompiledGraph` this module produces.
//
// The graph deliberately stops short of allocating GPU resources; that
// stays backend-owned. The graph's job is to:
//
//   * give every pass a stable identity ([`PassId`]),
//   * track read / write declarations so the compile pass can derive
//     pass order, transient resource lifetimes, and per-pass barriers,
//   * surface a `CompiledGraph` the per-backend executor consumes.

#![allow(unused_imports)]

mod alias;
mod builder;
mod compile;
mod frame;
mod passes;
mod types;

pub use alias::{AliasPlan, AliasSlot, plan_aliasing};
pub use builder::{GraphBuilder, PassBuilder, ResourceVersion};
pub use compile::{CompiledGraph, CompiledPass, CompiledResource, GraphError};
pub use frame::{FOG_FROXEL_X, FOG_FROXEL_Y, FOG_FROXEL_Z, FrameGraphInputs, build_frame_graph};
pub use passes::{PASS_COUNT, PASS_NAMES, PassId};
pub use types::{
    BarrierOp, BufferDesc, BufferHandle, BufferUsage, GraphResourceClass, PassKind, PassRange,
    PixelFormat, ReadStages, ResourceId, ResourceOrigin, ResourceState, TextureDesc, TextureHandle,
    TextureSize, TextureUsage,
};
