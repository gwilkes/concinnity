// src/metal/parallel_encoder.rs
//
// Send/Sync shims for parallel per-pass command-buffer recording. The
// render-graph executor in `metal/graph_exec.rs` fans non-composite
// passes onto rayon workers; each worker mints its own
// `MTLCommandBuffer`, encodes its pass, and hands the
// encoded-but-uncommitted buffer back through a per-pass slot. The main
// thread then commits the slots in topological pass order, so the single
// command queue's FIFO commit order is the GPU execution order; no
// `MTLEvent` wait/signal pairs are involved.
//
// `MtlContext` and the objc2 protocol objects it stores are not Send/Sync
// in Rust's type system: Apple's API contract makes shared, read-only
// access to Metal resources thread-safe, but objc2 cannot encode that
// without a hand claim. The wrappers below adopt that claim at the
// parallel-dispatch boundary. Workers reach `&MtlContext` through
// `ParallelCtxRef::as_ctx()` for strictly read-only encode work; the
// lone `&mut self` mutations (`frame_stats.draw_calls`,
// `particle_last_elapsed`, `particle_frame_index`, the per-emitter
// `spawn_state`) all happen on the main thread before the fan-out.

#![allow(clippy::incompatible_msrv)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandBuffer;

use super::context::MtlContext;

// `Send` wrapper around an encoded-but-uncommitted `MTLCommandBuffer`.
// Workers in the parallel-dispatch fan-out create + encode their cmd buf,
// then hand it back to the main thread to commit in topological order.
// Apple's command buffer is safe to transfer across thread boundaries as
// long as only one thread drives the encoder at a time; objc2 just lacks
// an auto Send impl.
pub(super) struct SendableCmdBuf(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: Each `SendableCmdBuf` is owned by exactly one worker for the
// span of encoding, then moves back to the main thread for the commit.
// No two threads access the inner `Retained` simultaneously.
unsafe impl Send for SendableCmdBuf {}

// A `Send + Sync` handle to a `&MtlContext` borrow. Worker closures use it
// to reach the immutable subset of `MtlContext` they need while encoding
// commands into their own command buffer.
//
// # Safety
//
// Construction takes `&'a MtlContext`. The wrapper is only used inside the
// parallel-encoder fan-out in `graph_exec.rs`, which joins all workers
// before the outer borrow returns. Workers perform strictly read-only field
// access; the encode helpers they call (`encode_main_pass`,
// `encode_shadow_pass`, …) all take `&self`. Apple's Metal device, queue,
// buffers, textures, and pipeline states are thread-safe for shared read.
// The wrapper itself is the shared generic shim in `gfx::parallel_ctx`; this
// alias keeps the `ParallelCtxRef<'a>` spelling at the metal call sites.
pub(super) type ParallelCtxRef<'a> = crate::gfx::parallel_ctx::ParallelCtxRef<'a, MtlContext>;

// SAFETY: see the type-level safety contract above. Workers reach `&MtlContext`
// for strictly read-only encode work; the lone `&mut self` mutations
// (`frame_stats.draw_calls`, `particle_last_elapsed`, `particle_frame_index`,
// the per-emitter `spawn_state`) all happen on the main thread before the
// fan-out, and Apple's Metal device, queue, buffers, textures, and pipeline
// states are thread-safe for shared read.
unsafe impl crate::gfx::parallel_ctx::ParallelEncodeCtx for MtlContext {}
