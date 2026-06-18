// src/directx/parallel_encoder.rs
//
// Send/Sync shims for parallel per-pass command-list recording. The
// render-graph executor in `directx/graph_exec.rs` fans non-composite
// passes onto rayon workers; each worker resets its assigned pass's
// allocator + cmd list, encodes its pass, and closes the cmd list. The
// main thread then submits every closed cmd list in topological pass
// order via `ExecuteCommandLists`.
//
// `DxContext` and the windows-crate COM smart pointers it stores are
// not Send/Sync in Rust's type system; Microsoft's API contract makes
// shared, read-only access to D3D12 resources thread-safe (commands
// being recorded against *separate* command lists with *separate*
// allocators is the canonical free-threaded pattern), but the windows
// crate cannot encode that without a hand claim. The wrappers below
// adopt that claim at the parallel-dispatch boundary. Workers reach
// `&DxContext` through `ParallelCtxRef::as_ctx()` for strictly
// read-only encode work; the only mutations during encode are bumps to
// `draw_calls_accum` (atomic), the `deformed_primed` velocity-priming
// store (atomic), and inline command-list recording into the worker's
// own dedicated cmd list (single-writer per allocator).
//
// Mirrors `metal/parallel_encoder.rs`.

use windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList;

use super::context::DxContext;

// `Send` wrapper around a closed but unsubmitted command list. Workers
// in the parallel-dispatch fan-out reset their cmd list, encode their
// pass into it, close it, then hand it back to the main thread to
// submit in topological order. D3D12 command lists are free-threaded
// for "record on one thread, submit on another" once closed; the
// windows crate just lacks an auto Send impl.
pub(super) struct SendableCmdList(pub ID3D12GraphicsCommandList);

// SAFETY: Each `SendableCmdList` is owned by exactly one worker for the
// span of encoding, then moves back to the main thread for submission.
// No two threads access the inner handle simultaneously.
unsafe impl Send for SendableCmdList {}

// A `Send + Sync` handle to a `&DxContext` borrow. Worker closures use
// it to reach the immutable subset of `DxContext` they need while
// recording commands into their own command list.
//
// # Safety
//
// Construction takes `&'a DxContext`. The wrapper is only used inside
// the parallel-encoder fan-out in `graph_exec.rs`, which joins all
// workers before the outer borrow returns. Workers perform strictly
// read-only field access; the encode helpers they call (`encode_main_pass`,
// `encode_shadow_pass`, …) all take `&self`. D3D12 resources (root
// signatures, PSOs, descriptor heaps, mapped upload buffers, fence)
// are thread-safe for shared read per Microsoft's free-threading rules
// for ID3D12Device-derived objects.
//
// The contract is **read-only**: any mutable field access from a
// worker (RefCell::borrow_mut, Cell::set on a frame-relevant field)
// is unsound. The audited mutations during encode are limited to the
// `draw_calls_accum: AtomicU32` bump via `inc_draw_calls` and the
// `skinned.deformed_primed: AtomicBool` priming store in the G-buffer
// pass, both already thread-safe.
// The wrapper itself is the shared generic shim in `gfx::parallel_ctx`; this
// alias keeps the `ParallelCtxRef<'a>` spelling at the directx call sites.
pub(super) type ParallelCtxRef<'a> = crate::gfx::parallel_ctx::ParallelCtxRef<'a, DxContext>;

// SAFETY: see the type-level safety contract above. The contract is
// **read-only**: any mutable field access from a worker (RefCell::borrow_mut,
// Cell::set on a frame-relevant field) is unsound. The audited mutations during
// encode are limited to the `draw_calls_accum: AtomicU32` bump via
// `inc_draw_calls` and the `skinned.deformed_primed: AtomicBool` priming store
// in the G-buffer pass, both already thread-safe; D3D12 device-derived objects
// (root signatures, PSOs, descriptor heaps, mapped upload buffers, fence) are
// thread-safe for shared read per Microsoft's free-threading rules.
unsafe impl crate::gfx::parallel_ctx::ParallelEncodeCtx for DxContext {}

// Index into the per-pass `pass_allocators` / `pass_cmd_lists` pools
// in `DxContext`. Layout: `frame_idx * PASS_COUNT + (PassId as usize)`.
pub(super) fn pool_index(frame_idx: usize, pass_id: crate::gfx::render_graph::PassId) -> usize {
    frame_idx * crate::gfx::render_graph::PASS_COUNT + pass_id as usize
}
