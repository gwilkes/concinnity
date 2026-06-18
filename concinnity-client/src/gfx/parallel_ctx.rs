// src/gfx/parallel_ctx.rs
//
// Generic Send/Sync shim for parallel per-pass command recording, shared by all
// three backend executors (`{metal,directx,vulkan}/graph_exec.rs`). Each backend
// fans its non-composite render-graph passes onto worker threads; every worker
// records into its own command buffer/list and reaches the immutable subset of
// the backend context it needs through a `ParallelCtxRef`.
//
// The backend context types are not Send/Sync in Rust's type system: they hold
// COM smart pointers, objc2 protocol objects, RefCells, and the like. The
// graphics APIs nonetheless permit shared, read-only access to device-derived
// resources from many threads. A backend adopts that claim for its own context
// type with a single `unsafe impl ParallelEncodeCtx`, where it documents the
// audit of every interior-mutable field reachable during encode. The Send/Sync
// impls on `ParallelCtxRef` below are then keyed off that marker, so the unsafe
// reasoning lives in one auditable place per backend instead of being repeated
// on three structurally identical wrapper types.

use std::marker::PhantomData;

// Marker for a backend context that may be shared, read-only, across the
// parallel-encode worker fan-out. Implementing this is an unsafe claim that
// concurrent `&Self` access during command recording is sound: the graphics API
// allows shared read of device-derived resources, and every interior-mutable
// field reachable during encode is either atomic or hoisted out of the fan-out
// before it begins. Each backend's impl carries that audit (see the module docs
// in each `*/parallel_encoder.rs`).
//
// The safety contract lives in the `//` comments above rather than a `///`
// `# Safety` section because this crate does not use rustdoc outside the asset
// API; the lint that wants the rustdoc form is suppressed accordingly.
#[allow(clippy::missing_safety_doc)]
pub(crate) unsafe trait ParallelEncodeCtx {}

// A handle to a `&'a T` borrow that is Send + Sync when `T: ParallelEncodeCtx`.
// Worker closures use it to reach the immutable subset of the backend context
// they need while recording commands into their own command buffer/list.
//
// # Safety
//
// `new` takes `&'a T`; the lifetime is preserved through `PhantomData`. The
// wrapper is only used inside each backend's parallel-encoder fan-out in
// `graph_exec.rs`, which joins all workers before the outer borrow returns. The
// Send/Sync claim rests entirely on the `T: ParallelEncodeCtx` marker.
pub(crate) struct ParallelCtxRef<'a, T> {
    ptr: *const T,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> ParallelCtxRef<'a, T> {
    pub(crate) fn new(ctx: &'a T) -> Self {
        Self {
            ptr: ctx as *const T,
            _marker: PhantomData,
        }
    }

    pub(crate) fn as_ctx(&self) -> &T {
        // SAFETY: lifetime tied to `'a` via PhantomData; the pointer is a
        // straight reborrow of `&'a T`, and the workers that hold this are all
        // joined before that borrow ends (see each backend's graph_exec).
        unsafe { &*self.ptr }
    }
}

impl<T> Clone for ParallelCtxRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ParallelCtxRef<'_, T> {}

// SAFETY: `T: ParallelEncodeCtx` is the backend's audited claim that shared,
// read-only `&T` access across the encode fan-out is sound. The wrapper only
// ever hands out `&T` (via `as_ctx`), so Send + Sync follow from that claim.
unsafe impl<T: ParallelEncodeCtx> Send for ParallelCtxRef<'_, T> {}
unsafe impl<T: ParallelEncodeCtx> Sync for ParallelCtxRef<'_, T> {}
