// src/directx/barrier_translate.rs
//
// Translate the render graph's coarse `ResourceState`, for a given resource
// class, into the concrete `D3D12_RESOURCE_STATES` the executor passes to
// `transition_barrier`. The graph tracks only Undefined / Read / Write; the
// resource class (assigned by the executor's resolver) disambiguates what a
// `Write` means: a colour render target writes `RENDER_TARGET`, a depth target
// writes `DEPTH_WRITE`.
//
// A `Read` maps by the consuming-stage union (`ReadStages`) carried on the
// barrier, not the class: a fragment consumer needs `PIXEL_SHADER_RESOURCE`, a
// compute consumer `NON_PIXEL_SHADER_RESOURCE`, and a resource read in both
// stages on one version needs both bits so the single transition makes the
// write visible to each. The class is irrelevant once a resource is being read.
//
// The `StorageImage` class covers a compute-written, fragment-sampled UAV
// resource (`fog_froxel_volume`): its `Write` is `UNORDERED_ACCESS` and its
// `Read` resolves through the same stage union (today fragment-only).
//
// `Undefined` never reaches here as a real transition: the executor resolves a
// barrier whose `from` is Undefined to the resource's resting state (returned
// by the resolver) before translating, so the first per-frame transition has a
// `from` matching the resource's actual state. The arm below is a fallback.

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::render_graph::{GraphResourceClass, ReadStages, ResourceState};

// Map a `Read`'s consuming-stage union to the matching shader-resource states.
// FRAGMENT -> `PIXEL_SHADER_RESOURCE`, COMPUTE -> `NON_PIXEL_SHADER_RESOURCE`,
// both -> both bits. An empty union (no Read side, or a resource no consumer
// reads) falls back to `PIXEL_SHADER_RESOURCE`, the historical default before
// stages were carried; the deriver never emits a `Read` barrier with an empty
// union, so the fallback is purely defensive.
fn read_state(stages: ReadStages) -> D3D12_RESOURCE_STATES {
    match (
        stages.contains(ReadStages::FRAGMENT),
        stages.contains(ReadStages::COMPUTE),
    ) {
        (true, true) => {
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        }
        (false, true) => D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        // FRAGMENT-only and the empty-union fallback both map to the
        // pixel-shader state.
        (true, false) | (false, false) => D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    }
}

pub(super) fn d3d12_state(
    class: GraphResourceClass,
    state: ResourceState,
    read_stages: ReadStages,
) -> D3D12_RESOURCE_STATES {
    match (class, state) {
        (_, ResourceState::Undefined) => D3D12_RESOURCE_STATE_COMMON,
        (_, ResourceState::Read) => read_state(read_stages),
        (GraphResourceClass::ColorTarget, ResourceState::Write) => {
            D3D12_RESOURCE_STATE_RENDER_TARGET
        }
        (GraphResourceClass::DepthTarget, ResourceState::Write) => D3D12_RESOURCE_STATE_DEPTH_WRITE,
        (GraphResourceClass::StorageImage, ResourceState::Write) => {
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        }
    }
}

// Resolve a graph barrier `from -> to` for a resource of `class` whose resting
// (created / cross-frame-restored) state is `resting`, into the concrete
// `(before, after)` D3D12 states the executor passes to `transition_barrier`. A
// first-use `Undefined` source resolves to `resting` so the before-state matches
// the resource's real state (the debug layer rejects a mismatch). Returns `None`
// when before == after: a no-op the executor skips, e.g. a depth or storage
// producer whose resting state already equals its write state.
//
// `read_stages` is the barrier's consuming-stage union (see `ReadStages`); it
// applies to whichever side is `Read` (the `to` of a consumer transition or the
// `from` of a Read -> Write WAR), and is ignored for the Write / Undefined side,
// so threading the single union through both `d3d12_state` calls is correct.
pub(super) fn d3d12_transition(
    class: GraphResourceClass,
    resting: D3D12_RESOURCE_STATES,
    from: ResourceState,
    to: ResourceState,
    read_stages: ReadStages,
) -> Option<(D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> {
    let before = if from == ResourceState::Undefined {
        resting
    } else {
        d3d12_state(class, from, read_stages)
    };
    let after = d3d12_state(class, to, read_stages);
    (before != after).then_some((before, after))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every graph-driven resource today is read in the fragment stage, so the
    // existing class-mapping assertions pass the fragment union.
    const FRAG: ReadStages = ReadStages::FRAGMENT;

    #[test]
    fn class_state_mapping_is_pinned() {
        // Colour target: ao_output. Sampled read, render-target write.
        assert_eq!(
            d3d12_state(GraphResourceClass::ColorTarget, ResourceState::Read, FRAG),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );
        assert_eq!(
            d3d12_state(GraphResourceClass::ColorTarget, ResourceState::Write, FRAG),
            D3D12_RESOURCE_STATE_RENDER_TARGET
        );
        // Depth target: shadow_map. Sampled read, depth write.
        assert_eq!(
            d3d12_state(GraphResourceClass::DepthTarget, ResourceState::Read, FRAG),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );
        assert_eq!(
            d3d12_state(GraphResourceClass::DepthTarget, ResourceState::Write, FRAG),
            D3D12_RESOURCE_STATE_DEPTH_WRITE
        );
        // Storage image: fog_froxel_volume. Compute write is UNORDERED_ACCESS;
        // the fog fragment samples it, so read is the shared
        // PIXEL_SHADER_RESOURCE.
        assert_eq!(
            d3d12_state(GraphResourceClass::StorageImage, ResourceState::Write, FRAG),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
        );
        assert_eq!(
            d3d12_state(GraphResourceClass::StorageImage, ResourceState::Read, FRAG),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );
    }

    #[test]
    fn read_state_maps_by_consuming_stage() {
        // The Read translation is class-independent: it follows the consuming
        // stage union. Fragment-only -> pixel shader; compute-only -> non-pixel
        // shader; both -> both bits (so one transition makes the producing write
        // visible to a compute consumer and a fragment consumer on one version,
        // the hdr_resolve case the union exists for).
        assert_eq!(
            d3d12_state(
                GraphResourceClass::ColorTarget,
                ResourceState::Read,
                ReadStages::FRAGMENT
            ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );
        assert_eq!(
            d3d12_state(
                GraphResourceClass::ColorTarget,
                ResourceState::Read,
                ReadStages::COMPUTE
            ),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        assert_eq!(
            d3d12_state(
                GraphResourceClass::ColorTarget,
                ResourceState::Read,
                ReadStages::FRAGMENT | ReadStages::COMPUTE
            ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
        );
        // Empty union falls back to the pixel-shader default.
        assert_eq!(
            d3d12_state(
                GraphResourceClass::ColorTarget,
                ResourceState::Read,
                ReadStages::empty()
            ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
        );
    }

    #[test]
    fn transition_resolves_resting_and_skips_no_ops() {
        // ao_output (ColorTarget, resting PIXEL_SHADER_RESOURCE): both producer
        // (PSR -> RT) and consumer (RT -> PSR) are real transitions.
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::ColorTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Undefined,
                ResourceState::Write,
                FRAG,
            ),
            Some((
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET
            ))
        );
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::ColorTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Write,
                ResourceState::Read,
                FRAG,
            ),
            Some((
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            ))
        );
        // shadow_map (DepthTarget, resting PIXEL_SHADER_RESOURCE): the
        // cross-frame reset is folded into the producer, so it is a real
        // PSR -> DEPTH_WRITE transition; the consumer is DEPTH_WRITE -> PSR.
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::DepthTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Undefined,
                ResourceState::Write,
                FRAG,
            ),
            Some((
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE
            ))
        );
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::DepthTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Write,
                ResourceState::Read,
                FRAG,
            ),
            Some((
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            ))
        );
        // fog_froxel_volume (StorageImage, resting PIXEL_SHADER_RESOURCE): the
        // cross-frame reset is folded into the producer, so it is a real
        // PSR -> UAV open; the consumer is the UAV -> PSR close.
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::StorageImage,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Undefined,
                ResourceState::Write,
                FRAG,
            ),
            Some((
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            ))
        );
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::StorageImage,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Write,
                ResourceState::Read,
                FRAG,
            ),
            Some((
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
            ))
        );
        // Generic no-op: a resource whose resting state already equals its write
        // state (e.g. a target left in DEPTH_WRITE between frames) skips the
        // producer transition. No migrated resource rests this way anymore, but
        // the translator still collapses it.
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::DepthTarget,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                ResourceState::Undefined,
                ResourceState::Write,
                FRAG,
            ),
            None
        );
    }

    #[test]
    fn transition_threads_compute_read_stage() {
        // A compute consumer of a colour resource (the hdr_resolve / AutoExposure
        // shape, once that resource migrates): the consumer Write -> Read resolves
        // its after-state to NON_PIXEL_SHADER_RESOURCE off the COMPUTE union, and
        // a mixed compute+fragment run resolves to both bits.
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::ColorTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Write,
                ResourceState::Read,
                ReadStages::COMPUTE,
            ),
            Some((
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            ))
        );
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::ColorTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Write,
                ResourceState::Read,
                ReadStages::FRAGMENT | ReadStages::COMPUTE,
            ),
            Some((
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            ))
        );
        // The WAR side: a Read -> Write whose prior run read in the compute stage
        // resolves its before-state to NON_PIXEL_SHADER_RESOURCE.
        assert_eq!(
            d3d12_transition(
                GraphResourceClass::ColorTarget,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ResourceState::Read,
                ResourceState::Write,
                ReadStages::COMPUTE,
            ),
            Some((
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_RENDER_TARGET
            ))
        );
    }
}
