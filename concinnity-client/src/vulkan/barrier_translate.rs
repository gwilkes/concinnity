// src/vulkan/barrier_translate.rs
//
// Translate the render graph's coarse `ResourceState`, for a given resource
// class, into the concrete Vulkan (layout, access, stage) triple the executor
// feeds into an image memory barrier. The graph tracks only Undefined / Read /
// Write; the resource class (assigned by the executor's resolver) disambiguates
// what a `Write` means: a colour target writes COLOR_ATTACHMENT, a depth target
// writes DEPTH_STENCIL_ATTACHMENT. Both are sampled (SHADER_READ_ONLY) when
// read.
//
// For a barrier `from -> to`, the executor uses `from`'s triple for the source
// and `to`'s triple for the destination, and skips the barrier when the two
// layouts match (a no-op).
//
// A `Read`'s layout is SHADER_READ_ONLY either way, but its pipeline stage
// follows the consuming-stage union (`ReadStages`) carried on the barrier: a
// fragment consumer waits in FRAGMENT_SHADER, a compute consumer in
// COMPUTE_SHADER, and a resource read in both stages on one version waits in
// both so the single transition makes the producing write visible to each.
//
// The `Undefined` (first-use) triple is class-specific and stage-fixed (it is a
// resting state, not a genuine read, so it does not consult `read_stages`). A
// colour target like `ao_output` is not pre-transitioned, so its first use
// discards (UNDEFINED layout). A depth target like `shadow_map` rests sampled
// (SHADER_READ_ONLY) between frames, where the prior frame's Main consumer left
// it, so its `Undefined` resolves to that sampled layout and the producer
// (Shadow) barrier is the real SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT
// cross-frame reset (folded off the old inline restore).
//
// A `StorageImage` like `fog_froxel_volume` is compute-written (GENERAL, in the
// compute stage) and fragment-sampled. Init rests it in SHADER_READ_ONLY (in
// the fragment stage, where the prior frame's Fog consumer left it), so its
// `Undefined` resolves to that resting layout + fragment stage and the producer
// (FogFroxel) barrier is a real SHADER_READ_ONLY -> GENERAL open the executor
// emits, with the consumer (Fog) barrier the matching GENERAL -> SHADER_READ_ONLY
// close.

use ash::vk;

use crate::gfx::render_graph::{GraphResourceClass, ReadStages, ResourceState};

// Map a `Read`'s consuming-stage union to the pipeline stages the transition
// must synchronise against. FRAGMENT -> FRAGMENT_SHADER, COMPUTE ->
// COMPUTE_SHADER, both -> both. An empty union (no Read side, or a resource no
// consumer reads) falls back to FRAGMENT_SHADER, the historical resting stage;
// the deriver never emits a `Read` barrier with an empty union, so the fallback
// is purely defensive.
fn read_stage_mask(stages: ReadStages) -> vk::PipelineStageFlags {
    let mut mask = vk::PipelineStageFlags::empty();
    if stages.contains(ReadStages::FRAGMENT) {
        mask |= vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    if stages.contains(ReadStages::COMPUTE) {
        mask |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    if mask.is_empty() {
        mask = vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    mask
}

pub(super) fn vk_state(
    class: GraphResourceClass,
    state: ResourceState,
    read_stages: ReadStages,
) -> (vk::ImageLayout, vk::AccessFlags, vk::PipelineStageFlags) {
    let depth_attachment = (
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
    );
    // The resting sampled triple, used for the `Undefined` first-use of a
    // resource whose cross-frame reset is folded into its producer barrier: the
    // depth `shadow_map` and the storage `fog_froxel_volume` both rest sampled
    // between frames (where the prior frame's Main / Fog consumer left them), so
    // their producer barrier opens from here. Its stage is fixed at the fragment
    // shader (the resting stage the prior consumer left it in), independent of
    // the current barrier's `read_stages` (which is empty on that producer open).
    let sampled_resting = (
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
    );
    match (class, state) {
        // Colour target first use: discard prior contents.
        (GraphResourceClass::ColorTarget, ResourceState::Undefined) => (
            vk::ImageLayout::UNDEFINED,
            vk::AccessFlags::empty(),
            vk::PipelineStageFlags::TOP_OF_PIPE,
        ),
        (GraphResourceClass::ColorTarget, ResourceState::Write) => (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        // Depth target (shadow_map) rests sampled between frames, so its producer
        // barrier is the real SHADER_READ_ONLY -> DEPTH_STENCIL_ATTACHMENT
        // cross-frame reset (folded off the old inline restore); its write is the
        // depth attachment.
        (GraphResourceClass::DepthTarget, ResourceState::Undefined) => sampled_resting,
        (GraphResourceClass::DepthTarget, ResourceState::Write) => depth_attachment,
        // Storage image first use == its resting sampled layout (init rests the
        // froxel volume in SHADER_READ_ONLY); its write is GENERAL in the compute
        // stage.
        (GraphResourceClass::StorageImage, ResourceState::Undefined) => sampled_resting,
        (GraphResourceClass::StorageImage, ResourceState::Write) => (
            vk::ImageLayout::GENERAL,
            vk::AccessFlags::SHADER_WRITE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        // Every class reads as a sampled image; the stage follows the consuming
        // run's union so a compute consumer synchronises in COMPUTE_SHADER.
        (_, ResourceState::Read) => (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::SHADER_READ,
            read_stage_mask(read_stages),
        ),
    }
}

// Resolve a graph barrier `from -> to` for a resource of `class` into the
// concrete (old, new, src_access, dst_access, src_stage, dst_stage) the executor
// feeds into an image memory barrier + `cmd_pipeline_barrier`. The first-use
// `Undefined` source maps to the class's resting layout (see `vk_state`).
// Returns `None` when old == new: a no-op the executor skips, e.g. a depth
// producer whose resting layout already equals its write layout.
//
// `read_stages` is the barrier's consuming-stage union (see `ReadStages`); it
// applies to whichever side is `Read` (the `to` of a consumer transition or the
// `from` of a Read -> Write WAR), driving that side's pipeline stage. The Write /
// Undefined side ignores it, so threading the single union through both
// `vk_state` calls is correct.
type VkTransition = (
    vk::ImageLayout,
    vk::ImageLayout,
    vk::AccessFlags,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
);

pub(super) fn vk_transition(
    class: GraphResourceClass,
    from: ResourceState,
    to: ResourceState,
    read_stages: ReadStages,
) -> Option<VkTransition> {
    let (old, src_access, src_stage) = vk_state(class, from, read_stages);
    let (new, dst_access, dst_stage) = vk_state(class, to, read_stages);
    (old != new).then_some((old, new, src_access, dst_access, src_stage, dst_stage))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every graph-driven resource today is read in the fragment stage, so the
    // existing assertions pass the fragment union.
    const FRAG: ReadStages = ReadStages::FRAGMENT;

    #[test]
    fn class_state_mapping_is_pinned() {
        // Colour target: ao_output. First use discards; write is colour
        // attachment; read is sampled.
        assert_eq!(
            vk_state(
                GraphResourceClass::ColorTarget,
                ResourceState::Undefined,
                FRAG
            )
            .0,
            vk::ImageLayout::UNDEFINED
        );
        let (layout, access, stage) =
            vk_state(GraphResourceClass::ColorTarget, ResourceState::Write, FRAG);
        assert_eq!(layout, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(access, vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        assert_eq!(stage, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);

        // Depth target: shadow_map. First use == its resting sampled layout (the
        // cross-frame reset is folded into the producer), so a producer
        // Undefined -> Write is a real SHADER_READ_ONLY -> DEPTH_STENCIL open;
        // write is the depth attachment; read is sampled.
        let read = vk_state(GraphResourceClass::DepthTarget, ResourceState::Read, FRAG);
        assert_eq!(
            vk_state(
                GraphResourceClass::DepthTarget,
                ResourceState::Undefined,
                FRAG
            ),
            read
        );
        let write = vk_state(GraphResourceClass::DepthTarget, ResourceState::Write, FRAG);
        assert_eq!(write.0, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        assert_eq!(read.0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(
            vk_state(GraphResourceClass::ColorTarget, ResourceState::Read, FRAG),
            read
        );

        // Storage image: fog_froxel_volume. First use == resting sampled layout,
        // so the producer barrier is a real SHADER_READ_ONLY -> GENERAL open;
        // write is GENERAL in the compute stage; read is sampled.
        assert_eq!(
            vk_state(
                GraphResourceClass::StorageImage,
                ResourceState::Undefined,
                FRAG
            ),
            vk_state(GraphResourceClass::StorageImage, ResourceState::Read, FRAG)
        );
        let (sl, sa, ss) = vk_state(GraphResourceClass::StorageImage, ResourceState::Write, FRAG);
        assert_eq!(sl, vk::ImageLayout::GENERAL);
        assert_eq!(sa, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(ss, vk::PipelineStageFlags::COMPUTE_SHADER);
        assert_eq!(
            vk_state(GraphResourceClass::StorageImage, ResourceState::Read, FRAG).0,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
    }

    #[test]
    fn read_stage_follows_consuming_union() {
        // The Read layout is always SHADER_READ_ONLY, but the stage tracks the
        // consuming union: fragment-only waits in FRAGMENT_SHADER, compute-only in
        // COMPUTE_SHADER, both in both (the hdr_resolve case the union exists for).
        // Layout + access are unchanged across the three.
        let frag = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::FRAGMENT,
        );
        assert_eq!(frag.0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(frag.2, vk::PipelineStageFlags::FRAGMENT_SHADER);

        let comp = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::COMPUTE,
        );
        assert_eq!(comp.0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(comp.2, vk::PipelineStageFlags::COMPUTE_SHADER);

        let both = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::FRAGMENT | ReadStages::COMPUTE,
        );
        assert_eq!(
            both.2,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        );

        // Empty union falls back to the fragment stage; this is the storage
        // resting stage the producer-open's source side keeps.
        let empty = vk_state(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ReadStages::empty(),
        );
        assert_eq!(empty.2, vk::PipelineStageFlags::FRAGMENT_SHADER);
    }

    #[test]
    fn transition_resolves_migrated_producers_and_consumers() {
        // Every migrated resource now rests sampled, so its producer is a real
        // open (the folded cross-frame reset) and its consumer a real close.
        // ao_output: producer UNDEFINED -> COLOR_ATTACHMENT (real), consumer
        // COLOR_ATTACHMENT -> SHADER_READ (real).
        assert!(
            vk_transition(
                GraphResourceClass::ColorTarget,
                ResourceState::Undefined,
                ResourceState::Write,
                FRAG,
            )
            .is_some()
        );
        assert!(
            vk_transition(
                GraphResourceClass::ColorTarget,
                ResourceState::Write,
                ResourceState::Read,
                FRAG,
            )
            .is_some()
        );
        // shadow_map: producer is the real folded reset SHADER_READ_ONLY ->
        // DEPTH_STENCIL_ATTACHMENT; consumer DEPTH_STENCIL_ATTACHMENT ->
        // SHADER_READ (real).
        let (po, pn, ..) = vk_transition(
            GraphResourceClass::DepthTarget,
            ResourceState::Undefined,
            ResourceState::Write,
            FRAG,
        )
        .unwrap();
        assert_eq!(po, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(pn, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let (old, new, ..) = vk_transition(
            GraphResourceClass::DepthTarget,
            ResourceState::Write,
            ResourceState::Read,
            FRAG,
        )
        .unwrap();
        assert_eq!(old, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        assert_eq!(new, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        // Generic no-op: a transition whose old and new layouts coincide (here a
        // sampled-resting resource asked for Undefined -> Read) is skipped. No
        // migrated resource emits one now, but the translator still collapses it.
        assert!(
            vk_transition(
                GraphResourceClass::DepthTarget,
                ResourceState::Undefined,
                ResourceState::Read,
                FRAG,
            )
            .is_none()
        );
        // fog_froxel_volume: producer SHADER_READ_ONLY -> GENERAL (real open),
        // consumer GENERAL -> SHADER_READ_ONLY (real close).
        let (po, pn, ..) = vk_transition(
            GraphResourceClass::StorageImage,
            ResourceState::Undefined,
            ResourceState::Write,
            FRAG,
        )
        .unwrap();
        assert_eq!(po, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(pn, vk::ImageLayout::GENERAL);
        let (co, cn, ..) = vk_transition(
            GraphResourceClass::StorageImage,
            ResourceState::Write,
            ResourceState::Read,
            FRAG,
        )
        .unwrap();
        assert_eq!(co, vk::ImageLayout::GENERAL);
        assert_eq!(cn, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }

    #[test]
    fn transition_threads_compute_read_stage() {
        // A compute consumer of a colour resource (the hdr_resolve / AutoExposure
        // shape, once that resource migrates): the consumer Write -> Read keeps the
        // SHADER_READ_ONLY layout but its dst_stage is COMPUTE_SHADER; a mixed run
        // waits in both stages.
        let (.., src_stage, dst_stage) = vk_transition(
            GraphResourceClass::ColorTarget,
            ResourceState::Write,
            ResourceState::Read,
            ReadStages::COMPUTE,
        )
        .unwrap();
        assert_eq!(src_stage, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
        assert_eq!(dst_stage, vk::PipelineStageFlags::COMPUTE_SHADER);

        let (.., dst_stage) = vk_transition(
            GraphResourceClass::ColorTarget,
            ResourceState::Write,
            ResourceState::Read,
            ReadStages::FRAGMENT | ReadStages::COMPUTE,
        )
        .unwrap();
        assert_eq!(
            dst_stage,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER
        );

        // The WAR side: a Read -> Write whose prior run read in the compute stage
        // waits its src_stage on COMPUTE_SHADER (the readers it must not race).
        let (old, new, .., src_stage, _dst_stage) = vk_transition(
            GraphResourceClass::ColorTarget,
            ResourceState::Read,
            ResourceState::Write,
            ReadStages::COMPUTE,
        )
        .unwrap();
        assert_eq!(old, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(new, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        assert_eq!(src_stage, vk::PipelineStageFlags::COMPUTE_SHADER);
    }
}
