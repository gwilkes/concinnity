// src/vulkan/transient_pool.rs
//
// Backing store for the render graph's transient images. Stage 1's
// `gfx::render_graph::alias` planner decides which transient resources may
// share physical memory; this pool is where the Vulkan backend realises that
// plan. Features stop owning these images and read them back by label, so the
// pool can repoint several labels at one shared allocation without touching the
// features. This mirrors how the graph plans barriers while each backend emits
// them.
//
// Structure: the pool is organised into alias *slots*. A slot owns one
// `VkDeviceMemory` per frame in flight; every member image of a slot binds into
// that one allocation at offset 0. Members of a slot must have pairwise-disjoint
// lifetimes (they are never live at the same time), so reusing the bytes is
// safe within a frame; the per-frame copies keep the reuse safe across frames in
// flight (the single-frame planner does not model frames-in-flight, so the
// backend supplies the per-frame buffering). A single-member slot is just a
// per-frame target with its own memory (no sharing); a multi-member slot is a
// realised alias.
//
// A resource is "managed" iff its owning feature is enabled at build time (e.g.
// `ao_output` only when SSAO is on); the `*_for` lookups return `None`
// otherwise and the consumer falls back exactly as it did before.

use ash::{Device, vk};

use super::texture::{create_image_view, find_memory_type};

// One managed transient image: the graph label plus the concrete Vulkan
// parameters the pool needs to allocate it. The label is the same string the
// shared `build_frame_graph` declares, so the barrier registry and every
// feature consumer agree on one identifier.
pub(super) struct ImageSpec {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
    pub aspect: vk::ImageAspectFlags,
}

// One alias slot: the set of images that share backing memory. A single-member
// slot is a plain per-frame target; a multi-member slot realises an alias (the
// members have disjoint lifetimes, so they reuse one allocation per frame).
pub(super) struct SlotSpec {
    pub members: Vec<ImageSpec>,
}

// One managed image, resolved for one frame in flight.
struct PooledImage {
    label: &'static str,
    frame: usize,
    image: vk::Image,
    view: vk::ImageView,
}

// The transient image pool owned by `VkContext`. Resolution-dependent, so it is
// rebuilt on swapchain resize.
pub(super) struct TransientImagePool {
    // One backing allocation per (slot, frame). Owned here, freed on destroy /
    // rebuild after the member images + views are gone.
    slot_memories: Vec<vk::DeviceMemory>,
    // Every member image across all slots + frames.
    images: Vec<PooledImage>,
    // The member labels of each slot, in lifetime order (the order they reuse the
    // slot's memory). Drives the executor's aliasing barriers: a member's
    // predecessor in this list is the resource it reuses memory from.
    slot_labels: Vec<Vec<&'static str>>,
}

impl TransientImagePool {
    // Allocate every slot's per-frame backing memory and bind its member images
    // into it. Each member image is created device-local and left `UNDEFINED`;
    // its first-use layout is established by its graph producer barrier or its
    // render pass, exactly as when the feature owned the image.
    pub(super) fn build(
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frames: usize,
        slots: &[SlotSpec],
    ) -> Result<Self, String> {
        let mut slot_memories = Vec::new();
        let mut images = Vec::new();
        let slot_labels: Vec<Vec<&'static str>> = slots
            .iter()
            .map(|s| s.members.iter().map(|m| m.label).collect())
            .collect();
        // Footprint accounting: `aliased_bytes` is what the pool actually
        // allocates (one slot allocation per (slot, frame)); `unaliased_bytes` is
        // what the same images would cost with no sharing. Their difference is the
        // VRAM aliasing reclaims, reported below.
        let mut aliased_bytes: u64 = 0;
        let mut unaliased_bytes: u64 = 0;
        for slot in slots {
            for f in 0..frames {
                // Create every member image (unbound), gathering the combined
                // memory requirements: the slot's allocation must be large
                // enough for the biggest member and of a type all members accept.
                let mut member_images: Vec<(&ImageSpec, vk::Image)> =
                    Vec::with_capacity(slot.members.len());
                let mut type_bits = u32::MAX;
                let mut slot_size: vk::DeviceSize = 0;
                for m in &slot.members {
                    let image = create_image_unbound(device, m)?;
                    let reqs = unsafe { device.get_image_memory_requirements(image) };
                    type_bits &= reqs.memory_type_bits;
                    slot_size = slot_size.max(reqs.size);
                    unaliased_bytes += reqs.size;
                    member_images.push((m, image));
                }
                aliased_bytes += slot_size;

                // One device-local allocation backs every member of this slot
                // for this frame; bind each member at offset 0 (their disjoint
                // lifetimes make the overlap safe).
                let memory = unsafe {
                    device.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(slot_size)
                            .memory_type_index(find_memory_type(
                                instance,
                                physical_device,
                                type_bits,
                                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                            )?),
                        None,
                    )
                }
                .map_err(|e| format!("transient pool slot memory: {e}"))?;
                slot_memories.push(memory);

                for (spec, image) in member_images {
                    unsafe { device.bind_image_memory(image, memory, 0) }
                        .map_err(|e| format!("transient pool bind {}: {e}", spec.label))?;
                    let view = create_image_view(device, image, spec.format, spec.aspect)?;
                    images.push(PooledImage {
                        label: spec.label,
                        frame: f,
                        image,
                        view,
                    });
                }
            }
        }
        tracing::info!(
            "transient image pool: {} slot allocation(s), {} KiB ({} KiB saved by aliasing)",
            slot_memories.len(),
            aliased_bytes / 1024,
            unaliased_bytes.saturating_sub(aliased_bytes) / 1024,
        );
        Ok(Self {
            slot_memories,
            images,
            slot_labels,
        })
    }

    // The label `label` reuses slot memory from, i.e. the member immediately
    // before it in its slot's lifetime order, or `None` when `label` is the
    // first member of its slot (or unmanaged, or alone). The executor emits an
    // aliasing barrier on `label` against this predecessor before `label`'s
    // first write, since they share one allocation.
    pub(super) fn alias_predecessor(&self, label: &str) -> Option<&'static str> {
        for members in &self.slot_labels {
            if let Some(pos) = members.iter().position(|&l| l == label) {
                return if pos == 0 {
                    None
                } else {
                    Some(members[pos - 1])
                };
            }
        }
        None
    }

    // The managed image for `label` at frame-in-flight `frame`, or `None` when
    // the owning feature was disabled at build time (so no image was allocated).
    pub(super) fn image_for(&self, label: &str, frame: usize) -> Option<vk::Image> {
        self.lookup(label, frame).map(|p| p.image)
    }

    // The sampled / attachment view for `label` at frame-in-flight `frame`.
    pub(super) fn view_for(&self, label: &str, frame: usize) -> Option<vk::ImageView> {
        self.lookup(label, frame).map(|p| p.view)
    }

    // Every frame-in-flight view for `label`, frames `0..frames` in order.
    // Empty when the label is unmanaged. When the label is managed the pool
    // holds one image per frame, so the result has exactly `frames` entries.
    pub(super) fn views_for_frames(&self, label: &str, frames: usize) -> Vec<vk::ImageView> {
        (0..frames)
            .filter_map(|f| self.view_for(label, f))
            .collect()
    }

    // Every (image, view) pair for `label`, frames `0..frames` in order. Empty
    // when unmanaged; one entry per frame when managed. Used to hand a per-frame
    // pooled image to a feature that wraps it (bloom mip 0).
    pub(super) fn pairs_for_frames(
        &self,
        label: &str,
        frames: usize,
    ) -> Vec<(vk::Image, vk::ImageView)> {
        (0..frames)
            .filter_map(|f| self.lookup(label, f).map(|p| (p.image, p.view)))
            .collect()
    }

    fn lookup(&self, label: &str, frame: usize) -> Option<&PooledImage> {
        self.images
            .iter()
            .find(|p| p.label == label && p.frame == frame)
    }

    // Rebuild every managed image at a new extent / frame count. The caller has
    // already idled the device. The old images are freed first, so any feature
    // framebuffer / descriptor that referenced their views must be rebuilt by
    // the caller afterward.
    pub(super) fn rebuild(
        &mut self,
        instance: &ash::Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        frames: usize,
        slots: &[SlotSpec],
    ) -> Result<(), String> {
        self.destroy(device);
        *self = Self::build(instance, device, physical_device, frames, slots)?;
        Ok(())
    }

    // Free every managed image, view, and slot allocation. The caller has
    // already idled the device and destroyed any framebuffer that referenced
    // these views.
    pub(super) fn destroy(&mut self, device: &Device) {
        unsafe {
            for p in &self.images {
                device.destroy_image_view(p.view, None);
                device.destroy_image(p.image, None);
            }
            for &mem in &self.slot_memories {
                device.free_memory(mem, None);
            }
        }
        self.images.clear();
        self.slot_memories.clear();
        self.slot_labels.clear();
    }
}

// Create a 2D `VkImage` without backing memory: the pool binds it into a slot
// allocation afterward (so several aliased images can share one allocation).
// Mirrors `texture::create_image` minus the allocate + bind.
fn create_image_unbound(device: &Device, spec: &ImageSpec) -> Result<vk::Image, String> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: spec.width.max(1),
            height: spec.height.max(1),
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(spec.format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(spec.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    unsafe { device.create_image(&info, None) }.map_err(|e| format!("transient pool image: {e}"))
}

// Build the alias-slot list for the transients the pool manages this build.
// Centralises the label to Vulkan format / usage mapping and the slot grouping
// so init and resize stay in lockstep. A transient is managed only while its
// owning feature is on: `ao_output` with SSAO, `bloom_top` with bloom. The
// shared `gfx::render_graph::alias` planner decides which of the managed
// transients may share a slot; `group_by_plan` packs them into one `SlotSpec`
// per planner slot, so disjoint-lifetime transients alias -- `ao_output` (used
// early, [SsaoBlur, Main]) and `bloom_top` (used late, [Bloom, Composite]) share
// ONE per-frame slot when both are on, with `ao_output` first since it reuses
// the memory first.
pub(super) fn transient_slots(
    ssao_enabled: bool,
    bloom_enabled: bool,
    ao_extent: vk::Extent2D,
    bloom_extent: vk::Extent2D,
) -> Vec<SlotSpec> {
    let mut specs = Vec::new();
    if ssao_enabled {
        specs.push(ao_output_spec(ao_extent));
    }
    if bloom_enabled {
        specs.push(bloom_top_spec(bloom_extent));
    }
    group_by_plan(specs, ssao_enabled, bloom_enabled)
}

// Group the managed specs into shared slots per the aliasing planner. The
// planner runs on a minimal worst-case graph (only the managed features on) so
// the generic greedy pairs exactly the pooled candidates -- on the full frame
// graph it could instead pair `bloom_top` with the unpooled `gbuffer`. The
// grouping is lifetime-based, so the extent passed for the planner's sizing is
// irrelevant and a fixed one is used. Members of a planner slot keep its order
// (lifetime-start), which the pool's cyclic predecessor wiring relies on. Falls
// back to one slot per spec if the worst-case graph fails to compile, leaving
// the build render-neutral. Unlike DirectX (where `bloom_top` is always managed
// and bloom toggles per frame, so the planner graph forces bloom on), Vulkan
// manages `bloom_top` only while bloom is on, so the planner graph uses the real
// bloom flag.
fn group_by_plan(specs: Vec<ImageSpec>, ssao_enabled: bool, bloom_enabled: bool) -> Vec<SlotSpec> {
    use crate::gfx::render_graph::{FrameGraphInputs, build_frame_graph, plan_aliasing};

    let mut inputs = FrameGraphInputs::all_off();
    inputs.ssao_enabled = ssao_enabled;
    inputs.bloom_enabled = bloom_enabled;

    let groups: Vec<Vec<usize>> = match build_frame_graph(&inputs) {
        Ok(graph) => {
            let plan = plan_aliasing(&graph, 1920, 1080);
            let mut by_label: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for (i, s) in specs.iter().enumerate() {
                by_label.insert(s.label, i);
            }
            let mut groups: Vec<Vec<usize>> = Vec::new();
            let mut grouped = vec![false; specs.len()];
            for slot in &plan.slots {
                let group: Vec<usize> = slot
                    .members
                    .iter()
                    .filter_map(|&res_idx| by_label.get(graph.resources[res_idx].label).copied())
                    .collect();
                for &si in &group {
                    grouped[si] = true;
                }
                if !group.is_empty() {
                    groups.push(group);
                }
            }
            // Any managed spec the planner did not place (no graph resource for
            // it) gets its own un-aliased slot.
            for (si, placed) in grouped.iter().enumerate() {
                if !placed {
                    groups.push(vec![si]);
                }
            }
            groups
        }
        Err(_) => (0..specs.len()).map(|i| vec![i]).collect(),
    };

    // Materialize: move each spec into its group's slot.
    let mut specs_opt: Vec<Option<ImageSpec>> = specs.into_iter().map(Some).collect();
    groups
        .into_iter()
        .map(|g| SlotSpec {
            members: g
                .into_iter()
                .map(|si| specs_opt[si].take().unwrap())
                .collect(),
        })
        .collect()
}

// `ao_output`: SSAO's blurred occlusion at full render resolution, single-channel
// R8, sampled by the main pass's ambient term.
fn ao_output_spec(extent: vk::Extent2D) -> ImageSpec {
    ImageSpec {
        label: "ao_output",
        width: extent.width,
        height: extent.height,
        format: super::post::ssao::SSAO_OCCLUSION_FORMAT,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        aspect: vk::ImageAspectFlags::COLOR,
    }
}

// `bloom_top`: bloom mip 0, half the output (swapchain) extent, HDR_FORMAT. The
// prefilter writes it, the downsample chain reads it, the final upsample
// accumulates into it, and composite samples it.
fn bloom_top_spec(extent: vk::Extent2D) -> ImageSpec {
    ImageSpec {
        label: "bloom_top",
        width: (extent.width >> 1).max(1),
        height: (extent.height >> 1).max(1),
        format: super::context::HDR_FORMAT,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        aspect: vk::ImageAspectFlags::COLOR,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `transient_slots` is pure CPU (it builds slot descriptions; no device), so
    // the planner-routed grouping is testable headlessly.

    fn extent(width: u32, height: u32) -> vk::Extent2D {
        vk::Extent2D { width, height }
    }

    #[test]
    fn ssao_and_bloom_alias_into_one_slot() {
        // Both on: the planner sees `ao_output` (early: SsaoBlur -> Main) and
        // `bloom_top` (late: Bloom -> Composite) with disjoint lifetimes, so the
        // pool packs them into one shared slot -- one allocation instead of two.
        let slots = transient_slots(true, true, extent(1024, 768), extent(1024, 768));
        assert_eq!(
            slots.len(),
            1,
            "ao_output + bloom_top should share one slot"
        );
        let labels: Vec<&str> = slots[0].members.iter().map(|m| m.label).collect();
        assert!(labels.contains(&"ao_output"), "{labels:?}");
        assert!(labels.contains(&"bloom_top"), "{labels:?}");
        // `ao_output` reuses the memory first, so it must be the first member:
        // the pool's cyclic predecessor wiring depends on lifetime-start order.
        assert_eq!(slots[0].members[0].label, "ao_output", "{labels:?}");
    }

    #[test]
    fn ao_output_alone_is_unshared() {
        // SSAO on, bloom off: `ao_output` is the only managed transient, so it
        // sits in its own single-member slot (no aliasing barriers).
        let slots = transient_slots(true, false, extent(1024, 768), extent(1024, 768));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "ao_output");
    }

    #[test]
    fn bloom_top_alone_is_unshared() {
        // Bloom on, SSAO off: `bloom_top` is the only managed transient.
        let slots = transient_slots(false, true, extent(1024, 768), extent(1024, 768));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "bloom_top");
    }

    #[test]
    fn nothing_managed_yields_no_slots() {
        // Neither feature on: the pool manages nothing, so there are no slots.
        let slots = transient_slots(false, false, extent(1024, 768), extent(1024, 768));
        assert!(slots.is_empty());
    }
}
