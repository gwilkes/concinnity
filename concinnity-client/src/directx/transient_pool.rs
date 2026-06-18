// src/directx/transient_pool.rs
//
// Backing store for the render graph's transient render targets on D3D12. The
// shared `gfx::render_graph::alias` planner decides which transients can share
// physical memory; this pool realises that on D3D12 with placed resources on an
// `ID3D12Heap` (the analogue of Vulkan's aliased `VkImage`s on a shared
// `VkDeviceMemory`). Features stop owning these resources and read them back by
// label, so the pool can repoint several labels at one heap region without
// touching the features.
//
// Buffering: D3D12 is single-buffered for these targets. The command queue runs
// frames in submission order and the per-resource state-transition barriers
// serialise a frame's writes against a prior frame's reads of the same resource,
// so a single resource is safe across frames in flight (unlike Vulkan, whose
// explicit-layout model led that backend to per-frame buffer its bloom chain).
// So a DX alias slot is ONE shared heap region (not per-frame): making the
// members per-frame would multiply already-single-buffered resources and cost
// more memory than aliasing saves. The cross-frame reuse ordering is carried by
// aliasing barriers at both reuse boundaries (added when sharing lands).
//
// A resource is "managed" iff its owning feature is enabled at build time (e.g.
// `ao_output` only when SSAO is on); `resource_for` returns `None` otherwise and
// the consumer keeps its disabled-feature fallback.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::texture::{HDR_FORMAT, one_shot_submit, transition_barrier};

// One managed transient: the graph label plus the concrete D3D12 parameters the
// pool needs to place it. The label is the same string `build_frame_graph`
// declares, so the barrier registry and every feature consumer agree on one id.
pub(super) struct ResourceSpec {
    pub label: &'static str,
    pub width: u32,
    pub height: u32,
    pub format: DXGI_FORMAT,
    pub clear_color: [f32; 4],
    pub initial_state: D3D12_RESOURCE_STATES,
}

// One alias slot: the members that share a heap region (each placed at offset 0).
// A single-member slot is a plain placed target; a multi-member slot realises an
// alias (the members have disjoint lifetimes, so they reuse one region).
pub(super) struct SlotSpec {
    pub members: Vec<ResourceSpec>,
}

struct PlacedResource {
    label: &'static str,
    resource: ID3D12Resource,
}

// The transient render-target pool owned by `DxContext`. Resolution-dependent,
// so it is rebuilt on swapchain resize. COM resources release on drop, so the
// old heaps + resources free when the pool is reassigned (the caller has already
// idled the device).
pub(super) struct TransientResourcePool {
    // One heap per slot. Held only to keep the heaps alive: a placed resource
    // does not keep its heap alive (D3D12 requires the heap to outlive the
    // resource), so the pool must retain them. Never read after construction.
    #[allow(dead_code)]
    heaps: Vec<ID3D12Heap>,
    resources: Vec<PlacedResource>,
    // For each member of a shared (multi-member) slot, its cyclic predecessor:
    // the label of the resource whose heap memory it reclaims. Cyclic because
    // D3D12 is single-buffered, so the first member reclaims from the last across
    // the frame boundary (the wrap), giving every shared member a predecessor.
    // Empty when no slot is shared. Drives the executor's aliasing barriers.
    alias_pred: Vec<(&'static str, &'static str)>,
}

impl TransientResourcePool {
    // Allocate one heap per slot, sized to the largest member, and place every
    // member resource at offset 0. Each is created in its `initial_state` with
    // its optimized clear value, exactly as the committed version was, so its
    // first-use barrier is unchanged.
    pub(super) fn build(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        slots: &[SlotSpec],
    ) -> Result<Self, String> {
        let mut heaps = Vec::new();
        let mut resources = Vec::new();
        let mut alias_pred: Vec<(&'static str, &'static str)> = Vec::new();
        // (resource, resting state) for the one-shot init below. A placed
        // render-target resource is NOT auto-zeroed like a committed one, so
        // D3D12 rejects its first draw/sample until a Clear/Discard/Copy
        // initializes it. Discard suffices (no need to define the contents):
        // every managed transient is fully written each frame before it is read
        // (the SSAO blur writes `ao_output`, the bloom prefilter writes
        // `bloom_top`), and a consumer that may run while a target is unwritten
        // guards its read (the composite skips `bloom_top` when bloom is off),
        // so the undefined initial contents are never observed. Only single-
        // member slots are initialized here; a shared slot's members are
        // re-initialized per frame by the executor's aliasing barrier + Discard
        // before each first write (Discarding them here, on shared memory with no
        // aliasing barrier between, would itself be an aliasing hazard).
        let mut to_init: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)> = Vec::new();
        for slot in slots {
            let shared = slot.members.len() > 1;
            // Size the heap to the largest member; offset 0 satisfies every
            // member's alignment, so aliased members all place there.
            let mut slot_size: u64 = 0;
            let mut slot_align: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
            let descs: Vec<(&ResourceSpec, D3D12_RESOURCE_DESC)> = slot
                .members
                .iter()
                .map(|m| {
                    let desc = rt_desc(m);
                    let info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
                    slot_size = slot_size.max(info.SizeInBytes);
                    slot_align = slot_align.max(info.Alignment);
                    (m, desc)
                })
                .collect();

            let heap_desc = D3D12_HEAP_DESC {
                SizeInBytes: slot_size,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                Alignment: slot_align,
                // These targets are all render targets, so a heap restricted to
                // RT/DS textures is valid on every resource-heap tier.
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
            };
            let mut heap: Option<ID3D12Heap> = None;
            unsafe { device.CreateHeap(&heap_desc, &mut heap) }
                .map_err(|e| format!("transient pool heap: {e}"))?;
            let heap = heap.ok_or("transient pool heap returned None")?;

            for (m, desc) in &descs {
                let clear = D3D12_CLEAR_VALUE {
                    Format: m.format,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        Color: m.clear_color,
                    },
                };
                let mut res: Option<ID3D12Resource> = None;
                unsafe {
                    device.CreatePlacedResource(
                        &heap,
                        0,
                        desc,
                        m.initial_state,
                        Some(&clear),
                        &mut res,
                    )
                }
                .map_err(|e| format!("transient pool place {}: {e}", m.label))?;
                let resource = res.ok_or("transient pool placed resource None")?;
                if !shared {
                    to_init.push((resource.clone(), m.initial_state));
                }
                resources.push(PlacedResource {
                    label: m.label,
                    resource,
                });
            }
            // Wire each shared-slot member to its cyclic predecessor (the prior
            // member, the first to the last) so the executor can claim the memory
            // before each first write.
            if shared {
                let n = slot.members.len();
                for i in 0..n {
                    alias_pred.push((slot.members[i].label, slot.members[(i + n - 1) % n].label));
                }
            }
            heaps.push(heap);
        }

        // Initialize every placed resource (Discard in its RENDER_TARGET state,
        // then back to its resting state) so its first real use is legal.
        if !to_init.is_empty() {
            one_shot_submit(device, queue, |cmd| {
                for (res, resting) in &to_init {
                    unsafe {
                        cmd.ResourceBarrier(&[transition_barrier(
                            res,
                            *resting,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                        )]);
                        cmd.DiscardResource(res, None);
                        cmd.ResourceBarrier(&[transition_barrier(
                            res,
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            *resting,
                        )]);
                    }
                }
            })?;
        }

        Ok(Self {
            heaps,
            resources,
            alias_pred,
        })
    }

    // The managed resource for `label`, or `None` when the owning feature was
    // disabled at build time (so nothing was placed).
    pub(super) fn resource_for(&self, label: &str) -> Option<&ID3D12Resource> {
        self.resources
            .iter()
            .find(|r| r.label == label)
            .map(|r| &r.resource)
    }

    // The label of the resource whose heap memory `label` reclaims (its cyclic
    // slot predecessor), or `None` when `label` is not a shared-slot member (so
    // it is not aliased and needs no aliasing barrier). The executor emits an
    // aliasing barrier before the pass that first-writes any resource for which
    // this returns `Some`.
    pub(super) fn alias_predecessor(&self, label: &str) -> Option<&'static str> {
        self.alias_pred
            .iter()
            .find(|(l, _)| *l == label)
            .map(|(_, p)| *p)
    }

    // Rebuild every managed resource at a new extent. The caller has already
    // idled the device; reassigning drops the old heaps + placed resources
    // (COM release), so any feature descriptor that referenced them must be
    // rewritten by the caller afterward.
    pub(super) fn rebuild(
        &mut self,
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        slots: &[SlotSpec],
    ) -> Result<(), String> {
        *self = Self::build(device, queue, slots)?;
        Ok(())
    }
}

// The resource desc for a managed render target: single-sample Texture2D, one
// mip, render-target-capable (matches `texture::create_rt_target`). `Alignment`
// 0 lets the runtime pick the default (64 KiB) placement alignment.
fn rt_desc(m: &ResourceSpec) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: m.width.max(1) as u64,
        Height: m.height.max(1),
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: m.format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
    }
}

// Build the alias-slot list for the transients the pool manages this build.
// Centralises the label to D3D12 format / state mapping so init and resize stay
// in lockstep. The managed transients are `bloom_top` (the bloom chain's
// half-res top mip, always present) and `ao_output` (SSAO's blurred occlusion,
// only when SSAO is on). The shared `gfx::render_graph::alias` planner decides
// which of them may share a heap region; `group_by_plan` packs them into one
// `SlotSpec` per planner slot, so disjoint-lifetime transients alias.
pub(super) fn transient_slots(
    ssao_enabled: bool,
    ao_extent: (u32, u32),
    bloom_top_extent: (u32, u32),
) -> Vec<SlotSpec> {
    // bloom_top is always managed: the bloom chain always exists and the
    // composite samples mip 0 even when bloom is disabled.
    let mut specs = vec![bloom_top_spec(bloom_top_extent)];
    if ssao_enabled {
        specs.push(ao_output_spec(ao_extent));
    }
    group_by_plan(specs, ssao_enabled)
}

// Group the managed specs into shared slots per the aliasing planner. The
// planner runs on a worst-case graph (bloom forced on, since it toggles per
// frame yet the physical allocation must cover the frames where it is live) at
// the actual SSAO setting; the grouping is lifetime-based, so the extent used
// for the planner's sizing is irrelevant and a fixed one is passed. Members of a
// planner slot are kept in its order (lifetime-start), which the pool's cyclic
// predecessor wiring relies on. Falls back to one slot per spec if the
// worst-case graph fails to compile, leaving the build render-neutral.
fn group_by_plan(specs: Vec<ResourceSpec>, ssao_enabled: bool) -> Vec<SlotSpec> {
    use crate::gfx::render_graph::{FrameGraphInputs, build_frame_graph, plan_aliasing};

    let mut inputs = FrameGraphInputs::all_off();
    inputs.bloom_enabled = true;
    inputs.ssao_enabled = ssao_enabled;

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
    let mut specs_opt: Vec<Option<ResourceSpec>> = specs.into_iter().map(Some).collect();
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

// `bloom_top`: bloom mip 0 (the chain's half-resolution top), HDR_FORMAT,
// sampled by the composite. The composite binds it even when bloom is disabled
// and the bloom passes never run, so its clear-colour init must hold. Rests in
// PIXEL_SHADER_RESOURCE like the committed mip it replaces; the bloom prefilter
// transitions it to RENDER_TARGET per frame.
fn bloom_top_spec((width, height): (u32, u32)) -> ResourceSpec {
    ResourceSpec {
        label: "bloom_top",
        width,
        height,
        format: HDR_FORMAT,
        clear_color: [0.0; 4],
        initial_state: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    }
}

// `ao_output`: SSAO's blurred occlusion at full render resolution, single-channel
// R8, sampled by the main pass's ambient term. Rests in PIXEL_SHADER_RESOURCE
// (matching `create_rt_target` + the executor's barrier registry resting state).
fn ao_output_spec((width, height): (u32, u32)) -> ResourceSpec {
    ResourceSpec {
        label: "ao_output",
        width,
        height,
        format: super::post::ssao::SSAO_OCCLUSION_FORMAT,
        clear_color: [0.0; 4],
        initial_state: D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `transient_slots` is pure CPU (it builds slot descriptions; no device), so
    // the planner-routed grouping is testable headlessly.

    #[test]
    fn ssao_and_bloom_alias_into_one_slot() {
        // With SSAO on, the planner sees `ao_output` (early: SsaoBlur -> Main) and
        // `bloom_top` (late: Bloom -> Composite) with disjoint lifetimes, so the
        // pool packs them into one shared slot. This is the memory saving: one
        // heap region instead of two.
        let slots = transient_slots(true, (1024, 768), (512, 384));
        assert_eq!(
            slots.len(),
            1,
            "ao_output + bloom_top should share one slot"
        );
        let labels: Vec<&str> = slots[0].members.iter().map(|m| m.label).collect();
        assert!(labels.contains(&"bloom_top"), "{labels:?}");
        assert!(labels.contains(&"ao_output"), "{labels:?}");
    }

    #[test]
    fn bloom_top_alone_is_unshared() {
        // SSAO off: `bloom_top` is the only managed transient, so it sits in its
        // own single-member slot (no aliasing, no aliasing barriers).
        let slots = transient_slots(false, (1024, 768), (512, 384));
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].members.len(), 1);
        assert_eq!(slots[0].members[0].label, "bloom_top");
    }
}
