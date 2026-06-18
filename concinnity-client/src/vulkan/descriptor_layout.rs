// src/vulkan/descriptor_layout.rs
//
// Canonical descriptor-set binding tables for the geometry render path, kept in
// one place so the `layout(set = N, binding = M)` indices the GLSL shaders use
// stay greppable and locked. `init.rs` builds the real `vk::DescriptorSetLayout`s
// from these via `create_descriptor_set_layout`, and the per-frame descriptor
// writes target the same binding numbers. The unit tests below assert each table
// is gap-free + unique and pin the binding -> (type, stage) contract, so a
// reordering, retype, or stage-flag change that would silently desync from the
// shaders fails `cargo test` instead of reading garbage on the GPU. Vulkan
// analogue of `directx/init/heap_layout.rs`'s slot tests.
//
// Only the geometry-path sets (global / per-object / shadow) are centralized
// here; the post-process sets (composite, bloom, text) are simpler 1-3 binding
// layouts still declared inline in `init.rs`.

use ash::vk;

// One descriptor binding: (binding index, descriptor type, shader stages).
pub(in crate::vulkan) type Binding = (u32, vk::DescriptorType, vk::ShaderStageFlags);

// Global set (set 0), shared by the main / instanced / skinned pipelines:
//   0  ViewUniforms UBO          (VS + FS)
//   1  LightUniforms UBO         (FS)
//   2  ShadowUniforms UBO        (VS + FS)
//   3  shadow-map cascade array  (FS)
//   4  irradiance cube           (FS)
//   5  prefiltered env cube      (FS)
//   6  SSAO occlusion / fallback (FS)
pub(in crate::vulkan) fn global_set() -> [Binding; 7] {
    use vk::DescriptorType as T;
    use vk::ShaderStageFlags as S;
    [
        (0, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
        (1, T::UNIFORM_BUFFER, S::FRAGMENT),
        (2, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
        (3, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (4, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (5, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (6, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
    ]
}

// Per-object set (set 1): albedo at 0, normal map at 1.
pub(in crate::vulkan) fn object_set() -> [Binding; 2] {
    use vk::DescriptorType as T;
    use vk::ShaderStageFlags as S;
    [
        (0, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
        (1, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
    ]
}

// Shadow global set (set 0 for the shadow pass): ShadowUniforms UBO, vertex-only
// (the shadow fragment stage is a depth-only no-op).
pub(in crate::vulkan) fn shadow_global_set() -> [Binding; 1] {
    [(
        0,
        vk::DescriptorType::UNIFORM_BUFFER,
        vk::ShaderStageFlags::VERTEX,
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sorted binding indices must be exactly 0..n: any duplicate or gap (a
    // fat-fingered binding number) breaks this.
    fn assert_gap_free_and_unique(bindings: &[Binding]) {
        let mut idx: Vec<u32> = bindings.iter().map(|b| b.0).collect();
        idx.sort_unstable();
        for (expected, &got) in idx.iter().enumerate() {
            assert_eq!(
                got, expected as u32,
                "descriptor bindings must be 0..n gap-free and unique, got {idx:?}"
            );
        }
    }

    #[test]
    fn geometry_path_sets_are_gap_free() {
        assert_gap_free_and_unique(&global_set());
        assert_gap_free_and_unique(&object_set());
        assert_gap_free_and_unique(&shadow_global_set());
    }

    // Golden lock: an independent copy of the binding -> (type, stage) contract.
    // Editing `global_set()` without updating this (and the matching shader
    // `layout(...)` qualifiers) is a deliberate review gate, not a silent change.
    #[test]
    fn global_set_contract_is_locked() {
        use vk::DescriptorType as T;
        use vk::ShaderStageFlags as S;
        assert_eq!(
            global_set(),
            [
                (0, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
                (1, T::UNIFORM_BUFFER, S::FRAGMENT),
                (2, T::UNIFORM_BUFFER, S::VERTEX | S::FRAGMENT),
                (3, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (4, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (5, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (6, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
            ]
        );
    }

    #[test]
    fn object_and_shadow_sets_contract_is_locked() {
        use vk::DescriptorType as T;
        use vk::ShaderStageFlags as S;
        assert_eq!(
            object_set(),
            [
                (0, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
                (1, T::COMBINED_IMAGE_SAMPLER, S::FRAGMENT),
            ]
        );
        assert_eq!(shadow_global_set(), [(0, T::UNIFORM_BUFFER, S::VERTEX)]);
    }
}
