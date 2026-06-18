// src/vulkan/resources/textures.rs
//
// Texture-pool slot management for VkContext: per-object / bindless / cluster /
// skinned / chunk descriptor rewires when an albedo or normal-map slot is
// streamed in or evicted. Mirrors the Metal pattern of "the texture pool gets
// re-read every frame," except Vulkan bakes texture *views* into descriptor
// sets at init, so a slot swap must walk every set that samples this slot.

use ash::vk;

use super::super::context::*;
use super::super::texture::upload_texture;

impl VkContext {
    pub(in crate::vulkan) fn write_object_image(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        view: vk::ImageView,
    ) {
        let info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(self.linear_sampler);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&info));
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    // Re-point bindless texture-pool element `index` of `set` to `view`.
    // Keeps the bindless texture pool in sync with a streamed albedo /
    // normal-map swap. The pool layout is `[albedo..] ++ [normal..]`.
    fn write_pool_image(&self, set: vk::DescriptorSet, index: u32, view: vk::ImageView) {
        let info = vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(self.linear_sampler);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(std::slice::from_ref(&info));
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    // Re-point every per-object / per-cluster descriptor set that samples
    // albedo `slot` at the (just-swapped) `self.textures[slot]` view.
    fn rewrite_albedo_slot(&self, slot: usize) {
        let last = self.textures.len().saturating_sub(1);
        let view = self.textures[slot].view;
        for (set, obj) in self
            .descriptors
            .object_sets
            .iter()
            .zip(self.draw_objects.iter())
        {
            if obj.texture_slot.min(last) == slot {
                self.write_object_image(*set, 0, view);
            }
        }
        // The bindless pool addresses albedo slot `s` at pool index `s`.
        for &set in &self.cull.bindless_sets {
            self.write_pool_image(set, slot as u32, view);
        }
        for (set, cluster) in self
            .instanced
            .object_sets
            .iter()
            .zip(self.instanced.clusters.iter())
        {
            if cluster.texture_slot.min(last) == slot {
                self.write_object_image(*set, 0, view);
            }
        }
        for (set, obj) in self
            .skinned
            .object_sets
            .iter()
            .zip(self.skinned.draw_objects.iter())
        {
            if obj.texture_slot.min(last) == slot {
                self.write_object_image(*set, 0, view);
            }
        }
        // The shared VoxelWorld chunk set bakes its albedo view at init too,
        // and is not part of object/cluster/skinned, so re-point it explicitly.
        if let (Some(set), Some(chunk_slot)) =
            (self.chunk_stream.object_set, self.chunk_stream.texture_slot)
            && chunk_slot == slot
        {
            self.write_object_image(set, 0, view);
        }
        // Per-decal albedo descriptors. Walk the decal-side slot tracker;
        // a world with no decals pays nothing here.
        self.rewrite_decal_albedo_slot(slot);
        // Runtime clones (from `clone_static_draw_object`) carry their own
        // (albedo, normal) descriptor sets, so re-point those that sample
        // this slot.
        for (offset, &clone_slot) in self.clone_texture_slots.iter().enumerate() {
            if clone_slot.min(last) == slot
                && let Some(&set) = self.clone_object_sets.get(offset)
            {
                self.write_object_image(set, 0, view);
            }
        }
        // Particle emitters sample an albedo from the texture pool into their
        // own render set, so re-point any whose source slot is this one.
        self.rewrite_particle_albedo_slot(slot);
    }

    // Re-point every per-object / per-cluster descriptor set that samples
    // normal-map `slot` at the (just-swapped) `self.normal_map_textures[slot]`
    // view.
    fn rewrite_normal_slot(&self, slot: usize) {
        let last = self.normal_map_textures.len().saturating_sub(1);
        let view = self.normal_map_textures[slot].view;
        for (set, obj) in self
            .descriptors
            .object_sets
            .iter()
            .zip(self.draw_objects.iter())
        {
            if obj.normal_map_slot.min(last) == slot {
                self.write_object_image(*set, 1, view);
            }
        }
        // The bindless pool ([albedo..] ++ [normal..]) addresses normal slot
        // `s` at pool index `albedo_count + s`.
        let albedo_count = self.textures.len();
        for &set in &self.cull.bindless_sets {
            self.write_pool_image(set, (albedo_count + slot) as u32, view);
        }
        for (set, cluster) in self
            .instanced
            .object_sets
            .iter()
            .zip(self.instanced.clusters.iter())
        {
            if cluster.normal_map_slot.min(last) == slot {
                self.write_object_image(*set, 1, view);
            }
        }
        for (set, obj) in self
            .skinned
            .object_sets
            .iter()
            .zip(self.skinned.draw_objects.iter())
        {
            if obj.normal_map_slot.min(last) == slot {
                self.write_object_image(*set, 1, view);
            }
        }
        // The shared VoxelWorld chunk set bakes its normal-map view at init
        // too, and is not part of object/cluster/skinned, so re-point it.
        if let (Some(set), Some(chunk_slot)) = (
            self.chunk_stream.object_set,
            self.chunk_stream.normal_map_slot,
        ) && chunk_slot == slot
        {
            self.write_object_image(set, 1, view);
        }
        // Runtime clones: re-point those that sample this normal-map slot.
        for (offset, &clone_slot) in self.clone_normal_map_slots.iter().enumerate() {
            if clone_slot.min(last) == slot
                && let Some(&set) = self.clone_object_sets.get(offset)
            {
                self.write_object_image(set, 1, view);
            }
        }
    }

    // Replace albedo texture-pool `slot` with freshly decoded RGBA8 pixels.
    pub fn update_texture_slot(
        &mut self,
        slot: usize,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        if slot >= self.textures.len() {
            return Err(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.textures.len()
            ));
        }
        self.wait_idle();
        let img = upload_texture(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            width,
            height,
            pixels,
        )?;
        // Swap in the new image, then rewrite every descriptor that samples
        // this slot BEFORE destroying the old view. The previous order
        // (destroy then rewrite) left a brief window where descriptor sets
        // referenced an already-destroyed VkImageView, spec-permissible
        // because vkUpdateDescriptorSets is write-only, but Vulkan validation
        // layers and some drivers will flag this when the descriptor pool
        // tracks live image-view handles per descriptor (the symptom is a
        // device-lost in worlds combining texture streaming with a
        // SkinnedMesh + VoxelWorld chunk material that share the
        // object_set_layout).
        let old = std::mem::replace(&mut self.textures[slot], img);
        self.rewrite_albedo_slot(slot);
        old.destroy(&self.device);
        Ok(())
    }

    // Reset albedo texture-pool `slot` to a 1x1 mid-grey placeholder.
    pub fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        self.update_texture_slot(slot, 1, 1, &[128, 128, 128, 255])
    }

    // Replace normal-map pool `slot` with freshly decoded RGBA8 pixels.
    pub fn update_normal_map_slot(
        &mut self,
        slot: usize,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        if slot >= self.normal_map_textures.len() {
            return Err(format!(
                "update_normal_map_slot: slot {} out of range (pool size {})",
                slot,
                self.normal_map_textures.len()
            ));
        }
        self.wait_idle();
        let img = upload_texture(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            width,
            height,
            pixels,
        )?;
        // Rewrite descriptors before destroying the old view; see
        // `update_texture_slot` for the rationale.
        let old = std::mem::replace(&mut self.normal_map_textures[slot], img);
        self.rewrite_normal_slot(slot);
        old.destroy(&self.device);
        Ok(())
    }

    // Reset normal-map pool `slot` to a 1x1 flat-normal placeholder.
    pub fn evict_normal_map_slot(&mut self, slot: usize) -> Result<(), String> {
        self.update_normal_map_slot(slot, 1, 1, &[128, 128, 255, 255])
    }

    // Replace the live colour-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only) when the file-backed
    // `ColorLut` source is saved. `wait_idle` first guarantees no in-flight
    // command buffer still references the old image. Builds the replacement
    // via the same `upload_color_lut` the init path uses, rewrites every
    // composite descriptor set's binding 2 to point at the new view, then
    // drops the previous image; same write-then-destroy order as the
    // texture-pool rewires above to keep validation layers happy. Mirrors
    // `DxContext::update_color_lut` / `MtlContext::update_color_lut`. Reached
    // only through the bin's `cn debug` runtime-mutation path (dead in the FFI
    // lib, live in the bin).
    #[allow(dead_code)]
    pub fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        self.wait_idle();
        let new_lut = super::super::texture::upload_color_lut(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            size,
            data,
        )?;
        // Rewrite composite descriptors before destroying the old image; see
        // the texture-pool rewires above for the rationale.
        let new_view = new_lut.view;
        let old = std::mem::replace(&mut self.color_lut, new_lut);
        for &set in &self.composite_sets {
            let info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(new_view)
                .sampler(self.composite_sampler);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&info));
            unsafe {
                self.device
                    .update_descriptor_sets(std::slice::from_ref(&write), &[])
            };
        }
        old.destroy(&self.device);
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). Decodes the byte stream
    // emitted by `gfx::build::environment_map::serialise`, then re-uploads
    // the irradiance + prefilter cubes via the same `upload_environment_map`
    // the init path uses. Every consumer that captured the old image views is
    // re-pointed at the new ones: each `global_sets` entry (irradiance +
    // prefilter), the SSR resolve sets (prefilter), and the raymarch view sets
    // (both cubes). `prefilter_mip_count` is refreshed on `self` so the next
    // frame's `ViewUniforms` upload picks up the new mip count. Unlike DirectX,
    // which re-uploads into the same SRV heap slots so its consumers need no
    // re-wire, every Vulkan `upload_environment_map` mints fresh `vk::ImageView`
    // handles, so each descriptor set must be re-written. Mirrors
    // `DxContext::update_environment_map`. Reached
    // only through the bin's `cn debug` runtime-mutation path (dead in the FFI
    // lib, live in the bin).
    #[allow(dead_code)]
    pub fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = crate::build::environment_map::deserialise(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {e}"))?;
        self.wait_idle();
        let new_env = super::super::texture::upload_environment_map(
            &self.instance,
            &self.device,
            self.physical_device,
            self.commands.command_pool,
            self.graphics_queue,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        let new_irradiance_view = new_env.irradiance.view;
        let new_prefilter_view = new_env.prefilter.view;
        let new_mip_count = new_env.prefilter_mip_count;
        // Rewrite global sets before destroying the previous cubes; see the
        // texture-pool rewires above for the rationale.
        let old = std::mem::replace(&mut self.env_map, new_env);
        self.prefilter_mip_count = new_mip_count;
        for &set in &self.descriptors.global_sets {
            let irr_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(new_irradiance_view)
                .sampler(self.cube_sampler);
            let pre_info = vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(new_prefilter_view)
                .sampler(self.cube_sampler);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(IRRADIANCE_CUBE_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&irr_info)),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(PREFILTER_CUBE_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(std::slice::from_ref(&pre_info)),
            ];
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        // The IBL cubes are also bound outside `global_sets`: the SSR resolve
        // sets sample the prefilter cube (set 0 binding 3) and the raymarch
        // view sets sample both cubes (bindings 4 + 5). They captured the old
        // image views too, so re-point them at the new cubes before the old
        // ones are destroyed below; otherwise the next SSR resolve / raymarch
        // draw reads a destroyed view and loses the device. Done here, not in
        // `global_sets`, because the resize path re-wires these the same way.
        if let Some(ssr) = self.ssr.as_ref() {
            let hdr_views: Vec<vk::ImageView> =
                self.hdr_resolve_images.iter().map(|img| img.view).collect();
            // Keep the SSR resolve's G-buffer / roughness bindings on the unified
            // pre-pass per-frame views when present (empty falls back to SSR's own
            // pre-pass targets); only binding 3 (the prefilter cube) actually
            // moved, but `wire_resolve_sets` rewrites all four bindings.
            let (nd_views, rough_views) = match self.gbuffer.as_ref() {
                Some(gb) => (gb.normal_depth_views(), gb.roughness_views()),
                None => (Vec::new(), Vec::new()),
            };
            ssr.wire_resolve_sets(
                &self.device,
                &hdr_views,
                &nd_views,
                &rough_views,
                new_prefilter_view,
                self.cube_sampler,
            );
        }
        if let Some(rm) = self.raymarch.as_ref() {
            rm.rewire_ibl_cubes(
                &self.device,
                new_irradiance_view,
                new_prefilter_view,
                self.cube_sampler,
            );
        }
        // The RT-reflection sets sample the prefilter cube at binding 8 (the miss
        // fallback + the metallic/roughness IBL hit shading); re-point them too.
        if let Some(rt) = self.rt_reflections.as_ref() {
            rt.rewire_prefilter(&self.device, new_prefilter_view, self.cube_sampler);
        }
        old.irradiance.destroy(&self.device);
        old.prefilter.destroy(&self.device);
        Ok(())
    }
}

// Set-0 binding indices for the IBL cubemaps. Must match the bindings the
// init path writes in `vulkan/init.rs` when wiring `global_sets`. Kept as
// documentation of that layout; `init.rs` writes the literals directly.
#[allow(dead_code)]
const IRRADIANCE_CUBE_BINDING: u32 = 4;
#[allow(dead_code)]
const PREFILTER_CUBE_BINDING: u32 = 5;

impl VkContext {
    // Append a new draw object that re-uses an existing slot's geometry
    // region with a fresh model matrix, texture / normal-map slots,
    // material, and cull distance. Driven by `world.jsonl` hot-reload
    // (`cn debug` only) when a newly authored Prop references a Mesh /
    // Model already present in the init world. The clone is non-cullable
    // (sentinel AABB) and joins `always_draw` since the init-time BVH
    // cannot refit; the dynamically added prop is drawn every frame,
    // like a streamed `VoxelWorld` chunk. Allocates the clone's (albedo,
    // normal) descriptor set from a `MAX_CLONE_DRAWS`-deep pool reserved
    // at init, and records `draw_idx → clone_offset` in
    // `clone_slot_by_draw_idx` so the legacy main pass + the texture-pool
    // rewires above can find it. Mirrors
    // `DxContext::clone_static_draw_object`. Reached only through the bin's
    // `cn debug` runtime-mutation path (dead in the FFI lib, live in the bin).
    #[allow(dead_code)]
    pub fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: crate::gfx::render_types::MaterialUniforms,
        cull_distance: f32,
    ) -> Result<usize, String> {
        use super::super::context::MAX_CLONE_DRAWS;

        if self.clone_object_sets.len() >= MAX_CLONE_DRAWS {
            return Err(format!(
                "clone_static_draw_object: MAX_CLONE_DRAWS ({MAX_CLONE_DRAWS}) exceeded"
            ));
        }
        let src = self.draw_objects.get(src_draw_idx).ok_or_else(|| {
            format!(
                "clone_static_draw_object: src draw {} out of range",
                src_draw_idx
            )
        })?;
        let obj = crate::gfx::render_types::DrawObject {
            vertex_offset: src.vertex_offset,
            vertex_count: src.vertex_count,
            index_offset: src.index_offset,
            index_count: src.index_count,
            base_vertex: src.base_vertex,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            // Sentinel AABB so the init-time BVH cull skips this draw; it
            // joins `always_draw` and is drawn every frame. Matches the
            // chunk pattern.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance,
            lod_alternates: src.lod_alternates.clone(),
        };

        // Lazily build the clone descriptor pool on first call. Sized for
        // MAX_CLONE_DRAWS (albedo, normal) sets: two samplers each.
        let pool = match self.clone_descriptor_pool {
            Some(p) => p,
            None => {
                let pool_sizes = [vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count((MAX_CLONE_DRAWS * 2) as u32)];
                let pool = unsafe {
                    self.device.create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&pool_sizes)
                            .max_sets(MAX_CLONE_DRAWS as u32),
                        None,
                    )
                }
                .map_err(|e| format!("clone descriptor pool: {e}"))?;
                self.clone_descriptor_pool = Some(pool);
                pool
            }
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(std::slice::from_ref(&self.descriptors.object_set_layout));
        let set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| format!("allocate clone descriptor set: {e}"))?[0];

        let last_tex = self.textures.len().saturating_sub(1);
        let last_nm = self.normal_map_textures.len().saturating_sub(1);
        let albedo_view = self.textures[texture_slot.min(last_tex)].view;
        let normal_view = self.normal_map_textures[normal_map_slot.min(last_nm)].view;
        self.write_object_image(set, 0, albedo_view);
        self.write_object_image(set, 1, normal_view);

        let clone_offset = self.clone_object_sets.len();
        self.clone_object_sets.push(set);
        self.clone_texture_slots.push(texture_slot);
        self.clone_normal_map_slots.push(normal_map_slot);

        self.draw_objects.push(obj);
        let new_idx = self.draw_objects.len() - 1;
        self.always_draw.push(new_idx as u32);
        self.clone_slot_by_draw_idx.insert(new_idx, clone_offset);
        Ok(new_idx)
    }
}
