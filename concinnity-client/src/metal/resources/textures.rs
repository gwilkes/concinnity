// src/metal/resources/textures.rs
//
// Texture-pool slot updates + IBL / colour-grading hot-swap. Driven both by
// the streaming subsystem (per-slot upload + eviction placeholders) and by
// asset hot-reload (`cn debug` only) for envmaps + LUTs.
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::incompatible_msrv)]

use crate::metal::context::MtlContext;
use crate::metal::texture::upload_texture;

impl MtlContext {
    // Replace albedo texture-pool `slot` with freshly decoded RGBA8 pixels.
    //
    // The asset-streaming subsystem calls this to bring a texture resident
    // after init. Both the bindless pool bind and the per-draw fallback bind
    // read `self.textures` fresh each frame, so the swapped texture is picked
    // up on the next `draw_frame` with no pipeline rebuild.
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
        self.textures[slot] = upload_texture(&self.device, width, height, pixels)?;
        Ok(())
    }

    // Reset albedo texture-pool `slot` to a 1x1 mid-grey placeholder.
    //
    // Used by the asset-streaming subsystem to mark a slot whose texture is
    // not yet resident; a later `update_texture_slot` brings the real texture
    // back. The grey is distinct from the white no-texture fallback so a
    // not-yet-streamed slot reads differently under inspection.
    pub fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        if slot >= self.textures.len() {
            return Err(format!(
                "evict_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.textures.len()
            ));
        }
        self.textures[slot] = upload_texture(&self.device, 1, 1, &[128, 128, 128, 255])?;
        Ok(())
    }

    // Replace normal-map pool `slot` with freshly decoded RGBA8 pixels.
    //
    // The normal-map counterpart of [`update_texture_slot`](Self::update_texture_slot):
    // the asset-streaming subsystem calls this to bring a normal map resident
    // after init. Slot 0 is the flat-normal fallback and is never streamed;
    // streamed maps occupy slots >= 1.
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
        self.normal_map_textures[slot] = upload_texture(&self.device, width, height, pixels)?;
        Ok(())
    }

    // Reset normal-map pool `slot` to a 1x1 flat-normal placeholder.
    //
    // The normal-map counterpart of [`evict_texture_slot`](Self::evict_texture_slot).
    // A not-yet-streamed normal map reads as tangent-space (0,0,1), so the
    // surface shades flat (no bump detail) until its real map is resident --
    // the same value the slot-0 fallback carries.
    pub fn evict_normal_map_slot(&mut self, slot: usize) -> Result<(), String> {
        if slot >= self.normal_map_textures.len() {
            return Err(format!(
                "evict_normal_map_slot: slot {} out of range (pool size {})",
                slot,
                self.normal_map_textures.len()
            ));
        }
        self.normal_map_textures[slot] = upload_texture(&self.device, 1, 1, &[128, 128, 255, 255])?;
        Ok(())
    }

    // Swap the live 3D colour-grading LUT for a fresh payload. Driven by
    // asset hot-reload (`cn debug` only). The composite pass binds
    // `self.color_lut` every frame, so the new texture is sampled on the
    // next `draw_frame` with no pipeline rebuild.
    pub fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        let tex = crate::metal::texture::upload_color_lut(&self.device, size, data)?;
        self.color_lut = tex;
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). The fragment shader binds
    // `self.env_map.irradiance` and `self.env_map.prefilter` every frame, so
    // the new cubes are sampled on the next `draw_frame` with no pipeline
    // rebuild. The new payload may declare different mip / face sizes than
    // the original -- `EnvironmentMapTextures` is replaced wholesale.
    pub fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = crate::build::environment_map::deserialise(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {}", e))?;
        let new_env = crate::metal::texture::upload_environment_map(
            &self.device,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )?;
        self.env_map = new_env;
        Ok(())
    }

    // Upload a baked reflection-probe payload to GPU cube textures and return
    // them. The caller installs the result into `probe_maps` (the specular
    // reflection source, distinct from `env_map`); the skybox + diffuse irradiance
    // keep sampling `env_map`, so the visible sky is never replaced by the capture.
    pub(in crate::metal) fn build_probe_textures(
        &self,
        payload: &[u8],
    ) -> Result<super::super::texture::EnvironmentMapTextures, String> {
        let view = crate::build::environment_map::deserialise(payload)
            .map_err(|e| format!("reflection probe payload malformed: {}", e))?;
        crate::metal::texture::upload_environment_map(
            &self.device,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
        )
    }
}
