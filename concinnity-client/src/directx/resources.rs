// src/directx/resources.rs
//
// Runtime GPU resource management for DxContext: texture-pool slot updates,
// mesh upload/eviction, chunk streaming, and skinned-mesh upload. Also owns
// the skinned-mesh pipelines (built lazily by `upload_skinned` the first time
// a SkinnedMesh is uploaded), mirroring metal/resources/skinning.rs.
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::gfx::mesh_payload::{SkinnedVertex, Vertex};
use crate::gfx::render_types::*;

use super::context::*;
use super::init::pipelines::{create_main_instanced_root_signature, create_main_pso};
use super::math::*;
use super::pipeline::{
    compile_hlsl, serialize_and_create_root_sig, shader_source, skinned_input_layout,
};
use super::texture::*;

// Skinned-mesh HLSL sources

// Skeletally animated sibling of MAIN_VERT_HLSL. Each vertex carries four joint
// indices + blend weights; the shader blends up to four joint matrices from the
// per-object structured buffer at t3 (linear blend skinning), applies the
// blended matrix to position/normal/tangent, then proceeds exactly like
// MAIN_VERT_HLSL. Paired with the regular MAIN_FRAG_HLSL.
const SKINNED_VERT_HLSL: &str = include_str!("shaders/skinned_vert.hlsl");

// Skeletally animated sibling of SHADOW_VERT_HLSL. Blends the joint matrices so
// a skinned mesh casts a correctly deformed shadow. Reads the per-object joint
// buffer at t0 (the shadow root signature has no texture registers, so t0 is
// free).
const SKINNED_SHADOW_VERT_HLSL: &str = include_str!("shaders/skinned_shadow_vert.hlsl");

const MAIN_FRAG_HLSL: &str = concinnity_core::build::shader::BUILTIN_DEFAULT_FRAG_HLSL;

// Skinned pipeline builders
//
// These mirror the static + shadow PSO builders in init/pipelines.rs but use
// the skinned vertex layout (80-byte SkinnedVertex with joint indices +
// weights) and pair with the skinned vertex shaders.

// Compile the skinned-mesh shader stages. Returns (main_skinned_vs,
// shadow_skinned_vs, frag_ps). The main skinned VS pairs with the standard
// fragment shader; the shadow skinned VS is depth-only. `frag_bytes`, when
// non-empty, is treated as pre-compiled DXBC (the same resolution the static
// path applies); otherwise the built-in `MAIN_FRAG_HLSL` is compiled.
// Compiled skinned-mesh shaders: main vertex, shadow vertex, fragment bytecode.
type SkinnedShaders = (Vec<u8>, Vec<u8>, Vec<u8>);

fn compile_skinned_shaders(frag_bytes: &[u8], hot_reload: bool) -> Result<SkinnedShaders, String> {
    let main_vs = compile_hlsl(
        &shader_source(hot_reload, "skinned_vert.hlsl", SKINNED_VERT_HLSL),
        "main",
        "vs_5_1",
    )?;
    let shadow_vs = compile_hlsl(
        &shader_source(
            hot_reload,
            "skinned_shadow_vert.hlsl",
            SKINNED_SHADOW_VERT_HLSL,
        ),
        "main",
        "vs_5_1",
    )?;
    let frag_ps = if !frag_bytes.is_empty() {
        frag_bytes.to_vec()
    } else {
        compile_hlsl(MAIN_FRAG_HLSL, "main", "ps_5_1")?
    };
    Ok((main_vs, shadow_vs, frag_ps))
}

// Same as the shadow root signature but with one extra root SRV at slot [2]
// (t0) carrying the per-object joint matrices. Used by the skinned shadow PSO.
fn create_skinned_shadow_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let params = [
        // [0] Root constants: model mat4 (16) + cascade_idx + 3 pad = 20 DWORDs at b0
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 20,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [1] Root CBV: shadow UBO (light_vps[4] + cascade_splits) at b1
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
        // [2] Root SRV: per-object joint matrices (t0, VS-only)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_VERTEX,
        },
    ];

    serialize_and_create_root_sig(device, &params, "skinned shadow root sig")
}

// Main-pass PSO for skinned geometry: the skinned vertex shader (80-byte
// layout) paired with the standard fragment shader. Uses the instanced root
// signature: its extra root SRV at slot [8] (t3) carries the joint matrices.
fn create_skinned_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
    ps: &[u8],
    rtv_format: DXGI_FORMAT,
    sample_count: u32,
) -> Result<ID3D12PipelineState, String> {
    let layout = skinned_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr() as _,
            BytecodeLength: ps.len(),
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: {
            let mut a = [DXGI_FORMAT_UNKNOWN; 8];
            a[0] = rtv_format;
            a
        },
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: {
                let mut arr = [D3D12_RENDER_TARGET_BLEND_DESC::default(); 8];
                arr[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: false.into(),
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                arr
            },
            ..Default::default()
        },
        ..Default::default()
    };

    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create skinned PSO: {e}"))
}

// Shadow-pass PSO for skinned geometry: the skinned shadow vertex shader
// (80-byte layout, depth-only). Uses the skinned shadow root signature.
fn create_skinned_shadow_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    vs: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let layout = skinned_input_layout();
    let pso_desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr() as _,
            BytecodeLength: vs.len(),
        },
        InputLayout: D3D12_INPUT_LAYOUT_DESC {
            pInputElementDescs: layout.as_ptr(),
            NumElements: layout.len() as u32,
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 0,
        DSVFormat: DXGI_FORMAT_D32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: true.into(),
            DepthBias: 1,
            DepthBiasClamp: 0.01,
            SlopeScaledDepthBias: 1.0,
            DepthClipEnable: true.into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        BlendState: D3D12_BLEND_DESC {
            ..Default::default()
        },
        ..Default::default()
    };

    unsafe { device.CreateGraphicsPipelineState(&pso_desc) }
        .map_err(|e| format!("create skinned shadow PSO: {e}"))
}

impl DxContext {
    // CPU descriptor handle for CBV/SRV/UAV heap `slot`.
    fn srv_slot_cpu(&self, slot: usize) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetCPUDescriptorHandleForHeapStart()
        };
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + slot * self.descriptors.srv_descriptor_size,
        }
    }

    // Re-point every per-object / per-cluster albedo SRV that resolves to
    // albedo-pool `slot` at the (just-swapped) `self.descriptors.textures[slot]` resource.
    //
    // The per-object / per-cluster SRVs are baked into the descriptor heap at
    // init from the texture selected by `texture_slot` (clamped to the pool
    // length), so a streamed swap must rewrite every heap slot that resolved
    // to this pool index -- mirroring the init loop in `Self::new`.
    fn rewrite_albedo_slot(&self, slot: usize) {
        let last = self.descriptors.textures.len() - 1;
        let resource = &self.descriptors.textures[slot];
        for (obj_idx, obj) in self.draw_objects.iter().enumerate() {
            if obj.texture_slot.min(last) != slot {
                continue;
            }
            // Runtime clones live in their own (albedo, normal) heap pool;
            // streamed `VoxelWorld` chunks share `chunk_srv_base_slot` and
            // are not refreshed here (their material is fixed at
            // `setup_chunk_streaming` time). Everything else is a build-time
            // draw at `3 + obj_idx * 2`.
            let albedo_cpu = if let Some(&clone_offset) = self.clone.slot_by_draw_idx.get(&obj_idx)
            {
                self.srv_slot_cpu(self.clone.srv_base_slot + clone_offset * 2)
            } else if obj_idx >= self.n_objects {
                continue;
            } else {
                self.srv_slot_cpu(3 + obj_idx * 2)
            };
            write_rgba8_srv(&self.device, resource, albedo_cpu);
        }
        let cluster_base = 3 + self.n_objects * 2;
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            if cluster.texture_slot.min(last) == slot {
                write_rgba8_srv(
                    &self.device,
                    resource,
                    self.srv_slot_cpu(cluster_base + cluster_idx * 2),
                );
            }
        }
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            if obj.texture_slot.min(last) == slot {
                write_rgba8_srv(
                    &self.device,
                    resource,
                    self.srv_slot_cpu(self.skinned.srv_base_slot + i * 2),
                );
            }
        }
        // Flat deduplicated pool: the swapped resource has exactly one descriptor
        // here (shared by every consumer), so one re-point refreshes the bindless
        // main pass + the RT hit shader at once.
        write_rgba8_srv(
            &self.device,
            resource,
            self.srv_slot_cpu(self.descriptors.flat_pool_base_slot + slot),
        );
    }

    // Re-point every per-object / per-cluster normal-map SRV that resolves to
    // normal-map-pool `slot` at the (just-swapped)
    // `self.descriptors.normal_map_textures[slot]` resource. The normal SRV sits one slot
    // after the albedo SRV in each (albedo, normal) heap pair.
    fn rewrite_normal_slot(&self, slot: usize) {
        let last = self.descriptors.normal_map_textures.len() - 1;
        let resource = &self.descriptors.normal_map_textures[slot];
        for (obj_idx, obj) in self.draw_objects.iter().enumerate() {
            if obj.normal_map_slot.min(last) != slot {
                continue;
            }
            // Same kind-routing as `rewrite_albedo_slot`: clones in the
            // clone pool, chunks skipped, build-time draws at the per-object
            // pair offset.
            let normal_cpu = if let Some(&clone_offset) = self.clone.slot_by_draw_idx.get(&obj_idx)
            {
                self.srv_slot_cpu(self.clone.srv_base_slot + clone_offset * 2 + 1)
            } else if obj_idx >= self.n_objects {
                continue;
            } else {
                self.srv_slot_cpu(3 + obj_idx * 2 + 1)
            };
            write_rgba8_srv(&self.device, resource, normal_cpu);
        }
        let cluster_base = 3 + self.n_objects * 2;
        for (cluster_idx, cluster) in self.instanced.clusters.iter().enumerate() {
            if cluster.normal_map_slot.min(last) == slot {
                write_rgba8_srv(
                    &self.device,
                    resource,
                    self.srv_slot_cpu(cluster_base + cluster_idx * 2 + 1),
                );
            }
        }
        for (i, obj) in self.skinned.draw_objects.iter().enumerate() {
            if obj.normal_map_slot.min(last) == slot {
                write_rgba8_srv(
                    &self.device,
                    resource,
                    self.srv_slot_cpu(self.skinned.srv_base_slot + i * 2 + 1),
                );
            }
        }
        // Flat deduplicated pool: the normal region follows the albedo region, so
        // normal pool slot `slot` lives at `flat_pool_base_slot + albedo_count +
        // slot`. One re-point refreshes the bindless main pass + RT hit shader.
        let albedo_count = self.descriptors.textures.len();
        write_rgba8_srv(
            &self.device,
            resource,
            self.srv_slot_cpu(self.descriptors.flat_pool_base_slot + albedo_count + slot),
        );
    }

    // Replace albedo texture-pool `slot` with freshly decoded RGBA8 pixels.
    //
    // The asset-streaming subsystem calls this to bring a texture resident
    // after init. Like Vulkan -- and unlike Metal, whose bind paths re-read
    // the texture pool every frame -- the D3D12 per-object / per-cluster SRVs
    // are baked into the descriptor heap at init, so a streamed swap must
    // rewrite every heap slot that samples this pool index. `wait_idle` first
    // guarantees no in-flight command list still reads the old descriptor (or
    // the old resource) before it is overwritten and dropped.
    pub fn update_texture_slot(
        &mut self,
        slot: usize,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        if slot >= self.descriptors.textures.len() {
            return Err(format!(
                "update_texture_slot: slot {} out of range (pool size {})",
                slot,
                self.descriptors.textures.len()
            ));
        }
        self.wait_idle();
        let texture =
            upload_texture_resource(&self.device, &self.command_queue, width, height, pixels)?;
        self.descriptors.textures[slot] = texture;
        self.rewrite_albedo_slot(slot);
        Ok(())
    }

    // Reset albedo texture-pool `slot` to a 1x1 mid-grey placeholder.
    //
    // Used by the asset-streaming subsystem to mark a slot whose texture is
    // not yet resident; a later `update_texture_slot` brings the real texture
    // back. The grey is distinct from the white no-texture fallback so a
    // not-yet-streamed slot reads differently under inspection.
    pub fn evict_texture_slot(&mut self, slot: usize) -> Result<(), String> {
        self.update_texture_slot(slot, 1, 1, &[128, 128, 128, 255])
    }

    // Replace normal-map pool `slot` with freshly decoded RGBA8 pixels.
    //
    // The normal-map counterpart of [`update_texture_slot`](Self::update_texture_slot).
    // Slot 0 is the flat-normal fallback and is never streamed; streamed maps
    // occupy slots >= 1.
    pub fn update_normal_map_slot(
        &mut self,
        slot: usize,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(), String> {
        if slot >= self.descriptors.normal_map_textures.len() {
            return Err(format!(
                "update_normal_map_slot: slot {} out of range (pool size {})",
                slot,
                self.descriptors.normal_map_textures.len()
            ));
        }
        self.wait_idle();
        let texture =
            upload_texture_resource(&self.device, &self.command_queue, width, height, pixels)?;
        self.descriptors.normal_map_textures[slot] = texture;
        self.rewrite_normal_slot(slot);
        Ok(())
    }

    // Reset normal-map pool `slot` to a 1x1 flat-normal placeholder.
    //
    // The normal-map counterpart of [`evict_texture_slot`](Self::evict_texture_slot).
    // A not-yet-streamed normal map reads as tangent-space (0,0,1), so the
    // surface shades flat (no bump detail) until its real map is resident --
    // the same value the slot-0 fallback carries.
    pub fn evict_normal_map_slot(&mut self, slot: usize) -> Result<(), String> {
        self.update_normal_map_slot(slot, 1, 1, &[128, 128, 255, 255])
    }

    // Replace the live colour-grading LUT with a fresh `size³` RGBA8 payload.
    // Driven by asset hot-reload (`cn debug` only) when the file-backed
    // `ColorLut` source is saved. Reuses the SRV heap slot the composite pass
    // already binds, so the new texture is picked up on the next `draw_frame`
    // with no pipeline or descriptor-table change. `wait_idle` first
    // guarantees no in-flight command list still references the old texture
    // (or the now-stale SRV) before it is overwritten and dropped. Mirrors
    // `MtlContext::update_color_lut`.
    #[allow(
        dead_code,
        reason = "cn-debug-only mutation/hot-reload; dead from the FFI lib crate's roots, live in the binary; see directx/decal.rs"
    )]
    pub fn update_color_lut(&mut self, size: u32, data: &[u8]) -> Result<(), String> {
        self.wait_idle();
        let srv_cpu = self.color_lut.srv_cpu;
        let srv_gpu = self.color_lut.srv_gpu;
        let new_lut = upload_color_lut(
            &self.device,
            &self.command_queue,
            size,
            data,
            srv_cpu,
            srv_gpu,
        )?;
        self.color_lut = new_lut;
        Ok(())
    }

    // Swap the live IBL cubemap pair for a freshly precomputed envmap payload.
    // Driven by asset hot-reload (`cn debug` only). Re-uploads into the same
    // SRV heap slots [1] (irradiance) + [2] (prefilter) the init path wrote,
    // so every pipeline that references those slots keeps working without a
    // descriptor-table rebind. The new payload may declare different mip /
    // face sizes than the original; `EnvironmentMapTextures` is replaced
    // wholesale and the next frame's `ViewUniforms` picks up the new
    // `prefilter_mip_count` from `self.env_map`. `wait_idle` first guarantees
    // no in-flight command list still references the old cubes (or the
    // now-stale SRVs) before they are overwritten and dropped. Mirrors
    // `MtlContext::update_environment_map`.
    #[allow(
        dead_code,
        reason = "cn-debug-only mutation/hot-reload; dead from the FFI lib crate's roots, live in the binary; see directx/decal.rs"
    )]
    pub fn update_environment_map(&mut self, payload: &[u8]) -> Result<(), String> {
        let view = crate::build::environment_map::deserialise(payload)
            .map_err(|e| format!("envmap hot-reload payload malformed: {e}"))?;
        self.wait_idle();
        let irr_srv_cpu = self.env_map.irradiance.srv_cpu;
        let irr_srv_gpu = self.env_map.irradiance.srv_gpu;
        let pre_srv_cpu = self.env_map.prefilter.srv_cpu;
        let pre_srv_gpu = self.env_map.prefilter.srv_gpu;
        let new_env = upload_environment_map(
            &self.device,
            &self.command_queue,
            view.irradiance_face,
            view.irradiance_bytes,
            view.prefilter_face,
            &view.prefilter_mip_bytes,
            irr_srv_cpu,
            irr_srv_gpu,
            pre_srv_cpu,
            pre_srv_gpu,
        )?;
        self.env_map = new_env;
        Ok(())
    }

    // GPU descriptor handle for a runtime clone's (albedo, normal) SRV pair.
    // The pair lives at `clone_srv_base_slot + clone_offset * 2`; the
    // 2-descriptor table the legacy main pass binds covers both slots.
    pub(super) fn clone_srv_gpu(&self, clone_offset: usize) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        let slot = self.clone.srv_base_slot + clone_offset * 2;
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + (slot * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // Append a new draw object that re-uses an existing slot's geometry
    // region (vertex / index offsets, base_vertex, LOD alternates) with a
    // fresh model matrix, texture / normal-map slots, material, and cull
    // distance. Driven by `world.jsonl` hot-reload (`cn debug` only) when a
    // newly authored Prop references a Mesh / Model already present in the
    // init world. The clone is non-cullable (sentinel AABB) and joins
    // `always_draw` since the init-time BVH cannot refit; the dynamically
    // added prop is drawn every frame, like a streamed `VoxelWorld` chunk.
    // Bakes the clone's (albedo, normal) SRV pair at the next free slot in
    // the clone descriptor pool reserved at init (`MAX_CLONE_DRAWS` pairs),
    // and records `draw_idx → clone_offset` in `clone_slot_by_draw_idx`
    // so the legacy main pass + `rewrite_albedo_slot` /
    // `rewrite_normal_slot` can find it. Mirrors
    // `MtlContext::clone_static_draw_object`.
    #[allow(
        dead_code,
        reason = "cn-debug-only mutation/hot-reload; dead from the FFI lib crate's roots, live in the binary; see directx/decal.rs"
    )]
    pub fn clone_static_draw_object(
        &mut self,
        src_draw_idx: usize,
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: MaterialUniforms,
        cull_distance: f32,
    ) -> Result<usize, String> {
        if self.clone.count >= MAX_CLONE_DRAWS {
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
        let obj = DrawObject {
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
            // Sentinel AABB so the init-time BVH cull skips the new draw:
            // it joins `always_draw` and is drawn every frame regardless of
            // camera position. Matches the runtime-streamed chunk pattern.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance,
            lod_alternates: src.lod_alternates.clone(),
        };

        let clone_offset = self.clone.count;
        let albedo_slot = self.clone.srv_base_slot + clone_offset * 2;
        let normal_slot = albedo_slot + 1;
        let last_tex = self.descriptors.textures.len().saturating_sub(1);
        let last_nm = self.descriptors.normal_map_textures.len().saturating_sub(1);
        write_rgba8_srv(
            &self.device,
            &self.descriptors.textures[texture_slot.min(last_tex)],
            self.srv_slot_cpu(albedo_slot),
        );
        write_rgba8_srv(
            &self.device,
            &self.descriptors.normal_map_textures[normal_map_slot.min(last_nm)],
            self.srv_slot_cpu(normal_slot),
        );

        self.draw_objects.push(obj);
        let new_idx = self.draw_objects.len() - 1;
        self.always_draw.push(new_idx as u32);
        self.clone.slot_by_draw_idx.insert(new_idx, clone_offset);
        self.clone.count += 1;
        Ok(new_idx)
    }

    // Copy `data` into a sub-region of a DEFAULT-heap geometry buffer.
    //
    // `dest` is a buffer currently in `usage_state` (the vertex or index
    // buffer). The copy goes through a temporary UPLOAD-heap staging buffer
    // and a one-shot command list that transitions the resource
    // `usage_state -> COPY_DEST -> usage_state` around a `CopyBufferRegion`.
    // The caller must `wait_idle` first: the COPY_DEST transition covers the
    // whole resource, so no in-flight command list may still reference it.
    fn write_geometry_region(
        &self,
        dest: &ID3D12Resource,
        usage_state: D3D12_RESOURCE_STATES,
        offset: u64,
        data: &[u8],
    ) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let upload = create_buffer(
            &self.device,
            data.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
        unsafe { upload.Map(0, None, Some(&mut ptr)) }
            .map_err(|e| format!("mesh region map: {e}"))?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            upload.Unmap(0, None);
        }
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let to_dst = transition_barrier(dest, usage_state, D3D12_RESOURCE_STATE_COPY_DEST);
            cmd.ResourceBarrier(&[to_dst]);
            cmd.CopyBufferRegion(dest, offset, &upload, 0, data.len() as u64);
            let back = transition_barrier(dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
            cmd.ResourceBarrier(&[back]);
        })
    }

    // Upload a streamed mesh's geometry into the shared vertex and index
    // buffers, place it via the sub-allocators, and mark the draw resident.
    //
    // The mesh-streaming subsystem calls this to bring a mesh resident after
    // init. The geometry is placed wherever the allocators find free space
    // (not the build-time region), so `DrawObject::vertex_offset` /
    // `index_offset` are rewritten here. `vertices` / `indices` must match the
    // fixed `vertex_count` / `index_count` recorded by `build_draw_list`.
    //
    // `indices` are mesh-relative (0-based); they are rebased onto the chosen
    // vertex region before upload, so the D3D12 draw can keep a 0 base-vertex.
    // `frame` reclaims deferred frees that have retired by then. `wait_idle`
    // runs first so the whole-resource COPY_DEST transition races no in-flight
    // command list (see `write_geometry_region`).
    pub fn upload_mesh(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        frame: u64,
    ) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("upload_mesh: draw object {} out of range", draw_idx))?;
        let (vertex_count, index_count) = (obj.vertex_count, obj.index_count);
        if vertices.len() != vertex_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} vertices, got {}",
                draw_idx,
                vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != index_count {
            return Err(format!(
                "upload_mesh: draw {} expects {} indices, got {}",
                draw_idx,
                index_count,
                indices.len()
            ));
        }

        // Reclaim frees whose in-flight frames have retired, then place the
        // geometry. build_draw_list never emits a zero-length mesh, so an
        // empty allocation request is treated as a hard error.
        self.mesh_stream.vtx_alloc.reclaim(frame);
        self.mesh_stream.idx_alloc.reclaim(frame);
        let v_len = std::mem::size_of_val(vertices);
        // Static IB is u32 (the per-scene total can exceed u16); per-mesh
        // indices come in as u16 (each mesh fits in u16, enforced by the
        // build-time splitter) and get widened on write below. Size the
        // allocation against the u32 stride. Mirrors metal's upload_mesh.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .mesh_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "upload_mesh: draw {}: no free vertex space for {} bytes",
                    draw_idx, v_len
                )
            })? as usize;
        let i_off = match self.mesh_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                // hand the vertex region back so a half-failed upload leaks no
                // space (frame 0: it was never written or drawn)
                self.mesh_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "upload_mesh: draw {}: no free index space for {} bytes",
                    draw_idx, i_len
                ));
            }
        };

        self.wait_idle();

        // Vertices copy verbatim. Indices are mesh-relative, so rebase them to
        // the vertex region the allocator chose: v_off is always a multiple of
        // size_of::<Vertex>() (every seed region and allocation is), so the
        // base is an exact vertex index.
        let vert_bytes =
            unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, v_len) };
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off as u64,
            vert_bytes,
        )?;
        let base = (v_off / std::mem::size_of::<Vertex>()) as u32;
        // Widen u16 → u32 while rebasing onto the chosen vertex region.
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = unsafe { std::slice::from_raw_parts(rebased.as_ptr() as *const u8, i_len) };
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off as u64,
            idx_bytes,
        )?;

        let obj = &mut self.draw_objects[draw_idx];
        obj.vertex_offset = v_off;
        obj.index_offset = i_off / std::mem::size_of::<u32>();
        obj.resident = true;
        Ok(())
    }

    // Overwrite a `Mesh` draw slot's vertex / index data in place. Driven by
    // asset hot-reload (`cn debug` only). New `vertices` / `indices` are
    // written at the draw object's existing offsets in the shared vertex /
    // index buffers, so the slot's count must match init-time; size-changing
    // reloads need `rebuild_static_geometry`, not this call. Each entry in
    // `lod_alternates` is written to the matching slot's pre-allocated LOD
    // region; LOD counts and per-LOD index counts must match init-time too.
    // Per-LOD `switch_distance`s are re-stored so JSON-side tweaks to
    // `lod_distances` propagate without a process restart. `wait_idle` is
    // folded into each `write_geometry_region` call (the whole-resource
    // COPY_DEST transition needs no in-flight command list referencing the
    // buffer). Mirrors `MtlContext::update_mesh_geometry`.
    #[allow(
        dead_code,
        reason = "cn-debug-only mutation/hot-reload; dead from the FFI lib crate's roots, live in the binary; see directx/decal.rs"
    )]
    pub fn update_mesh_geometry(
        &mut self,
        draw_idx: usize,
        vertices: &[Vertex],
        indices: &[u16],
        lod_alternates: &[(f32, Vec<u16>)],
    ) -> Result<(), String> {
        let obj = self.draw_objects.get(draw_idx).ok_or_else(|| {
            format!(
                "update_mesh_geometry: draw object {} out of range",
                draw_idx
            )
        })?;
        if vertices.len() != obj.vertex_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} vertices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.vertex_count,
                vertices.len()
            ));
        }
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_static_geometry)",
                draw_idx,
                obj.index_count,
                indices.len()
            ));
        }
        if lod_alternates.len() != obj.lod_alternates.len() {
            return Err(format!(
                "update_mesh_geometry: draw {} expects {} LOD alternate(s), got {} \
                 (LOD-count changes need rebuild_static_geometry)",
                draw_idx,
                obj.lod_alternates.len(),
                lod_alternates.len()
            ));
        }
        for (lod_idx, ((_, alt_idx), slice)) in lod_alternates
            .iter()
            .zip(obj.lod_alternates.iter())
            .enumerate()
        {
            if alt_idx.len() != slice.index_count {
                return Err(format!(
                    "update_mesh_geometry: draw {} LOD{} expects {} indices, got {} \
                     (LOD size changes need rebuild_static_geometry)",
                    draw_idx,
                    lod_idx + 1,
                    slice.index_count,
                    alt_idx.len()
                ));
            }
        }
        let v_off = obj.vertex_offset as u64;
        let i_off_bytes = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        // Static draws keep indices absolute (base_vertex == 0), so rebase
        // mesh-relative u16 indices onto the slot's vertex_offset and widen to
        // u32 before writing, matching the shared u32 index buffer and the
        // streaming upload_mesh path. `v_off` is always a multiple of
        // size_of::<Vertex>() (every region build_draw_list emits starts on a
        // vertex boundary).
        let base = (obj.vertex_offset / std::mem::size_of::<Vertex>()) as u32;
        let lod_byte_offsets: Vec<u64> = obj
            .lod_alternates
            .iter()
            .map(|s| (s.index_offset * std::mem::size_of::<u32>()) as u64)
            .collect();

        self.wait_idle();

        let vert_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off,
            vert_bytes,
        )?;
        let rebased: Vec<u32> = indices.iter().map(|&i| u32::from(i) + base).collect();
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                rebased.as_ptr() as *const u8,
                std::mem::size_of_val(rebased.as_slice()),
            )
        };
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off_bytes,
            idx_bytes,
        )?;
        // LOD alternate slots were laid out at init alongside LOD0 in the
        // same shared index buffer. Each alternate shares LOD0's vertex
        // region (LOD decimation never touches vertices), so rebase onto the
        // same `base`.
        for ((_, alt_idx), &alt_off_bytes) in lod_alternates.iter().zip(lod_byte_offsets.iter()) {
            let alt_rebased: Vec<u32> = alt_idx.iter().map(|&i| u32::from(i) + base).collect();
            let alt_bytes = unsafe {
                std::slice::from_raw_parts(
                    alt_rebased.as_ptr() as *const u8,
                    std::mem::size_of_val(alt_rebased.as_slice()),
                )
            };
            self.write_geometry_region(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                alt_off_bytes,
                alt_bytes,
            )?;
        }
        // Refresh the per-LOD switch distances so JSON-side tweaks to
        // `lod_distances` propagate without a process restart.
        let slot = &mut self.draw_objects[draw_idx];
        for ((switch_distance, _), slice) in
            lod_alternates.iter().zip(slot.lod_alternates.iter_mut())
        {
            slice.switch_distance = *switch_distance;
        }
        Ok(())
    }

    // Return a streamed mesh's geometry region to the sub-allocators and mark
    // the draw non-resident so it is skipped in every pass.
    //
    // `retire_frame` is the frame from which the freed region may be reused:
    // pass `current_frame + frames_in_flight` for a runtime eviction so a
    // still-in-flight command list never has its region overwritten by a
    // later `upload_mesh`, and `0` at init, where nothing has been drawn.
    // The region is not zeroed -- a non-resident draw is skipped everywhere,
    // and an `alloc` hands back exactly `size` bytes that `upload_mesh` then
    // fully overwrites, so no pass ever reads stale geometry.
    pub fn evict_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("evict_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.mesh_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.mesh_stream.idx_alloc.free(i_off, i_len, retire_frame);
        self.draw_objects[draw_idx].resident = false;
        Ok(())
    }

    // Seed the streamed-mesh sub-allocators with one reserved headroom block
    // (byte ranges in the shared vertex / index buffers), for the
    // shrinkable-seed path.
    //
    // The streamed geometry is not baked into the buffers at build time;
    // instead the buffers carry one zeroed headroom region (sized to the
    // cap-many resident meshes) at these offsets, which `compact_for_streaming`
    // appended before init. `retire_frame 0`: nothing has been drawn yet, so
    // the space is allocatable immediately -- mirrors `setup_chunk_streaming`'s
    // seeding. From then on `upload_mesh` / `evict_mesh` place and free streamed
    // meshes within it. Mirrors `MtlContext::seed_mesh_streaming`.
    pub fn seed_mesh_streaming(
        &mut self,
        vtx_offset: u64,
        vtx_bytes: u64,
        idx_offset: u64,
        idx_bytes: u64,
    ) {
        self.mesh_stream.vtx_alloc.free(vtx_offset, vtx_bytes, 0);
        self.mesh_stream.vtx_alloc.reclaim(0);
        self.mesh_stream.idx_alloc.free(idx_offset, idx_bytes, 0);
        self.mesh_stream.idx_alloc.reclaim(0);
    }

    // GPU descriptor handle for the shared chunk (albedo, normal) SRV pair.
    // Valid only after `setup_chunk_streaming` has populated the two slots.
    pub(super) fn chunk_srv_gpu(&self) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        let base = unsafe {
            self.descriptors
                .srv_heap
                .GetGPUDescriptorHandleForHeapStart()
        };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr
                + (self.chunk_stream.srv_base_slot * self.descriptors.srv_descriptor_size) as u64,
        }
    }

    // Grow the shared vertex/index buffers by a headroom region for streamed
    // `VoxelWorld` chunks, seed the chunk sub-allocators with it, and bake the
    // shared chunk (albedo, normal) SRV pair from the world's chunk material.
    //
    // Called once at init by `GraphicsSystem` when a `VoxelWorld` is present.
    // The build-time geometry is copied verbatim into the start of the new
    // (larger) DEFAULT-heap buffers; chunks are placed in the appended
    // headroom by `add_chunk_mesh`. This runs before the first frame, so no
    // in-flight command list references the replaced buffers.
    pub fn setup_chunk_streaming(
        &mut self,
        chunk_vtx_bytes: usize,
        chunk_idx_bytes: usize,
        texture_slot: usize,
        normal_map_slot: usize,
    ) -> Result<(), String> {
        self.wait_idle();
        let old_v_len = self.geometry.vertex_buffer_view.SizeInBytes as u64;
        let old_i_len = self.geometry.index_buffer_view.SizeInBytes as u64;
        let new_v_len = old_v_len + chunk_vtx_bytes as u64;
        let new_i_len = old_i_len + chunk_idx_bytes as u64;

        // Buffers are created in COMMON; the CopyBufferRegion below implicitly
        // promotes the destination COMMON -> COPY_DEST.
        let new_vbuf = create_buffer(
            &self.device,
            new_v_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let new_ibuf = create_buffer(
            &self.device,
            new_i_len,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // Copy the build-time geometry into the start of the grown buffers so
        // every existing draw's offsets stay valid.
        one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
            let v_src = transition_barrier(
                &self.geometry.vertex_buffer,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            let i_src = transition_barrier(
                &self.geometry.index_buffer,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            cmd.ResourceBarrier(&[v_src, i_src]);
            cmd.CopyBufferRegion(&new_vbuf, 0, &self.geometry.vertex_buffer, 0, old_v_len);
            cmd.CopyBufferRegion(&new_ibuf, 0, &self.geometry.index_buffer, 0, old_i_len);
            let v_dst = transition_barrier(
                &new_vbuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            );
            let i_dst = transition_barrier(
                &new_ibuf,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            );
            cmd.ResourceBarrier(&[v_dst, i_dst]);
        })?;

        self.geometry.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { new_vbuf.GetGPUVirtualAddress() },
            SizeInBytes: new_v_len as u32,
            StrideInBytes: std::mem::size_of::<Vertex>() as u32,
        };
        self.geometry.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { new_ibuf.GetGPUVirtualAddress() },
            SizeInBytes: new_i_len as u32,
            // Static IB is u32 (matches the `Format` chosen in init/mod.rs).
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
        };
        self.geometry.vertex_buffer = new_vbuf;
        self.geometry.index_buffer = new_ibuf;

        // Seed the chunk allocators with the appended headroom. retire_frame 0:
        // nothing has been drawn, so the space is reusable immediately.
        self.chunk_stream
            .vtx_alloc
            .free(old_v_len, chunk_vtx_bytes as u64, 0);
        self.chunk_stream
            .idx_alloc
            .free(old_i_len, chunk_idx_bytes as u64, 0);

        // Bake the shared chunk (albedo, normal) SRV pair from the chunk
        // material's texture-pool slots, clamped to the pool lengths.
        let last_tex = self.descriptors.textures.len().saturating_sub(1);
        let last_nm = self.descriptors.normal_map_textures.len().saturating_sub(1);
        write_rgba8_srv(
            &self.device,
            &self.descriptors.textures[texture_slot.min(last_tex)],
            self.srv_slot_cpu(self.chunk_stream.srv_base_slot),
        );
        write_rgba8_srv(
            &self.device,
            &self.descriptors.normal_map_textures[normal_map_slot.min(last_nm)],
            self.srv_slot_cpu(self.chunk_stream.srv_base_slot + 1),
        );
        Ok(())
    }

    // Place one streamed chunk's geometry in the chunk headroom region and
    // add (or recycle) a `DrawObject` for it; returns the draw-list index.
    //
    // The chunk is non-cullable and joins the `always_draw` set: the streaming
    // window already bounds the resident chunk count. Indices stay
    // mesh-relative (0-based) and the draw passes the vertex region's base as
    // `base_vertex`, so a chunk placed past the 65 535-vertex `u16` index
    // range still renders. `frame` reclaims retired deferred frees first.
    // `wait_idle` runs before the geometry copy so the whole-resource
    // COPY_DEST transition races no in-flight command list.
    #[allow(clippy::too_many_arguments)]
    pub fn add_chunk_mesh(
        &mut self,
        vertices: &[Vertex],
        indices: &[u16],
        model: [[f32; 4]; 4],
        texture_slot: usize,
        normal_map_slot: usize,
        material: crate::gfx::render_types::MaterialUniforms,
        frame: u64,
    ) -> Result<usize, String> {
        if vertices.is_empty() || indices.is_empty() {
            return Err("add_chunk_mesh: empty chunk geometry".to_string());
        }
        self.chunk_stream.vtx_alloc.reclaim(frame);
        self.chunk_stream.idx_alloc.reclaim(frame);

        let v_len = std::mem::size_of_val(vertices);
        // Static IB is u32; chunk indices come in as u16 and get widened on
        // write. Size the allocation against the u32 stride.
        let i_len = indices.len() * std::mem::size_of::<u32>();
        let v_off = self
            .chunk_stream
            .vtx_alloc
            .alloc(v_len as u64)
            .ok_or_else(|| {
                format!(
                    "add_chunk_mesh: no free chunk vertex space for {} bytes",
                    v_len
                )
            })? as usize;
        let i_off = match self.chunk_stream.idx_alloc.alloc(i_len as u64) {
            Some(o) => o as usize,
            None => {
                self.chunk_stream
                    .vtx_alloc
                    .free(v_off as u64, v_len as u64, 0);
                return Err(format!(
                    "add_chunk_mesh: no free chunk index space for {} bytes",
                    i_len
                ));
            }
        };

        self.wait_idle();

        // Vertices and indices both copy verbatim: the indices stay 0-based and
        // the draw fixes them up with `base_vertex`.
        let vert_bytes =
            unsafe { std::slice::from_raw_parts(vertices.as_ptr() as *const u8, v_len) };
        self.write_geometry_region(
            &self.geometry.vertex_buffer,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_off as u64,
            vert_bytes,
        )?;
        // Chunk indices stay mesh-relative; the draw fixes them up with
        // `base_vertex`. Widen u16 → u32 to match the static IB's stride.
        let widened: Vec<u32> = indices.iter().map(|&i| u32::from(i)).collect();
        let idx_bytes = unsafe { std::slice::from_raw_parts(widened.as_ptr() as *const u8, i_len) };
        self.write_geometry_region(
            &self.geometry.index_buffer,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_off as u64,
            idx_bytes,
        )?;

        // v_off is a multiple of size_of::<Vertex>() (the headroom start and
        // every alloc are), so the base is an exact vertex index.
        let base_vertex = (v_off / std::mem::size_of::<Vertex>()) as i32;
        let obj = DrawObject {
            vertex_offset: v_off,
            vertex_count: vertices.len(),
            index_offset: i_off / std::mem::size_of::<u32>(),
            index_count: indices.len(),
            base_vertex,
            model,
            texture_slot,
            normal_map_slot,
            material,
            visible: true,
            resident: true,
            // Non-cullable: degenerate AABB disables frustum/distance culling.
            bb_min: [f32::NAN; 3],
            bb_max: [f32::NAN; 3],
            cull_distance: 0.0,
            // Streamed chunks always render at the build-time mesh; no LOD.
            lod_alternates: Vec::new(),
        };

        // Reuse a vacated chunk slot when one is free, else append. A reused
        // slot is already in `always_draw`; a fresh one must be added.
        let draw_idx = if let Some(slot) = self.chunk_stream.free_slots.pop() {
            self.draw_objects[slot] = obj;
            slot
        } else {
            self.draw_objects.push(obj);
            let idx = self.draw_objects.len() - 1;
            self.always_draw.push(idx as u32);
            idx
        };
        // Seed the G-buffer pre-pass's previous-model snapshot for a recycled
        // slot so a fresh chunk does not inherit the removed chunk's transform
        // and ghost for one frame. A fresh append is past the snapshot's end
        // and the pre-pass falls back to the current model itself.
        if let Some(gbuffer) = &self.gbuffer {
            let mut prev = gbuffer.prev_models.borrow_mut();
            if draw_idx < prev.len() {
                prev[draw_idx] = model;
            }
        }
        Ok(draw_idx)
    }

    // Free a streamed chunk's geometry region and retire its `DrawObject`
    // slot for reuse.
    //
    // `retire_frame` is `current_frame + frames_in_flight` so an in-flight
    // command list never has the freed region overwritten by a later
    // `add_chunk_mesh`. The slot stays in `draw_objects` / `always_draw` but
    // is marked non-resident and invisible, so every pass skips it. The region
    // is not zeroed -- a non-resident draw is skipped everywhere and an
    // `alloc` hands back exactly `size` bytes that `add_chunk_mesh` fully
    // overwrites.
    pub fn remove_chunk_mesh(&mut self, draw_idx: usize, retire_frame: u64) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get(draw_idx)
            .ok_or_else(|| format!("remove_chunk_mesh: draw object {} out of range", draw_idx))?;
        let v_off = obj.vertex_offset as u64;
        let v_len = (obj.vertex_count * std::mem::size_of::<Vertex>()) as u64;
        let i_off = (obj.index_offset * std::mem::size_of::<u32>()) as u64;
        let i_len = (obj.index_count * std::mem::size_of::<u32>()) as u64;
        self.chunk_stream.vtx_alloc.free(v_off, v_len, retire_frame);
        self.chunk_stream.idx_alloc.free(i_off, i_len, retire_frame);
        let obj = &mut self.draw_objects[draw_idx];
        obj.visible = false;
        obj.resident = false;
        self.chunk_stream.free_slots.push(draw_idx);
        Ok(())
    }

    // Rewrite a resident chunk's model matrix.
    //
    // Used by camera-relative rendering: when the camera crosses into a new
    // chunk the render origin follows it, so every resident chunk is rebased
    // onto the new origin. Only the model matrix changes -- the geometry stays
    // where it was uploaded.
    pub fn set_chunk_model(&mut self, draw_idx: usize, model: [[f32; 4]; 4]) -> Result<(), String> {
        let obj = self
            .draw_objects
            .get_mut(draw_idx)
            .ok_or_else(|| format!("set_chunk_model: draw object {} out of range", draw_idx))?;
        obj.model = model;
        Ok(())
    }
}

impl DxContext {
    // Upload skinned-mesh geometry and build the skinned render pipelines.
    //
    // Called once at init by `GraphicsSystem` when the world declares at least
    // one `SkinnedMesh`. The skinned vertex + shadow shaders are compiled from
    // the inline HLSL; the fragment shader is shared with the static path. The
    // joint matrices live in per-(frame, object) upload buffers the skinned
    // passes bind as a root SRV. With no skinned meshes this is never called
    // and every skinned pass is skipped.
    pub fn upload_skinned(
        &mut self,
        vertices: &[SkinnedVertex],
        indices: &[u16],
        draw_objects: Vec<SkinnedDrawObject>,
        frag_bytes: &[u8],
    ) -> Result<(), String> {
        if draw_objects.is_empty() || vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }
        if draw_objects.len() > MAX_SKINNED_OBJECTS {
            return Err(format!(
                "skinned: {} skinned meshes exceeds MAX_SKINNED_OBJECTS ({})",
                draw_objects.len(),
                MAX_SKINNED_OBJECTS
            ));
        }
        self.wait_idle();

        let (skinned_vs, skinned_shadow_vs, frag_ps) =
            compile_skinned_shaders(frag_bytes, self.hot_reload.enabled)?;

        // Main skinned pipeline: reuses the instanced root signature (its root
        // SRV at t3 carries the joint matrices) and the off-screen HDR target.
        let skinned_root_sig = dump_on_err(
            self.info_queue.as_ref(),
            create_main_instanced_root_signature(&self.device),
        )?;
        let skinned_pso = dump_on_err(
            self.info_queue.as_ref(),
            create_skinned_pso(
                &self.device,
                &skinned_root_sig,
                &skinned_vs,
                &frag_ps,
                HDR_FORMAT,
                self.hdr.msaa_samples,
            ),
        )?;

        // Skinned shadow pipeline: built only when the static shadow pass is
        // active, so a skinned mesh casts a correctly deformed shadow.
        let (skinned_shadow_root_sig, skinned_shadow_pso) = if self.shadow_pso.is_some() {
            let sr = dump_on_err(
                self.info_queue.as_ref(),
                create_skinned_shadow_root_signature(&self.device),
            )?;
            let sp = dump_on_err(
                self.info_queue.as_ref(),
                create_skinned_shadow_pso(&self.device, &sr, &skinned_shadow_vs),
            )?;
            (Some(sr), Some(sp))
        } else {
            (None, None)
        };

        // Shared skinned vertex/index buffers (DEFAULT heap, GPU-copied once).
        let vtx_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                std::mem::size_of_val(indices),
            )
        };
        // GENERIC_READ (rather than the narrower VERTEX_AND_CONSTANT_BUFFER /
        // INDEX_BUFFER) so these stay both vertex/index-bindable for the skinned
        // main + shadow passes AND shader-readable as raw root SRVs for the RT
        // skin compute dispatch (bind-pose VB) and the RT reflection trace (u16
        // IB). GENERIC_READ is a superset of both, so no per-frame transition on
        // these shared resources is needed.
        let skinned_vertex_buffer = upload_buffer(
            &self.device,
            &self.command_queue,
            vtx_bytes,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let skinned_index_buffer = upload_buffer(
            &self.device,
            &self.command_queue,
            idx_bytes,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        self.skinned.vertex_buffer_view = D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: unsafe { skinned_vertex_buffer.GetGPUVirtualAddress() },
            SizeInBytes: vtx_bytes.len() as u32,
            StrideInBytes: std::mem::size_of::<SkinnedVertex>() as u32,
        };
        self.skinned.index_buffer_view = D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: unsafe { skinned_index_buffer.GetGPUVirtualAddress() },
            SizeInBytes: idx_bytes.len() as u32,
            Format: DXGI_FORMAT_R16_UINT,
        };

        // Per-(frame, object) joint-matrix upload buffers, each MAX_JOINTS
        // float4x4 matrices, persistently mapped.
        //
        // The buffer is seeded with `MAX_JOINTS` identity matrices once at
        // creation. `upload_joint_matrices` later overwrites only the first
        // `mats.len()` slots each frame; anything past the live pose count
        // keeps the identity seed, so a vertex whose `joints.{xyzw}` indexes
        // past the live range degenerates into an LBS of identity matrices
        // (i.e. its bind-pose position) instead of reading uninitialised
        // UPLOAD-heap memory and producing an arbitrary spike. The seed is
        // also what the renderer wants on frame 0 before the first pose
        // arrives: every joint is identity, so the mesh shows in bind pose.
        let joint_buf_bytes = (MAX_JOINTS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
        let identity_seed: Vec<[[f32; 4]; 4]> = vec![IDENTITY4; MAX_JOINTS];
        let mut joint_buffers: Vec<Vec<ID3D12Resource>> = Vec::with_capacity(FRAMES);
        let mut joint_ptrs: Vec<Vec<*mut u8>> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let mut frame_bufs: Vec<ID3D12Resource> = Vec::with_capacity(draw_objects.len());
            let mut frame_ptrs: Vec<*mut u8> = Vec::with_capacity(draw_objects.len());
            for _ in 0..draw_objects.len() {
                let buf = create_buffer(
                    &self.device,
                    joint_buf_bytes,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )
                .map_err(|e| format!("skinned joint buf: {e}"))?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                unsafe {
                    buf.Map(0, None, Some(&mut ptr))
                        .map_err(|e| format!("map skinned joint buf: {e}"))?;
                    std::ptr::copy_nonoverlapping(
                        identity_seed.as_ptr() as *const u8,
                        ptr as *mut u8,
                        joint_buf_bytes as usize,
                    );
                }
                frame_bufs.push(buf);
                frame_ptrs.push(ptr as *mut u8);
            }
            joint_buffers.push(frame_bufs);
            joint_ptrs.push(frame_ptrs);
        }

        // Bake each skinned object's (albedo, normal) SRV pair from its
        // material's texture-pool slots, clamped to the pool lengths.
        let last_tex = self.descriptors.textures.len().saturating_sub(1);
        let last_nm = self.descriptors.normal_map_textures.len().saturating_sub(1);
        for (i, obj) in draw_objects.iter().enumerate() {
            write_rgba8_srv(
                &self.device,
                &self.descriptors.textures[obj.texture_slot.min(last_tex)],
                self.srv_slot_cpu(self.skinned.srv_base_slot + i * 2),
            );
            write_rgba8_srv(
                &self.device,
                &self.descriptors.normal_map_textures[obj.normal_map_slot.min(last_nm)],
                self.srv_slot_cpu(self.skinned.srv_base_slot + i * 2 + 1),
            );
        }

        // Seed each object's joint matrices to identity (bind pose) so the mesh
        // renders undeformed until the first `update_skinned_pose`.
        self.skinned.joint_matrices = draw_objects
            .iter()
            .map(|o| vec![IDENTITY4; o.joint_count.max(1)])
            .collect();

        self.skinned.pso = Some(skinned_pso);
        self.skinned.root_sig = Some(skinned_root_sig);
        self.skinned.shadow_pso = skinned_shadow_pso;
        self.skinned.shadow_root_sig = skinned_shadow_root_sig;
        self.skinned.vertex_buffer = Some(skinned_vertex_buffer);
        self.skinned.index_buffer = Some(skinned_index_buffer);
        self.skinned.joint_buffers = joint_buffers;
        self.skinned.joint_ptrs = joint_ptrs;
        self.skinned.draw_objects = draw_objects;

        // GPU-driven main-pass skinning fold. When the bindless cull path is
        // active, build the `rt_skin` compute pipeline (reused independently of
        // RT) + one UAV-writable deformed-vertex buffer per frame-in-flight, sized
        // to all skinned verts. Each frame `encode_skin` poses the bind-pose verts
        // into this frame's buffer and the bindless main pass's 2nd ExecuteIndirect
        // draws the skinned records the cull buffers reserved. Setting
        // `self.n_skinned` here (not at init) engages the fold; a build failure
        // (e.g. DXC unavailable) leaves it 0 and the legacy skinned main pass runs.
        // The cull / object / draw-args / indirect buffers already reserved the
        // skinned tail at init via the threaded `n_skinned` capacity.
        if self.cull.main_bindless_pso.is_some() && self.cull_count() > 0 {
            let stride = std::mem::size_of::<Vertex>();
            let deformed_bytes = (vertices.len() * stride).max(stride) as u64;
            let mut deformed_buffers: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
            let mut deformed_vbvs: Vec<D3D12_VERTEX_BUFFER_VIEW> = Vec::with_capacity(FRAMES);
            for _ in 0..FRAMES {
                let buf =
                    create_uav_buffer(&self.device, deformed_bytes, D3D12_RESOURCE_STATE_COMMON)?;
                let vbv = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: unsafe { buf.GetGPUVirtualAddress() },
                    SizeInBytes: deformed_bytes as u32,
                    StrideInBytes: stride as u32,
                };
                deformed_buffers.push(buf);
                deformed_vbvs.push(vbv);
            }
            // Move COMMON -> VERTEX_AND_CONSTANT_BUFFER so the per-frame skin
            // pass's VERTEX -> UAV -> VERTEX transition cycle is valid from frame 0.
            one_shot_submit(&self.device, &self.command_queue, |cmd| unsafe {
                let barriers: Vec<D3D12_RESOURCE_BARRIER> = deformed_buffers
                    .iter()
                    .map(|b| {
                        transition_barrier(
                            b,
                            D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
                        )
                    })
                    .collect();
                cmd.ResourceBarrier(&barriers);
            })?;
            match super::raytrace::build_rt_skin_pipeline(&self.device, self.hot_reload.enabled) {
                Ok(skin) => {
                    self.skinned.skin_pipeline = Some(skin);
                    self.skinned.deformed_buffers = deformed_buffers;
                    self.skinned.deformed_vbvs = deformed_vbvs;
                    // Fresh ring: no slot has been posed yet, so the G-buffer
                    // velocity must treat the previous deformed buffer as the
                    // current one until a full frame has primed it.
                    self.skinned
                        .deformed_primed
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                    self.n_skinned = self.skinned.draw_objects.len();
                }
                Err(e) => {
                    tracing::warn!(
                        "skinned: rt_skin pipeline build failed ({e}); skinned meshes use \
                         the legacy main pass"
                    );
                }
            }
        }

        // Build the unified G-buffer pre-pass's skinned PSO now that the
        // skinned vertex layout exists. Without it, skinned meshes are missing
        // from the normal+depth / roughness / velocity targets, so they ghost
        // under TAA, fail to occlude in SSAO, and do not appear in the SSR
        // reflection ray-march. A no-op when no screen-space consumer drives the
        // pre-pass (the G-buffer is absent).
        if let Some(gbuffer) = self.gbuffer.as_mut() {
            gbuffer.ensure_skinned_pso(
                &self.device,
                self.hot_reload.enabled,
                self.info_queue.as_ref(),
            )?;
        }

        Ok(())
    }

    // Overwrite a `SkinnedMesh` draw slot's vertex + index data in the shared
    // skinned vertex / index buffers in place. Driven by asset hot-reload
    // (`cn debug` only). The slot's vertex region starts at
    // `vertex_base * size_of::<SkinnedVertex>()` and is `vertices.len()`
    // vertices wide; the index region lives at the slot's init-time
    // `index_offset` / `index_count`. Indices are rebased onto `vertex_base`
    // before writing (matching the init-time `upload_skinned` rebasing).
    // `indices.len()` must match init-time; size-changing reloads route
    // through `rebuild_skinned_geometry`. Joint-count
    // changes resize the per-slot joint-matrix buffers via
    // `update_skinned_skeleton`. Pipelines stay untouched.
    // Mirrors `MtlContext::update_skinned_mesh_geometry`.
    #[allow(
        dead_code,
        reason = "cn-debug-only mutation/hot-reload; dead from the FFI lib crate's roots, live in the binary; see directx/decal.rs"
    )]
    pub fn update_skinned_mesh_geometry(
        &mut self,
        skinned_index: usize,
        vertex_base: u16,
        vertices: &[SkinnedVertex],
        indices: &[u16],
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .draw_objects
            .get(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_mesh_geometry: skinned object {} out of range",
                    skinned_index
                )
            })?;
        if indices.len() != obj.index_count {
            return Err(format!(
                "update_skinned_mesh_geometry: skinned {} expects {} indices, got {} \
                 (in-place path is size-matched only; size changes route through \
                 rebuild_skinned_geometry)",
                skinned_index,
                obj.index_count,
                indices.len()
            ));
        }
        let v_buf = self.skinned.vertex_buffer.as_ref().cloned().ok_or(
            "update_skinned_mesh_geometry: no skinned vertex buffer (was upload_skinned called?)",
        )?;
        let i_buf = self.skinned.index_buffer.as_ref().cloned().ok_or(
            "update_skinned_mesh_geometry: no skinned index buffer (was upload_skinned called?)",
        )?;
        // Check the vertex region fits inside the live buffer. The shared
        // buffer was sized once at `upload_skinned` to hold every skinned
        // mesh's vertices; vertex_base + vertices.len() must stay within that
        // region or a neighbouring slot would be overwritten.
        let v_byte_off = (vertex_base as usize) * std::mem::size_of::<SkinnedVertex>();
        let v_byte_len = std::mem::size_of_val(vertices);
        let v_buf_len = self.skinned.vertex_buffer_view.SizeInBytes as usize;
        if v_byte_off + v_byte_len > v_buf_len {
            return Err(format!(
                "update_skinned_mesh_geometry: vertex region [{}, {}) overruns skinned \
                 vertex buffer length {}",
                v_byte_off,
                v_byte_off + v_byte_len,
                v_buf_len
            ));
        }
        let i_byte_off = (obj.index_offset * std::mem::size_of::<u16>()) as u64;
        let rebased: Vec<u16> = indices
            .iter()
            .map(|&i| i.checked_add(vertex_base))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                format!(
                    "update_skinned_mesh_geometry: index rebase by {} overflows u16 \
                     (skinned slot {})",
                    vertex_base, skinned_index
                )
            })?;

        self.wait_idle();

        let vert_bytes = unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices),
            )
        };
        self.write_geometry_region(
            &v_buf,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            v_byte_off as u64,
            vert_bytes,
        )?;
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                rebased.as_ptr() as *const u8,
                std::mem::size_of_val(rebased.as_slice()),
            )
        };
        self.write_geometry_region(
            &i_buf,
            D3D12_RESOURCE_STATE_INDEX_BUFFER,
            i_byte_off,
            idx_bytes,
        )?;
        Ok(())
    }

    // Update a skinned slot's joint count and resize its per-slot CPU
    // joint-matrix buffer to match. Driven by asset hot-reload (`cn debug`
    // only) when a re-imported `.glb`'s skeleton has a different joint
    // count than the slot was initialised with. New entries are seeded to
    // identity so the slot renders undeformed until the next
    // `update_skinned_pose` writes the new pose. The shared skinned
    // pipelines + per-frame GPU joint buffers stay untouched; the GPU
    // buffers are sized for `MAX_JOINTS` at init, so a joint-count change
    // only resizes the CPU-side `skinned_joint_matrices[skinned_index]`
    // Vec (and `SkinnedDrawObject.joint_count`); the next
    // `upload_joint_matrices` writes the new (capped at `MAX_JOINTS`)
    // count of matrices into the per-frame ring. The velocity pre-pass
    // reads the previous-frame pose from `(frame_idx + FRAMES - 1) %
    // FRAMES` of the same ring rather than a separate CPU mirror, so no
    // "prev" array needs resizing; joints past the previous pose's
    // length retain the init identity seed (or stale prior data) for one
    // post-reload frame and then catch up. Mirrors
    // `MtlContext::update_skinned_skeleton`.
    #[allow(
        dead_code,
        reason = "cn-debug-only mutation/hot-reload; dead from the FFI lib crate's roots, live in the binary; see directx/decal.rs"
    )]
    pub fn update_skinned_skeleton(
        &mut self,
        skinned_index: usize,
        new_joint_count: usize,
    ) -> Result<(), String> {
        let obj = self
            .skinned
            .draw_objects
            .get_mut(skinned_index)
            .ok_or_else(|| {
                format!(
                    "update_skinned_skeleton: skinned object {} out of range",
                    skinned_index
                )
            })?;
        let capped = new_joint_count.min(MAX_JOINTS);
        obj.joint_count = capped;
        let size = capped.max(1);
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.resize(size, IDENTITY4);
        }
        Ok(())
    }

    // Replace the skinning matrices for one skinned object. Called each frame
    // from `GraphicsSystem` with the pose `AnimationSystem` computed. Out-of-
    // range indices are ignored.
    pub fn update_skinned_pose(&mut self, skinned_index: usize, matrices: &[[[f32; 4]; 4]]) {
        if let Some(slot) = self.skinned.joint_matrices.get_mut(skinned_index) {
            slot.clear();
            slot.extend_from_slice(matrices);
            if slot.is_empty() {
                slot.push(IDENTITY4);
            }
        }
    }

    // Copy this frame's skinning matrices into the per-frame joint buffers.
    // Called from `record_frame` before the skinned shadow + main passes.
    pub(super) fn upload_joint_matrices(&self, frame_idx: usize) {
        let Some(frame_ptrs) = self.skinned.joint_ptrs.get(frame_idx) else {
            return;
        };
        for (i, mats) in self.skinned.joint_matrices.iter().enumerate() {
            let Some(&dst) = frame_ptrs.get(i) else {
                continue;
            };
            let n = mats.len().min(MAX_JOINTS);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mats.as_ptr() as *const u8,
                    dst,
                    n * std::mem::size_of::<[[f32; 4]; 4]>(),
                );
            }
        }
    }

    // GPU virtual address of skinned object `i`'s joint buffer for `frame_idx`.
    pub(super) fn skinned_joint_gva(&self, frame_idx: usize, i: usize) -> u64 {
        unsafe { self.skinned.joint_buffers[frame_idx][i].GetGPUVirtualAddress() }
    }
}

// World-ShaderStage runtime hot-swap (RenderBackend::update_world_shader_pipelines)

// cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots,
// live in the concinnity binary. See the note on the analogous block in
// [directx/particle.rs].
#[allow(
    dead_code,
    reason = "cn-debug-only runtime-mutation surface; dead from the FFI lib crate's roots, live in the concinnity binary"
)]
impl DxContext {
    // Rebuild the world-driven graphics pipelines from freshly compiled
    // `ShaderStage` bytes and hot-swap them, for the live-reload path
    // (`reload_shader_stages` -> here). A custom-shader world's vertex +
    // fragment stages drive the legacy static main pipeline; the instanced
    // pipeline pairs the world's instanced vertex stage with the same fragment;
    // the skinned main pipeline keeps its engine-internal 80-byte vertex shader
    // and only swaps the fragment (matching `upload_skinned`, which ignores the
    // static vertex bytes). The shadow, bindless, and cull pipelines are
    // engine-internal and reload through `reload_shaders`, not here.
    //
    // Everything is built into temporaries first; any compile / PSO-create
    // failure early-returns with the live pipelines untouched, mirroring
    // `reload_shaders`. Mirrors `MtlContext::update_world_shader_pipelines`.
    pub fn update_world_shader_pipelines(
        &mut self,
        vert_bytes: Option<&[u8]>,
        frag_bytes: Option<&[u8]>,
        _shadow_bytes: Option<&[u8]>,
        vert_instanced_bytes: Option<&[u8]>,
    ) -> Result<(), String> {
        let vert = vert_bytes.ok_or_else(|| {
            "update_world_shader_pipelines: vertex shader bytes are required".to_string()
        })?;
        let frag = frag_bytes.ok_or_else(|| {
            "update_world_shader_pipelines: fragment shader bytes are required".to_string()
        })?;
        let iq = self.info_queue.as_ref();
        let msaa = self.hdr.msaa_samples;

        // Legacy static main pipeline (the path a custom-shader world uses; the
        // bindless variant stays engine-internal). Reuses the live root sig.
        let new_main = dump_on_err(
            iq,
            create_main_pso(
                &self.device,
                &self.main_root_sig,
                vert,
                frag,
                HDR_FORMAT,
                msaa,
            ),
        )?;

        // Instanced pipeline: rebuilt only when one is live. Needs the world's
        // instanced vertex stage paired with the fresh fragment.
        let new_instanced = if let (Some(_), Some(root_sig)) = (
            self.instanced.pso.as_ref(),
            self.instanced.root_sig.as_ref(),
        ) {
            let inst = vert_instanced_bytes.ok_or_else(|| {
                "update_world_shader_pipelines: instanced vertex shader bytes are required \
                 when an instanced pipeline is live"
                    .to_string()
            })?;
            Some(dump_on_err(
                iq,
                create_main_pso(&self.device, root_sig, inst, frag, HDR_FORMAT, msaa),
            )?)
        } else {
            None
        };

        // Skinned main pipeline: rebuilt only when one is live. Keeps its
        // engine-internal skinned vertex shader; only the fragment changes
        // (`compile_skinned_shaders` treats the fresh `frag` as the precompiled
        // pixel shader, exactly as `upload_skinned` does at init).
        let new_skinned = if let (Some(_), Some(root_sig)) =
            (self.skinned.pso.as_ref(), self.skinned.root_sig.as_ref())
        {
            let (skinned_vs, _skinned_shadow_vs, frag_ps) =
                compile_skinned_shaders(frag, self.hot_reload.enabled)?;
            Some(dump_on_err(
                iq,
                create_skinned_pso(
                    &self.device,
                    root_sig,
                    &skinned_vs,
                    &frag_ps,
                    HDR_FORMAT,
                    msaa,
                ),
            )?)
        } else {
            None
        };

        // All builds succeeded: swap into the live context. The next frame's
        // draw calls bind the freshly compiled pipelines.
        self.main_pso = new_main;
        if let Some(p) = new_instanced {
            self.instanced.pso = Some(p);
        }
        if let Some(p) = new_skinned {
            self.skinned.pso = Some(p);
        }
        Ok(())
    }
}
