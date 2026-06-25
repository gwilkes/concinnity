// src/directx/cull.rs
//
// Compute-driven GPU culling: one thread per build-time `DrawObject`
// frustum/distance-tests the object's `GpuObjectData` AABB against the six
// CPU-extracted frustum planes, optionally Hi-Z-occlusion-tests it against the
// previous frame's depth pyramid (see `directx/hiz.rs`), and writes one
// `ExecuteIndirect` command into the per-frame argument buffer: survivors get
// `instance_count = 1`, culled or disabled objects get `instance_count = 0`
// (a no-op draw). The main bindless pass then issues the whole buffer with a
// single `ExecuteIndirect`, so the CPU never walks the draw list.
//
// The frustum and distance maths mirror `gfx::frustum` exactly (the six
// planes are extracted CPU-side already normalised) so the GPU path culls
// identically to the CPU BVH path it replaces. `GpuObjectData` / `GpuDrawArgs`
// mirror `gfx::render_types`; `IndirectCommand` is a b0 root constant (the
// object id) followed by `D3D12_DRAW_INDEXED_ARGUMENTS`, matching the command
// signature built by `create_cull_command_signature`. Mirrors src/metal/cull.rs.

use windows::Win32::Graphics::Direct3D12::*;

use crate::directx::context::DxContext;
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::transition_barrier;

// HLSL source

pub const CULL_COMPUTE_HLSL: &str = include_str!("shaders/cull.hlsl");

// DWORD count of the cull kernel's `CullParams` cbuffer: `float4 planes[6]`
// (24) + `float3 cam_pos` + `uint object_count` (4) + `float4x4
// prev_view_proj` (16) + `float2 hiz_size` + `uint hiz_mip_count` + `uint
// hiz_enabled` (4) = 48 DWORDs. Pushed inline as a root-constant block.
pub(in crate::directx) const CULL_PARAMS_DWORDS: u32 = 48;

// Byte stride of one `IndirectCommand` in the cull kernel's output buffer: a
// 1-DWORD object-id root constant + `D3D12_DRAW_INDEXED_ARGUMENTS` (5 DWORDs).
pub(in crate::directx) const INDIRECT_COMMAND_STRIDE: u32 = 24;

// Root-constant block for the GPU-cull compute kernel (192 bytes = 48 DWORDs).
// Must match the `CullParams` cbuffer at b0 in CULL_COMPUTE_HLSL: six
// already-normalised frustum planes (xyz = normal, w = d), the camera position
// + object count (sharing the last 16-byte cbuffer row), the previous frame's
// view-projection (4x4), and the Hi-Z metadata (dims, mip count, enable flag).
#[derive(Copy, Clone)]
#[repr(C)]
struct CullParams {
    planes: [[f32; 4]; 6],
    cam_pos: [f32; 3],
    object_count: u32,
    prev_view_proj: [[f32; 4]; 4],
    hiz_size: [f32; 2],
    hiz_mip_count: u32,
    hiz_enabled: u32,
}

// Pipeline + command signature builders

// Compile the phase-1 GPU-cull compute kernel (`main`) to DXBC.
pub(in crate::directx) fn compile_cull_shader(hot_reload: bool) -> Result<Vec<u8>, String> {
    compile_hlsl(
        &shader_source(hot_reload, "cull.hlsl", CULL_COMPUTE_HLSL),
        "main",
        "cs_5_1",
    )
}

// Compile the phase-2 GPU-cull compute kernel (`main_phase2`) for two-pass
// occlusion. Same source / root signature as phase 1, different entry point.
pub(in crate::directx) fn compile_cull_shader_phase2(hot_reload: bool) -> Result<Vec<u8>, String> {
    compile_hlsl(
        &shader_source(hot_reload, "cull.hlsl", CULL_COMPUTE_HLSL),
        "main_phase2",
        "cs_5_1",
    )
}

// Compile the GPU-driven shadow cull kernel (`main_shadow`): light-frustum only
// (no Hi-Z, no distance cull, no status write). Same source / root signature as
// phase 1, different entry point.
pub(in crate::directx) fn compile_cull_shader_shadow(hot_reload: bool) -> Result<Vec<u8>, String> {
    compile_hlsl(
        &shader_source(hot_reload, "cull.hlsl", CULL_COMPUTE_HLSL),
        "main_shadow",
        "cs_5_1",
    )
}

// Root signature for the GPU-cull compute kernel: a `CullParams` root-constant
// block at b0, the `GpuObjectData` + `GpuDrawArgs` inputs as root SRVs (t0,
// t1), an SRV descriptor table at t2 (Hi-Z texture covering all mips), the
// indirect-command output as a root UAV (u0), and the per-object cull-status
// output as a root UAV (u1). The Hi-Z table is the only reason this root
// signature needs a descriptor table: root SRVs can only carry raw / structured
// buffers, not Texture2D. Shared by the phase-1 (`main`) and phase-2
// (`main_phase2`) cull kernels; phase 1 writes both `commands` + `cull_status`,
// phase 2 reads `cull_status` and writes the phase-2 `commands`.
pub(in crate::directx) fn create_cull_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let hiz_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 2, // t2
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // [0] Root constants b0: CullParams (planes + cam_pos + object_count +
        //     prev_view_proj + hiz metadata)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: CULL_PARAMS_DWORDS,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Root SRV t0: StructuredBuffer<GpuObjectData>
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root SRV t1: StructuredBuffer<GpuDrawArgs>
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [3] Descriptor table SRV t2: Hi-Z Texture2D<float> covering all mips
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &hiz_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [4] Root UAV u0: RWStructuredBuffer<IndirectCommand>
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [5] Root UAV u1: RWStructuredBuffer<uint> cull_status (two-pass)
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "cull root sig")
}

// Compute pipeline state for the GPU-cull kernel.
pub(in crate::directx) fn create_cull_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
) -> Result<ID3D12PipelineState, String> {
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // Borrow the root signature without an AddRef. `pRootSignature` is a
        // `ManuallyDrop`, so a `clone()` here is never released and leaks one
        // reference per PSO creation. The caller's `&root_sig` outlives the
        // synchronous pipeline-state creation, so copying the raw pointer is sound.
        pRootSignature: unsafe { std::mem::transmute_copy(root_sig) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: cs.as_ptr() as _,
            BytecodeLength: cs.len(),
        },
        ..Default::default()
    };
    unsafe { device.CreateComputePipelineState(&desc) }.map_err(|e| format!("create cull PSO: {e}"))
}

// Command signature for the GPU-driven main pass `ExecuteIndirect`: each
// command sets the b0 object-id root constant, then issues a `DrawIndexed`.
// The command layout must match the `IndirectCommand` struct the cull kernel
// writes. A command signature that touches a root constant must be created
// against the consuming root signature (the bindless main root signature).
pub(in crate::directx) fn create_cull_command_signature(
    device: &ID3D12Device,
    bindless_root_sig: &ID3D12RootSignature,
) -> Result<ID3D12CommandSignature, String> {
    let arg_descs = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: 0,
                    DestOffsetIn32BitValues: 0,
                    Num32BitValuesToSet: 1,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED,
            ..Default::default()
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: INDIRECT_COMMAND_STRIDE,
        NumArgumentDescs: arg_descs.len() as u32,
        pArgumentDescs: arg_descs.as_ptr(),
        NodeMask: 0,
    };
    let mut sig: Option<ID3D12CommandSignature> = None;
    unsafe { device.CreateCommandSignature(&desc, bindless_root_sig, &mut sig) }
        .map_err(|e| format!("create cull command signature: {e}"))?;
    sig.ok_or_else(|| "create cull command signature: returned None".to_string())
}

// Per-frame buffer fill + encoder

impl DxContext {
    // Total records the GPU-driven cull + bindless main pass processes: the
    // build-time static objects, the instanced-cluster instances folded in after
    // them, then the skinned objects (`n_objects + n_instances + n_skinned`). The
    // cull dispatch + the `GpuObjectData` / `GpuDrawArgs` / indirect buffers all
    // count this; the main pass then draws the static+instance prefix and the
    // skinned tail with two `ExecuteIndirect` calls. With no instanced props /
    // skinned meshes (or a non-bindless world) the extra terms are 0, leaving it
    // equal to the static `n_objects`.
    pub(in crate::directx) fn cull_count(&self) -> usize {
        self.n_objects + self.n_instances + self.n_chunk + self.n_skinned
    }

    // Buffer index of the first streamed-chunk record. The chunk reserve is
    // `[chunk_record_base(), skinned_record_base())`; resident chunks are packed
    // into the front of it each frame and the unused tail is disabled. Chunks ride
    // the static + instance prefix `ExecuteIndirect` (their geometry already lives
    // in the shared VB/IB), so this is just the instance tail.
    pub(in crate::directx) fn chunk_record_base(&self) -> usize {
        self.n_objects + self.n_instances
    }

    // Buffer index of the first skinned record. The static + instance + chunk
    // prefix the first `ExecuteIndirect` draws ends here; the skinned tail
    // `[skinned_record_base(), cull_count())` is the second `ExecuteIndirect` (over
    // the per-frame deformed VB). The chunk reserve sits inside the prefix, so the
    // skinned base is past it.
    pub(in crate::directx) fn skinned_record_base(&self) -> usize {
        self.n_objects + self.n_instances + self.n_chunk
    }

    // Rebuild this frame's `StructuredBuffer<GpuDrawArgs>` for the GPU-cull
    // compute kernel: one 16-byte record per build-time `DrawObject`, carrying
    // the indexed-draw arguments the kernel encodes plus the per-frame
    // cull-decision bits (`update_visibility` / streaming residency). Streamed
    // chunks (past `n_objects`) are skipped; a no-op when the bindless pass is
    // inactive. The per-object `(index_offset, index_count)` is the active LOD
    // slice picked by camera distance, so the bindless main pass renders the
    // chosen LOD with no shader-side change. Mirrors `metal/cull.rs`.
    pub(in crate::directx) fn build_draw_args_buffer(&self, frame_idx: usize, cam_pos: [f32; 3]) {
        use crate::gfx::render_types::{GpuDrawArgs, draw_args_flags};
        let Some(&ptr) = self.cull.draw_args_buffer_ptrs.get(frame_idx) else {
            return;
        };
        let stride = std::mem::size_of::<GpuDrawArgs>();
        for (i, obj) in self.draw_objects.iter().take(self.n_objects).enumerate() {
            // Per-frame active LOD pick. Objects with no alternates fall
            // straight through to LOD0.
            let d = crate::gfx::lod::camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                flags: draw_args_flags(obj.visible, obj.resident, obj.cullable()),
            };
            // SAFETY: the buffer was sized for `n_objects` records and the
            // loop is bounded by `take(n_objects)`, so `i * stride` is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuDrawArgs as *const u8,
                    ptr.add(i * stride),
                    stride,
                );
            }
        }

        // Streamed chunks: one draw-arg each in the reserved region at
        // `[chunk_record_base() + k]`. Chunk geometry lives in the shared VB/IB, so
        // the args carry the chunk's own `base_vertex` + index slice and the chunk
        // rides the static + instance prefix `ExecuteIndirect`. Chunks are
        // non-cullable (NaN AABB -> `cullable()` false), so a resident chunk draws
        // unconditionally; a freed slot's `resident` clear disables it. The unused
        // reserve tail is disabled (ENABLED clear -> the cull kernel emits a no-op
        // and never reads its stale object record).
        let chunk_base = self.chunk_record_base();
        let n_resident_chunks = self.for_each_chunk_record(|k, obj| {
            // Chunks have no LOD alternates; `active_lod(0.0)` returns the base
            // slice (and avoids a NaN camera distance from the chunk's NaN AABB).
            let (index_offset, index_count) = obj.active_lod(0.0);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: obj.base_vertex as u32,
                flags: draw_args_flags(obj.visible, obj.resident, obj.cullable()),
            };
            // SAFETY: the chunk reserve is `[chunk_base, chunk_base + n_chunk)` and
            // `for_each_chunk_record` caps `k < n_chunk`, so the write is in range.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuDrawArgs as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        });
        // Disable the unused chunk reserve tail so vacated / never-used slots draw
        // nothing (the cull kernel skips `objects[i]` for an ENABLED-clear record).
        let disabled = GpuDrawArgs {
            index_count: 0,
            index_offset: 0,
            base_vertex: 0,
            flags: 0,
        };
        for k in n_resident_chunks..self.n_chunk {
            // SAFETY: `k < n_chunk`, so `chunk_base + k < skinned_record_base()`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &disabled as *const GpuDrawArgs as *const u8,
                    ptr.add((chunk_base + k) * stride),
                    stride,
                );
            }
        }

        // Skinned objects: one record each in the reserved tail at
        // `[skinned_record_base(), cull_count())`. The main pass's 2nd
        // `ExecuteIndirect` draws them against the per-frame deformed-vertex
        // buffer with the skinned u16 index buffer bound, so `base_vertex = 0`
        // and the active-LOD slice is the element offset into the skinned IB.
        // Active LOD is picked from the camera distance to the model translation.
        let base = self.skinned_record_base();
        for (k, obj) in self
            .skinned
            .draw_objects
            .iter()
            .take(self.n_skinned)
            .enumerate()
        {
            let d = crate::gfx::lod::skinned_camera_distance(obj, cam_pos);
            let (index_offset, index_count) = obj.active_lod(d);
            let rec = GpuDrawArgs {
                index_count: index_count as u32,
                index_offset: index_offset as u32,
                base_vertex: 0,
                // Skinned objects always carry a finite padded bind-pose AABB
                // (`pack_skinned_record`), so they are cullable + resident; the
                // cull kernel frustum/Hi-Z tests them like any static object.
                flags: draw_args_flags(obj.visible, true, true),
            };
            // SAFETY: the buffers reserved `n_skinned` records past
            // `skinned_record_base()` at init (threaded capacity), and the loop
            // is bounded by `self.skinned.draw_objects.len() == self.n_skinned`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &rec as *const GpuDrawArgs as *const u8,
                    ptr.add((base + k) * stride),
                    stride,
                );
            }
        }
    }

    // Dispatch the cull compute pass: fills the per-frame draw-args buffer,
    // packs the camera frustum planes + Hi-Z metadata + previous frame's VP,
    // runs one thread per build-time object to test it against the frustum,
    // distance, and (when valid) the Hi-Z pyramid, and writes the surviving
    // `ExecuteIndirect` commands into this frame's indirect buffer. The
    // indirect buffer rests in `INDIRECT_ARGUMENT` between frames; the
    // dispatch flips it to `UAV` and back. Caller must already have built
    // the per-frame `GpuObjectData` buffer (the bindless main pass owns
    // that step).
    pub(in crate::directx) fn encode_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
    ) {
        self.build_draw_args_buffer(frame_idx, cam_pos);

        let cull_pso = self
            .cull
            .cull_pso
            .as_ref()
            .expect("encode_cull: cull_pso missing");
        let cull_root = self
            .cull
            .cull_root_sig
            .as_ref()
            .expect("encode_cull: cull_root_sig missing");
        let indirect = &self.cull.indirect_cmd_buffers[frame_idx];
        let object_gva =
            unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let draw_args_gva =
            unsafe { self.cull.draw_args_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        // Per-object cull-status buffer (u1): always allocated alongside the
        // indirect buffer, always written, read by phase 2 under two-pass
        // occlusion (ignored under single-pass). Resting state is
        // `UNORDERED_ACCESS`, so it binds as a root UAV with no transition.
        let cull_status_gva =
            unsafe { self.cull.cull_status_buffers[frame_idx].GetGPUVirtualAddress() };

        // Pack the six already-normalised frustum planes + the previous frame's
        // VP + Hi-Z metadata for the kernel. Hi-Z is gated on the per-context
        // `hiz_valid` flag (false on the very first frame, before any Hi-Z
        // pyramid has been built) and on whether a `HiZResources` was built
        // at init (`self.cull.hiz.is_some()`).
        let (hiz_size, hiz_mip_count, hiz_srv, hiz_enabled) = match self.cull.hiz.as_ref() {
            Some(h) if self.cull.hiz_valid.get() => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
                1u32,
            ),
            Some(h) => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
                0u32,
            ),
            None => ([1.0, 1.0], 1, None, 0u32),
        };
        let mut cull_params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos,
            object_count: self.cull_count() as u32,
            prev_view_proj: self.cull.prev_view_proj.get(),
            hiz_size,
            hiz_mip_count,
            hiz_enabled,
        };
        for (i, p) in frustum.planes.iter().enumerate() {
            cull_params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }

        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(cull_root);
            cmd.SetPipelineState(cull_pso);
            // The Hi-Z SRV lives in the global SRV heap; the descriptor table
            // at root param [3] needs that heap bound. Bind it
            // unconditionally: when Hi-Z is disabled the kernel just skips
            // sampling, but the descriptor table still has to point at a
            // valid (live) descriptor.
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRoot32BitConstants(
                0,
                CULL_PARAMS_DWORDS,
                &cull_params as *const CullParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetComputeRootShaderResourceView(1, object_gva);
            cmd.SetComputeRootShaderResourceView(2, draw_args_gva);
            if let Some(srv) = hiz_srv {
                cmd.SetComputeRootDescriptorTable(3, srv);
            }
            cmd.SetComputeRootUnorderedAccessView(4, indirect.GetGPUVirtualAddress());
            cmd.SetComputeRootUnorderedAccessView(5, cull_status_gva);
            // One thread per build-time object, 64-wide threadgroups.
            cmd.Dispatch((self.cull_count() as u32).div_ceil(64), 1, 1);
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }

    // Cull one reflection-probe cube face into the reserved capture slot
    // (`directx/probe.rs::bake_ring_slot` = `FRAMES`). Mirrors `encode_cull` but
    // (a) indexes the reserved ring slot the frame never touches, (b) forces Hi-Z
    // occlusion OFF (the only pyramid is in the main camera's screen space, useless
    // for a cube face), and (c) does NOT rebuild the draw-args buffer -- the bake
    // builds object + draw-args into the reserved slot once at capture start (they
    // are frustum-independent) and re-culls per face here. `frustum` is the face's
    // view frustum and `cam_pos` the probe capture point. The object count is the
    // same `cull_count()` the frame uses; the bake renders only the static + instance
    // prefix, so culled skinned records are written but never executed.
    pub(in crate::directx) fn encode_probe_cull(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        slot: usize,
        frustum: &crate::gfx::frustum::Frustum,
        cam_pos: [f32; 3],
    ) {
        let cull_pso = self
            .cull
            .cull_pso
            .as_ref()
            .expect("encode_probe_cull: cull_pso missing");
        let cull_root = self
            .cull
            .cull_root_sig
            .as_ref()
            .expect("encode_probe_cull: cull_root_sig missing");
        let indirect = &self.cull.indirect_cmd_buffers[slot];
        let object_gva = unsafe { self.cull.object_buffer_resources[slot].GetGPUVirtualAddress() };
        let draw_args_gva =
            unsafe { self.cull.draw_args_buffer_resources[slot].GetGPUVirtualAddress() };
        let cull_status_gva = unsafe { self.cull.cull_status_buffers[slot].GetGPUVirtualAddress() };

        // A Hi-Z SRV is still bound (the root signature's descriptor table at [3]
        // must point at a live descriptor) but `hiz_enabled = 0` makes the kernel
        // skip the occlusion test. The probe captures whatever is in the face
        // frustum, occluded or not.
        let hiz_srv = self.cull.hiz.as_ref().map(|h| h.srv_gpu);
        let mut cull_params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos,
            object_count: self.cull_count() as u32,
            prev_view_proj: self.cull.prev_view_proj.get(),
            hiz_size: [1.0, 1.0],
            hiz_mip_count: 1,
            hiz_enabled: 0,
        };
        for (i, p) in frustum.planes.iter().enumerate() {
            cull_params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }

        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(cull_root);
            cmd.SetPipelineState(cull_pso);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRoot32BitConstants(
                0,
                CULL_PARAMS_DWORDS,
                &cull_params as *const CullParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetComputeRootShaderResourceView(1, object_gva);
            cmd.SetComputeRootShaderResourceView(2, draw_args_gva);
            if let Some(srv) = hiz_srv {
                cmd.SetComputeRootDescriptorTable(3, srv);
            }
            cmd.SetComputeRootUnorderedAccessView(4, indirect.GetGPUVirtualAddress());
            cmd.SetComputeRootUnorderedAccessView(5, cull_status_gva);
            cmd.Dispatch((self.cull_count() as u32).div_ceil(64), 1, 1);
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }

    // Per-cascade GPU cull for the GPU-driven shadow pass. Uses the frustum-only
    // shadow cull kernel (`cull_pso_shadow` = `main_shadow`): one dispatch per
    // re-rendered cascade tests every record (static + instances + skinned) against
    // that cascade's light frustum -- extracted from `light_vps[c]` -- with NO Hi-Z
    // (sun cascades have no light-space depth pyramid) and NO per-object distance
    // cull (the cascade frustum already bounds the shadow draw distance; the view
    // `cull_distance` must not silence shadows). Writes the surviving `ExecuteIndirect`
    // commands into cascade `c`'s region of this frame's shadow indirect buffer. The region is selected by binding the cull output UAV at a
    // per-cascade GPU-address offset (`c * cull_count` records), so `commands[i]`
    // lands at physical index `c*cull_count + i`; the object + draw-args inputs are
    // the same camera-independent buffers the main cull reads, so only the frustum
    // + output region differ. Status writes go to a scratch buffer (never read;
    // the shared `cull_status` is reserved for the phase-2 main cull, which runs
    // after this pass). The whole indirect buffer flips INDIRECT_ARGUMENT -> UAV
    // for the dispatches and back; the shadow pass then issues each cascade region
    // with `ExecuteIndirect`. Skipped cascades (not in `render_mask`) keep their
    // prior region untouched (and their depth slice is not re-rendered). A no-op
    // when the GPU-driven shadow resources are absent or `cull_count() == 0`.
    pub(in crate::directx) fn encode_shadow_culls(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        render_mask: u32,
        cam_pos: [f32; 3],
    ) {
        use crate::gfx::render_types::NUM_SHADOW_CASCADES;
        let (Some(shadow_cull_pso), Some(cull_root), Some(indirect), Some(status)) = (
            self.cull.cull_pso_shadow.as_ref(),
            self.cull.cull_root_sig.as_ref(),
            self.cull.shadow_indirect_buffers.get(frame_idx),
            self.cull.shadow_cull_status_buffers.get(frame_idx),
        ) else {
            return;
        };
        let n_cull = self.cull_count();
        if n_cull == 0 {
            return;
        }

        let object_gva =
            unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let draw_args_gva =
            unsafe { self.cull.draw_args_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let status_gva = unsafe { status.GetGPUVirtualAddress() };
        let base_gva = unsafe { indirect.GetGPUVirtualAddress() };

        // Hi-Z is disabled for the shadow cull (`hiz_enabled = 0`), so the kernel
        // never samples the pyramid; the descriptor table at root [3] still has to
        // point at a live descriptor, bound exactly like `encode_cull`.
        let (hiz_size, hiz_mip_count, hiz_srv) = match self.cull.hiz.as_ref() {
            Some(h) => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
            ),
            None => ([1.0, 1.0], 1, None),
        };

        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(cull_root);
            cmd.SetPipelineState(shadow_cull_pso);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRootShaderResourceView(1, object_gva);
            cmd.SetComputeRootShaderResourceView(2, draw_args_gva);
            if let Some(srv) = hiz_srv {
                cmd.SetComputeRootDescriptorTable(3, srv);
            }
            cmd.SetComputeRootUnorderedAccessView(5, status_gva);

            for c in 0..NUM_SHADOW_CASCADES {
                if render_mask & (1u32 << c) == 0 {
                    continue;
                }
                let frustum = crate::gfx::frustum::Frustum::from_view_projection(
                    self.shadow.uniforms.light_vps[c],
                );
                let mut cull_params = CullParams {
                    planes: [[0.0; 4]; 6],
                    cam_pos,
                    object_count: n_cull as u32,
                    // Unused with Hi-Z disabled (the projection is never taken).
                    prev_view_proj: [[0.0; 4]; 4],
                    hiz_size,
                    hiz_mip_count,
                    hiz_enabled: 0,
                };
                for (i, p) in frustum.planes.iter().enumerate() {
                    cull_params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
                }
                cmd.SetComputeRoot32BitConstants(
                    0,
                    CULL_PARAMS_DWORDS,
                    &cull_params as *const CullParams as *const std::ffi::c_void,
                    0,
                );
                // Cascade `c`'s output region: offset the indirect UAV's GPU address
                // by `c * n_cull` commands so the kernel's `commands[i]` write lands
                // in this cascade's slice. Root UAV GPU addresses only need element
                // alignment (the stride is a multiple of 4), like the instanced
                // path's per-bucket root-SRV bumps.
                let region_gva = base_gva + (c * n_cull * INDIRECT_COMMAND_STRIDE as usize) as u64;
                cmd.SetComputeRootUnorderedAccessView(4, region_gva);
                cmd.Dispatch((n_cull as u32).div_ceil(64), 1, 1);
            }

            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }

    // Reflected-frustum mirror cull for the planar reflection pass. For each
    // `(reflected frustum, reflected eye)` in `planes`, re-runs the GPU cull into
    // that plane's region of `indirect` (one region of `region_count` commands per
    // plane), reading the FRAME's camera-independent object + draw-args buffers --
    // so geometry visible only in the reflection (behind / beside the main camera,
    // outside its frustum) is captured, not just the main camera's visible set. The
    // reflected view-proj already carries the oblique near-plane clip, so the
    // extracted frustum also rejects geometry behind the reflector. Uses the main
    // single-pass cull kernel (frustum + distance, by the reflected eye) with Hi-Z
    // OFF (the only pyramid is the main camera's screen space, useless here),
    // exactly like the probe capture. Status writes go to a scratch buffer (never
    // read). The whole indirect buffer flips INDIRECT_ARGUMENT -> UAV for the
    // dispatches and back; the per-plane face render then issues each region with
    // `ExecuteIndirect`. Mirrors `encode_shadow_culls`. A no-op when the cull path
    // is inactive or `cull_count() == 0`.
    pub(in crate::directx) fn encode_planar_culls(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        planes: &[(crate::gfx::frustum::Frustum, [f32; 3])],
        indirect: &ID3D12Resource,
        status_gva: u64,
        // Per-plane region stride, in commands: the FIXED build-time record capacity
        // the indirect buffer was sized with (`PlanarReflectionSet::n_cull`) and that
        // the face render's `region_offset` reads with. MUST single-source with the
        // reader -- NOT the live `cull_count()`, which can be smaller than the
        // capacity (e.g. the skinned tail goes inactive when the RT-skin pipeline
        // fails to build), shifting plane >= 1's read offset off the written region.
        region_count: usize,
    ) {
        let (Some(cull_pso), Some(cull_root)) = (
            self.cull.cull_pso.as_ref(),
            self.cull.cull_root_sig.as_ref(),
        ) else {
            return;
        };
        let n_cull = self.cull_count();
        if n_cull == 0 || planes.is_empty() {
            return;
        }
        // The live count never exceeds the capacity the buffer + reader stride by, so
        // the kernel's `n_cull` written commands always land within plane's region.
        debug_assert!(n_cull <= region_count);

        let object_gva =
            unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let draw_args_gva =
            unsafe { self.cull.draw_args_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let base_gva = unsafe { indirect.GetGPUVirtualAddress() };

        // Hi-Z disabled (`hiz_enabled = 0`): the kernel never samples the pyramid,
        // but the descriptor table at root [3] must still point at a live descriptor.
        let (hiz_size, hiz_mip_count, hiz_srv) = match self.cull.hiz.as_ref() {
            Some(h) => (
                [h.width as f32, h.height as f32],
                h.mip_count,
                Some(h.srv_gpu),
            ),
            None => ([1.0, 1.0], 1, None),
        };

        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            cmd.SetComputeRootSignature(cull_root);
            cmd.SetPipelineState(cull_pso);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRootShaderResourceView(1, object_gva);
            cmd.SetComputeRootShaderResourceView(2, draw_args_gva);
            if let Some(srv) = hiz_srv {
                cmd.SetComputeRootDescriptorTable(3, srv);
            }
            cmd.SetComputeRootUnorderedAccessView(5, status_gva);

            for (plane_idx, (frustum, eye)) in planes.iter().enumerate() {
                let mut cull_params = CullParams {
                    planes: [[0.0; 4]; 6],
                    cam_pos: *eye,
                    object_count: n_cull as u32,
                    // Unused with Hi-Z disabled (the reprojection is never taken).
                    prev_view_proj: [[0.0; 4]; 4],
                    hiz_size,
                    hiz_mip_count,
                    hiz_enabled: 0,
                };
                for (i, p) in frustum.planes.iter().enumerate() {
                    cull_params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
                }
                cmd.SetComputeRoot32BitConstants(
                    0,
                    CULL_PARAMS_DWORDS,
                    &cull_params as *const CullParams as *const std::ffi::c_void,
                    0,
                );
                // Plane `plane_idx`'s output region: offset the indirect UAV's GPU
                // address by `plane_idx * region_count` commands so the kernel's
                // `commands[i]` write lands in this plane's slice (strided by the
                // SAME capacity the face render reads with, not the live count).
                let region_gva =
                    base_gva + (plane_idx * region_count * INDIRECT_COMMAND_STRIDE as usize) as u64;
                cmd.SetComputeRootUnorderedAccessView(4, region_gva);
                cmd.Dispatch((n_cull as u32).div_ceil(64), 1, 1);
            }

            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }

    // Dispatch the phase-2 cull compute pass for two-pass occlusion. Runs after
    // the Hi-Z pyramid has been rebuilt mid-frame from phase-1 depth (the
    // `HizBuild` graph node). One thread per build-time object re-tests the
    // objects phase 1 marked `STATUS_HIZ_CANDIDATE` against the fresh pyramid,
    // projecting through *this* frame's un-jittered view-projection (`cur_vp`),
    // and writes the surviving `ExecuteIndirect` commands into this frame's
    // second indirect buffer; `Main2` then issues it. A no-op when the phase-2
    // pipeline / buffers are not built (two-pass off). The phase-2 indirect
    // buffer rests in `INDIRECT_ARGUMENT`; the dispatch flips it to `UAV` and
    // back. Mirrors `metal/cull.rs::encode_cull_phase2`.
    pub(in crate::directx) fn encode_cull_phase2(
        &self,
        cmd: &ID3D12GraphicsCommandList,
        frame_idx: usize,
        frustum: &crate::gfx::frustum::Frustum,
        cur_vp: [[f32; 4]; 4],
    ) {
        let (Some(cull_pso2), Some(cull_root), Some(hiz), Some(indirect)) = (
            self.cull.cull_pso_phase2.as_ref(),
            self.cull.cull_root_sig.as_ref(),
            self.cull.hiz.as_ref(),
            self.cull.indirect_cmd_buffers_2.get(frame_idx),
        ) else {
            return;
        };
        if self.cull_count() == 0 {
            return;
        }

        let object_gva =
            unsafe { self.cull.object_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let draw_args_gva =
            unsafe { self.cull.draw_args_buffer_resources[frame_idx].GetGPUVirtualAddress() };
        let cull_status_gva =
            unsafe { self.cull.cull_status_buffers[frame_idx].GetGPUVirtualAddress() };

        // Project AABBs through this frame's un-jittered VP against the pyramid
        // just rebuilt from this frame's depth. `hiz_enabled = 1`: HizBuild
        // always precedes this dispatch in the graph, so a valid pyramid is
        // guaranteed (the kernel still guards defensively). Frustum planes are
        // unused by the phase-2 kernel (candidates already passed the frustum
        // test in phase 1) but the cbuffer layout is shared, so pack them anyway.
        let mut cull_params = CullParams {
            planes: [[0.0; 4]; 6],
            cam_pos: [0.0; 3],
            object_count: self.cull_count() as u32,
            prev_view_proj: cur_vp,
            hiz_size: [hiz.width as f32, hiz.height as f32],
            hiz_mip_count: hiz.mip_count,
            hiz_enabled: 1,
        };
        for (i, p) in frustum.planes.iter().enumerate() {
            cull_params.planes[i] = [p.normal[0], p.normal[1], p.normal[2], p.d];
        }

        unsafe {
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            // Order phase-1's `cull_status` writes (an earlier per-pass cmd
            // list, so earlier in GPU execution order on the serial queue)
            // before this dispatch's reads. cull_status has no state transition
            // between phases (UAV in both), so the UAV barrier is the only thing
            // that flushes the phase-1 writes.
            cmd.ResourceBarrier(&[uav_barrier(&self.cull.cull_status_buffers[frame_idx])]);
            cmd.SetComputeRootSignature(cull_root);
            cmd.SetPipelineState(cull_pso2);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRoot32BitConstants(
                0,
                CULL_PARAMS_DWORDS,
                &cull_params as *const CullParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetComputeRootShaderResourceView(1, object_gva);
            cmd.SetComputeRootShaderResourceView(2, draw_args_gva);
            // The rebuilt Hi-Z pyramid (same all-mips SRV phase 1 sampled; the
            // HizBuild node rewrote the texels in place).
            cmd.SetComputeRootDescriptorTable(3, hiz.srv_gpu);
            cmd.SetComputeRootUnorderedAccessView(4, indirect.GetGPUVirtualAddress());
            cmd.SetComputeRootUnorderedAccessView(5, cull_status_gva);
            cmd.Dispatch((self.cull_count() as u32).div_ceil(64), 1, 1);
            cmd.ResourceBarrier(&[transition_barrier(
                indirect,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }

    // True when two-pass Hi-Z occlusion runs this frame: the world requested
    // `occlusion_two_pass`, the phase-2 cull PSO + Hi-Z resource are built, and
    // the bindless GPU-cull path is active with build-time geometry. This is the
    // exact condition under which the shared graph inserts the HizBuild / Cull2
    // / Main2 chain, so `seed_inputs`, `encode_main_pass` (resolve skip), and
    // the executor's phase-2 arms all gate on it identically. Mirrors Metal's
    // `CullState.two_pass_occlusion` gate.
    pub(in crate::directx) fn two_pass_occlusion_active(&self) -> bool {
        self.cull.occlusion_two_pass
            && self.cull.cull_pso_phase2.is_some()
            && self.cull.hiz.is_some()
            && self.cull.main_bindless_pso.is_some()
            && self.cull_count() > 0
    }
}

// UAV barrier on a single resource. Used to order phase-1 cull_status writes
// before the phase-2 read (no state transition sits between them). `pResource`
// is borrowed (no AddRef) via `transmute_copy`: it is a `ManuallyDrop`, so a
// `clone()` here would never be released and would leak one reference to the
// resource on every barrier. The caller's `&resource` outlives the
// `ResourceBarrier` call, so the raw pointer stays valid. Mirrors
// `transition_barrier`.
fn uav_barrier(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(resource) },
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CullParams must match the `CullParams` cbuffer (b0) in cull.hlsl: six
    // frustum planes, cam_pos sharing its row with object_count, the previous
    // view-projection, then the Hi-Z metadata (192 B total).
    #[test]
    fn cull_params_layout_matches_hlsl() {
        assert_eq!(std::mem::size_of::<CullParams>(), 192);
        assert_eq!(std::mem::offset_of!(CullParams, planes), 0);
        assert_eq!(std::mem::offset_of!(CullParams, cam_pos), 96);
        assert_eq!(std::mem::offset_of!(CullParams, object_count), 108);
        assert_eq!(std::mem::offset_of!(CullParams, prev_view_proj), 112);
        assert_eq!(std::mem::offset_of!(CullParams, hiz_size), 176);
        assert_eq!(std::mem::offset_of!(CullParams, hiz_mip_count), 184);
        assert_eq!(std::mem::offset_of!(CullParams, hiz_enabled), 188);
    }
}
