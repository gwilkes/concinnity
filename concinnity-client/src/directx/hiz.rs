// src/directx/hiz.rs
//
// Hi-Z (depth-mip pyramid) build pass used by the GPU-cull compute kernel for
// occlusion culling. Each frame, after the main depth buffer has been written
// by the graph, we copy/reduce it into a Texture2D mip chain (R32_FLOAT, MAX
// reduction). The *next* frame's `Cull` pass projects each `DrawObject` AABB
// through the previous frame's view-projection, picks the Hi-Z mip whose
// texels are roughly the size of the projected rect, and culls the AABB when
// its nearest projected depth is behind the rasterised occluder depth.
//
// Three compute kernels share one root signature (see `shaders/hiz_build.hlsl`):
//
//   * `init_single`: copy a single-sample main depth resource into HiZ mip 0.
//   * `init_msaa`  : reduce an MSAA main depth resource into HiZ mip 0,
//                    taking the MAX over every sample so the result is
//                    conservative.
//   * `downsample` : MAX-reduce 2x2 source texels into the next mip.
//
// The pyramid is *not* a graph node; it runs inline on the outer "end" cmd
// list after `execute_graph` returns (see `directx/draw/mod.rs`). Treating
// it as an end-of-frame action keeps it off the graph's RMW chain on the
// main depth attachment (decals, fog, and SSAO/SSR pre-passes already share
// that target).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::directx::context::dump_on_err;
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::transition_barrier;

pub const HIZ_BUILD_HLSL: &str = include_str!("shaders/hiz_build.hlsl");

// DWORD count of the `HizParams` cbuffer (dst_w, dst_h, src_mip, sample_count).
const HIZ_PARAMS_DWORDS: u32 = 4;

#[derive(Copy, Clone)]
#[repr(C)]
struct HizParams {
    dst_w: u32,
    dst_h: u32,
    src_mip: u32,
    sample_count: u32,
}

// Compute pipelines + texture + per-mip descriptors for the Hi-Z build. Built
// alongside the GPU-cull pipeline (same gating condition: bindless main pass
// active with build-time static geometry).
pub(super) struct HiZResources {
    pub(super) root_sig: ID3D12RootSignature,
    pub(super) init_single_pso: ID3D12PipelineState,
    pub(super) init_msaa_pso: ID3D12PipelineState,
    pub(super) downsample_pso: ID3D12PipelineState,

    // R32_FLOAT 2D texture with a full mip chain. UAV-writable; the cull
    // kernel reads it via `Texture2D<float>.Load(int3(x, y, mip))`. Held
    // only to keep the resource alive; the per-mip CPU UAV handles below
    // reference it.
    #[allow(dead_code)]
    pub(super) texture: ID3D12Resource,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) mip_count: u32,
    // Resting resource state between Hi-Z build pulses. The encoder
    // transitions to `UNORDERED_ACCESS` for the duration of the build and
    // back to this state afterwards so the next frame's cull dispatch can
    // sample it.
    pub(super) rest_state: D3D12_RESOURCE_STATES,

    // CPU descriptor handle for the SRV covering the whole mip chain. Used
    // by the `downsample` kernel (which Loads from the prior mip) and by
    // the GPU-cull kernel (4 corner Loads at a picked mip).
    pub(super) srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(super) srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // CPU descriptor handle of the depth-source SRV the init kernel binds at
    // t0 (the main-depth SRV the decal/fog passes also share). The CPU
    // handle is captured at init so a future resize can rewrite it in
    // place without rebinding the cull root signature.
    #[allow(dead_code)]
    pub(super) depth_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pub(super) depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    // Per-mip UAV CPU/GPU descriptor pairs. Length = `mip_count`.
    pub(super) mip_uav_cpus: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
    pub(super) mip_uav_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
}

// Compile every Hi-Z compute kernel against the same root signature.
// Compiled Hi-Z kernels: init_single, init_msaa, downsample bytecode.
type HizShaders = (Vec<u8>, Vec<u8>, Vec<u8>);

pub(in crate::directx) fn compile_hiz_shaders(hot_reload: bool) -> Result<HizShaders, String> {
    let src = shader_source(hot_reload, "hiz_build.hlsl", HIZ_BUILD_HLSL);
    let init_single = compile_hlsl(&src, "init_single", "cs_5_1")?;
    let init_msaa = compile_hlsl(&src, "init_msaa", "cs_5_1")?;
    let downsample = compile_hlsl(&src, "downsample", "cs_5_1")?;
    Ok((init_single, init_msaa, downsample))
}

// Root signature: 4 root constants (HizParams at b0), one SRV descriptor
// table (depth source for init / HiZ for downsample), one UAV descriptor
// table (destination mip).
pub(in crate::directx) fn create_hiz_root_signature(
    device: &ID3D12Device,
) -> Result<ID3D12RootSignature, String> {
    let srv_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let uav_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // u0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: HIZ_PARAMS_DWORDS,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &srv_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &uav_range,
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
    serialize_desc_and_create(device, &desc, "hiz root sig")
}

fn create_hiz_pso(
    device: &ID3D12Device,
    root_sig: &ID3D12RootSignature,
    cs: &[u8],
    label: &str,
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
    unsafe { device.CreateComputePipelineState(&desc) }
        .map_err(|e| format!("create {label} PSO: {e}"))
}

// Mip count for a Hi-Z of size (w, h): `floor(log2(max(w, h))) + 1`. Power-
// of-two sources end exactly at 1x1; non-power-of-two sources stop one mip
// short of true 1x1 in the smaller dimension, which is fine; the cull
// kernel clamps to the actual mip dims.
pub(super) fn hiz_mip_count(width: u32, height: u32) -> u32 {
    let m = width.max(height).max(1);
    32 - m.leading_zeros()
}

// Create the Hi-Z texture (R32_FLOAT, full mip chain, UAV + SRV capable)
// plus the resource. Resting state is `NON_PIXEL_SHADER_RESOURCE` so the
// next-frame cull dispatch can sample it without a transition.
fn create_hiz_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    mip_count: u32,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: mip_count as u16,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut tex: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            None,
            &mut tex,
        )
    }
    .map_err(|e| format!("create hiz texture: {e}"))?;
    tex.ok_or_else(|| "create hiz texture returned None".to_string())
}

// Write the all-mips SRV that the cull kernel and the downsample kernel
// share. `MipLevels: u32::MAX` means "every mip".
pub(in crate::directx) fn write_hiz_srv(
    device: &ID3D12Device,
    tex: &ID3D12Resource,
    mip_count: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: mip_count,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(tex, Some(&desc), srv_cpu) };
}

// Write a UAV pointing at a single mip slice.
pub(in crate::directx) fn write_hiz_mip_uav(
    device: &ID3D12Device,
    tex: &ID3D12Resource,
    mip: u32,
    uav_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV {
                MipSlice: mip,
                PlaneSlice: 0,
            },
        },
    };
    unsafe { device.CreateUnorderedAccessView(tex, None, Some(&desc), uav_cpu) };
}

impl HiZResources {
    // Build the Hi-Z resource + every PSO. Called from the init path when
    // the bindless static pass + cull pipeline are active. Each of the
    // supplied descriptor handles points at a pre-reserved slot in the
    // SRV heap; the resource owns the descriptors but not the heap.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        device: &ID3D12Device,
        info_queue: Option<&ID3D12InfoQueue>,
        width: u32,
        height: u32,
        srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        depth_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
        mip_uav_cpus: Vec<D3D12_CPU_DESCRIPTOR_HANDLE>,
        mip_uav_gpus: Vec<D3D12_GPU_DESCRIPTOR_HANDLE>,
        hot_reload: bool,
    ) -> Result<Self, String> {
        let mip_count = hiz_mip_count(width, height).min(mip_uav_cpus.len() as u32);
        if mip_count == 0 {
            return Err("hiz: zero mip count".into());
        }
        let (init_single_cs, init_msaa_cs, downsample_cs) = compile_hiz_shaders(hot_reload)?;
        let root_sig = dump_on_err(info_queue, create_hiz_root_signature(device))?;
        let init_single_pso = dump_on_err(
            info_queue,
            create_hiz_pso(device, &root_sig, &init_single_cs, "hiz init_single"),
        )?;
        let init_msaa_pso = dump_on_err(
            info_queue,
            create_hiz_pso(device, &root_sig, &init_msaa_cs, "hiz init_msaa"),
        )?;
        let downsample_pso = dump_on_err(
            info_queue,
            create_hiz_pso(device, &root_sig, &downsample_cs, "hiz downsample"),
        )?;

        let texture = create_hiz_texture(device, width, height, mip_count)?;
        write_hiz_srv(device, &texture, mip_count, srv_cpu);
        for (mip, &cpu) in mip_uav_cpus.iter().take(mip_count as usize).enumerate() {
            write_hiz_mip_uav(device, &texture, mip as u32, cpu);
        }
        Ok(Self {
            root_sig,
            init_single_pso,
            init_msaa_pso,
            downsample_pso,
            texture,
            width,
            height,
            mip_count,
            rest_state: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            srv_cpu,
            srv_gpu,
            depth_srv_cpu,
            depth_srv_gpu,
            mip_uav_cpus,
            mip_uav_gpus,
        })
    }

    // Recreate the texture at new render-target dimensions. Re-uses the
    // existing descriptor heap slots; the live cull-kernel binding stays
    // valid because the GPU descriptor handles point at the same slots.
    pub(super) fn resize_to(
        &mut self,
        device: &ID3D12Device,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let new_mip_count = hiz_mip_count(width, height).min(self.mip_uav_cpus.len() as u32);
        let texture = create_hiz_texture(device, width, height, new_mip_count)?;
        write_hiz_srv(device, &texture, new_mip_count, self.srv_cpu);
        for (mip, &cpu) in self
            .mip_uav_cpus
            .iter()
            .take(new_mip_count as usize)
            .enumerate()
        {
            write_hiz_mip_uav(device, &texture, mip as u32, cpu);
        }
        self.texture = texture;
        self.width = width;
        self.height = height;
        self.mip_count = new_mip_count;
        self.rest_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        Ok(())
    }

    // Swap the freshly-rebuilt PSOs into the live resources. Used by the
    // shader hot-reload pass.
    pub(super) fn swap_pipelines(
        &mut self,
        init_single_pso: ID3D12PipelineState,
        init_msaa_pso: ID3D12PipelineState,
        downsample_pso: ID3D12PipelineState,
    ) {
        self.init_single_pso = init_single_pso;
        self.init_msaa_pso = init_msaa_pso;
        self.downsample_pso = downsample_pso;
    }
}

// UAV barrier helper, mirrors `auto_exposure::uav_barrier`. `pResource` is
// borrowed (no AddRef) via `transmute_copy`: it is a `ManuallyDrop`, so a
// `clone()` here would never be released and would leak one reference to the
// resource on every barrier. The caller's `&resource` outlives the
// `ResourceBarrier` call, so the raw pointer stays valid.
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

impl crate::directx::context::DxContext {
    // Encode the Hi-Z build pass on `cmd`. Runs after the render graph
    // returns, before the per-frame restore barriers. Assumes the main
    // depth resource is in `DEPTH_WRITE` state (every depth-reading pass
    // in the graph restores it back to DEPTH_WRITE after sampling). A
    // no-op when bindless cull isn't active (no Hi-Z resource was built).
    pub(in crate::directx) fn encode_hiz_build(&self, cmd: &ID3D12GraphicsCommandList) {
        let Some(hiz) = self.cull.hiz.as_ref() else {
            return;
        };
        // 1. Transition main depth so a compute SRV can sample it, and
        //    transition the Hi-Z texture so the compute UAV can write it.
        let depth_to_srv = transition_barrier(
            &self.depth_resource,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        let hiz_to_uav = transition_barrier(
            &hiz.texture,
            hiz.rest_state,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        unsafe { cmd.ResourceBarrier(&[depth_to_srv, hiz_to_uav]) };

        // 2. Init kernel: mip 0 from main depth (MAX over MSAA samples when
        //    MSAA is on).
        let msaa = self.hdr.msaa_samples > 1;
        let init_params = HizParams {
            dst_w: hiz.width,
            dst_h: hiz.height,
            src_mip: 0,
            sample_count: self.hdr.msaa_samples.max(1),
        };
        unsafe {
            cmd.SetComputeRootSignature(&hiz.root_sig);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetPipelineState(if msaa {
                &hiz.init_msaa_pso
            } else {
                &hiz.init_single_pso
            });
            cmd.SetComputeRoot32BitConstants(
                0,
                HIZ_PARAMS_DWORDS,
                &init_params as *const HizParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetComputeRootDescriptorTable(1, hiz.depth_srv_gpu);
            cmd.SetComputeRootDescriptorTable(2, hiz.mip_uav_gpus[0]);
            cmd.Dispatch(hiz.width.div_ceil(8), hiz.height.div_ceil(8), 1);
        }
        unsafe { cmd.ResourceBarrier(&[uav_barrier(&hiz.texture)]) };

        // 3. Downsample chain. Each dispatch reads the prior mip via the
        //    all-mips SRV and writes the next mip via its UAV.
        let mut cur_w = hiz.width;
        let mut cur_h = hiz.height;
        for mip in 1..hiz.mip_count {
            let next_w = (cur_w / 2).max(1);
            let next_h = (cur_h / 2).max(1);
            let params = HizParams {
                dst_w: next_w,
                dst_h: next_h,
                src_mip: mip - 1,
                sample_count: 0,
            };
            unsafe {
                cmd.SetPipelineState(&hiz.downsample_pso);
                cmd.SetComputeRoot32BitConstants(
                    0,
                    HIZ_PARAMS_DWORDS,
                    &params as *const HizParams as *const std::ffi::c_void,
                    0,
                );
                cmd.SetComputeRootDescriptorTable(1, hiz.srv_gpu);
                cmd.SetComputeRootDescriptorTable(2, hiz.mip_uav_gpus[mip as usize]);
                cmd.Dispatch(next_w.div_ceil(8), next_h.div_ceil(8), 1);
                cmd.ResourceBarrier(&[uav_barrier(&hiz.texture)]);
            }
            cur_w = next_w;
            cur_h = next_h;
        }

        // 4. Restore: main depth back to DEPTH_WRITE for next frame's main
        //    pass; Hi-Z back to NON_PIXEL_SHADER_RESOURCE so the next frame's
        //    cull dispatch finds it where the cull kernel expects.
        let depth_back = transition_barrier(
            &self.depth_resource,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
        );
        let hiz_back = transition_barrier(
            &hiz.texture,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            hiz.rest_state,
        );
        unsafe { cmd.ResourceBarrier(&[depth_back, hiz_back]) };
    }
}
