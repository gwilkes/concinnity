// src/directx/auto_exposure.rs
//
// Auto-exposure (EV adaptation) on D3D12: a per-frame CPU readback of the
// previous frame's average log-luminance, an EMA step that updates the adapted
// EV, and the histogram build + average compute dispatches that produce next
// frame's average. The compute passes are encoded after the main HDR resolve
// (where `hdr_srv_gpu` carries this frame's scene colour) and read CPU-side at
// the top of a later frame, so there is `FRAMES - 1` frames of latency between
// the scene's actual luminance and the exposure applied to it, invisible at
// human-scale eye-adaptation rates. Mirrors `metal/auto_exposure.rs`.

use windows::Win32::Graphics::Direct3D12::*;

use crate::gfx::auto_exposure::HISTOGRAM_BINS;

use crate::directx::context::{DxContext, FRAMES};
use crate::directx::pipeline::{compile_hlsl, serialize_desc_and_create, shader_source};
use crate::directx::texture::{create_buffer, create_uav_buffer, transition_barrier};

// Build a D3D12 UAV barrier for one buffer resource. `pResource` is borrowed
// (no AddRef) via `transmute_copy`: it is a `ManuallyDrop`, so a `clone()` here
// would never be released and would leak one reference to the resource on every
// barrier. The caller's `&resource` outlives the `ResourceBarrier` call, so the
// raw pointer stays valid. Mirrors `transition_barrier`.
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

// HLSL source for the two compute kernels (build + average). Compiled with
// entry points `build` and `average`.
pub const AUTO_EXPOSURE_HLSL: &str = include_str!("shaders/auto_exposure.hlsl");

// Compile the auto-exposure `build` + `average` compute kernels. Used at
// init and by shader hot-reload to rebuild the two compute PSOs.
pub(in crate::directx) fn compile_auto_exposure_shaders(
    hot_reload: bool,
) -> Result<(Vec<u8>, Vec<u8>), String> {
    let src = shader_source(hot_reload, "auto_exposure.hlsl", AUTO_EXPOSURE_HLSL);
    let build_cs = compile_hlsl(&src, "build", "cs_5_1")?;
    let average_cs = compile_hlsl(&src, "average", "cs_5_1")?;
    Ok((build_cs, average_cs))
}

// DWORD count of the `AutoExposureParams` root-constant block. Must match the
// `cbuffer AutoExposureParams` declaration in `shaders/auto_exposure.hlsl`.
const AUTO_EXPOSURE_PARAMS_DWORDS: u32 = 4;

// Inputs to the auto-exposure compute kernels (root constants at b0).
// Mirrors `metal::uniforms::AutoExposureParams` and the `cbuffer` in the HLSL.
// 16 bytes.
#[derive(Copy, Clone)]
#[repr(C)]
struct AutoExposureParams {
    lum_log2_min: f32,
    lum_log2_range: f32,
    lum_to_bin_scale: f32,
    _pad: f32,
}

// Pair of compute pipelines + GPU buffers + per-frame readback driving the
// auto-exposure histogram path. Built only when the world's
// `PostProcessConfig` opts in; the encoder is a no-op otherwise.
pub(super) struct AutoExposureResources {
    // Build kernel: one thread per HDR-resolve pixel; produces the 256-bin
    // log-luminance histogram in the global UAV.
    build_pso: ID3D12PipelineState,
    build_root_sig: ID3D12RootSignature,
    // Average kernel: one threadgroup of `HISTOGRAM_BINS` threads that reduces
    // the histogram, clears it, and writes the weighted-average log-luminance
    // to the output UAV.
    average_pso: ID3D12PipelineState,
    average_root_sig: ID3D12RootSignature,

    // 256-bin u32 histogram (UNORDERED_ACCESS, DEFAULT heap). The build kernel
    // `InterlockedAdd`s into it; the average kernel reads and clears each bin.
    // Held only to keep the resource alive; the GVA is read directly through
    // `histogram.GetGPUVirtualAddress()` at encode time.
    #[allow(dead_code)]
    histogram: ID3D12Resource,
    // Single f32 (UNORDERED_ACCESS, DEFAULT heap) the average kernel writes
    // the weighted-average log-luminance into.
    output_buf: ID3D12Resource,
    // Per-frame-slot READBACK buffer (4 bytes). Persistently mapped (READBACK
    // resources allow leaving Map active across submissions). The end of each
    // frame's command list copies `output_buf` into the matching slot; at the
    // top of a later frame (after the fence wait gates this slot's previous
    // use) the CPU reads its pointer for the EMA update.
    #[allow(dead_code)]
    readback_bufs: Vec<ID3D12Resource>,
    readback_ptrs: Vec<*const f32>,
}

impl AutoExposureResources {
    // Root signature for the build kernel. Exposed so the DirectX shader
    // hot-reload pass can rebuild the `build_pso` against the same root sig.
    pub(in crate::directx) fn build_root_sig(&self) -> &ID3D12RootSignature {
        &self.build_root_sig
    }
    // Root signature for the average kernel. Same purpose as
    // [`Self::build_root_sig`].
    pub(in crate::directx) fn average_root_sig(&self) -> &ID3D12RootSignature {
        &self.average_root_sig
    }
    // Swap the freshly-built build + average PSOs into the live resources.
    // Driven by the DirectX shader hot-reload pass after every replacement
    // successfully compiled.
    pub(in crate::directx) fn swap_pipelines(
        &mut self,
        build_pso: ID3D12PipelineState,
        average_pso: ID3D12PipelineState,
    ) {
        self.build_pso = build_pso;
        self.average_pso = average_pso;
    }

    // Build all auto-exposure resources. Called from `DxContext::new` only
    // when `PostProcessConfig.auto_exposure` is enabled.
    pub(super) fn new(device: &ID3D12Device, hot_reload: bool) -> Result<Self, String> {
        let (build_cs, average_cs) = compile_auto_exposure_shaders(hot_reload)?;

        let build_root_sig = create_build_root_signature(device)?;
        let build_pso =
            create_compute_pso(device, &build_root_sig, &build_cs, "auto-exposure build")?;

        let average_root_sig = create_average_root_signature(device)?;
        let average_pso = create_compute_pso(
            device,
            &average_root_sig,
            &average_cs,
            "auto-exposure average",
        )?;

        // Histogram: 256 * 4 bytes UAV, cleared by the average kernel each
        // frame. Created in COMMON; a one-shot transition before the first
        // dispatch flips it into UNORDERED_ACCESS for steady state.
        let histogram = create_uav_buffer(
            device,
            (HISTOGRAM_BINS * std::mem::size_of::<u32>()) as u64,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // Output: a single f32 the average kernel writes into.
        let output_buf = create_uav_buffer(
            device,
            std::mem::size_of::<f32>() as u64,
            D3D12_RESOURCE_STATE_COMMON,
        )?;

        // Per-frame readback buffers. READBACK heap resources start in
        // COPY_DEST and never need a barrier.
        let mut readback_bufs: Vec<ID3D12Resource> = Vec::with_capacity(FRAMES);
        let mut readback_ptrs: Vec<*const f32> = Vec::with_capacity(FRAMES);
        for _ in 0..FRAMES {
            let buf = create_buffer(
                device,
                std::mem::size_of::<f32>() as u64,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
            // READBACK heaps allow leaving Map active across submissions; the
            // pointer stays valid for the resource's lifetime.
            unsafe { buf.Map(0, None, Some(&mut ptr)) }
                .map_err(|e| format!("auto-exposure readback map: {e}"))?;
            readback_ptrs.push(ptr as *const f32);
            readback_bufs.push(buf);
        }

        Ok(Self {
            build_pso,
            build_root_sig,
            average_pso,
            average_root_sig,
            histogram,
            output_buf,
            readback_bufs,
            readback_ptrs,
        })
    }
}

// Root signature for the build kernel: 4 root constants (b0, AutoExposureParams),
// a single-SRV descriptor table for the HDR texture (t0), and a root UAV for
// the histogram (u0). The HDR SRV needs a descriptor table because root SRVs
// are limited to raw / structured buffers, not Texture2D.
fn create_build_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let hdr_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0, // t0
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
    };
    let params = [
        // [0] Root constants b0: AutoExposureParams.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: AUTO_EXPOSURE_PARAMS_DWORDS,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [1] Descriptor table SRV t0: HDR texture.
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &hdr_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        // [2] Root UAV u0: histogram.
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
    ];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        ..Default::default()
    };
    serialize_desc_and_create(device, &desc, "auto-exposure build root sig")
}

// Root signature for the average kernel: 4 root constants (b0), root UAV for
// the histogram (u0, read + clear), root UAV for the output (u1, write-once).
fn create_average_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature, String> {
    let params = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: AUTO_EXPOSURE_PARAMS_DWORDS,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
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
    serialize_desc_and_create(device, &desc, "auto-exposure average root sig")
}

// Compute pipeline state for one of the auto-exposure kernels. Exposed to
// the DirectX shader hot-reload pass so it can rebuild both PSOs against the
// existing root signatures.
pub(in crate::directx) fn create_compute_pso(
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

impl DxContext {
    // Build the per-frame compute params. The log-luminance range and the
    // precomputed `bins / range` scale match the `gfx::auto_exposure::LUM_LOG2_*`
    // constants exactly.
    fn auto_exposure_params(&self) -> AutoExposureParams {
        use crate::gfx::auto_exposure::{LUM_LOG2_MAX, LUM_LOG2_MIN};
        let range = LUM_LOG2_MAX - LUM_LOG2_MIN;
        AutoExposureParams {
            lum_log2_min: LUM_LOG2_MIN,
            lum_log2_range: range,
            lum_to_bin_scale: HISTOGRAM_BINS as f32 / range,
            _pad: 0.0,
        }
    }

    // Step the auto-exposure EMA from a previous frame's GPU measurement, then
    // push the new exposure multiplier into `self.post_process.exposure`.
    // A no-op when auto-exposure is disabled: the static authored EV then
    // drives `exposure` unchanged.
    //
    // Called at the top of `draw_frame` after the fence wait for this slot's
    // previous use completes, so the matching readback buffer holds a fully
    // committed GPU result (one or two frames stale, smoothed by the EMA).
    // `elapsed` is the total elapsed seconds since startup; the per-call diff
    // drives `dt` for the EMA.
    pub(super) fn update_auto_exposure(&mut self, elapsed: f32, frame_idx: usize) {
        let Some(settings) = self.auto_exposure.settings else {
            return;
        };
        let Some(resources) = self.auto_exposure.resources.as_ref() else {
            return;
        };
        let Some(state) = self.auto_exposure.state.as_mut() else {
            return;
        };
        let Some(&ptr) = resources.readback_ptrs.get(frame_idx) else {
            return;
        };

        // Read the previous frame's average log-luminance for this slot. The
        // fence wait above this call already gated the GPU work that wrote it,
        // so the READBACK-heap mapping reflects the committed value.
        let avg_log_lum = unsafe { ptr.read() };
        let avg_log_lum = if avg_log_lum.is_finite() {
            avg_log_lum
        } else {
            crate::gfx::auto_exposure::LUM_LOG2_MIN
        };

        let dt = (elapsed - self.auto_exposure.last_elapsed).max(0.0);
        self.auto_exposure.last_elapsed = elapsed;

        let adapted_ev = state.update(avg_log_lum, self.auto_exposure.bias_ev, &settings, dt);
        // `self.post_process.exposure` is the linear multiplier the bloom
        // prefilter and composite consume; it already folds in the authored
        // exposure_ev when auto-exposure is off, so we only overwrite it here
        // when the GPU path owns the value. `state.update` already folds the
        // bias into the target; re-adding it would double the bias.
        self.post_process.exposure = adapted_ev.exp2();
    }

    // Resource the histogram_build kernel samples through `hdr_srv_gpu`: the
    // resolved single-sample HDR scene with MSAA on, otherwise the raw
    // (single-sample) `hdr_color`. The auto-exposure measurement runs between
    // the main HDR resolve and any post pass that mutates the scene (decals,
    // fog, SSR, TAA, bloom, composite), so it samples the pre-post scene.
    fn auto_exposure_source(&self) -> &ID3D12Resource {
        self.hdr.resolve.as_ref().unwrap_or(&self.hdr.color)
    }

    // Encode the auto-exposure histogram passes against the resolved HDR
    // scene. The build kernel runs one thread per HDR pixel; the average
    // kernel runs one threadgroup of 256 threads that reduces the histogram,
    // clears it for the next frame, and writes the average log-luminance to
    // the output UAV. The end of the encoder copies the output to this slot's
    // readback buffer for the CPU's EMA step at the top of a later frame.
    // A no-op when auto-exposure is disabled.
    pub(super) fn encode_auto_exposure(&self, cmd: &ID3D12GraphicsCommandList, frame_idx: usize) {
        let Some(resources) = self.auto_exposure.resources.as_ref() else {
            return;
        };

        let params = self.auto_exposure_params();
        let source = self.auto_exposure_source();

        // The build kernel needs the HDR scene readable in a compute (i.e.
        // NON_PIXEL_SHADER_RESOURCE) stage. After encode_main_pass the
        // resolved scene rests in PIXEL_SHADER_RESOURCE; flip it to the
        // compute-readable state for the dispatch and back so the downstream
        // decal / fog / SSR / TAA / bloom / composite passes find it where
        // they expect.
        let to_compute = transition_barrier(
            source,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[to_compute]) };

        let histogram_gva = unsafe { resources.histogram.GetGPUVirtualAddress() };
        let output_gva = unsafe { resources.output_buf.GetGPUVirtualAddress() };

        // Build dispatch: 16×16 threadgroups, one thread per HDR pixel.
        unsafe {
            cmd.SetComputeRootSignature(&resources.build_root_sig);
            cmd.SetPipelineState(&resources.build_pso);
            cmd.SetDescriptorHeaps(&[Some(self.descriptors.srv_heap.clone())]);
            cmd.SetComputeRoot32BitConstants(
                0,
                AUTO_EXPOSURE_PARAMS_DWORDS,
                &params as *const AutoExposureParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetComputeRootDescriptorTable(1, self.hdr.srv_gpu);
            cmd.SetComputeRootUnorderedAccessView(2, histogram_gva);

            let groups_x = self.render_width.div_ceil(16);
            let groups_y = self.render_height.div_ceil(16);
            cmd.Dispatch(groups_x, groups_y, 1);
        }

        // UAV barrier so the average dispatch sees the build kernel's writes.
        let barrier = uav_barrier(&resources.histogram);
        unsafe { cmd.ResourceBarrier(&[barrier]) };

        // Average dispatch: one threadgroup of 256 threads.
        unsafe {
            cmd.SetComputeRootSignature(&resources.average_root_sig);
            cmd.SetPipelineState(&resources.average_pso);
            cmd.SetComputeRoot32BitConstants(
                0,
                AUTO_EXPOSURE_PARAMS_DWORDS,
                &params as *const AutoExposureParams as *const std::ffi::c_void,
                0,
            );
            cmd.SetComputeRootUnorderedAccessView(1, histogram_gva);
            cmd.SetComputeRootUnorderedAccessView(2, output_gva);
            cmd.Dispatch(1, 1, 1);
        }

        // UAV barrier so the readback copy sees the average kernel's write.
        let barrier = uav_barrier(&resources.output_buf);
        unsafe { cmd.ResourceBarrier(&[barrier]) };

        // Copy the freshly-written average to this slot's readback buffer. A
        // later frame using the same slot reads it from the matching
        // `readback_ptrs[frame_idx]` after the fence wait gates the copy.
        let to_copy_src = transition_barrier(
            &resources.output_buf,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[to_copy_src]) };
        if let Some(readback) = resources.readback_bufs.get(frame_idx) {
            unsafe {
                cmd.CopyBufferRegion(
                    readback,
                    0,
                    &resources.output_buf,
                    0,
                    std::mem::size_of::<f32>() as u64,
                );
            }
        }
        let to_uav = transition_barrier(
            &resources.output_buf,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        unsafe { cmd.ResourceBarrier(&[to_uav]) };

        // Restore the HDR source to PIXEL_SHADER_RESOURCE for the post stack.
        let back_to_psr = transition_barrier(
            source,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[back_to_psr]) };
    }
}
