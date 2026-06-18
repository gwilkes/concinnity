// D3D12 resource creation helpers.
// All texture uploads use an upload heap (CPU-visible) that is copied to a
// default heap (GPU-local) via CopyTextureRegion on a one-shot command list.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

// GPU resource handle

// Opaque handle to a committed D3D12 resource (buffer or texture).
#[allow(dead_code)]
pub(super) struct GpuResource {
    pub resource: ID3D12Resource,
    // CPU descriptor handle for the SRV (zero/invalid for buffers that don't need one).
    pub srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    // GPU descriptor handle for the SRV.
    pub srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
}

// One-shot command list helper

// Execute f on a freshly allocated command list, submit, and wait for idle.
pub(super) fn one_shot_submit<F>(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    f: F,
) -> Result<(), String>
where
    F: FnOnce(&ID3D12GraphicsCommandList),
{
    let allocator: ID3D12CommandAllocator =
        unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
            .map_err(|e| format!("one_shot allocator: {e}"))?;

    let cmd: ID3D12GraphicsCommandList =
        unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None) }
            .map_err(|e| format!("one_shot cmd list: {e}"))?;

    f(&cmd);

    unsafe { cmd.Close() }.map_err(|e| format!("one_shot close: {e}"))?;

    let cmd_list: ID3D12CommandList = cmd.cast().map_err(|e| format!("one_shot cast: {e}"))?;
    unsafe { queue.ExecuteCommandLists(&[Some(cmd_list)]) };

    // Fence-wait for completion.
    let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
        .map_err(|e| format!("one_shot fence: {e}"))?;
    let event =
        unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None) }
            .map_err(|e| format!("one_shot event: {e}"))?;
    unsafe { queue.Signal(&fence, 1) }.map_err(|e| format!("one_shot signal: {e}"))?;
    if unsafe { fence.GetCompletedValue() } < 1 {
        unsafe { fence.SetEventOnCompletion(1, event) }
            .map_err(|e| format!("one_shot set event: {e}"))?;
        unsafe { windows::Win32::System::Threading::WaitForSingleObject(event, u32::MAX) };
    }
    unsafe { windows::Win32::Foundation::CloseHandle(event) }.ok();
    Ok(())
}

// Buffer helpers

// Create a committed buffer in the given heap type.
pub(super) fn create_buffer(
    device: &ID3D12Device,
    size: u64,
    heap_type: D3D12_HEAP_TYPE,
    initial_state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initial_state,
            None,
            &mut resource,
        )
    }
    .map_err(|e| format!("create_buffer: {e}"))?;
    resource.ok_or_else(|| "create_buffer returned None".to_string())
}

// Create a default-heap buffer with `ALLOW_UNORDERED_ACCESS`, suitable for a
// compute shader to write through a UAV. Used by the compute-cull
// pass for the per-frame indirect-command buffers.
pub(super) fn create_uav_buffer(
    device: &ID3D12Device,
    size: u64,
    initial_state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            initial_state,
            None,
            &mut resource,
        )
    }
    .map_err(|e| format!("create_uav_buffer: {e}"))?;
    resource.ok_or_else(|| "create_uav_buffer returned None".to_string())
}

// Upload raw bytes to a GPU-local buffer via a temporary upload heap.
// Returns the device-local buffer.
pub(super) fn upload_buffer(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    data: &[u8],
    usage_state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let size = data.len().max(4) as u64;

    let upload = create_buffer(
        device,
        size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Map and copy.
    let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { upload.Map(0, None, Some(&mut ptr)) }.map_err(|e| format!("upload map: {e}"))?;
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
        upload.Unmap(0, None);
    }

    // Buffers are always created in COMMON regardless of requested state, so
    // pass COMMON explicitly to avoid the debug layer warning.
    let dest = create_buffer(
        device,
        size,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_STATE_COMMON,
    )?;

    one_shot_submit(device, queue, |cmd| unsafe {
        cmd.CopyBufferRegion(&dest, 0, &upload, 0, size);
        // CopyBufferRegion implicitly promotes the buffer COMMON -> COPY_DEST,
        // so the transition barrier's before-state must be COPY_DEST.
        let barrier = transition_barrier(&dest, D3D12_RESOURCE_STATE_COPY_DEST, usage_state);
        cmd.ResourceBarrier(&[barrier]);
    })?;

    Ok(dest)
}

// Texture helpers

// Upload RGBA pixel data to a GPU-local RGBA8_UNORM texture.
// Returns just the resource; call `write_rgba8_srv` to bind it into a slot.
// Multiple SRVs may reference the same resource (one per object using it).
pub(super) fn upload_texture_resource(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> Result<ID3D12Resource, String> {
    let base = (width as usize) * (height as usize) * 4;
    if pixels.len() < base {
        return Err(format!(
            "pixel data too short for {}x{} texture ({} bytes, need {})",
            width,
            height,
            pixels.len(),
            base
        ));
    }

    // Box-filtered mip chain so the texture minifies through hardware trilinear /
    // aniso selection instead of aliasing from a single mip-0 sample.
    let chain = crate::gfx::mipmap::generate_mip_chain(width, height, pixels);
    let mip_count = chain.len() as u32;

    // Texture resource (default heap, copy-dest initially), full mip chain.
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
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create texture: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create texture returned None".to_string())?;

    // Footprints for every subresource (one per mip).
    let mut layouts = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); mip_count as usize];
    let mut row_counts = vec![0u32; mip_count as usize];
    let mut row_sizes = vec![0u64; mip_count as usize];
    let mut total_size: u64 = 0;
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            mip_count,
            0,
            Some(layouts.as_mut_ptr()),
            Some(row_counts.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total_size),
        );
    }

    // Upload heap holding every mip packed at its footprint offset.
    let upload = create_buffer(
        device,
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("upload tex map: {e}"))?;
    for (m, level) in chain.iter().enumerate() {
        let src_row = (level.width * 4) as usize;
        let dst_pitch = layouts[m].Footprint.RowPitch as usize;
        let base_off = layouts[m].Offset as usize;
        for row in 0..level.height as usize {
            let src = &level.pixels[row * src_row..(row + 1) * src_row];
            let dst = unsafe { (map_ptr as *mut u8).add(base_off + row * dst_pitch) };
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
        }
    }
    unsafe { upload.Unmap(0, None) };

    // Copy each mip subresource, then transition all subresources to shader-read.
    // The copy-location structs are created once and reused across the loop (only
    // the footprint / subresource index changes). `pResource` borrows the upload /
    // texture pointer without an AddRef: the field is a `ManuallyDrop`, so a
    // `clone()` would never be released and would leak a reference to the transient
    // upload buffer (a real memory leak) and the destination texture (a VRAM leak
    // under streaming eviction) on every upload. Both outlive the synchronous
    // `CopyTextureRegion` calls.
    one_shot_submit(device, queue, |cmd| {
        let mut src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[0],
            },
        };
        let mut dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        for m in 0..mip_count {
            src.Anonymous = D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layouts[m as usize],
            };
            dst.Anonymous = D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: m,
            };
            unsafe { cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        }
        let barrier = transition_barrier(
            &texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[barrier]) };
    })?;

    Ok(texture)
}

// Write an RGBA8_UNORM Texture2D SRV at the given heap slot, exposing the
// resource's full mip chain so minified samples trilinear-select down it.
pub(super) fn write_rgba8_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let mip_levels = unsafe { resource.GetDesc() }.MipLevels as u32;
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: mip_levels,
                ..Default::default()
            },
        },
    };
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Upload RGBA pixel data to a GPU-local RGBA8_UNORM texture and write its SRV
// at the given heap slot. Used for resources that bind to a single slot
// (text atlases, etc.). Per-object scene textures use `upload_texture_resource`
// + `write_rgba8_srv` directly so one resource can feed multiple per-object slots.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_texture(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    width: u32,
    height: u32,
    pixels: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let texture = upload_texture_resource(device, queue, width, height, pixels)?;
    write_rgba8_srv(device, &texture, srv_cpu);
    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Create a 1×1 opaque white RGBA texture (no SRV write; caller binds it).
pub(super) fn create_fallback_white_resource(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
) -> Result<ID3D12Resource, String> {
    upload_texture_resource(device, queue, 1, 1, &[255u8, 255, 255, 255])
}

// Create a 1×1 flat-normal RGBA texture (tangent-space no-op 128,128,255,255).
pub(super) fn create_fallback_flat_normal_resource(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
) -> Result<ID3D12Resource, String> {
    upload_texture_resource(device, queue, 1, 1, &[128u8, 128, 255, 255])
}

// Create a 1×1×1 R32_FLOAT Texture2DArray fallback for when no shadow pass is
// configured. Value 0.0 ensures SampleCmpLevelZero (LESS_EQUAL) always passes,
// returning 1.0 (fully lit). R32_FLOAT is required for comparison sampling.
// The SRV is declared as Texture2DArray (ArraySize=1) so the fragment shader's
// binding type stays identical between the disabled and CSM-enabled cases.
pub(super) fn create_fallback_shadow_array(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create fallback shadow array: {e}"))?;
    let texture =
        tex_opt.ok_or_else(|| "create fallback shadow array returned None".to_string())?;

    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    unsafe {
        device.GetCopyableFootprints(&desc, 0, 1, 0, Some(&mut layout), None, None, None);
    }

    let upload = create_buffer(
        device,
        layout.Footprint.RowPitch as u64,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("map fallback shadow array: {e}"))?;
    unsafe {
        *(map_ptr as *mut f32) = 0.0f32;
        upload.Unmap(0, None);
    }

    // `pResource` borrows the upload / texture pointer without an AddRef: the
    // field is a `ManuallyDrop`, so a `clone()` would never be released and would
    // leak a reference to the transient upload buffer and the destination texture
    // on every upload. Both outlive the synchronous `CopyTextureRegion` call.
    one_shot_submit(device, queue, |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        unsafe {
            cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            let barrier = transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            cmd.ResourceBarrier(&[barrier]);
        }
    })?;

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                FirstArraySlice: 0,
                ArraySize: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(&texture, Some(&srv_desc), srv_cpu) };

    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Create the main pass depth buffer. DEPTH_STENCIL only, no SRV.
// `shader_readable` drops the `DENY_SHADER_RESOURCE` flag so the resource
// can also be bound as a `Texture2D[MS]<float>` SRV, needed by the
// projected-decal pass, which samples scene depth to reconstruct world
// positions. The HiZ cost is acceptable for the cases that opt in; the
// SSR / SSAO pre-pass depth buffers leave the flag set (depth-only).
pub(super) fn create_main_depth_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    dsv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    sample_count: u32,
    shader_readable: bool,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let mut flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    if !shader_readable {
        flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create main depth texture: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create main depth texture returned None".to_string())?;

    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: DXGI_FORMAT_D32_FLOAT,
        ViewDimension: if sample_count > 1 {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        },
        Flags: D3D12_DSV_FLAG_NONE,
        Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
        },
    };
    unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };

    Ok(texture)
}

// Create a `layers`-slice Texture2DArray shadow map. Returns the resource
// (initial state DEPTH_WRITE), one DSV cpu handle per slice (written at
// `dsv_cpu_base + i * dsv_stride`), and an SRV pointing at the whole array
// suitable for sampling as a `Texture2DArray<float>` with SampleCmpLevelZero.
pub(super) fn create_shadow_map_array(
    device: &ID3D12Device,
    size: u32,
    layers: u32,
    dsv_cpu_base: D3D12_CPU_DESCRIPTOR_HANDLE,
    dsv_stride: usize,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<(GpuResource, Vec<D3D12_CPU_DESCRIPTOR_HANDLE>), String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: DXGI_FORMAT_D32_FLOAT,
        Anonymous: D3D12_CLEAR_VALUE_0 {
            DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                Depth: 1.0,
                Stencil: 0,
            },
        },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: layers as u16,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            // Rest in the sampled state. The graph's Shadow producer barrier
            // transitions this to DEPTH_WRITE before each shadow pass and Main's
            // consumer returns it here, so the cross-frame reset is the graph's
            // producer barrier, not an inline end-of-frame transition. Creating
            // it sampled makes frame 0's producer barrier (sampled -> DEPTH_WRITE)
            // start from the resource's real state.
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&clear_value),
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create shadow map array: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create shadow map array returned None".to_string())?;

    let mut dsvs = Vec::with_capacity(layers as usize);
    for i in 0..layers {
        let dsv_cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: dsv_cpu_base.ptr + (i as usize) * dsv_stride,
        };
        let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: DXGI_FORMAT_D32_FLOAT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
            Flags: D3D12_DSV_FLAG_NONE,
            Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                    MipSlice: 0,
                    FirstArraySlice: i,
                    ArraySize: 1,
                },
            },
        };
        unsafe { device.CreateDepthStencilView(&texture, Some(&dsv_desc), dsv_cpu) };
        dsvs.push(dsv_cpu);
    }

    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                FirstArraySlice: 0,
                ArraySize: layers,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(&texture, Some(&srv_desc), srv_cpu) };

    Ok((
        GpuResource {
            resource: texture,
            srv_cpu,
            srv_gpu,
        },
        dsvs,
    ))
}

// Off-screen HDR colour format. The main + instanced passes render
// linear-light HDR into a target of this format; the composite pass tonemaps
// it down to the swapchain backbuffer.
pub(super) const HDR_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

// Create the off-screen HDR colour render target the main pass draws into.
// `sample_count` matches the depth buffer's MSAA; with MSAA off this target
// is single-sample and the composite pass samples it directly. Created in the
// RENDER_TARGET state, with an RTV written at `rtv_cpu`.
pub(super) fn create_hdr_color_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    sample_count: u32,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    clear_color: [f32; 4],
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: HDR_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear_color },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: HDR_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some(&clear_value),
            &mut res_opt,
        )
    }
    .map_err(|e| format!("create hdr color target: {e}"))?;
    let res = res_opt.ok_or_else(|| "create hdr color returned None".to_string())?;

    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: if sample_count > 1 {
            D3D12_RTV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_RTV_DIMENSION_TEXTURE2D
        },
        ..Default::default()
    };
    unsafe { device.CreateRenderTargetView(&res, Some(&rtv_desc), rtv_cpu) };

    Ok(res)
}

// Create the single-sample HDR resolve target. The MSAA `create_hdr_color_target`
// resolves into this each frame; the composite pass then samples it. Created in
// the PIXEL_SHADER_RESOURCE state (the per-frame cycle flips it to RESOLVE_DEST
// and back). Only needed when MSAA is on.
pub(super) fn create_hdr_resolve_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    // `ALLOW_RENDER_TARGET` so the projected-decal pass can flip the
    // resolved target back to RENDER_TARGET to stamp decals onto the
    // scene; the resolve copy still works as before.
    let clear_value = D3D12_CLEAR_VALUE {
        Format: HDR_FORMAT,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: HDR_FORMAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&clear_value),
            &mut res_opt,
        )
    }
    .map_err(|e| format!("create hdr resolve target: {e}"))?;
    res_opt.ok_or_else(|| "create hdr resolve returned None".to_string())
}

// Write an `HDR_FORMAT` Texture2D SRV at the given heap slot so the composite
// pass can sample the HDR scene target.
pub(super) fn write_hdr_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: HDR_FORMAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Single-sample colour render targets
//
// Used by the TAA velocity + history images and the SSAO G-buffer / occlusion
// targets. The bloom mip chain is its own family (see post/bloom.rs).

// Create a single-sample colour render target usable as both a render target
// and a sampled texture. Created in the PIXEL_SHADER_RESOURCE state so the
// first frame can bind it before it has been rendered (the TAA velocity
// buffer and the ping-pong history images). The per-frame cycle flips it to
// RENDER_TARGET for the draw and back. Caller writes the RTV + SRV.
// Cleared to transparent black; see `create_rt_target_with_clear` for targets
// whose per-frame clear is a different value.
pub(super) fn create_rt_target(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
) -> Result<ID3D12Resource, String> {
    create_rt_target_with_clear(device, width, height, format, [0.0; 4])
}

// As `create_rt_target`, but bakes `clear_color` as the resource's optimized
// clear value. This must match the colour the caller passes to
// ClearRenderTargetView every frame, else D3D12 falls back to a slower clear
// path and warns. Defaulting to transparent black covers most targets;
// non-zero backgrounds (e.g. roughness 1.0) pass their value here.
pub(super) fn create_rt_target_with_clear(
    device: &ID3D12Device,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    clear_color: [f32; 4],
) -> Result<ID3D12Resource, String> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let clear_value = D3D12_CLEAR_VALUE {
        Format: format,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: clear_color },
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width.max(1) as u64,
        Height: height.max(1),
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut res_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            Some(&clear_value),
            &mut res_opt,
        )
    }
    .map_err(|e| format!("create rt target: {e}"))?;
    res_opt.ok_or_else(|| "create rt target returned None".to_string())
}

// Write a single-sample Texture2D render-target view of the given format.
pub(super) fn write_format_rtv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    rtv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    unsafe { device.CreateRenderTargetView(resource, Some(&rtv_desc), rtv_cpu) };
}

// Write a single-sample Texture2D shader-resource view of the given format.
pub(super) fn write_format_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    format: DXGI_FORMAT,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Resource barriers

pub(super) fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // Borrow the resource pointer into the barrier without an
                // AddRef. `pResource` is wrapped in `ManuallyDrop`, so a
                // `clone()` here is never released and leaks one reference to
                // the resource on every barrier; against the swapchain back
                // buffers that accumulates until `ResizeBuffers` rejects the
                // resize ("outstanding buffer references"). The caller's
                // `&resource` outlives the `ResourceBarrier` call, so copying
                // the raw pointer (no refcount change) is sound, and the
                // `ManuallyDrop` guarantees it is never released.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                StateBefore: before,
                StateAfter: after,
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
            }),
        },
    }
}

// Aliasing barrier announcing that `after` is about to use heap memory another
// placed resource may have just occupied. `pResourceBefore` is left NULL ("any
// resource could have aliased here"), which is the conservative form: it makes
// no assumption about which resource last owned the memory, so it is correct on
// the first frame (nothing has used it yet) and across the single-buffered
// cyclic reuse without tracking the live occupant. After it, `after`'s contents
// are undefined and `after` must be re-initialized (a Clear/Discard/Copy) before
// any non-overwriting use. Mirrors the Vulkan executor's `UNDEFINED -> ...`
// aliasing transition.
pub(super) fn aliasing_barrier(after: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                // Borrow the resource pointer without an AddRef, same rationale
                // as `transition_barrier`: the caller's `&after` outlives the
                // `ResourceBarrier` call and the `ManuallyDrop` never releases it.
                pResourceAfter: unsafe { std::mem::transmute_copy(after) },
            }),
        },
    }
}

// RAII guard for the common read-modify-write barrier pattern: transition a
// resource into a working state now, and restore it to its resting state when
// the guard drops. Pairing the two halves means an early `return`/`?` between
// them, or a pass inserted mid-scope, can't leave the resource in the wrong
// state for the next pass; the reverse barrier is issued by construction.
//
// Scoped to the single-resource "transition in, draw, transition back" pattern.
// Passes that batch several resources into one `ResourceBarrier`, or that hand a
// resource to the next pass in a new state on purpose, stay explicit.
pub(super) struct ScopedBarrier<'a> {
    list: &'a ID3D12GraphicsCommandList,
    resource: &'a ID3D12Resource,
    // The resting state the guard entered from and restores on drop.
    resting: D3D12_RESOURCE_STATES,
    // The working state the guard transitioned into (restored *from* on drop).
    working: D3D12_RESOURCE_STATES,
}

impl<'a> ScopedBarrier<'a> {
    // Transition `resource` from `resting` to `working` immediately; the reverse
    // transition is recorded onto `list` when the returned guard drops.
    pub(super) fn new(
        list: &'a ID3D12GraphicsCommandList,
        resource: &'a ID3D12Resource,
        resting: D3D12_RESOURCE_STATES,
        working: D3D12_RESOURCE_STATES,
    ) -> Self {
        let forward = transition_barrier(resource, resting, working);
        unsafe { list.ResourceBarrier(&[forward]) };
        Self {
            list,
            resource,
            resting,
            working,
        }
    }
}

impl Drop for ScopedBarrier<'_> {
    fn drop(&mut self) {
        let reverse = transition_barrier(self.resource, self.working, self.resting);
        unsafe { self.list.ResourceBarrier(&[reverse]) };
    }
}

// IBL textures produced by a single `EnvironmentMap` asset. Mirrors the Metal
// `EnvironmentMapTextures` shape so the fragment-shader code stays portable.
// `prefilter_mip_count == 0` is the runtime signal for "IBL disabled"; the
// fragment shader keys off it and falls back to the legacy ambient path.
// The `irradiance` / `prefilter` fields hold the COM resources alive while
// their SRVs are referenced via the shader-visible descriptor heap; the SRV
// GPU handles are read by `draw.rs`, not the GpuResource itself.
#[allow(dead_code)]
pub(super) struct EnvironmentMapTextures {
    pub irradiance: GpuResource,
    pub prefilter: GpuResource,
    pub prefilter_mip_count: u32,
}

// Write a TextureCube SRV (1 mip) at the given heap slot.
fn write_cube_srv_single_mip(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Write a multi-mip TextureCube SRV at the given heap slot.
fn write_cube_srv_mips(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    mip_count: u32,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            TextureCube: D3D12_TEXCUBE_SRV {
                MostDetailedMip: 0,
                MipLevels: mip_count,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Create a 1×1 RGBA32F cube of `value` for every face. Used as the IBL
// fallback when no `EnvironmentMap` is bound; the fragment shader keys off
// `prefilter_mip_count == 0` and skips IBL math, but the cube SRV must still
// resolve to a valid texture.
pub(super) fn create_fallback_cubemap(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    value: [f32; 4],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let face_bytes = [value; 1]; // 16 bytes = one RGBA32F pixel per face
    let mut all_faces = Vec::with_capacity(6 * 16);
    for _ in 0..6 {
        for v in &face_bytes {
            all_faces.extend_from_slice(&v[0].to_le_bytes());
            all_faces.extend_from_slice(&v[1].to_le_bytes());
            all_faces.extend_from_slice(&v[2].to_le_bytes());
            all_faces.extend_from_slice(&v[3].to_le_bytes());
        }
    }
    let resource = upload_cube_resource(device, queue, 1, 1, &all_faces)?;
    write_cube_srv_single_mip(device, &resource, srv_cpu);
    Ok(GpuResource {
        resource,
        srv_cpu,
        srv_gpu,
    })
}

// Upload a six-face HDR cubemap from a CubemapTexture payload. `bytes` is the
// raw RGBA32F face-major data emitted by build/cubemap.rs::compile_cubemap_payload:
// 6 * face_size² * 16 bytes in face order +X, -X, +Y, -Y, +Z, -Z. Single-mip.
#[allow(dead_code)]
pub(super) fn upload_cubemap(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    face_size: u32,
    bytes: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let resource = upload_cube_resource(device, queue, face_size, 1, bytes)?;
    write_cube_srv_single_mip(device, &resource, srv_cpu);
    Ok(GpuResource {
        resource,
        srv_cpu,
        srv_gpu,
    })
}

// Upload an EnvironmentMap payload into two cube textures: a single-mip
// irradiance cube and a multi-mip prefiltered radiance cube. Both are
// RGBA32F TextureCube SRVs.
//
// `irradiance_face` / `prefilter_face` are the mip-0 face sizes. `mip_bytes`
// is one slice per mip in order 0..mip_count; `mip_count` must equal
// `mip_bytes.len()`.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_environment_map(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    irradiance_face: u32,
    irradiance_bytes: &[u8],
    prefilter_face: u32,
    mip_bytes: &[&[u8]],
    irr_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    irr_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    pre_srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    pre_srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<EnvironmentMapTextures, String> {
    if mip_bytes.is_empty() {
        return Err("envmap upload: prefilter mip_bytes must not be empty".into());
    }
    let irradiance_res = upload_cube_resource(device, queue, irradiance_face, 1, irradiance_bytes)
        .map_err(|e| format!("envmap irradiance: {e}"))?;
    write_cube_srv_single_mip(device, &irradiance_res, irr_srv_cpu);

    let prefilter_res = upload_prefilter_cube_resource(device, queue, prefilter_face, mip_bytes)
        .map_err(|e| format!("envmap prefilter: {e}"))?;
    write_cube_srv_mips(device, &prefilter_res, mip_bytes.len() as u32, pre_srv_cpu);

    Ok(EnvironmentMapTextures {
        irradiance: GpuResource {
            resource: irradiance_res,
            srv_cpu: irr_srv_cpu,
            srv_gpu: irr_srv_gpu,
        },
        prefilter: GpuResource {
            resource: prefilter_res,
            srv_cpu: pre_srv_cpu,
            srv_gpu: pre_srv_gpu,
        },
        prefilter_mip_count: mip_bytes.len() as u32,
    })
}

// Create a single-mip RGBA32F TextureCube resource and upload `bytes` (six
// faces in +X,-X,+Y,-Y,+Z,-Z order). Transitions to PIXEL_SHADER_RESOURCE.
fn upload_cube_resource(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    face_size: u32,
    mip_count: u32,
    bytes: &[u8],
) -> Result<ID3D12Resource, String> {
    let face_bytes_mip0 = (face_size as usize) * (face_size as usize) * 16;
    let needed = 6 * face_bytes_mip0 * mip_count as usize;
    if mip_count == 1 && bytes.len() < needed {
        return Err(format!(
            "cubemap data too short for face_size {}: {} bytes, need {}",
            face_size,
            bytes.len(),
            needed
        ));
    }

    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: face_size as u64,
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: mip_count as u16,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create cube texture: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create cube texture returned None".to_string())?;

    upload_face_major_into_cube(
        device,
        queue,
        &texture,
        &desc,
        face_size,
        mip_count,
        &[bytes],
    )?;
    Ok(texture)
}

// Upload a multi-mip prefilter cube. `mip_bytes[m]` is 6 * (face_size >> m)² * 16 bytes
// in face-major order. Mip 0 corresponds to `face_size`; each subsequent mip halves.
fn upload_prefilter_cube_resource(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    face_size: u32,
    mip_bytes: &[&[u8]],
) -> Result<ID3D12Resource, String> {
    let mip_count = mip_bytes.len() as u32;
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: face_size as u64,
        Height: face_size,
        DepthOrArraySize: 6,
        MipLevels: mip_count as u16,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create prefilter cube: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create prefilter cube returned None".to_string())?;

    upload_face_major_into_cube(
        device, queue, &texture, &desc, face_size, mip_count, mip_bytes,
    )?;
    Ok(texture)
}

// Copy face-major RGBA32F bytes into a 6-slice cube `texture`. For each mip
// `m` (0..mip_count), `mip_bytes[m]` is expected to be
// `6 * (face_size >> m)² * 16` bytes in face order +X,-X,+Y,-Y,+Z,-Z.
// Transitions the resource to PIXEL_SHADER_RESOURCE at the end.
fn upload_face_major_into_cube(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    texture: &ID3D12Resource,
    desc: &D3D12_RESOURCE_DESC,
    face_size: u32,
    mip_count: u32,
    mip_bytes: &[&[u8]],
) -> Result<(), String> {
    let num_subresources = 6 * mip_count;
    let mut layouts: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> =
        vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); num_subresources as usize];
    let mut row_counts: Vec<u32> = vec![0; num_subresources as usize];
    let mut row_sizes: Vec<u64> = vec![0; num_subresources as usize];
    let mut total_bytes: u64 = 0;
    unsafe {
        device.GetCopyableFootprints(
            desc,
            0,
            num_subresources,
            0,
            Some(layouts.as_mut_ptr()),
            Some(row_counts.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total_bytes),
        );
    }

    let upload = create_buffer(
        device,
        total_bytes.max(4),
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("cube upload map: {e}"))?;

    // Layout in D3D12: subresource index = mip + face * MipLevels.
    // Source data layout: per-mip slab `mip_bytes[m]`, face-major within each.
    for mip in 0..mip_count {
        let mip_face_size = (face_size >> mip).max(1);
        let face_bytes = (mip_face_size as usize) * (mip_face_size as usize) * 16;
        let slab = mip_bytes[mip as usize];
        if slab.len() < 6 * face_bytes {
            unsafe { upload.Unmap(0, None) };
            return Err(format!(
                "cube upload mip {} too short: {} bytes, need {}",
                mip,
                slab.len(),
                6 * face_bytes
            ));
        }
        for face in 0..6u32 {
            let subres = mip + face * mip_count;
            let layout = &layouts[subres as usize];
            let row_pitch = layout.Footprint.RowPitch as usize;
            let src_row = (mip_face_size as usize) * 16;
            let face_src_offset = (face as usize) * face_bytes;
            for row in 0..mip_face_size as usize {
                let src =
                    &slab[face_src_offset + row * src_row..face_src_offset + (row + 1) * src_row];
                let dst =
                    unsafe { (map_ptr as *mut u8).add(layout.Offset as usize + row * row_pitch) };
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
            }
        }
    }
    unsafe { upload.Unmap(0, None) };

    // `pResource` borrows the upload / texture pointer without an AddRef: the
    // field is a `ManuallyDrop`, so a `clone()` would never be released and would
    // leak a reference to the transient upload buffer and the destination texture
    // on every subresource copy. Both outlive the synchronous `CopyTextureRegion`
    // calls (`texture` is borrowed from the caller).
    one_shot_submit(device, queue, |cmd| {
        for subres in 0..num_subresources {
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { std::mem::transmute_copy(&upload) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: layouts[subres as usize],
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: subres,
                },
            };
            unsafe { cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
        }
        let barrier = transition_barrier(
            texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        );
        unsafe { cmd.ResourceBarrier(&[barrier]) };
    })?;

    Ok(())
}

// Colour-grading LUT (3D texture)

// Write a Texture3D R8G8B8A8_UNORM SRV at the given heap slot.
fn write_lut_srv(
    device: &ID3D12Device,
    resource: &ID3D12Resource,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
) {
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE3D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe { device.CreateShaderResourceView(resource, Some(&srv_desc), srv_cpu) };
}

// Upload a deserialised `ColorLut` payload into a 3D R8G8B8A8_UNORM texture and
// write its Texture3D SRV at the given heap slot. `data` is `size³ * 4` bytes
// in red-fastest, then green, then blue order, the same texel order the
// composite shader samples with the display-referred `(r, g, b)` colour as the
// coordinate. Mirrors `vulkan/texture.rs::upload_color_lut`.
pub(super) fn upload_color_lut(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    size: u32,
    data: &[u8],
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    let n = size as usize;
    let needed = n * n * n * 4;
    if data.len() < needed {
        return Err(format!(
            "color LUT data too short for size {}: {} bytes, need {}",
            size,
            data.len(),
            needed
        ));
    }

    // 3D texture resource (default heap, copy-dest initially).
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: size as u64,
        Height: size,
        DepthOrArraySize: size as u16,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        ..Default::default()
    };
    let mut tex_opt: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut tex_opt,
        )
    }
    .map_err(|e| format!("create color LUT texture: {e}"))?;
    let texture = tex_opt.ok_or_else(|| "create color LUT texture returned None".to_string())?;

    // Query upload size/layout. A 3D texture is one subresource.
    let mut layout = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total_size: u64 = 0;
    unsafe {
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut layout),
            None,
            None,
            Some(&mut total_size),
        );
    }

    let upload = create_buffer(
        device,
        total_size,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;

    // Map and copy row-by-row to match D3D12's row pitch alignment. The placed
    // 3D footprint is `n` depth slices, each `n` rows of `RowPitch` bytes, so
    // the slice pitch is `RowPitch * n`.
    let mut map_ptr = std::ptr::null_mut::<std::ffi::c_void>();
    unsafe { upload.Map(0, None, Some(&mut map_ptr)) }
        .map_err(|e| format!("color LUT upload map: {e}"))?;
    let src_row = n * 4;
    let dst_pitch = layout.Footprint.RowPitch as usize;
    let slice_pitch = dst_pitch * n;
    for z in 0..n {
        for y in 0..n {
            let src_off = (z * n + y) * src_row;
            let src = &data[src_off..src_off + src_row];
            let dst = unsafe {
                (map_ptr as *mut u8).add(layout.Offset as usize + z * slice_pitch + y * dst_pitch)
            };
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_row) };
        }
    }
    unsafe { upload.Unmap(0, None) };

    // Copy upload → texture, then transition to shader-read. `pResource` borrows
    // the upload / texture pointer without an AddRef: the field is a `ManuallyDrop`,
    // so a `clone()` would never be released and would leak a reference to the
    // transient upload buffer and the destination texture on every upload. Both
    // outlive the synchronous `CopyTextureRegion` call.
    one_shot_submit(device, queue, |cmd| {
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&upload) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: layout,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: unsafe { std::mem::transmute_copy(&texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        unsafe {
            cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            let barrier = transition_barrier(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            );
            cmd.ResourceBarrier(&[barrier]);
        }
    })?;

    write_lut_srv(device, &texture, srv_cpu);
    Ok(GpuResource {
        resource: texture,
        srv_cpu,
        srv_gpu,
    })
}

// Build a 2×2×2 identity colour LUT so the composite pass always binds a valid
// Texture3D even when the world declares no `ColorLut`. With the identity LUT
// the grade is a no-op at any `lut_strength`. Mirrors
// `vulkan/texture.rs::create_fallback_color_lut`.
pub(super) fn create_fallback_color_lut(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    srv_cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    srv_gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
) -> Result<GpuResource, String> {
    // Red-fastest, then green, then blue, matching the payload texel order.
    let mut data = Vec::with_capacity(2 * 2 * 2 * 4);
    for b in 0..2u8 {
        for g in 0..2u8 {
            for r in 0..2u8 {
                data.extend_from_slice(&[r * 255, g * 255, b * 255, 255]);
            }
        }
    }
    upload_color_lut(device, queue, 2, &data, srv_cpu, srv_gpu)
}
