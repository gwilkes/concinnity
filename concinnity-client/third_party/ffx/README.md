# Vendored FidelityFX Vulkan runtime (patched)

`amd_fidelityfx_vk.dll` here is a locally rebuilt FidelityFX SDK **v1.1.4**
Vulkan runtime with a one-line shader fix applied. `build.rs`
(`setup_fidelityfx_vk_sdk`) copies it next to the executable in preference to the
stock SDK copy when present; delete it to fall back to the unmodified SDK DLL.

## Why

The stock SDK declares the FSR3 upscaler's `rw_luma_history` storage image as
`rgba8` in its GLSL callback, while the C++ creates the resource as
`R16G16B16A16_FLOAT`. On Vulkan that trips a validation-layer format-mismatch
warning on every FSR dispatch, and the reads/stores are per-spec undefined. It is
upstream issue
[#161](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/issues/161)
(open, no AMD fix). DX12 is unaffected (typed UAVs, no SPIR-V format operand).

## The fix

`patches/fsr3upscaler_luma_history_rgba16f.patch`: change the qualifier
`rgba8` -> `rgba16f` so the shader declaration matches the view. Output is
visually identical (independently confirmed on issue #161); validation goes clean.

## How to rebuild

Run the helper from the repo root (needs CMake, Visual Studio Build Tools with
the C++ x64 toolset, and the Vulkan SDK with `VULKAN_SDK` set):

```
pwsh scripts/setup_ffx_vk_dll.ps1                 # uses C:\FidelityFX-SDK-v1.1.4
pwsh scripts/setup_ffx_vk_dll.ps1 -CloneIfMissing # clone v1.1.4 source if absent
```

The script applies the patch, builds `ffx-api` for `VK_X64` (which pulls in the
`sdk` subproject and recompiles the shader permutations), and copies the built
`amd_fidelityfx_vk.dll` here.

## Provenance / caveats

- Source: FidelityFX SDK v1.1.4 (`https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK`, tag `v1.1.4`).
- This DLL is **locally built and unsigned**, unlike AMD's `PrebuiltSignedDLL`.
- ABI-identical to the stock v1.1.4 runtime (only a shader format qualifier
  changed), so the engine's `ffx_api` FFI and its struct-size tests are unchanged.
