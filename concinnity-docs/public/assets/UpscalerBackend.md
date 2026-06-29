<!-- Auto-generated - do not edit. -->

# UpscalerBackend

Upscaler backend selector for `PostProcessConfig.temporal_upscaling`.
`Auto` resolves at runtime to the best available (DLSS, then XeSS, then
FSR3); the explicit variants request a specific backend and fall back when
it is unavailable. DLSS (NVIDIA NGX) and XeSS (Intel) are DirectX-only;
Metal uses MetalFX and Vulkan has no upscaler yet, so both treat any value
as their native path.

## Values

- `auto`
- `fsr3`
- `dlss`
- `xess`
