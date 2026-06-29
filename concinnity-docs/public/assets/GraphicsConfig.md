<!-- Auto-generated - do not edit. -->

# GraphicsConfig

Rendering settings for the world: frame pacing, shadows, and clear colour.
One per world. The GPU backend is chosen by the engine for the platform and
is not user-configurable.

```json
{
  "name": "gfx",
  "type": "GraphicsConfig",
  "args": { "clear_color": [0.1, 0.1, 0.15, 1.0], "frames_in_flight": 2 }
}
```

## Parameters

- `max_frames`: An integer. Cap the render loop at this many frames, then exit. Unset runs until the window is closed.
- `frames_in_flight`: An integer. Preferred number of frames in flight (1-3). Higher can smooth pacing at the cost of input latency. Defaults to `2`.
- `vsync`: A boolean. Cap the frame rate to the display refresh (vsync). Defaults to `false`: the render loop runs uncapped (DirectX presents with tearing allowed, Vulkan uses a mailbox present mode), which is what a benchmark wants. Set to `true` to lock presentation to the monitor refresh, eliminating tearing and the wasted frames that never reach the screen.
- `fps_cap`: An integer. Cap the frame rate to this many frames per second. `0` (default) leaves the loop uncapped. The cap is a CPU-side frame pacer, so it composes with `vsync`: the more restrictive of the two wins. Useful for limiting heat, fan noise, and power draw, or matching a fixed refresh.
- `clear_color`: An array of 4 floats. Background clear colour [r, g, b, a] in linear 0..1 space. Defaults to `[0.01, 0.01, 0.02, 1.0]`.
- `rotation_speed`: A float. Rotation speed of the demo object in radians per second. Only used when no camera is present. Defaults to `1.0`.
- `shadow_map_size`: An integer. Shadow map resolution in texels (e.g. 2048). Set to 0 to disable shadows. Defaults to `2048`.
- `shadow_update`: A string (see [ShadowUpdate](ShadowUpdate.md)). How often shadow cascades are re-rendered. `hybrid` (default) amortizes the far cascades across frames; `every_frame` refreshes them all every frame.
- `shadow_distance`: An integer. How far from the camera shadows are cast, in world units (e.g. 80). The cascades cover from the near plane out to this distance; a larger value shadows more of the scene but spreads the same shadow-map resolution over more area (softer, blockier shadows). Capped at the camera far plane. Defaults to `80`.
- `shadow_cascades`: An integer. Number of shadow cascades, 1 to 4 (`4` is the default and the maximum). More cascades keep distant shadows sharper by splitting the view range into finer slices, at the cost of an extra shadow-map render per cascade; fewer is cheaper but blockier far from the camera. The slice count covers the same `shadow_distance` regardless.
- `anisotropy`: An integer. Maximum anisotropic-filtering degree for the scene texture sampler (albedo + normal maps), e.g. 8. Higher keeps textures viewed at a grazing angle (floors, walls receding into the distance) sharp instead of blurring along the minor axis, at a small sampling cost. `1` disables anisotropy (plain trilinear). Clamped to the GPU's supported range (1..16) at init. Defaults to `8`.
