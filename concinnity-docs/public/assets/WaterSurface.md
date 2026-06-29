<!-- Auto-generated - do not edit. -->

# WaterSurface

A translucent animated water surface.

A flat, subdivided horizontal surface whose vertices ripple with summed
waves. It refracts and reflects the scene, blends from a shallow to a deep
colour with depth, and adds shoreline foam.

The surface is positioned by `centre` and sized by `extent` (XZ
half-widths). The mesh itself is flat; all height variation comes from the
animated waves.

```jsonl
{"name":"pond","type":"WaterSurface","args":{
  "centre":[0.0,0.4,0.0],
  "extent":[12.0,8.0],
  "subdivisions":96,
  "waves":[
    {"amplitude":0.10,"wavelength":3.0,"speed":0.7,"direction":[1.0,0.0],"steepness":0.4},
    {"amplitude":0.05,"wavelength":1.5,"speed":1.1,"direction":[-0.4,0.8],"steepness":0.3}
  ]
}}
```

## Parameters

- `centre`: An array of 3 floats. World-space position of the surface's centre. Defaults to `[0.0, 0.0, 0.0]`.
- `extent`: An array of 2 floats. Half-width and half-depth of the surface `[x, z]`, in world units. Defaults to `[10.0, 10.0]`.
- `subdivisions`: An integer. Grid subdivisions across the surface. Higher gives smoother waves. Clamped to [8, 255]. Defaults to `64`.
- `waves`: An array of [WaterWave](WaterWave.md) objects. The waves summed to animate the surface (up to 4). Defaults to a single gentle wave.
- `deep_colour`: An array of 3 floats. Linear-space RGB colour of deep water. Defaults to `[0.02, 0.05, 0.15]`.
- `shallow_colour`: An array of 3 floats. Linear-space RGB colour of shallow water near the shore. Defaults to `[0.20, 0.50, 0.55]`.
- `depth_falloff_metres`: A float. Depth over which the colour blends from shallow to deep, in metres. Defaults to `4.0`.
- `foam_width_metres`: A float. Width of the shoreline foam band, in metres. Defaults to `0.30`.
- `foam_intensity`: A float. Strength of the shoreline foam, in [0, 1]. Defaults to `0.8`.
- `fresnel_power`: A float. Sharpness of the grazing-angle reflection. Higher confines reflections to steeper viewing angles. Defaults to `5.0`.
- `roughness`: A float. Surface roughness in [0, 1]. Higher gives blurrier reflections. Defaults to `0.05`.
- `refraction_strength`: A float. How strongly the surface bends the view of what's beneath it. Defaults to `0.15`.
- `visible`: A boolean. When false the surface is skipped each frame. Defaults to `true`.
