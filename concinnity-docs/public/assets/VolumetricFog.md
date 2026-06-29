<!-- Auto-generated - do not edit. -->

# VolumetricFog

Environmental volumetric fog: a single lit medium that wraps the scene,
thicker near the ground and thinning with height, with extra glow around the
sun.

Only one `VolumetricFog` is honoured: the first declared instance wins;
later instances are silently dropped. With none declared, there is no fog.

```jsonl
{"name":"fog","type":"VolumetricFog","args":{"density":0.08,"color":[0.75,0.82,0.95],"height_falloff":0.18,"max_distance":160.0,"phase_g":0.5}}
```

## Parameters

- `enabled`: A boolean. Master toggle. `false` disables the fog even when this asset is present. Defaults to `true`.
- `color`: An array of 3 floats. Linear-space RGB tint of the fog: the colour the camera sees in the far distance. Defaults to `[0.7, 0.78, 0.85]`.
- `density`: A float. Base thickness of the fog at `height_reference` (per world unit). Higher is thicker. Floored at 0. Defaults to `0.05`.
- `height_falloff`: A float. How quickly the fog thins with height above `height_reference`. 0 keeps it uniform; larger values pin it to the ground. Defaults to `0.2`.
- `height_reference`: A float. World-space Y at which the fog reaches full `density`. It thickens below this height and thins above it. Defaults to `0.0`.
- `max_distance`: A float. Maximum distance the fog covers from the camera, in world units. Past this, distant geometry stays clear. Defaults to `200.0`.
- `phase_g`: A float. Sun-glow anisotropy in `(-1, 1)`. Positive values concentrate brightness around the sun (haloes), negative values scatter away from it, 0 is uniform. Defaults to `0.4`.
- `ambient`: A float. Constant ambient brightness so the fog keeps some colour in shaded areas. Defaults to `0.15`.
