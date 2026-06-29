<!-- Auto-generated - do not edit. -->

# PointLight

A spherical point light with quadratic distance attenuation.

Up to 8 point lights may be declared; extras beyond 8 are silently ignored.

```jsonl
{"name":"lamp","type":"PointLight","args":{"position":[2.0,2.5,-3.0],"color":[1.0,0.8,0.5],"intensity":8.0,"range":6.0}}
```

## Parameters

- `position`: An array of 3 floats. World-space position of the light source. Defaults to `[0.0, 2.5, 0.0]`.
- `color`: An array of 3 floats. Linear-space RGB colour of the light. Defaults to `[1.0, 1.0, 1.0]`.
- `intensity`: A float. Intensity multiplier applied to the colour. Defaults to `8.0`.
- `range`: A float. Maximum reach in world units; attenuation is zero at this distance. Defaults to `6.0`.
