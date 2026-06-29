<!-- Auto-generated - do not edit. -->

# WaterWave

One wave in a water surface's motion. A surface sums up to
[`MAX_WATER_WAVES`] of these to displace its flat grid. Each wave travels
horizontally along `direction`, rising and falling with `amplitude` peak
height, `wavelength` distance between crests, and `speed` metres per second.
`steepness` in [0, 1] pinches the crests and broadens the troughs (choppier
water).

## Parameters

- `amplitude`: A float. Peak height of the wave, in world units. Defaults to `0.15`.
- `wavelength`: A float. Distance between successive crests, in world units. Defaults to `4.0`.
- `speed`: A float. Horizontal travel speed, in metres per second. Defaults to `1.0`.
- `direction`: An array of 2 floats. Horizontal travel direction `[x, z]`. Defaults to `[1.0, 0.0]`.
- `steepness`: A float. Crest sharpness in [0, 1]. 0 is a smooth sine; higher pinches crests and broadens troughs. Defaults to `0.4`.
