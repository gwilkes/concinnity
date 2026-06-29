<!-- Auto-generated - do not edit. -->

# ReflectionBlurResolution

Internal render resolution of the roughness-aware reflection blur (only
meaningful when `ssr` or `ray_traced_reflections` is on). The blur is the
expensive multi-tap part of the reflection composite and is low-frequency
(a widening glossy cone), so running it at a fraction of the pixels and
bilinearly upsampling is visually free. `half` (the default) blurs at a
quarter of the pixels; `full` keeps it at native resolution; `quarter` is
the cheapest. Mirrors stay sharp regardless: the composite lerps in the
full-resolution reflection for low roughness.

## Values

- `full`
- `half`
- `quarter`
