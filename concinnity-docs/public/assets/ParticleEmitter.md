<!-- Auto-generated - do not edit. -->

# ParticleEmitter

A billboard particle emitter.

Particles spawn from `position` in a cone centred on `direction` (half-angle
`spread_deg`), with a speed drawn from `[speed_min, speed_max]` and a
lifetime from `[lifetime_min, lifetime_max]`. Over each particle's life its
size interpolates from `size_start` to `size_end` and its colour from
`color_start` to `color_end`. Each particle is drawn as a camera-facing quad
textured by `texture`.

The pool holds `max_particles` particles; new ones spawn at `spawn_rate` per
second, reusing slots as old particles die.

```jsonl
{"name":"sparks","type":"ParticleEmitter","args":{"texture":"tex_spark","position":[0,1,0],"direction":[0,1,0],"spread_deg":25,"speed_min":2,"speed_max":5,"lifetime_min":0.5,"lifetime_max":1.5,"spawn_rate":80,"max_particles":512,"size_start":0.08,"size_end":0.02,"color_start":[1,0.8,0.3,1],"color_end":[1,0.1,0,0]}}
```

## Parameters

- `texture`: A string. [Texture](Texture.md) sampled per particle. `None` uses a white fallback so the colour gradient still shows. Optional.
- `position`: An array of 3 floats. World-space spawn origin. Defaults to `[0.0, 0.0, 0.0]`.
- `direction`: An array of 3 floats. Mean emission direction. The cone of width `spread_deg` is centred on this vector. Normalised on load; a zero vector falls back to `[0, 1, 0]`. Defaults to `[0.0, 1.0, 0.0]`.
- `spread_deg`: A float. Cone half-angle in degrees around `direction`. `0` emits a straight jet; `180` emits in all directions. Defaults to `15.0`.
- `speed_min`: A float. Lower bound on initial speed (m/s). Floored at 0. Defaults to `1.0`.
- `speed_max`: A float. Upper bound on initial speed (m/s). Lifted to at least `speed_min`. Defaults to `2.0`.
- `lifetime_min`: A float. Lower bound on particle lifetime (seconds). Must be > 0. Defaults to `1.0`.
- `lifetime_max`: A float. Upper bound on particle lifetime (seconds). Lifted to at least `lifetime_min`. Defaults to `2.0`.
- `gravity`: An array of 3 floats. Constant acceleration applied to each particle, in world units per second squared. Defaults to `[0.0, -9.8, 0.0]`.
- `spawn_rate`: A float. Particles spawned per second. `0` produces a one-shot burst that then empties as particles age out. Defaults to `32.0`.
- `max_particles`: An integer. Maximum number of particles alive at once. Clamped to `[1, 65536]`. Defaults to `256`.
- `size_start`: A float. Billboard side length at spawn, in world units. Defaults to `0.2`.
- `size_end`: A float. Billboard side length at death, in world units. Defaults to `0.05`.
- `color_start`: An array of 4 floats. Linear-space RGBA multiplier applied to the texture at spawn. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
- `color_end`: An array of 4 floats. Linear-space RGBA multiplier applied to the texture at death. Defaults to `[1.0, 1.0, 1.0, 0.0]`.
- `visible`: A boolean. When false the emitter is skipped each frame. Defaults to `true`.
