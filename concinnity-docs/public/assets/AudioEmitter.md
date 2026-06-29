<!-- Auto-generated - do not edit. -->

# AudioEmitter

A point source of sound in the world.

Plays its `clip` (an [AudioClip](AudioClip.md) reference) from `position`,
attenuated and panned relative to the camera. When `prop` names a
[Prop](Prop.md), the emitter tracks that prop's position every frame, so the
sound follows a moving object.

```jsonl
{"name":"fire_sound","type":"AudioEmitter","args":{"clip":"fire_loop","position":[6.0,4.0,-6.0]}}
```

## Parameters

- `clip`: A string. The [AudioClip](AudioClip.md) this emitter plays. Optional.
- `position`: An array of 3 floats. World-space position of the sound source.
- `volume`: A float. Linear gain multiplier applied to the clip. Defaults to `1.0`.
- `looping`: A boolean. Whether the clip restarts when it ends. Defaults to `true`.
- `prop`: A string. Optional [Prop](Prop.md) whose position the emitter tracks each frame.
