<!-- Auto-generated - do not edit. -->

# AudioClip

A baked audio clip: the sound an [AudioEmitter](AudioEmitter.md) plays.

The build reads the `source` file (any format the engine can decode:
`.ogg`, `.wav`, `.flac`, `.mp3`) and packs it into the world.

An `AudioClip` is inert on its own: reference it from an
[AudioEmitter](AudioEmitter.md)'s `clip` field to place the sound in the world.

```jsonl
{"name":"fire_loop","type":"AudioClip","args":{"source":"audio/fire_crackle.ogg"}}
```

## Parameters

- `source`: A string. Path to the source audio file.
