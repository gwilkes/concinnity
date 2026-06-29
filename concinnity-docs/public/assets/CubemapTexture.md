<!-- Auto-generated - do not edit. -->

# CubemapTexture

A six-face HDR cubemap baked from an equirectangular Radiance HDR source.

The build resamples the source into six square HDR faces of `face_size`
pixels each, used as an environment / image-based-lighting source.

```jsonl
{"name":"env_studio","type":"CubemapTexture","args":{"source":"assets/hdri/studio.hdr","face_size":512}}
```

## Parameters

- `source`: A string. Path to the source equirectangular HDR (`.hdr`) file, relative to the project root.
- `face_size`: An integer. Edge length of each cube face in pixels. Must be a power of two. Defaults to `256`.
