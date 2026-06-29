<!-- Auto-generated - do not edit. -->

# EnvironmentMap

A baked lighting environment built from a Radiance HDR equirectangular
source (or a built-in generator). It provides the scene's ambient
image-based lighting (soft diffuse fill plus glossy reflections that follow
surface roughness) and the on-screen sky.

**`prefilter_face_size` note:** this controls both the reflection detail and
the on-screen sky sharpness. 512 is the default balance: 256 visibly
pixelates a 4K-source HDR sky, 1024 sharpens it further at 4× the size.

**Built-in generators:** `sky` produces a procedural blue sky with a soft
sun, useful when no HDR file is available.

```jsonl
{"name":"env_studio","type":"EnvironmentMap","args":{"source":"assets/hdri/studio.hdr"}}
{"name":"env_outdoor","type":"EnvironmentMap","args":{"source":"assets/hdri/sky.hdr","prefilter_face_size":512}}
{"name":"env_proc","type":"EnvironmentMap","args":{"generator":"sky"}}
```

## Parameters

- `source`: A string. Path to the source equirectangular HDR (`.hdr`) file, relative to the project root. Mutually exclusive with `generator`.
- `generator`: A string. Built-in source name (e.g. "sky"). Mutually exclusive with `source`.
- `prefilter_face_size`: An integer. Face size of the reflection/sky cubemap, in pixels. Higher is sharper but larger. Defaults to `512`.
- `irradiance_face_size`: An integer. Face size of the diffuse ambient cubemap, in pixels. Defaults to `8`.
- `prefilter_samples`: An integer. Number of samples used to filter each reflection texel. Higher reduces noise at the cost of build time. Defaults to `1024`.
- `prefilter_clamp`: A float. Upper bound on how bright a single source texel may count while building the glossy reflection mips. A clear-sky HDR holds a few sun or sky texels thousands of times brighter than their surroundings; left unbounded they survive into the small (coarse) reflection mips as lone hot texels and smear across glossy floors as hard bright squares. This caps each sampled texel so that energy spreads smoothly across the reflection instead. It affects reflections only, never the on-screen sky. Set to `0` to disable (no cap); lower values clamp harder. Defaults to `12.0`.
