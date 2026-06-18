# Builtin shader assets

These are the engine's default `ShaderStage` **assets** — user-facing shader
content referenced by name from a world's JSONL (e.g. `default.metal`).

- Compiled at **world-build time** by `src/build/shader.rs` (`xcrun metal` +
  `metallib` for MSL, `D3DCompile` for HLSL) into bytecode that is packed into
  the asset blob.
- Loaded at runtime as pre-compiled libraries (e.g. `MtlContext::new` →
  `load_library`), never compiled from source at runtime.
- A world may override them with its own shader assets.

Engine-internal pipeline shaders (post-process, TAA, SSAO/SSR, cull, text, …)
are NOT here — those live per backend, e.g. `src/metal/shaders/`, and are
compiled from source at runtime.
