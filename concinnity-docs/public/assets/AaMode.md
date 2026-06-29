<!-- Auto-generated - do not edit. -->

# AaMode

Anti-aliasing mode for `PostProcessConfig.aa_mode`. `Off` runs no edge
smoothing; `Fxaa` (default) applies the composite's single-frame edge
filter, which is nearly free; `Taa` adds a temporal pass that jitters the
projection and reprojects detail across frames for the cleanest edges, at
the cost of a velocity pre-pass and a per-frame history buffer.

## Values

- `off`
- `fxaa`
- `taa`
