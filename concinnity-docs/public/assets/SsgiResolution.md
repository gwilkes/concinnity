<!-- Auto-generated - do not edit. -->

# SsgiResolution

Internal render resolution of the SSGI gather pass (only meaningful when
`indirect_lighting` is `ssgi`). The gather is the expensive part (a
hemisphere ray-march per pixel), and its composite is a depth-aware
bilateral filter that upsamples a lower-resolution gather back to full
resolution at little visible cost. `half` (the default) gathers at a quarter
of the pixels for a large saving; `full` keeps the gather at native
resolution; `quarter` is the cheapest, for low-end GPUs or debugging.

## Values

- `full`
- `half`
- `quarter`
