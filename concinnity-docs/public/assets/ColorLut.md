<!-- Auto-generated - do not edit. -->

# ColorLut

A 3D colour-grading lookup table applied as a final post-process step. The
build bakes the source into a colour cube; the graded result is blended over
the image by [PostProcessConfig](PostProcessConfig.md)'s `lut_strength`.

A world declares at most one `ColorLut`; the first wins. When none is
present, colour grading is skipped regardless of `lut_strength`.

Two source formats are accepted, picked by file extension:
  - `.cube`  Adobe Cube LUT (plain-text interchange format).
  - `.png`   A horizontal slice strip: `(n*n)` wide by `n` tall.

```jsonl
{"name":"grade","type":"ColorLut","args":{"source":"luts/cinematic_warm.cube"}}
```

## Parameters

- `source`: A string. Path to the source `.cube` or `.png` LUT file.
