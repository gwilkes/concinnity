<!-- Auto-generated - do not edit. -->

# Font

Rasterises a TrueType font into a glyph atlas at build time.

Reference a Font by name from a [TextLabel](TextLabel.md).

```jsonl
{
  "type": "Font",
  "name": "fps_font",
  "args": {
    "path": "assets/fonts/JetBrainsMono-Regular.ttf",
    "size_px": 20
  }
}
```

## Parameters

- `path`: A string. Path to the TTF file, relative to the project root.
- `size_px`: An integer. Rasterisation size in pixels. Determines the rendered glyph height. Defaults to `20`.
