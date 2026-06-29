<!-- Auto-generated - do not edit. -->

# TextLabel

Screen-space text drawn as a UI overlay on top of the 3D scene each frame.

Text is laid out using the referenced [Font](Font.md). The `content` field can
be updated every frame (e.g. by an [FpsCounter](FpsCounter.md)).

A `\n` in `content` starts a new line. When `background` has an alpha > 0, a
box is filled behind the glyphs, extended outward by `padding` pixels,
useful for HUD chips.

```jsonl
{
  "type": "TextLabel",
  "name": "fps_text",
  "args": {
    "font": "fps_font",
    "content": "FPS: --",
    "x": 10,
    "y": 10,
    "color": [
      1,
      1,
      1
    ],
    "scale": 1
  }
}
```

## Parameters

- `font`: A string. The [Font](Font.md) asset to use for rendering. Optional.
- `content`: A string. Text to display. Can be updated each frame.
- `x`: A float. Horizontal position in pixels from the left edge of the window. Defaults to `10.0`.
- `y`: A float. Vertical position in pixels from the top edge of the window. Defaults to `10.0`.
- `color`: An array of 3 floats. Linear-space RGB text colour. Defaults to `[1.0, 1.0, 1.0]`.
- `scale`: A float. Uniform scale applied on top of the font's `size_px`. 1.0 = native size. Defaults to `1.0`.
- `centered`: A boolean. When true, center the label in the viewport each frame; x and y are ignored. Defaults to `false`.
- `background`: An array of 4 floats. RGBA fill of a box drawn behind the text. An alpha of 0 (the default) draws no box; any alpha > 0 draws the box at that opacity.
- `padding`: A float. Pixels the background box extends past the text on every side. Only meaningful when `background` is visible. Defaults to `0.0`.
- `visible`: A boolean. When false, the label is hidden. Defaults to `true`.
- `view`: A string. [View](View.md) this label belongs to. Resolved automatically from the naming convention (`<view>_*`); you don't set this directly. `None` means the label is always visible. Optional.
