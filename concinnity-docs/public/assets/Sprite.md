<!-- Auto-generated - do not edit. -->

# Sprite

Screen-space 2D rectangle drawn as a UI overlay each frame.

Sprites are pixel-anchored quads with an RGBA tint. They draw alongside
[TextLabel](TextLabel.md)s, ordered behind labels so text sits on top.

Currently only the tint is drawn (solid-coloured rectangles). The `texture`
field is reserved for forward compatibility: a sprite with `texture` set
renders exactly as if it were unset.

```jsonl
{
  "name": "title_menu_bg",
  "type": "Sprite",
  "args": {
    "x": 0, "y": 0, "width": 1280, "height": 720,
    "tint": [0.04, 0.06, 0.10, 1.0]
  }
}
```

## Parameters

- `x`: A float. Left edge in screen pixels from the window's top-left. Defaults to `0.0`.
- `y`: A float. Top edge in screen pixels from the window's top-left. Defaults to `0.0`.
- `width`: A float. Width in screen pixels. Defaults to `100.0`.
- `height`: A float. Height in screen pixels. Defaults to `100.0`.
- `texture`: A string. [Texture](Texture.md) to draw (reserved; not yet sampled). Optional.
- `tint`: An array of 4 floats. RGBA colour the rectangle is filled with, each channel in [0, 1]. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
- `follow_cursor`: A boolean. When true, the sprite acts as an in-engine cursor: it is drawn on top of the other overlays as an arrow pointer tracking the mouse, with the pointer at the arrow's tip. `tint` is the arrow fill (a contrasting outline is added automatically) and `height` its size; `width` is ignored so the arrow keeps its shape. The system cursor is hidden while a visible `follow_cursor` sprite exists. Defaults to `false`.
- `visible`: A boolean. When false the sprite is skipped each frame. Defaults to `true`.
- `view`: A string. [View](View.md) this sprite belongs to. Resolved automatically from the naming convention (`<view>_*`); you don't set this directly. `None` means the sprite is always visible (e.g. a scene background). Optional.
