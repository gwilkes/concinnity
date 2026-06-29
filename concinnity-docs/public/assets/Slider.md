<!-- Auto-generated - do not edit. -->

# Slider

A settings row that sets a continuous value by dragging a handle along a
track.

`Slider` is a build-time shorthand for one row of a settings menu: a
left-aligned name, a draggable track with a handle, and a right-aligned
current value. It expands into a [TextLabel](TextLabel.md) for the name, a
[TextLabel](TextLabel.md) for the value, two [Sprite](Sprite.md)s (the track and
the handle), and a [HitRegion](HitRegion.md) covering the track that fires a
`"setting:<setting>:drag"` action. While the region is pressed the handle
follows the cursor and the value updates live.

The `setting` field names an engine setting the runtime knows how to map
from a fraction, apply, and format (e.g. `"exposure"`); its value range and
display format live in the engine, not here. The value label and handle
show a placeholder position at build time and are corrected to the live
value when the world starts.

```jsonl
{"name":"sld_exposure","type":"Slider","args":{"setting":"exposure","label":"Exposure"}}
```

Generated names are prefixed with this asset's `name` (`<name>_label`,
`<name>_value`, `<name>_track`, `<name>_handle`, `<name>_drag`), so they
never clash with hand-authored assets.

## Parameters

- `setting`: A string. Engine setting this row controls (e.g. `"exposure"`). Must be a setting the runtime recognises as a slider; an unknown key renders but does nothing on drag.
- `label`: A string. Display name shown at the left of the row.
- `x`: A float. Left edge of the row in window pixels. Defaults to `0.0`.
- `y`: A float. Top edge of the row in window pixels. Defaults to `0.0`.
- `width`: A float. Row width in window pixels (name sits at the left, track and value at the right). Defaults to `360.0`.
- `height`: A float. Row height in window pixels (the draggable region's height). Defaults to `48.0`.
- `font`: A string. [Font](Font.md) for the row text. Empty uses the built-in font.
- `font_px`: A float. Pixel size of the row text when it uses the built-in font (that is, when `font` is empty). Ignored when `font` names a [Font](Font.md), which carries its own size. Defaults to `48.0`.
- `text_color`: An array of 3 floats. Linear-space RGB color of the name text. Defaults to `[0.85, 0.85, 0.85]`.
- `value_color`: An array of 3 floats. Linear-space RGB color of the value text. Defaults to `[0.85, 0.85, 0.85]`.
- `text_scale`: A float. Scale applied to the row text. Defaults to `1.0`.
- `track_color`: An array of 4 floats. RGBA color of the track bar behind the handle. Defaults to `[0.28, 0.28, 0.32, 1.0]`.
- `handle_color`: An array of 4 floats. RGBA color of the draggable handle. Defaults to `[1.0, 0.85, 0.3, 1.0]`.
