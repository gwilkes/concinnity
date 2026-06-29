<!-- Auto-generated - do not edit. -->

# OptionSelect

A settings row that cycles through a fixed set of values on click.

`OptionSelect` is a build-time shorthand for one row of a settings menu: a
left-aligned name, a right-aligned current value, and a clickable region
that advances the value. It expands into a [TextLabel](TextLabel.md) for the
name, a `TextLabel` for the value, and a [HitRegion](HitRegion.md) that fires a
`"setting:<setting>:next"` action.

The `setting` field names an engine setting the runtime knows how to read,
cycle, and apply (e.g. `"vsync"`); its option list lives in the engine, not
here. The value label shows a placeholder at build time and is corrected to
the live value when the world starts.

```jsonl
{"name":"opt_vsync","type":"OptionSelect","args":{"setting":"vsync","label":"Vsync"}}
```

Generated names are prefixed with this asset's `name` (`<name>_label`,
`<name>_value`, `<name>_btn`), so they never clash with hand-authored assets.

## Parameters

- `setting`: A string. Engine setting this row controls (e.g. `"vsync"`). Must be a setting the runtime recognises; an unknown key renders but does nothing on click.
- `label`: A string. Display name shown at the left of the row.
- `x`: A float. Left edge of the row in window pixels. Defaults to `0.0`.
- `y`: A float. Top edge of the row in window pixels. Defaults to `0.0`.
- `width`: A float. Row width in window pixels (name sits at the left, value at the right). Defaults to `360.0`.
- `height`: A float. Row height in window pixels (the clickable region's height). Defaults to `48.0`.
- `font`: A string. [Font](Font.md) for the row text. Empty uses the built-in font.
- `font_px`: A float. Pixel size of the row text when it uses the built-in font (that is, when `font` is empty). Ignored when `font` names a [Font](Font.md), which carries its own size. Defaults to `48.0`.
- `text_color`: An array of 3 floats. Linear-space RGB color of the name text. Defaults to `[0.85, 0.85, 0.85]`.
- `value_color`: An array of 3 floats. Linear-space RGB color of the value text. Defaults to `[0.85, 0.85, 0.85]`.
- `text_scale`: A float. Scale applied to the row text. Defaults to `1.0`.
- `hover_color`: An array of 3 floats. RGB color of the value text while the row is hovered. Defaults to `[1.0, 0.85, 0.3]`.
- `hover_scale`: A float. Scale of the value text while the row is hovered. Defaults to `1.08`.
- `stepper_width`: A float. Width in pixels of the `<` previous-value click region. The `>` next-value region spans the rest of the row's right half (the value sits inside it), so a click on the value advances to the next option. Defaults to `40.0`.
