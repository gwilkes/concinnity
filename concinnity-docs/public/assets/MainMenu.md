<!-- Auto-generated - do not edit. -->

# MainMenu

A ready-made menu declared in a single line.

`MainMenu` is a build-time shorthand. It expands into the assets a menu is
built from: a [View](View.md) layer, a dim backdrop [Sprite](Sprite.md), a
[TextLabel](TextLabel.md) and [HitRegion](HitRegion.md) for each item, an
optional [KeyBinding](KeyBinding.md) that toggles the menu, and an optional
in-engine mouse cursor [Sprite](Sprite.md). So `world.jsonl` stays small.

The bare form gives a centered Return / Settings / Quit menu shown on load:

```jsonl
{"name":"main_menu","type":"MainMenu"}
```

**Items.** Each item has a `label` (the text) and an `action` fired on
click. `action` takes the same vocabulary as [HitRegion](HitRegion.md)
(`"scene:<name>"`, `"quit"`, `"view:show:<name>"`, `"view:hide"`,
`"view:toggle:<name>"`) plus two conveniences resolved against this menu:
- `"return"`: hide this menu (the same as `"view:hide"`).
- `"settings"`: open a generated settings sub-menu that has a Back button.

```jsonl
{"name":"title","type":"MainMenu","args":{"items":[
  {"label":"New Game","action":"scene:level_1"},
  {"label":"Quit","action":"quit"}
]}}
```

**Generated names** are prefixed with the menu's `name` (`<name>_btn_0`,
`<name>_label_0`, `<name>_cursor`, ...), so they never clash with
hand-authored assets and you never reference them by hand.

## Parameters

- `items`: An array of [MainMenuItem](MainMenuItem.md) objects. Menu entries, top to bottom. Each one is a clickable button.
- `title`: A string. Optional heading drawn above the items. Empty draws no heading.
- `initial`: A boolean. Show the menu as soon as the world loads. Defaults to `true`.
- `toggle_key`: A string. Key that toggles the menu while the cursor is free. Empty binds no key. Only `"Escape"` is currently recognised by the runtime. Defaults to `"Escape"`.
- `dim`: An array of 4 floats. RGBA fill drawn across the whole window behind the items. Defaults to opaque black: a fully opaque alpha (1.0) hides the scene completely, which lets the renderer skip the entire world render while the menu is open, so the frame costs only the menu overlay. Lower the alpha to keep the world visible behind a translucent fade (the world then keeps rendering); an alpha of 0 draws no backdrop at all.
- `centered`: A boolean. Horizontally center the menu and align it to the top of the window. When false, `x` is the column's center and `y` is the top of the first item. The menu is a screen overlay laid out against a fixed reference resolution and uniformly scaled to fill the window, so it keeps the same proportions at any window size. All pixel fields below are in that reference space, not raw window pixels. Defaults to `true`.
- `x`: A float. Column center x in reference-space pixels, used when `centered` is false. Defaults to `640.0`.
- `y`: A float. Top of the first item in reference-space pixels, used when `centered` is false. Defaults to `300.0`.
- `button_width`: A float. Width of each item's clickable region in pixels. Defaults to `360.0`.
- `button_height`: A float. Height of each item's clickable region in pixels. Defaults to `60.0`.
- `row_gap`: A float. Pixels between adjacent items. Defaults to `14.0`.
- `font`: A string. [Font](Font.md) for the item text. Empty uses the built-in font.
- `font_px`: A float. Pixel size of the item text when this menu emits its own built-in font (that is, when `font` is empty). Ignored when `font` names a [Font](Font.md), which carries its own size. In reference-space pixels. Defaults to `48.0`.
- `text_color`: An array of 3 floats. Linear-space RGB color of the item text. Defaults to `[0.85, 0.85, 0.85]`.
- `text_scale`: A float. Scale applied to the item text. Defaults to `1.1`.
- `hover_color`: An array of 3 floats. RGB color of an item's text while it is hovered. Defaults to `[1.0, 0.85, 0.3]`.
- `hover_scale`: A float. Multiplier applied to an item's text size while it is hovered. The default `1.0` keeps the size and position fixed, so only the color changes on hover; a value like `1.1` grows the hovered text by 10%.
- `cursor`: A boolean. Draw an in-engine arrow cursor while the menu is shown (the system cursor is hidden). When false the system cursor is used. Defaults to `true`.
- `cursor_color`: An array of 4 floats. RGBA fill color of the arrow cursor. A contrasting outline is added automatically so it stays legible over any scene. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
- `cursor_size`: A float. Arrow cursor height in pixels (its width follows the arrow's shape). Defaults to `22.0`.
