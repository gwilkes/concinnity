<!-- Auto-generated - do not edit. -->

# KeyBinding

Maps a keyboard key to an action string.

When the bound key is pressed, the action fires once per press (like a
[HitRegion](HitRegion.md) click). Bindings only run while the cursor is free:
they're inactive in worlds that capture the cursor for camera control.

The action vocabulary is the same as [HitRegion](HitRegion.md)'s:
- `"scene:<name>"`:       jump to the named [Scene](Scene.md)
- `"quit"`:               stop the application
- `"view:show:<name>"`:   show the named [View](View.md) overlay
- `"view:hide"`:          hide the active [View](View.md)
- `"view:toggle:<name>"`: toggle the named [View](View.md)

Recognised key names are case-sensitive; currently `"Escape"` is supported.

```jsonl
{"name":"esc_binding","type":"KeyBinding","args":{"key":"Escape","action":"view:toggle:pause_menu"}}
```

## Parameters

- `key`: A string. The key name to bind (e.g. `"Escape"`).
- `action`: A string. The action to fire when the key is pressed.
