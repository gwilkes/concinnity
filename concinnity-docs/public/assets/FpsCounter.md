<!-- Auto-generated - do not edit. -->

# FpsCounter

Requests a frames-per-second counter; optionally writes it to a
[TextLabel](TextLabel.md).

Declaring an `FpsCounter` updates the named [TextLabel](TextLabel.md) with the
current rate once per second. Omit `label` to suppress on-screen display.

To display an FPS overlay, declare a [Font](Font.md), a
[TextLabel](TextLabel.md), and an `FpsCounter` that references the label:

```jsonl
{"$include":"assets/fps_font.json"}
{"$include":"assets/fps_text.json"}
{"type":"FpsCounter","name":"fps","args":{"label":"fps_text"}}
```

## Parameters

- `label`: A string. A [TextLabel](TextLabel.md) to update with the current FPS each second. Leave unset to suppress on-screen display.
