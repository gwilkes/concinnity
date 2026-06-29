<!-- Auto-generated - do not edit. -->

# Window

Declares the application window.

```json
{
  "name": "main_window",
  "type": "Window",
  "args": {
    "title": "Game",
    "width": 1280,
    "height": 720,
    "mode": "windowed",
    "resizable": true
  }
}
```

## Parameters

- `title`: A string. Window title shown in the title bar. Defaults to `"Concinnity"`.
- `width`: An integer. Initial window width in pixels. Defaults to `1024`.
- `height`: An integer. Initial window height in pixels. Defaults to `768`.
- `mode`: A string (one of `windowed`, `fullscreen`, or `borderless`). How the window is displayed.
- `resizable`: A boolean. Whether the user can resize the window. Defaults to `false`.
