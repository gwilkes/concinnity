<!-- Auto-generated - do not edit. -->

# LightRig

A named grouping of lights.

Use `preset` to expand a built-in setup into named
[DirectionalLight](DirectionalLight.md)/[PointLight](PointLight.md) assets
(`<rig_name>_<light_name>`), or declare lights directly and list their names
in `lights`.

**Library presets:**

```jsonl
// From preset (expands into rig_sun + rig_fill):
{"name":"rig","type":"LightRig","args":{"preset":"rig_outdoor_sun_fill"}}

// Referencing pre-declared lights:
{"name":"sun",  "type":"DirectionalLight","args":{"direction":[-0.4,0.7,0.3],"color":[1.0,0.95,0.8],"intensity":1.2}}
{"name":"torch","type":"PointLight",      "args":{"position":[3.0,2.0,-5.0],"color":[1.0,0.7,0.3],"intensity":10.0,"range":6.0}}
{"name":"rig",  "type":"LightRig","args":{"lights":["sun","torch"]}}
```

## Parameters

- `preset`: A string. Name of a built-in or file-backed preset (e.g. "rig_outdoor_sun_fill"). When set, `lights` is ignored.
- `lights`: An array of strings. Names of existing [DirectionalLight](DirectionalLight.md) or [PointLight](PointLight.md) assets to include in this rig. Ignored when `preset` is set.
