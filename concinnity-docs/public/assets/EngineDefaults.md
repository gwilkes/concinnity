<!-- Auto-generated - do not edit. -->

# EngineDefaults

Opts a world out of individual engine-injected defaults.

A rendering world is completed at build time with standard assets it does
not declare itself: the [DebugHud](DebugHud.md) with its chip
[TextLabel](TextLabel.md)s and font, the [StatHud](StatHud.md) and its chips
when the world declares a [MainMenu](MainMenu.md), and, when an
[EnvironmentMap](EnvironmentMap.md) is present, the sky mesh that displays
it. Declaring the same asset yourself replaces the injected one; declaring
`EngineDefaults` with a flag set to `false` removes it entirely.

The build records every injected asset in `world-lock.json`; copy an entry
from there (or from `cn explain <name>`) into `world.jsonl` to override it.

```jsonl
{"name":"defaults","type":"EngineDefaults","args":{"debug_hud":false,"sky":false}}
```

## Parameters

- `hud`: A boolean. Inject the [StatHud](StatHud.md) with its chip labels and font when the world declares a [MainMenu](MainMenu.md) but no `StatHud`. Defaults to `true`.
- `debug_hud`: A boolean. Inject the [DebugHud](DebugHud.md) with its chip labels when the world declares no `DebugHud`. Defaults to `true`.
- `sky`: A boolean. Inject the sky mesh (a skybox [ProceduralMesh](ProceduralMesh.md), [Material](Material.md), and [Prop](Prop.md)) when the world has an [EnvironmentMap](EnvironmentMap.md) but no skybox mesh. Disable to use an `EnvironmentMap` for image-based lighting only, with the background left to `clear_color` or your own geometry. Defaults to `true`.
