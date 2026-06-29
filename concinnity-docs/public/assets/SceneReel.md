<!-- Auto-generated - do not edit. -->

# SceneReel

An ordered playlist of named [Scene](Scene.md)s.

The current scene's [Prop](Prop.md)s are shown, then it advances to the next
based on that scene's `duration_secs`. Timing and transition style are
declared on each [Scene](Scene.md) asset. Props not prefixed by any scene name
remain visible in all scenes.

```jsonl
{"name":"day",  "type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
{"name":"night","type":"Scene","args":{"duration_secs":5.0,"transition":"FadeBlack"}}
{"name":"day_sun",  "type":"Prop","args":{"model":"model_sun_disc","position":[0,80,-200]}}
{"name":"night_moon","type":"Prop","args":{"model":"model_moon_disc","position":[0,80,-200]}}
{"name":"reel","type":"SceneReel","args":{"looping":true,"scenes":["day","night"]}}
```

## Parameters

- `scenes`: An array of strings. Ordered list of [Scene](Scene.md) assets to play.
- `looping`: A boolean. When true, wraps back to the first scene after the last one ends.
- `start_index`: An integer. Index of the entry that is active at world start.
