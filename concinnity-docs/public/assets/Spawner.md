<!-- Auto-generated - do not edit. -->

# Spawner

Periodically instantiates copies of an existing placement at this entity's
position.

A spawner clones `template` (the name of another placement in the world)
every `interval` seconds, giving each copy a `lifetime` after which it is
automatically removed. Pairing a short lifetime with a short interval keeps a
bounded population churning (an enemy wave, a particle of debris, a fountain
of props) and is what exercises GPU draw-slot recycling: each expiry frees a
slot the next spawn reuses.

The spawner's own `Transform` (its position) is where copies appear, so place
the spawner where you want the stream to originate.

```jsonl
{"name":"crate","type":"Prop","args":{"mesh":"box_mesh","material":"mat_brick","position":[0.0,1.0,-6.0]}}
{"name":"fountain","type":"Prop","args":{"mesh":"box_mesh","position":[0.0,1.0,-3.0]}}
{"name":"fountain_spawner","type":"Spawner","args":{"template":"crate","interval":0.5,"lifetime":2.0}}
```

## Parameters

- `template`: A string. Name of the placement to copy on each spawn.
- `interval`: A float. Seconds between spawns. Defaults to `1.0`.
- `lifetime`: A float. Seconds each spawned copy lives before auto-removal; 0 keeps it forever. Defaults to `0.0`.
