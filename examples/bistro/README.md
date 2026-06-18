# Amazon Lumberyard Bistro, Open Research Content Archive (ORCA)

- `author`: Amazon Lumberyard
- `year`: 2017
- `month`: July
- `url`: [http://developer.nvidia.com/orca/amazon-lumberyard-bistro](http://developer.nvidia.com/orca/amazon-lumberyard-bistro)

The Amazon Lumberyard Bistro exterior (~2.8M triangles) rendered through the
engine's full PBR pipeline: HDR-IBL, cascaded shadow maps, ray-traced
reflections, SSAO, SSGI, bloom, and TAA on a bindless GPU-driven path. The scene
is one `SceneImport` of `BistroExterior.fbx` plus a sun, sky, post-process
config, and a stat HUD.

## Running

```
cargo run -p bistro --release
```

The first run downloads the Bistro asset pack (~833 MB) into
`examples/bistro/assets/` and unpacks it (~1.6 GB); later runs detect the assets and skip
straight to rendering. The download is a runtime preflight, so `cargo build`
never touches the network.

Use `--release`: the scene is GPU-heavy and a debug build runs well below full
frame rate. Free-fly the camera with WASD + mouse.

### Asset fetch overrides

| Environment variable                 | Effect                                             |
| ------------------------------------ | -------------------------------------------------- |
| `BISTRO_ARCHIVE=/path/to/bistro.zip` | Use an already-downloaded ZIP instead of fetching. |
| `BISTRO_URL=...`                     | Override the download URL.                         |

The fetcher only handles ZIP archives. If the download URL ever serves a
different format, download the pack manually and point `BISTRO_ARCHIVE` at the
resulting ZIP.
