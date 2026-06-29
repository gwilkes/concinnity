# Concinnity

Application software for [`concinnity.gg`](https://concinnity.gg).

## Overview

Concinnity is an asset-driven 3D rendering engine. Runtime behavior is defined
declaratively through a static set of assets compiled into binary blobs. There
are no scripting languages (yet); behavior emerges entirely from asset
composition.

## Installation

This project is in **early development** and no releases are available yet.
For now, you'll need to [build it manually](concinnity-docs/public/development/building.md).

After a successful build, you may optionally install the `concinnity` executable:

```sh
cargo install --path concinnity-editor
concinnity help
```

## Getting Started

Since this project is in early development and CLI commands are bound to change,
there currently isn't any CLI documentation. Run `concinnity help` for a list of
supported commands.

Check out the [asset documentation](concinnity-docs/public/assets/index.md) for all
supported asset types and fields.

This project currently has one example, which uses the
[Amazon Lumberyard Bistro](examples/bistro/README.md) assets and can be run
with `cargo`:

```sh
cargo run -p bistro --release
```

## License

See [LICENSE](LICENSE).
