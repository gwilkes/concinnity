<!-- Auto-generated - do not edit. -->

# File

References a source file by path.

For supported kinds the build compiles the file into the world (an `.obj`
becomes mesh data); other kinds are path-only references.

## Parameters

- `path`: A string. Path to the source file, relative to the project root.
- `kind`: A string (see [FileKind](FileKind.md)). File category. Inferred from the path extension when absent.
