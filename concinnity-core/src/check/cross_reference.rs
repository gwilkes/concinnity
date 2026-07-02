// src/check/cross_reference.rs
//
// Cross-asset name reference declarations. Every referencing asset (Prop ->
// Mesh, Material -> Texture, etc.) declares its own references by implementing
// `CrossReferenced` in its asset file. The build crate's resolver consumes
// these declarations: it resolves each `RefKind` to the matching set of asset
// names and detects Prop parent cycles. Adding a new referencing asset means
// writing one impl plus one dispatch arm in the build crate.

// The category of asset a name reference must resolve to. Reference kinds are
// deliberately not 1:1 with asset types: `MeshSource` accepts several types
// and `CameraShot` resolves to `Camera3D` names.
#[derive(Debug, Clone, Copy)]
pub enum RefKind {
    // Mesh, ProceduralMesh, VoxelChunk, or a mesh-kind File.
    MeshSource,
    Texture,
    Material,
    Model,
    Prop,
    Scene,
    // A camera shot target, resolves to declared `Camera3D` names.
    CameraShot,
    BlockType,
    TextLabel,
}

// One item produced by a referencing asset's `cross_refs`.
pub enum CrossRef {
    // `target` must resolve to an asset in `kind`'s name-set; if it does not,
    // `error` is collected verbatim.
    Resolve {
        kind: RefKind,
        target: String,
        error: String,
    },
    // A problem the asset detected on its own: a missing required field, a
    // malformed array entry, an empty list. Collected verbatim.
    Issue(String),
}

// Implemented by every asset type that references other assets by name.
// `cross_refs` extracts those references (and any structural problems) from
// the asset's args; the resolver resolves each `Resolve` against the world.
pub trait CrossReferenced {
    fn cross_refs(name: &str, args: &serde_json::Value) -> Vec<CrossRef>;
}

// Shared extractor for HUD-style assets whose args hold optional TextLabel
// references: one `Resolve` per non-empty label field.
pub fn label_refs(
    asset_type: &str,
    name: &str,
    args: &serde_json::Value,
    fields: &[&str],
) -> Vec<CrossRef> {
    let mut refs = Vec::new();
    for field in fields {
        let target = args.get(field).and_then(|v| v.as_str()).unwrap_or("");
        if target.is_empty() {
            continue;
        }
        refs.push(CrossRef::Resolve {
            kind: RefKind::TextLabel,
            target: target.to_string(),
            error: format!(
                "{} '{}': {} '{}' not found, add a TextLabel asset with that name",
                asset_type, name, field, target
            ),
        });
    }
    refs
}
