use crate::assets::FileKind;

// Compile a File asset's source into a binary payload.
//
// Only mesh-kind files (e.g. OBJ) produce a blob. Call `kind.is_mesh()` before
// invoking this to skip non-mesh kinds at the call site.
pub(crate) fn compile_file_payload(path: &str, kind: &FileKind) -> Result<Vec<u8>, String> {
    match kind {
        FileKind::Obj => {
            let source = std::fs::read_to_string(path)
                .map_err(|e| format!("failed to read '{}': {}", path, e))?;
            let (vertices, indices) = super::wavefront::parse_obj(&source)?;
            Ok(crate::geometry::compile_mesh_from_vertex_data(
                &vertices, &indices,
            ))
        }
        other => Err(format!(
            "File kind '{:?}' does not produce a compiled payload",
            other
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_mesh_kind_returns_error() {
        assert!(compile_file_payload("irrelevant.png", &FileKind::Png).is_err());
    }
}
