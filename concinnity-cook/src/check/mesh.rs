pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    // A `.glb`-sourced Mesh has no inline vertices yet: the build's desugar
    // pass fills them in later. Skip the compile-time check; full validation
    // happens when desugar reads the GLB.
    let source = args.get("source").and_then(|v| v.as_str()).unwrap_or("");
    if !source.is_empty() {
        return Ok(());
    }
    // Full compile catches unknown generators and structural errors.
    // Pure geometry math, no I/O: cost is negligible for per-asset validation.
    crate::mesh_compile::compile_mesh_payload(args)
        .map(|_| ())
        .map_err(|e| format!("Asset '{}' mesh compile error: {}", name, e))
}
