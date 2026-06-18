pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let has_model = !args
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty();
    let has_mesh = !args
        .get("mesh")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty();
    let has_prefab = !args
        .get("prefab")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .is_empty();
    if has_prefab && (has_model || has_mesh) {
        return Err(format!(
            "Asset '{}': Prop with 'prefab' cannot also set 'model' or 'mesh'",
            name
        ));
    }
    if !has_model && !has_mesh && !has_prefab {
        return Err(format!(
            "Asset '{}': Prop requires either a 'prefab', 'model', or 'mesh' field",
            name
        ));
    }
    Ok(())
}
