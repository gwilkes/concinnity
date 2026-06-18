pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    crate::cubemap::validate_cubemap_args(args).map_err(|e| format!("Asset '{}': {}", name, e))
}
