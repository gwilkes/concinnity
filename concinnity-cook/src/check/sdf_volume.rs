pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    concinnity_core::assets::sdf_volume::check(args).map_err(|e| format!("Asset '{}': {}", name, e))
}
