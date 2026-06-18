pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    concinnity_core::assets::shader_stage::check(args)
        .map_err(|e| format!("Asset '{}': {}", name, e))
}
