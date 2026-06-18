pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    crate::texture::validate_texture_generator(args).map_err(|e| format!("Asset '{}': {}", name, e))
}
