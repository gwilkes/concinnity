pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    crate::environment_map::validate_environment_map_args(args)
        .map_err(|e| format!("Asset '{}': {}", name, e))
}
