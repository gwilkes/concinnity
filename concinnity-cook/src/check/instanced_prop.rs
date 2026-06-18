// Structural validation for InstancedProp args. Cross-asset mesh/material
// lookups are handled by build/pipeline.rs::validate_cross_references; this
// check enforces only the things we can see from the asset's own args.

// Soft cap on instances per cluster. The expansion path produces one
// DrawObject per instance, so a runaway cluster can blow out the draw list
// budget without warning.
pub(crate) const MAX_INSTANCES_PER_CLUSTER: usize = 16_384;

pub fn check(name: &str, args: &serde_json::Value) -> Result<(), String> {
    let instances = args
        .get("instances")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            format!(
                "Asset '{}': InstancedProp `instances` must be an array of transforms",
                name
            )
        })?;
    if instances.len() > MAX_INSTANCES_PER_CLUSTER {
        return Err(format!(
            "Asset '{}': InstancedProp has {} instances; current cap is {}. Split into multiple clusters.",
            name,
            instances.len(),
            MAX_INSTANCES_PER_CLUSTER
        ));
    }
    for (i, entry) in instances.iter().enumerate() {
        if !entry.is_object() {
            return Err(format!(
                "Asset '{}': InstancedProp instances[{}] must be an object",
                name, i
            ));
        }
        if let Some(p) = entry.get("position") {
            check_f32x3(p, &format!("Asset '{}': instances[{}].position", name, i))?;
        }
        if let Some(r) = entry.get("rotation_deg") {
            check_f32x3(
                r,
                &format!("Asset '{}': instances[{}].rotation_deg", name, i),
            )?;
        }
        if let Some(s) = entry.get("scale") {
            check_f32x3(s, &format!("Asset '{}': instances[{}].scale", name, i))?;
        }
    }
    Ok(())
}

fn check_f32x3(v: &serde_json::Value, label: &str) -> Result<(), String> {
    let arr = v
        .as_array()
        .ok_or_else(|| format!("{label} must be an array of 3 numbers"))?;
    if arr.len() < 3 {
        return Err(format!("{label} must have 3 elements, got {}", arr.len()));
    }
    for (i, e) in arr.iter().take(3).enumerate() {
        if e.as_f64().is_none() {
            return Err(format!("{label}[{i}] must be a number"));
        }
    }
    Ok(())
}
