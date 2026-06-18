// src/gfx/transparent.rs
//
// Backend-agnostic helpers for the transparent (translucent) pass. The pass
// itself is encoded per backend; this module owns only the CPU-side ordering
// policy so it can be unit-tested without a GPU and reused as the Vulkan /
// DirectX transparent ports land.
//
// Transparent fragments use SRC_ALPHA / ONE_MINUS_SRC_ALPHA blending, which is
// order-dependent: a draw must be composited after everything behind it.
// `back_to_front_order` returns the draw indices sorted farthest-first by
// camera distance so the blend resolves correctly. This is a single fixed
// sorted draw list, not order-independent transparency.

// Return the indices `0..distances.len()` ordered farthest camera distance
// first (back-to-front). The sort is stable, so draws at equal distance keep
// their original (declaration) order. Non-finite distances (NaN) are treated
// as nearest so a degenerate value never pushes a draw behind valid geometry.
pub fn back_to_front_order(distances: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..distances.len()).collect();
    order.sort_by(|&a, &b| {
        // Farther (larger distance) sorts first. Map NaN to -inf so it lands
        // last (nearest), keeping a total order for `sort_by`.
        let da = if distances[a].is_finite() {
            distances[a]
        } else {
            f32::NEG_INFINITY
        };
        let db = if distances[b].is_finite() {
            distances[b]
        } else {
            f32::NEG_INFINITY
        };
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orders_farthest_first() {
        let d = [1.0, 5.0, 3.0];
        assert_eq!(back_to_front_order(&d), vec![1, 2, 0]);
    }

    #[test]
    fn empty_is_empty() {
        assert!(back_to_front_order(&[]).is_empty());
    }

    #[test]
    fn equal_distances_keep_declaration_order() {
        let d = [2.0, 2.0, 2.0];
        assert_eq!(back_to_front_order(&d), vec![0, 1, 2]);
    }

    #[test]
    fn nan_sorts_last() {
        let d = [4.0, f32::NAN, 2.0];
        // 4.0 (farthest) → index 0, then 2.0 → index 2, then NaN → index 1.
        assert_eq!(back_to_front_order(&d), vec![0, 2, 1]);
    }
}
