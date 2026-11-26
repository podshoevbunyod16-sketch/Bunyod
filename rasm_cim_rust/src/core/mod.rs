//! Core shared primitives. Currently thin: error type lives in lib.rs,
//! similarity is a one-liner kept per-module (memory, graph). Kept as the
//! place for future shared kernels (SIMD cosine, buffer pools) — documented
//! as intentionally minimal, not a stub of removed functionality.

/// Cosine-like similarity for bipolar HV vectors (dot / d), matches
/// memory_system.py `sims = (K @ qhv) / D`.
#[inline]
pub fn hv_similarity(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_is_one() {
        let v = vec![1.0, -1.0, 1.0, 1.0];
        assert!((hv_similarity(&v, &v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn opposite_is_minus_one() {
        let v = vec![1.0, -1.0];
        let w = vec![-1.0, 1.0];
        assert!((hv_similarity(&v, &w) + 1.0).abs() < 1e-12);
    }
}
