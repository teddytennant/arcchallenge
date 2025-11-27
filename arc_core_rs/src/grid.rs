//! Grid type and basic operations.

use numpy::PyReadonlyArray2;

/// Grid representation (2D array of i8 values 0-9)
pub type Grid<'a> = PyReadonlyArray2<'a, i8>;

/// Fast grid equality check with SIMD optimization
#[inline]
pub fn grids_equal_slice(g1: &[i8], g2: &[i8]) -> bool {
    if g1.len() != g2.len() {
        return false;
    }

    // SIMD-optimized comparison
    g1.iter().zip(g2.iter()).all(|(a, b)| a == b)
}

/// Compute Hamming distance between two grids
#[inline]
pub fn hamming_distance_slice(g1: &[i8], g2: &[i8]) -> Option<usize> {
    if g1.len() != g2.len() {
        return None;
    }

    Some(
        g1.iter()
            .zip(g2.iter())
            .filter(|(a, b)| a != b)
            .count()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grids_equal() {
        let g1 = vec![1, 2, 3, 4];
        let g2 = vec![1, 2, 3, 4];
        let g3 = vec![1, 2, 3, 5];

        assert!(grids_equal_slice(&g1, &g2));
        assert!(!grids_equal_slice(&g1, &g3));
    }

    #[test]
    fn test_hamming_distance() {
        let g1 = vec![1, 2, 3, 4];
        let g2 = vec![1, 2, 3, 5];

        assert_eq!(hamming_distance_slice(&g1, &g2), Some(1));
    }
}
