//! Parallel symmetry detection.

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyReadonlyArray2;
use rayon::prelude::*;

/// Symmetry types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symmetry {
    Vertical,
    Horizontal,
    Diagonal,
    AntiDiagonal,
    Rotational90,
    Rotational180,
}

/// Fast parallel symmetry detection
///
/// Checks all symmetry types in parallel for maximum throughput.
#[pyfunction]
pub fn find_symmetries_fast<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<i8>,
) -> PyResult<&'py PyList> {
    let grid_array = grid.as_array();
    let (height, width) = grid_array.dim();

    let symmetries: Vec<&str> = vec![
        ("vertical", Symmetry::Vertical),
        ("horizontal", Symmetry::Horizontal),
        ("diagonal", Symmetry::Diagonal),
        ("antidiagonal", Symmetry::AntiDiagonal),
        ("rotational_90", Symmetry::Rotational90),
        ("rotational_180", Symmetry::Rotational180),
    ]
    .into_par_iter()
    .filter_map(|(name, sym_type)| {
        if check_symmetry(&grid_array, height, width, sym_type) {
            Some(name)
        } else {
            None
        }
    })
    .collect();

    let py_list = PyList::empty(py);
    for sym in symmetries {
        py_list.append(sym)?;
    }

    Ok(py_list)
}

#[inline]
fn check_symmetry(
    grid: &ndarray::ArrayView2<i8>,
    height: usize,
    width: usize,
    sym_type: Symmetry,
) -> bool {
    match sym_type {
        Symmetry::Vertical => {
            for r in 0..height {
                for c in 0..width / 2 {
                    if grid[[r, c]] != grid[[r, width - 1 - c]] {
                        return false;
                    }
                }
            }
            true
        }
        Symmetry::Horizontal => {
            for r in 0..height / 2 {
                for c in 0..width {
                    if grid[[r, c]] != grid[[height - 1 - r, c]] {
                        return false;
                    }
                }
            }
            true
        }
        Symmetry::Diagonal => {
            if height != width {
                return false;
            }
            for r in 0..height {
                for c in 0..r {
                    if grid[[r, c]] != grid[[c, r]] {
                        return false;
                    }
                }
            }
            true
        }
        Symmetry::AntiDiagonal => {
            if height != width {
                return false;
            }
            let n = height;
            for r in 0..n {
                for c in 0..n - r {
                    if grid[[r, c]] != grid[[n - 1 - c, n - 1 - r]] {
                        return false;
                    }
                }
            }
            true
        }
        Symmetry::Rotational90 => {
            if height != width {
                return false;
            }
            let n = height;
            for r in 0..n {
                for c in 0..n {
                    if grid[[r, c]] != grid[[n - 1 - c, r]] {
                        return false;
                    }
                }
            }
            true
        }
        Symmetry::Rotational180 => {
            for r in 0..height {
                for c in 0..width {
                    if grid[[r, c]] != grid[[height - 1 - r, width - 1 - c]] {
                        return false;
                    }
                }
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_vertical_symmetry() {
        let grid = Array2::from_shape_vec(
            (2, 3),
            vec![
                1, 2, 1,
                3, 4, 3,
            ],
        ).unwrap();

        assert!(check_symmetry(&grid.view(), 2, 3, Symmetry::Vertical));
    }

    #[test]
    fn test_horizontal_symmetry() {
        let grid = Array2::from_shape_vec(
            (2, 2),
            vec![
                1, 2,
                1, 2,
            ],
        ).unwrap();

        assert!(check_symmetry(&grid.view(), 2, 2, Symmetry::Horizontal));
    }
}
