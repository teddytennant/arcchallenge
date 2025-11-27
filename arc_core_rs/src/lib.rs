//! High-performance grid operations for ARC challenge.
//!
//! This crate provides optimized implementations of core grid operations:
//! - Connected components with SIMD optimizations
//! - Fast grid comparisons
//! - Parallel symmetry detection
//! - Efficient spatial transformations

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray2, PyReadonlyArray2};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

mod grid;
mod components;
mod symmetry;

pub use grid::*;
pub use components::*;
pub use symmetry::*;

/// Python module initialization
#[pymodule]
fn arc_core_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(connected_components_fast, m)?)?;
    m.add_function(wrap_pyfunction!(grids_equal_fast, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance_fast, m)?)?;
    m.add_function(wrap_pyfunction!(find_symmetries_fast, m)?)?;
    Ok(())
}
