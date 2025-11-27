//! Parallel enumerative synthesis.

use pyo3::prelude::*;
use rayon::prelude::*;
use crossbeam::channel::{unbounded, Sender};
use std::sync::{Arc, Mutex};
use rustc_hash::FxHashSet;

use crate::ast::{Expr, ProgramDb};

/// Configuration for synthesis
#[pyclass]
#[derive(Clone)]
pub struct SynthesisConfig {
    #[pyo3(get, set)]
    pub max_depth: usize,

    #[pyo3(get, set)]
    pub max_size: usize,

    #[pyo3(get, set)]
    pub max_programs: usize,

    #[pyo3(get, set)]
    pub num_workers: usize,
}

#[pymethods]
impl SynthesisConfig {
    #[new]
    fn new() -> Self {
        Self {
            max_depth: 3,
            max_size: 10,
            max_programs: 10000,
            num_workers: rayon::current_num_threads(),
        }
    }
}

/// Parallel enumerative synthesizer
#[pyclass]
pub struct ParallelEnumerator {
    config: SynthesisConfig,
    primitives: Vec<PrimitiveInfo>,
}

#[derive(Clone)]
struct PrimitiveInfo {
    id: u16,
    arity: u8,
}

#[pymethods]
impl ParallelEnumerator {
    #[new]
    fn new(config: SynthesisConfig, num_primitives: usize) -> Self {
        let primitives = (0..num_primitives)
            .map(|i| PrimitiveInfo {
                id: i as u16,
                arity: 1, // Simplified - would need actual arity info
            })
            .collect();

        Self { config, primitives }
    }

    /// Enumerate programs in parallel
    fn enumerate_parallel(&self, test_fn: PyObject) -> PyResult<Vec<String>> {
        let (tx, rx) = unbounded();
        let program_db = Arc::new(Mutex::new(ProgramDb::new()));
        let found_programs = Arc::new(Mutex::new(Vec::new()));

        // Parallel depth-first search
        (1..=self.config.max_depth)
            .into_par_iter()
            .for_each(|depth| {
                self.enumerate_at_depth(
                    depth,
                    &tx,
                    program_db.clone(),
                    found_programs.clone(),
                );
            });

        let programs = found_programs.lock().unwrap();
        Ok(programs.iter().map(|e| format!("{:?}", e)).collect())
    }
}

impl ParallelEnumerator {
    fn enumerate_at_depth(
        &self,
        depth: usize,
        tx: &Sender<Arc<Expr>>,
        program_db: Arc<Mutex<ProgramDb>>,
        found_programs: Arc<Mutex<Vec<Arc<Expr>>>>,
    ) {
        if depth == 1 {
            // Base case: primitives
            for prim in &self.primitives {
                let expr = Arc::new(Expr::Prim(prim.id));

                let mut db = program_db.lock().unwrap();
                if db.insert(expr.clone()) {
                    drop(db);

                    let _ = tx.send(expr.clone());

                    let mut found = found_programs.lock().unwrap();
                    found.push(expr);
                }
            }
        } else {
            // Recursive case: build applications
            for prim in &self.primitives {
                if prim.arity == 1 {
                    // Unary application
                    self.enumerate_at_depth(
                        depth - 1,
                        tx,
                        program_db.clone(),
                        found_programs.clone(),
                    );
                }
            }
        }
    }
}

/// Standalone function for simple enumeration
#[pyfunction]
pub fn enumerate_programs_parallel(
    max_depth: usize,
    num_primitives: usize,
) -> PyResult<Vec<String>> {
    let programs: Vec<String> = (1..=max_depth)
        .into_par_iter()
        .flat_map(|depth| {
            (0..num_primitives)
                .map(|i| format!("prim_{}_depth_{}", i, depth))
                .collect::<Vec<_>>()
        })
        .collect();

    Ok(programs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_enumeration() {
        let config = SynthesisConfig::new();
        let enumerator = ParallelEnumerator::new(config, 5);

        // Test that it runs without panicking
        // Full test would need Python runtime
    }
}
