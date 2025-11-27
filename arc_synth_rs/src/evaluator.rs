//! Fast program evaluation with memoization.

use rustc_hash::FxHashMap;
use std::sync::{Arc, RwLock};

use crate::ast::Expr;

/// Value that can be returned from evaluation
#[derive(Debug, Clone)]
pub enum Value {
    Grid(Vec<Vec<i8>>),
    Int(i32),
    Bool(bool),
}

/// Evaluation cache for memoization
pub struct EvalCache {
    cache: Arc<RwLock<FxHashMap<Arc<Expr>, Value>>>,
}

impl EvalCache {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(FxHashMap::default())),
        }
    }

    /// Evaluate expression with memoization
    pub fn eval(&self, expr: &Arc<Expr>) -> Option<Value> {
        // Check cache
        {
            let cache = self.cache.read().unwrap();
            if let Some(value) = cache.get(expr) {
                return Some(value.clone());
            }
        }

        // Evaluate
        let value = self.eval_impl(expr)?;

        // Store in cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(expr.clone(), value.clone());
        }

        Some(value)
    }

    fn eval_impl(&self, expr: &Arc<Expr>) -> Option<Value> {
        match expr.as_ref() {
            Expr::Const(n) => Some(Value::Int(*n)),
            Expr::Prim(_) => None, // Would need primitive implementations
            Expr::Apply { func, args } => {
                // Evaluate function and arguments
                let _func_val = self.eval(func)?;
                let _arg_vals: Vec<_> = args.iter()
                    .map(|a| self.eval(&Arc::new(a.clone())))
                    .collect::<Option<_>>()?;

                // Apply function (would need actual implementation)
                None
            }
            Expr::Var(_) => None, // Would need environment
        }
    }

    pub fn cache_size(&self) -> usize {
        self.cache.read().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_cache() {
        let cache = EvalCache::new();
        let expr = Arc::new(Expr::Const(42));

        let val = cache.eval(&expr);
        assert!(matches!(val, Some(Value::Int(42))));

        assert_eq!(cache.cache_size(), 1);
    }
}
