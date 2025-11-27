//! Compact AST representation for fast synthesis.

use rustc_hash::FxHashSet;
use std::sync::Arc;

/// Compact program representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Primitive operation (index into primitive table)
    Prim(u16),

    /// Application of function to arguments
    Apply {
        func: Arc<Expr>,
        args: Arc<[Expr]>,
    },

    /// Variable reference
    Var(u8),

    /// Constant integer
    Const(i32),
}

impl Expr {
    /// Compute size (number of nodes)
    pub fn size(&self) -> usize {
        match self {
            Expr::Prim(_) | Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Apply { func, args } => {
                1 + func.size() + args.iter().map(|a| a.size()).sum::<usize>()
            }
        }
    }

    /// Compute depth
    pub fn depth(&self) -> usize {
        match self {
            Expr::Prim(_) | Expr::Var(_) | Expr::Const(_) => 1,
            Expr::Apply { func, args } => {
                1 + func.depth().max(
                    args.iter().map(|a| a.depth()).max().unwrap_or(0)
                )
            }
        }
    }
}

/// Program database for deduplication
pub struct ProgramDb {
    seen: FxHashSet<Arc<Expr>>,
}

impl ProgramDb {
    pub fn new() -> Self {
        Self {
            seen: FxHashSet::default(),
        }
    }

    /// Insert program, returning true if new
    pub fn insert(&mut self, expr: Arc<Expr>) -> bool {
        self.seen.insert(expr)
    }

    /// Check if program was seen
    pub fn contains(&self, expr: &Expr) -> bool {
        self.seen.contains(expr)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_size() {
        let expr = Expr::Prim(0);
        assert_eq!(expr.size(), 1);

        let app = Expr::Apply {
            func: Arc::new(Expr::Prim(0)),
            args: Arc::new([Expr::Prim(1)]),
        };
        assert_eq!(app.size(), 3);
    }

    #[test]
    fn test_program_db() {
        let mut db = ProgramDb::new();
        let expr = Arc::new(Expr::Prim(0));

        assert!(db.insert(expr.clone()));
        assert!(!db.insert(expr.clone()));
        assert_eq!(db.len(), 1);
    }
}
