"""
Interpreter for ARC DSL programs.

Provides execution environment for DSL expressions including:
- Environment management
- Primitive operations execution
- Lambda evaluation with closures
- Error handling and debugging
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
import traceback

from .ast import *
from .primitives import PRIMITIVES_BY_NAME, Primitive


class InterpreterError(Exception):
    """Base exception for interpreter errors."""
    pass


class UndefinedVariableError(InterpreterError):
    """Variable not found in environment."""
    pass


class TypeError_(InterpreterError):
    """Type error during execution."""
    pass


class RuntimeError_(InterpreterError):
    """Runtime error during execution."""
    pass


# ============================================================================
# Environment
# ============================================================================

@dataclass
class Environment:
    """Execution environment mapping variables to values."""

    bindings: Dict[str, Any] = field(default_factory=dict)
    parent: Optional['Environment'] = None

    def lookup(self, var: str) -> Any:
        """Lookup variable in environment."""
        if var in self.bindings:
            return self.bindings[var]
        if self.parent is not None:
            return self.parent.lookup(var)
        raise UndefinedVariableError(f"Undefined variable: {var}")

    def extend(self, var: str, value: Any) -> 'Environment':
        """Create new environment extending this one."""
        new_bindings = {var: value}
        return Environment(new_bindings, parent=self)

    def extend_many(self, vars: List[str], values: List[Any]) -> 'Environment':
        """Create new environment with multiple bindings."""
        new_bindings = dict(zip(vars, values))
        return Environment(new_bindings, parent=self)

    def define(self, var: str, value: Any):
        """Define variable in current environment."""
        self.bindings[var] = value

    def __repr__(self) -> str:
        return f"Env({self.bindings})"


# ============================================================================
# Closure (for lambda values)
# ============================================================================

@dataclass
class Closure:
    """Closure: function value with captured environment."""

    params: List[str]
    body: Expr
    env: Environment

    def __repr__(self) -> str:
        params_str = ", ".join(self.params)
        return f"<closure λ{params_str}. {self.body}>"


# ============================================================================
# Interpreter
# ============================================================================

class Interpreter:
    """Interpreter for DSL expressions."""

    def __init__(self, primitives: Optional[Dict[str, Primitive]] = None,
                 debug: bool = False):
        """Initialize interpreter.

        Args:
            primitives: Dictionary of primitive operations
            debug: Enable debug output
        """
        self.primitives = primitives or PRIMITIVES_BY_NAME
        self.debug = debug
        self.step_count = 0
        self.max_steps = 100000  # Prevent infinite loops

    def eval(self, expr: Expr, env: Optional[Environment] = None) -> Any:
        """Evaluate an expression in an environment.

        Args:
            expr: Expression to evaluate
            env: Execution environment (creates empty if None)

        Returns:
            Result value

        Raises:
            InterpreterError: If evaluation fails
        """
        if env is None:
            env = Environment()

        self.step_count += 1
        if self.step_count > self.max_steps:
            raise RuntimeError_(f"Exceeded maximum steps ({self.max_steps})")

        if self.debug:
            print(f"[{self.step_count}] Evaluating: {expr}")

        try:
            return self._eval(expr, env)
        except InterpreterError:
            raise
        except Exception as e:
            raise RuntimeError_(f"Runtime error: {e}\n{traceback.format_exc()}")

    def _eval(self, expr: Expr, env: Environment) -> Any:
        """Internal evaluation method."""

        # Variable
        if isinstance(expr, Var):
            return env.lookup(expr.name)

        # Constant
        elif isinstance(expr, Const):
            return expr.value

        # Primitive
        elif isinstance(expr, Prim):
            if expr.name not in self.primitives:
                raise UndefinedVariableError(f"Unknown primitive: {expr.name}")
            return self.primitives[expr.name]

        # Function application
        elif isinstance(expr, Apply):
            func = self._eval(expr.func, env)
            args = [self._eval(arg, env) for arg in expr.args]

            # Apply primitive
            if isinstance(func, Primitive):
                try:
                    return func.impl(*args)
                except Exception as e:
                    raise RuntimeError_(
                        f"Error applying primitive {func.name}: {e}\n"
                        f"Arguments: {args}"
                    )

            # Apply closure
            elif isinstance(func, Closure):
                if len(args) != len(func.params):
                    raise TypeError_(
                        f"Arity mismatch: function expects {len(func.params)} "
                        f"arguments, got {len(args)}"
                    )
                new_env = func.env.extend_many(func.params, args)
                return self._eval(func.body, new_env)

            # Apply callable (Python function)
            elif callable(func):
                try:
                    return func(*args)
                except Exception as e:
                    raise RuntimeError_(f"Error applying function: {e}")

            else:
                raise TypeError_(f"Cannot apply non-function: {type(func)}")

        # Lambda
        elif isinstance(expr, Lambda):
            return Closure(expr.params, expr.body, env)

        # Let binding
        elif isinstance(expr, Let):
            value = self._eval(expr.value, env)
            new_env = env.extend(expr.var, value)
            return self._eval(expr.body, new_env)

        # Conditional
        elif isinstance(expr, IfThenElse):
            cond = self._eval(expr.cond, env)
            if not isinstance(cond, bool):
                raise TypeError_(f"Condition must be boolean, got {type(cond)}")
            if cond:
                return self._eval(expr.then_branch, env)
            else:
                return self._eval(expr.else_branch, env)

        else:
            raise RuntimeError_(f"Unknown expression type: {type(expr)}")

    def run_program(self, program: Program, input_grid: Any) -> Any:
        """Run a complete program on an input grid.

        Args:
            program: Program to execute
            input_grid: Input grid

        Returns:
            Output grid
        """
        env = Environment()
        env.define("input", input_grid)
        self.step_count = 0

        return self.eval(program.expr, env)

    def reset(self):
        """Reset interpreter state."""
        self.step_count = 0


# ============================================================================
# Convenience Functions
# ============================================================================

def interpret(expr: Expr, env: Optional[Environment] = None,
              debug: bool = False) -> Any:
    """Interpret an expression.

    Args:
        expr: Expression to interpret
        env: Environment (optional)
        debug: Enable debug output

    Returns:
        Result value
    """
    interpreter = Interpreter(debug=debug)
    return interpreter.eval(expr, env)


def run_on_grid(program: Program, grid: Any, debug: bool = False) -> Any:
    """Run a program on a grid.

    Args:
        program: Program to run
        grid: Input grid
        debug: Enable debug output

    Returns:
        Output grid
    """
    interpreter = Interpreter(debug=debug)
    return interpreter.run_program(program, grid)


def test_program(program: Program, test_cases: List[tuple],
                 debug: bool = False) -> Dict[str, Any]:
    """Test a program on multiple input/output pairs.

    Args:
        program: Program to test
        test_cases: List of (input_grid, expected_output) tuples
        debug: Enable debug output

    Returns:
        Dictionary with test results
    """
    interpreter = Interpreter(debug=debug)
    results = {
        'passed': 0,
        'failed': 0,
        'errors': 0,
        'details': []
    }

    for i, (input_grid, expected_output) in enumerate(test_cases):
        try:
            interpreter.reset()
            output = interpreter.run_program(program, input_grid)

            # Check if output matches expected
            import numpy as np
            if isinstance(output, np.ndarray) and isinstance(expected_output, np.ndarray):
                matches = np.array_equal(output, expected_output)
            else:
                matches = output == expected_output

            if matches:
                results['passed'] += 1
                results['details'].append({
                    'test': i,
                    'status': 'passed'
                })
            else:
                results['failed'] += 1
                results['details'].append({
                    'test': i,
                    'status': 'failed',
                    'expected': expected_output,
                    'got': output
                })

        except Exception as e:
            results['errors'] += 1
            results['details'].append({
                'test': i,
                'status': 'error',
                'error': str(e)
            })

    results['total'] = len(test_cases)
    results['accuracy'] = results['passed'] / results['total'] if results['total'] > 0 else 0

    return results
