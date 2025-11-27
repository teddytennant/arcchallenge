import numpy as np
import pytest
from dsl.interpreter import (
    Interpreter, Environment, Closure,
    InterpreterError, UndefinedVariableError, TypeError_, RuntimeError_,
    interpret, run_on_grid, run_test_program
)
from dsl.ast import Var, Const, Prim, Apply, Lambda, Let, IfThenElse, Program
from dsl.primitives import ADD, SUB, ROTATE_90
from arc_core.grid import load_grid


def test_environment_creation():
    """Test environment creation."""
    env = Environment()
    assert env.bindings == {}
    assert env.parent is None


def test_environment_define():
    """Test defining variable in environment."""
    env = Environment()
    env.define("x", 42)

    assert env.bindings["x"] == 42


def test_environment_lookup():
    """Test looking up variable in environment."""
    env = Environment()
    env.define("x", 42)

    assert env.lookup("x") == 42


def test_environment_lookup_undefined():
    """Test looking up undefined variable."""
    env = Environment()

    with pytest.raises(UndefinedVariableError):
        env.lookup("x")


def test_environment_extend():
    """Test extending environment."""
    env = Environment()
    env.define("x", 42)

    new_env = env.extend("y", 100)

    assert new_env.lookup("y") == 100
    assert new_env.lookup("x") == 42  # Parent lookup
    assert "y" not in env.bindings  # Original unchanged


def test_environment_extend_many():
    """Test extending environment with multiple bindings."""
    env = Environment()
    new_env = env.extend_many(["x", "y"], [1, 2])

    assert new_env.lookup("x") == 1
    assert new_env.lookup("y") == 2


def test_environment_shadowing():
    """Test variable shadowing in environment."""
    env = Environment()
    env.define("x", 42)

    new_env = env.extend("x", 100)

    assert new_env.lookup("x") == 100
    assert env.lookup("x") == 42


def test_closure_creation():
    """Test closure creation."""
    env = Environment()
    body = Var("x")
    closure = Closure(["x"], body, env)

    assert closure.params == ["x"]
    assert closure.body == body
    assert closure.env == env


def test_interpreter_eval_const():
    """Test evaluating constant."""
    interp = Interpreter()
    result = interp.eval(Const(42))

    assert result == 42


def test_interpreter_eval_var():
    """Test evaluating variable."""
    interp = Interpreter()
    env = Environment()
    env.define("x", 100)

    result = interp.eval(Var("x"), env)
    assert result == 100


def test_interpreter_eval_var_undefined():
    """Test evaluating undefined variable."""
    interp = Interpreter()

    with pytest.raises(UndefinedVariableError):
        interp.eval(Var("x"))


def test_interpreter_eval_prim():
    """Test evaluating primitive."""
    interp = Interpreter()
    result = interp.eval(Prim("add"))

    assert result == ADD


def test_interpreter_eval_prim_undefined():
    """Test evaluating undefined primitive."""
    interp = Interpreter()

    with pytest.raises(UndefinedVariableError):
        interp.eval(Prim("nonexistent"))


def test_interpreter_eval_apply_primitive():
    """Test applying primitive function."""
    interp = Interpreter()
    expr = Apply(Prim("add"), [Const(2), Const(3)])

    result = interp.eval(expr)
    assert result == 5


def test_interpreter_eval_lambda():
    """Test evaluating lambda."""
    interp = Interpreter()
    expr = Lambda(["x"], Var("x"))

    result = interp.eval(expr)
    assert isinstance(result, Closure)
    assert result.params == ["x"]


def test_interpreter_eval_apply_lambda():
    """Test applying lambda function."""
    interp = Interpreter()
    # (λx. x + 1)(5)
    lam = Lambda(["x"], Apply(Prim("add"), [Var("x"), Const(1)]))
    expr = Apply(lam, [Const(5)])

    result = interp.eval(expr)
    assert result == 6


def test_interpreter_eval_let():
    """Test evaluating let binding."""
    interp = Interpreter()
    # let x = 5 in x + 1
    expr = Let("x", Const(5), Apply(Prim("add"), [Var("x"), Const(1)]))

    result = interp.eval(expr)
    assert result == 6


def test_interpreter_eval_if_then_else_true():
    """Test conditional with true condition."""
    interp = Interpreter()
    expr = IfThenElse(Const(True), Const(1), Const(2))

    result = interp.eval(expr)
    assert result == 1


def test_interpreter_eval_if_then_else_false():
    """Test conditional with false condition."""
    interp = Interpreter()
    expr = IfThenElse(Const(False), Const(1), Const(2))

    result = interp.eval(expr)
    assert result == 2


def test_interpreter_eval_if_non_boolean():
    """Test conditional with non-boolean condition."""
    interp = Interpreter()
    expr = IfThenElse(Const(42), Const(1), Const(2))

    with pytest.raises(TypeError_):
        interp.eval(expr)


def test_interpreter_nested_let():
    """Test nested let bindings."""
    interp = Interpreter()
    # let x = 5 in (let y = x + 1 in y + 1)
    inner = Let("y", Apply(Prim("add"), [Var("x"), Const(1)]),
                Apply(Prim("add"), [Var("y"), Const(1)]))
    expr = Let("x", Const(5), inner)

    result = interp.eval(expr)
    assert result == 7


def test_interpreter_closure_captures_env():
    """Test that closures capture their environment."""
    interp = Interpreter()
    # let x = 5 in (λy. x + y)
    lam = Lambda(["y"], Apply(Prim("add"), [Var("x"), Var("y")]))
    expr = Let("x", Const(5), lam)

    closure = interp.eval(expr)
    assert isinstance(closure, Closure)

    # Apply the closure
    result = interp.eval(Apply(Const(closure), [Const(3)]))
    assert result == 8


def test_interpreter_arity_mismatch():
    """Test arity mismatch error."""
    interp = Interpreter()
    lam = Lambda(["x", "y"], Var("x"))
    expr = Apply(lam, [Const(1)])  # Only 1 arg, expects 2

    with pytest.raises(TypeError_):
        interp.eval(expr)


def test_interpreter_max_steps():
    """Test maximum steps limit."""
    interp = Interpreter()
    interp.max_steps = 10

    # Create expression that will exceed steps
    expr = Const(1)
    for _ in range(20):
        expr = Apply(Prim("add"), [expr, Const(1)])

    with pytest.raises(RuntimeError_):
        interp.eval(expr)


def test_interpreter_run_program():
    """Test running a complete program."""
    interp = Interpreter()
    grid = load_grid([[1, 2], [3, 4]])

    # Program: rotate_90(input)
    program = Program(Apply(Prim("rotate_90"), [Var("input")]))

    result = interp.run_program(program, grid)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_interpreter_reset():
    """Test resetting interpreter state."""
    interp = Interpreter()
    interp.step_count = 100

    interp.reset()
    assert interp.step_count == 0


def test_interpret_function():
    """Test convenience interpret function."""
    expr = Apply(Prim("add"), [Const(2), Const(3)])
    result = interpret(expr)

    assert result == 5


def test_run_on_grid_function():
    """Test convenience run_on_grid function."""
    grid = load_grid([[1, 2], [3, 4]])
    program = Program(Apply(Prim("rotate_90"), [Var("input")]))

    result = run_on_grid(program, grid)
    assert isinstance(result, np.ndarray)


def test_test_program_function():
    """Test test_program function."""
    grid1 = load_grid([[1, 2]])
    grid2 = load_grid([[2, 1]])

    program = Program(Apply(Prim("reflect_h"), [Var("input")]))
    test_cases = [(grid1, grid2)]

    results = run_test_program(program, test_cases)

    assert results['total'] == 1
    assert results['passed'] == 1
    assert results['failed'] == 0
    assert results['errors'] == 0
    assert results['accuracy'] == 1.0


def test_test_program_with_failure():
    """Test test_program with failing test."""
    grid1 = load_grid([[1, 2]])
    grid2 = load_grid([[1, 2]])  # Wrong expected output

    program = Program(Apply(Prim("reflect_h"), [Var("input")]))
    test_cases = [(grid1, grid2)]

    results = run_test_program(program, test_cases)

    assert results['total'] == 1
    assert results['passed'] == 0
    assert results['failed'] == 1


def test_test_program_with_error():
    """Test test_program with runtime error."""
    grid1 = load_grid([[1, 2]])

    # Program that will cause an error
    program = Program(Apply(Prim("nonexistent"), [Var("input")]))
    test_cases = [(grid1, grid1)]

    results = run_test_program(program, test_cases)

    assert results['total'] == 1
    assert results['errors'] == 1


def test_interpreter_debug_mode():
    """Test interpreter in debug mode."""
    interp = Interpreter(debug=True)
    expr = Const(42)

    # Should not raise, but will print debug output
    result = interp.eval(expr)
    assert result == 42


def test_complex_program():
    """Test complex program with multiple operations."""
    interp = Interpreter()

    # Program: let x = rotate_90(input) in reflect_h(x)
    inner = Apply(Prim("reflect_h"), [Var("x")])
    expr = Let("x", Apply(Prim("rotate_90"), [Var("input")]), inner)
    program = Program(expr)

    grid = load_grid([[1, 2], [3, 4]])
    result = interp.run_program(program, grid)

    assert isinstance(result, np.ndarray)


def test_higher_order_function():
    """Test higher-order function (function that returns function)."""
    interp = Interpreter()

    # (λx. λy. x + y)(5)(3)
    inner_lam = Lambda(["y"], Apply(Prim("add"), [Var("x"), Var("y")]))
    outer_lam = Lambda(["x"], inner_lam)
    expr = Apply(Apply(outer_lam, [Const(5)]), [Const(3)])

    result = interp.eval(expr)
    assert result == 8


def test_recursive_let():
    """Test multiple let bindings."""
    interp = Interpreter()

    # let a = 1 in let b = 2 in let c = 3 in a + b + c
    expr = Let("a", Const(1),
           Let("b", Const(2),
           Let("c", Const(3),
           Apply(Prim("add"), [Var("a"),
           Apply(Prim("add"), [Var("b"), Var("c")])]))))

    result = interp.eval(expr)
    assert result == 6
