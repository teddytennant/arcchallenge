import pytest
import json
from dsl.ast import (
    Var, Const, Prim, Apply, Lambda, Let, IfThenElse, Program,
    PrimitiveType, FunctionType, ListType,
    GridType, IntType, BoolType, ObjectType,
    size, depth, get_primitives, get_variables, substitute
)


def test_primitive_type():
    """Test primitive type creation and equality."""
    grid_type = PrimitiveType("Grid")
    int_type = PrimitiveType("Int")

    assert str(grid_type) == "Grid"
    assert grid_type == PrimitiveType("Grid")
    assert grid_type != int_type


def test_list_type():
    """Test list type."""
    list_int = ListType(IntType)

    assert str(list_int) == "List[Int]"
    assert list_int == ListType(IntType)


def test_function_type():
    """Test function type."""
    func_type = FunctionType((IntType, IntType), BoolType)

    assert str(func_type) == "(Int, Int) -> Bool"
    assert func_type == FunctionType((IntType, IntType), BoolType)


def test_var_node():
    """Test variable AST node."""
    var = Var("x")

    assert repr(var) == "x"
    assert var.name == "x"


def test_const_node():
    """Test constant AST node."""
    const = Const(42)

    assert repr(const) == "42"
    assert const.value == 42


def test_prim_node():
    """Test primitive AST node."""
    prim = Prim("rotate_90")

    assert repr(prim) == "rotate_90"
    assert prim.name == "rotate_90"


def test_apply_node():
    """Test function application node."""
    func = Prim("rotate_90")
    arg = Var("grid")
    apply = Apply(func, [arg])

    assert "rotate_90" in repr(apply)
    assert "grid" in repr(apply)


def test_lambda_node():
    """Test lambda abstraction node."""
    body = Var("x")
    lam = Lambda(["x"], body)

    assert "λ" in repr(lam) or "lambda" in repr(lam).lower()


def test_let_node():
    """Test let binding node."""
    value = Const(5)
    body = Var("x")
    let = Let("x", value, body)

    assert "let" in repr(let)
    assert let.var == "x"


def test_if_then_else_node():
    """Test conditional node."""
    cond = Const(True)
    then_branch = Const(1)
    else_branch = Const(2)
    ite = IfThenElse(cond, then_branch, else_branch)

    assert "if" in repr(ite)


def test_program():
    """Test program creation."""
    expr = Apply(Prim("rotate_90"), [Var("input")])
    prog = Program(expr, name="test_prog", description="Test program")

    assert prog.name == "test_prog"
    assert prog.expr == expr


def test_serialization_var():
    """Test variable serialization."""
    var = Var("x")
    d = var.to_dict()

    assert d["type"] == "var"
    assert d["name"] == "x"

    var2 = Var.from_dict(d)
    assert var2.name == var.name


def test_serialization_const():
    """Test constant serialization."""
    const = Const(42)
    d = const.to_dict()

    assert d["type"] == "const"
    assert d["value"] == 42

    const2 = Const.from_dict(d)
    assert const2.value == const.value


def test_serialization_prim():
    """Test primitive serialization."""
    prim = Prim("rotate_90")
    d = prim.to_dict()

    assert d["type"] == "prim"
    assert d["name"] == "rotate_90"

    prim2 = Prim.from_dict(d)
    assert prim2.name == prim.name


def test_serialization_apply():
    """Test application serialization."""
    apply = Apply(Prim("add"), [Const(1), Const(2)])
    d = apply.to_dict()

    assert d["type"] == "apply"
    assert len(d["args"]) == 2

    apply2 = Apply.from_dict(d)
    assert len(apply2.args) == len(apply.args)


def test_serialization_lambda():
    """Test lambda serialization."""
    lam = Lambda(["x", "y"], Var("x"))
    d = lam.to_dict()

    assert d["type"] == "lambda"
    assert d["params"] == ["x", "y"]

    lam2 = Lambda.from_dict(d)
    assert lam2.params == lam.params


def test_program_json():
    """Test program JSON serialization."""
    expr = Apply(Prim("rotate_90"), [Var("input")])
    prog = Program(expr, name="test", description="Test program")

    json_str = prog.to_json()
    prog2 = Program.from_json(json_str)

    assert prog2.name == prog.name
    assert prog2.description == prog.description


def test_size_simple():
    """Test size calculation for simple expressions."""
    assert size(Var("x")) == 1
    assert size(Const(42)) == 1
    assert size(Prim("rotate_90")) == 1


def test_size_complex():
    """Test size calculation for complex expressions."""
    # apply(prim, [const, var]) = 1 + 1 + 1 + 1 = 4
    expr = Apply(Prim("add"), [Const(1), Var("x")])
    assert size(expr) == 4


def test_depth_simple():
    """Test depth calculation for simple expressions."""
    assert depth(Var("x")) == 1
    assert depth(Const(42)) == 1
    assert depth(Prim("rotate_90")) == 1


def test_depth_complex():
    """Test depth calculation for complex expressions."""
    # apply(prim, [const]) has depth 2
    expr = Apply(Prim("rotate_90"), [Const(1)])
    assert depth(expr) == 2


def test_get_primitives():
    """Test extracting primitives from expression."""
    expr = Apply(Prim("rotate_90"), [Apply(Prim("reflect_h"), [Var("x")])])
    prims = get_primitives(expr)

    assert "rotate_90" in prims
    assert "reflect_h" in prims


def test_get_variables():
    """Test extracting variables from expression."""
    expr = Apply(Prim("add"), [Var("x"), Var("y")])
    vars = get_variables(expr)

    assert "x" in vars
    assert "y" in vars


def test_substitute_var():
    """Test variable substitution in variable."""
    expr = Var("x")
    subst = substitute(expr, "x", Const(42))

    assert isinstance(subst, Const)
    assert subst.value == 42


def test_substitute_apply():
    """Test variable substitution in application."""
    expr = Apply(Prim("add"), [Var("x"), Const(1)])
    subst = substitute(expr, "x", Const(5))

    assert isinstance(subst, Apply)
    assert isinstance(subst.args[0], Const)
    assert subst.args[0].value == 5


def test_substitute_lambda_shadowing():
    """Test variable substitution with lambda shadowing."""
    # λx. x should not substitute x
    expr = Lambda(["x"], Var("x"))
    subst = substitute(expr, "x", Const(42))

    assert isinstance(subst, Lambda)
    assert isinstance(subst.body, Var)


def test_substitute_let():
    """Test variable substitution in let."""
    # let y = x in y
    expr = Let("y", Var("x"), Var("y"))
    subst = substitute(expr, "x", Const(42))

    assert isinstance(subst, Let)
    assert isinstance(subst.value, Const)
    assert subst.value.value == 42


def test_substitute_if():
    """Test variable substitution in conditional."""
    expr = IfThenElse(Var("x"), Const(1), Const(2))
    subst = substitute(expr, "x", Const(True))

    assert isinstance(subst, IfThenElse)
    assert isinstance(subst.cond, Const)


def test_empty_apply():
    """Test application with no arguments."""
    apply = Apply(Prim("identity"), [])
    assert repr(apply) == "identity()"


def test_nested_let():
    """Test nested let expressions."""
    inner = Let("y", Const(2), Var("y"))
    outer = Let("x", Const(1), inner)

    assert size(outer) > size(inner)
    assert depth(outer) > depth(inner)


def test_complex_program():
    """Test complex program creation and serialization."""
    # Create: let f = λx. rotate_90(x) in f(input)
    lam = Lambda(["x"], Apply(Prim("rotate_90"), [Var("x")]))
    body = Apply(Var("f"), [Var("input")])
    expr = Let("f", lam, body)

    prog = Program(expr, name="rotate_program")
    json_str = prog.to_json()

    # Should be valid JSON
    data = json.loads(json_str)
    assert data["name"] == "rotate_program"

    # Should deserialize correctly
    prog2 = Program.from_json(json_str)
    assert isinstance(prog2.expr, Let)
