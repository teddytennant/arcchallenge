import numpy as np
import pytest
from dsl.primitives import (
    Primitive, get_primitive, list_primitives,
    ROTATE_90, ROTATE_180, ROTATE_270,
    REFLECT_H, REFLECT_V, REFLECT_D, REFLECT_AD,
    TRIM_BORDERS, SCALE_2X, DOWNSCALE_2X,
    EXTRACT_OBJECTS, DETECT_BACKGROUND,
    FILTER_COLOR, ARGMAX_AREA, ARGMIN_AREA,
    ADD, SUB, MUL, DIV, MOD, MIN, MAX, ABS,
    EQ, NEQ, LT, GT, AND, OR, NOT,
    IF_THEN_ELSE, HAS_SYMMETRY_V, HAS_SYMMETRY_H,
    PRIMITIVES_BY_NAME, ALL_PRIMITIVES
)
from arc_core.grid import load_grid
from arc_core.objects import Object


def test_primitive_class():
    """Test Primitive class creation."""
    prim = Primitive("test", lambda x: x + 1, "Int -> Int", "Test primitive")

    assert prim.name == "test"
    assert prim.type_sig == "Int -> Int"
    assert prim.description == "Test primitive"
    assert prim(5) == 6


def test_get_primitive():
    """Test getting primitive by name."""
    prim = get_primitive("rotate_90")

    assert prim is not None
    assert prim.name == "rotate_90"


def test_get_primitive_not_found():
    """Test getting non-existent primitive."""
    prim = get_primitive("nonexistent")
    assert prim is None


def test_list_primitives():
    """Test listing all primitives."""
    prims = list_primitives()
    assert len(prims) > 0
    assert all(isinstance(p, Primitive) for p in prims)


def test_list_primitives_filtered():
    """Test listing primitives with type filter."""
    grid_prims = list_primitives(type_filter="Grid")
    assert len(grid_prims) > 0
    assert all("Grid" in p.type_sig for p in grid_prims)


def test_rotate_90():
    """Test 90-degree rotation primitive."""
    grid = load_grid([[1, 2], [3, 4]])
    result = ROTATE_90(grid)

    assert result.shape == (2, 2)
    assert result[0, 0] == 3
    assert result[0, 1] == 1


def test_rotate_180():
    """Test 180-degree rotation primitive."""
    grid = load_grid([[1, 2], [3, 4]])
    result = ROTATE_180(grid)

    assert result.shape == (2, 2)
    assert result[0, 0] == 4
    assert result[1, 1] == 1


def test_rotate_270():
    """Test 270-degree rotation primitive."""
    grid = load_grid([[1, 2], [3, 4]])
    result = ROTATE_270(grid)

    assert result.shape == (2, 2)
    assert result[0, 0] == 2
    assert result[1, 1] == 3


def test_reflect_h():
    """Test horizontal reflection primitive."""
    grid = load_grid([[1, 2, 3]])
    result = REFLECT_H(grid)

    assert result.shape == (1, 3)
    assert result[0, 0] == 3
    assert result[0, 2] == 1


def test_reflect_v():
    """Test vertical reflection primitive."""
    grid = load_grid([[1], [2], [3]])
    result = REFLECT_V(grid)

    assert result.shape == (3, 1)
    assert result[0, 0] == 3
    assert result[2, 0] == 1


def test_trim_borders():
    """Test trim borders primitive."""
    grid = load_grid([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    result = TRIM_BORDERS(grid)

    assert result.shape == (1, 1)
    assert result[0, 0] == 1


def test_scale_2x():
    """Test 2x scaling primitive."""
    grid = load_grid([[1, 2]])
    result = SCALE_2X(grid)

    assert result.shape == (2, 4)


def test_downscale_2x():
    """Test 2x downscaling primitive."""
    grid = load_grid([[1, 1, 2, 2], [1, 1, 2, 2]])
    result = DOWNSCALE_2X(grid)

    assert result.shape == (1, 2)


def test_extract_objects():
    """Test object extraction primitive."""
    grid = load_grid([
        [0, 1, 1],
        [0, 0, 2]
    ])
    objects = EXTRACT_OBJECTS(grid)

    assert len(objects) == 2
    colors = {obj.color for obj in objects}
    assert colors == {1, 2}


def test_detect_background():
    """Test background detection primitive."""
    grid = load_grid([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    bg = DETECT_BACKGROUND(grid)

    assert bg == 0


def test_filter_color():
    """Test color filtering primitive."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0)}, color=2)
    obj3 = Object({(2, 0)}, color=1)

    result = FILTER_COLOR([obj1, obj2, obj3], 1)
    assert len(result) == 2
    assert all(obj.color == 1 for obj in result)


def test_argmax_area():
    """Test argmax area primitive."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0), (1, 1), (1, 2)}, color=2)

    result = ARGMAX_AREA([obj1, obj2])
    assert result.area == 3


def test_argmin_area():
    """Test argmin area primitive."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0), (1, 1), (1, 2)}, color=2)

    result = ARGMIN_AREA([obj1, obj2])
    assert result.area == 1


def test_arithmetic_add():
    """Test addition primitive."""
    assert ADD(5, 3) == 8


def test_arithmetic_sub():
    """Test subtraction primitive."""
    assert SUB(5, 3) == 2


def test_arithmetic_mul():
    """Test multiplication primitive."""
    assert MUL(5, 3) == 15


def test_arithmetic_div():
    """Test integer division primitive."""
    assert DIV(7, 3) == 2
    assert DIV(5, 0) == 0  # Division by zero returns 0


def test_arithmetic_mod():
    """Test modulo primitive."""
    assert MOD(7, 3) == 1
    assert MOD(5, 0) == 0  # Modulo by zero returns 0


def test_arithmetic_min():
    """Test minimum primitive."""
    assert MIN(5, 3) == 3


def test_arithmetic_max():
    """Test maximum primitive."""
    assert MAX(5, 3) == 5


def test_arithmetic_abs():
    """Test absolute value primitive."""
    assert ABS(-5) == 5
    assert ABS(5) == 5


def test_comparison_eq():
    """Test equality primitive."""
    assert EQ(5, 5) == True
    assert EQ(5, 3) == False


def test_comparison_neq():
    """Test not equal primitive."""
    assert NEQ(5, 3) == True
    assert NEQ(5, 5) == False


def test_comparison_lt():
    """Test less than primitive."""
    assert LT(3, 5) == True
    assert LT(5, 3) == False


def test_comparison_gt():
    """Test greater than primitive."""
    assert GT(5, 3) == True
    assert GT(3, 5) == False


def test_boolean_and():
    """Test logical AND primitive."""
    assert AND(True, True) == True
    assert AND(True, False) == False
    assert AND(False, False) == False


def test_boolean_or():
    """Test logical OR primitive."""
    assert OR(True, False) == True
    assert OR(False, False) == False
    assert OR(True, True) == True


def test_boolean_not():
    """Test logical NOT primitive."""
    assert NOT(True) == False
    assert NOT(False) == True


def test_if_then_else():
    """Test conditional primitive."""
    assert IF_THEN_ELSE(True, 1, 2) == 1
    assert IF_THEN_ELSE(False, 1, 2) == 2


def test_has_symmetry_v():
    """Test vertical symmetry detection."""
    symmetric = load_grid([[1, 2, 1]])
    assert HAS_SYMMETRY_V(symmetric) == True

    asymmetric = load_grid([[1, 2, 3]])
    assert HAS_SYMMETRY_V(asymmetric) == False


def test_has_symmetry_h():
    """Test horizontal symmetry detection."""
    symmetric = load_grid([[1], [2], [1]])
    assert HAS_SYMMETRY_H(symmetric) == True

    asymmetric = load_grid([[1], [2], [3]])
    assert HAS_SYMMETRY_H(asymmetric) == False


def test_primitives_by_name():
    """Test PRIMITIVES_BY_NAME dictionary."""
    assert "rotate_90" in PRIMITIVES_BY_NAME
    assert "add" in PRIMITIVES_BY_NAME
    assert PRIMITIVES_BY_NAME["rotate_90"] == ROTATE_90


def test_all_primitives():
    """Test ALL_PRIMITIVES list."""
    assert len(ALL_PRIMITIVES) > 0
    assert ROTATE_90 in ALL_PRIMITIVES
    assert ADD in ALL_PRIMITIVES


def test_primitive_type_signatures():
    """Test that all primitives have type signatures."""
    for prim in ALL_PRIMITIVES:
        assert prim.type_sig is not None
        assert len(prim.type_sig) > 0


def test_primitive_names_unique():
    """Test that all primitive names are unique."""
    names = [p.name for p in ALL_PRIMITIVES]
    assert len(names) == len(set(names))


def test_primitive_callable():
    """Test that all primitives are callable."""
    # Just test a few to make sure they have implementations
    assert callable(ROTATE_90.impl)
    assert callable(ADD.impl)
    assert callable(IF_THEN_ELSE.impl)
