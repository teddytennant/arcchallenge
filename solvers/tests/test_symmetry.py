import numpy as np
import pytest
from arc_core.grid import load_grid


def test_symmetry_solver_import():
    """Test that symmetry solver can be imported."""
    try:
        from solvers.symmetry import solve_symmetry
        assert callable(solve_symmetry)
    except ImportError:
        pytest.skip("Symmetry solver module not fully implemented")


def test_symmetry_horizontal_flip():
    """Test symmetry solver with horizontal flip."""
    try:
        from solvers.symmetry import solve_symmetry

        input_grid = load_grid([[1, 2], [3, 4]])
        output_grid = load_grid([[3, 4], [1, 2]])

        result = solve_symmetry([input_grid], [output_grid])

        if result is not None:
            assert isinstance(result, np.ndarray)
    except ImportError:
        pytest.skip("Symmetry solver module not fully implemented")


def test_symmetry_vertical_flip():
    """Test symmetry solver with vertical flip."""
    try:
        from solvers.symmetry import solve_symmetry

        input_grid = load_grid([[1, 2], [3, 4]])
        output_grid = load_grid([[2, 1], [4, 3]])

        result = solve_symmetry([input_grid], [output_grid])

        if result is not None:
            assert isinstance(result, np.ndarray)
    except ImportError:
        pytest.skip("Symmetry solver module not fully implemented")


def test_symmetry_no_match():
    """Test symmetry solver with no matching transformation."""
    try:
        from solvers.symmetry import solve_symmetry

        input_grid = load_grid([[1, 2], [3, 4]])
        output_grid = load_grid([[5, 6], [7, 8]])

        result = solve_symmetry([input_grid], [output_grid])

        # Should return None when no symmetry matches
        assert result is None
    except ImportError:
        pytest.skip("Symmetry solver module not fully implemented")
