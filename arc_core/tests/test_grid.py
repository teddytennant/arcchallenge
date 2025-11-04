import numpy as np
from arc_core.grid import load_grid, canonicalize_colors, trim_borders, detect_background

def test_load_grid():
    data = [[1, 2], [3, 4]]
    grid = load_grid(data)
    assert isinstance(grid, np.ndarray)
    assert grid.shape == (2, 2)
    assert grid.dtype == np.int8

def test_canonicalize_colors():
    grid = np.array([[1, 2, 2], [2, 1, 1]], dtype=np.int8)
    new_grid, mapping = canonicalize_colors(grid)
    # Both 1 and 2 appear 3 times, so order arbitrary, but mapping should have 0 and 1
    assert set(mapping.values()) == {0, 1}
    assert np.array_equal(new_grid, np.vectorize(mapping.get)(grid))

def test_trim_borders():
    grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    trimmed = trim_borders(grid)
    assert trimmed.shape == (1, 1)
    assert trimmed[0, 0] == 1

def test_detect_background():
    grid = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    bg = detect_background(grid)
    assert bg == 0