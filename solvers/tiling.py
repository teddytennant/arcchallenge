"""
Tiling pattern solver.

Detects and solves tasks involving grid tiling and repetition patterns.
"""

from typing import List, Optional, Tuple, Callable
import numpy as np

from ..arc_core.grid import Grid
from ..arc_core import grid as grid_ops


def solve_by_tiling(train_examples: List[Tuple[Grid, Grid]]) -> Optional[Callable]:
    """
    Solve task by detecting tiling patterns.

    Args:
        train_examples: List of (input, output) grid pairs

    Returns:
        Transformation function if tiling pattern found, None otherwise
    """

    for input_grid, output_grid in train_examples:
        # Check if output is a tiled version of input
        tile_info = grid_ops.is_tiled(output_grid)

        if tile_info is not None:
            rows, cols, base_tile = tile_info

            # Check if base tile matches input (possibly after transformation)
            if _grids_match(base_tile, input_grid):
                # Output is direct tiling of input
                return lambda g: grid_ops.tile_grid(g, rows, cols)

            # Check for transformed tiles
            transform = _find_tile_transform(input_grid, base_tile)
            if transform:
                def tile_with_transform(g: Grid) -> Grid:
                    transformed = transform(g)
                    return grid_ops.tile_grid(transformed, rows, cols)
                return tile_with_transform

        # Check if input is tiled and output is the base
        tile_info = grid_ops.is_tiled(input_grid)
        if tile_info is not None:
            rows, cols, base_tile = tile_info
            if _grids_match(base_tile, output_grid):
                # Output is the base tile (de-tiling)
                return lambda g: _extract_base_tile(g)

    return None


def _grids_match(grid1: Grid, grid2: Grid) -> bool:
    """Check if two grids match exactly."""
    if grid1.shape != grid2.shape:
        return False
    return np.array_equal(grid1, grid2)


def _find_tile_transform(input_grid: Grid, target_tile: Grid) -> Optional[Callable]:
    """Find transformation that converts input to target tile."""

    # Try common transformations
    transforms = [
        ("identity", lambda g: g),
        ("rotate_90", grid_ops.rotate_90),
        ("rotate_180", grid_ops.rotate_180),
        ("rotate_270", grid_ops.rotate_270),
        ("reflect_h", grid_ops.reflect_horizontal),
        ("reflect_v", grid_ops.reflect_vertical),
        ("trim", grid_ops.trim_borders),
    ]

    for name, transform in transforms:
        try:
            result = transform(input_grid)
            if _grids_match(result, target_tile):
                return transform
        except:
            continue

    return None


def _extract_base_tile(grid: Grid) -> Grid:
    """Extract base tile from a tiled grid."""
    tile_info = grid_ops.is_tiled(grid)
    if tile_info:
        _, _, base_tile = tile_info
        return base_tile
    return grid


def detect_repetition_pattern(grid: Grid) -> Optional[Tuple[str, int]]:
    """
    Detect if grid has a repetition pattern.

    Returns:
        Tuple of (axis, period) if pattern found, None otherwise
        axis: 'row' or 'col'
    """

    # Check row repetition
    row_period = grid_ops.detect_period(grid, axis=0)
    if row_period:
        return ('row', row_period)

    # Check column repetition
    col_period = grid_ops.detect_period(grid, axis=1)
    if col_period:
        return ('col', col_period)

    return None


def solve_by_repetition(train_examples: List[Tuple[Grid, Grid]]) -> Optional[Callable]:
    """
    Solve task by detecting repetition patterns.

    Args:
        train_examples: List of (input, output) grid pairs

    Returns:
        Transformation function if pattern found, None otherwise
    """

    patterns = []

    for input_grid, output_grid in train_examples:
        pattern = detect_repetition_pattern(output_grid)
        if pattern:
            patterns.append(pattern)
        else:
            return None

    # Check consistency
    if not patterns:
        return None

    first_pattern = patterns[0]
    if not all(p == first_pattern for p in patterns):
        return None

    axis, period = first_pattern

    # Build transformation
    if axis == 'row':
        def repeat_rows(g: Grid) -> Grid:
            return np.repeat(g, period, axis=0)
        return repeat_rows
    else:
        def repeat_cols(g: Grid) -> Grid:
            return np.repeat(g, period, axis=1)
        return repeat_cols
