import numpy as np
from ..arc_core.grid import Grid

def solve_symmetry(input_grids: List[Grid], output_grids: List[Grid]) -> Grid:
    """Try to solve by symmetry operations."""
    # Simple: if input and output differ by mirror, return mirrored input
    input_grid = input_grids[0]
    output_grid = output_grids[0]
    
    # Check horizontal mirror
    mirrored_h = np.flipud(input_grid)
    if np.array_equal(mirrored_h, output_grid):
        return np.flipud(input_grid)  # for test input
    
    # Vertical
    mirrored_v = np.fliplr(input_grid)
    if np.array_equal(mirrored_v, output_grid):
        return np.fliplr(input_grid)
    
    return None  # no match