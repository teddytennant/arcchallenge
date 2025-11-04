import numpy as np
from typing import List, Tuple, Set

Grid = np.ndarray  # 2D array of int (0-9)

def load_grid(data: List[List[int]]) -> Grid:
    """Load grid from list of lists."""
    return np.array(data, dtype=np.int8)

def save_grid(grid: Grid) -> List[List[int]]:
    """Save grid to list of lists."""
    return grid.tolist()

def canonicalize_colors(grid: Grid) -> Tuple[Grid, dict]:
    """Reindex colors by frequency, return new grid and mapping."""
    unique, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)  # descending frequency
    mapping = {old: new for new, old in enumerate(unique[order])}
    new_grid = np.vectorize(mapping.get)(grid)
    return new_grid, mapping

def trim_borders(grid: Grid, bg_color: int = 0) -> Grid:
    """Trim constant borders."""
    # Find rows/cols that are not all bg
    rows = np.any(grid != bg_color, axis=1)
    cols = np.any(grid != bg_color, axis=0)
    return grid[rows][:, cols]

def detect_background(grid: Grid) -> int:
    """Detect background color as the most frequent on borders."""
    borders = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    unique, counts = np.unique(borders, return_counts=True)
    return unique[np.argmax(counts)]

def connected_components(grid: Grid, connectivity: int = 4) -> List[Set[Tuple[int, int]]]:
    """Extract connected components (4 or 8 connected)."""
    # Simple flood fill implementation
    visited = set()
    components = []
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    def flood_fill(r, c, color):
        stack = [(r, c)]
        component = set()
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited or not (0 <= cr < rows and 0 <= cc < cols) or grid[cr, cc] != color:
                continue
            visited.add((cr, cc))
            component.add((cr, cc))
            for dr, dc in directions:
                stack.append((cr + dr, cc + dc))
        return component
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r, c] != 0:  # assume 0 is bg
                comp = flood_fill(r, c, grid[r, c])
                if comp:
                    components.append(comp)
    return components