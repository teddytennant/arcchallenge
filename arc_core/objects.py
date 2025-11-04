from typing import List, Tuple, Set, Dict
import numpy as np
from .grid import Grid

class Object:
    def __init__(self, positions: Set[Tuple[int, int]], color: int):
        self.positions = positions
        self.color = color
        self.bbox = self._compute_bbox()
        self.area = len(positions)
        self.centroid = self._compute_centroid()
    
    def _compute_bbox(self) -> Tuple[int, int, int, int]:  # min_r, min_c, max_r, max_c
        if not self.positions:
            return 0, 0, 0, 0
        rows = [r for r, c in self.positions]
        cols = [c for r, c in self.positions]
        return min(rows), min(cols), max(rows), max(cols)
    
    def _compute_centroid(self) -> Tuple[float, float]:
        if not self.positions:
            return 0.0, 0.0
        r_sum = sum(r for r, c in self.positions)
        c_sum = sum(c for r, c in self.positions)
        return r_sum / self.area, c_sum / self.area

def extract_objects(grid: Grid) -> List[Object]:
    """Extract objects from grid."""
    components = connected_components(grid)
    objects = []
    for comp in components:
        if not comp:
            continue
        # Get color from first position
        r, c = next(iter(comp))
        color = grid[r, c]
        obj = Object(comp, color)
        objects.append(obj)
    return objects

def connected_components(grid: Grid) -> List[Set[Tuple[int, int]]]:
    """Alias for grid.connected_components."""
    from .grid import connected_components
    return connected_components(grid)