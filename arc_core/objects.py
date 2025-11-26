"""
Object detection and shape analysis for ARC challenge.

Provides comprehensive object-oriented analysis including:
- Object extraction and properties (bbox, centroid, area)
- Shape classification and analysis
- Spatial relationships between objects
- Object transformations and manipulations
"""

from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from .grid import Grid, connected_components as grid_connected_components


class ShapeType(Enum):
    """Common shape types in ARC tasks."""
    RECTANGLE = "rectangle"
    SQUARE = "square"
    LINE_H = "line_horizontal"
    LINE_V = "line_vertical"
    LINE_D = "line_diagonal"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    CROSS = "cross"
    HOLLOW_RECT = "hollow_rectangle"
    IRREGULAR = "irregular"


@dataclass
class Object:
    """Represents a connected component in a grid with rich properties."""

    positions: Set[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int] = field(init=False)  # min_r, min_c, max_r, max_c
    area: int = field(init=False)
    centroid: Tuple[float, float] = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    perimeter: int = field(init=False)
    shape_type: ShapeType = field(init=False)
    is_convex: bool = field(init=False)
    holes: int = field(init=False)

    def __post_init__(self):
        """Compute derived properties."""
        self.bbox = self._compute_bbox()
        self.area = len(self.positions)
        self.centroid = self._compute_centroid()
        self.width = self.bbox[3] - self.bbox[1] + 1 if self.positions else 0
        self.height = self.bbox[2] - self.bbox[0] + 1 if self.positions else 0
        self.perimeter = self._compute_perimeter()
        self.shape_type = self._classify_shape()
        self.is_convex = self._check_convexity()
        self.holes = self._count_holes()

    def _compute_bbox(self) -> Tuple[int, int, int, int]:
        """Compute bounding box (min_r, min_c, max_r, max_c)."""
        if not self.positions:
            return 0, 0, 0, 0
        rows = [r for r, c in self.positions]
        cols = [c for r, c in self.positions]
        return min(rows), min(cols), max(rows), max(cols)

    def _compute_centroid(self) -> Tuple[float, float]:
        """Compute centroid (center of mass)."""
        if not self.positions:
            return 0.0, 0.0
        r_sum = sum(r for r, c in self.positions)
        c_sum = sum(c for r, c in self.positions)
        return r_sum / self.area, c_sum / self.area

    def _compute_perimeter(self) -> int:
        """Compute perimeter (number of boundary cells)."""
        if not self.positions:
            return 0
        boundary = 0
        for r, c in self.positions:
            # Check 4 neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (r + dr, c + dc) not in self.positions:
                    boundary += 1
        return boundary

    def _classify_shape(self) -> ShapeType:
        """Classify the shape of the object."""
        if not self.positions:
            return ShapeType.IRREGULAR

        # Check if it's a filled rectangle
        expected_area = self.width * self.height
        if self.area == expected_area:
            if self.width == self.height:
                return ShapeType.SQUARE
            return ShapeType.RECTANGLE

        # Check if it's a line
        if self.width == 1 and self.height > 1:
            return ShapeType.LINE_V
        if self.height == 1 and self.width > 1:
            return ShapeType.LINE_H

        # Check diagonal line
        if self.area == max(self.width, self.height):
            return ShapeType.LINE_D

        # Check hollow rectangle
        if self._is_hollow_rectangle():
            return ShapeType.HOLLOW_RECT

        # Check for L, T, or cross shapes
        if self._is_cross():
            return ShapeType.CROSS
        if self._is_t_shape():
            return ShapeType.T_SHAPE
        if self._is_l_shape():
            return ShapeType.L_SHAPE

        return ShapeType.IRREGULAR

    def _is_hollow_rectangle(self) -> bool:
        """Check if object is a hollow rectangle."""
        if self.width < 3 or self.height < 3:
            return False

        min_r, min_c, max_r, max_c = self.bbox
        # Check if border is filled and interior is empty
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                is_border = (r == min_r or r == max_r or c == min_c or c == max_c)
                is_filled = (r, c) in self.positions

                if is_border and not is_filled:
                    return False
                if not is_border and is_filled:
                    return False

        return True

    def _is_cross(self) -> bool:
        """Check if object is a cross shape."""
        min_r, min_c, max_r, max_c = self.bbox
        mid_r = (min_r + max_r) / 2
        mid_c = (min_c + max_c) / 2

        # Simple heuristic: check if center row and column are mostly filled
        center_row_count = sum(1 for r, c in self.positions if abs(r - mid_r) < 1)
        center_col_count = sum(1 for r, c in self.positions if abs(c - mid_c) < 1)

        return (center_row_count > self.width * 0.6 and
                center_col_count > self.height * 0.6)

    def _is_t_shape(self) -> bool:
        """Check if object is a T shape."""
        # Simplified heuristic
        return False

    def _is_l_shape(self) -> bool:
        """Check if object is an L shape."""
        # Simplified heuristic
        return False

    def _check_convexity(self) -> bool:
        """Check if object is convex."""
        # Simple check: object is convex if it fills its bounding box
        return self.area == self.width * self.height

    def _count_holes(self) -> int:
        """Count number of holes (empty regions inside object)."""
        if self.area == 0:
            return 0

        # Create a grid representation
        min_r, min_c, max_r, max_c = self.bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        grid = np.zeros((h, w), dtype=np.int8)
        for r, c in self.positions:
            grid[r - min_r, c - min_c] = 1

        # Find background components (0s) that don't touch the border
        bg_components = grid_connected_components(1 - grid, connectivity=4, background=0)

        holes = 0
        for comp in bg_components:
            # Check if component touches border
            touches_border = any(
                r == 0 or r == h - 1 or c == 0 or c == w - 1
                for r, c in comp
            )
            if not touches_border:
                holes += 1

        return holes

    def to_grid(self) -> Grid:
        """Convert object to a grid (bounding box)."""
        if not self.positions:
            return np.array([[]], dtype=np.int8)

        min_r, min_c, max_r, max_c = self.bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        grid = np.zeros((h, w), dtype=np.int8)
        for r, c in self.positions:
            grid[r - min_r, c - min_c] = self.color

        return grid

    def translate(self, dr: int, dc: int) -> 'Object':
        """Translate object by (dr, dc)."""
        new_positions = {(r + dr, c + dc) for r, c in self.positions}
        return Object(new_positions, self.color)

    def rotate_90_cw(self, center: Optional[Tuple[float, float]] = None) -> 'Object':
        """Rotate object 90 degrees clockwise around center (default: centroid)."""
        if center is None:
            center = self.centroid

        cr, cc = center
        new_positions = set()
        for r, c in self.positions:
            # Translate to origin, rotate, translate back
            r_rel = r - cr
            c_rel = c - cc
            r_new = c_rel + cr
            c_new = -r_rel + cc
            new_positions.add((int(round(r_new)), int(round(c_new))))

        return Object(new_positions, self.color)

    def scale(self, factor: int) -> 'Object':
        """Scale object by repeating each cell factor x factor times."""
        new_positions = set()
        min_r, min_c, _, _ = self.bbox

        for r, c in self.positions:
            r_rel = r - min_r
            c_rel = c - min_c
            for dr in range(factor):
                for dc in range(factor):
                    new_positions.add((min_r + r_rel * factor + dr,
                                     min_c + c_rel * factor + dc))

        return Object(new_positions, self.color)

    def overlaps(self, other: 'Object') -> bool:
        """Check if this object overlaps with another."""
        return bool(self.positions & other.positions)

    def contains(self, position: Tuple[int, int]) -> bool:
        """Check if object contains a position."""
        return position in self.positions

    def distance_to(self, other: 'Object') -> float:
        """Compute minimum distance to another object."""
        min_dist = float('inf')
        for r1, c1 in self.positions:
            for r2, c2 in other.positions:
                dist = abs(r1 - r2) + abs(c1 - c2)  # Manhattan distance
                min_dist = min(min_dist, dist)
        return min_dist

    def __repr__(self) -> str:
        return f"Object(color={self.color}, area={self.area}, bbox={self.bbox}, shape={self.shape_type.value})"


# ============================================================================
# Object Extraction
# ============================================================================

def extract_objects(grid: Grid, connectivity: int = 4,
                   background: Optional[int] = None) -> List[Object]:
    """Extract objects from grid using connected components.

    Args:
        grid: Input grid
        connectivity: 4 or 8 connectivity
        background: Background color to ignore (default: 0)

    Returns:
        List of Object instances
    """
    components = grid_connected_components(grid, connectivity, background)
    objects = []

    for comp in components:
        if not comp:
            continue
        # Get color from first position
        r, c = next(iter(comp))
        color = int(grid[r, c])
        obj = Object(comp, color)
        objects.append(obj)

    return objects


def extract_objects_by_color(grid: Grid, connectivity: int = 4,
                             background: Optional[int] = None) -> Dict[int, List[Object]]:
    """Extract objects grouped by color."""
    objects = extract_objects(grid, connectivity, background)
    by_color: Dict[int, List[Object]] = {}

    for obj in objects:
        if obj.color not in by_color:
            by_color[obj.color] = []
        by_color[obj.color].append(obj)

    return by_color


# ============================================================================
# Spatial Relationships
# ============================================================================

class Relationship(Enum):
    """Spatial relationships between objects."""
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    CONTAINED_BY = "contained_by"
    ADJACENT = "adjacent"
    ALIGNED_H = "aligned_horizontal"
    ALIGNED_V = "aligned_vertical"


def get_relationships(obj1: Object, obj2: Object) -> Set[Relationship]:
    """Determine spatial relationships between two objects."""
    relationships = set()

    # Overlapping
    if obj1.overlaps(obj2):
        relationships.add(Relationship.OVERLAPS)

    # Containment
    if obj1.positions.issuperset(obj2.positions):
        relationships.add(Relationship.CONTAINS)
    if obj1.positions.issubset(obj2.positions):
        relationships.add(Relationship.CONTAINED_BY)

    # Relative position
    c1_r, c1_c = obj1.centroid
    c2_r, c2_c = obj2.centroid

    if c1_r < c2_r - 1:
        relationships.add(Relationship.ABOVE)
    if c1_r > c2_r + 1:
        relationships.add(Relationship.BELOW)
    if c1_c < c2_c - 1:
        relationships.add(Relationship.LEFT)
    if c1_c > c2_c + 1:
        relationships.add(Relationship.RIGHT)

    # Alignment
    if abs(c1_r - c2_r) < 1:
        relationships.add(Relationship.ALIGNED_H)
    if abs(c1_c - c2_c) < 1:
        relationships.add(Relationship.ALIGNED_V)

    # Adjacency
    if not obj1.overlaps(obj2) and obj1.distance_to(obj2) == 1:
        relationships.add(Relationship.ADJACENT)

    return relationships


def find_largest_object(objects: List[Object]) -> Optional[Object]:
    """Find the largest object by area."""
    if not objects:
        return None
    return max(objects, key=lambda obj: obj.area)


def find_smallest_object(objects: List[Object]) -> Optional[Object]:
    """Find the smallest object by area."""
    if not objects:
        return None
    return min(objects, key=lambda obj: obj.area)


def filter_by_color(objects: List[Object], color: int) -> List[Object]:
    """Filter objects by color."""
    return [obj for obj in objects if obj.color == color]


def filter_by_shape(objects: List[Object], shape: ShapeType) -> List[Object]:
    """Filter objects by shape type."""
    return [obj for obj in objects if obj.shape_type == shape]


def filter_by_area(objects: List[Object], min_area: int = 0,
                   max_area: int = float('inf')) -> List[Object]:
    """Filter objects by area range."""
    return [obj for obj in objects if min_area <= obj.area <= max_area]


def sort_objects_by_position(objects: List[Object],
                             by: str = 'row') -> List[Object]:
    """Sort objects by position (row or column).

    Args:
        objects: List of objects
        by: 'row' or 'column'

    Returns:
        Sorted list of objects
    """
    if by == 'row':
        return sorted(objects, key=lambda obj: obj.centroid[0])
    elif by == 'column':
        return sorted(objects, key=lambda obj: obj.centroid[1])
    else:
        raise ValueError(f"Invalid sort key: {by}")


# ============================================================================
# Grid Reconstruction
# ============================================================================

def objects_to_grid(objects: List[Object], height: int, width: int,
                   background: int = 0) -> Grid:
    """Reconstruct grid from objects.

    Args:
        objects: List of objects
        height: Grid height
        width: Grid width
        background: Background color

    Returns:
        Reconstructed grid
    """
    grid = np.full((height, width), background, dtype=np.int8)

    for obj in objects:
        for r, c in obj.positions:
            if 0 <= r < height and 0 <= c < width:
                grid[r, c] = obj.color

    return grid