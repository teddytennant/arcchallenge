"""
Core grid operations for ARC challenge.

This module provides fundamental grid manipulation primitives including:
- Spatial transformations (rotation, reflection, scaling)
- Color operations (remapping, canonicalization)
- Pattern detection (symmetry, periodicity, connected components)
- Grid analysis (histogram, shape detection, spatial relationships)
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

Grid = np.ndarray  # 2D array of int (0-9)


# ============================================================================
# Grid I/O and Construction
# ============================================================================

def load_grid(data: List[List[int]]) -> Grid:
    """Load grid from list of lists."""
    return np.array(data, dtype=np.int8)


def save_grid(grid: Grid) -> List[List[int]]:
    """Save grid to list of lists."""
    return grid.tolist()


def create_grid(height: int, width: int, fill_value: int = 0) -> Grid:
    """Create a new grid with specified dimensions."""
    return np.full((height, width), fill_value, dtype=np.int8)


def clone_grid(grid: Grid) -> Grid:
    """Create a deep copy of a grid."""
    return grid.copy()


# ============================================================================
# Spatial Transformations
# ============================================================================

def rotate_90(grid: Grid) -> Grid:
    """Rotate grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)


def rotate_180(grid: Grid) -> Grid:
    """Rotate grid 180 degrees."""
    return np.rot90(grid, k=2)


def rotate_270(grid: Grid) -> Grid:
    """Rotate grid 270 degrees clockwise (or 90 counter-clockwise)."""
    return np.rot90(grid, k=1)


def reflect_horizontal(grid: Grid) -> Grid:
    """Reflect grid horizontally (flip left-right)."""
    return np.fliplr(grid)


def reflect_vertical(grid: Grid) -> Grid:
    """Reflect grid vertically (flip up-down)."""
    return np.flipud(grid)


def reflect_diagonal(grid: Grid) -> Grid:
    """Reflect grid along main diagonal (transpose)."""
    return grid.T


def reflect_antidiagonal(grid: Grid) -> Grid:
    """Reflect grid along anti-diagonal."""
    return np.fliplr(grid.T)


def scale_grid(grid: Grid, factor: int) -> Grid:
    """Scale grid by repeating each cell factor x factor times."""
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def downscale_grid(grid: Grid, factor: int) -> Optional[Grid]:
    """Downscale grid by sampling every factor-th cell.

    Returns None if grid dimensions are not divisible by factor.
    """
    h, w = grid.shape
    if h % factor != 0 or w % factor != 0:
        return None
    return grid[::factor, ::factor]


def crop_grid(grid: Grid, r_start: int, c_start: int, height: int, width: int) -> Grid:
    """Crop a rectangular region from the grid."""
    h, w = grid.shape
    r_end = min(r_start + height, h)
    c_end = min(c_start + width, w)
    return grid[r_start:r_end, c_start:c_end]


def pad_grid(grid: Grid, top: int, bottom: int, left: int, right: int,
             fill_value: int = 0) -> Grid:
    """Pad grid with fill_value on all sides."""
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=fill_value)


def resize_grid(grid: Grid, new_height: int, new_width: int, fill_value: int = 0) -> Grid:
    """Resize grid to new dimensions, cropping or padding as needed."""
    h, w = grid.shape
    result = create_grid(new_height, new_width, fill_value)
    copy_h = min(h, new_height)
    copy_w = min(w, new_width)
    result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
    return result


def tile_grid(grid: Grid, rows: int, cols: int) -> Grid:
    """Tile the grid in a rows x cols pattern."""
    return np.tile(grid, (rows, cols))


# ============================================================================
# Color Operations
# ============================================================================

def remap_colors(grid: Grid, color_map: Dict[int, int]) -> Grid:
    """Remap colors according to the provided mapping."""
    result = grid.copy()
    for old_color, new_color in color_map.items():
        result[grid == old_color] = new_color
    return result


def swap_colors(grid: Grid, color1: int, color2: int) -> Grid:
    """Swap two colors in the grid."""
    result = grid.copy()
    mask1 = grid == color1
    mask2 = grid == color2
    result[mask1] = color2
    result[mask2] = color1
    return result


def canonicalize_colors(grid: Grid) -> Tuple[Grid, Dict[int, int]]:
    """Reindex colors by frequency, return new grid and mapping."""
    unique, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)  # descending frequency
    mapping = {old: new for new, old in enumerate(unique[order])}
    new_grid = np.vectorize(mapping.get)(grid)
    return new_grid, mapping


def normalize_colors(grid: Grid, start: int = 0) -> Tuple[Grid, Dict[int, int]]:
    """Normalize colors to consecutive integers starting from start."""
    unique = np.unique(grid)
    mapping = {old: start + i for i, old in enumerate(unique)}
    new_grid = np.vectorize(mapping.get)(grid)
    return new_grid, mapping


def get_color_histogram(grid: Grid) -> Dict[int, int]:
    """Get histogram of color frequencies."""
    unique, counts = np.unique(grid, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def get_dominant_color(grid: Grid, exclude_background: bool = False,
                      bg_color: int = 0) -> int:
    """Get the most frequent color in the grid."""
    histogram = get_color_histogram(grid)
    if exclude_background and bg_color in histogram:
        del histogram[bg_color]
    if not histogram:
        return 0
    return max(histogram.items(), key=lambda x: x[1])[0]


def replace_color(grid: Grid, old_color: int, new_color: int) -> Grid:
    """Replace all instances of old_color with new_color."""
    result = grid.copy()
    result[grid == old_color] = new_color
    return result


# ============================================================================
# Background Detection and Border Operations
# ============================================================================

def detect_background(grid: Grid) -> int:
    """Detect background color as the most frequent on borders."""
    if grid.size == 0:
        return 0
    h, w = grid.shape
    if h == 0 or w == 0:
        return 0
    borders = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    unique, counts = np.unique(borders, return_counts=True)
    return int(unique[np.argmax(counts)])


def trim_borders(grid: Grid, bg_color: Optional[int] = None) -> Grid:
    """Trim constant borders. If bg_color not provided, auto-detect."""
    if bg_color is None:
        bg_color = detect_background(grid)

    # Find rows/cols that are not all bg
    rows = np.any(grid != bg_color, axis=1)
    cols = np.any(grid != bg_color, axis=0)

    if not np.any(rows) or not np.any(cols):
        return create_grid(0, 0)

    return grid[rows][:, cols]


def extract_borders(grid: Grid) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract border rows and columns (top, bottom, left, right)."""
    return grid[0], grid[-1], grid[:, 0], grid[:, -1]


# ============================================================================
# Pattern Detection
# ============================================================================

class Symmetry(Enum):
    """Types of symmetry."""
    NONE = 0
    VERTICAL = 1
    HORIZONTAL = 2
    BOTH = 3
    DIAGONAL = 4
    ANTIDIAGONAL = 5
    ROTATIONAL_90 = 6
    ROTATIONAL_180 = 7


def detect_symmetry(grid: Grid) -> Set[Symmetry]:
    """Detect all symmetries present in the grid."""
    symmetries = set()

    if np.array_equal(grid, reflect_vertical(grid)):
        symmetries.add(Symmetry.VERTICAL)
    if np.array_equal(grid, reflect_horizontal(grid)):
        symmetries.add(Symmetry.HORIZONTAL)
    if grid.shape[0] == grid.shape[1] and np.array_equal(grid, reflect_diagonal(grid)):
        symmetries.add(Symmetry.DIAGONAL)
    if grid.shape[0] == grid.shape[1] and np.array_equal(grid, reflect_antidiagonal(grid)):
        symmetries.add(Symmetry.ANTIDIAGONAL)
    if grid.shape[0] == grid.shape[1] and np.array_equal(grid, rotate_90(grid)):
        symmetries.add(Symmetry.ROTATIONAL_90)
    if np.array_equal(grid, rotate_180(grid)):
        symmetries.add(Symmetry.ROTATIONAL_180)

    if not symmetries:
        symmetries.add(Symmetry.NONE)

    return symmetries


def detect_period(grid: Grid, axis: int) -> Optional[int]:
    """Detect if grid has periodic pattern along given axis (0=row, 1=col).

    Returns the period if found, None otherwise.
    """
    size = grid.shape[axis]

    # Try periods from 1 to size//2
    for period in range(1, size // 2 + 1):
        if size % period != 0:
            continue

        # Check if repeating with this period
        is_periodic = True
        for i in range(period, size):
            if axis == 0:
                if not np.array_equal(grid[i], grid[i % period]):
                    is_periodic = False
                    break
            else:
                if not np.array_equal(grid[:, i], grid[:, i % period]):
                    is_periodic = False
                    break

        if is_periodic:
            return period

    return None


def is_tiled(grid: Grid) -> Optional[Tuple[int, int, Grid]]:
    """Check if grid is a tiling of smaller grid.

    Returns (rows, cols, base_grid) if tiled, None otherwise.
    """
    h, w = grid.shape

    # Try different tile sizes
    for tile_h in range(1, h // 2 + 1):
        if h % tile_h != 0:
            continue
        for tile_w in range(1, w // 2 + 1):
            if w % tile_w != 0:
                continue

            # Extract base tile
            base = grid[:tile_h, :tile_w]

            # Check if entire grid is this tile repeated
            rows = h // tile_h
            cols = w // tile_w
            tiled = tile_grid(base, rows, cols)

            if np.array_equal(grid, tiled):
                return rows, cols, base

    return None


# ============================================================================
# Connected Components
# ============================================================================

def connected_components(grid: Grid, connectivity: int = 4,
                        background: Optional[int] = None) -> List[Set[Tuple[int, int]]]:
    """Extract connected components (4 or 8 connected).

    Args:
        grid: Input grid
        connectivity: 4 or 8
        background: Background color to ignore (default: 0)

    Returns:
        List of sets of (row, col) positions for each component
    """
    if background is None:
        background = 0

    visited = set()
    components = []
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def flood_fill(r: int, c: int, color: int) -> Set[Tuple[int, int]]:
        stack = [(r, c)]
        component = set()
        while stack:
            cr, cc = stack.pop()
            if ((cr, cc) in visited or
                not (0 <= cr < rows and 0 <= cc < cols) or
                grid[cr, cc] != color):
                continue
            visited.add((cr, cc))
            component.add((cr, cc))
            for dr, dc in directions:
                stack.append((cr + dr, cc + dc))
        return component

    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited and grid[r, c] != background:
                comp = flood_fill(r, c, grid[r, c])
                if comp:
                    components.append(comp)
    return components


def connected_components_by_color(grid: Grid, connectivity: int = 4,
                                 background: Optional[int] = None) -> Dict[int, List[Set[Tuple[int, int]]]]:
    """Group connected components by color."""
    components = connected_components(grid, connectivity, background)
    by_color: Dict[int, List[Set[Tuple[int, int]]]] = {}

    for comp in components:
        r, c = next(iter(comp))
        color = int(grid[r, c])
        if color not in by_color:
            by_color[color] = []
        by_color[color].append(comp)

    return by_color


# ============================================================================
# Grid Comparison and Matching
# ============================================================================

def grids_equal(grid1: Grid, grid2: Grid) -> bool:
    """Check if two grids are exactly equal."""
    return np.array_equal(grid1, grid2)


def grids_equal_up_to_transform(grid1: Grid, grid2: Grid) -> Optional[str]:
    """Check if grid2 can be obtained from grid1 by a transformation.

    Returns the transformation name if found, None otherwise.
    """
    transforms = {
        'identity': lambda g: g,
        'rotate_90': rotate_90,
        'rotate_180': rotate_180,
        'rotate_270': rotate_270,
        'reflect_h': reflect_horizontal,
        'reflect_v': reflect_vertical,
    }

    if grid1.shape == grid2.shape:
        transforms['reflect_d'] = reflect_diagonal
        transforms['reflect_ad'] = reflect_antidiagonal

    for name, transform in transforms.items():
        try:
            if np.array_equal(transform(grid1), grid2):
                return name
        except:
            continue

    return None


def compute_diff(grid1: Grid, grid2: Grid) -> Optional[Grid]:
    """Compute difference between grids.

    Returns grid where 0 = same, 1 = different. None if shapes differ.
    """
    if grid1.shape != grid2.shape:
        return None
    return (grid1 != grid2).astype(np.int8)


def hamming_distance(grid1: Grid, grid2: Grid) -> Optional[int]:
    """Compute Hamming distance (number of differing cells).

    Returns None if shapes differ.
    """
    if grid1.shape != grid2.shape:
        return None
    return int(np.sum(grid1 != grid2))


# ============================================================================
# Grid Overlays and Composition
# ============================================================================

def overlay_grid(base: Grid, overlay: Grid, offset: Tuple[int, int],
                 transparent_color: Optional[int] = None) -> Grid:
    """Overlay one grid onto another at specified offset.

    Args:
        base: Base grid
        overlay: Grid to overlay
        offset: (row_offset, col_offset) position to place overlay
        transparent_color: Color in overlay to treat as transparent

    Returns:
        New grid with overlay applied
    """
    result = base.copy()
    r_off, c_off = offset
    h, w = overlay.shape
    base_h, base_w = base.shape

    for r in range(h):
        for c in range(w):
            target_r = r + r_off
            target_c = c + c_off

            if not (0 <= target_r < base_h and 0 <= target_c < base_w):
                continue

            color = overlay[r, c]
            if transparent_color is None or color != transparent_color:
                result[target_r, target_c] = color

    return result


def merge_grids(grids: List[Grid], fill_value: int = 0) -> Grid:
    """Merge multiple grids into one by finding bounding box."""
    if not grids:
        return create_grid(0, 0)

    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)

    result = create_grid(max_h, max_w, fill_value)

    for grid in grids:
        h, w = grid.shape
        result[:h, :w] = np.maximum(result[:h, :w], grid)

    return result