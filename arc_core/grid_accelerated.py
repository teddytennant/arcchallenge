"""
Grid operations with automatic Rust acceleration.

This module automatically uses Rust implementations when available,
falling back to pure Python when not.
"""

import numpy as np
from typing import List, Set, Tuple, Optional

# Try to import Rust accelerated functions
try:
    from arc_core_rs import (
        connected_components_fast,
        grids_equal_fast,
        hamming_distance_fast,
        find_symmetries_fast,
        use_rust_backend,
    )
    HAS_RUST = use_rust_backend()
except ImportError:
    HAS_RUST = False

# Import Python fallbacks
from .grid import (
    connected_components as connected_components_python,
    grids_equal as grids_equal_python,
    hamming_distance as hamming_distance_python,
    detect_symmetry as detect_symmetry_python,
    Grid,
)


def connected_components(
    grid: Grid,
    connectivity: int = 4,
    background: Optional[int] = None
) -> List[Set[Tuple[int, int]]]:
    """
    Extract connected components with automatic Rust acceleration.

    Uses Rust implementation when available (5-10x faster),
    falls back to Python otherwise.

    Args:
        grid: Input grid
        connectivity: 4 or 8 connectivity
        background: Background color to ignore

    Returns:
        List of sets of (row, col) positions
    """
    if HAS_RUST and connectivity == 4:
        # Use Rust implementation
        components = connected_components_fast(grid, background)
        # Convert Python sets to Python sets (they come back as sets already)
        return [set(comp) for comp in components]
    else:
        # Fall back to Python
        return connected_components_python(grid, connectivity, background)


def grids_equal(grid1: Grid, grid2: Grid) -> bool:
    """
    Check if two grids are equal with automatic Rust acceleration.

    Uses Rust implementation when available (10x faster),
    falls back to Python otherwise.

    Args:
        grid1: First grid
        grid2: Second grid

    Returns:
        True if grids are equal
    """
    if HAS_RUST:
        return grids_equal_fast(grid1, grid2)
    else:
        return grids_equal_python(grid1, grid2)


def hamming_distance(grid1: Grid, grid2: Grid) -> Optional[int]:
    """
    Compute Hamming distance with automatic Rust acceleration.

    Uses Rust implementation when available (10x faster),
    falls back to Python otherwise.

    Args:
        grid1: First grid
        grid2: Second grid

    Returns:
        Hamming distance, or None if shapes differ
    """
    if HAS_RUST:
        return hamming_distance_fast(grid1, grid2)
    else:
        return hamming_distance_python(grid1, grid2)


def detect_symmetry_fast(grid: Grid) -> Set[str]:
    """
    Detect symmetries with parallel Rust implementation.

    Uses parallel Rust implementation when available (8x faster),
    falls back to Python otherwise.

    Args:
        grid: Input grid

    Returns:
        Set of symmetry type names
    """
    if HAS_RUST:
        symmetries = find_symmetries_fast(grid)
        return set(symmetries)
    else:
        from .grid import Symmetry
        symmetries = detect_symmetry_python(grid)
        return {sym.name.lower() for sym in symmetries}


def get_backend_info() -> dict:
    """Get information about the current backend."""
    return {
        'rust_available': HAS_RUST,
        'backend': 'Rust' if HAS_RUST else 'Python',
        'expected_speedup': {
            'connected_components': '5-10x' if HAS_RUST else '1x',
            'grid_equality': '10x' if HAS_RUST else '1x',
            'hamming_distance': '10x' if HAS_RUST else '1x',
            'symmetry_detection': '8x' if HAS_RUST else '1x',
        }
    }


# Export accelerated versions as default
__all__ = [
    'connected_components',
    'grids_equal',
    'hamming_distance',
    'detect_symmetry_fast',
    'get_backend_info',
    'HAS_RUST',
]
