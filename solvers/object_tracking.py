"""
Object transformation tracking solver.

Tracks how objects transform between input and output grids.
Useful for tasks involving object movement, rotation, scaling, or color changes.
"""

from typing import List, Optional, Tuple, Dict, Callable
import numpy as np

from ..arc_core.grid import Grid
from ..arc_core import grid as grid_ops
from ..arc_core import objects as obj_ops
from ..arc_core.objects import Object


def solve_by_object_tracking(train_examples: List[Tuple[Grid, Grid]]) -> Optional[Callable]:
    """
    Infer transformation by tracking object changes.

    Args:
        train_examples: List of (input, output) grid pairs

    Returns:
        Transformation function if pattern found, None otherwise
    """
    # Extract objects from all examples
    transformations = []

    for input_grid, output_grid in train_examples:
        # Extract objects
        input_objs = obj_ops.extract_objects(input_grid, connectivity=4)
        output_objs = obj_ops.extract_objects(output_grid, connectivity=4)

        # Try to match objects
        transform = _match_objects(input_objs, output_objs)
        if transform is None:
            return None

        transformations.append(transform)

    # Check if all transformations are consistent
    if not _check_consistency(transformations):
        return None

    # Return the transformation function
    return _build_transform_function(transformations[0])


def _match_objects(input_objs: List[Object], output_objs: List[Object]) -> Optional[Dict]:
    """Match input objects to output objects and infer transformation."""

    if len(input_objs) != len(output_objs):
        # Handle case where number of objects changes
        return {
            'type': 'count_change',
            'input_count': len(input_objs),
            'output_count': len(output_objs)
        }

    # Try to match by color
    input_by_color = {}
    for obj in input_objs:
        if obj.color not in input_by_color:
            input_by_color[obj.color] = []
        input_by_color[obj.color].append(obj)

    output_by_color = {}
    for obj in output_objs:
        if obj.color not in output_by_color:
            output_by_color[obj.color] = []
        output_by_color[obj.color].append(obj)

    # Check for simple color mapping
    if set(input_by_color.keys()) == set(output_by_color.keys()):
        # Same colors, check for spatial transformations
        transform = _infer_spatial_transform(input_objs, output_objs)
        if transform:
            return transform

    # Check for color change
    if len(input_objs) == len(output_objs):
        color_mapping = {}
        for i, in_obj in enumerate(input_objs):
            # Find best match in output
            best_match = None
            best_score = -1

            for out_obj in output_objs:
                score = _object_similarity(in_obj, out_obj)
                if score > best_score:
                    best_score = score
                    best_match = out_obj

            if best_match and best_score > 0.5:
                color_mapping[in_obj.color] = best_match.color

        if len(color_mapping) > 0:
            return {
                'type': 'color_change',
                'mapping': color_mapping
            }

    return None


def _object_similarity(obj1: Object, obj2: Object) -> float:
    """Compute similarity between two objects (0 to 1)."""
    score = 0.0

    # Position similarity
    pos_diff = abs(obj1.centroid[0] - obj2.centroid[0]) + abs(obj1.centroid[1] - obj2.centroid[1])
    pos_score = 1.0 / (1.0 + pos_diff / 10.0)
    score += pos_score * 0.3

    # Area similarity
    area_ratio = min(obj1.area, obj2.area) / max(obj1.area, obj2.area)
    score += area_ratio * 0.3

    # Shape similarity
    if obj1.shape_type == obj2.shape_type:
        score += 0.4

    return score


def _infer_spatial_transform(input_objs: List[Object], output_objs: List[Object]) -> Optional[Dict]:
    """Infer spatial transformation between object sets."""

    if len(input_objs) == 0:
        return None

    # Check for translation
    translations = []
    for in_obj in input_objs:
        for out_obj in output_objs:
            if in_obj.color == out_obj.color and in_obj.area == out_obj.area:
                dr = out_obj.centroid[0] - in_obj.centroid[0]
                dc = out_obj.centroid[1] - in_obj.centroid[1]
                translations.append((dr, dc))

    if len(translations) == len(input_objs):
        # Check if all translations are the same
        if all(t == translations[0] for t in translations):
            return {
                'type': 'translation',
                'delta': translations[0]
            }

    # Check for rotation
    # (Simplified - would need more sophisticated matching)

    # Check for scaling
    scales = []
    for in_obj in input_objs:
        for out_obj in output_objs:
            if in_obj.color == out_obj.color:
                if in_obj.width > 0 and in_obj.height > 0:
                    scale_w = out_obj.width / in_obj.width
                    scale_h = out_obj.height / in_obj.height
                    if abs(scale_w - scale_h) < 0.1:  # Uniform scaling
                        scales.append(scale_w)

    if len(scales) == len(input_objs):
        if all(abs(s - scales[0]) < 0.1 for s in scales):
            return {
                'type': 'scaling',
                'factor': int(round(scales[0]))
            }

    return None


def _check_consistency(transformations: List[Dict]) -> bool:
    """Check if all transformations are consistent."""
    if not transformations:
        return False

    first_type = transformations[0]['type']

    for transform in transformations:
        if transform['type'] != first_type:
            return False

    # Type-specific consistency checks
    if first_type == 'translation':
        first_delta = transformations[0]['delta']
        for transform in transformations:
            if transform['delta'] != first_delta:
                return False

    elif first_type == 'scaling':
        first_factor = transformations[0]['factor']
        for transform in transformations:
            if transform['factor'] != first_factor:
                return False

    elif first_type == 'color_change':
        first_mapping = transformations[0]['mapping']
        for transform in transformations:
            if transform['mapping'] != first_mapping:
                return False

    return True


def _build_transform_function(transform: Dict) -> Callable:
    """Build a transformation function from inferred pattern."""

    transform_type = transform['type']

    if transform_type == 'translation':
        dr, dc = transform['delta']
        def translate_grid(grid: Grid) -> Grid:
            objs = obj_ops.extract_objects(grid)
            h, w = grid.shape
            result = grid_ops.create_grid(h, w, 0)

            for obj in objs:
                translated = obj.translate(int(dr), int(dc))
                for r, c in translated.positions:
                    if 0 <= r < h and 0 <= c < w:
                        result[r, c] = translated.color

            return result

        return translate_grid

    elif transform_type == 'scaling':
        factor = transform['factor']
        def scale_grid(grid: Grid) -> Grid:
            return grid_ops.scale_grid(grid, factor)

        return scale_grid

    elif transform_type == 'color_change':
        mapping = transform['mapping']
        def remap_grid(grid: Grid) -> Grid:
            return grid_ops.remap_colors(grid, mapping)

        return remap_grid

    else:
        return lambda grid: grid


def solve_test_case(test_input: Grid, transform_fn: Callable) -> Grid:
    """Apply learned transformation to test input."""
    return transform_fn(test_input)
