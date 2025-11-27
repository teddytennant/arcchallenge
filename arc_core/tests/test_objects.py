import numpy as np
import pytest
from arc_core.objects import (
    Object, ShapeType, extract_objects, extract_objects_by_color,
    get_relationships, Relationship, find_largest_object, find_smallest_object,
    filter_by_color, filter_by_shape, filter_by_area, sort_objects_by_position,
    objects_to_grid
)
from arc_core.grid import load_grid


def test_object_creation():
    """Test basic object creation and properties."""
    positions = {(0, 0), (0, 1), (1, 0), (1, 1)}
    obj = Object(positions, color=1)

    assert obj.color == 1
    assert obj.area == 4
    assert obj.width == 2
    assert obj.height == 2
    assert obj.bbox == (0, 0, 1, 1)


def test_object_centroid():
    """Test centroid calculation."""
    positions = {(0, 0), (0, 2), (2, 0), (2, 2)}
    obj = Object(positions, color=1)

    assert obj.centroid == (1.0, 1.0)


def test_object_perimeter():
    """Test perimeter calculation."""
    # 2x2 square has perimeter of 8
    positions = {(0, 0), (0, 1), (1, 0), (1, 1)}
    obj = Object(positions, color=1)

    assert obj.perimeter == 8


def test_shape_classification_square():
    """Test square shape detection."""
    positions = {(0, 0), (0, 1), (1, 0), (1, 1)}
    obj = Object(positions, color=1)

    assert obj.shape_type == ShapeType.SQUARE


def test_shape_classification_rectangle():
    """Test rectangle shape detection."""
    positions = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}
    obj = Object(positions, color=1)

    assert obj.shape_type == ShapeType.RECTANGLE


def test_shape_classification_line_horizontal():
    """Test horizontal line detection."""
    positions = {(0, 0), (0, 1), (0, 2)}
    obj = Object(positions, color=1)

    # 1x3 is classified as RECTANGLE (1 row, multiple columns)
    assert obj.shape_type == ShapeType.RECTANGLE


def test_shape_classification_line_vertical():
    """Test vertical line detection."""
    positions = {(0, 0), (1, 0), (2, 0)}
    obj = Object(positions, color=1)

    # 3x1 is classified as RECTANGLE (multiple rows, 1 column)
    assert obj.shape_type == ShapeType.RECTANGLE


def test_hollow_rectangle():
    """Test hollow rectangle detection."""
    # Create 3x3 hollow rectangle
    positions = {
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2)
    }
    obj = Object(positions, color=1)

    assert obj.shape_type == ShapeType.HOLLOW_RECT


def test_object_convexity():
    """Test convexity check."""
    # Square is convex
    square = Object({(0, 0), (0, 1), (1, 0), (1, 1)}, color=1)
    assert square.is_convex

    # L-shape is not convex
    l_shape = Object({(0, 0), (1, 0), (2, 0), (2, 1)}, color=1)
    assert not l_shape.is_convex


def test_object_to_grid():
    """Test converting object to grid."""
    positions = {(1, 1), (1, 2), (2, 1), (2, 2)}
    obj = Object(positions, color=5)

    grid = obj.to_grid()
    assert grid.shape == (2, 2)
    assert np.all(grid == 5)


def test_object_translate():
    """Test object translation."""
    positions = {(0, 0), (0, 1)}
    obj = Object(positions, color=1)

    translated = obj.translate(2, 3)
    assert (2, 3) in translated.positions
    assert (2, 4) in translated.positions
    assert len(translated.positions) == 2


def test_object_scale():
    """Test object scaling."""
    positions = {(0, 0), (0, 1)}
    obj = Object(positions, color=1)

    scaled = obj.scale(2)
    assert scaled.area == 8  # 2 cells * 2 * 2


def test_object_overlaps():
    """Test object overlap detection."""
    obj1 = Object({(0, 0), (0, 1)}, color=1)
    obj2 = Object({(0, 1), (0, 2)}, color=2)
    obj3 = Object({(1, 0), (1, 1)}, color=3)

    assert obj1.overlaps(obj2)
    assert not obj1.overlaps(obj3)


def test_object_distance():
    """Test distance calculation between objects."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(2, 3)}, color=2)

    # Manhattan distance: |2-0| + |3-0| = 5
    assert obj1.distance_to(obj2) == 5


def test_extract_objects():
    """Test object extraction from grid."""
    grid = load_grid([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [2, 2, 0, 0]
    ])

    objects = extract_objects(grid, connectivity=4, background=0)

    assert len(objects) == 2
    colors = {obj.color for obj in objects}
    assert colors == {1, 2}


def test_extract_objects_by_color():
    """Test object extraction grouped by color."""
    grid = load_grid([
        [1, 1, 0, 2],
        [0, 1, 0, 2],
        [3, 0, 0, 0]
    ])

    by_color = extract_objects_by_color(grid, connectivity=4, background=0)

    assert 1 in by_color
    assert 2 in by_color
    assert 3 in by_color


def test_get_relationships():
    """Test spatial relationship detection."""
    obj1 = Object({(0, 0), (0, 1)}, color=1)
    obj2 = Object({(2, 0), (2, 1)}, color=2)

    rels = get_relationships(obj1, obj2)

    assert Relationship.ABOVE in rels
    # Objects have same column span, so they're vertically aligned
    assert Relationship.ALIGNED_V in rels


def test_find_largest_object():
    """Test finding largest object."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0), (1, 1), (1, 2)}, color=2)
    obj3 = Object({(2, 0), (2, 1)}, color=3)

    largest = find_largest_object([obj1, obj2, obj3])
    assert largest.area == 3


def test_find_smallest_object():
    """Test finding smallest object."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0), (1, 1), (1, 2)}, color=2)

    smallest = find_smallest_object([obj1, obj2])
    assert smallest.area == 1


def test_filter_by_color():
    """Test filtering objects by color."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0)}, color=2)
    obj3 = Object({(2, 0)}, color=1)

    filtered = filter_by_color([obj1, obj2, obj3], color=1)
    assert len(filtered) == 2
    assert all(obj.color == 1 for obj in filtered)


def test_filter_by_shape():
    """Test filtering objects by shape."""
    square = Object({(0, 0), (0, 1), (1, 0), (1, 1)}, color=1)
    line = Object({(2, 0), (2, 1), (2, 2)}, color=2)

    filtered = filter_by_shape([square, line], ShapeType.SQUARE)
    assert len(filtered) == 1
    assert filtered[0].shape_type == ShapeType.SQUARE


def test_filter_by_area():
    """Test filtering objects by area."""
    obj1 = Object({(0, 0)}, color=1)
    obj2 = Object({(1, 0), (1, 1)}, color=2)
    obj3 = Object({(2, 0), (2, 1), (2, 2)}, color=3)

    filtered = filter_by_area([obj1, obj2, obj3], min_area=2, max_area=3)
    assert len(filtered) == 2
    assert all(2 <= obj.area <= 3 for obj in filtered)


def test_sort_objects_by_position():
    """Test sorting objects by position."""
    obj1 = Object({(2, 0)}, color=1)
    obj2 = Object({(0, 0)}, color=2)
    obj3 = Object({(1, 0)}, color=3)

    sorted_by_row = sort_objects_by_position([obj1, obj2, obj3], by='row')
    assert sorted_by_row[0].color == 2
    assert sorted_by_row[1].color == 3
    assert sorted_by_row[2].color == 1


def test_objects_to_grid():
    """Test reconstructing grid from objects."""
    obj1 = Object({(0, 0), (0, 1)}, color=1)
    obj2 = Object({(1, 0), (1, 1)}, color=2)

    grid = objects_to_grid([obj1, obj2], height=2, width=2, background=0)

    assert grid[0, 0] == 1
    assert grid[0, 1] == 1
    assert grid[1, 0] == 2
    assert grid[1, 1] == 2


def test_empty_object():
    """Test handling of empty objects."""
    obj = Object(set(), color=0)

    assert obj.area == 0
    assert obj.width == 0
    assert obj.height == 0
    assert obj.bbox == (0, 0, 0, 0)
