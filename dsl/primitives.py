"""
Domain-Specific Language Primitives for ARC.

Provides a rich library of primitive operations organized by type:
- Grid operations (transformations, filters, compositions)
- Object operations (extraction, manipulation, queries)
- Numeric operations (arithmetic, comparisons)
- Boolean operations (logic, conditions)
- Set operations (filters, aggregations)
"""

from typing import Callable, Any, List, Set as PySet, Tuple, Optional, Dict
import numpy as np
from ..arc_core.grid import Grid
from ..arc_core import grid as grid_ops
from ..arc_core import objects as obj_ops


# ============================================================================
# Type Definitions
# ============================================================================

# Runtime types for DSL
GridT = Grid
ObjectT = obj_ops.Object
ObjectSetT = List[obj_ops.Object]
IntT = int
BoolT = bool
ColorT = int


class Primitive:
    """Represents a primitive operation in the DSL."""

    def __init__(self, name: str, impl: Callable, type_sig: str, description: str = ""):
        self.name = name
        self.impl = impl
        self.type_sig = type_sig
        self.description = description

    def __call__(self, *args, **kwargs):
        return self.impl(*args, **kwargs)

    def __repr__(self):
        return f"Primitive({self.name}: {self.type_sig})"


# ============================================================================
# Grid Primitives
# ============================================================================

# Grid -> Grid transformations
ROTATE_90 = Primitive(
    "rotate_90",
    grid_ops.rotate_90,
    "Grid -> Grid",
    "Rotate grid 90 degrees clockwise"
)

ROTATE_180 = Primitive(
    "rotate_180",
    grid_ops.rotate_180,
    "Grid -> Grid",
    "Rotate grid 180 degrees"
)

ROTATE_270 = Primitive(
    "rotate_270",
    grid_ops.rotate_270,
    "Grid -> Grid",
    "Rotate grid 270 degrees clockwise"
)

REFLECT_H = Primitive(
    "reflect_h",
    grid_ops.reflect_horizontal,
    "Grid -> Grid",
    "Reflect grid horizontally"
)

REFLECT_V = Primitive(
    "reflect_v",
    grid_ops.reflect_vertical,
    "Grid -> Grid",
    "Reflect grid vertically"
)

REFLECT_D = Primitive(
    "reflect_d",
    grid_ops.reflect_diagonal,
    "Grid -> Grid",
    "Reflect grid along main diagonal"
)

REFLECT_AD = Primitive(
    "reflect_ad",
    grid_ops.reflect_antidiagonal,
    "Grid -> Grid",
    "Reflect grid along anti-diagonal"
)

TRIM_BORDERS = Primitive(
    "trim_borders",
    lambda g: grid_ops.trim_borders(g),
    "Grid -> Grid",
    "Trim constant borders from grid"
)


# Grid -> Grid with parameters
def make_scale(factor: int) -> Callable:
    return lambda g: grid_ops.scale_grid(g, factor)


SCALE_2X = Primitive(
    "scale_2x",
    make_scale(2),
    "Grid -> Grid",
    "Scale grid 2x"
)

SCALE_3X = Primitive(
    "scale_3x",
    make_scale(3),
    "Grid -> Grid",
    "Scale grid 3x"
)


def make_downscale(factor: int) -> Callable:
    return lambda g: grid_ops.downscale_grid(g, factor)


DOWNSCALE_2X = Primitive(
    "downscale_2x",
    make_downscale(2),
    "Grid -> Grid",
    "Downscale grid by 2"
)

DOWNSCALE_3X = Primitive(
    "downscale_3x",
    make_downscale(3),
    "Grid -> Grid",
    "Downscale grid by 3"
)


# Grid tiling
def make_tile(rows: int, cols: int) -> Callable:
    return lambda g: grid_ops.tile_grid(g, rows, cols)


TILE_2X2 = Primitive(
    "tile_2x2",
    make_tile(2, 2),
    "Grid -> Grid",
    "Tile grid 2x2"
)

TILE_3X3 = Primitive(
    "tile_3x3",
    make_tile(3, 3),
    "Grid -> Grid",
    "Tile grid 3x3"
)


# Grid -> ObjectSet
EXTRACT_OBJECTS = Primitive(
    "extract_objects",
    lambda g: obj_ops.extract_objects(g, connectivity=4),
    "Grid -> ObjectSet",
    "Extract all objects from grid (4-connected)"
)

EXTRACT_OBJECTS_8 = Primitive(
    "extract_objects_8",
    lambda g: obj_ops.extract_objects(g, connectivity=8),
    "Grid -> ObjectSet",
    "Extract all objects from grid (8-connected)"
)


# Grid queries
DETECT_BACKGROUND = Primitive(
    "detect_background",
    grid_ops.detect_background,
    "Grid -> Color",
    "Detect background color"
)

GRID_WIDTH = Primitive(
    "grid_width",
    lambda g: g.shape[1],
    "Grid -> Int",
    "Get grid width"
)

GRID_HEIGHT = Primitive(
    "grid_height",
    lambda g: g.shape[0],
    "Grid -> Int",
    "Get grid height"
)

GET_COLOR_HISTOGRAM = Primitive(
    "get_color_histogram",
    grid_ops.get_color_histogram,
    "Grid -> Dict[Int, Int]",
    "Get color frequency histogram"
)


# Grid composition
OVERLAY_GRID = Primitive(
    "overlay_grid",
    grid_ops.overlay_grid,
    "(Grid, Grid, (Int, Int), Int?) -> Grid",
    "Overlay one grid on another"
)

MERGE_GRIDS = Primitive(
    "merge_grids",
    grid_ops.merge_grids,
    "List[Grid] -> Grid",
    "Merge multiple grids"
)


# ============================================================================
# Object Primitives
# ============================================================================

# Object -> Grid
OBJECT_TO_GRID = Primitive(
    "object_to_grid",
    lambda obj: obj.to_grid(),
    "Object -> Grid",
    "Convert object to grid"
)


# Object transformations
def translate_object(obj: obj_ops.Object, dr: int, dc: int) -> obj_ops.Object:
    return obj.translate(dr, dc)


TRANSLATE = Primitive(
    "translate",
    translate_object,
    "(Object, Int, Int) -> Object",
    "Translate object by (dr, dc)"
)

ROTATE_OBJECT_90 = Primitive(
    "rotate_object_90",
    lambda obj: obj.rotate_90_cw(),
    "Object -> Object",
    "Rotate object 90 degrees clockwise"
)

SCALE_OBJECT = Primitive(
    "scale_object",
    lambda obj, factor: obj.scale(factor),
    "(Object, Int) -> Object",
    "Scale object by factor"
)


# Object queries
GET_COLOR = Primitive(
    "get_color",
    lambda obj: obj.color,
    "Object -> Color",
    "Get object color"
)

GET_AREA = Primitive(
    "get_area",
    lambda obj: obj.area,
    "Object -> Int",
    "Get object area"
)

GET_WIDTH = Primitive(
    "get_width",
    lambda obj: obj.width,
    "Object -> Int",
    "Get object width"
)

GET_HEIGHT = Primitive(
    "get_height",
    lambda obj: obj.height,
    "Object -> Int",
    "Get object height"
)

GET_BBOX = Primitive(
    "get_bbox",
    lambda obj: obj.bbox,
    "Object -> (Int, Int, Int, Int)",
    "Get object bounding box"
)

GET_CENTROID = Primitive(
    "get_centroid",
    lambda obj: obj.centroid,
    "Object -> (Float, Float)",
    "Get object centroid"
)

GET_PERIMETER = Primitive(
    "get_perimeter",
    lambda obj: obj.perimeter,
    "Object -> Int",
    "Get object perimeter"
)

IS_SQUARE = Primitive(
    "is_square",
    lambda obj: obj.shape_type == obj_ops.ShapeType.SQUARE,
    "Object -> Bool",
    "Check if object is a square"
)

IS_RECTANGLE = Primitive(
    "is_rectangle",
    lambda obj: obj.shape_type == obj_ops.ShapeType.RECTANGLE,
    "Object -> Bool",
    "Check if object is a rectangle"
)


# ============================================================================
# ObjectSet Primitives
# ============================================================================

# ObjectSet -> ObjectSet filters
def filter_by_color(objs: ObjectSetT, color: ColorT) -> ObjectSetT:
    return obj_ops.filter_by_color(objs, color)


FILTER_COLOR = Primitive(
    "filter_color",
    filter_by_color,
    "(ObjectSet, Color) -> ObjectSet",
    "Filter objects by color"
)


def filter_by_area_range(objs: ObjectSetT, min_area: int, max_area: int) -> ObjectSetT:
    return obj_ops.filter_by_area(objs, min_area, max_area)


FILTER_AREA = Primitive(
    "filter_area",
    filter_by_area_range,
    "(ObjectSet, Int, Int) -> ObjectSet",
    "Filter objects by area range"
)


def filter_by_shape(objs: ObjectSetT, shape: obj_ops.ShapeType) -> ObjectSetT:
    return obj_ops.filter_by_shape(objs, shape)


FILTER_SQUARES = Primitive(
    "filter_squares",
    lambda objs: filter_by_shape(objs, obj_ops.ShapeType.SQUARE),
    "ObjectSet -> ObjectSet",
    "Filter only square objects"
)

FILTER_RECTANGLES = Primitive(
    "filter_rectangles",
    lambda objs: filter_by_shape(objs, obj_ops.ShapeType.RECTANGLE),
    "ObjectSet -> ObjectSet",
    "Filter only rectangular objects"
)


# ObjectSet -> Object aggregations
ARGMAX_AREA = Primitive(
    "argmax_area",
    obj_ops.find_largest_object,
    "ObjectSet -> Object",
    "Find largest object by area"
)

ARGMIN_AREA = Primitive(
    "argmin_area",
    obj_ops.find_smallest_object,
    "ObjectSet -> Object",
    "Find smallest object by area"
)


def get_first(objs: ObjectSetT) -> Optional[obj_ops.Object]:
    return objs[0] if objs else None


FIRST = Primitive(
    "first",
    get_first,
    "ObjectSet -> Object?",
    "Get first object from set"
)


# ObjectSet -> Int
COUNT = Primitive(
    "count",
    len,
    "ObjectSet -> Int",
    "Count objects in set"
)

COUNT_COLORS = Primitive(
    "count_colors",
    lambda objs: len(set(obj.color for obj in objs)),
    "ObjectSet -> Int",
    "Count unique colors in object set"
)


# ObjectSet -> Grid
OBJECTS_TO_GRID = Primitive(
    "objects_to_grid",
    lambda objs, h, w: obj_ops.objects_to_grid(objs, h, w),
    "(ObjectSet, Int, Int) -> Grid",
    "Reconstruct grid from objects"
)


# ObjectSet sorting
SORT_BY_ROW = Primitive(
    "sort_by_row",
    lambda objs: obj_ops.sort_objects_by_position(objs, by='row'),
    "ObjectSet -> ObjectSet",
    "Sort objects by row position"
)

SORT_BY_COL = Primitive(
    "sort_by_col",
    lambda objs: obj_ops.sort_objects_by_position(objs, by='column'),
    "ObjectSet -> ObjectSet",
    "Sort objects by column position"
)

SORT_BY_AREA = Primitive(
    "sort_by_area",
    lambda objs: sorted(objs, key=lambda o: o.area),
    "ObjectSet -> ObjectSet",
    "Sort objects by area (ascending)"
)


# ============================================================================
# Color Primitives
# ============================================================================

def make_replace_color(old_color: ColorT, new_color: ColorT) -> Callable:
    return lambda g: grid_ops.replace_color(g, old_color, new_color)


def swap_colors(g: Grid, c1: ColorT, c2: ColorT) -> Grid:
    return grid_ops.swap_colors(g, c1, c2)


SWAP_COLORS = Primitive(
    "swap_colors",
    swap_colors,
    "(Grid, Color, Color) -> Grid",
    "Swap two colors in grid"
)


def remap_colors(g: Grid, mapping: Dict[ColorT, ColorT]) -> Grid:
    return grid_ops.remap_colors(g, mapping)


REMAP_COLORS = Primitive(
    "remap_colors",
    remap_colors,
    "(Grid, Dict[Color, Color]) -> Grid",
    "Remap colors according to mapping"
)


# Common color constants
COLOR_BLACK = 0
COLOR_BLUE = 1
COLOR_RED = 2
COLOR_GREEN = 3
COLOR_YELLOW = 4
COLOR_GRAY = 5
COLOR_MAGENTA = 6
COLOR_ORANGE = 7
COLOR_LIGHT_BLUE = 8
COLOR_BROWN = 9


# ============================================================================
# Numeric Primitives
# ============================================================================

ADD = Primitive("add", lambda a, b: a + b, "(Int, Int) -> Int", "Addition")
SUB = Primitive("sub", lambda a, b: a - b, "(Int, Int) -> Int", "Subtraction")
MUL = Primitive("mul", lambda a, b: a * b, "(Int, Int) -> Int", "Multiplication")
DIV = Primitive("div", lambda a, b: a // b if b != 0 else 0, "(Int, Int) -> Int", "Integer division")
MOD = Primitive("mod", lambda a, b: a % b if b != 0 else 0, "(Int, Int) -> Int", "Modulo")

MIN = Primitive("min", min, "(Int, Int) -> Int", "Minimum")
MAX = Primitive("max", max, "(Int, Int) -> Int", "Maximum")
ABS = Primitive("abs", abs, "Int -> Int", "Absolute value")


# ============================================================================
# Boolean Primitives
# ============================================================================

EQ = Primitive("eq", lambda a, b: a == b, "(Int, Int) -> Bool", "Equal")
NEQ = Primitive("neq", lambda a, b: a != b, "(Int, Int) -> Bool", "Not equal")
LT = Primitive("lt", lambda a, b: a < b, "(Int, Int) -> Bool", "Less than")
GT = Primitive("gt", lambda a, b: a > b, "(Int, Int) -> Bool", "Greater than")
LTE = Primitive("lte", lambda a, b: a <= b, "(Int, Int) -> Bool", "Less than or equal")
GTE = Primitive("gte", lambda a, b: a >= b, "(Int, Int) -> Bool", "Greater than or equal")

AND = Primitive("and", lambda a, b: a and b, "(Bool, Bool) -> Bool", "Logical AND")
OR = Primitive("or", lambda a, b: a or b, "(Bool, Bool) -> Bool", "Logical OR")
NOT = Primitive("not", lambda a: not a, "Bool -> Bool", "Logical NOT")


# ============================================================================
# Conditional Primitives
# ============================================================================

def if_then_else(cond: BoolT, then_val: Any, else_val: Any) -> Any:
    return then_val if cond else else_val


IF_THEN_ELSE = Primitive(
    "if_then_else",
    if_then_else,
    "(Bool, T, T) -> T",
    "Conditional expression"
)


# ============================================================================
# Pattern Detection Primitives
# ============================================================================

HAS_SYMMETRY_V = Primitive(
    "has_symmetry_v",
    lambda g: grid_ops.Symmetry.VERTICAL in grid_ops.detect_symmetry(g),
    "Grid -> Bool",
    "Check if grid has vertical symmetry"
)

HAS_SYMMETRY_H = Primitive(
    "has_symmetry_h",
    lambda g: grid_ops.Symmetry.HORIZONTAL in grid_ops.detect_symmetry(g),
    "Grid -> Bool",
    "Check if grid has horizontal symmetry"
)

IS_TILED = Primitive(
    "is_tiled",
    lambda g: grid_ops.is_tiled(g) is not None,
    "Grid -> Bool",
    "Check if grid is a tiling pattern"
)


# ============================================================================
# Primitive Registry
# ============================================================================

ALL_PRIMITIVES = [
    # Grid transformations
    ROTATE_90, ROTATE_180, ROTATE_270,
    REFLECT_H, REFLECT_V, REFLECT_D, REFLECT_AD,
    TRIM_BORDERS,
    SCALE_2X, SCALE_3X,
    DOWNSCALE_2X, DOWNSCALE_3X,
    TILE_2X2, TILE_3X3,

    # Grid -> ObjectSet
    EXTRACT_OBJECTS, EXTRACT_OBJECTS_8,

    # Grid queries
    DETECT_BACKGROUND, GRID_WIDTH, GRID_HEIGHT,
    GET_COLOR_HISTOGRAM,

    # Grid composition
    OVERLAY_GRID, MERGE_GRIDS,

    # Object operations
    OBJECT_TO_GRID, TRANSLATE, ROTATE_OBJECT_90, SCALE_OBJECT,

    # Object queries
    GET_COLOR, GET_AREA, GET_WIDTH, GET_HEIGHT,
    GET_BBOX, GET_CENTROID, GET_PERIMETER,
    IS_SQUARE, IS_RECTANGLE,

    # ObjectSet operations
    FILTER_COLOR, FILTER_AREA, FILTER_SQUARES, FILTER_RECTANGLES,
    ARGMAX_AREA, ARGMIN_AREA, FIRST,
    COUNT, COUNT_COLORS,
    OBJECTS_TO_GRID,
    SORT_BY_ROW, SORT_BY_COL, SORT_BY_AREA,

    # Color operations
    SWAP_COLORS, REMAP_COLORS,

    # Numeric
    ADD, SUB, MUL, DIV, MOD, MIN, MAX, ABS,

    # Boolean
    EQ, NEQ, LT, GT, LTE, GTE,
    AND, OR, NOT,

    # Conditional
    IF_THEN_ELSE,

    # Pattern detection
    HAS_SYMMETRY_V, HAS_SYMMETRY_H, IS_TILED,
]

PRIMITIVES_BY_NAME = {p.name: p for p in ALL_PRIMITIVES}


def get_primitive(name: str) -> Optional[Primitive]:
    """Get primitive by name."""
    return PRIMITIVES_BY_NAME.get(name)


def list_primitives(type_filter: Optional[str] = None) -> List[Primitive]:
    """List all primitives, optionally filtered by return type."""
    if type_filter is None:
        return ALL_PRIMITIVES

    return [p for p in ALL_PRIMITIVES if type_filter in p.type_sig]
