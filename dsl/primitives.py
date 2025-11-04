from .ast import Primitive, GridType, ObjSetType, ObjType, IntType, BoolType

# Grid primitives
components = Primitive("components")  # Grid -> ObjSet
crop = Primitive("crop")  # Grid, Int, Int, Int, Int -> Grid
paint = Primitive("paint")  # Grid, ObjSet, Int -> Grid
mirror_h = Primitive("mirror_h")  # Grid -> Grid
mirror_v = Primitive("mirror_v")  # Grid -> Grid

# ObjSet primitives
filter_color = Primitive("filter_color")  # ObjSet, Int -> ObjSet
argmax_area = Primitive("argmax_area")  # ObjSet -> Obj
count = Primitive("count")  # ObjSet -> Int

# Obj primitives
translate = Primitive("translate")  # Obj, Int, Int -> Obj
get_color = Primitive("get_color")  # Obj -> Int

# Int primitives
add = Primitive("add")  # Int, Int -> Int
sub = Primitive("sub")  # Int, Int -> Int

# Bool primitives
eq = Primitive("eq")  # Int, Int -> Bool
gt = Primitive("gt")  # Int, Int -> Bool