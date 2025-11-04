from typing import List, Generator
from ..dsl.ast import Expr, Apply, Primitive
from ..dsl.primitives import *

def enumerate_programs(max_depth: int = 3) -> Generator[Expr, None, None]:
    """Simple enumerative generator of programs."""
    # For now, just yield some hardcoded examples
    yield mirror_h
    yield Apply(mirror_h, [])
    yield Apply(components, [])
    yield Apply(paint, [Apply(components, []), Primitive("1")])  # dummy