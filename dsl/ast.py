from typing import Any, List, Union
from abc import ABC, abstractmethod

class Expr(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass

class GridExpr(Expr):
    pass

class ObjSetExpr(Expr):
    pass

class ObjExpr(Expr):
    pass

class IntExpr(Expr):
    pass

class BoolExpr(Expr):
    pass

# Primitives
class Primitive(Expr):
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return self.name

class Apply(Expr):
    def __init__(self, func: Expr, args: List[Expr]):
        self.func = func
        self.args = args
    
    def __repr__(self):
        return f"{self.func}({', '.join(repr(a) for a in self.args)})"

# Types
GridType = type
ObjSetType = type
ObjType = type
IntType = type
BoolType = type