"""
Abstract Syntax Tree for ARC DSL.

Defines AST nodes for representing programs in the domain-specific language.
Includes:
- Expression types (primitives, applications, lambdas, variables)
- Type inference system
- Program serialization/deserialization
- Pretty printing
"""

from typing import Any, List, Optional, Dict, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json


# ============================================================================
# Type System
# ============================================================================

class Type(ABC):
    """Base class for types in the DSL."""

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass


@dataclass(frozen=True)
class PrimitiveType(Type):
    """Primitive type (Grid, Object, Int, Bool, etc.)."""
    name: str

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return isinstance(other, PrimitiveType) and self.name == other.name


@dataclass(frozen=True)
class ListType(Type):
    """List type."""
    elem_type: Type

    def __str__(self) -> str:
        return f"List[{self.elem_type}]"

    def __eq__(self, other) -> bool:
        return isinstance(other, ListType) and self.elem_type == other.elem_type


@dataclass(frozen=True)
class FunctionType(Type):
    """Function type."""
    arg_types: tuple
    return_type: Type

    def __str__(self) -> str:
        if len(self.arg_types) == 0:
            return f"() -> {self.return_type}"
        args_str = ", ".join(str(t) for t in self.arg_types)
        return f"({args_str}) -> {self.return_type}"

    def __eq__(self, other) -> bool:
        return (isinstance(other, FunctionType) and
                self.arg_types == other.arg_types and
                self.return_type == other.return_type)


# Common types
GridType = PrimitiveType("Grid")
ObjectType = PrimitiveType("Object")
ObjectSetType = ListType(ObjectType)
IntType = PrimitiveType("Int")
BoolType = PrimitiveType("Bool")
ColorType = PrimitiveType("Color")


# ============================================================================
# AST Nodes
# ============================================================================

class Expr(ABC):
    """Base class for all expressions."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Expr':
        """Deserialize from dictionary."""
        pass


@dataclass
class Var(Expr):
    """Variable reference."""
    name: str
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "var",
            "name": self.name,
            "var_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Var':
        return cls(d["name"])


@dataclass
class Const(Expr):
    """Constant value (literal)."""
    value: Any
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "const",
            "value": self.value,
            "const_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Const':
        return cls(d["value"])


@dataclass
class Prim(Expr):
    """Primitive operation."""
    name: str
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "prim",
            "name": self.name,
            "prim_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Prim':
        return cls(d["name"])


@dataclass
class Apply(Expr):
    """Function application."""
    func: Expr
    args: List[Expr]
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        if not self.args:
            return f"{self.func}()"
        args_str = ", ".join(repr(arg) for arg in self.args)
        return f"{self.func}({args_str})"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "apply",
            "func": self.func.to_dict(),
            "args": [arg.to_dict() for arg in self.args],
            "apply_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Apply':
        func = _expr_from_dict(d["func"])
        args = [_expr_from_dict(arg) for arg in d["args"]]
        return cls(func, args)


@dataclass
class Lambda(Expr):
    """Lambda abstraction."""
    params: List[str]
    body: Expr
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        params_str = ", ".join(self.params)
        return f"λ{params_str}. {self.body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "lambda",
            "params": self.params,
            "body": self.body.to_dict(),
            "lambda_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Lambda':
        params = d["params"]
        body = _expr_from_dict(d["body"])
        return cls(params, body)


@dataclass
class Let(Expr):
    """Let binding."""
    var: str
    value: Expr
    body: Expr
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        return f"let {self.var} = {self.value} in {self.body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "let",
            "var": self.var,
            "value": self.value.to_dict(),
            "body": self.body.to_dict(),
            "let_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Let':
        var = d["var"]
        value = _expr_from_dict(d["value"])
        body = _expr_from_dict(d["body"])
        return cls(var, value, body)


@dataclass
class IfThenElse(Expr):
    """Conditional expression."""
    cond: Expr
    then_branch: Expr
    else_branch: Expr
    type_: Optional[Type] = None

    def __repr__(self) -> str:
        return f"if {self.cond} then {self.then_branch} else {self.else_branch}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "if",
            "cond": self.cond.to_dict(),
            "then": self.then_branch.to_dict(),
            "else": self.else_branch.to_dict(),
            "if_type": str(self.type_) if self.type_ else None
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IfThenElse':
        cond = _expr_from_dict(d["cond"])
        then_branch = _expr_from_dict(d["then"])
        else_branch = _expr_from_dict(d["else"])
        return cls(cond, then_branch, else_branch)


def _expr_from_dict(d: Dict[str, Any]) -> Expr:
    """Helper to deserialize expressions."""
    expr_type = d["type"]
    if expr_type == "var":
        return Var.from_dict(d)
    elif expr_type == "const":
        return Const.from_dict(d)
    elif expr_type == "prim":
        return Prim.from_dict(d)
    elif expr_type == "apply":
        return Apply.from_dict(d)
    elif expr_type == "lambda":
        return Lambda.from_dict(d)
    elif expr_type == "let":
        return Let.from_dict(d)
    elif expr_type == "if":
        return IfThenElse.from_dict(d)
    else:
        raise ValueError(f"Unknown expression type: {expr_type}")


# ============================================================================
# Program Class
# ============================================================================

@dataclass
class Program:
    """Complete ARC program."""
    expr: Expr
    name: Optional[str] = None
    description: Optional[str] = None

    def __repr__(self) -> str:
        if self.name:
            return f"Program({self.name}): {self.expr}"
        return f"Program: {self.expr}"

    def to_json(self) -> str:
        """Serialize program to JSON."""
        return json.dumps({
            "name": self.name,
            "description": self.description,
            "expr": self.expr.to_dict()
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Program':
        """Deserialize program from JSON."""
        data = json.loads(json_str)
        expr = _expr_from_dict(data["expr"])
        return cls(
            expr=expr,
            name=data.get("name"),
            description=data.get("description")
        )


# ============================================================================
# Utility Functions
# ============================================================================

def size(expr: Expr) -> int:
    """Compute the size of an expression (number of nodes)."""
    if isinstance(expr, (Var, Const, Prim)):
        return 1
    elif isinstance(expr, Apply):
        return 1 + size(expr.func) + sum(size(arg) for arg in expr.args)
    elif isinstance(expr, Lambda):
        return 1 + size(expr.body)
    elif isinstance(expr, Let):
        return 1 + size(expr.value) + size(expr.body)
    elif isinstance(expr, IfThenElse):
        return 1 + size(expr.cond) + size(expr.then_branch) + size(expr.else_branch)
    return 1


def depth(expr: Expr) -> int:
    """Compute the depth of an expression."""
    if isinstance(expr, (Var, Const, Prim)):
        return 1
    elif isinstance(expr, Apply):
        arg_depths = [depth(arg) for arg in expr.args] if expr.args else [0]
        return 1 + max(depth(expr.func), max(arg_depths))
    elif isinstance(expr, Lambda):
        return 1 + depth(expr.body)
    elif isinstance(expr, Let):
        return 1 + max(depth(expr.value), depth(expr.body))
    elif isinstance(expr, IfThenElse):
        return 1 + max(depth(expr.cond), depth(expr.then_branch), depth(expr.else_branch))
    return 1


def get_primitives(expr: Expr) -> List[str]:
    """Get all primitives used in an expression."""
    if isinstance(expr, Prim):
        return [expr.name]
    elif isinstance(expr, Apply):
        result = get_primitives(expr.func)
        for arg in expr.args:
            result.extend(get_primitives(arg))
        return result
    elif isinstance(expr, Lambda):
        return get_primitives(expr.body)
    elif isinstance(expr, Let):
        return get_primitives(expr.value) + get_primitives(expr.body)
    elif isinstance(expr, IfThenElse):
        return (get_primitives(expr.cond) +
                get_primitives(expr.then_branch) +
                get_primitives(expr.else_branch))
    return []


def get_variables(expr: Expr) -> List[str]:
    """Get all variables used in an expression."""
    if isinstance(expr, Var):
        return [expr.name]
    elif isinstance(expr, Apply):
        result = get_variables(expr.func)
        for arg in expr.args:
            result.extend(get_variables(arg))
        return result
    elif isinstance(expr, Lambda):
        return get_variables(expr.body)
    elif isinstance(expr, Let):
        return get_variables(expr.value) + get_variables(expr.body)
    elif isinstance(expr, IfThenElse):
        return (get_variables(expr.cond) +
                get_variables(expr.then_branch) +
                get_variables(expr.else_branch))
    return []


def substitute(expr: Expr, var: str, value: Expr) -> Expr:
    """Substitute variable with value in expression."""
    if isinstance(expr, Var):
        return value if expr.name == var else expr
    elif isinstance(expr, Const):
        return expr
    elif isinstance(expr, Prim):
        return expr
    elif isinstance(expr, Apply):
        new_func = substitute(expr.func, var, value)
        new_args = [substitute(arg, var, value) for arg in expr.args]
        return Apply(new_func, new_args)
    elif isinstance(expr, Lambda):
        if var in expr.params:
            return expr  # Variable shadowed
        new_body = substitute(expr.body, var, value)
        return Lambda(expr.params, new_body)
    elif isinstance(expr, Let):
        new_value = substitute(expr.value, var, value)
        if expr.var == var:
            return Let(expr.var, new_value, expr.body)  # Variable shadowed
        new_body = substitute(expr.body, var, value)
        return Let(expr.var, new_value, new_body)
    elif isinstance(expr, IfThenElse):
        new_cond = substitute(expr.cond, var, value)
        new_then = substitute(expr.then_branch, var, value)
        new_else = substitute(expr.else_branch, var, value)
        return IfThenElse(new_cond, new_then, new_else)
    return expr
