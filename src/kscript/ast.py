"""AST node definitions for KScript language."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

@dataclass
class NodeType:
    id: str
    line: int
    column: int

@dataclass 
class Signature(NodeType):
    """A signature identifier [A-Z]+ with optional comment."""

    comment: str | None  # optional (...)

@dataclass
class StringLiteral(NodeType):
    """A string literal "..."."""

@dataclass
class NumberLiteral(NodeType):
    """A number literal [0-9]+."""

# Node types that can appear in constructs
Node: TypeAlias = Signature | StringLiteral | NumberLiteral


class ConstructType(Enum):
    """Types of construct operators."""

    COUNTERSIGN = "=="  # bidirectional
    CANONIZE_FWD = "=>"  # forward canonization
    CANONIZE_BWD = "<="  # backward canonization
    CONNOTATE_FWD = ">"  # forward connotation
    CONNOTATE_BWD = "<"  # backward connotation
    UNDERSIGN = "="  # undersign


@dataclass
class Construct:
    """A construct: operator followed by nodes."""

    type: ConstructType
    nodes: list[Node]
    line: int
    has_leading_nodes: bool = False  # True if nodes include signatures before operator


@dataclass
class Script:
    """A script: signature with constructs and optional subscripts."""

    signature: Signature
    constructs: list[Construct]
    subscripts: list["Script"]
    line: int


@dataclass
class KScriptFile:
    """A complete KScript file with multiple top-level scripts."""

    scripts: list[Script]
