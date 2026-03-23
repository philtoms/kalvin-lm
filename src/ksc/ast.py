"""AST node definitions for KScript language."""

from dataclasses import dataclass
from enum import Enum


@dataclass
class Signature:
    """A signature identifier [A-Z]+ with optional comment."""

    name: str  # A-Z+ identifier
    comment: str | None  # optional (...)
    line: int
    column: int


@dataclass
class StringLiteral:
    """A string literal "..."."""

    value: str
    line: int
    column: int


@dataclass
class NumberLiteral:
    """A number literal [0-9]+."""

    value: str
    line: int
    column: int


# Node types that can appear in constructs
Node = Signature | StringLiteral | NumberLiteral


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
