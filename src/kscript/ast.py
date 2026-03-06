"""AST node dataclasses for KScript parser."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class SignificanceType(Enum):
    """Significance level for KLine relationships."""

    S1 = "="  # Prefix match
    S2 = "=>"  # Partial positional
    S3_FORWARD = ">"  # Unordered (forward)
    S3_BACKWARD = "<"  # Unordered (backward)
    S4 = "!="  # No match


@dataclass
class Identifier:
    """An unquoted string identifier (KNode name or file path)."""

    name: str
    line: int
    column: int


@dataclass
class LoadStatement:
    """Load command: load <path>"""

    path: Identifier


@dataclass
class SaveStatement:
    """Save command: save [path]"""

    path: Identifier | None = None


# Forward reference for recursive types
KScriptStatement = Union["LoadStatement", "SaveStatement", "KLineExpr"]


@dataclass
class KScriptAst:
    """Root node containing a sequence of statements."""

    statements: list[KScriptStatement] = field(default_factory=list)

    # Convenience properties for backward compatibility
    @property
    def root(self) -> KScriptStatement | None:
        """Get the first relationship's significance, if any."""
        return self.statements[0] if self.statements else None

@dataclass
class KNodeRef:
    """
    Reference to a KNode.

    Can be either:
    - A simple identifier reference: `hello`
    - A nested KLine expression: `S < M` (when used in indented blocks)
    """

    identifier: Identifier
    # Optional nested KLine - when set, this node is actually a nested KLine
    # that should be compiled and then referenced
    nested_kline: "KLineExpr | None" = None


# KSig can be either an identifier or a nested KLine expression
KSig = Union[Identifier, "KLineExpr"]


@dataclass
class KLineRelationship:
    """A single significance relationship in a KLine."""

    significance: SignificanceType
    nodes: list[KNodeRef] = field(default_factory=list)


@dataclass
class KLineExpr:
    """
    KLine expression with optional significance relationships.

    Forms:
    - `name` -> Simple KLine with just a name
    - `name = node` -> KLine with S1 relationship to node
    - `name => node1 node2` -> KLine with S2 relationship to nodes
    - `name > node1 node2` -> KLine with S3 forward relationship
    - `name < node1 node2` -> KLine with S3 backward relationship
    - `name != node` -> KLine with S4 relationship
    - `kline1 > nodes` -> KLine referencing another KLine

    Multi-line form:
        MHALL = SVO =>
            S < M
            V < H
            O < ALL

    This creates MHALL with S1->SVO and S2->[S<M, V<H, O<ALL]
    """

    sig: KSig  # Identifier or nested KLineExpr
    relationships: list[KLineRelationship] = field(default_factory=list)
    line: int = 0
    column: int = 0

    # Convenience properties for backward compatibility
    @property
    def significance(self) -> SignificanceType | None:
        """Get the first relationship's significance, if any."""
        return self.relationships[0].significance if self.relationships else None

    @property
    def nodes(self) -> list[KNodeRef]:
        """Get the first relationship's nodes, if any."""
        return self.relationships[0].nodes if self.relationships else []
