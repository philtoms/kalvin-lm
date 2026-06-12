"""AST node definitions for KScript v3 compiler.

Key differences from old src/kscript/ast.py:
- No chain_right field. OperatorScope replaces old Construct with chain links.
  Scopes are identified by operator boundaries (§3).
- Annotation replaces Comment — reflects BPE encoding purpose.
- ScopeItem type for OperatorScope.items includes bare Signature nodes.
- KScriptFile has no Script wrapper — single script per file.

Spec ref: @specs/kscript.md v3.0 §4–5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from .token import TokenType


@dataclass
class Signature:
    """An uppercase identifier [A-Z]+.

    Can appear as a scope's signature or as a bare node within a scope's items.
    """
    id: str       # uppercase
    line: int
    column: int


@dataclass
class Annotation:
    """BPE annotation node — parenthesised expression providing word text for
    BPE token encoding.

    Renamed from Comment in the old AST to reflect its semantic purpose.
    The text includes enclosing parentheses.
    """
    text: str
    line: int
    column: int


@dataclass
class OperatorScope:
    """A scope created by an operator boundary (§3).

    Replaces the old Construct with chain_right. Scopes are identified by
    operator boundaries, not by chain links.

    Fields:
        sig: The identifier preceding the operator (the scope's signature).
        op: The operator token type, or None for bare (unsigned) signatures.
        items: Nodes and child constructs within this scope. Typed as
            ScopeItem (Signature | Annotation | OperatorScope) to accommodate
            bare Signature nodes per grammar §4: item ::= sig | annotation | operator_scope.
        child_block: Indented child scope extending this operator's scope (§3 Rule S4).
        inline_annotation: Annotation attached to sig-side, e.g. S(ubject) = M.
            Singular — one annotation per signature position.
        node_inline_annotation: Annotation attached to the first node-side
            Signature, e.g. A = D(et). Singular — for multi-node scopes with
            multiple inline-annotated nodes, the parser (KB-187) is responsible
            for attaching subsequent annotations via the items list (as Annotation
            entries preceding the annotated Signature) or another mechanism.
    """
    sig: Signature
    op: TokenType | None = None
    items: list[ScopeItem] = field(default_factory=list)
    child_block: Block | None = None
    inline_annotation: Annotation | None = None       # sig-side
    node_inline_annotation: Annotation | None = None   # node-side (first node)


@dataclass
class Block:
    """INDENT construct+ DEDENT — an indented block of constructs."""
    constructs: list[ConstructItem]


@dataclass
class KScriptFile:
    """Top-level file container.

    Deviation from spec §5: no Script wrapper — KScriptFile holds constructs
    directly since there is exactly one script per file.
    """
    constructs: list[ConstructItem]


# Type aliases

ConstructItem: TypeAlias = "Annotation | OperatorScope | Block"
"""Top-level construct types — what can appear in Block.constructs and KScriptFile.constructs.
Per grammar §4: construct ::= block | annotation | operator_scope.
"""

ScopeItem: TypeAlias = "Signature | Annotation | OperatorScope"
"""Items within an OperatorScope — what can appear in OperatorScope.items.
Per grammar §4: item ::= sig | annotation | operator_scope.
Note: Block is NOT a ScopeItem. Indented content goes to child_block, not items.
"""
