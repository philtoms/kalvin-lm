"""AST node definitions for KScript v3 compiler.

The AST models operator-delimited scopes (spec §3). Each OperatorScope
holds a signature, an optional operator, items (nodes), and an optional
indented child block. Annotations provide BPE encoding word text.

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
    When used as an item, may carry an ``inline_annotation`` — an annotation
    that immediately follows it in source (e.g. the ``(ave)`` on the second H
    in ``H => D H(ave)``), bound unconditionally to this item by Word Binding.
    """

    id: str  # uppercase
    line: int
    column: int
    inline_annotation: Annotation | None = None  # inline (item-side)


@dataclass
class Annotation:
    """BPE annotation node — parenthesised expression whose text (including the
    enclosing parens) provides word text for BPE token encoding."""

    text: str
    line: int
    column: int


@dataclass
class OperatorScope:
    """A scope created by an operator boundary (§3).

    Scopes are identified by operator boundaries (spec §3).

    Fields:
        sig: The identifier preceding the operator (the scope's signature).
        op: The operator token type, or None for bare (unsigned) signatures.
        items: Nodes and child constructs within this scope. Typed as
            ScopeItem (Signature | Annotation | OperatorScope) to accommodate
            bare Signature nodes per grammar §4: item ::= sig | annotation | operator_scope.
            A Signature item may carry its own ``inline_annotation`` (bound
            unconditionally to that item per Word Binding).
        child_block: Indented child scope extending this operator's scope (§3 Rule S4).
        inline_annotation: Annotation attached to sig-side, e.g. S(ubject) = M.
            A top-level (signature-prefix) annotation: binds fill-if-empty per
            Word Binding (never overriding an outer binding on the same char).
    """

    sig: Signature
    op: TokenType | None = None
    items: list[ScopeItem] = field(default_factory=list)
    child_block: Block | None = None
    inline_annotation: Annotation | None = None  # sig-side (top-level)


@dataclass
class Block:
    """INDENT construct+ DEDENT — an indented block of constructs."""

    constructs: list[ConstructItem]


@dataclass
class KScriptFile:
    """Top-level file container.

    No Script wrapper — KScriptFile holds constructs directly since
    there is exactly one script per file.
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
