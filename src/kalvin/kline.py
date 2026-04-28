"""KLine - Fundamental unit of the knowledge graph.

A Kline is an identified, ordered sequence of zero or more nodes.
See openspec/kline.md for the full specification.
"""

from __future__ import annotations

from typing import TypeAlias

# === Core Types ===

KNode: TypeAlias = int

# Type alias for KNodes — accepted input representations
KNodes: TypeAlias = int | None | list[int]

# Type alias for Signatures (uint64)
KSig: TypeAlias = int


class KLine:
    """An identified, ordered sequence of zero or more nodes.

    Attributes:
        signature: uint64 identity key (produced by make_signature).
        nodes: list of uint64 node values (always a list, never None).
        literal: whether this kline represents an exact token.
        dbg_text: optional debug label (not spec'd).
    """

    __slots__ = ("signature", "nodes", "literal", "dbg_text")

    def __init__(
        self,
        signature: KSig,
        nodes: KNodes | KNode | None = None,
        literal: bool = False,
        dbg_text: str = "",
    ):
        self.signature = signature
        self.nodes = _normalize_nodes(nodes)
        self.literal = literal
        self.dbg_text = dbg_text

    def is_literal(self) -> bool:
        """Return whether this kline is literal."""
        return self.literal

    # ── Backwards-compatible helpers ──────────────────────────────────

    def as_node_list(self) -> list[KNode]:
        """Get nodes as a list. Always returns self.nodes (already a list)."""
        return self.nodes

    # ── Equality, hashing ─────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KLine):
            return NotImplemented
        if self.signature != other.signature:
            return False
        if len(self.nodes) != len(other.nodes):
            return False
        return self.nodes == other.nodes

    def __hash__(self) -> int:
        return hash((self.signature, tuple(self.nodes)))

    # ── Repr ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        text = f" {self.dbg_text!r}" if self.dbg_text else ""
        return f"KLine(sig={self.signature:#x}, nodes={self.nodes!r}, lit={self.literal}{text})"

    def __len__(self) -> int:
        return len(self.nodes)


# Type alias for an iterator of KLines
KGraph: TypeAlias = "object"  # Iterator[KLine] — for compat


def _normalize_nodes(nodes: KNodes | KNode | None) -> list[KNode]:
    """Normalize node input to a list[int].

    - None → []
    - int → [int]
    - list → list (as-is)
    """
    if nodes is None:
        return []
    if isinstance(nodes, int):
        return [nodes]
    return list(nodes)
