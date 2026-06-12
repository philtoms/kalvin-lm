"""KLine - Fundamental unit of the knowledge graph.

A Kline is an identified, ordered sequence of zero or more nodes.
See specs/kline.md for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

# === Core Types ===

KNode: TypeAlias = int

# Type alias for KNodes — accepted input representations
KNodes: TypeAlias = int | None | list[int]

# Type alias for Signatures (uint64)
KSig: TypeAlias = int


@dataclass
class KDbg:
    """Rich debug info for a KLine or CompiledEntry (not spec'd).

    Populated in dev mode by the token encoder.  Forwarded as-is
    by misfit expansions and model duplication.

    Attributes:
        label: Origin word or operator context.
        decoded: Tokenizer decode of the signature (actual subword text).
        pos: Part-of-speech tag (e.g. "PROPN", "VERB", "NOUN").
        dep: Dependency relation (e.g. "appos", "nsubj").
        morph: Morphological features (e.g. "Number=Sing").
    """

    label: str = ""
    decoded: str = ""
    pos: str = ""
    dep: str = ""
    morph: str = ""

    def __bool__(self) -> bool:
        """Truthy when any field is non-empty."""
        return bool(self.label or self.decoded or self.pos or self.dep or self.morph)

    def __repr__(self) -> str:
        parts = []
        if self.label:
            parts.append(self.label)
        if self.decoded and self.decoded != self.label:
            parts.append(f"decoded={self.decoded!r}")
        if self.pos:
            parts.append(f"pos={self.pos}")
        if self.dep:
            parts.append(f"dep={self.dep}")
        if self.morph:
            parts.append(f"morph={self.morph}")
        return f"KDbg({', '.join(parts)})" if parts else "KDbg()"


class KLine:
    """An identified, ordered sequence of zero or more nodes.

    Attributes:
        signature: uint64 identity key (produced by make_signature).
        nodes: list of uint64 node values (always a list, never None).
        dbg: optional debug info (not spec'd).
    """

    __slots__ = ("signature", "nodes", "dbg", "sig_level")

    def __init__(
        self,
        signature: KSig,
        nodes: KNodes | KNode | None = None,
        dbg: KDbg | None = None,
        sig_level: str | None = None,
    ):
        self.signature = signature
        self.nodes = _normalize_nodes(nodes)
        self.dbg = dbg
        self.sig_level = sig_level

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
        text = f" {self.dbg}" if self.dbg else ""
        return f"KLine(sig={self.signature:#x}, nodes={self.nodes!r}{text})"

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
