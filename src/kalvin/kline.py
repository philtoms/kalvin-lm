"""KLine - Fundamental unit of Kalvin's memory.

A Kline is an identified, ordered sequence of zero or more nodes.
See specs/kline.md for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

# === Core Types ===

KNode: TypeAlias = int

# Type alias for KNodes — accepted input representations
KNodes: TypeAlias = int | None | list[int]

# Type alias for Signatures (uint64)
KSig: TypeAlias = int


@dataclass
class KDbg:
    """Provenance metadata for a KLine (not spec'd).

    Populated by the token encoder during compilation.  Forwarded as-is
    by misfit expansions and model duplication.

    Attributes:
        op: Structural state (COUNTERSIGNED, UNDERSIGNED, CONNOTED,
            CANONIZED, IDENTITY).
        label: Origin word or operator context.
        decoded: Tokenizer decode of the signature (actual subword text).
        type_info: Short debug summary of the node's type-dictionary entry
            (e.g. NLP POS/DEP/MORPH labels when the dictionary was generated
            by NLP tooling). Opaque to kalvin.
    """

    op: str = "IDENTITY"
    label: str = ""
    decoded: str = ""
    type_info: str = ""

    def __bool__(self) -> bool:
        """Truthy when any field is non-empty."""
        return bool(
            self.op != "IDENTITY"
            or self.label
            or self.decoded
            or self.type_info
        )

    def __repr__(self) -> str:
        parts = []
        if self.op != "IDENTITY":
            parts.append(f"op={self.op}")
        if self.label:
            parts.append(self.label)
        if self.decoded and self.decoded != self.label:
            parts.append(f"decoded={self.decoded!r}")
        if self.type_info:
            parts.append(f"type_info={self.type_info!r}")
        return f"KDbg({', '.join(parts)})" if parts else "KDbg()"


class KLine:
    """An identified, ordered sequence of zero or more nodes.

    Attributes:
        signature: uint64 identity key (produced by make_signature).
        nodes: list of uint64 node values (always a list, never None).
        dbg: optional debug info (not spec'd).
    """

    __slots__ = ("signature", "nodes", "dbg")

    def __init__(
        self,
        signature: KSig,
        nodes: KNodes | KNode | None = None,
        dbg: KDbg | None = None,
    ):
        self.signature = signature
        self.nodes = _normalize_nodes(nodes)
        self.dbg = dbg

    # Backwards-compatible helpers

    def as_node_list(self) -> list[KNode]:
        """Get nodes as a list. Always returns self.nodes (already a list)."""
        return self.nodes

    # Equality, hashing

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

    # Repr

    def __repr__(self) -> str:
        text = f" {self.dbg}" if self.dbg else ""
        return f"KLine(sig={self.signature:#x}, nodes={self.nodes!r}{text})"

    def __len__(self) -> int:
        return len(self.nodes)


# Type alias for an iterator of KLines
KGraph: TypeAlias = "object"  # Iterator[KLine] — for compat


# === Structural predicates ===
#
# Identity and canon are structural properties of a KLine (they depend only
# on its signature and nodes, not on model state). Defined here so every
# module agrees on what counts as identity vs canon. See @kline spec and
# @cogitator spec §Universal Constraint.


def is_identity(kline: KLine) -> bool:
    """Test whether a kline is an identity.

    A kline is identity when it carries no decomposition — either form:
      - empty nodes: ``{S: []}``, or
      - self-referential: ``{S: [S]}`` — its own signature is its sole node.

    The self-referential form is identity *by definition*: a value that
    decomposes into itself carries no further information. This overrules any
    canon classification (see :func:`is_canon`).
    """
    if not kline.nodes:
        return True
    return kline.nodes == [kline.signature]


def is_canon(kline: KLine) -> bool:
    """Test whether a kline is canonical.

    A kline is canonical when it is neither identity nor self-referential AND
    its signature is the OR-reduction of its nodes
    (``signature == make_signature(nodes)``).

    ``{S: [S]}`` is NOT canonical — it is identity (see :func:`is_identity`).
    Allowing it as a canon would let it displace the genuine canon for the
    same signature under Recency Precedence, collapsing ``unpack()`` to
    identity. No relational token (``==``, ``=``, ``>``, ``=>``) produces a
    self-referential kline.
    """
    if not kline.nodes:
        return False  # identity
    if kline.signature in kline.nodes:
        return False  # self-referential — identity, not canon
    from kalvin.signature import make_signature

    return kline.signature == make_signature(kline.nodes)


# Display helper

_OP_SYMBOLS = {
    "COUNTERSIGNED": "==",
    "UNDERSIGNED": "=",
    "CONNOTED": ">",
    "CANONIZED": "=>",
    "IDENTITY": None,
}

_SIG_LEVELS = {
    "COUNTERSIGNED": "S1",
    "UNDERSIGNED": "S3",
    "CANONIZED": "S2",
    "CONNOTED": "S3",
    "IDENTITY": "S4",
}


def sig_level(kline: KLine) -> str:
    """Return significance level (S1–S4) for a KLine.

    Uses dbg.op when available, infers from structure otherwise.
    """
    if kline.dbg and kline.dbg.op:
        return _SIG_LEVELS.get(kline.dbg.op, "S4")
    return _infer_level(kline)


def kline_display(kline: KLine, tokenizer: object) -> str:
    """Format a KLine as human-readable KScript source.

    Uses dbg provenance when available (label, op). Falls back to
    tokenizer decoding and structural inference when dbg is absent.

    Args:
        kline: The KLine to display.
        tokenizer: A KTokenizer for decoding uint64 values to strings.

    Returns:
        KScript-like source string (e.g. "M == H", "ABC => A B C").
    """
    # Resolve signature name
    if kline.dbg and kline.dbg.label:
        sig_name = kline.dbg.label
    else:
        sig_name = _decode_token(tokenizer, kline.signature)

    # No nodes → identity / bare label
    if not kline.nodes:
        return sig_name

    # Resolve operator
    if kline.dbg and kline.dbg.op:
        op_sym = _OP_SYMBOLS.get(kline.dbg.op, ">")
    else:
        op_sym = _infer_op_symbol(kline)

    # Resolve node names
    node_names = []
    for n in kline.nodes:
        # Try to get label from another kline's dbg if available
        name = _decode_token(tokenizer, n)
        node_names.append(name)

    return f"{sig_name} {op_sym} {' '.join(node_names)}"


def _decode_token(tokenizer: object, token: int) -> str:
    """Decode a uint64 token to a string, falling back to hex."""
    try:
        result = tokenizer.decode([token])
        if result:
            return result
    except Exception:
        pass
    return f"<{token:#x}>"


def _infer_level(kline: KLine) -> str:
    """Infer significance level from KLine structure."""
    nodes = kline.nodes
    if not nodes:
        return "S4"
    from kalvin.signature import make_signature

    nodes_sig = make_signature(nodes)
    if kline.signature == nodes_sig:
        return "S2"  # perfect fit → canonize
    if len(nodes) == 1:
        combined = kline.signature & nodes[0]
        return "S2" if combined != 0 else "S3"
    combined = kline.signature & nodes_sig
    return "S2" if combined != 0 else "S3"


def _infer_op_symbol(kline: KLine) -> str:
    """Infer operator symbol from KLine structure."""
    if not kline.nodes:
        return None
    from kalvin.signature import make_signature

    nodes_sig = make_signature(kline.nodes)
    if kline.signature == nodes_sig:
        return "=>"  # perfect fit → canonize
    return ">"  # default: connotate


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
