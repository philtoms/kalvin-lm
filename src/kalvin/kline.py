"""KLine - Fundamental unit of Kalvin's memory.

A Kline is an identified, ordered sequence of zero or more nodes.
See specs/kline.md for the full specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier

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
        op: Structural relationship (COUNTERSIGNS, DENOTES, CONNOTES,
            CANONIZES, IDENTITY).
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

#: The **compound marker token**, re-exported from the kalvin↔NLP boundary
#: (:mod:`kalvin.nlp_tokenizer`). The compiler appends it to the nodes of a
#: §11.3 compound-word kline (``Mary: [COMPOUND_TOKEN, M, ary]``); the token
#: participates in the signature algebra, so the marker needs no masking.
#: See :data:`kalvin.nlp_tokenizer.COMPOUND_TOKEN` for the full rationale.
from kalvin.nlp_tokenizer import COMPOUND_TOKEN  # noqa: E402


def _is_compound_word(kline: KLine) -> bool:
    """Test whether a kline is a §11.3 compound-word identity.

    True iff :data:`COMPOUND_TOKEN` is among the kline's nodes. The compiler
    appends the token only to a compound-word's nodes (a single word the
    #: external tokenizer split into BPE subwords), so its presence is the
    #: structural signal. Purely structural — no signifier, no provenance,
    #: no bit masking.
    """
    return COMPOUND_TOKEN in kline.nodes


def is_identity(kline: KLine) -> bool:
    """Test whether a kline is an identity.

    A kline is identity when it carries no decomposition — either form:
      - empty nodes: ``{S: []}``, or
      - self-referential: ``{S: [S]}`` — its own signature is its sole node, or
      - compound-word: ``{S: [COMPOUND_TOKEN, M, ary]}`` — a single word
        whose nodes include :data:`COMPOUND_TOKEN` because the external
        tokenizer split it into multiple BPE subwords. The word is one
        lexical item; the decomposition is an encoding artefact, not a
        declared aggregation.

    The self-referential form is identity *by definition*: a value that
    decomposes into itself carries no further information. The compound-word
    form is identity *by external tokenisation*: the word does not aggregate
    its subwords. Both overrule any canon classification (see :func:`is_canon`).
    """
    if not kline.nodes:
        return True
    if kline.nodes == [kline.signature]:
        return True
    return _is_compound_word(kline)


def is_canon(kline: KLine, signifier: KSignifier) -> bool:
    """Test whether a kline is canonical.

    A kline is a canon when it has multiple nodes and each of them is
    represented in its signature but does not constitute a compound identity. 
    """
    return not is_identity(kline) and kline.signature == signifier.make_signature(kline.nodes)

def is_misfit(kline: KLine, signifier: KSignifier) -> bool:
    """Test whether a kline is a misfit.
    
    A kline is a misfit when it has multiple nodes and at least one node is 
    not represented by its signature (ie, kline is not a canon) but does not 
    constitute a compound identity. 
    """
    return len(kline.nodes) > 1 and not _is_compound_word(kline) and not is_canon(kline, signifier)

# Display helper

_OP_SYMBOLS = {
    "COUNTERSIGNS": "==",
    "DENOTES": "=",
    "CONNOTES": ">",
    "CANONIZES": "=>",
    "IDENTITY": None,
}


def sig_level(kline: KLine, signifier: KSignifier) -> str:
    """Return significance level (S1–S4) for a KLine.

    Uses dbg.op when available, infers from structure otherwise.
    """
    nodes = kline.nodes
    if not nodes:
        return "S4"
    if len(nodes) == 1:
        return "S3"
    return "S1" if kline.signature == signifier.make_signature(kline.nodes) else "S2"


def kline_display(kline: KLine, tokenizer: object, signifier: KSignifier) -> str:
    """Format a KLine as human-readable KScript source.

    Uses dbg provenance when available (label, op). Falls back to
    tokenizer decoding and structural inference when dbg is absent.

    Args:
        kline: The KLine to display.
        tokenizer: A KTokenizer for decoding uint64 values to strings.
        signifier: A KSignifier for inferring structure when dbg is absent.

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
        op_sym = _infer_op_symbol(kline, signifier)

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


def _infer_op_symbol(kline: KLine, signifier: KSignifier) -> str:
    """Infer operator symbol from KLine structure."""
    if not kline.nodes:
        return ""
    nodes_sig = signifier.make_signature(kline.nodes)
    if kline.signature == nodes_sig:
        return "=>"  # perfect fit → canonize
    return ">"  # default: connote


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
