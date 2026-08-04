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
            CANONIZES, UNKNOWN).
        label: Origin word or operator context.
        decoded: Tokenizer decode of the signature (actual subword text).
        type_info: Short debug summary of the node's type-dictionary entry
            (e.g. NLP POS/DEP/MORPH labels when the dictionary was generated
            by NLP tooling). Opaque to kalvin.
    """

    op: str = "UNKNOWN"
    label: str = ""
    decoded: str = ""
    type_info: str = ""

    def __bool__(self) -> bool:
        """Truthy when any field is non-empty."""
        return bool(
            self.op != "UNKNOWN"
            or self.label
            or self.decoded
            or self.type_info
        )

    def __repr__(self) -> str:
        parts = []
        if self.op != "UNKNOWN":
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
        signature: uint64 identity key (produced by signature_of).
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


def is_terminal(kline: KLine) -> bool:
    """Test whether a kline is a terminal — a leaf that stops traversal.

    A terminal carries no further decomposition. Two shapes are terminal:
      - empty nodes: ``{S: []}`` (an Unknown), or
      - self-referential: ``{S: [S]}`` (an Identity; this includes §11.3
        compound-words, which are self-referential identities whose
        signature is the OR-reduction of their subword tokens).

    Terminal is the genus of :func:`is_unknown` and :func:`is_identity`;
    the canon/misfit distinction applies only to non-terminals
    (@CONTEXT.md §Terminal).
    """
    if not kline.nodes:
        return True
    return kline.nodes == [kline.signature]


def is_unknown(kline: KLine) -> bool:
    """Test whether a kline is an Unknown — the empty form ``{S: []}``.

    An Unknown claims S4: nothing held for this signature, the structural
    form of an ask (@CONTEXT.md §Unknown).
    """
    return not kline.nodes


def is_identity(kline: KLine) -> bool:
    """Test whether a kline is a decodable Identity terminal.

    An Identity is a terminal that translates to a known value in the
    outside world — directly decodable. The sole structural shape is the
    self-referential form ``{S: [S]}``: a value that decodes into itself.
    A §11.3 compound-word is an identity by this same rule — its signature
    is the OR-reduction of its subword tokens, so it is a self-ref with no
    marker.

    The empty form ``{S: []}`` is an :func:`is_unknown`, not an Identity.
    Identity overrules any canon classification (see :func:`is_canon` and
    @CONTEXT.md §Identity).
    """
    if not kline.nodes:
        return False
    return kline.nodes == [kline.signature]


def is_canon(kline: KLine, signifier: KSignifier) -> bool:
    """Test whether a kline is a canon.

    A kline is a canon when it is a non-terminal whose signature equals
    ``signature_of(nodes)`` (@CONTEXT.md §Canon). A terminal is never a canon.
    """
    return not is_terminal(kline) and kline.signature == signifier.signature_of(kline.nodes)

def is_misfit(kline: KLine, signifier: KSignifier) -> bool:
    """Test whether a kline is a misfit.

    A kline is a misfit when it is a non-terminal whose signature does not
    equal ``signature_of(nodes)`` (@CONTEXT.md §Misfit).
    """
    return len(kline.nodes) > 1 and not is_terminal(kline) and not is_canon(kline, signifier)


def classify_misfit(
    kline: KLine, signifier: KSignifier
) -> tuple[bool, bool]:
    """Classify whether a kline's signature faithfully covers its nodes.

    A structural classification of the relationship between a kline's
    signature and the signature of its nodes. Returns ``(underfit, overfit)``:

    - underfit — the signature claims more than its nodes deliver
      (``residual(signature, nodes_sig) != 0``).
    - overfit — the nodes carry more than the signature captures
      (``residual(nodes_sig, signature) != 0``).

    The residual representation and its masking remain Signifier concerns
    (``residual``); this function only orchestrates the two directions and
    the emptiness test, like the other structural predicates here. Used by
    the misfit/expansion pipeline during rationalisation.
    """
    nodes_sig = signifier.signature_of(kline.nodes)
    underfit = signifier.residual(kline.signature, nodes_sig) != 0
    overfit = signifier.residual(nodes_sig, kline.signature) != 0
    return underfit, overfit

# Display helper

_OP_SYMBOLS = {
    "COUNTERSIGNS": "==",
    "DENOTES": "=",
    "CONNOTES": ">",
    "CANONIZES": "=>",
    "UNKNOWN": None,
}


def sig_level(kline: KLine, signifier: KSignifier) -> str:
    """Return structural significance level (S1–S4) for a KLine.
    """
    nodes = kline.nodes
    if not nodes:
        return "S4"
    if len(nodes) == 1:
        return "S1" if kline.signature == kline.nodes[0] else "S3"
    return "S1" if kline.signature == signifier.signature_of(kline.nodes) else "S2"


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
    nodes_sig = signifier.signature_of(kline.nodes)
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
