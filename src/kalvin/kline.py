"""KLine - Fundamental unit of Kalvin's memory.

A Kline is an identified, ordered sequence of zero or more nodes.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from kalvin.abstract import KSignifier, KTokenizer

# === Core Types ===


class KNode(int):
    """A node: a uint64 value with an optional label.

    Subclasses ``int`` so nodes hash, compare, and mask as plain values
    everywhere in the engine; ``.label`` carries the human-readable name
    and ``.value`` exposes the underlying int.
    """

    def __new__(cls, value: int, label: str = "") -> "KNode":
        self = super().__new__(cls, value)
        self.label = label
        return self

    @property
    def value(self) -> int:
        return int(self)

    def with_label(self, label: str) -> "KNode":
        """A copy of this node carrying ``label``."""
        return KNode(self, label)

    def merge(self, other: int) -> "KNode":
        """A new node OR-combined with *other*, labels composed with ``|``.

        The node-level counterpart of the Signifier's OR-reduction
        (``signature_of``): accumulates values into a running signature
        where no Signifier is at hand. Unlike ``|``, the result stays a
        KNode. An empty label on either side yields the other side's
        label; both empty stays empty.
        """
        parts = (self.label, getattr(other, "label", ""))
        return KNode(self | other, "|".join(p for p in parts if p))

    def __repr__(self) -> str:
        return f"KNode({int(self)}, {self.label!r})" if self.label else f"KNode({int(self)})"


# Accepted input representations for KLine's ``nodes`` parameter.
# Sequence (covariant) so list[KNode] is assignable to it.
KNodes: TypeAlias = Sequence[int]

# Type alias for Signatures (uint64)
KSig: TypeAlias = KNode


# === Decode resolver context ===
#
# A session-scoped resolver maps a node value to the KLine that heads it
# (a model's ``resolve``, an encoder's index, etc.). When one is active,
# the KLine constructor populates ``KDbg.decoded`` on freshly minted klines
# via :func:`kline_decode` — so runtime-emitted klines become debuggable
# without each emission site wiring it explicitly. Probe klines (membership
# checks, codec deserialisation) run outside any resolver context and pay
# only a contextvar peek, never the per-node resolve.

KResolver: TypeAlias = Callable[[KNode], "KLine | None"]
_resolver: contextvars.ContextVar[KResolver | None] = contextvars.ContextVar(
    "kalvin.kline.resolver", default=None
)


@contextlib.contextmanager
def using_resolver(resolver: KResolver) -> Iterator[None]:
    """Install *resolver* as the active decode resolver for the current context.

    KLines constructed inside the block populate ``KDbg.decoded`` from the
    resolver; on exit the previous resolver (or none) is restored. Use as a
    context manager around emission / compilation scopes.
    """
    token = _resolver.set(resolver)
    try:
        yield
    finally:
        _resolver.reset(token)


@contextlib.contextmanager
def _resolver_reset() -> Iterator[None]:
    """Temporarily clear the active resolver.

    Used inside :func:`kline_decode` so that klines the resolver itself
    constructs (e.g. a compiler index lookup) do not re-enter ``kline_decode``.
    """
    token = _resolver.set(None)
    try:
        yield
    finally:
        _resolver.reset(token)


@dataclass
class KDbg:
    """Provenance metadata for a KLine (not spec'd).

    Populated by the token encoder during compilation.  Forwarded as-is
    by misfit expansions and model duplication.

    Attributes:
        op: Structural relationship (ASK, DENOTES, CONNOTES,
            CANONICALISES, UNKNOWN).
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
    annotation: str = ""
    scope: int = 0

    def __bool__(self) -> bool:
        """Truthy when any field is non-empty."""
        return bool(
            self.op != "UNKNOWN"
            or self.label
            or self.decoded
            or self.type_info
            or self.annotation
            or self.scope
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
        acq_depth: acquisition depth — the unratified correspondence edges
            crossed to win this kline's content (kalvin-algebra.md §11). 0 for given
            content; flattened by grounding. Identity ignores it: two klines
            with the same signature and nodes are the same kline whatever
            they cost.
    """

    __slots__ = ("signature", "nodes", "dbg", "acq_depth")

    def __init__(
        self,
        signature: KSig,
        nodes: KNodes | KNode | None = None,
        dbg: KDbg | None = None,
        acq_depth: int = 0,
    ):
        self.signature = signature if isinstance(signature, KNode) else KNode(signature)
        self.acq_depth = acq_depth
        if not self.signature.label and dbg is not None and dbg.label:
            self.signature = self.signature.with_label(dbg.label)
        self.nodes = _normalize_nodes(nodes)
        resolver = _resolver.get()
        if resolver is None:
            self.dbg = dbg
            return
        # A resolver is active: populate ``decoded`` on this kline. A passed
        # ``dbg`` may be shared with another kline (emission sites forward
        # ``kline.dbg``), so copy it before mutating to avoid aliasing.
        if dbg is None:
            self.dbg = KDbg()
        else:
            self.dbg = KDbg(
                op=dbg.op, label=dbg.label, decoded=dbg.decoded,
                type_info=dbg.type_info, annotation=dbg.annotation, scope=dbg.scope,
            )
        self.dbg.decoded = kline_decode(self, resolver)

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
# module agrees on what counts as identity vs canon.


def is_terminal(kline: KLine) -> bool:
    """Test whether a kline is a terminal — a leaf that stops traversal.

    A terminal carries no further decomposition. Two shapes are terminal:
      - empty nodes: ``{S: []}`` (an Unknown), or
      - self-referential: ``{S: [S]}`` (an Identity; a multi-subword
        word is a plain identity — one word bit, its subword token ids
        OR-reduced into the value).

    Terminal is the genus of :func:`is_unknown` and :func:`is_identity`;
    the canon/misfit distinction applies only to non-terminals.
    """
    if not kline.nodes:
        return True
    return kline.nodes == [kline.signature]


def is_unknown(kline: KLine) -> bool:
    """Test whether a kline is an Unknown — the empty form ``{S: []}``.

    An Unknown claims S4: nothing held for this signature, the structural
    form of an ask.
    """
    return not kline.nodes


def is_identity(kline: KLine) -> bool:
    """Test whether a kline is a decodable Identity terminal.

    An Identity is a terminal that translates to a known value in the
    outside world — directly decodable. The sole structural shape is the
    self-referential form ``{S: [S]}``: a value that decodes into itself.
    A multi-subword word is an identity by this same rule — one word bit,
    its subword token ids OR-reduced into the value.

    The empty form ``{S: []}`` is an :func:`is_unknown`, not an Identity.
    Identity overrules any canon classification (see :func:`is_canon`).
    """
    if not kline.nodes:
        return False
    return kline.nodes == [kline.signature]


def is_exact(kline: KLine, signifier: KSignifier) -> bool:
    """s = signature_of(ν) over the atom space (kalvin-algebra Def 6) — equivalent
    to gap = ∅ and excess = ∅ (Def 9). BPE packing bits are not atoms."""
    nodes_sig = signifier.signature_of(kline.nodes)
    return (
        signifier.residual(kline.signature, nodes_sig) == 0
        and signifier.residual(nodes_sig, kline.signature) == 0
    )


def is_canon(kline: KLine, signifier: KSignifier) -> bool:
    """Test whether a kline is a canon.

    A kline is a canon when it is a non-terminal whose signature equals
    ``signature_of(nodes)`` over the atom space. A terminal is never a
    canon.
    """
    return not is_terminal(kline) and is_exact(kline, signifier)


def is_canon_evidence(kline: KLine, signifier: KSignifier) -> bool:
    """A canon usable as replace evidence: exact and well-founded (kalvin-algebra
    Def 13 — the signature does not occur in its own witness). A
    self-containing canon is an inert witness class, like an identity."""
    return is_canon(kline, signifier) and kline.signature not in kline.nodes

def is_relationship(kline: KLine) -> bool:
    """Test whether a kline is a 1:1 relationship.

    The denote/connote structural shape: a non-terminal misfit with exactly
    one node (``{A: [B]}``, ``A != B``). The signature associates with a
    single other value. Band-agnostic — the band-true species are
    :func:`is_denotation` (case 4) and :func:`is_connotation` (case 6).
    """
    return (
        len(kline.nodes) == 1
        and not is_terminal(kline)
        and not is_identity(kline)
    )


def is_denotation(kline: KLine, signifier: KSignifier) -> bool:
    """Case 4: a 1:1 relationship whose node shares no atom with its signature.

    Uncovered (no word-bit overlap) — S3. The `=` DENOTES shape.
    """
    return is_relationship(kline) and not signifier.signifies(
        kline.nodes[0], kline.signature
    )


def is_connotation(kline: KLine, signifier: KSignifier) -> bool:
    """Case 6: a 1:1 relationship, covered, gap-only.

    The node overlaps the signature and carries no excess (``AB:[B]``) — S2.
    The `>` CONNOTES shape.
    """
    if not is_relationship(kline):
        return False
    node = kline.nodes[0]
    return signifier.signifies(node, kline.signature) and signifier.residual(
        node, kline.signature
    ) == 0

def is_misfit(kline: KLine, signifier: KSignifier) -> bool:
    """Test whether a kline is a misfit.

    A kline is a misfit when it is a non-terminal whose signature does not
    equal ``signature_of(nodes)``. This includes the single-node connote/denote
    shape ``{A: [B]}``; misfits differ in the band they claim (S2 when at
    least one node is covered by the signature, S3 when none is), not in
    whether they are misfits.
    """
    return not is_terminal(kline) and not is_canon(kline, signifier)


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

# Display helpers

_OP_SYMBOLS = {
    "DENOTES": "=",
    "CONNOTES": ">",
    "CANONICALISES": "=>",
    "ASK": "?",
    "UNKNOWN": None,
}


def sig_level(kline: KLine, signifier: KSignifier) -> str:
    """Return structural significance level (S1–S4) for a KLine.

    The significance level derives from the signature–nodes relationship
    alone, independent of node count and of the relational token that
    compiled the shape:

    - S1 — the signature covers its nodes exactly (canon, identity).
    - S2 — at least one node is covered by the signature (shares a word
      bit — Def 8 overlap, not containment).
    - S3 — no node is covered (denotation, misfit).
    - S4 — no nodes (unknown).
    """
    nodes = kline.nodes
    if not nodes:
        return "S4"
    if is_exact(kline, signifier):
        return "S1"
    return "S2" if any(signifier.signifies(n, kline.signature) for n in nodes) else "S3"


def kline_display(kline: KLine, tokenizer: KTokenizer, signifier: KSignifier) -> str:
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


def _decode_token(tokenizer: KTokenizer, token: int) -> str:
    """Decode a uint64 token to a string, falling back to hex."""
    try:
        result = tokenizer.decode([token])
        if result:
            return result
    except Exception:
        pass
    return f"<{token:#x}>"


def _node_label(node: int, resolver: KResolver) -> str:
    """Resolve a node value to a readable label via *resolver*, falling back to hex.

    *resolver* maps a node value to the KLine that heads it (e.g. a model's
    ``resolve``); the kline's ``dbg.label``/``dbg.annotation`` supply the name.
    No tokenizer decoding — hex is the only fallback.
    """
    kl = resolver(node)
    if kl is not None and kl.dbg:
        if kl.dbg.label:
            return kl.dbg.label
        if kl.dbg.annotation:
            return kl.dbg.annotation
    return f"<{node:#x}>"


def kline_decode(
    kline: KLine,
    resolver: KResolver,
) -> str:
    """Format a KLine as a readable ``sig:[node, ...]`` provenance string.

    Uses ``dbg`` provenance (label/annotation) for the signature and resolves
    each node through *resolver* to read its heading kline's ``dbg``. Falls
    back to hex when no label is available — no tokenizer decoding. Used by
    the compiler and the dialogue subsystem (via the constructor's resolver
    context) to populate ``KDbg.decoded``.

    Runs with the contextvar resolver cleared, so any ``KLine`` the resolver
    itself constructs (e.g. a compiler index lookup) does not re-enter
    ``kline_decode``.

    Args:
        kline: The KLine to decode.
        resolver: Maps a node value to the KLine that heads it (a model's
            ``resolve``, or an equivalent index), or ``None`` when unknown.

    Returns:
        ``"sig:[n0, n1, ...]"``; empty nodes → ``"sig:[]"``.
    """
    if kline.dbg and kline.dbg.label:
        sig_name = kline.dbg.label
    elif kline.dbg and kline.dbg.annotation:
        sig_name = kline.dbg.annotation
    else:
        sig_name = f"<{kline.signature:#x}>"
    with _resolver_reset():
        nodes = ", ".join(_node_label(n, resolver) for n in kline.nodes)
    return f"{sig_name}:[{nodes}]"


def _infer_op_symbol(kline: KLine, signifier: KSignifier) -> str:
    """Infer operator symbol from KLine structure."""
    if not kline.nodes:
        return ""
    nodes_sig = signifier.signature_of(kline.nodes)
    if kline.signature == nodes_sig:
        return "=>"  # perfect fit → canonicalise
    return ">"  # default: connote


def _normalize_nodes(nodes: KNodes | KNode | None) -> list[KNode]:
    """Normalize node input to a list of KNode.

    - None → []
    - int → [KNode]
    - list → list (KNodes kept, plain ints wrapped)
    """
    if nodes is None:
        return []
    if isinstance(nodes, int):
        return [nodes if isinstance(nodes, KNode) else KNode(nodes)]
    return [n if isinstance(n, KNode) else KNode(n) for n in nodes]
