"""Stateless Trainer supply rule (spec ``@specs/dialogue-driven-training.md``
§Partition and the Supply Rule, §Significance the Trainer Attaches, §Ratification,
§Opening; plan Phase 2).

The Trainer carries **no temporal state** about the dialogue (D1, DDT-5): no
provenance ledger, no open-proposal tracking. Its response to an incoming turn
is a pure function of ``(incoming turn, compiled script)``. This module is that
function and the held index it consults.

What lives here
---------------
- :class:`HeldIndex` — the held-kline lookup, built once at configuration time
  from the compiled script. Withheld = Identities + Canons (subword canons
  unfiltered, DDT-19). The index is keyed by **symbolic label** (the table's
  vocabulary): a request "Det" is a request for the Det *concept*, and the
  Trainer supplies whichever held kline that label names under canon-first full
  shadowing.
- :func:`supply` — the pure canon-first supply function with full shadowing
  (DDT-8, DDT-9, DDT-11).
- :func:`terminal_significance` — node-terminality significance (terminal → S1,
  non-terminal → S2, DDT-12).
- :func:`respond` — the pure dispatch over an incoming K turn (S4 → supply,
  S3 → ratify reciprocal at S1, S1 → no-op, DDT-6/13/14).
- :func:`opening` — the computed primary-half opening at S2 (DDT-7).

Statelessness contract
----------------------
Every public function is pure: same inputs → same output, no mutation of shared
state. :class:`HeldIndex` is built once and only read thereafter. There is no
trainer-side provenance, no open-proposal ledger — that state is Kalvin's, not
the Trainer's (D1).

Label-driven supply (why not signature-driven?)
------------------------------------------------
The K S4 request is an IDENTITY turn whose label is the requested concept (e.g.
``"Det"``), but the decoder resolves that label to the *atom* signature (e.g.
the ``D`` subword). The Det **canon** (``Det => D, et``) has a different
signature, so a signature-keyed lookup would return the atom identity, not the
canon. The table is prescriptive at the symbolic level: the request "Det" must
supply the Det canon. The held index is therefore keyed by the symbolic label
carried on each compiled kline's ``dbg.label``. Full shadowing (DDT-9) then falls
out naturally: a label whose canon exists supplies the canon; a label with only a
relation (e.g. ``a``) supplies the relation (DDT-11).

Symbolic-label node-terminality (D3)
------------------------------------
"Terminal" is evaluated at the **table's symbolic node-label level**: a held
kline's node is terminal iff no canon or relation in the compiled script has that
node's signature. This matches the spec's worked example: ``{a:[Det]}`` is S2
because the Det operand is a non-terminal role (it has a canon), even though the
compiled ``a`` relation binds to a subword-atom signature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kalvin.expand import SIG_S1, SIG_S2
from kalvin.kline import is_identity
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn


def _classify(kvalue: KValue) -> str:
    """Classify a compiled KValue into its held-index bucket.

    Buckets are the supply-priority classes: ``canon`` (the compiler's
    ``CANONIZED`` aggregate), ``relation`` (any non-identity structural state —
    ``COUNTERSIGNED``/``CONNOTED``/``UNDERSIGNED``), ``identity``. Identity is
    the fallback bucket for atoms.
    """
    op = kvalue.kline.dbg.op if kvalue.kline.dbg else "IDENTITY"
    if op == "CANONIZED":
        return "canon"
    if is_identity(kvalue.kline):
        return "identity"
    return "relation"


def _label_of(kvalue: KValue) -> str | None:
    """The symbolic label of a compiled kline (``dbg.label``), else None."""
    return kvalue.kline.dbg.label if kvalue.kline.dbg else None


@dataclass(frozen=True)
class HeldIndex:
    """The held-kline lookup, keyed by symbolic label (spec §Partition).

    Withheld entries are Identities and Canons (DDT-19: subword canons are not
    filtered). The index maps each label to its canon (if any), relation (if
    any), and identity (if any) — the three supply-priority classes. Canon-first
    full shadowing (DDT-9) is: if a label has a canon, supply the canon and never
    its relation; a label with only a relation (DDT-11) supplies the relation.

    Also tracks ``terminal_signatures`` — signatures no canon/relation decomposes
    — for node-terminality (D3). Built once at configuration time; read-only
    thereafter (D1).
    """

    by_label: dict[str, dict[str, KValue | None]] = field(default_factory=dict)
    terminal_signatures: frozenset[int] = field(default_factory=frozenset)

    def canon_for(self, label: str) -> KValue | None:
        return self.by_label.get(label, {}).get("canon")

    def relation_for(self, label: str) -> KValue | None:
        return self.by_label.get(label, {}).get("relation")

    def identity_for(self, label: str) -> KValue | None:
        return self.by_label.get(label, {}).get("identity")


def build_held_index(entries: Sequence[KValue]) -> HeldIndex:
    """Build the held index from compiled entries (plan task 2.1).

    Partitions compiled entries by their ``dbg.label`` into canon/relation/
    identity buckets. A label may carry several classes (e.g. ``Mary`` has both a
    canon and an undersigned relation) — they coexist so canon-first shadowing is
    "prefer canon". Also computes the terminal-signature set for node-terminality
    (D3): a signature is terminal iff no canon/relation's signature equals it.
    """
    by_label: dict[str, dict[str, KValue | None]] = {}
    decomposable: set[int] = set()
    for e in entries:
        label = _label_of(e)
        if label is None:
            continue
        slot = by_label.setdefault(label, {"canon": None, "relation": None, "identity": None})
        cls = _classify(e)
        # Keep the first compiled entry per class (stable; the MHALL table has no
        # duplicate-class collisions per label).
        if slot[cls] is None:
            slot[cls] = e
        if cls in ("canon", "relation"):
            decomposable.add(e.kline.signature)

    terminal = frozenset(
        e.kline.signature
        for e in entries
        if is_identity(e.kline) and e.kline.signature not in decomposable
    )
    return HeldIndex(by_label=by_label, terminal_signatures=terminal)


# ── Node-terminality significance (DDT-12, D3) ────────────────────────────


def terminal_significance(supplied: KValue, held: HeldIndex) -> int:
    """The significance the Trainer attaches to a supplied entry (DDT-12).

    Terminal nodes (atoms needing no further decomposition in the script's role
    graph) → S1; non-terminal nodes (compounds/structures that themselves need
    resolving) → S2. Terminality is decided by membership in
    :attr:`HeldIndex.terminal_signatures`: a node is terminal iff no canon or
    relation decomposes it further.
    """
    nodes = supplied.kline.nodes
    is_terminal = bool(nodes) and all(n in held.terminal_signatures for n in nodes)
    return SIG_S1 if is_terminal else SIG_S2


# ── Canon-first supply with full shadowing (DDT-8, DDT-9, DDT-11) ──────────


class SupplyMiss(Exception):
    """No held kline for the requested label (escalates per DDT-20).

    For the bootstrap run the held index always resolves; we surface the miss so
    the loop can route it to the supervisor-decision path rather than silently
    emitting nothing.
    """


def supply(request_label: str, held: HeldIndex) -> KValue:
    """Pure canon-first supply with full shadowing (DDT-8, DDT-9, DDT-11).

    Looks up ``request_label`` and returns the held kline in priority order:
    canon → relation → identity. Full shadowing (DDT-9): when a label carries
    both a canon and a relation, the canon is returned and the relation is never
    supplied — it is K-discovered (S3) then T-ratified (S1). A label with only a
    relation (DDT-11, e.g. ``a``) is supplied. The significance is attached by
    node-terminality (:func:`terminal_significance`).

    The returned KValue is the held kline re-marked with the Trainer's
    node-terminality significance (the compiled entry's own significance is the
    band derived from its op, not the dialogic stance the Trainer takes).
    """
    slot = held.by_label.get(request_label)
    if not slot:
        raise SupplyMiss(
            f"no held kline for label {request_label!r} (DDT-20 escalation)"
        )
    for cls in ("canon", "relation", "identity"):
        kv = slot.get(cls)
        if kv is not None:
            return KValue(kv.kline, terminal_significance(kv, held))
    raise SupplyMiss(
        f"no held kline for label {request_label!r} (DDT-20 escalation)"
    )


# ── Pure response dispatch (DDT-5, DDT-6, DDT-13, DDT-14) ─────────────────


def _op_of(kvalue: KValue) -> str:
    """Map a KValue's compiled structural state onto the dialogue op vocabulary.

    The held index stores compiled KValues whose ``dbg.op`` is the compiler's
    vocabulary (``CANONIZED`` for canons). The dialogue op vocabulary is the
    table's (``CANON`` for the aggregate directive). This maps between them so a
    computed T turn carries the same op the table uses, enabling Model A
    turn-equality in the loop.
    """
    op = kvalue.kline.dbg.op if kvalue.kline.dbg else "IDENTITY"
    if op == "CANONIZED":
        return "CANON"
    return op


@dataclass(frozen=True)
class TrainerResponse:
    """The result of a pure dispatch: zero or one T turns, or a miss.

    The Trainer emits at most one turn per incoming K turn (the canonical table
    is 1:1; a multi-supply table would yield a multi-turn T-run, handled by the
    loop's greedy cursor in Phase 3). ``miss`` is set when a request cannot be
    auto-resolved (DDT-20); the loop routes it to escalation.
    """

    turn: DecodedTurn | None = None
    miss: "SupplyMiss | None" = None


def respond(incoming: DecodedTurn, held: HeldIndex) -> TrainerResponse:
    """The pure Trainer response to an incoming K turn (DDT-5, DDT-6).

    Dispatch by the incoming turn's significance (the dialogic stance):

    - **S4** (request, ``X:[]``) → :func:`supply` the held kline for X's
      **label**, marked at node-terminality significance (DDT-8, DDT-12).
    - **S3** (proposal) → ratify: the reciprocal at S1 (DDT-13).
    - **S1** (terminal) → no action; advance (DDT-14). Returns no turn.

    No provenance, no open-proposal ledger — the response depends only on the
    incoming turn and the held index (D1).
    """
    from kalvin.expand import SIG_S3, SIG_S4

    sig = incoming.value.significance
    if sig == SIG_S4:
        label = _label_of(incoming.value)
        if label is None:
            return TrainerResponse(
                miss=SupplyMiss("S4 request carries no label (DDT-20 escalation)")
            )
        try:
            supplied = supply(label, held)
        except SupplyMiss as miss:
            return TrainerResponse(miss=miss)
        return TrainerResponse(
            turn=DecodedTurn(actor="T", op=_op_of(supplied), value=supplied)
        )
    if sig == SIG_S3:
        # Ratify (DDT-13): respond with the reciprocal at S1. The reciprocal's
        # structural state is the **script's authoritative op** for the proposed
        # kline (the relation the script compiles), not a mechanical reversal
        # of K's tentative proposal op. K proposes ``CONNOTED`` (it does not yet
        # know the direction); the script's compiled relation (e.g.
        # ``UNDERSIGNED Mary:[Subject]``) is the structure T endorses. We look
        # the relation up by the proposal's label and fall back to the proposal's
        # own op if the script declares no such relation.
        label = _label_of(incoming.value)
        rel = held.relation_for(label) if label is not None else None
        ratified_op = _op_of(rel) if rel is not None else _op_of(incoming.value)
        ratified = KValue(incoming.value.kline, SIG_S1)
        return TrainerResponse(
            turn=DecodedTurn(actor="T", op=ratified_op, value=ratified)
        )
    # S1 (terminal): no action; advance (DDT-14).
    return TrainerResponse()


# ── Opening (DDT-7) ───────────────────────────────────────────────────────


def opening(primary: KValue, held: HeldIndex) -> DecodedTurn:
    """The computed opening turn: the primary ``==`` half at S2 (DDT-7).

    The primary is the root relationship (the top-level ``==``), emitted as a
    single half at S2. Opening is itself a proposal. It is **computed** (an
    opening entry point), not read from the table, so every T turn — including
    turn 0 — flows through the supply function and is validated against the table
    by the loop's Model A check (Phase 3, DDT-22).

    ``primary`` is the compiled primary KValue (a ``COUNTERSIGNED`` relation);
    its opening half is the same kline re-marked at S2 (the proposal stance).
    """
    return DecodedTurn(
        actor="T",
        op="COUNTERSIGNED",
        value=KValue(primary.kline, SIG_S2),
    )
