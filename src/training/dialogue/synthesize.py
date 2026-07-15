"""Synthesize a trainer turn from the compiled script.

:func:`synthesize` is the core of
:class:`~training.dialogue.actors.SynthesizingTrainer`. It derives the next
trainer KValue from the compiled script and the trainee's last KValue using
structural predicates only (no ``dbg`` reads). The runner checks the
synthesised turn against the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kalvin.expand import SIG_S1, SIG_S2, SIG_S4
from kalvin.kline import KLine, is_canon, is_identity
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["synthesize"]

# Op precedence for R2's reply-to-identity: the canon (a signature's own
# decomposition) outranks any relation sharing that signature. Canon is detected
# structurally via is_canon, so this only orders the non-canon relations.
_RELATION_PRECEDENCE = {"DENOTES": 0, "CONNOTES": 1}


def synthesize(
    compiled: list[KValue],
    incoming: KValue | None,
    signifier: KSignifier,
) -> KValue:
    """Synthesize the next trainer KValue from ``(compiled, incoming)``.

    Pure: ``compiled`` is indexed once internally; the result is otherwise a
    pure function of ``incoming``. Implements R1 (opening), R2 (reply to an
    identity), and R3 (echo a matching compiled kline).
    """
    decompositions: dict[int, list[KLine]] = {}
    for value in compiled:
        kline = value.kline
        if kline.nodes:
            decompositions.setdefault(kline.signature, []).append(kline)
    primary = compiled[0].kline

    if incoming is None:
        return _opening(primary)

    proposal = incoming.kline
    if is_identity(proposal):
        return _reply_identity(proposal.signature, decompositions, signifier)
    return _echo_compiled(proposal, decompositions, signifier)


# ── R1 — Opening ─────────────────────────────────────────────────────────


def _opening(primary: KLine) -> KValue:
    """R1 — emit the first compiled entry at S2."""
    return KValue(primary, SIG_S2)


# ── R2 — Reply to an identity ────────────────────────────────────────────


def _reply_identity(
    signature: int,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
) -> KValue:
    """R2 — the trainee does not recognise ``signature``; supply its decomposition.

    Emit the first decomposition by op-precedence (canon > relation; among
    relations DENOTES > CONNOTES), then compilation order. Significance:
    S1 when every node is a leaf, S2 when any node is itself decomposable, S4
    when no decomposition exists.
    """
    candidates = decompositions.get(signature, [])
    chosen = _precedence_pick(candidates, signifier)
    if chosen is None:
        return KValue(KLine(signature, []), SIG_S4)
    significance = (
        SIG_S1
        if all(node not in decompositions for node in chosen.nodes)
        else SIG_S2
    )
    return KValue(chosen, significance)


def _precedence_pick(
    candidates: list[KLine], signifier: KSignifier
) -> KLine | None:
    """First canon by compilation order; else lowest-precedence relation."""
    for kline in candidates:
        if is_canon(kline, signifier):
            return kline
    relations = [k for k in candidates if _relation_op(k) is not None]
    if not relations:
        return None
    relations.sort(key=_relation_key)
    return relations[0]


def _relation_op(kline: KLine) -> str | None:
    """The relation op on ``kline.dbg`` if it is a known relation, else None."""
    op = kline.dbg.op if kline.dbg else None
    return op if op in _RELATION_PRECEDENCE else None


def _relation_key(kline: KLine) -> int:
    """Sort key for R2 relation precedence (DENOTES before CONNOTES)."""
    op = _relation_op(kline)
    return _RELATION_PRECEDENCE[op] if op is not None else len(_RELATION_PRECEDENCE)


# ── R3 — Echo a matching compiled kline ──────────────────────────────────


def _echo_compiled(
    proposal: KLine,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
) -> KValue:
    """R3 — the trainee proposed a kline the trainer has compiled: echo it verbatim.

    A match is a compiled kline under ``proposal.signature`` whose nodes equal
    ``proposal.nodes``. Significance: S1 for a relation (ratify), S2 for a
    canon (confirm). No match → CONNOTES,S4.
    """
    for kline in decompositions.get(proposal.signature, []):
        if list(kline.nodes) == list(proposal.nodes):
            significance = SIG_S2 if is_canon(kline, signifier) else SIG_S1
            return KValue(kline, significance)
    return KValue(KLine(proposal.signature, proposal.nodes), SIG_S4)
