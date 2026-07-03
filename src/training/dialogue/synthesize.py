"""Synthesize a trainer turn from the compiled script (not the table).

Plan: ``@plans/implement-synthesizing-trainer.md``.
Spec seam: ``@specs/dialogue-driven-training.md`` §Actor, §The Runner,
§Validation (the synthesizer satisfies the existing Actor contract).

The :func:`synthesize` function is the drop-in core of a
:class:`~training.dialogue.runner.SynthesizingTrainer`. It derives the next
trainer KValue from two inputs — the compiled script and the trainee's last
KValue — using structural predicates only (D2: ``dbg``-free). It never reads
the authored dialogue table; that table is only the validation oracle the
runner checks the synthesised turn against.

See the plan's §The Synthesis Rules for R1–R3 (verified against all 16 trainer
turns of the MHALL golden master).
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
_RELATION_PRECEDENCE = {"UNDERSIGNED": 0, "CONNOTED": 1}


def synthesize(
    compiled: list[KValue],
    incoming: KValue | None,
    signifier: KSignifier,
) -> KValue:
    """Synthesize the next trainer KValue from ``(compiled, incoming)``.

    Pure: ``compiled`` is indexed once internally; the result is otherwise a
    pure function of ``incoming``. Implements R1 (opening), R2 (reply to an
    identity), and R3 (echo a matching compiled kline). ``dbg`` is never read
    for decisions — canon is detected structurally via ``is_canon`` (plan D5),
    and relation-vs-canon in R3 via ``is_canon`` as well.
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
    """R1 — emit the first compiled entry at S2.

    ``compiled`` is assumed non-empty (single-primary, single-cascade scripts —
    the spec's existing Out-of-Scope).
    """
    return KValue(primary, SIG_S2)


# ── R2 — Reply to an identity ────────────────────────────────────────────


def _reply_identity(
    signature: int,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
) -> KValue:
    """R2 — the trainee does not recognise ``signature``; supply its decomposition.

    Emit the first decomposition by op-precedence (canon > relation; among
    relations UNDERSIGNED > CONNOTED), then compilation order. Significance:
    S1 when every node is a leaf (no further decomposition), S2 when any node
    is itself decomposable (the trainee will re-ask), S4 when no decomposition
    exists (stalemate).
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
    """The relation op carried on ``kline.dbg`` if it is a known relation, else None.

    A non-canon kline with nodes is a relation; its op distinguishes
    UNDERSIGNED from CONNOTED. dbg is read only here (an ordering hint), never
    for the canon/identity decisions that D2 guards.
    """
    op = kline.dbg.op if kline.dbg else None
    return op if op in _RELATION_PRECEDENCE else None


def _relation_key(kline: KLine) -> int:
    """Sort key for R2 relation precedence (UNDERSIGNED before CONNOTED)."""
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
    ``proposal.nodes``. Significance: S1 for a relation (the trainer ratifies),
    S2 for a canon (the trainer confirms). No match → CONNOTED,S4.
    """
    for kline in decompositions.get(proposal.signature, []):
        if list(kline.nodes) == list(proposal.nodes):
            significance = SIG_S2 if is_canon(kline, signifier) else SIG_S1
            return KValue(kline, significance)
    return KValue(KLine(proposal.signature, proposal.nodes), SIG_S4)
