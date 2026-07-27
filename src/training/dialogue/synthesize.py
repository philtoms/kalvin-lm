"""Synthesize a trainer turn from the compiled script.

The supervisor the rationalising trainer escalates to (and the engine behind
:class:`~training.dialogue.actors.SynthesizingTrainer`). See
@specs/dialogue-cogitation.md §Identities for the R2 precedence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kalvin.expand import SIG_S1, SIG_S2, SIG_S4
from kalvin.kline import KLine, is_canon, is_identity
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["synthesize"]


def synthesize(
    compiled: list[KValue],
    incoming: KValue | None,
    signifier: KSignifier,
    grounded: set[int] | None = None,
) -> KValue:
    """Reply to ``incoming`` (or open) from ``compiled``. ``grounded`` is the
    trainer's view of K's ratified S1 signatures."""
    grounded = grounded if grounded is not None else set()
    decompositions: dict[int, list[KLine]] = {}
    for value in compiled:
        kline = value.kline
        if kline.nodes:
            decompositions.setdefault(kline.signature, []).append(kline)
    primary = compiled[0].kline

    if incoming is None:
        return KValue(primary, SIG_S2)

    proposal = incoming.kline
    if is_identity(proposal):
        return _reply_identity(proposal.signature, decompositions, signifier, grounded)
    return _echo_compiled(proposal, decompositions, signifier)


def _reply_identity(
    signature: int,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
    grounded: set[int],
) -> KValue:
    """K asks ``{sig: []}`` (S4); reply by the first rule that admits:

    1. canon — teach its parts (S1 if K grounded every node, else S2);
    2. CONNOTES — a teachable gloss at S2 (a DENOTES role-binding is left for
       the S3 phase, where K proposes and T ratifies);
    3. compound identity ``{sig: [CT, x, y]}`` — the subword grounding at S1;
    4. otherwise — forge the self-identity ``{sig: [sig]}`` at S1.
    """
    candidates = decompositions.get(signature, [])

    canon = _best_canon(candidates, signifier, grounded)
    if canon is not None:
        significance = SIG_S1 if all(n in grounded for n in canon.nodes) else SIG_S2
        return KValue(canon, significance)

    connotes = _first_connotes(candidates)
    if connotes is not None:
        return KValue(connotes, SIG_S2)

    compound = _first_compound(candidates, signifier)
    if compound is not None:
        return KValue(compound, SIG_S1)

    return KValue(KLine(signature, [signature]), SIG_S1)


def _best_canon(
    candidates: list[KLine],
    signifier: KSignifier,
    grounded: set[int],
) -> KLine | None:
    """The canon decomposition closest to K's grounding: most nodes already in
    ``grounded`` (the flattest form K can now ratify), ties broken by compiled
    order. With a single canon this is the first canon."""
    best = None
    best_score = -1
    for kline in candidates:
        if not is_canon(kline, signifier):
            continue
        score = sum(1 for n in kline.nodes if n in grounded)
        if score > best_score:
            best = kline
            best_score = score
    return best


def _first_connotes(candidates: list[KLine]) -> KLine | None:
    """First CONNOTES among ``candidates`` (a DENOTES is a role-binding, not a gloss)."""
    for kline in candidates:
        op = kline.dbg.op if kline.dbg else None
        if op == "CONNOTES":
            return kline
    return None


def _first_compound(candidates: list[KLine], signifier: KSignifier) -> KLine | None:
    """First compound-word identity ``{sig: [CT, x, y]}`` among ``candidates``."""
    for kline in candidates:
        if is_identity(kline) and kline.nodes:
            return kline
    return None


def _echo_compiled(
    proposal: KLine,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
) -> KValue:
    """R3 — respond to K's non-identity proposal: echo an exact compiled match
    (S1 for a relation, S2 for a canon), else bounce it back at S4."""
    for kline in decompositions.get(proposal.signature, []):
        if list(kline.nodes) == list(proposal.nodes):
            significance = SIG_S2 if is_canon(kline, signifier) else SIG_S1
            return KValue(kline, significance)
    return KValue(KLine(proposal.signature, proposal.nodes), SIG_S4)
