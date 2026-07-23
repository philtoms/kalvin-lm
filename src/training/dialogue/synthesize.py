"""Synthesize a trainer turn from the compiled script.

:func:`synthesize` is the core of
:class:`~training.dialogue.actors.SynthesizingTrainer`. It derives the next
trainer KValue from the compiled script and the trainee's last KValue. The
runner checks the synthesised turn against the table.

Derivation mixes structural predicates (``is_canon`` / ``is_identity``, via
``signifier.make_signature``) with relation-op reads (``dbg.op``): a DENOTES
is a role-binding that belongs to the S3 phase, so R2 does not supply it,
whereas a CONNOTES is a teachable gloss that R2 will offer. See the dialogue
specs for the script ↔ code ↔ rules triad this keeps in agreement.
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
    """Synthesize the next trainer KValue from ``(compiled, incoming, grounded)``.

    ``compiled`` is indexed once internally; the result is a function of
    ``incoming`` and ``grounded`` (the signatures K has grounded at S1 — the
    trainer's view of K's ratified knowledge). Implements R1 (opening), R2
    (reply to an identity), and R3 (echo a matching compiled kline).
    """
    grounded = grounded if grounded is not None else set()
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
        return _reply_identity(
            proposal.signature, decompositions, signifier, grounded
        )
    return _echo_compiled(proposal, decompositions, signifier)


# ── R1 — Opening ─────────────────────────────────────────────────────────


def _opening(primary: KLine) -> KValue:
    """R1 — emit the first compiled entry at S2."""
    return KValue(primary, SIG_S2)


# ── R2 — Reply to an identity ────────────────────────────────────────────
#
# When K asks ``{sig: []}`` at S4 ("I don't recognise sig"), T replies by the
# first rule that admits, in precedence order:
#
#   1. CANON  — a real canon of ``sig`` (``sig == make_signature(nodes)``):
#      teach its parts. Significance is pedagogical: S1 when K has grounded
#      every node, else S2 (some part is still unknown to K).
#   2. CONNOTES — a single-node connotation ``{sig: [node]}``: a teachable
#      gloss (e.g. ``a:[Det]``). Emitted at S2 (the operand is unknown to K).
#      A DENOTES is deliberately NOT supplied here: it is a role-binding
#      (e.g. ``Mary:[Subject]``) that belongs to the S3 phase, where K
#      proposes the binding and T ratifies it. Supplying it on the S4 ask
#      would pre-empt K's proposal.
#   3. otherwise — ``sig`` has only a compound-word identity, only DENOTES
#      role-bindings, or nothing at all. A compound-word identity
#      (``{sig: [CT, x, y]}``) is the grounding that decodes back into text,
#      so it is supplied at S1 in preference to the bare identity. Failing
#      that, T ratifies the identity at S1: ``{sig: []}`` at S1, signalling
#      "this is a primitive you may ground".


def _reply_identity(
    signature: int,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
    grounded: set[int],
) -> KValue:
    """R2 — the trainee does not recognise ``signature``; reply per §R2."""
    candidates = decompositions.get(signature, [])

    # 1. Canon — teach the parts. Significance is pedagogical: S1 when K has
    #    grounded every node, else S2 (some part is still unknown to K).
    canon = _first_canon(candidates, signifier)
    if canon is not None:
        significance = SIG_S1 if all(n in grounded for n in canon.nodes) else SIG_S2
        return KValue(canon, significance)

    # 2. CONNOTES — a teachable gloss at S2.
    connotes = _first_connotes(candidates)
    if connotes is not None:
        return KValue(connotes, SIG_S2)

    # 3. Compound identity — the subword decomposition
    #    ``{sig: [COMPOUND_TOKEN, x, y]}``. This is the grounding that decodes
    #    back into text, so it must be supplied (not the bare identity) when
    #    one is available. Ratified at S1.
    compound = _first_compound(candidates, signifier)
    if compound is not None:
        return KValue(compound, SIG_S1)

    # 4. Otherwise ratify the identity at S1 (a primitive K may ground).
    return KValue(KLine(signature, []), SIG_S1)


def _first_canon(
    candidates: list[KLine], signifier: KSignifier
) -> KLine | None:
    """The first real canon (``is_canon``) among ``candidates``, else None."""
    for kline in candidates:
        if is_canon(kline, signifier):
            return kline
    return None


def _first_connotes(candidates: list[KLine]) -> KLine | None:
    """The first CONNOTES relation among ``candidates``, else None.

    Distinguished from DENOTES by ``dbg.op``: a CONNOTES is a teachable gloss
    T supplies on the S4 ask; a DENOTES is a role-binding T leaves for the S3
    phase (K proposes, T ratifies).
    """
    for kline in candidates:
        op = kline.dbg.op if kline.dbg else None
        if op == "CONNOTES":
            return kline
    return None


def _first_compound(
    candidates: list[KLine], signifier: KSignifier
) -> KLine | None:
    """The first compound-word identity among ``candidates``, else None.

    A compound identity ``{sig: [COMPOUND_TOKEN, x, y]}`` is the subword
    decomposition of a single lexical item. It is the grounding that decodes
    back into text, so the supervisor supplies it in preference to the bare
    identity ratification whenever one is compiled.
    """
    for kline in candidates:
        if is_identity(kline) and kline.nodes:
            return kline
    return None


# ── R3 — Echo a matching compiled kline ──────────────────────────────────


def _echo_compiled(
    proposal: KLine,
    decompositions: dict[int, list[KLine]],
    signifier: KSignifier,
) -> KValue:
    """R3 — respond to K's non-identity proposal.

    Two cases:

    1. **Exact compiled match** — a compiled kline under ``proposal.signature``
       whose nodes equal ``proposal.nodes``: echo it verbatim. S1 for a
       relation (ratify), S2 for a canon (confirm).
    2. otherwise — emit the proposal back at S4 (T cannot endorse it here). A
       close that ratifies K's recombined misfit is the trainer's driving move:
       it is supplied by the scripted fallback on K's subsequent PASS, not
       synthesised here (the recombination's node order is the author's to fix).
    """
    for kline in decompositions.get(proposal.signature, []):
        if list(kline.nodes) == list(proposal.nodes):
            significance = SIG_S2 if is_canon(kline, signifier) else SIG_S1
            return KValue(kline, significance)
    return KValue(KLine(proposal.signature, proposal.nodes), SIG_S4)
