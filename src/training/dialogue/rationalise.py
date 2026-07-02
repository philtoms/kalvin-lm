"""The Rationalising Trainee — a drop-in replacement for ``TableTrainee``.

Plan: ``@plans/implement-rationalising-trainee.md``.
Spec seam: ``@specs/dialogue-driven-training.md`` §Actor, §The Runner,
§Validation (unchanged — the rationaliser satisfies the existing contract).

Unlike the table-reading ``TableTrainee`` (which yields the table's K-rows in
order), the Rationaliser **rationalises**: it maintains a minimal model of what
it has grounded and derives each turn from ``(incoming, state)``, never reading
the authored table or the compiled script. The table is the **golden master** —
the validation oracle the runner checks every emitted turn against; this module
reads it zero times.

The Rationaliser is a **bootstrap double**: a genuine rationaliser in mechanism,
destined to be replaced only by another rationaliser (the full ``KAgent``),
never by a synthesizer. Its cogitation is deliberately simplified —
synchronous, deterministic, inline — standing in for the real Kalvin's async
``expand()`` / ``propose_expansions()`` slow path. See the plan's Design
Decisions (D1–D12) and §The Rationaliser Mechanism.

This is the **initial shell** (plan Phase 1.1, in progress). The entry rule,
the two cogitation levels (Identity, Relationships), the grouping convention,
and group-size escalation are stubbed below and filled in subsequent steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S4, boundaries, classify
from kalvin.kline import KLine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Rationaliser"]

# Significance band classification — used by the entry rule to classify an
# incoming query's significance. S4 is a sentinel detected by *value*
# (``== SIG_S4``), not by band: ``classify()`` collapses the S3|S4 boundary
# so the band function can never return S4 (see @agent spec §Phase 1b).

# The role the Rationaliser announces on every emitted event (the routing key
# the runner validates against — spec §Validation, DDT-17).
_ROLE = "K"
# Every Rationaliser event is a "frame" event (the dialogue vocabulary's only
# kind for an actor turn).
_KIND = "frame"


# ── Minimal bespoke state (plan D3) ───────────────────────────────────────
#
# The Rationaliser does NOT use ``kalvin.Model`` (STM/Frame/LTM cascade). Its
# simplified cogitation needs only: the work-list of outstanding S2/S3 entries
# and a ``grounded`` memory that mimics ``KModel`` — a dict keyed by signature,
# each value the list of grounded klines under that signature. There is no
# ordered distinction between identities and relationships: they are all
# klines. See plan §The Minimal State.


@dataclass
class _State:
    """The Rationaliser's mutable memory (plan D3, §The Minimal State).

    - ``work_list`` — outstanding S2/S3 entries, selected LIFO (plan D6; the
      LIFO order is a convention standing in for future significance-based
      selection).
    - ``grounded`` — signatures K has grounded, mirroring ``KModel``: a dict
      keyed by signature, each value the list of grounded klines under it.
      Identities and relationships are stored alike — there is no ordered
      distinction; everything is a kline. Level 0 vs Level 1 is decided by
      lookup ("does K have a record of this signature?"), and Level 1's
      "matched by an entry" is a (signature, nodes) membership check against
      the klines under a signature.
    """

    work_list: list[KLine] = field(default_factory=list)
    grounded: dict[int, list[KLine]] = field(default_factory=dict)


# ── The Rationaliser ──────────────────────────────────────────────────────


class Rationaliser:
    """A stateful trainee actor that rationalises (plan §Purpose, D1–D12).

    Satisfies the ``Actor`` protocol (``@specs/dialogue-driven-training.md``
    §Actor) with ``role="K"``. Each :meth:`respond` call:

    1. Applies the **entry rule** to ``incoming`` as bookkeeping (S4 → pop a
       matching entry; S1 → ground immediately and pop a matching entry; S2/S3
       → push and enter cogitation at Level 0).
    2. Runs **cogitation** on the selected work-list entry (LIFO) — Level 0
       (Identity) or Level 1 (Relationships) — and emits exactly one event.
    3. Returns ``None`` when the work-list is empty after the entry rule (the
       runner then signals termination — plan D12; termination is the runner's
       job, not K's).

    Reads neither the table nor the compiled script nor ``dbg`` (plan D1, D2,
    D4). Constructs synthetic signatures via ``signifier.make_signature`` when
    the grouping convention requires it (plan D9, D10).
    """

    def __init__(self, signifier: KSignifier) -> None:
        self._signifier = signifier
        self._state = _State()
        # Significance boundaries, computed once (they are constants).
        self._s12, self._s23, self._s34 = boundaries()

    @property
    def role(self) -> str:
        """The role this actor announces on its events (the routing key)."""
        return _ROLE

    # ── Actor protocol ────────────────────────────────────────────────────

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None:
        """Rationalise one turn: process the query, then process the next step.

        Applies the **entry rule** to ``incoming`` as bookkeeping, then runs
        cogitation on the selected work-list entry and emits exactly one event.
        Returns ``None`` when the work-list is empty after the entry rule (the
        runner then signals termination — plan D12).

        .. note::

            Targets the **refactored** ``Actor`` contract
            (``RationaliseEvent | None`` — synthesizing-trainer plan D7), NOT
            the current cursor-leaking ``tuple[int, RationaliseEvent] | None``.
            The runner/actor refactor (synthesizing-trainer D7) is a
            prerequisite for integration; until then the Rationaliser cannot
            be driven by the unrefactored runner.
        """
        # Step 1 — entry rule: process the query (bookkeeping).
        if incoming is not None:
            self._process_query(incoming)

        # Step 2 — process next step: cogitation emits exactly one event,
        # or returns None when the work-list is empty (D12).
        raise NotImplementedError(
            "Rationaliser.respond: cogitation (Levels 0 and 1) is filled in "
            "a subsequent step."
        )

    # ── Entry rule (plan §Entry rule) ─────────────────────────────────────

    def _process_query(self, incoming: RationaliseEvent) -> None:
        """Apply the entry rule to an incoming query (plan §Entry rule).

        Fires on every received query, before cogitation. It is bookkeeping
        only — it never emits; emission is cogitation's job (D7). Three cases
        by the query's significance band:

        - **S4** matching a work-list entry → pop the entry (the other side
          says "I don't know this either" — stalemate accepted, leaf bottomed
          out). S4 is a sentinel detected by value, not by band (see @agent
          spec §Phase 1b).
        - **S1** → ground it immediately in K's state; if it matches a
          work-list entry, pop it (the other side says "I understand this").
        - **S2 or S3** → push onto the work-list; cogitation will work it.
        """
        query = incoming.proposal
        sig = query.significance

        # S4 sentinel — detected by value (classify() cannot return S4).
        if sig == SIG_S4:
            self._pop_matching(query.kline)
            return

        band = classify(sig, self._s12, self._s23, self._s34)
        if band == "S1":
            self._ground(query.kline)
            self._pop_matching(query.kline)
            return

        # S2 or S3 — push for cogitation.
        self._state.work_list.append(query.kline)

    # ── State helpers ─────────────────────────────────────────────────────

    def _ground(self, kline: KLine) -> None:
        """Record that K has grounded ``kline`` (the trainee's own S1).

        Mirrors ``KModel``: append the kline under its signature. Shape
        (identity, canon, relationship) is irrelevant to storage — everything
        is a kline. Idempotent: grounding an already-grounded kline is a
        no-op (K already understands it).
        """
        bucket = self._state.grounded.setdefault(kline.signature, [])
        for existing in bucket:
            if existing.nodes == kline.nodes:
                return  # already grounded
        bucket.append(kline)

    def _is_grounded_sig(self, signature: int) -> bool:
        """Does K have a record of ``signature``? (decides Level 0 vs Level 1)."""
        return signature in self._state.grounded

    def _pop_matching(self, kline: KLine) -> None:
        """Pop the first work-list entry whose (signature, nodes) match.

        Used by the entry rule's S4 and S1 branches to retire an entry the
        other side has now resolved. No-op if no entry matches.
        """
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == kline.signature and entry.nodes == kline.nodes:
                del self._state.work_list[i]
                return
