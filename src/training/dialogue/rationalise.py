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
from kalvin.kline import KLine, is_identity
from kalvin.kvalue import KValue

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
        return self._cogitate(incoming)

    # ── Entry rule (plan §Entry rule) ─────────────────────────────────────

    def _process_query(self, incoming: RationaliseEvent) -> None:
        """Apply the entry rule to an incoming query (plan §Entry rule).

        Fires on every received query, before cogitation. It is bookkeeping
        only — it never emits; emission is cogitation's job (D7). Three cases
        by the query's significance band:

        - **S4** → retire the matching identity work-item (the other side says
          "I don't know this either" — stalemate accepted, leaf bottomed
          out). S4 is a sentinel detected by value, not by band (see @agent
          spec §Phase 1b).
        - **S1** → run cleanup, which grounds the kline and recurses: any
          work-list kline whose nodes are all now grounded is itself grounded
          and removed, transitively. Silent — an S1 emission (broadcast)
          happens only when K grounds the opening query, not on every pop
          (Correction 1) — so this branch never emits.
        - **S2 or S3** → **unpack**: push the query kline itself AND identity
          work-items ``{node: []}`` for each unrecognised node, all into the
          one work-list. The query kline is held pending until cleanup grounds
          it (its nodes all ground); the identities are what K asks about.
        """
        query = incoming.proposal
        sig = query.significance

        # S4 sentinel — detected by value (classify() cannot return S4).
        if sig == SIG_S4:
            self._pop_identity(query.kline.signature)
            return

        band = classify(sig, self._s12, self._s23, self._s34)
        if band == "S1":
            self._cleanup(query.kline)
            return

        # S2 or S3 — unpack the query kline and its unrecognised nodes.
        self._unpack(query.kline)

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
        """Is ``signature`` grounded?"""
        return signature in self._state.grounded

    def _recognised(self, signature: int) -> bool:
        """Has K seen ``signature`` before, as an identity or grounded kline?

        Recognised iff the signature is grounded OR already has an *identity*
        work-item in flight. A pending multi-node query kline under the
        signature does NOT count — K still wants to ask about that signature.
        This is the unpack push-decision: a recognised signature (or node) is
        not pushed as a fresh identity work-item.
        """
        if signature in self._state.grounded:
            return True
        return any(
            entry.signature == signature and is_identity(entry)
            for entry in self._state.work_list
        )

    def _unpack(self, kline: KLine) -> None:
        """Unpack an S2/S3 kline: push the kline, its unrecognised nodes, and (if
        the signature is new) an identity work-item for the signature.

        The query kline ``{S: [n1, ...]}`` is pushed first (held pending until
        cleanup grounds it via node-resolution), then each unrecognised node,
        right-to-left, as an identity work-item ``{node: []}``, then — if the
        signature is itself unrecognised — ``{S: []}`` last (LIFO top), so K
        asks about the signature before descending into its nodes. A reply
        (signature already in flight) does not re-push ``{S: []}``.

        Idempotent per signature: a node or signature already recognised
        (grounded or in flight) is not pushed again.
        """
        self._state.work_list.append(kline)
        for node in reversed(kline.nodes):
            if not self._recognised(node):
                self._state.work_list.append(KLine(node, []))
        if not self._recognised(kline.signature):
            self._state.work_list.append(KLine(kline.signature, []))

    def _cleanup(self, kline: KLine) -> None:
        """Ground ``kline`` and recursively ground every kline it unblocks.

        The grounding engine (Correction 3, recursive). Ground the triggering
        kline, then repeatedly scan the work-list for any kline whose **nodes
        are all now grounded** — ground it, remove it, and recurse (grounding
        one kline may unblock others). The rule is uniform: an identity
        ``{sig: []}`` ≡ ``{sig: [sig]}`` is groundable iff its own signature is
        grounded; a multi-node kline is groundable iff all its nodes are.

        No S2→S1 promotion: a kline is grounded only structurally (an S1
        arrived for it, or all its nodes grounded). This is what retires a
        pending query kline like ``{a:[Det]}`` once Det grounds — `a` is
        grounded by the structural fact that its relationship's node resolved.
        """
        self._ground(kline)
        # Repeatedly ground any work-list kline whose nodes are all grounded,
        # removing it and continuing until no more can be grounded.
        changed = True
        while changed:
            changed = False
            for i, entry in enumerate(self._state.work_list):
                if self._nodes_all_grounded(entry):
                    del self._state.work_list[i]
                    newly = not self._is_grounded_sig(entry.signature)
                    self._ground(entry)
                    changed = True
                    if newly:
                        # Grounding this signature may unblock others; loop.
                        pass
                    break

    def _nodes_all_grounded(self, kline: KLine) -> bool:
        """Are all of ``kline``'s nodes grounded?

        Uniform over identity and multi-node klines: an identity ``{sig: []}``
        ≡ ``{sig: [sig]}``, so its nodes are all grounded iff its own signature
        is grounded (@CONTEXT.md §Identity).
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        return all(node in self._state.grounded for node in kline.nodes)

    def _pop_identity(self, signature: int) -> None:
        """Pop the first identity work-item under ``signature`` (S4 branch).

        Used by the entry rule's S4 branch to retire an identity the other
        side also doesn't know. No-op if no matching identity entry exists.
        """
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == signature and is_identity(entry):
                del self._state.work_list[i]
                return

    # ── Cogitation (plan §Cogitation) ──────────────────────────────────────

    def _cogitate(self, incoming: RationaliseEvent | None) -> RationaliseEvent | None:
        """Work the selected work-list entry and emit exactly one event.

        Selection (D6): LIFO — the most-recently-added entry. Dispatch is by
        kline **shape** (structural), not state:

        - **Level 0 (Identity)** — the entry is an identity ``{sig: []}`` ≡
          ``{sig: [sig]}``. Emit IDENTITY ``{sig: []}`` at S4. The entry stays
          in the list; it is retired by cleanup when its signature grounds, or
          by the S4 branch on a matching stalemate.
        - **Level 1 (Relationships)** — the entry is a multi-node (pending
          query) kline. Filled in a subsequent step.

        Returns ``None`` when the work-list is empty (D12 — termination is the
        runner's job, not K's).
        """
        if not self._state.work_list:
            return None  # D12 — no work; the runner terminates.

        entry = self._state.work_list[-1]  # LIFO (D6, convention)

        if is_identity(entry):
            return self._level0_identity(entry.signature, incoming)

        # Level 1 (Relationships) — subsequent step.
        raise NotImplementedError(
            "Rationaliser Level 1 (Relationships) is filled in a subsequent step."
        )

    def _level0_identity(
        self, signature: int, incoming: RationaliseEvent | None
    ) -> RationaliseEvent:
        """Level 0 — emit IDENTITY ``{signature: []}`` at S4.

        K has never seen ``signature``, so it calls it out as an identity:
        "I don't know this; what is it?" The identity work-item stays on the
        work-list; it is retired later by the entry rule (S1 cleanup when K
        grounds the signature, or the S4 branch on a matching stalemate).

        The emitted event's ``query`` voice is the incoming proposal (the
        thing K is responding to), or the identity itself on the opening —
        matching the table-actor convention. The runner validates only
        ``proposal`` and ``role`` (spec §Validation), so ``query`` is
        diagnostic only.
        """
        identity = KLine(signature, [])
        proposal = KValue(identity, SIG_S4)
        query = incoming.proposal if incoming is not None else proposal
        return RationaliseEvent(_KIND, query, proposal, role=_ROLE)
