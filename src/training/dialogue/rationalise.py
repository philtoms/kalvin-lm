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

        - **S4** matching a work-list entry → pop the entry (the other side
          says "I don't know this either" — stalemate accepted, leaf bottomed
          out). S4 is a sentinel detected by value, not by band (see @agent
          spec §Phase 1b).
        - **S1** → ground it immediately, then clean up the work-list (pop
          any entry whose nodes include this signature and whose nodes are all
          now grounded). The other side says "I understand this." An S1
          emission (broadcast) happens only when K grounds the opening query —
          not on every pop (Correction 1) — so this branch is silent.
        - **S2 or S3** → **unpack** the kline right-to-left (nodes then
          signature), pushing an identity work-item ``{sig: []}`` for each
          signature not already recognised (grounded or in flight). This makes
          node processing identical to any other work-item (Correction 2).
        """
        query = incoming.proposal
        sig = query.significance

        # S4 sentinel — detected by value (classify() cannot return S4).
        if sig == SIG_S4:
            self._pop_matching_identity(query.kline.signature)
            return

        band = classify(sig, self._s12, self._s23, self._s34)
        if band == "S1":
            self._ground(query.kline)
            self._cleanup_s1(query.kline.signature)
            return

        # S2 or S3 — two cases:
        # (a) The signature matches an in-flight identity work-item: this S2/S3
        #   is the trainer's *reply* to K's ask (e.g. K asks IDENTITY `a`;
        #   trainer replies `{a:[Det]}` S2). Ground the kline (K now has the
        #   answer — a decomposition or relationship) and retire the identity.
        # (b) Unsolicited (no matching identity, e.g. the opening turn):
        #   push the signature's identity so K asks about it.
        # Nodes are unpacked first so the signature (when pushed) is the LIFO
        # top — K asks about the signature before descending into its nodes.
        self._unpack_nodes(query.kline)
        if self._has_identity_in_flight(query.kline.signature):
            self._ground(query.kline)
            self._pop_matching_identity(query.kline.signature)
        elif not self._recognised(query.kline.signature):
            self._state.work_list.append(KLine(query.kline.signature, []))

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
        """Is ``signature`` a grounded signature? (the Level-0 ask-decision).

        Distinct from :meth:`_recognised` (the unpack push-decision): an
        identity work-item ``{sig: []}`` is in flight but not grounded, so it
        must still be asked about — the ask-decision keys on grounding alone.
        """
        return signature in self._state.grounded

    def _recognised(self, signature: int) -> bool:
        """Has K seen ``signature`` before?

        A signature is **recognised** if K has any kline under it — either
        grounded (K holds it) or in flight (the work-list has an entry under
        it). This is the push-decision predicate for unpacking (Correction 2)
        and the Level-0 ask-decision for cogitation: a recognised signature is
        not re-asked as an identity.
        """
        if signature in self._state.grounded:
            return True
        return any(entry.signature == signature for entry in self._state.work_list)

    def _has_identity_in_flight(self, signature: int) -> bool:
        """Is there an in-flight identity work-item under ``signature``?

        True iff the work-list contains an identity entry ``{signature: []}``.
        Used by the S2/S3 branch to distinguish a *reply* to K's ask (signature
        matches an in-flight identity → ground + retire) from an unsolicited
        S2 (no matching identity → push the signature's identity).
        """
        return any(
            entry.signature == signature and not entry.nodes
            for entry in self._state.work_list
        )

    def _unpack_nodes(self, kline: KLine) -> None:
        """Unpack an S2/S3 kline's *nodes* right-to-left into identity work-items.

        For ``{S: [n1, n2, ...]}``: each node, right-to-left, is checked for
        recognition; if not recognised (grounded or in flight), an identity
        work-item ``{node: []}`` is pushed. Right-to-left ordering with LIFO
        popping means the first node is worked first (Correction 2).

        Only the nodes — the signature is handled by the caller, which decides
        whether to ground it (a reply) or push its identity (unsolicited).
        Idempotent per node: a node already recognised is not pushed again.
        """
        for node in reversed(kline.nodes):
            if not self._recognised(node):
                self._state.work_list.append(KLine(node, []))

    def _cleanup_s1(self, signature: int) -> None:
        """After grounding ``signature``, retire resolved identity work-items.

        Pop every work-list identity entry whose own signature is now grounded
        (Correction 3). An identity work-item ``{sig: []}`` is retired when K
        grounds ``{sig: ...}``. (Multi-node entries are Level-1 territory,
        handled later.)
        """
        self._state.work_list = [
            entry
            for entry in self._state.work_list
            if not (not entry.nodes and entry.signature in self._state.grounded)
        ]

    def _pop_matching_identity(self, signature: int) -> None:
        """Pop the first identity work-item under ``signature`` (S4 branch).

        Used by the entry rule's S4 branch to retire an identity the other
        side also doesn't know. No-op if no matching identity entry exists.
        """
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == signature and not entry.nodes:
                del self._state.work_list[i]
                return

    # ── Cogitation (plan §Cogitation) ──────────────────────────────────────

    def _cogitate(self, incoming: RationaliseEvent | None) -> RationaliseEvent | None:
        """Work the selected work-list entry and emit exactly one event.

        Selection (D6): LIFO — the most-recently-added entry. After unpacking
        (Correction 2), the work-list is a stack of identity work-items
        ``{sig: []}``; the LIFO top is the next signature to ask about.

        - **Level 0 (Identity)** — the entry's signature is NOT grounded
          (K does not yet understand it). Emit IDENTITY ``{sig: []}`` at S4.
          The entry stays in the list; it is retired by the entry rule's S1
          cleanup when K later grounds the signature, or by the S4 branch on a
          matching stalemate. (The ask-decision is *grounded*, not *recognised*:
          the identity entry itself is, by definition, not yet grounded.)
        - **Level 1 (Relationships)** — the entry's signature IS grounded.
          Filled in a subsequent step.

        Returns ``None`` when the work-list is empty (D12 — termination is the
        runner's job, not K's).
        """
        if not self._state.work_list:
            return None  # D12 — no work; the runner terminates.

        entry = self._state.work_list[-1]  # LIFO (D6, convention)

        if not self._is_grounded_sig(entry.signature):
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
