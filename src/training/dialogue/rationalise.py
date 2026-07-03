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
from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4, boundaries, classify
from kalvin.kline import KLine, is_canon, is_identity
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
    # Signatures K has already asked about as identities (Level 0). Tracked
    # separately because identities are popped on emission (Strategy B): an
    # asked signature must not be re-asked when a later unpack encounters it,
    # even though its identity work-item is gone. Consulted by _recognised.
    asked: set[int] = field(default_factory=set)


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
        - **S2 or S3** → **elevation check, then unpack**. If K's *own* current
          assessment of the kline is S1 — i.e. it is a relationship whose nodes
          are all now grounded — K elevates it: ground it via cleanup (the
          sender's declared S2 is honoured no higher than K's own re-derived
          S1; this is the two-way significance dialog). Otherwise unpack: push
          the query kline itself AND identity work-items ``{node: []}`` for
          each unrecognised node. This is how an asynchronously-grounded
          relationship (``{a:[Det]}`` re-received after Det grounded) finally
          grounds.
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

        # S2 or S3 — elevation check: a relationship whose nodes are all now
        # grounded is elevated to S1 (K's own assessment outranks the sender's
        # declared S2). This grounds async relationships on re-receipt.
        if self._elevatable(query.kline):
            self._cleanup(query.kline)
            return

        # Otherwise unpack the query kline and its unrecognised nodes.
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

    def _recognised(self, signature: int, *, is_signature: bool = False) -> bool:
        """Has K seen ``signature`` before?

        - **Nodes** (``is_signature=False``): recognised iff grounded, asked, or
          an identity work-item is in flight. Re-traversing an already-asked
          node in a NEW canon is legitimate (the new context may resolve it —
          e.g. re-asking ``a`` while traversing ALL, so the re-receipt can
          elevate ``{a:[Det]}``), so the asked set does NOT suppress node asks;
          it only suppresses *duplicate* identity work-items in flight.
        - **Signature** (``is_signature=True``): recognised iff grounded or
          asked. A signature K has already asked about is not re-asked when a
          later unpack encounters it as the signature (its identity work-item
          was popped on emission — Strategy B).
        """
        if signature in self._state.grounded:
            return True
        if is_signature and signature in self._state.asked:
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
        if not self._recognised(kline.signature, is_signature=True):
            self._state.work_list.append(KLine(kline.signature, []))

    def _cleanup(self, kline: KLine) -> None:
        """Ground ``kline`` and recursively ground every kline it unblocks.

        The grounding engine (Correction 3, recursive). Ground the triggering
        kline, then repeatedly ground any *groundable* work-list kline (a canon
        or an identity whose nodes are all grounded), removing it and
        continuing (grounding one kline may unblock others).

        **Only canons and identities ground by node-resolution** — never
        relationships. A canon ``{S:[nodes]}`` grounds when all its nodes
        ground; an identity ``{sig: []}`` ≡ ``{sig: [sig]}`` grounds when its
        own signature grounds. A relationship grounds by **elevation on
        re-receipt** (the entry rule), not here — so the opening relationship
        never grounds prematurely (it is received once, never re-received, and
        is closed by K's own S1 broadcast once its canon's operands resolve).
        """
        self._ground(kline)
        changed = True
        while changed:
            changed = False
            for i, entry in enumerate(self._state.work_list):
                if self._groundable(entry):
                    del self._state.work_list[i]
                    self._ground(entry)
                    changed = True
                    break

    def _groundable(self, kline: KLine) -> bool:
        """Is ``kline`` eligible to ground by node-resolution?

        Only **canons and identities** — never relationships:
        - **Identity** ``{sig: []}`` ≡ ``{sig: [sig]}`` — groundable iff its
          own signature is grounded (@CONTEXT.md §Identity).
        - **Canon** ``{S:[nodes]}`` — groundable iff all its nodes are grounded.
        - **Relationship** (non-canon, non-identity) — never groundable here;
          it grounds by elevation on re-receipt (the entry rule).
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        if is_canon(kline, self._signifier):
            return all(node in self._state.grounded for node in kline.nodes)
        return False  # a relationship — grounds only by elevation on re-receipt

    def _elevatable(self, kline: KLine) -> bool:
        """Should K elevate an incoming S2/S3 relationship to S1?

        True iff ``kline`` is a relationship (non-canon, non-identity) whose
        nodes are all now grounded. The sender declared S2 (e.g. the stateless
        trainer resubmitting ``{a:[Det]}`` after Det grounded); K re-derives its
        own significance from current model state and, finding the node
        grounded, elevates to S1. This is the async-grounding mechanism and the
        two-way significance dialog (K does not blindly accept the sender's
        significance). Canons/identities are handled by the S1/unpack branches.
        """
        if is_identity(kline) or is_canon(kline, self._signifier):
            return False
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

        Selection: LIFO among **workable** entries (a convention, placeholder
        for future significance-based selection — the work-list is a list, not a
        queue, because not every entry is workable at all times). An entry is
        workable iff it is:
        - an **identity** ``{sig: []}`` (always askable), or
        - a **Level-1-eligible opening** (a relationship whose operands both
          have grounded canons).
        An **async-pending relationship** (e.g. ``{a:[Det]}`` awaiting elevation
        on re-receipt) is NOT workable — K cannot emit about it now; it is
        skipped and removed later by cleanup when elevation grounds it.

        Returns ``None`` when no entry is workable (D12 — termination is the
        runner's job, not K's).
        """
        for idx in range(len(self._state.work_list) - 1, -1, -1):
            entry = self._state.work_list[idx]
            if is_identity(entry):
                return self._level0_identity(idx, entry.signature, incoming)
            if self._level1_eligible(entry):
                return self._level1_relationships(entry, incoming)
            # else: async-pending relationship — skip, look further down.
        return None  # D12 — nothing workable; the runner terminates.

    def _level1_eligible(self, entry: KLine) -> bool:
        """Is ``entry`` an opening relationship ready for Level 1?

        True iff ``entry`` is a single-node relationship ``{L:[R]}`` whose
        operands L and R BOTH have canons K has seen (in the work-list or
        grounded) — so K can read their operands to pair. The canons need not be
        *grounded*: Level 1 fires as soon as K has the operand lists, which may
        precede full grounding (e.g. MHALL's canon is seen before ``a`` grounds
        async). A single-node async-pending relationship like ``{a:[Det]}`` is
        not eligible (``a`` has no canon).
        """
        if is_identity(entry) or len(entry.nodes) != 1:
            return False
        return (
            self._find_canon_nodes(entry.signature) is not None
            and self._find_canon_nodes(entry.nodes[0]) is not None
        )

    def _level0_identity(
        self, idx: int, signature: int, incoming: RationaliseEvent | None
    ) -> RationaliseEvent:
        """Level 0 — emit IDENTITY ``{signature: []}`` at S4 and pop the entry.

        K calls out the signature as an identity: "I don't know this; what is
        it?" The identity work-item is **popped on emission** (Strategy B):
        the ask is fire-and-forget — K does not retain identities to re-ask.
        This is what lets K move past an async-blocked signature (e.g. ``a``,
        whose ``{a:[Det]}`` awaits elevation) to ask the next workable entry,
        rather than banging against a lingering ``{a: []}`` under pure LIFO.
        The signature becomes grounded later via the entry rule (S1, or
        elevation on re-receipt), independent of the popped identity.

        The emitted event's ``query`` voice is the incoming proposal (the
        thing K is responding to), or the identity itself on the opening —
        matching the table-actor convention. The runner validates only
        ``proposal`` and ``role`` (spec §Validation), so ``query`` is
        diagnostic only.
        """
        del self._state.work_list[idx]
        self._state.asked.add(signature)
        identity = KLine(signature, [])
        proposal = KValue(identity, SIG_S4)
        query = incoming.proposal if incoming is not None else proposal
        return RationaliseEvent(_KIND, query, proposal, role=_ROLE)

    def _level1_relationships(
        self, entry: KLine, incoming: RationaliseEvent | None
    ) -> RationaliseEvent:
        """Level 1 — propose operand relationships for a pending relationship, or close.

        ``entry`` is a multi-node relationship ``{L:[R]}`` whose operands L and
        R are MTS signatures with grounded canons (e.g. the opening
        ``{MHALL:[SVO]}``). K pairs the operands of L's canon and R's canon
        left-to-right at group size 1, grouping one side's residual into a
        single synthetic operand when the other reaches a single node (D10).

        Each call emits the next not-yet-ratified relationship:
        - a **1:1 relationship** ``{lhs:[rhs]}`` is emitted CONNOTED at S3
          (a tentative connoted relationship, inviting ratification).
        - a **synthesised residual** requires inventing a signature K was never
          taught; rather than assert a relationship to it (a leap too far), K
          emits a **canonical request** ``{make_signature(residual): residual}``
          at S2 — "is this a thing?" — the same shape as every other S2 query.
          The trainer replies with the scaffolding; K traverses it; async
          relationships (``{a:[Det]}``) ground by elevation on re-receipt.

        A pair is **resolved** when:
        - a 1:1 pair ``{lhs:[rhs]}`` is ratified (its kline grounded, trainer
          replied S1), OR
        - a grouped pair is resolved when its synthesised canon
          ``{make_signature(residual): residual}`` is grounded (the canonical
          request was traversed and its async relationships elevated).
        When every pair is resolved, K closes by emitting the entry itself at
        S1 (COUNTERSIGNED) — the broadcast that K grounds the opening query
        (Correction 1) — and the entry is removed from the work-list.

        Group-size escalation on a trainer S4 refusal (D11) is deferred.
        """
        right = entry.nodes  # the operand side of {L:[R]} — e.g. [SVO]
        assert len(right) == 1, "Level 1 expects a single-node relationship entry"
        left_sig = entry.signature  # e.g. MHALL

        # LHS signatures come from left_sig's canon (MHALL -> Mary, had, ...);
        # RHS nodes come from the operand's canon (SVO -> Subject, Verb, Object).
        left_nodes = self._find_canon_nodes(left_sig)
        right_nodes = self._find_canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            raise NotImplementedError(
                "Level 1: an operand canon is missing; cannot relate."
            )

        for lhs_sig, rhs_node, residual in self._relationship_plan(left_nodes, right_nodes):
            if self._pair_resolved(lhs_sig, rhs_node, residual):
                continue
            if residual:
                # A synthesised lhs (a signature K was never taught): emit a
                # canonical request {synthesised: residual} at S2 — "is this a
                # thing?" — not a relationship assertion. K cannot legitimately
                # relate to a signature it invented. Mark it asked so a later
                # unpack doesn't re-ask it as an identity.
                synth_sig = self._signifier.make_signature(residual)
                proposal_kline = KLine(synth_sig, residual)
                sig_band = SIG_S2
                self._state.asked.add(synth_sig)
            else:
                # A 1:1 relationship between known signatures: CONNOTED at S3.
                proposal_kline = KLine(lhs_sig, [rhs_node])
                sig_band = SIG_S3
            proposal = KValue(proposal_kline, sig_band)
            query = incoming.proposal if incoming is not None else proposal
            return RationaliseEvent(_KIND, query, proposal, role=_ROLE)

        # All pairs resolved — close at S1 (Correction 1: the only S1
        # broadcast). Ground the entry and remove it from the work-list.
        # (entry may be anywhere in the work-list, not just LIFO top, since
        # cogitation may have skipped async-pending entries above it.)
        self._state.work_list.remove(entry)
        self._ground(entry)
        proposal = KValue(entry, SIG_S1)
        query = incoming.proposal if incoming is not None else proposal
        return RationaliseEvent(_KIND, query, proposal, role=_ROLE)

    # ── Level 1 helpers ───────────────────────────────────────────────────

    def _canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s grounded canon, or None if none grounded.

        An MTS signature has a canon decomposition in K's grounded memory; this
        returns its node list. Used where grounded status matters.
        """
        for kline in self._state.grounded.get(signature, []):
            if is_canon(kline, self._signifier):
                return list(kline.nodes)
        return None

    def _find_canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, wherever K has seen it.

        Searches grounded memory AND the work-list. Used by Level 1: K can read
        a canon's operands as soon as it has seen the canon, even before the
        canon fully grounds (e.g. MHALL's canon is seen before ``a`` grounds
        async). Returns the first canon found's node list, or None.
        """
        for kline in self._state.grounded.get(signature, []):
            if is_canon(kline, self._signifier):
                return list(kline.nodes)
        for kline in self._state.work_list:
            if kline.signature == signature and is_canon(kline, self._signifier):
                return list(kline.nodes)
        return None

    def _relationship_plan(
        self, left_nodes: list[int], right_nodes: list[int]
    ) -> list[tuple[int, int, list[int]]]:
        """Pair two canons' operands into ``(lhs_sig, rhs_node, residual)`` tuples.

        Group-size-1 convention (D10): pair left-to-right while both sides have
        more than one node remaining; when one side reaches a single node,
        group the other side's entire residual into one synthetic operand and
        pair it with the remaining single. ``residual`` is empty for a 1:1 pair;
        for a grouped pair it carries the residual node list (so the caller can
        emit a canonical request ``{make_signature(residual): residual}``).
        """
        plan: list[tuple[int, int, list[int]]] = []
        i = j = 0
        while i < len(left_nodes) and j < len(right_nodes):
            left_rem = len(left_nodes) - i
            right_rem = len(right_nodes) - j
            if left_rem == 1 and right_rem == 1:
                plan.append((left_nodes[i], right_nodes[j], []))
                i += 1
                j += 1
            elif left_rem == 1:
                # Group right's residual; lhs is the single left node.
                residual = list(right_nodes[j:])
                plan.append((left_nodes[i], self._signifier.make_signature(residual), residual))
                break
            elif right_rem == 1:
                # Group left's residual into a synthetic lhs; rhs is single.
                residual = list(left_nodes[i:])
                plan.append((self._signifier.make_signature(residual), right_nodes[j], residual))
                break
            else:
                plan.append((left_nodes[i], right_nodes[j], []))
                i += 1
                j += 1
        return plan

    def _pair_resolved(
        self, lhs_sig: int, rhs_node: int, residual: list[int]
    ) -> bool:
        """Is this relationship-plan pair resolved?

        - **1:1 pair** (``residual`` empty): resolved iff ``{lhs_sig:[rhs_node]}``
          is grounded (the trainer ratified it at S1).
        - **Grouped pair** (``residual`` non-empty): resolved iff the synthesised
          canon ``{make_signature(residual): residual}`` is grounded — i.e. the
          canonical request was traversed and any async relationships within it
          elevated. The synthesised canon grounds via cleanup once all its
          residual nodes ground.
        """
        if residual:
            synth_sig = self._signifier.make_signature(residual)
            return any(
                is_canon(kl, self._signifier) for kl in self._state.grounded.get(synth_sig, [])
            )
        return self._relationship_ratified(lhs_sig, rhs_node)

    def _relationship_ratified(self, lhs_sig: int, rhs_node: int) -> bool:
        """Has the trainer ratified the relationship ``{lhs_sig:[rhs_node]}``?

        True iff that relationship kline is grounded (the trainer replied S1,
        which cleanup grounded). Used by Level 1 to find the next unratified
        relationship.
        """
        for kline in self._state.grounded.get(lhs_sig, []):
            if list(kline.nodes) == [rhs_node]:
                return True
        return False
