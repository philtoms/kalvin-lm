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
from kalvin.expand import SIG_S1, SIG_S3, SIG_S4, boundaries, classify
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
        kline, then repeatedly ground any *groundable* work-list kline (a canon
        or an identity whose nodes are all grounded), removing it and
        continuing (grounding one kline may unblock others).

        **Only canons and identities ground by node-resolution** — never
        relationships. A canon ``{S:[nodes]}`` grounds when all its nodes
        ground (K has all the pieces of the decomposition); an identity
        ``{sig: []}`` ≡ ``{sig: [sig]}`` grounds when its own signature grounds.
        A relationship (``{MHALL:[SVO]}``, ``{a:[Det]}``) does NOT ground this
        way — it grounds only by explicit ratification (an S1 for that kline).
        This stops the opening relationship from grounding prematurely when
        its operands' nodes resolve.
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

        - **Identity** ``{sig: []}`` ≡ ``{sig: [sig]}`` — groundable iff its
          own signature is grounded (@CONTEXT.md §Identity).
        - **Canon** ``{S:[nodes]}`` — groundable iff all its nodes are grounded
          (K has all the pieces of the decomposition).
        - **Relationship** ``{S:[node]}`` (non-canon, non-identity) — groundable
          iff all its nodes are grounded **and ``S`` is not an MTS**. A
          single-token (non-MTS) signature's relationship is *terminal*: when
          its node grounds, the signature is fully resolved (e.g. ``a`` ↔ Det).
          An MTS signature's relationship is *non-terminal*: grounding its node
          does not resolve it — understanding ``MHALL`` ↔ ``SVO`` requires the
          operand binding (Level 1), not just ``SVO`` grounding. So an MTS
          relationship grounds only by explicit ratification.
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        if is_canon(kline, self._signifier):
            return all(node in self._state.grounded for node in kline.nodes)
        # A relationship: groundable only if non-MTS and all nodes grounded.
        if self._is_mts(kline.signature):
            return False
        return all(node in self._state.grounded for node in kline.nodes)

    def _is_mts(self, signature: int) -> bool:
        """Is ``signature`` a Multi-Token Signature (a compound)?

        An MTS is the OR-reduction of two or more token values
        (@specs/signifier SIG-14; @specs/kscript §8 MTS expansion). Operationally,
        for the rationaliser: a signature is an MTS iff K has a grounded **canon**
        under it — i.e. K knows a decomposition of the signature into multiple
        tokens. A single-token signature (e.g. ``a``) has no canon, so it is not
        an MTS; its relationships are terminal. A compound (``MHALL``, ``SVO``,
        ``Det``) has a canon, so it is an MTS; its relationships need binding.
        """
        for kline in self._state.grounded.get(signature, []):
            if is_canon(kline, self._signifier):
                return True
        return False

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

        # Level 1 — a pending multi-node (relationship) kline. Propose the
        # operand bindings between its signature's canon and its node's canon,
        # or close at S1 when all bindings are ratified.
        return self._level1_relationships(entry, incoming)

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

    def _level1_relationships(
        self, entry: KLine, incoming: RationaliseEvent | None
    ) -> RationaliseEvent:
        """Level 1 — propose operand bindings for a pending relationship, or close.

        ``entry`` is a multi-node relationship ``{L:[R]}`` whose operands L and
        R are MTS signatures with grounded canons (e.g. the opening
        ``{MHALL:[SVO]}``). K pairs the operands of L's canon and R's canon
        left-to-right at group size 1, grouping one side's residual into a
        single synthetic operand when the other reaches a single node (D10).

        Each call emits the **first not-yet-ratified** binding as a CONNOTED
        relationship at S3 (1:1) or S2 (grouped). A binding is ratified once
        its kline is grounded (the trainer replied S1). When every binding is
        ratified, K closes by emitting the entry itself at S1 (COUNTERSIGNED)
        — the broadcast that K grounds the opening query (Correction 1) — and
        the entry is removed from the work-list.

        Group-size escalation on a trainer S4 refusal (D11) is deferred: the
        bootstrap targets golden masters the convention satisfies at size 1.
        """
        right = entry.nodes  # the operand side of {L:[R]} — e.g. [SVO]
        assert len(right) == 1, "Level 1 expects a single-node relationship entry"
        left_sig = entry.signature  # e.g. MHALL

        # LHS signatures come from left_sig's canon (MHALL -> Mary, had, ...);
        # RHS nodes come from the operand's canon (SVO -> Subject, Verb, Object).
        left_nodes = self._canon_nodes(left_sig)
        right_nodes = self._canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            # A canon is missing — cannot bind. Defer (should not happen on a
            # well-formed golden master where both operands grounded).
            raise NotImplementedError(
                "Level 1: an operand canon is missing; cannot bind."
            )

        plan = self._binding_plan(left_nodes, right_nodes)

        for lhs_sig, rhs_node in plan:
            if not self._binding_ratified(lhs_sig, rhs_node):
                # Emit the next unratified binding. Significance is the emitted
                # kline's signature-to-node count (grill Q3): a 1:1 proposal
                # (one signature, one node) is S3; a multi-node proposal is S2.
                # MHALL's proposals are all 1:1 (grouping makes a synthetic lhs,
                # not multiple rhs nodes), so all are S3 here. The S2 (multi-node)
                # branch is coverage gap G1, unexercised by MHALL.
                proposal_kline = KLine(lhs_sig, [rhs_node])
                proposal = KValue(proposal_kline, SIG_S3)
                query = incoming.proposal if incoming is not None else proposal
                return RationaliseEvent(_KIND, query, proposal, role=_ROLE)

        # All bindings ratified — close at S1 (Correction 1: the only S1
        # broadcast). Ground the entry and remove it from the work-list.
        self._state.work_list.pop()  # the entry is the LIFO top
        self._ground(entry)
        proposal = KValue(entry, SIG_S1)
        query = incoming.proposal if incoming is not None else proposal
        return RationaliseEvent(_KIND, query, proposal, role=_ROLE)

    # ── Level 1 helpers ───────────────────────────────────────────────────

    def _canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s grounded canon, or None if none.

        An MTS signature has a canon decomposition in K's grounded memory; this
        returns its node list. Used by Level 1 to get the operands of each side
        of a pending relationship.
        """
        for kline in self._state.grounded.get(signature, []):
            if is_canon(kline, self._signifier):
                return list(kline.nodes)
        return None

    def _binding_plan(
        self, left_nodes: list[int], right_nodes: list[int]
    ) -> list[tuple[int, int]]:
        """Pair two canons' operands into a list of ``(lhs_sig, rhs_node)``.

        Group-size-1 convention (D10): pair left-to-right while both sides have
        more than one node remaining; when one side reaches a single node,
        group the other side's entire residual into one synthetic operand
        (``make_signature(residual)``) and pair it with the remaining single.
        Returns the plan as emitted-relationship shapes ``(signature, [node])``;
        a grouped lhs carries a synthetic signature.
        """
        plan: list[tuple[int, int]] = []
        i = j = 0
        while i < len(left_nodes) and j < len(right_nodes):
            left_rem = len(left_nodes) - i
            right_rem = len(right_nodes) - j
            if left_rem == 1 and right_rem == 1:
                plan.append((left_nodes[i], right_nodes[j]))
                i += 1
                j += 1
            elif left_rem == 1:
                # Group right's residual; lhs is the single left node.
                residual = right_nodes[j:]
                plan.append((left_nodes[i], self._signifier.make_signature(residual)))
                break
            elif right_rem == 1:
                # Group left's residual into a synthetic lhs; rhs is single.
                residual = left_nodes[i:]
                plan.append((self._signifier.make_signature(residual), right_nodes[j]))
                break
            else:
                plan.append((left_nodes[i], right_nodes[j]))
                i += 1
                j += 1
        return plan

    def _binding_ratified(self, lhs_sig: int, rhs_node: int) -> bool:
        """Has the trainer ratified the binding ``{lhs_sig:[rhs_node]}``?

        True iff that relationship kline is grounded (the trainer replied S1,
        which cleanup grounded). Used by Level 1 to find the next unratified
        binding.
        """
        for kline in self._state.grounded.get(lhs_sig, []):
            if list(kline.nodes) == [rhs_node]:
                return True
        return False
