r"""The rationalising engine — the pure-logic core of a rationalising trainee.

This module holds the rationalising **engine**: a stateless object that
derives each turn from ``(state, incoming)`` and returns a ``(batch,
observations)`` pair. It knows nothing of ``RationaliseEvent``, roles, or
event kinds — the :class:`~training.dialogue.actors.RationalisingTrainee`
wrapper owns the :class:`RationaliserState` and wraps each emitted
``KValue`` in a ``RationaliseEvent``.

The **batch** holds dialogue emissions — speech acts K addresses to T (S4
identity asks, S3 connotation proposals, S2 similar-fit proposals). The
**observations** hold K's internal S1 grounding events (same shape as the
batch: a list of ``KValue``\ s), surfaced for white-box verification. Every
kline K grounds produces one observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.expand import (
    SIG_S1,
    SIG_S2,
    SIG_S3,
    SIG_S4,
    boundaries,
    structural_significance,
)
from kalvin.kline import KLine, is_canon, is_identity, is_misfit
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Rationaliser", "RationaliserState"]


@dataclass
class RationaliserState:
    """The Rationaliser's mutable memory, owned by the actor.

    A work-list of pending klines (work-list entries carry no
    significance band — dispatch is structural) and the grounded model
    (keyed by signature). Passed into
    :meth:`Rationaliser.rationalise` and mutated in place by reference.
    Emission deduplication is the actor's responsibility, not the engine's:
    the engine is free to re-derive an emission, and the actor drops any it
    has already published.

    The **frame** holds previously cogitated (emitted) klines — same
    structure as ``grounded`` (keyed by signature, deduplicated on nodes).
    The fast route matches incoming subjective-S1/S4 queries against it by
    structural signature; only genuinely-new emissions are added to it (see
    :meth:`_Turn.frame_batch`).
    """

    work_list: list[KLine] = field(default_factory=list)
    grounded: dict[int, list[KLine]] = field(default_factory=dict)
    frame: dict[int, list[KLine]] = field(default_factory=dict)
    _dbg_step: int = 0

class Rationaliser:
    r"""The rationalising engine — derives each turn from ``(state, incoming)``.

    Stateless: holds only the signifier and the significance boundaries. Each
    :meth:`rationalise` call applies the entry rule to the incoming query as
    bookkeeping, then cogitates. Returns a ``(batch, observations)`` pair:
    the batch is the dialogue emissions (empty when nothing is workable), the
    observations are the S1 grounding events K performed this turn.
    """

    def __init__(self, signifier: KSignifier) -> None:
        self._signifier = signifier
        self._s12, self._s23, self._s34 = boundaries()

    # ── The turn ─────────────────────────────────────────────────────

    def rationalise(
        self, state: RationaliserState, incoming: Sequence[KValue]
    ) -> tuple[list[KValue], list[KValue]]:
        """Route every incoming query, then cogitate.

        Mutates ``state`` in place. Returns ``(batch, observations)``: the
        batch holds dialogue emissions (speech acts for T), the observations
        hold the S1 grounding events K performed this turn (for white-box
        verification).

        Two routes, dispatched on the query's **subjective** significance
        (the producer's surface stamp): the **fast route** (subjective S1/S4)
        matches the query against ``state.frame`` and promotes or pops;
        the **slow route** (everything else) is the work-list cogitation.
        Frame is the last thing updated before returning: the batch is
        filtered to genuinely-new emissions, those are added to the frame,
        and the filtered batch is returned.
        """
        state._dbg_step += 1
        turn = _Turn(state, self._signifier, self._s12, self._s23, self._s34)
        for query in incoming:
            turn.route(query)
        batch = turn.cogitate()
        batch = turn.frame_batch(batch)
        return batch, turn.observations


class _Turn:
    """A single rationalise turn: scoped accumulators over a shared state.

    Confines mutation to one ``rationalise`` call. Holds the state (mutated by
    reference), the signifier, the significance boundaries, and accumulates the
    turn's ``observations`` (the batch is returned from :meth:`cogitate`).
    """

    def __init__(
        self,
        state: RationaliserState,
        signifier: KSignifier,
        s12: int,
        s23: int,
        s34: int,
    ) -> None:
        self._state = state
        self._signifier = signifier
        self._s12, self._s23, self._s34 = s12, s23, s34
        self.observations: list[KValue] = []

    # ── Routing ──────────────────────────────────────────────────────

    def route(self, query: KValue) -> None:
        """Route an incoming query first on subjective significance, then on structural significance.

        Two routes, fast and slow, dispatched on the query's **subjective**
        significance (the producer's surface stamp, ``query.significance``):

        - **Fast route** (subjective S1, S4) — structurally match the query
          against ``state.frame`` (same signature, same *structural*
          significance). An S1 match **promotes** the kline (grounds it at S1
          and cascades any node-resolution it unblocks); an S4 match **pops**
          the matching identity ask from the work-list. Either way the framed
          kline is consumed (removed from the frame). An unmatched query is
          dropped silently. The fast route never appends to the work-list.
        - **Slow route** (subjective S2, S3) — append to the work-list at the
          query's *structural* band, then unpack unrecognised nodes and
          signature as S4 identity asks for :meth:`cogitate`.
        """

        if query.significance in (SIG_S1, SIG_S4):
            self._fast_route(query)
            return

        kline = query.kline
        self._state.work_list.append(kline)

        # Unrecognised nodes and signature become identity asks (S4) on the
        # work-list, for cogitate to emit.
        for node in kline.nodes:
            if not self._is_recognised(node):
                self._state.work_list.append(KLine(node, [], kline.dbg))
        if not self._is_recognised(kline.signature):
            self._state.work_list.append(KLine(kline.signature, [], kline.dbg))

    def _fast_route(self, query: KValue) -> None:
        """Match ``query`` (S1 or S4) against the frame.

        Structural match: a framed kline with the same signature and the same
        *structural* significance as ``query.kline``. On a match the framed kline is
        consumed (unframed) and:

        - **S4** — pop the pending identity ask for ``kline``'s
          signature from the work-list (T has answered K's ask).
        - **S1** — promote ``kline``: ground it at S1 and cascade
          any node-resolution the new grounding unblocks.

        An unmatched query is dropped silently.
        """
        kline = query.kline
        sig = query.significance

        if not self._match_in_frame(kline):
            return  # unmatched — dropped silently

        self._unframe(kline)
        if sig == SIG_S4:
            self._pop_identity(kline.signature)
            return

        self._promote(kline)

    def _promote(self, kline: KLine) -> None:
        """Ground ``kline`` at S1, then drain any node-resolution it unblocks.

        A subjective-S1 match in the frame means another participant has
        countersigned a kline K previously cogitated; K now grounds it. The
        new grounding may unblock identities or canons whose nodes newly
        resolve, so this cascades: any work-list entry that becomes groundable
        is grounded in turn, until no further grounding is possible. Each
        grounding flows through :meth:`_ground`, which grounds the reciprocal
        of a relationship kline as part of its own bookkeeping.
        """
        self._ground(kline)
        changed = True
        while changed:
            changed = False
            for i, entry in enumerate(self._state.work_list):
                if self._is_groundable(entry):
                    del self._state.work_list[i]
                    self._ground(entry)
                    changed = True
                    break

    def _pop_identity(self, signature: int) -> None:
        """Pop the first identity work-item under ``signature`` (S4 branch).

        Called from the fast route on a subjective-S4 match: T has supplied
        an identity K previously asked about, so the pending ask is withdrawn.
        A no-op when no identity ask is pending.
        """
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == signature and is_identity(entry):
                del self._state.work_list[i]
                return

    def _match_in_frame(self, kline: KLine) -> bool:
        """The framed kline matching ``kline``, else None.

        Match is on signature and *structural* significance: the query's
        subjective band selects the route, but the frame lookup is structural
        (two klines frame-match when they are the same kline structurally).
        """
        target = structural_significance(kline, self._signifier)
        for framed in self._state.frame.get(kline.signature, []):
            if structural_significance(framed, self._signifier) == target:
                return True
        return False

    def _unframe(self, kline: KLine) -> None:
        """Remove ``kline`` from the frame (exact signature + nodes match)."""
        bucket = self._state.frame.get(kline.signature)
        if not bucket:
            return
        self._state.frame[kline.signature] = [
            k for k in bucket if not (k.signature == kline.signature and k.nodes == kline.nodes)
        ]
        if not self._state.frame[kline.signature]:
            del self._state.frame[kline.signature]

    def cogitate(self) -> list[KValue]:
        """Ground, countersign, or propose: amongst all entries in the work-list
        are those ready to be grounded, countersigned, or expanded into new proposals.
        """
        batch: list[KValue] = []
        for idx in range(len(self._state.work_list) - 1, -1, -1):
            kline = self._state.work_list[idx]

            if is_identity(kline):
                batch.append(self._emit_identity(idx, kline.signature))
                continue

            if structural_significance(kline, self._signifier) == SIG_S1:
                self._promote(kline)
                continue
            
            if not batch:
                # First workable non-identity dispatches on structure:
                #   S3 connoted (single node) → canon-countersign
                #   S2 misfit (multi-node) → similar-fit proposal
                if self._is_countersignable(kline):
                    pairings = self._expand_connotation(kline)
                    if pairings:
                        return pairings
                    continue
                if is_misfit(kline, self._signifier):
                    batch = self._propose_similar_fit(kline)

        return batch

    # ── State ─────────────────────────────────────────────────────────────

    def _is_groundable(self, kline: KLine) -> bool:
        """Is ``kline`` ready to ground at S1?

        Two cases: an identity whose signature is grounded; or a
        canon whose nodes are all grounded. S2 multi-node misfits never ground
        here — they propose.
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        if is_canon(kline, self._signifier):
            return all(node in self._state.grounded for node in kline.nodes)
        if len(kline.nodes) != 1:
            return False  # S2 misfit — propose, never ground
        reciprocal = KLine(kline.nodes[0], [kline.signature])
        return self._is_grounded(reciprocal)

    def _is_grounded(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in grounded memory?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self._state.grounded.get(kline.signature, [])
        )

    def _ground(self, kline: KLine, countersigning = False) -> None:
        """Record that K has grounded ``kline`` and observe it at S1.

        Idempotent on nodes — an already-grounded kline is not re-observed.
        """
        bucket = self._state.grounded.setdefault(kline.signature, [])
        if any(existing.nodes == kline.nodes for existing in bucket):
            return
        bucket.append(kline)
        self.observations.append(KValue(kline, SIG_S1))
        if not countersigning and self._is_countersignable(kline):
            reciprocal_sig = self._signifier.make_signature(kline.nodes)
            reciprocal = KLine(reciprocal_sig, [kline.signature])
            self._ground(reciprocal, True)

    def _is_recognised(self, signature: int) -> bool:
        """Has K seen ``signature`` — grounded or pending as an identity?"""
        if signature in self._state.grounded:
            return True
        return any(
            entry.signature == signature and is_identity(entry)
            for entry in self._state.work_list
        )

    def _is_countersignable(self, entry: KLine) -> bool:
        """Is ``entry`` a single-node relationship whose two operands have seen
        canons, so K can pursue an S1 countersignature for it?"""
        if is_identity(entry) or len(entry.nodes) != 1:
            return False
        return (
            self._find_canon_nodes(entry.signature) is not None
            and self._find_canon_nodes(entry.nodes[0]) is not None
        )

    def frame_batch(self, batch: list[KValue]) -> list[KValue]:
        """Filter ``batch`` to genuinely-new emissions and add them to the frame.

        The frame holds previously cogitated (emitted) klines. An emission
        whose kline is already framed (exact signature + nodes match, like
        :meth:`_is_grounded`) is dropped from the returned batch; the rest
        are added to the frame and returned. Run as the last action of a
        turn, before returning the batch.
        """
        new_batch: list[KValue] = []
        for value in batch:
            if self._is_framed(value.kline):
                continue
            self._frame(value.kline)
            new_batch.append(value)
        return new_batch

    def _is_framed(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in the frame?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self._state.frame.get(kline.signature, [])
        )

    def _frame(self, kline: KLine) -> None:
        """Record that K has emitted ``kline`` into a batch (add to frame).

        Idempotent on nodes — call sites filter via :meth:`_is_framed` first.
        """
        self._state.frame.setdefault(kline.signature, []).append(kline)

    # ── Cogitation ────────────────────────────────────────────────────────

    def _emit_identity(self, idx: int, signature: int) -> KValue:
        """S4 — emit IDENTITY ``{signature: []}`` at S4 and pop the entry."""
        del self._state.work_list[idx]
        return KValue(KLine(signature, []), SIG_S4)

    def _expand_connotation(self, entry: KLine) -> list[KValue]:
        """Emit every unresolved S3 pairing for ``entry`` in one batch, or ``[]``
        if all pairings are already grounded.

        Pair the two canons' operands left-to-right at group size 1, grouping
        one side's residual into a single synthesised operand when the other
        reaches a single node. Both a 1:1 pair and a grouped residual are
        CONNOTES at S3: the residual is synthesised into a signature that is
        substituted for ``lhs_sig``, so the grouped pair takes the same S3
        connotation path as a 1:1 pair.
        """
        right = entry.nodes
        assert len(right) == 1, "S3 pairings expect a single-node relationship entry"
        left_nodes = self._find_canon_nodes(entry.signature)
        right_nodes = self._find_canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            raise NotImplementedError(
                "S3 pairings: an operand canon is missing; cannot relate."
            )

        batch: list[KValue] = []
        for lhs_sig, rhs_node, residual in self._relationship_plan(left_nodes, right_nodes):
            if self._pair_resolved(lhs_sig, rhs_node, residual):
                continue
            # A grouped residual is synthesised into a signature and
            # substituted for lhs_sig, so the grouped pair takes the same
            # S3 connotation path as a 1:1 pair.
            head_sig = (
                self._signifier.make_signature(residual) if residual else lhs_sig
            )
            proposal_kline = KLine(head_sig, [rhs_node])
            significance = SIG_S3
            batch.append(KValue(proposal_kline, significance))

        # Every unresolved pairing collected in one batch (empty iff all were
        # already grounded) — the signal for the countersignature to proceed.
        return batch

    # ── S2 path (misfit canonisation) ────────────────────────────────────

    def _propose_similar_fit(self, entry: KLine) -> list[KValue]:
        """S2 path — originate a misfit proposal by accumulated canonisation (shaping).

        Shape one proposal by processing candidates in preference order:
        (1) node-expansion — replace each node that is a grounded kline's
        signature with that kline's nodes; (2) node-graft — resolve each
        shared-node candidate against the accumulated target. The accumulated
        target is emitted at S2 (dropped if already grounded). The entry stays
        in the work-list.
        """
        target = list(entry.nodes)
        target = self._apply_node_expansions(target)
        for candidate in self._s2_candidates(entry):
            # Resolve ``target`` against ``candidate.nodes`` into a resolved core.
            # The candidate fires iff the core is non-empty (a foothold); on firing
            # the new target is the resolved core + the candidate's nodes not in
            # the core. If it does not fire, ``target`` is returned unchanged.
            core = self._resolve_target(target, list(candidate.nodes))
            if core:
                c_open = [n for n in candidate.nodes if n not in core]
                target = core + c_open

        # Drop if the shaped proposal is already grounded.
        proposal = KLine(entry.signature, target)
        if self._is_grounded(proposal):
            return []
        return [KValue(proposal, SIG_S2)]

    def _resolve_target(
        self, target: list[int], candidate_nodes: list[int]
    ) -> list[int]:
        """Resolve ``target`` against ``candidate_nodes`` into the resolved core.

        The **core** is the portion of ``target`` that resolves into the
        candidate (directly or via a grounded kline whose signature is in the
        candidate). Target nodes that resolve to neither are dropped (open
        slots filled by the candidate's surplus in the caller). Iterates to
        fixed point.
        """
        candidate_set = set(candidate_nodes)
        core: list[int] = []
        remaining = list(target)
        while True:
            direct = [n for n in remaining if n in candidate_set]
            failed = [n for n in remaining if n not in candidate_set]
            if not failed:
                core.extend(direct)
                return core
            resolved = self._partition_and_resolve(failed)
            # Of the resolved failed nodes, those whose resolution landed in
            # candidate_set join the core; the rest are dropped (open slots).
            newly_matched = [n for n in resolved if n in candidate_set]
            core.extend(direct)
            core.extend(newly_matched)
            if not newly_matched:
                # No progress: remaining nodes are genuinely unresolvable.
                return core
            remaining = [n for n in resolved if n not in candidate_set]

    def _apply_node_expansions(self, target: list[int]) -> list[int]:
        """Rule 1 — replace each node that is a grounded kline's signature with
        that kline's nodes."""
        expanded: list[int] = []
        for node in target:
            sub = self._find_grounded_nodes(node)
            expanded.extend(sub if sub is not None else [node])
        return expanded

    def _find_grounded_nodes(self, signature: int) -> list[int] | None:
        """The nodes of any grounded kline under ``signature`` with non-empty
        nodes, else None."""
        for kline in self._state.grounded.get(signature, []):
            if kline.nodes:
                return list(kline.nodes)
        return None

    def _partition_and_resolve(self, failed: list[int]) -> list[int]:
        """Maximally cover ``failed`` with disjoint grounded-kline node-sets.

        Each coverable subset is replaced by its grounded kline's signature,
        plus any uncoverable leftovers. Greedy is insufficient (a larger kline
        may block two smaller ones that together cover more), so this searches
        for a maximal-disjoint cover.
        """
        # Collect cover-set candidates: (kline_nodes, kline_signature), deduped.
        # A grounded kline is a candidate cover-set iff its nodes are a subset
        # of `failed`.
        covers: list[tuple[tuple[int, ...], int]] = []
        seen_sigs: set[int] = set()
        failed_set = set(failed)
        for bucket in self._state.grounded.values():
            for kline in bucket:
                if kline.signature in seen_sigs:
                    continue
                if not kline.nodes:
                    continue
                if set(kline.nodes).issubset(failed_set):
                    covers.append((tuple(kline.nodes), kline.signature))
                    seen_sigs.add(kline.signature)
        # Search for a maximal disjoint cover (by node count covered).
        best = self._best_disjoint_cover(covers, failed_set)
        if not best:
            return list(failed)
        covered: set[int] = set()
        resolved: list[int] = []
        for canon_nodes, canon_sig in best:
            resolved.append(canon_sig)
            covered.update(canon_nodes)
        # Append any failed nodes no cover-set accounted for.
        resolved.extend(n for n in failed if n not in covered)
        return resolved

    def _best_disjoint_cover(
        self, covers: list[tuple[tuple[int, ...], int]], failed_set: set[int]
    ) -> list[tuple[tuple[int, ...], int]]:
        """The maximal-disjoint subset of ``covers`` (by total nodes covered)."""
        best: list[tuple[tuple[int, ...], int]] = []
        best_covered = 0

        def _recurse(
            idx: int,
            chosen: list[tuple[tuple[int, ...], int]],
            used: set[int],
            covered: int,
        ) -> None:
            nonlocal best, best_covered
            if covered > best_covered:
                best_covered = covered
                best = list(chosen)
            for i in range(idx, len(covers)):
                kline_nodes, _ = covers[i]
                if used & set(kline_nodes):
                    continue  # overlaps an already-chosen cover
                chosen.append(covers[i])
                _recurse(i + 1, chosen, used | set(kline_nodes), covered + len(kline_nodes))
                chosen.pop()

        _recurse(0, [], set(), 0)
        return best

    def _s2_candidates(self, entry: KLine) -> list[KLine]:
        """Grounded klines sharing at least one node value with ``entry.nodes``.

        Admission is keyed on the entry's nodes (not its head signature). The
        entry itself is excluded, and a canon under the entry's own signature is
        never admitted — it is the entry's resolution, not a recombination
        ingredient. Identities drop out (no nodes).
        """
        entry_nodes = set(entry.nodes)
        candidates: list[KLine] = []
        for bucket in self._state.grounded.values():
            for kline in bucket:
                if kline is entry:
                    continue
                if not kline.nodes:
                    continue  # identities carry no nodes
                # The canonical resolution of the entry's own signature is the
                # answer, not an ingredient: never admit it as a candidate.
                if kline.signature == entry.signature and is_canon(
                    kline, self._signifier
                ):
                    continue
                if entry_nodes & set(kline.nodes):
                    candidates.append(kline)
        return candidates

    # ── S3 helpers ────────────────────────────────────────────────────────

    def _find_canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, searching grounded memory and the
        work-list."""
        for kline in self._state.grounded.get(signature, []):
            if is_canon(kline, self._signifier):
                return list(kline.nodes)
        for entry in self._state.work_list:
            if entry.signature == signature and is_canon(entry, self._signifier):
                return list(entry.nodes)
        return None

    def _relationship_plan(
        self, left_nodes: list[int], right_nodes: list[int]
    ) -> list[tuple[int, int, list[int]]]:
        """Pair two canons' operands into ``(lhs_sig, rhs_node, residual)`` tuples.

        Group-size-1 convention: pair left-to-right while both sides have more
        than one node remaining; when one side reaches a single node, group the
        other side's entire residual into one synthesised operand.
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
                residual = list(right_nodes[j:])
                plan.append(
                    (left_nodes[i], self._signifier.make_signature(residual), residual)
                )
                break
            elif right_rem == 1:
                residual = list(left_nodes[i:])
                plan.append(
                    (self._signifier.make_signature(residual), right_nodes[j], residual)
                )
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

        A pair — whether a 1:1 pair or a grouped residual — is resolved when its
        CONNOTES proposal ``{head_sig:[rhs_node]}`` is grounded (ratified). For a
        grouped residual ``head_sig`` is the signature synthesised from the
        residual; for a 1:1 pair it is ``lhs_sig``.
        """
        head_sig = (
            self._signifier.make_signature(residual) if residual else lhs_sig
        )
        return any(
            list(kline.nodes) == [rhs_node]
            for kline in self._state.grounded.get(head_sig, [])
        )
