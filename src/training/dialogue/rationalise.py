r"""The rationalising engine — the pure-logic core of a rationalising trainee.

This module holds the rationalising **engine**: a stateful object that
derives each turn from ``(incoming, state)`` and returns a batch of
``KValue``\ s. It knows nothing of ``RationaliseEvent``, roles, or event
kinds — the :class:`~training.dialogue.actors.RationalisingTrainee` wrapper
wraps each emitted ``KValue`` in a ``RationaliseEvent``.

The engine maintains a minimal model of what it has grounded. Each
:meth:`Rationaliser.rationalise` call applies the entry rule to the incoming
query as bookkeeping, then emits a batch from cogitation. Returns an empty
list when nothing is workable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4, boundaries, classify
from kalvin.kline import KLine, is_canon, is_identity, is_misfit
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Rationaliser"]


@dataclass
class _State:
    """The Rationaliser's mutable memory: a work-list, the grounded model, and
    the set of signatures already asked about as identities."""

    work_list: list[KLine] = field(default_factory=list)
    grounded: dict[int, list[KLine]] = field(default_factory=dict)
    asked: set[int] = field(default_factory=set)


class Rationaliser:
    r"""The rationalising engine — derives each turn from ``(incoming, state)``.

    Returns a **batch** of ``KValue``\ s per :meth:`rationalise` call (or an
    empty list when nothing is workable). Constructs synthetic signatures via
    ``signifier.make_signature`` when grouping requires it.
    """

    def __init__(self, signifier: KSignifier) -> None:
        self._signifier = signifier
        self._state = _State()
        self._s12, self._s23, self._s34 = boundaries()

    # ── The turn ─────────────────────────────────────────────────────

    def rationalise(self, incoming: Sequence[KValue]) -> list[KValue]:
        """Route every incoming query, then cogitate.

        A route may emit a batch directly (the S1 canon-countersignature); the
        first such batch is returned verbatim and cogitation is skipped.
        """
        for query in incoming:
            emitted = self._route(query)
            if emitted is not None:
                return emitted
        return self._cogitate()

    def _route(self, query: KValue) -> list[KValue] | None:
        """Route an incoming query based on its surface significance.

        S1 grounds/cleans (or takes the canon-countersignature branch when T
        ratifies a misfit whose nodes are all grounded); S4 pops the identity
        ask; S2 and S3 might be promotable - these are also grounded.
        Otherwise the query is unpacked for cogitation. Returns a batch when a
        route emits directly, else ``None``.
        """
        kline = query.kline
        sig = query.significance

        if sig == SIG_S4:
            self._pop_identity(kline.signature)
            return None

        if classify(sig, self._s12, self._s23, self._s34) == "S1":
            if self._is_canon_countersignable(kline):
                return self._canon_countersign(kline)
            self._promote(kline)
            return None

        # S2 or S3.
        if self._is_promotable(kline):
            self._promote(kline)
        else:
            self._unpack(kline)
        return None

    def _cogitate(self) -> list[KValue]:
        """Work the next workable entry (LIFO) and emit a batch."""
        batch: list[KValue] = []
        for idx in range(len(self._state.work_list) - 1, -1, -1):
            entry = self._state.work_list[idx]
            if is_identity(entry):
                batch.append(self._emit_identity(idx, entry.signature))
                continue
            if not batch:
                # First workable non-identity dispatches on structure:
                #   S3 connoted (single node) → countersign
                #   S2 misfit (multi-node) → canonize
                if self._is_countersignable(entry):
                    batch = self._expand_connotation(entry)
                    if batch:
                        return batch
                    return self.countersign(entry)
                if is_misfit(entry, self._signifier):
                    batch = self._propose_similar_fit(entry)

        return batch

    # ── State ─────────────────────────────────────────────────────────────

    def _is_groundable(self, kline: KLine) -> bool:
        """Is ``kline`` eligible to ground by node-resolution?

        Only identities and canons — relationships ground by elevation on
        re-receipt.
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        if is_canon(kline, self._signifier):
            return all(node in self._state.grounded for node in kline.nodes)
        return False

    def _is_grounded(self, kline: KLine) -> bool:
        """Is an isomorphic kline (same signature and nodes) in grounded memory?"""
        return any(
            existing.nodes == kline.nodes
            for existing in self._state.grounded.get(kline.signature, [])
        )

    def _ground(self, kline: KLine) -> None:
        """Record that K has grounded ``kline``. Idempotent on nodes."""
        bucket = self._state.grounded.setdefault(kline.signature, [])
        if not any(existing.nodes == kline.nodes for existing in bucket):
            bucket.append(kline)

    def _is_promotable(self, kline: KLine) -> bool:
        """Should K promote an incoming S2/S3 relationship to S1?

        True iff ``kline`` is a relationship whose nodes are all now grounded.
        """
        if is_identity(kline) or is_canon(kline, self._signifier):
            return False
        return all(node in self._state.grounded for node in kline.nodes)

    def _promote(self, kline: KLine) -> None:
        """Ground ``kline``, then repeatedly ground any work-list kline it unblocks."""
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

    def _is_recognised(self, signature: int, *, is_signature: bool = False) -> bool:
        """Has K seen ``signature``?"""
        if signature in self._state.grounded:
            return True
        if is_signature and signature in self._state.asked:
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

    def _pop_identity(self, signature: int) -> None:
        """Pop the first identity work-item under ``signature`` (S4 branch)."""
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == signature and is_identity(entry):
                del self._state.work_list[i]
                return

    # ── Cogitation ────────────────────────────────────────────────────────


    def _unpack(self, kline: KLine) -> None:
        """Push an S2/S3 kline, its unrecognised nodes, and (if new) its signature."""
        self._state.work_list.append(kline)
        for node in kline.nodes:
            if not self._is_recognised(node):
                self._state.work_list.append(KLine(node, [], kline.dbg))
        if not self._is_recognised(kline.signature, is_signature=True):
            self._state.work_list.append(KLine(kline.signature, [],kline.dbg))

    def _emit_identity(self, idx: int, signature: int) -> KValue:
        """S4 — emit IDENTITY ``{signature: []}`` at S4 and pop the entry."""
        del self._state.work_list[idx]
        self._state.asked.add(signature)
        return KValue(KLine(signature, []), SIG_S4)

    def countersign(self, entry: KLine) -> list[KValue]:
        """Establish the S1 countersignature for ``entry``.

        ``entry`` is ``{L:[R]}`` whose operands L and R are signatures with
        seen canons. Emit both directions at S1.
        """
        self._state.work_list.remove(entry)
        self._ground(entry)
        reciprocal_sig = self._signifier.make_signature(entry.nodes)
        reciprocal = KLine(reciprocal_sig, [entry.signature])
        self._ground(reciprocal)
        return [KValue(entry, SIG_S1), KValue(reciprocal, SIG_S1)]

    def _is_canon_countersignable(self, kline: KLine) -> bool:
        """Should ``kline`` take the S1 canon-countersignature branch?

        True iff ``kline`` is an S1 relationship ``{S: nodes}`` with more than
        one node, all grounded. 1:1 shapes (connotations, denotations) and
        identities fall through to promotion — they ground directly without a
        countersignature emission.
        """
        if len(kline.nodes) <= 1:
            return False
        return all(node in self._state.grounded for node in kline.nodes)

    def _canon_countersign(self, entry: KLine) -> list[KValue]:
        """Establish the S1 canon-countersignature for a ratified misfit.

        T ratified ``{S: nodes}`` at S1. K does not ground the misfit; it
        computes ``C = make_signature(nodes)`` (the canon the ratified nodes
        form), promotes C by grounding ``{C: nodes}``, then emits both
        directions of the reciprocal pair ``{S: [C]}`` and ``{C: [S]}`` at S1.
        Pending work-list entries under S are dropped.
        """
        canon_sig = self._signifier.make_signature(entry.nodes)
        canon = KLine(canon_sig, list(entry.nodes))
        self._ground(canon)
        left = KLine(entry.signature, [canon_sig])
        right = KLine(canon_sig, [entry.signature])
        self._ground(left)
        self._ground(right)
        # Drop pending work-list entries under S (the original misfits).
        self._state.work_list = [
            kl for kl in self._state.work_list if kl.signature != entry.signature
        ]
        return [KValue(left, SIG_S1), KValue(right, SIG_S1)]

    def _expand_connotation(self, entry: KLine) -> list[KValue]:
        """Emit every unresolved S3 pairing for ``entry`` in one batch, or ``[]``
        if all pairings are already grounded.

        Pair the two canons' operands left-to-right at group size 1, grouping
        one side's residual into a single synthesised operand when the other
        reaches a single node. A 1:1 pair is CONNOTES at S3; a grouped residual
        is a canonical request at S2.
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
            if residual:
                # A synthesised lhs K was untaught: emit a canonical request
                # at S2, and mark it asked so unpack won't re-ask it as an identity.
                synth_sig = self._signifier.make_signature(residual)
                proposal_kline = KLine(synth_sig, residual)
                significance = SIG_S2
                self._state.asked.add(synth_sig)
            else:
                proposal_kline = KLine(lhs_sig, [rhs_node])
                significance = SIG_S3
            batch.append(KValue(proposal_kline, significance))

        # Every unresolved pairing collected in one batch (empty iff all were
        # already grounded) — the signal for the countersignature to proceed.
        return batch

    # ── S2 path (misfit origination) ────────────────────────────────────

    def _propose_similar_fit(self, entry: KLine) -> list[KValue]:
        """S2 path — originate a misfit proposal by accumulated shaping.

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
            # Resolve ``target`` against ``candidate.nodes`` into a resolved core and
            # the target's open nodes. The candidate fires iff the core is non-empty;
            # on firing the new target is the resolved core + the candidate's nodes
            # not in the core. If it does not fire, ``target`` is returned unchanged.
            core, e_open = self._resolve_target(target, list(candidate.nodes))
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
    ) -> tuple[list[int], list[int]]:
        """Resolve ``target`` against ``candidate_nodes`` into (core, open).

        The **core** is the portion of ``target`` that resolves into the
        candidate (directly or via a grounded kline whose signature is in the
        candidate); the **open** list is the target nodes that resolve to
        neither. Iterates to fixed point.
        """
        candidate_set = set(candidate_nodes)
        core: list[int] = []
        remaining = list(target)
        while True:
            direct = [n for n in remaining if n in candidate_set]
            failed = [n for n in remaining if n not in candidate_set]
            if not failed:
                core.extend(direct)
                return core, []
            resolved = self._partition_and_resolve(failed)
            # Of the resolved failed nodes, those whose resolution landed in
            # candidate_set join the core; the rest stay unresolved (open or
            # pending further resolution).
            newly_matched = [n for n in resolved if n in candidate_set]
            still_failed = [n for n in resolved if n not in candidate_set]
            core.extend(direct)
            core.extend(newly_matched)
            if not newly_matched:
                # No progress: still_failed are genuinely open (unresolvable).
                return core, still_failed
            remaining = still_failed

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

    # ── S2 rule 2 precondition: must_match resolution ──────────────────

    def _resolve_must_match(
        self, must_match: list[int], candidate_nodes: list[int]
    ) -> tuple[list[int], bool]:
        """Resolve ``must_match`` against a graft candidate's nodes to fixed point.

        Returns ``(resolved_must_match, fully_matched)``. A node is directly
        matched if it appears in ``candidate_nodes``; the failed set resolves
        through grounded klines, iterating until every node is matched (graft
        proceeds) or a pass produces no change (candidate rejected).
        """
        current = list(must_match)
        while True:
            failed = [n for n in current if n not in candidate_nodes]
            if not failed:
                return current, True
            resolved = self._partition_and_resolve(failed)
            if resolved == failed:
                # No kline covered any failed node this pass: stuck. Any
                # remaining failed node means the candidate cannot account for
                # the accumulated structure.
                return current, False
            # Rebuild must_match: keep matched nodes, replace failed with their
            # resolution (resolved signatures + uncoverable leftovers).
            current = [n for n in current if n in candidate_nodes] + resolved

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
        for kline in self._state.work_list:
            if kline.signature == signature and is_canon(kline, self._signifier):
                return list(kline.nodes)
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
        """Is this relationship-plan pair resolved? A 1:1 pair is resolved when
        ``{lhs_sig:[rhs_node]}`` is grounded; a grouped pair when its
        synthesised canon is grounded."""
        if residual:
            synth_sig = self._signifier.make_signature(residual)
            return any(
                is_canon(kl, self._signifier)
                for kl in self._state.grounded.get(synth_sig, [])
            )
        # 1:1 pair: resolved iff {lhs_sig:[rhs_node]} is grounded.
        return any(
            list(kline.nodes) == [rhs_node]
            for kline in self._state.grounded.get(lhs_sig, [])
        )
