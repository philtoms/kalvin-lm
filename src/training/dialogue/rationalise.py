r"""The rationalising engine — the pure-logic core of a rationalising trainee.

Plan: ``@plans/implement-rationalising-trainee.md``.
Spec seam: ``@specs/dialogue-driven-training.md`` §Actor (the trainee side).

This module holds the rationalising **engine**: a stateful object that
derives each turn from ``(incoming, state)`` and returns a ``KValue``. It knows
nothing of ``RationaliseEvent``, roles, or event kinds — the actor wrapper
(:class:`~training.dialogue.runner.RationalisingTrainee`) lives in the runner
and wraps each emitted ``KValue`` in a ``RationaliseEvent``, mirroring how
:func:`~training.dialogue.synthesize.synthesize` is the engine for
:class:`~training.dialogue.runner.SynthesizingTrainer`.

The engine maintains a minimal model of what it has grounded and never reads
the authored table or the compiled script. The table is only the validation
oracle the runner checks every emitted turn against.

Cogitation is a deliberate simplification of the real Kalvin's async
``expand()`` / ``propose_expansions()`` slow path — synchronous, deterministic,
inline (plan D3, D5). Each :meth:`Rationaliser.rationalise` call applies the
entry rule to the incoming query as bookkeeping, then emits a **batch** of
``KValue``\ s from cogitation — an identity blast (zero or more S4 asks) or a
single relationship emission (a relationship always terminates the batch and
is never appended to identities). Returns an empty list when nothing is
workable (plan D7, D12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4, boundaries, classify
from kalvin.kline import KLine, is_canon, is_identity
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Rationaliser"]


@dataclass
class _State:
    """The Rationaliser's mutable memory.

    ``grounded`` mirrors ``KModel``: keyed by signature, each value the list of
    grounded klines under it. Identities and relationships are stored alike.
    ``asked`` tracks signatures already asked about as identities (popped on
    emission) so unpack does not re-ask them.
    """

    work_list: list[KLine] = field(default_factory=list)
    grounded: dict[int, list[KLine]] = field(default_factory=dict)
    asked: set[int] = field(default_factory=set)


class Rationaliser:
    r"""The rationalising engine — derives each turn from ``(incoming, state)``.

    Returns a **batch** of ``KValue``\ s per :meth:`rationalise` call (or an
    empty list when nothing is workable) — an identity blast or a single
    relationship emission. The :class:`~training.dialogue.runner.RationalisingTrainee`
    actor wraps each emitted value in a ``RationaliseEvent``. Reads neither the
    table nor the compiled script nor ``dbg``; constructs synthetic signatures
    via ``signifier.make_signature`` when grouping requires it.
    """

    def __init__(self, signifier: KSignifier) -> None:
        self._signifier = signifier
        self._state = _State()
        self._s12, self._s23, self._s34 = boundaries()

    # ── The turn ─────────────────────────────────────────────────────

    def rationalise(self, incoming: KValue | None) -> list[KValue]:
        """Apply the entry rule to ``incoming``, then emit a batch.

        Returns a list of emitted values: an identity blast (zero or more S4
        identity asks), a single relationship emission (S2/S3/S1), or an empty
        list when nothing is workable. Identities and relationships are never
        mixed in one batch (a relationship always terminates the batch)."""
        if incoming is not None:
            self._process_query(incoming)
        return self._cogitate()

    # ── Entry rule ────────────────────────────────────────────────────────

    def _process_query(self, incoming: KValue) -> None:
        """Bookkeeping for an incoming query; never emits.

        - **S4** (sentinel by value — ``classify`` never returns it): pop the
          matching identity work-item (stalemate accepted).
        - **S1**: cleanup — ground the kline and recurse over what it unblocks.
        - **S2/S3**: elevate an elevatable relationship to S1 (grounding it via
          cleanup), else unpack it.
        """
        kline = incoming.kline
        sig = incoming.significance

        if sig == SIG_S4:
            self._pop_identity(kline.signature)
            return

        if classify(sig, self._s12, self._s23, self._s34) == "S1":
            self._cleanup(kline)
            return

        # S2 or S3.
        if self._elevatable(kline):
            self._cleanup(kline)
        else:
            self._unpack(kline)

    # ── State ─────────────────────────────────────────────────────────────

    def _ground(self, kline: KLine) -> None:
        """Record that K has grounded ``kline``. Idempotent on nodes."""
        bucket = self._state.grounded.setdefault(kline.signature, [])
        if not any(existing.nodes == kline.nodes for existing in bucket):
            bucket.append(kline)

    def _recognised(self, signature: int, *, is_signature: bool = False) -> bool:
        """Has K seen ``signature``?

        Nodes are recognised when grounded, asked, or with an identity in
        flight. Signatures are recognised when grounded or asked (the asked set
        suppresses duplicate identity asks, not legitimate node re-traversals
        in a new canon).
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
        """Push an S2/S3 kline, its unrecognised nodes, and (if new) its signature.

        The kline is pushed first, then each unrecognised node right-to-left as
        an identity ``{node: []}``, then ``{signature: []}`` last — so the
        signature is asked before its nodes descend (LIFO).
        """
        self._state.work_list.append(kline)
        for node in reversed(kline.nodes):
            if not self._recognised(node):
                self._state.work_list.append(KLine(node, []))
        if not self._recognised(kline.signature, is_signature=True):
            self._state.work_list.append(KLine(kline.signature, []))

    def _cleanup(self, kline: KLine) -> None:
        """Ground ``kline``, then repeatedly ground any work-list kline it unblocks."""
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

        Only identities and canons — relationships ground by elevation on
        re-receipt (``_elevatable``), never here.
        """
        if is_identity(kline):
            return kline.signature in self._state.grounded
        if is_canon(kline, self._signifier):
            return all(node in self._state.grounded for node in kline.nodes)
        return False

    def _elevatable(self, kline: KLine) -> bool:
        """Should K elevate an incoming S2/S3 relationship to S1?

        True iff ``kline`` is a relationship whose nodes are all now grounded —
        K's re-derived significance outranks the sender's declared S2.
        """
        if is_identity(kline) or is_canon(kline, self._signifier):
            return False
        return all(node in self._state.grounded for node in kline.nodes)

    def _pop_identity(self, signature: int) -> None:
        """Pop the first identity work-item under ``signature`` (S4 branch)."""
        for i, entry in enumerate(self._state.work_list):
            if entry.signature == signature and is_identity(entry):
                del self._state.work_list[i]
                return

    # ── Cogitation ────────────────────────────────────────────────────────

    def _cogitate(self) -> list[KValue]:
        """Work the next workable entry (LIFO) and emit a batch.

        Batch every workable identity into the list (each emitted at S4 and
        popped). A relationship always terminates the batch and is never
        appended to identities: the first workable non-identity returns
        ``[entry]`` if no identities were collected, else the identities
        collected so far (the relationship waits for the next call).

        Returns an empty list when nothing is workable.
        """
        batch: list[KValue] = []
        for idx in range(len(self._state.work_list) - 1, -1, -1):
            entry = self._state.work_list[idx]
            if is_identity(entry):
                batch.append(self._emit_identity(idx, entry.signature))
                continue
            # First workable non-identity dispatches by significance routing
            # (scripts/dialogue-rationalisation-behaviours.md §3a):
            #   S3 structure (1:1 relationship)    → operand pairing (if workable)
            #   S2 structure (multi-node misfit)    → misfit origination
            # A single-node S3 relationship whose operand canons are not yet
            # seen is S3-structure but NOT workable — skip it (it awaits
            # elevation/cleanup), do not route it to S2.
            if self._s3_pairable(entry):
                if batch:
                    return batch  # identities collected; relationship waits
                return [self._emit_s3_pairing(entry)]
            if self._s2_eligible(entry):
                if batch:
                    return batch
                emitted = self._originate_s2(entry)
                return [emitted] if emitted is not None else batch
        return batch

    def _s3_pairable(self, entry: KLine) -> bool:
        """Is ``entry`` a workable S3 structure — a 1:1 relationship whose
        operands both have seen canons, so K can pair their operands?

        S3 path precondition (scripts/dialogue-rationalisation-behaviours.md
        §3a). The S3 path pairs the operands of two canons; it requires a
        single-node relationship ``{L:[R]}`` whose operands L and R each have a
        seen canon (in grounded memory or the work-list). A single-node
        relationship whose operand canons are not yet seen is S3-*structure* but
        not *workable* — cogitation skips it (it awaits elevation/cleanup) and
        does NOT route it to S2. Multi-node entries are never S3-pairable.
        """
        if is_identity(entry) or len(entry.nodes) != 1:
            return False
        return (
            self._find_canon_nodes(entry.signature) is not None
            and self._find_canon_nodes(entry.nodes[0]) is not None
        )

    def _s2_eligible(self, entry: KLine) -> bool:
        """Is ``entry`` an S2 structure — a multi-node misfit routed to misfit
        origination?

        S2 path precondition (scripts/dialogue-rationalisation-behaviours.md
        §3a, B1). The S2 path originates substitutions onto an entry's own
        nodes; it requires a **multi-node** misfit (a single-node relationship
        is S3-structure, routed to — or awaiting — the S3 path, never S2 even
        when not yet pairable). Identities and canons never route here.
        """
        if is_identity(entry) or len(entry.nodes) < 2:
            return False
        return not is_canon(entry, self._signifier)

    def _emit_identity(self, idx: int, signature: int) -> KValue:
        """S4 — emit IDENTITY ``{signature: []}`` at S4 and pop the entry.

        The ask is fire-and-forget: the identity is popped on emission so it
        cannot block cogitation under LIFO while its signature grounds async.
        """
        del self._state.work_list[idx]
        self._state.asked.add(signature)
        return KValue(KLine(signature, []), SIG_S4)

    def _emit_s3_pairing(self, entry: KLine) -> KValue:
        """S3 path — emit the next unresolved operand pair, or close at S1.

        ``entry`` is ``{L:[R]}`` whose operands L and R are signatures with seen
        canons. K pairs the operands of the two canons left-to-right at group
        size 1, grouping one side's residual into a single synthesised operand
        when the other reaches a single node.

        Each call emits the next unresolved pair: a 1:1 pair ``{lhs:[rhs]}`` is
        CONNOTED at S3; a grouped residual is emitted as a canonical request
        ``{make_signature(residual): residual}`` at S2 (K cannot assert a
        relationship to a signature it invented — it must first confirm it).
        When every pair is resolved, K closes by emitting ``entry`` at S1
        (COUNTERSIGNED) and removes it from the work-list.
        """
        right = entry.nodes
        assert len(right) == 1, "S3 pairing expects a single-node relationship entry"
        left_nodes = self._find_canon_nodes(entry.signature)
        right_nodes = self._find_canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            raise NotImplementedError(
                "S3 pairing: an operand canon is missing; cannot relate."
            )

        for lhs_sig, rhs_node, residual in self._relationship_plan(left_nodes, right_nodes):
            if self._pair_resolved(lhs_sig, rhs_node, residual):
                continue
            if residual:
                # A synthesised lhs K was never taught: emit a canonical request
                # at S2, and mark it asked so unpack won't re-ask it as an identity.
                synth_sig = self._signifier.make_signature(residual)
                proposal_kline = KLine(synth_sig, residual)
                significance = SIG_S2
                self._state.asked.add(synth_sig)
            else:
                proposal_kline = KLine(lhs_sig, [rhs_node])
                significance = SIG_S3
            return KValue(proposal_kline, significance)

        # All pairs resolved — close at S1.
        self._state.work_list.remove(entry)
        self._ground(entry)
        return KValue(entry, SIG_S1)

    # ── S2 path (misfit origination) — scripts/dialogue-rationalisation- ─
    # ─ behaviours.md §4. Stubbed: returns no emission until steps 2+ land. ─

    def _originate_s2(self, entry: KLine) -> KValue | None:
        """S2 path — originate a misfit proposal by accumulated shaping.

        Stubbed in step 1 of the misfit implementation: returns ``None`` (no
        emission) so an S2 entry idles harmlessly in the work-list until the
        generation mechanism (candidate admission, node-expansion, ``must_match``
        node-graft) lands in later steps. See
        scripts/dialogue-rationalisation-behaviours.md §4–§5.
        """
        return None

    # ── S3 helpers ────────────────────────────────────────────────────────

    def _find_canon_nodes(self, signature: int) -> list[int] | None:
        """The nodes of ``signature``'s canon, searching grounded memory and the work-list.

        A canon's operands are readable as soon as K has seen the canon, even
        before it fully grounds.
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

        Group-size-1 convention: pair left-to-right while both sides have more
        than one node remaining; when one side reaches a single node, group the
        other side's entire residual into one synthesised operand. ``residual``
        is empty for a 1:1 pair.
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

        A 1:1 pair is resolved when ``{lhs_sig:[rhs_node]}`` is grounded (the
        trainer ratified it). A grouped pair is resolved when its synthesised
        canon ``{make_signature(residual): residual}`` is grounded.
        """
        if residual:
            synth_sig = self._signifier.make_signature(residual)
            return any(
                is_canon(kl, self._signifier)
                for kl in self._state.grounded.get(synth_sig, [])
            )
        # 1:1 pair: resolved iff {lhs_sig:[rhs_node]} is grounded (ratified).
        return any(
            list(kline.nodes) == [rhs_node]
            for kline in self._state.grounded.get(lhs_sig, [])
        )
