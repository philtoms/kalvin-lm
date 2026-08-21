r"""The rationalising engine.

A :class:`Engine` derives one turn from ``(state, incoming)`` and returns
``(batch, observations)`` — dialogue emissions and K's internal S1 groundings
this turn. The engine is stateless about its own emissions; dedup lives in the
actor.

The engine is pure mechanism: it holds a :class:`EngineState` and a
:class:`MisfitStrategy`, both fully constructed by the caller. The factories
that assemble them (signifier, state, strategy, engine) live in
:mod:`dialogue.harness`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

from dialogue.engine_state import EngineState
from kalvin.kline import (
    KLine,
    is_canon,
    is_identity,
    is_misfit,
    is_unknown,
    sig_level,
    using_resolver,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    SIG_S1,
    SIG_S3,
    SIG_S4,
    SIG8_MAX,
    SIG_MASK,
    BandLayout,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Engine", "EngineState", "MisfitStrategy"]

# Default band layout, used to classify a query's stamped significance byte
# into a structural level for routing.
_LAYOUT = BandLayout()


@runtime_checkable
class MisfitStrategy(Protocol):
    """Propose for one pending misfit ``entry`` against the ratified store.

    A lazy generator yielding S2 proposals (S1 when the proposal is already
    grounded), discovered breadth-first and halting at a proposal budget —
    nearer proposals first, so significance order is discovery order. The
    strategy shares the engine's :class:`EngineState` (set at construction).
    """

    def propose(
        self,
        entry: KLine,
    ) -> Iterator[KValue]:
        ...


class Engine:
    #: Shape-route containment answering (exploratory, off by default).
    SHAPE_ANSWERS = False
    """Derives one turn from ``incoming``.

    Holds the :class:`EngineState` it mutates in place and the
    :class:`MisfitStrategy` it consults for the S2 arm — both supplied fully
    constructed. The signifier is read off the state.
    """

    def __init__(
        self,
        state: EngineState,
        misfit: MisfitStrategy,
    ) -> None:
        self._state: EngineState = state
        self._misfit: MisfitStrategy = misfit

    @property
    def state(self) -> EngineState:
        """The engine's mutable memory, mutated in place each turn."""
        return self._state

    @property
    def signifier(self) -> KSignifier:
        """The state's signifier (the single source of truth)."""
        return self._state.signifier

    def rationalise(
        self, incoming: Sequence[KValue]
    ) -> tuple[list[KValue], list[KValue]]:
        """Route every incoming query, then cogitate. Returns ``(batch, observations)``."""
        self._state._dbg_step += 1
        self.observations: list[KValue] = []

        resolver = self._state.find
        with using_resolver(resolver):
            batch: list[KValue] = []
            for query in incoming:
                batch.extend(self.route(query))
            batch.extend(self.cogitate())
            return batch, self.observations


    # ── Routing ──────────────────────────────────────────────────────

    def route(self, query: KValue) -> list[KValue]:
        """Apply one incoming query; return any immediate emissions.

        Dispatch is on the query's **structural** significance:

        - **S1/S4 (fast route)** — an S1 (identity or canon) match grounds the
          kline (and cascades) and **answers from LTM**: every grounded kline
          under the query's signature, other than the query itself, is said —
          after the hard work of cogitation, a question K already holds the
          answer to is answered directly. An S4 rejection (empty ask or
          refused proposal) drops the matching kline from attention.
        - **S2/S3 (slow route)** — append to STM, then unpack an S2
          misfit's unrecognised nodes and signature as identity asks.
        """
        kline = query.kline
        structural_sig = sig_level(kline, self._state.signifier)
        query_sig = _LAYOUT.classify(query.significance)

        if query_sig == "S4":
            self._state.refuse(query.kline)
            self._state.remove_stm(query.kline)
            return []

        # A stamped-S1 query is a ratification: ground on receipt, before
        # any answering — the ratified kline is the answer just granted.
        if query_sig == "S1" or (
            structural_sig == query_sig and structural_sig == "S1"
        ):
            if self._fast_route(query):
                return []

        # Fast path: a question (unknown or misfit) whose signature already
        # holds grounded knowledge — say it, whatever the query's band. After
        # the hard work of cogitation, a question K holds the answer to is
        # answered directly from LTM. Statements (identities, countersigns,
        # ratifications) do not trigger answering.
        is_question = is_unknown(kline) or is_misfit(kline, self._state.signifier)
        if is_question:
            answers = self._answers_from_ltm(query)
            if answers:
                return answers

        # Shape route (GATED OFF — exploratory): a question whose signature
        # holds nothing yet — an unknown, or a self-signed query K has never
        # grounded (raw words that just self-signed into a new signature).
        # Resolved nodes resolve through grounded canons; a grounded kline
        # containing them is the answer, graded by coverage. Answers can
        # preempt cogitation's sharper proposals — hence the gate.
        if self.SHAPE_ANSWERS and (
            is_question
            or (
                is_canon(kline, self._state.signifier)
                and not self._state.ltm.get(kline.signature)
            )
        ):
            answers = self._answers_by_containment(query)
            if answers:
                return answers


        self._slow_route(query)
        return []

    def _answers_from_ltm(self, query: KValue) -> list[KValue]:
        """Grounded knowledge under the query's signature, said aloud.

        Identities and canons are excluded — identities are asks or facts,
        canons are ground truth; neither is an answer K earned.
        """
        signifier = self._state.signifier
        return [
            KValue(k, SIG_S1)
            for k in self._state.ltm.get(query.kline.signature, [])
            if k.nodes != query.kline.nodes
            and not is_identity(k)
            and not is_canon(k, signifier)
        ]

    def _resolved_nodes(self, nodes: list[int]) -> list[int]:
        """Resolve node groups through grounded canon resolutions.

        A grounded canon (e.g. DH:[did,have]) whose signature also holds a
        grounded resolution (DH:[had]) names its group: the group's nodes
        collapse to the resolution node — slot accounting, no semantics.
        """
        resolved = list(nodes)
        for bucket in self._state.ltm.values():
            # canon resolutions: same signature, one node, not the canon itself
            for canon in bucket:
                group = set(canon.nodes)
                if len(group) < 2 or not group <= set(resolved):
                    continue
                for other in bucket:
                    if other is canon:
                        continue
                    if len(other.nodes) == 1 and other.nodes[0] not in group:
                        node = other.nodes[0]
                        resolved = [n for n in resolved if n not in group] + [node]
                        break
                else:
                    continue
                break
        return resolved

    def _answers_by_containment(self, query: KValue) -> list[KValue]:
        """Grounded klines containing the query's resolved nodes, said aloud.

        Full containment is a ratify-grade answer; the grade scales with
        coverage (contained / containing). Identities are not answers; a
        canon here is the sentence itself — exactly what K should say.
        """
        signifier = self._state.signifier
        resolved = self._resolved_nodes(list(query.kline.nodes))
        if not resolved:
            return []
        resolved_set = set(resolved)
        answers: list[KValue] = []
        for bucket in self._state.ltm.values():
            for k in bucket:
                if is_identity(k):
                    continue
                if resolved_set <= set(k.nodes):
                    coverage = len(resolved_set) / len(k.nodes)
                    if coverage <= 0.5:
                        # A weak overlap is not an answer — let the query
                        # take the slow route instead.
                        continue
                    answers.append(
                        KValue(k, int(SIG8_MAX * coverage) & SIG_MASK or SIG_S1)
                    )
        return answers

    def _ground(self, kline: KLine) -> None:
        """Ground ``kline`` at S1, then cascade any node-resolution it unblocks.

        A grounding may make other STM entries groundable (an identity
        whose signature just landed, a canon whose nodes are now all seen, a
        relationship whose reciprocal just grounded). Cascade until fixed point.
        """
        if self._state.ground(kline):
            self.observations.append(KValue(kline, SIG_S1))
        sweep = True
        while sweep:
            sweep = False
            for entry in self._state.stm:
                if self._state._is_groundable(entry) and self._state._is_denoted(entry):
                    if self._state.ground(entry):
                        self.observations.append(KValue(entry, SIG_S1))
                        sweep = True
                        break

    def _fast_route(self, query: KValue) -> bool:
        if not self._state._is_groundable(query.kline):
            return False
        self._ground(query.kline)
        return True

    def _slow_route(self, query: KValue) -> None:
        kline = query.kline
        self._state.add_stm(kline)
        for node in kline.nodes:
            if not self._state.is_seen(node):
                self._state.add_stm(KLine(node, [], kline.dbg))
        if not self._state.is_seen(kline.signature):
            self._state.add_stm(KLine(kline.signature, [], kline.dbg))

    # ── Cogitation ───────────────────────────────────────────────────

    def cogitate(self) -> list[KValue]:
        """One oldest-first pass over STM: ask, countersign, propose, or ground.

        Per entry, in priority order: an identity becomes an S4 ask; a
        countersignable entry takes the S3 path and eventually grounds; a misfit
        takes the S2 path. a structurally-S1 entry is promoted (grounded).
        Entries that match no path persist for a later turn.
        """
        batch: list[KValue] = []

        idx = 0
        count = len(self._state.stm)
        while idx < len(self._state.stm):
            # Re-check the index each iteration: the _promote cascade (via the
            # S2 strategy's ground callback, or the countersign/groundable
            # arms) can remove arbitrary STM entries, shrinking the list
            # below the index this loop intends to visit.
            if idx >= len(self._state.stm):
                continue
            kline = self._state.stm[idx]

            if is_unknown(kline):
                self._state.remove_stm_at(idx)
                batch.append(KValue(KLine(kline.signature, []), SIG_S4))
                continue
            else:
                if self._state._is_groundable(kline) and self._state._is_denoted(kline):
                    self._ground(kline)

                # if self._state.is_countersignable(kline):
                #     pairings = self._countersignature_proposals(kline)
                #     if pairings:
                #         proposals.extend(pairings)
                #     else:
                #         # All pairings resolved: the countersignature is complete.
                #         self._state.remove_stm_at(idx)
                #         self._ground(kline)

                if is_misfit(kline, self._state.signifier):
                    proposals = list(self._misfit.propose(kline))
                    if proposals:
                        # Framing does not consume the misfit: it stays in
                        # STM until its proposal is ratified (grounded) or
                        # every shape is refused.
                        batch.extend(proposals)

                if self._state.is_grounded(kline):
                    self._state.remove_stm_at(idx)
                    continue

            idx += 1

        if count != len(self._state.stm):
            batch.extend(self.cogitate())

        return batch


    # ── S3 path: countersignature ────────────────────────────────────

    def _countersignature_proposals(self, entry: KLine) -> list[KValue]:
        """Every unresolved operand pairing for ``entry`` as CONNOTES at S3.

        Pair the two canons' operands left-to-right at group size 1; when one
        side reaches a single node, synthesise the other's residual into one
        operand. Returns ``[]`` once every pairing is grounded — the signal
        that the countersignature is complete and the entry should ground itself.
        """
        right = entry.nodes
        assert len(right) == 1, "S3 pairings expect a single-node relationship entry"
        left_nodes = self._state.canon_nodes(entry.signature)
        right_nodes = self._state.canon_nodes(right[0])
        if left_nodes is None or right_nodes is None:
            raise NotImplementedError("S3 pairings: an operand canon is missing")

        batch: list[KValue] = []
        for lhs_sig, rhs_node, residual in self._operand_pairings(left_nodes, right_nodes):
            if self._pairing_resolved(lhs_sig, rhs_node, residual):
                continue
            head_sig = self._state.signifier.signature_of(residual) if residual else lhs_sig
            batch.append(KValue(KLine(head_sig, [rhs_node]), SIG_S3))
        return batch

    def _operand_pairings(
        self, left_nodes: list[int], right_nodes: list[int]
    ) -> list[tuple[int, int, list[int]]]:
        """Pair two canons' operands into ``(lhs_sig, rhs_node, residual)`` tuples.

        Pair left-to-right while both sides have more than one node remaining;
        when one side reaches a single node, group the other's entire residual
        into one synthesised operand (returned as ``residual``).
        """
        signifier = self._state.signifier
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
                plan.append((left_nodes[i], signifier.signature_of(residual), residual))
                break
            elif right_rem == 1:
                residual = list(left_nodes[i:])
                plan.append((signifier.signature_of(residual), right_nodes[j], residual))
                break
            else:
                plan.append((left_nodes[i], right_nodes[j], []))
                i += 1
                j += 1
        return plan

    def _pairing_resolved(self, lhs_sig: int, rhs_node: int, residual: list[int]) -> bool:
        """Is this pairing's CONNOTES proposal ``{head_sig:[rhs_node]}`` grounded?

        For a grouped residual, ``head_sig`` is synthesised from the residual;
        for a 1:1 pair it is ``lhs_sig``.
        """
        head_sig = self._state.signifier.signature_of(residual) if residual else lhs_sig
        return any(
            list(kline.nodes) == [rhs_node]
            for kline in self._state.ltm.get(head_sig, [])
        )
