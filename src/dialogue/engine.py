r"""The rationalising engine.

A :class:`Engine` derives one turn from ``(state, incoming)`` and returns
``(batch, observations)`` — dialogue emissions and K's internal S1 groundings
this turn. The engine is stateless about its own emissions; dedup lives in the
actor.

The engine is pure mechanism: it holds an :class:`EngineState`, constructing
the S2 strategy (:class:`ExpandFit`) over it itself. The factories that
assemble signifier, state, and engine live in :mod:`dialogue.harness`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from dialogue.expand_fit import ExpandFit
from dialogue.reentry import Reentry
from kalvin.kline import (
    KLine,
    KNode,
    is_canon,
    is_identity,
    is_misfit,
    is_unknown,
    sig_level,
    using_resolver,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    SIG8_MAX,
    SIG_MASK,
    SIG_S1,
    SIG_S4,
    BandLayout,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Engine", "EngineState"]

# Default band layout, used to classify a query's stamped significance byte
# into a structural level for routing.
_LAYOUT = BandLayout()


class Engine:
    #: Shape-route containment answering (exploratory, off by default).
    SHAPE_ANSWERS = False
    #: User-significance teaching (S2/S3 supervisor stamps as training
    #: material; exploratory, off by default).
    TRAINING = False
    """Derives one turn from ``incoming``.

    Holds the :class:`EngineState` it mutates in place and constructs the
    :class:`ExpandFit` S2 strategy over it. The signifier is read off the
    state.
    """

    def __init__(self, state: EngineState) -> None:
        self._state: EngineState = state
        # self._misfit = PivotFill(state)
        self._misfit = ExpandFit(state)
        # self._misfit = Reentry(state)

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
                batch.extend(self.route(query) or [])
            batch.extend(self.cogitate())
            return batch, self.observations


    # ── Routing ──────────────────────────────────────────────────────

    def route(self, query: KValue) -> list[KValue] | None:
        """Apply one incoming query; return any immediate emissions.

        Three dispatch axes, consulted in order:

        1. **Stamped significance** (the sender's band on the KValue):
           an S4 stamp refuses the kline (drops it from attention and
           spends any open ask under its signature); an S1 stamp is a
           ratification — the kline grounds on receipt and the turn emits
           nothing further for it.
        2. **Question vs statement** (structural significance): an unknown
           or misfit whose signature already holds grounded knowledge is
           answered from LTM directly (the fast route).
        3. Otherwise the query takes the slow route: appended to STM, its
           unrecognised nodes and signature added as identity asks.
        """
        kline = query.kline
        structural_sig = sig_level(kline, self._state.signifier)
        query_sig = _LAYOUT.classify(query.significance)

        if self.TRAINING and query_sig in ("S2", "S3") and (
            kline.signature in self._state.asked
        ):
            # A graded response to K's own proposal under this signature:
            # teaching material. S2 patterns the answer shape; S3 pivots it.
            # The ask context is the canon K was attending to when asked.
            self._state.teaching.record(
                query_sig,
                kline,
                self._ask_context(kline.signature),
            )

        if query_sig == "S4":
            self._state.refuse(kline)
            self._state.remove_stm(kline)
            if kline.nodes:
                # Refusing a proposal spends the ask under its signature:
                # the signature is now seen, not asked.
                self._state.asked.discard(kline.signature)
            return None

        # A stamped-S1 query is a ratification: ground on receipt, before
        # any answering — the ratified kline is the answer just granted.
        if query_sig == "S1" or (
            structural_sig == query_sig and structural_sig == "S1"
        ):
            if self._state._is_groundable(kline):
                self._ground(kline)
                return None

        result = self._fast_route(query)
        if result is None:
            self._slow_route(query)
            return None
        return result

    def _taught_pattern(self, kline: KLine) -> KValue | None:
        """A taught answer for the ask ``kline`` poses, not already refused."""
        taught = self._state.teaching.pattern_for(kline, self._state.signifier)
        if taught is None or self._state.is_refused(taught.kline):
            return None
        return taught

    def _ask_context(self, signature: KNode) -> KLine | None:
        """The canon ``signature`` asked about, if one is in attention."""
        signifier = self._state.signifier
        for entry in self._state.stm:
            if entry.signature == signature and is_canon(entry, signifier):
                return entry
        return None

    def _fast_route(self, query: KValue) -> list[KValue] | None:
        """Answer a question directly from LTM.

        A question (unknown or misfit, by structure) whose signature holds
        grounded klines gets them said at S1 — the stamped band is not
        consulted. Statements (identities, canons, ratifications) never
        take this route.

        The gated shape route (``SHAPE_ANSWERS``, off by default) instead
        answers questions whose signature holds nothing yet: grounded
        klines containing the query's resolved nodes, graded by coverage.
        It can preempt cogitation's sharper proposals — hence the gate.
        """
        kline = query.kline
        is_question = is_unknown(kline) or is_misfit(kline, self._state.signifier)
        if is_question:
            answers = self._answers_from_ltm(query)
            if answers:
                return answers

        if self.SHAPE_ANSWERS and (
            is_question
            or (
                is_canon(kline, self._state.signifier)
                and not self._state.ltm.get(kline.signature)
            )
        ):
            return self._answers_by_containment(query)
        return None

    def _slow_route(self, query: KValue) -> None:
        """Attend to the query: append it and its unknown parts to STM.

        An S2-stamped feed is an ask — the stamp reads as a question about
        this signature, not a fact to ground.
        """
        kline = query.kline
        if _LAYOUT.classify(query.significance) == "S2":
            self._state.asked.add(kline.signature)
        self._state.add_stm(kline)
        for node in kline.nodes:
            if not self._state.is_seen(node):
                self._state.add_stm(KLine(node, [], kline.dbg))
        if not self._state.is_seen(kline.signature):
            self._state.add_stm(KLine(kline.signature, [], kline.dbg))

    # ── Cogitation ───────────────────────────────────────────────────

    def cogitate(self) -> list[KValue]:
        """One oldest-first pass over STM: ask, propose, or ground.

        Per entry, in priority order: an unknown becomes an S4 ask; an
        unasked, denoted, groundable entry grounds; a misfit or asked
        entry draws proposals from the strategy; a grounded entry leaves
        attention. Entries that match no path persist for a later turn.
        The pass repeats until stable — grounding can unblock further
        entries.
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
                asked = kline.signature in self._state.asked
                if (
                    (not asked or is_canon(kline, self._state.signifier))
                    and self._state._is_groundable(kline)
                    and self._state._is_denoted(kline)
                ):
                    # A canon is the script's own ground truth — it grounds
                    # even under an asked signature (the ask under the
                    # signature is answered by the canon itself).
                    self._ground(kline)

                if is_misfit(kline, self._state.signifier) or asked:
                    taught = self._taught_pattern(kline) if self.TRAINING else None
                    if taught is not None:
                        # Learned behaviour: a supervisor-taught answer for
                        # this ask shape preempts structural proposals. Like
                        # a structural proposal, it is emitted and the misfit
                        # stays in STM until ratified or refused.
                        batch.append(taught)
                    else:
                        proposals = list(self._misfit.propose(kline))
                        if proposals:
                            batch.extend(proposals)

                if self._state.is_grounded(kline):
                    self._state.remove_stm_at(idx)
                    continue

            idx += 1

        if count != len(self._state.stm):
            batch.extend(self.cogitate())

        return batch

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

    def _resolved_nodes(self, nodes: list[KNode]) -> list[KNode]:
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
