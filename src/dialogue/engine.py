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
    is_canon,
    is_misfit,
    sig_level,
    using_resolver,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    SIG_S1,
    BandLayout,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Engine", "EngineState"]

# Default band layout, used to classify a query's stamped significance byte
# into a structural level for routing.
_LAYOUT = BandLayout()


class Engine:
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
                if not self._fast_route(query):
                    self._slow_route(query)
            batch.extend(self.cogitate())
            return batch, self.observations


    # ── Routing ──────────────────────────────────────────────────────

    def _fast_route(self, query: KValue) -> bool:
        kline = query.kline
        structural_sig = sig_level(kline, self._state.signifier)
        query_sig = _LAYOUT.classify(query.significance)

        if query_sig == "S4":
            self._state.refuse(kline)
            self._state.remove_stm(kline)
            return True

        # A stamped-S1 query is a ratification: ground on receipt, before
        # any answering — the ratified kline is the answer just granted.
        if query_sig == "S1" or (
            structural_sig == query_sig and structural_sig == "S1"
        ):
            if self._state._is_groundable(kline):
                self._ground(kline)
                return True

        return False

    def _slow_route(self, query: KValue) -> None:
        """Attend to the query: append it and its unknown parts to STM.

        """
        kline = query.kline
        self._state.add_stm(kline)

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
                break

            kline = self._state.stm[idx]
            if (
                is_canon(kline, self._state.signifier)
                and self._state._is_groundable(kline)
                and self._state._is_denoted(kline)
            ):
                # A canon is the script's own ground truth — it grounds
                # even under an asked signature (the ask under the
                # signature is answered by the canon itself).
                self._ground(kline)

            if self.signifier.is_ask(kline.signature) or is_misfit(kline, self._state.signifier):
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
