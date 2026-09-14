r"""The rationalising engine.

A :class:`Engine` derives one turn from ``(state, incoming)`` and returns
the batch — the dialogue emissions. The engine is stateless about its own
emissions; dedup lives in the
actor.

The engine is pure mechanism: it holds an :class:`EngineState`, constructing
the S2 strategy (:class:`ExpandFit`) over it itself. The factories that
assemble signifier, state, and engine live in :mod:`dialogue.harness`.
"""

from __future__ import annotations

from collections.abc import Sequence, Iterator
from typing import TYPE_CHECKING

from dialogue.engine_state import EngineState
from dialogue.derivation import Derivation, DerivationResult
from kalvin.kline import (
    KLine,
    is_canon,
    is_terminal,
    is_misfit,
    sig_level,
    using_resolver,
)
from kalvin.kvalue import KValue
from kalvin.significance import (
    BandLayout,
    gamma_to_byte,
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
        self._misfit = Derivation(state)

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
    ) -> list[KValue]:
        """Route every incoming query, then cogitate. Returns the dialogue batch."""
        self._state._dbg_step += 1

        resolver = self._state.find
        with using_resolver(resolver):
            batch: list[KValue] = []
            for query in incoming:
                if not self._fast_route(query):
                    self._slow_route(query)
            batch.extend(self.cogitate())
            return batch


    # ── Routing ──────────────────────────────────────────────────────

    def _fast_route(self, query: KValue) -> bool:
        kline = query.kline
        structural_sig = sig_level(kline, self._state.signifier)
        query_sig = _LAYOUT.classify(query.significance)

        if query_sig == "S4":
            self._state.refuse(kline)
            self._state.remove_work(kline)
            return True

        # A stamped-S1 query is a ratification: ground on receipt, before
        # any answering — the ratified kline is the answer just granted.
        if query_sig == "S1" or (
            structural_sig == query_sig and structural_sig == "S1"
        ):
            if self._state.is_groundable(kline):
                self._ground(kline)
                return True

        return False

    def _slow_route(self, query: KValue) -> None:
        """Attend to the query: append it and its unknown parts to the work list.

        """
        kline = query.kline
        self._state.add_work(kline)

    # ── Cogitation ───────────────────────────────────────────────────

    def cogitate(self) -> list[KValue]:
        """One oldest-first pass over the work list: ask, propose, or ground.

        Per entry, in priority order: a groundable entry grounds; a misfit
        entry draws proposals from the strategy; a grounded entry leaves
        attention. Entries that match no path persist for a later turn.
        The pass repeats until stable — grounding can unblock further
        entries.
        """
        batch: list[KValue] = []

        idx = 0
        count = len(self._state.work_list)
        while idx < len(self._state.work_list):
            # Re-check the index each iteration: the _promote cascade (via the
            # S2 strategy's ground callback, or the countersign/groundable
            # arms) can remove arbitrary work-list entries, shrinking the list
            # below the index this loop intends to visit.
            if idx >= len(self._state.work_list):
                break

            kline = self._state.work_list[idx]
            if self._state.is_groundable(kline):
                self._ground(kline)

            for candidate in self._select(kline):
                result = self._misfit.derive(kline, candidate)
                if result.ending != "done" or len(result.trace) < 2:
                    # Stuck and abandoned ask; done at entry is the ground
                    # path's, not a proposal.
                    continue
                proposal = KLine(kline.signature, result.trace[-1])
                if not self._state.is_refused(proposal):
                    batch.append(KValue(proposal, gamma_to_byte(result.gamma)))

            if self._state.is_grounded(kline):
                self._state.remove_work_at(idx)
                continue

            idx += 1

        if count != len(self._state.work_list):
            batch.extend(self.cogitate())

        return batch

    def _ground(self, kline: KLine) -> None:
        """ground ``kline`` at S1, then cascade any node-resolution it unblocks.

        A grounding may make other work-list entries groundable (an identity
        whose signature just landed, a canon whose nodes are now all seen, a
        relationship whose reciprocal just grounded). Cascade until fixed point.
        """
        self._state.ground(kline)
        sweep = True
        while sweep:
            sweep = False
            for entry in self._state.work_list:
                if self._state.is_groundable(entry):
                    if self._state.ground(entry):
                        sweep = True
                        break

    def _select(self, entry: KLine) -> Iterator[KLine]:
        """Held non-terminal klines whose signature occurs as a node of
        ``entry`` — Def 16: selection is occurrence, not content overlap;
        terminals (unknowns, identities) offer no second side."""
        for kline in self._state.where(
            lambda k: k.signature != entry.signature
            and not is_terminal(k)
            and k.signature in entry.nodes
        ):
            yield kline
