r"""The one-shot kline rationaliser.

The :class:`Rationaliser` feeds memory from ``(state, incoming)``: a query
grounds directly on the fast path (S1 ratification, S4 refusal) or queues on
the work list. It never cogitates — the work-list pass lives in
:mod:`kalvin.cogitator`; the harness runs rationalise, then cogitate.

The rationaliser is pure mechanism: it holds an :class:`Memory` and
mutates it in place. The factories that
assemble signifier, state, and rationaliser live in :mod:`dev.dialogue.harness`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kalvin.kline import is_ask, sig_level, using_resolver
from kalvin.kvalue import KValue
from kalvin.memory import Memory
from kalvin.significance import BandLayout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier

__all__ = ["Rationaliser", "Memory"]

# Default band layout, used to classify a query's stamped significance byte
# into a structural level for routing.
_LAYOUT = BandLayout()


class Rationaliser:
    """Feeds memory from ``incoming``: fast path grounds, the rest queue.

    Holds the :class:`Memory` it mutates in place. The signifier is
    read off the state.
    """

    def __init__(self, state: Memory) -> None:
        self._state: Memory = state

    @property
    def state(self) -> Memory:
        """The rationaliser's mutable memory, mutated in place each turn."""
        return self._state

    @property
    def signifier(self) -> KSignifier:
        """The state's signifier (the single source of truth)."""
        return self._state.signifier

    def rationalise(self, incoming: Sequence[KValue]) -> None:
        """Feed memory: ground every incoming query on the fast path, queue the rest."""
        self._state._dbg_step += 1

        resolver = self._state.find
        with using_resolver(resolver):
            for query in incoming:
                if not self._fast_route(query):
                    self._slow_route(query)

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
        # The stamp, not structure, is the licence: the reached-goal
        # answer is a misfit in the question's head, never groundable
        # structurally. An ask never grounds however stamped (a question
        # is not an answer), nor an empty unknown.
        if query_sig == "S1" or (
            structural_sig == query_sig and structural_sig == "S1"
        ):
            if not is_ask(kline.signature) and kline.nodes:
                self._state.ground_cascade(kline)
                return True

        return False

    def _slow_route(self, query: KValue) -> None:
        """Attend to the query: append it and its unknown parts to the work list.

        """
        kline = query.kline
        self._state.add_work(kline)
