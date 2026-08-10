"""The S2 (misfit) cogitation strategies.

Each strategy proposes for a pending misfit kline by recombining what the
engine has already grounded. Two implementations live alongside this module
(``similar_fit`` / ``expand_fit``); the engine selects one at construction
via the ``strategy`` knob. Shared grounding-store helpers and the read-only
``Model`` adapter that ``kalvin.expand`` and ``dialogue.proposals`` expect
live here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kalvin.kline import KLine, is_canon
from kalvin.kvalue import KValue

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dialogue.engine import EngineState
    from kalvin.abstract import KSignifier

__all__ = [
    "MisfitStrategy",
    "GroundedModel",
    "similar_fit_candidates",
    "is_grounded",
    "grounded_nodes",
]


@runtime_checkable
class MisfitStrategy(Protocol):
    """Propose for one pending misfit ``entry`` against the grounded store.

    Returns S2 proposals for the actor to emit, and may ground an entry
    directly via ``ground`` when a candidate fully accounts for it (the
    expand strategy's S1 case). Strategies are stateless: all memory lives
    on ``state``.
    """

    def propose(
        self,
        state: EngineState,
        signifier: KSignifier,
        entry: KLine,
        ground: Callable[[KLine], None],
    ) -> list[KValue]:
        ...


class GroundedModel:
    """A read-only ``Model``-shaped view over an :class:`EngineState`'s grounded store.

    Satisfies the subset of the :class:`kalvin.model.Model` contract that
    ``kalvin.expand.expand`` and ``dialogue.proposals.propose_expansions``
    call: ``find``, ``find_all``, ``grounded``, and ``where``. Lets the expand
    strategy traverse the engine's grounded model without importing the full
    Model (the lean engine keeps none).
    """

    def __init__(self, state: EngineState) -> None:
        self._grounded = state.grounded

    def find(self, signature: int) -> KLine | None:
        bucket = self._grounded.get(signature)
        return bucket[-1] if bucket else None

    def find_all(self, signature: int) -> list[KLine]:
        return list(self._grounded.get(signature, []))

    def grounded(self, kline: KLine) -> bool:
        return any(
            existing.nodes == kline.nodes
            for existing in self._grounded.get(kline.signature, [])
        )

    def where(self, predicate: Callable[[KLine], bool] | int) -> list[KLine]:
        """All grounded klines matching ``predicate`` (callable, unlike Model)."""
        return [
            kline
            for bucket in self._grounded.values()
            for kline in bucket
            if predicate(kline)
        ]


def similar_fit_candidates(
    state: EngineState, signifier: KSignifier, entry: KLine
) -> list[KLine]:
    """Grounded klines sharing at least one but not all node values with ``entry``, 
    excluding the entry's own canon (its resolution, not a recombination ingredient)."""
    entry_nodes = set(entry.nodes)
    candidates: list[KLine] = []
    for bucket in state.grounded.values():
        for kline in bucket:
            if kline is entry or not kline.nodes or entry.nodes == kline.nodes:
                continue
            if kline.signature == entry.signature and is_canon(kline, signifier):
                continue
            kline_nodes = set(kline.nodes)
            if entry_nodes & kline_nodes and len(kline_nodes.difference(entry_nodes)):
                candidates.append(kline)
    return candidates


def is_grounded(state: EngineState, signature: int, nodes: list[int]) -> bool:
    """Is an isomorphic kline (same signature and nodes) in grounded memory?"""
    return any(
        existing.nodes == nodes
        for existing in state.grounded.get(signature, [])
    )


def grounded_nodes(state: EngineState, signature: int) -> list[int] | None:
    """The nodes of any grounded kline under ``signature`` with non-empty nodes."""
    for kline in state.grounded.get(signature, []):
        if kline.nodes:
            return list(kline.nodes)
    return None
