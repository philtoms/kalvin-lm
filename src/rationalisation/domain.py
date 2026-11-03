"""The domain — the insertion-ordered set of held klines, the model's
memory (docs/rationalisation.md §1 Definition 3). It only grows; the
first-stored canon per signature is its witness index."""

from __future__ import annotations

from kalvin.kline import KNode
from rationalisation.algebra import Kline, can


class Domain:
    """D ⊆_fin Kline with the first-stored witness index and lookups."""

    def __init__(self, klines: list[Kline] | tuple = ()) -> None:
        self._klines: set[Kline] = set()
        self._order: list[Kline] = []  # insertion order — first-stored is observable
        self._canonical: dict[KNode, frozenset[KNode]] = {}
        for k in klines:
            self.add(k)

    def add(self, k: Kline) -> bool:
        """R1 store — idempotent; returns True when new."""
        if k in self._klines:
            return False
        self._klines.add(k)
        self._order.append(k)
        if self.is_canonical(k):
            self._canonical.setdefault(k.sig, k.adj)  # first-stored wins
        return True

    def is_canonical(self, k: Kline) -> bool:
        return k.sig == can(k.adj)

    def __contains__(self, k: Kline) -> bool:
        return k in self._klines

    def __iter__(self):
        return iter(self._order)

    def witness(self, s: KNode) -> frozenset[KNode]:
        """The adjacency of the first-stored canon holding s; ∅ when none."""
        return self._canonical.get(s, frozenset())
