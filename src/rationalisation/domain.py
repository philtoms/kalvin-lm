"""The domain — the insertion-ordered set of held klines, the model's
memory (docs/rationalisation.md §1 Definition 3). It only grows."""

from __future__ import annotations

from kalvin.kline import KLine
from rationalisation.algebra import can


class Domain:
    """D ⊆_fin Kline — insertion-ordered, idempotent add, grows only."""

    def __init__(self, klines: list[KLine] | tuple = ()) -> None:
        self._klines: set[KLine] = set()
        self._order: list[KLine] = []  # insertion order — first-stored is observable
        for k in klines:
            self.add(k)

    def add(self, k: KLine) -> bool:
        """R1 store — idempotent; returns True when new."""
        if k in self._klines:
            return False
        self._klines.add(k)
        self._order.append(k)
        return True

    def is_canonical(self, k: KLine) -> bool:
        return k.signature == can(k.nodes)

    def __contains__(self, k: KLine) -> bool:
        return k in self._klines

    def __iter__(self):
        return iter(self._order)
