"""Event bus for Agent pub/sub pattern."""

from __future__ import annotations

import threading
from typing import Callable

from kalvin.kline import KLine


class RationaliseEvent:
    """Event emitted during rationalisation processing."""

    __slots__ = ("kind", "query", "proposal", "significance")

    def __init__(self, kind: str, query: KLine, proposal: KLine, significance: int):
        self.kind = kind
        self.query = query
        self.proposal = proposal
        self.significance = significance

    def __repr__(self) -> str:
        return f"RationaliseEvent({self.kind!r}, sig={self.significance:#x})"


class EventBus:
    """Single-channel pub/sub for rationalisation events.

    Thread-safe: publish() may be called from any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[RationaliseEvent], None]] = []

    def subscribe(self, callback: Callable[[RationaliseEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, event: RationaliseEvent) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for cb in subscribers:
            cb(event)

    def on_event(self, event: RationaliseEvent) -> None:
        """Adapter protocol: delegates to publish()."""
        self.publish(event)
