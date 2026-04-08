"""Event bus for Kalvin pub/sub pattern."""

from __future__ import annotations

import threading
from typing import Callable

from kalvin.kline import KLine


class RationaliseEvent:
    """Event emitted during rationalisation processing."""

    __slots__ = ("kind", "kline", "query")

    def __init__(self, kind: str, kline: KLine, query: KLine):
        self.kind = kind
        self.kline = kline
        self.query = query

    def __repr__(self) -> str:
        return f"RationaliseEvent({self.kind!r}, {self.kline!r})"


class EventBus:
    """Single-channel pub/sub for rationalisation events.

    Thread-safe: publish() may be called from any thread (e.g. the cogitate
    background thread). Callbacks are invoked synchronously under a lock so
    subscribers always see a consistent event order.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[RationaliseEvent], None]] = []

    def subscribe(self, callback: Callable[[RationaliseEvent], None]) -> None:
        """Register a callback that receives ALL events."""
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, event: RationaliseEvent) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            subscribers = list(self._subscribers)
        for cb in subscribers:
            cb(event)
