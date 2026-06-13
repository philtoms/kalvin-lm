"""Event bus for Agent pub/sub pattern.

EventBus is a test-friendly adapter that satisfies the KAgentAdapter protocol
(defined in ``kalvin.agent``).  It provides ``on_event(event)`` for the
adapter contract plus ``subscribe(callback)`` / ``publish(event)`` for
pub/sub use in tests and the TUI dev mode.

For production wiring, use ``KAgentAdapter`` from ``harness.adapter`` instead.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from kalvin.kline import KLine


class RationaliseEvent:
    """Event emitted during rationalisation processing."""

    __slots__ = ("kind", "query", "proposal", "significance", "candidate")

    def __init__(
        self,
        kind: str,
        query: KLine,
        proposal: KLine,
        significance: int,
        candidate: KLine | None = None,
    ):
        self.kind = kind
        self.query = query
        self.proposal = proposal
        self.significance = significance
        # Original candidate kline for expansion events.
        # For expansion proposals (S2/S3), this is the misfit candidate
        # that triggered the expansion. None for fast-path events (S1/S4).
        self.candidate = candidate

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
