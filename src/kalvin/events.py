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

from kalvin.kvalue import KValue


class RationaliseEvent:
    """Event emitted during rationalisation processing.

    Carries KValues, not bare KLines, and exposes no top-level significance
    field (@kvalue spec §Exchange, KE-3). ``query`` is the inbound KValue
    (the sender's declared assessment); ``proposal`` is Kalvin's assessment
    of the same (or an expansion-proposal) KLine. Each KValue supplies its
    own significance.

    ``role`` is the self-declared role of the emitting actor (the routing key the
    harness bus uses to address participants). It is ``None`` for events that are
    not part of a routed dialogue (e.g. internal cogitation emissions); dialogue
    actors set it so a runner can route and validate their responses without
    inferring the sender from context.
    """

    __slots__ = ("kind", "query", "proposal", "role")

    def __init__(
        self,
        kind: str,
        query: KValue,
        proposal: KValue,
        *,
        role: str | None = None,
    ):
        self.kind = kind
        self.query = query
        self.proposal = proposal
        self.role = role

    def __repr__(self) -> str:
        # Report Kalvin's assessment (the proposal's significance).
        return f"RationaliseEvent({self.kind!r}, sig={self.proposal.significance:#x})"


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
