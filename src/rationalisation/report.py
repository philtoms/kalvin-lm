"""The path's report — the events of a rationalisation, and the
Rationalisation record the runner returns. Events are reentry events:
each is one thing that happened to an entry — a frame minted, a stall,
an exchange, a grounding. The claiming path's trace; reporting only,
never read by a search. Claims and stalls are not report: they broadcast
through the sinks while the run is live."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from kalvin.kline import KLine, KNode, KNone


@dataclass(eq=False)  # an event is an occurrence, never a value
class Event:
    """One reentry event — the run's trace."""

    kind: str  # entry | frame | stall | exchange | close | wait | known
    entry: KNode  # the reentry's signature — the event's subject
    candidate: KLine | None = None
    F: KNode = KNone
    U: KNode = KNone
    O: KNode = KNone
    frame: KNode = KNone  # content the event minted or settled


@dataclass(frozen=True)
class Proposal:
    """A stalled kline read as a request — the harness's pending shape."""

    kind: str  # "connect" — fit wanted, nobody to talk to
    kline: KLine
    effort: Fraction


@dataclass
class Rationalisation:
    """One path's report: its events in order — the story the finishing
    entry carried forward, the carrying lineage's account."""

    events: list[Event] = field(default_factory=list)
