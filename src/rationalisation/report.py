"""The run's report — the events of a rationalisation, its halts read as
proposals, and the Rationalisation record both engines return. Events are
reentry events: each is one thing that happened to an entry — a pairing,
a frame minted, a stall, an exchange, a grounding. The run's trace;
reporting only, never read by a search."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from kalvin.kline import KNode, KNone
from rationalisation.algebra import Kline, effort


@dataclass(eq=False)  # an event is an occurrence, never a value
class Event:
    """One reentry event — the run's trace."""

    kind: str  # pair | frame | stall | exchange | close | wait | known
    depth: int
    entry: KNode  # the reentry's signature — the event's subject
    candidate: Kline | None = None
    F: KNode = KNone
    U: KNode = KNone
    O: KNode = KNone
    frame: KNode = KNone  # content the event minted or settled


@dataclass(frozen=True)
class End:
    """A terminal arrival — with the halt measure where a kline halted."""

    reason: str
    kline: Kline | None
    effort: Fraction | None = None


@dataclass(frozen=True)
class Proposal:
    """A halted kline read as a request (§5 Definition 15)."""

    kind: str  # "acquire" | "connect" | "continue"
    kline: Kline
    effort: Fraction


@dataclass
class Rationalisation:
    """One run's report: its events in order, its halts, its stores, its claim."""

    stores: list[Kline]
    ends: list[End]
    events: list[Event] = field(default_factory=list)
    claim: Kline | None = None


def proposals(r: Rationalisation) -> list[Proposal]:
    """Every halt generates one proposal (§5 Definition 15), ordered by ascending
    effort — least entropy first, most significant next step."""
    out = []
    for e in r.ends:
        if e.kline is None:
            continue
        if e.reason.startswith("stalled") or "no candidates" in e.reason:
            kind = "connect"  # fit wanted, nobody to talk to — the sides starved
        elif e.reason.startswith("bound"):
            kind = "continue"
        else:
            kind = "acquire"
        out.append(Proposal(kind, e.kline, e.effort if e.effort is not None else effort(e.kline)))
    return sorted(out, key=lambda p: p.effort)
