"""Peer dialogue runner — a sink, not a driver (spec ``@specs/peer-dialogue.md``).

After the trainer delivers the opening entry to the trainee, both sides emit
on their own schedule, in any order and any count. :class:`PeerRunner` is a
**sink**: it receives emissions via :meth:`PeerRunner.receive`, validates each
against the unconsumed same-role middle rows by content (with duplicate
collapse), and watches for the closing entry. It does not call into actors,
decide whose turn it is, or pace the exchange.

This is a deliberate departure from the synchronous :func:`training.dialogue.runner.run`,
which drives the exchange via ``Actor.respond()``. See
``@docs/adr/0001-peer-runner-is-a-sink.md`` for the decision and the rejected
alternatives.

Spec mapping
------------
- PDT-5/PDT-6 — the sink contract and permitted (coverage-only) state.
- PDT-7..PDT-10 — matching (content equality, duplicate collapse, divergence,
  closing).
- PDT-11/PDT-12 — anticipation (permitted, unflagged, middle-only).
- PDT-13 — completion (closing seen AND middle distinct-set exhausted).
- PDT-14/PDT-15 — :class:`PeerDivergence` / :class:`PeerRunResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.events import RationaliseEvent


# ── Divergence (PDT-14) ───────────────────────────────────────────────────


class PeerDivergence(Exception):  # noqa: N818 - spec names this type
    """A peer-run emission matched no unconsumed same-role row (spec §Matching).

    Raised by :meth:`PeerRunner.receive` under ``on_divergence="fail"`` when an
    event's ``(role, kline, significance)`` matches neither the closing nor any
    unconsumed middle row for its role. Carries the role, the emitted proposal,
    and the unconsumed same-role rows at the moment of divergence (so a human
    can see what *could* have been emitted). Distinct from the synchronous
    :class:`~training.dialogue.runner.ActorDivergence`, which is cursor-shaped;
    peer divergence has no cursor.
    """

    def __init__(
        self,
        role: str,
        emitted: KValue,
        unconsumed: tuple[DecodedTurn, ...],
    ) -> None:
        self.role = role
        self.emitted = emitted
        self.unconsumed = unconsumed
        super().__init__(
            f"{role} divergence: emitted sig={emitted.significance:#x} "
            f"matches no unconsumed same-role row "
            f"({len(unconsumed)} unconsumed for {role})"
        )


# ── Result (PDT-15) ───────────────────────────────────────────────────────


@dataclass
class PeerRunResult:
    """Outcome of a peer dialogue run (spec §Types).

    ``events`` is **arrival-ordered** — every received emission, in the order
    the actors pushed them — not table-ordered. This is a deliberate difference
    from the synchronous :class:`~training.dialogue.runner.RunResult.events`,
    which is table/cursor-ordered. ``unmatched`` is populated only under
    ``on_divergence="accept"``; ``uncovered`` is populated on incomplete runs.
    """

    events: list[RationaliseEvent] = field(default_factory=list)
    complete: bool = False
    covered: bool = False
    unmatched: list[RationaliseEvent] = field(default_factory=list)
    uncovered: list[DecodedTurn] = field(default_factory=list)
