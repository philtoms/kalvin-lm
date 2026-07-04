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

from collections.abc import Sequence
from dataclasses import dataclass, field

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, turn_content_key

# A peer content key: (role, kline_signature, kline_nodes_tuple, significance).
# The canonical form returned by ``turn_content_key``; aliased here for the
# coverage set's type.
ContentKey = tuple[str, int, tuple[int, ...], int]


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


# ── The sink (PDT-5..PDT-13) ──────────────────────────────────────────────


class PeerRunner:
    """The peer-dialogue sink (spec §The Runner, §Sink contract).

    A sink, not a driver. After the caller has delivered the opening entry to
    the trainee, the actors emit on their own and push emissions into
    :meth:`receive`. The runner validates each against the unconsumed same-role
    middle rows by content (duplicates collapse), watches for the closing, and
    reports completion. It holds **coverage bookkeeping only** (the unconsumed
    middle content set, a closing reference, a closing-seen flag) — no
    actor-coupling state (no turn tracking, no cursors, no pacing) (PDT-6).

    Construct via :func:`run_peer`. The caller is responsible for delivering the
    opening to the trainee; the runner performs no outbound delivery (spec Out
    of Scope).
    """

    def __init__(
        self,
        decoded: Sequence[DecodedTurn],
        *,
        on_divergence: str = "fail",
    ) -> None:
        if on_divergence not in ("fail", "accept"):
            raise ValueError(f"on_divergence must be 'fail' or 'accept', got {on_divergence!r}")
        if len(decoded) < 2:
            raise ValueError(
                "peer run needs at least an opening and a closing turn"
            )
        self._on_divergence = on_divergence
        # Coverage bookkeeping (PDT-6): the unconsumed distinct middle content
        # set, plus the closing reference and a closing-seen flag. The opening
        # (decoded[0]) seeds the run but is not part of the coverage set — it
        # is delivered to the trainee by the caller and consumed positionally
        # before any emission reaches the sink.
        self._unconsumed: set[ContentKey] = {
            turn_content_key(t) for t in decoded[1:-1]
        }
        self._closing: DecodedTurn = decoded[-1]
        self._closing_key: ContentKey = turn_content_key(self._closing)
        self._closing_seen: bool = False
        self._events: list[RationaliseEvent] = []
        self._unmatched: list[RationaliseEvent] = []

    # -- sink contract (PDT-5) ----------------------------------------------

    def receive(self, event: RationaliseEvent) -> None:
        """Receive one emission and update coverage (spec §Sink contract).

        Order of checks (PDT-10 then PDT-7..9): the closing is tested first so
        that an emission matching the closing marks it seen and consumes it
        without also being treated as a middle match or a divergence. Then the
        middle coverage set is consulted: a match consumes the distinct content
        (duplicates collapse); a non-match is divergence (fail raises, accept
        records). The event is always appended to ``events`` in arrival order.
        """
        self._events.append(event)
        key = self._event_key(event)

        # PDT-10: closing takes precedence (it is content-distinct from the
        # opening by the decode-time invariant, but may coincide with a middle
        # row; closing wins either way).
        if key == self._closing_key:
            self._closing_seen = True
            return

        # PDT-7/PDT-8: content match against the unconsumed middle set; the set
        # membership collapses duplicates in one step.
        if key in self._unconsumed:
            self._unconsumed.discard(key)
            return

        # PDT-9: divergence. Under "fail" raise with the unconsumed same-role
        # rows for diagnostics; under "accept" record and continue.
        if self._on_divergence == "fail":
            unconsumed_for_role = tuple(
                t
                for t in self._unconsumed_rows_for_role(event.role)
            )
            raise PeerDivergence(
                role=event.role or "?",
                emitted=event.proposal,
                unconsumed=unconsumed_for_role,
            )
        self._unmatched.append(event)

    @property
    def complete(self) -> bool:
        """``closing-seen AND middle distinct-set exhausted`` (PDT-13)."""
        return self._closing_seen and not self._unconsumed

    @property
    def result(self) -> PeerRunResult:
        """The current :class:`PeerRunResult` snapshot (PDT-15)."""
        return PeerRunResult(
            events=list(self._events),
            complete=self.complete,
            covered=not self._unconsumed,
            unmatched=list(self._unmatched),
            uncovered=list(self._unconsumed_rows()),
        )

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _event_key(event: RationaliseEvent) -> ContentKey:
        """The content key of an emitted event (role, sig, nodes, significance)."""
        return (
            event.role or "?",
            event.proposal.kline.signature,
            tuple(event.proposal.kline.nodes),
            event.proposal.significance,
        )

    def _unconsumed_rows_for_role(self, role: str | None) -> list[DecodedTurn]:
        """Placeholder rows for unconsumed same-role content (diagnostics only).

        Coverage is tracked by content key, not by the original ``DecodedTurn``
        rows (duplicates have collapsed). For diagnostics we reconstruct minimal
        ``DecodedTurn``\\ s carrying the role and a KValue of that content, so a
        human reading ``PeerDivergence.unconsumed`` or ``PeerRunResult.uncovered``
        sees what content remains outstanding per role.
        """
        r = role if role is not None else "?"
        return [
            DecodedTurn(role=_role(k[0]), op="?", value=KValue(
                _kline_from_key(k), k[3]
            ))
            for k in sorted(self._unconsumed, key=lambda x: (x[0], x[1], x[2], x[3]))
            if k[0] == r
        ]

    def _unconsumed_rows(self) -> list[DecodedTurn]:
        """All unconsumed middle rows as diagnostic placeholder ``DecodedTurn``\\ s."""
        return [
            DecodedTurn(role=_role(k[0]), op="?", value=KValue(
                _kline_from_key(k), k[3]
            ))
            for k in sorted(self._unconsumed, key=lambda x: (x[0], x[1], x[2], x[3]))
        ]


def _kline_from_key(k: ContentKey):
    """Reconstruct a KLine from a content key's (sig, nodes) for diagnostics."""
    from kalvin.kline import KLine

    return KLine(k[1], list(k[2]))


def _role(label: str):
    """Cast a content-key role label to ``Role`` for diagnostic DecodedTurns."""
    from typing import cast

    from training.dialogue.decoder import Role

    return cast(Role, label)


def run_peer(
    decoded: Sequence[DecodedTurn],
    *,
    on_divergence: str = "fail",
) -> PeerRunner:
    """Construct a :class:`PeerRunner` for ``decoded`` (spec §The Runner).

    The runner is a sink; the caller delivers the opening to the trainee and
    then pushes emissions into ``runner.receive``. ``complete`` becomes True
    once the closing is seen and the middle distinct-set is exhausted.

    ``on_divergence`` (``"fail"`` default, ``"accept"``) selects the
    non-matching-emission policy (spec §Matching). Both are authoring knobs on
    the table (``DialogueTable.on_divergence``); the loader resolves them into
    this parameter.
    """
    return PeerRunner(decoded, on_divergence=on_divergence)
