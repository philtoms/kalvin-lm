"""Peer dialogue runner — a sink, not a driver (spec ``@specs/peer-dialogue.md``).

After the trainer delivers the opening entry to the trainee, both sides emit
on their own schedule, in any order and any count. :class:`PeerRunner` is a
**sink**: it receives emissions via :meth:`PeerRunner.receive`, validates each
against the table's distinct middle contents and the closing, and reports
completion. It does not call into actors, decide whose turn it is, or pace
the exchange.

This is a deliberate departure from the synchronous :func:`training.dialogue.runner.run`,
which drives the exchange via ``Actor.respond()``. See
``@docs/adr/0001-peer-runner-is-a-sink.md`` for the decision and the rejected
alternatives.

Spec mapping
------------
- PDT-5/PDT-6 — the sink contract and permitted (coverage-only) state.
- PDT-7..PDT-10 — matching (content presence, idempotent coverage, divergence,
  closing).
- PDT-11/PDT-12 — anticipation (permitted, unflagged, middle-only).
- PDT-13 — completion (closing-seen; coverage is a diagnostic, not a gate).
- PDT-14/PDT-15 — :class:`PeerDivergence` / :class:`PeerRunResult`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, Role, turn_content_key

# A peer content key: (role, kline_signature, kline_nodes_tuple, significance).
# The canonical form returned by ``turn_content_key``; aliased here for the
# coverage set's type.
ContentKey = tuple[str, int, tuple[int, ...], int]


# ── Divergence (PDT-14) ───────────────────────────────────────────────────


class PeerDivergence(Exception):  # noqa: N818 - spec names this type
    """A peer-run emission matched neither the closing nor any middle content.

    Raised by :meth:`PeerRunner.receive` under ``on_divergence="fail"`` when an
    event's ``(role, kline, significance)`` matches neither the closing nor any
    of the table's distinct middle contents. Carries the role, the emitted
    proposal, and the uncovered same-role contents at the moment of divergence
    (so a human can see what *could* have been emitted). Distinct from the
    synchronous :class:`~training.dialogue.runner.ActorDivergence`, which is
    cursor-shaped; peer divergence has no cursor.
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
            f"matches no closing or middle content "
            f"({len(unconsumed)} uncovered same-role contents)"
        )


# ── Result (PDT-15) ───────────────────────────────────────────────────────


@dataclass
class PeerRunResult:
    """Outcome of a peer dialogue run (spec §Types).

    ``events`` is **arrival-ordered** — every received emission, in the order
    the actors pushed them — not table-ordered. This is a deliberate difference
    from the synchronous :class:`~training.dialogue.runner.RunResult.events`,
    which is table/cursor-ordered. ``unmatched`` is populated only under
    ``on_divergence="accept"``; ``uncovered`` lists the distinct middle contents
    never seen (a coverage/efficiency diagnostic).
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
    :meth:`receive`. The runner validates each against the table's distinct
    middle contents and the closing, and reports completion. It holds
    **coverage bookkeeping only** (the table's fixed distinct middle content
    set, a growing covered subset, a closing reference, a closing-seen flag) —
    no actor-coupling state (no turn tracking, no cursors, no pacing) (PDT-6).

    Coverage is a measure of **efficiency**, not a matching count: duplicate
    table rows collapse to one distinct content, covered the first time any
    emission matches it; subsequent emissions of the same content are not
    divergence (coverage is idempotent). Completion is the closing entry alone
    (PDT-13) — the only really important goal. ``covered`` reports whether the
    distinct middle was fully seen, as a diagnostic, not a gate.

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
            raise ValueError(
                f"on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
            )
        if len(decoded) < 2:
            raise ValueError("peer run needs at least an opening and a closing turn")
        self._on_divergence = on_divergence
        # Coverage bookkeeping (PDT-6). The middle is held as a **fixed set of
        # distinct contents** (duplicates in the table collapse to one entry);
        # a **covered subset** grows monotonically as emissions match. Coverage
        # is idempotent — re-emitting covered content is not divergence. The
        # opening (decoded[0]) is not part of the coverage set: it is delivered
        # to the trainee by the caller and consumed positionally before any
        # emission reaches the sink.
        self._distinct_middle: set[ContentKey] = {
            turn_content_key(t) for t in decoded[1:-1]
        }
        self._covered: set[ContentKey] = set()
        self._closing: DecodedTurn = decoded[-1]
        self._closing_key: ContentKey = turn_content_key(self._closing)
        self._closing_seen: bool = False
        self._events: list[RationaliseEvent] = []
        self._unmatched: list[RationaliseEvent] = []

    # -- sink contract (PDT-5) ----------------------------------------------

    def receive(self, event: RationaliseEvent) -> None:
        """Receive one emission and update coverage (spec §Sink contract).

        Order of checks (PDT-10 then PDT-7..9): the closing is tested first —
        an emission equal to the closing marks it seen. Then the table's
        distinct middle contents are consulted: a content present in the table
        (covered or not) marks it covered (idempotent — re-emission is not
        divergence); a content present nowhere in the table is divergence (fail
        raises, accept records). The event is always appended to ``events`` in
        arrival order.
        """
        self._events.append(event)
        key = self._event_key(event)

        # PDT-10: closing takes precedence (it is content-distinct from every
        # middle row by the decode-time invariant, so closing-vs-middle cannot
        # arise; the check is first for clarity).
        if key == self._closing_key:
            self._closing_seen = True
            return

        # PDT-7/PDT-8: a content present in the table's distinct middle marks
        # it covered. Coverage is idempotent — duplicates in the table collapsed
        # to this one entry, and re-emitting it leaves it covered (no divergence).
        if key in self._distinct_middle:
            self._covered.add(key)
            return

        # PDT-9: divergence — the emission matches neither the closing nor any
        # distinct middle content. Under "fail" raise with the uncovered
        # same-role contents for diagnostics; under "accept" record and continue.
        if self._on_divergence == "fail":
            raise PeerDivergence(
                role=event.role or "?",
                emitted=event.proposal,
                unconsumed=tuple(self._uncovered_rows_for_role(event.role)),
            )
        self._unmatched.append(event)

    @property
    def complete(self) -> bool:
        """``closing-seen`` (PDT-13).

        Completion is the closing entry alone — the only really important goal.
        Coverage is a separate efficiency diagnostic (``covered``), not a
        terminal condition: a run is complete the moment the closing arrives,
        regardless of how much of the middle was seen. Extreme anticipation
        (closing-first, zero middle coverage) is technically complete, though
        rare in practice; ``covered`` makes the inefficiency visible.
        """
        return self._closing_seen

    @property
    def covered(self) -> bool:
        """True when every distinct middle content has been seen (efficiency).

        A diagnostic, not a terminal condition: completion is closing-seen
        (PDT-13). ``covered`` reports whether the distinct middle was fully
        traversed. Meaningful especially when a training strategy thins the
        middle before start, making the coverage fraction a real signal.
        """
        return self._distinct_middle <= self._covered

    @property
    def result(self) -> PeerRunResult:
        """The current :class:`PeerRunResult` snapshot (PDT-15)."""
        return PeerRunResult(
            events=list(self._events),
            complete=self.complete,
            covered=self.covered,
            unmatched=list(self._unmatched),
            uncovered=list(self._uncovered_rows()),
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

    def _uncovered_rows_for_role(self, role: str | None) -> list[DecodedTurn]:
        """Placeholder rows for uncovered same-role content (diagnostics only).

        Coverage is tracked by content key, not by the original ``DecodedTurn``
        rows (duplicates collapsed to one entry). For diagnostics we
        reconstruct minimal ``DecodedTurn`` objects carrying the role and a
        KValue of that content, so a human reading
        ``PeerDivergence.unconsumed`` or ``PeerRunResult.uncovered`` sees what
        distinct content remains unseen per role.
        """
        r = role if role is not None else "?"
        unseen = self._distinct_middle - self._covered
        return [
            _placeholder_turn(k)
            for k in sorted(unseen, key=_key_sort)
            if k[0] == r
        ]

    def _uncovered_rows(self) -> list[DecodedTurn]:
        """All uncovered distinct middle content as diagnostic placeholder rows."""
        unseen = self._distinct_middle - self._covered
        return [_placeholder_turn(k) for k in sorted(unseen, key=_key_sort)]


def _key_sort(k: ContentKey):
    return (k[0], k[1], k[2], k[3])


def _placeholder_turn(k: ContentKey) -> DecodedTurn:
    """Reconstruct a minimal ``DecodedTurn`` from a content key for diagnostics."""
    from kalvin.kline import KLine

    return DecodedTurn(role=cast(Role, k[0]), op="?", value=KValue(KLine(k[1], list(k[2])), k[3]))


def run_peer(
    decoded: Sequence[DecodedTurn],
    *,
    on_divergence: str = "fail",
) -> PeerRunner:
    """Construct a :class:`PeerRunner` for ``decoded`` (spec §The Runner).

    The runner is a sink; the caller delivers the opening to the trainee and
    then pushes emissions into ``runner.receive``. ``complete`` becomes True
    the moment the closing is seen (PDT-13).

    ``on_divergence`` (``"fail"`` default, ``"accept"``) selects the
    non-matching-emission policy (spec §Matching). Both are authoring knobs on
    the table (``DialogueTable.on_divergence``); the loader resolves them into
    this parameter.
    """
    return PeerRunner(decoded, on_divergence=on_divergence)
