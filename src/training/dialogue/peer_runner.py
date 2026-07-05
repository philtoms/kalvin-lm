"""Peer dialogue runner — a coverage-tracking subscriber over the harness
``MessageBus`` (spec ``@specs/peer-dialogue.md``).

The harness :class:`~training.harness.bus.MessageBus` is the **sink and the
relay**; the peer runner is a **coverage-tracking wildcard subscriber** plus a
thin driver that seeds the opening and runs the bus until the closing is seen.
Actors reply fire-and-forget via the thread-safe bus (true non-blocking, no
``asyncio``), addressing replies to the other role; the bus delivers; the
runner never reroutes. This is where the peer runner belongs — next to the
harness (ADR-0002, supersedes ADR-0001).

The dialogue is messy and real: there is **no synchronised alternation** (an
actor may reply zero-or-many times per ``accept``), and **anticipation** and
**interjection** are first-class and unflagged. Agents must rationalise and
cogitate to make sense of the stream — the point of Kalvin.

Spec mapping
------------
- PDT-5/PDT-6 — the runner is a bus subscriber holding coverage bookkeeping
  only; the bus is the sink/relay.
- PDT-7..PDT-10 — matching (content presence, idempotent coverage, divergence,
  closing).
- PDT-11/PDT-12 — anticipation + interjection (permitted, unflagged, middle-only).
- PDT-13 — completion (closing-seen; coverage is a diagnostic, not a gate).
- PDT-14/PDT-15 — :class:`PeerDivergence` / :class:`PeerRunResult`.
- PDT-17 — ``BusSink`` + ``Actor.accept`` (fire-and-forget, zero-or-many replies).
- PDT-18 — no synchronised alternation; route-to-other via bus addressing.
- PDT-19 — terminate on closing-seen; idle timeout ends a stall as incomplete.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, turn_content_key
from training.harness.bus import WILDCARD_ROLE, MessageBus
from training.harness.message import Message

# A peer content key: (role, kline_signature, kline_nodes_tuple, significance).
# The canonical form returned by ``turn_content_key``.
ContentKey = tuple[str, int, tuple[int, ...], int]

# The bus action used for peer emissions: the recipient's handler is the
# recipient actor's ``accept``.
_ACCEPT_ACTION = "accept"


# ── BusSink (PDT-17) ──────────────────────────────────────────────────────


@runtime_checkable
class BusSink(Protocol):
    """The narrow reply channel handed to actors (spec §Actor contract).

    Exposes only ``send``: an actor's sole interaction with the run is to emit
    replies. The bus satisfies this; the runner hands actors a ``BusSink``-typed
    view of the bus so they cannot reach coverage state or the driver. This is
    least-privilege: the actor replies, and nothing else.
    """

    def send(self, msg: Message) -> None: ...


# ── Actor.accept (PDT-17) ─────────────────────────────────────────────────


class PeerActor(Protocol):
    """The peer-regime actor contract (spec §Actor contract).

    ``accept`` is **fire-and-forget**: it returns immediately; the actor replies
    **zero or many** times by calling ``sink.send(Message(role=<other>,
    action="accept", message=<RationaliseEvent>))``. The actor decides when and
    whether to reply, possibly later (from a cogitation thread), possibly many
    times, possibly never. ``event=None`` signals "you open".

    This is separate from the ordered regime's ``Actor.respond`` (the blocking
    pull); the two regimes have different contracts and live in different
    modules. A concrete actor may implement one or both.
    """

    @property
    def role(self) -> str: ...

    def accept(self, event: RationaliseEvent | None, sink: BusSink) -> None: ...


# ── Divergence (PDT-14) ───────────────────────────────────────────────────


class PeerDivergence(Exception):  # noqa: N818 - spec names this type
    """A peer-run emission matched neither the closing nor any middle content.

    Raised by the coverage handler under ``on_divergence="fail"`` when an
    emission's ``(role, kline, significance)`` matches neither the closing nor
    any of the table's distinct middle contents. Carries the role, the emitted
    proposal, and the uncovered same-role contents at the moment of divergence.
    Distinct from the synchronous :class:`~training.dialogue.runner.ActorDivergence`,
    which is cursor-shaped; peer divergence has no cursor.
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

    ``events`` is **arrival-ordered** — every observed emission, in the order
    the bus delivered them — not table-ordered. ``unmatched`` is populated only
    under ``on_divergence="accept"``; ``uncovered`` lists the distinct middle
    contents never seen (a coverage/efficiency diagnostic).
    """

    events: list[RationaliseEvent] = field(default_factory=list)
    complete: bool = False
    covered: bool = False
    unmatched: list[RationaliseEvent] = field(default_factory=list)
    uncovered: list[DecodedTurn] = field(default_factory=list)


# ── The runner as a MessageBus subscriber (PDT-5..PDT-19) ─────────────────


class PeerRunner:
    """The peer-dialogue run: a bus subscriber + driver (spec §The Runner).

    The harness :class:`MessageBus` is the sink and the relay. The runner:

    - owns a ``MessageBus``;
    - subscribes itself as a **wildcard handler** for coverage bookkeeping;
    - subscribes each actor to its own role (its ``accept`` is the handler);
    - drives the run by seeding the opening (addressed to the trainer) and
      running ``bus.run()`` on a thread until the closing is seen or the idle
      timeout fires.

    Holds **coverage bookkeeping only** (the table's fixed distinct middle
    content set, a growing covered subset, a closing reference, a closing-seen
    flag). No actor-coupling state — no turns, no cursors, no pacing. The relay
    lives in the bus; the runner only observes and records.

    Construct via :func:`run_peer`; call :meth:`run` to drive.
    """

    def __init__(
        self,
        decoded: Sequence[DecodedTurn],
        trainer: PeerActor,
        trainee: PeerActor,
        *,
        on_divergence: str = "fail",
        idle_timeout: float = 5.0,
    ) -> None:
        if on_divergence not in ("fail", "accept"):
            raise ValueError(
                f"on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
            )
        if len(decoded) < 2:
            raise ValueError("peer run needs at least an opening and a closing turn")
        self._on_divergence = on_divergence
        self._idle_timeout = idle_timeout
        self._trainer = trainer
        self._trainee = trainee

        # Coverage bookkeeping (PDT-6). Fixed distinct middle set (duplicates
        # collapse to one entry); covered subset grows monotonically. The
        # opening (decoded[0]) is consumed positionally — it is neither coverage
        # nor divergence, so its key is tracked and skipped by the handler.
        self._distinct_middle: set[ContentKey] = {
            turn_content_key(t) for t in decoded[1:-1]
        }
        self._covered: set[ContentKey] = set()
        self._opening_key: ContentKey = turn_content_key(decoded[0])
        self._closing: DecodedTurn = decoded[-1]
        self._closing_key: ContentKey = turn_content_key(self._closing)
        self._closing_seen: bool = False
        self._events: list[RationaliseEvent] = []
        self._unmatched: list[RationaliseEvent] = []
        # An exception raised on the bus dispatch thread (e.g. PeerDivergence)
        # is captured here and re-raised by run() on the caller's thread, so
        # callers see peer failures rather than silent thread warnings.
        self._thread_exc: BaseException | None = None

        # The bus is the sink and relay. The runner is a wildcard subscriber
        # (coverage); each actor is subscribed to its own role (its accept).
        self._bus = MessageBus()
        self._bus.subscribe(WILDCARD_ROLE, self._on_emission)
        self._bus.subscribe(trainer.role, self._make_handler(trainer))
        self._bus.subscribe(trainee.role, self._make_handler(trainee))

    # -- the driver (PDT-18, PDT-19) -----------------------------------------

    def run(self) -> PeerRunResult:
        """Drive the peer run to completion (spec §The Runner).

        Seeds the opening by addressing an ``accept`` message to the trainer
        (``message=None`` signals "you open"), then runs ``bus.run()`` on a
        dedicated thread. The trainer replies (addressed to the trainee), the
        bus delivers, the trainee replies (addressed to the trainer), and so on
        — the bus relays; the runner only observes. Terminates when the coverage
        handler sees the closing (it calls ``bus.stop()``), or when the idle
        timeout fires with no closing (a stall: ``complete = False``).
        """
        # Seed the opening: tell the trainer to open. message=None = "you open".
        self._bus.send(
            Message(role=self._trainer.role, action=_ACCEPT_ACTION, message=None)
        )
        # Run the bus on a thread with an idle timeout. The bus's run() blocks
        # on queue.get(); we bound it by stopping after the idle deadline if the
        # closing hasn't arrived. The closing-seen path calls bus.stop() itself.
        bus_thread = threading.Thread(target=self._run_bus_bounded, daemon=True)
        bus_thread.start()
        bus_thread.join()
        # Re-raise any exception captured on the bus thread (e.g. divergence).
        if self._thread_exc is not None:
            raise self._thread_exc
        return self.result

    def _run_bus_bounded(self) -> None:
        """Run the bus, enforcing the idle timeout via a watchdog.

        ``MessageBus.run()`` blocks indefinitely on ``queue.get()`` until
        ``stop()``. We run it on an inner thread and join with the idle timeout
        in a loop: if the join times out (no message processed within the idle
        window) and the closing hasn't been seen, it's a stall — stop the bus.
        """
        inner = threading.Thread(target=self._bus.run, daemon=True)
        inner.start()
        while True:
            inner.join(self._idle_timeout)
            if not inner.is_alive():
                return  # bus stopped (closing-seen called bus.stop())
            if self._closing_seen:
                return  # race: closing seen but bus not yet exited — done
            # Idle window elapsed with no closing: stall. Stop the bus.
            self._bus.stop()
            inner.join(self._idle_timeout)
            return

    # -- coverage handler (the wildcard subscriber) (PDT-7..PDT-13) ---------

    def _on_emission(self, msg: Message) -> None:
        """Wildcard handler: record coverage on every observed emission.

        The opening seed (``message=None``) is not an emission — skip it. Every
        real emission carries a ``RationaliseEvent`` in ``message``.
        """
        event = msg.message
        if event is None:
            return  # the opening seed, not an emission
        assert isinstance(event, RationaliseEvent)
        self._events.append(event)
        key = self._event_key(event)

        # The opening is consumed positionally (PDT-3): it is neither coverage
        # nor divergence. The trainer emits it as its first turn; skip it here.
        if key == self._opening_key and not self._closing_seen:
            return

        # PDT-10: closing seen → mark and stop the bus (terminal).
        if key == self._closing_key:
            self._closing_seen = True
            self._bus.stop()
            return

        # PDT-7/PDT-8: a content present in the distinct middle marks it covered
        # (idempotent — re-emitting covered content is not divergence).
        if key in self._distinct_middle:
            self._covered.add(key)
            return

        # PDT-9: divergence — present nowhere in the table.
        if self._on_divergence == "fail":
            exc = PeerDivergence(
                role=event.role or "?",
                emitted=event.proposal,
                unconsumed=tuple(self._uncovered_rows_for_role(event.role)),
            )
            self._thread_exc = exc
            self._bus.stop()
            return
        self._unmatched.append(event)

    # -- actor handler adapter ----------------------------------------------

    def _make_handler(self, actor: PeerActor):
        """Adapt an actor's ``accept`` to the bus's ``(msg) -> None`` handler.

        The bus delivers a ``Message`` addressed to the actor's role; this
        handler unpacks the incoming ``RationaliseEvent`` (or ``None`` for the
        opening seed) and the bus-as-sink, and calls ``actor.accept``. The actor
        replies zero-or-many via ``sink.send`` (addressed to the other role);
        the bus relays — the runner does not.
        """
        other = self._trainee.role if actor.role == self._trainer.role else self._trainer.role

        def handler(msg: Message) -> None:
            incoming = msg.message  # RationaliseEvent | None
            # Wrap the bus so the actor can only address the other role. This
            # enforces the route-to-other rule (PDT-18) structurally: the actor
            # calls sink.send(event) and the wrapper sets role=other.
            sink = _OtherRoleSink(self._bus, other)
            actor.accept(incoming, sink)

        return handler

    # -- result + diagnostics -----------------------------------------------

    @property
    def complete(self) -> bool:
        """``closing-seen`` (PDT-13). Completion is the closing entry alone."""
        return self._closing_seen

    @property
    def covered(self) -> bool:
        """True when every distinct middle content has been seen (efficiency).

        A diagnostic, not a terminal condition (PDT-13).
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
        return (
            event.role or "?",
            event.proposal.kline.signature,
            tuple(event.proposal.kline.nodes),
            event.proposal.significance,
        )

    def _uncovered_rows_for_role(self, role: str | None) -> list[DecodedTurn]:
        r = role if role is not None else "?"
        unseen = self._distinct_middle - self._covered
        return [
            _placeholder_turn(k)
            for k in sorted(unseen, key=_key_sort)
            if k[0] == r
        ]

    def _uncovered_rows(self) -> list[DecodedTurn]:
        unseen = self._distinct_middle - self._covered
        return [_placeholder_turn(k) for k in sorted(unseen, key=_key_sort)]


# ── Sink wrapper: enforces route-to-other (PDT-18) ────────────────────────


class _OtherRoleSink:
    """A ``BusSink`` that forces every emission to the *other* role.

    The route-to-other rule (PDT-18) is enforced structurally: an actor calls
    ``sink.send(event)`` with an event carrying its own role (the emitter), and
    this wrapper addresses the bus ``Message`` to the *other* role. The actor
    cannot misroute. (A future relaxation could let the actor choose the
    recipient; today the dialogue is strictly cross-role.)
    """

    def __init__(self, bus: MessageBus, other_role: str) -> None:
        self._bus = bus
        self._other = other_role

    def send(self, msg: Message) -> None:  # type: ignore[override]
        # Re-address to the other role, preserving the action and the event
        # payload (which carries the emitter's role on event.role).
        self._bus.send(
            Message(role=self._other, action=_ACCEPT_ACTION, message=msg.message)
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _key_sort(k: ContentKey):
    return (k[0], k[1], k[2], k[3])


def _placeholder_turn(k: ContentKey) -> DecodedTurn:
    """Reconstruct a minimal ``DecodedTurn`` from a content key for diagnostics."""
    from typing import cast

    from kalvin.kline import KLine
    from training.dialogue.decoder import Role

    return DecodedTurn(
        role=cast(Role, k[0]),
        op="?",
        value=KValue(KLine(k[1], list(k[2])), k[3]),
    )


def run_peer(
    decoded: Sequence[DecodedTurn],
    trainer: PeerActor,
    trainee: PeerActor,
    *,
    on_divergence: str = "fail",
    idle_timeout: float = 5.0,
) -> PeerRunner:
    """Construct a :class:`PeerRunner` for ``decoded`` (spec §The Runner).

    The caller calls :meth:`PeerRunner.run` to drive. ``on_divergence``
    (``"fail"`` default) and ``idle_timeout`` (seconds, default 5.0) select the
    divergence policy and the stall deadline.
    """
    return PeerRunner(
        decoded,
        trainer,
        trainee,
        on_divergence=on_divergence,
        idle_timeout=idle_timeout,
    )
