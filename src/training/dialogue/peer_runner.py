"""Peer dialogue runner — a coverage-tracking subscriber over the harness
``MessageBus`` (spec ``@specs/peer-dialogue.md``).

The harness :class:`~training.harness.bus.MessageBus` is the **sink and the
relay**; the peer runner is a **coverage-tracking wildcard subscriber** plus a
thin driver that seeds the opening and runs the bus until the closing is seen.

This module mirrors the :class:`~kalvin.agent.KAgent` adapter pattern: actors
take an :class:`EventSink` at construction (as KAgent takes an adapter) and
publish events to it (as KAgent publishes via its adapter). The runner builds a
bus-wired sink per actor (the sink bridges ``on_event`` to a bus ``Message``
addressed to the other role), so any adapter-driven actor is drop-in.

The dialogue is messy and real: **no synchronised alternation** (an actor may
publish zero-or-many per incoming), and **anticipation** and **interjection**
are first-class and unflagged. Agents must rationalise and cogitate to make
sense of the stream — the point of Kalvin.

Spec mapping
------------
- PDT-5/PDT-6 — the runner is a bus subscriber holding coverage bookkeeping
  only; the bus is the sink/relay.
- PDT-7..PDT-10 — matching (content presence, idempotent coverage, divergence,
  closing).
- PDT-11/PDT-12 — anticipation + interjection (permitted, unflagged, middle-only).
- PDT-13 — completion (closing-seen; coverage is a diagnostic, not a gate).
- PDT-14/PDT-15 — :class:`PeerDivergence` / :class:`PeerRunResult`.
- PDT-17 — ``EventSink`` + adapter-driven actors (mirrors KAgent).
- PDT-18 — no synchronised alternation; route-to-other via the bus-wired sink.
- PDT-19 — terminate on closing-seen; idle timeout ends a stall as incomplete.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, turn_content_key
from training.harness.bus import WILDCARD_ROLE, MessageBus
from training.harness.message import Message

# A peer content key: (role, kline_signature, kline_nodes_tuple, significance).
ContentKey = tuple[str, int, tuple[int, ...], int]

# The bus action used for peer emissions: the recipient's handler is the
# recipient actor's ``accept``.
_ACCEPT_ACTION = "accept"


# ── EventSink (PDT-17) — mirrors KAgent's adapter ─────────────────────────


@runtime_checkable
class EventSink(Protocol):
    """The publish target an actor holds (mirrors ``KAgentAdapter``).

    An actor is constructed with an ``EventSink`` and publishes events to it via
    ``on_event`` — exactly as :class:`~kalvin.agent.KAgent` holds an adapter and
    publishes via ``_publish`` → ``adapter.on_event``. The bus-wired sink
    (:class:`_BusEventSink`) bridges each ``on_event`` to a bus ``Message``
    addressed to the other role, so the actor's published events flow onto the
    relay without the actor knowing about the bus. This is least-coupling: the
    actor publishes, the sink routes.
    """

    def on_event(self, event: RationaliseEvent) -> None: ...


@runtime_checkable
class PeerActor(Protocol):
    """An adapter-driven dialogue actor (mirrors KAgent's shape).

    Holds an :class:`EventSink` (injected at construction) and publishes events
    to it. ``accept`` receives an incoming event (or ``None`` for the opening
    seed) and the actor decides whether/when/how-many events to publish via its
    sink — fire-and-forget, zero-or-many.
    """

    @property
    def role(self) -> str: ...

    def accept(self, event: RationaliseEvent | None) -> None: ...


# Actor factory: the runner builds the bus-wired sink (it owns the bus) and
# constructs the actor via this callable, passing the sink. Mirrors how a
# harness wires ``KAgentAdapter(bus)`` into ``KAgent(...)``: only the component
# that owns the bus can build the adapter, so it builds the actor too.
ActorFactory = Callable[[EventSink], PeerActor]


# ── Divergence (PDT-14) ───────────────────────────────────────────────────


class PeerDivergence(Exception):  # noqa: N818 - spec names this type
    """A peer-run emission matched neither the closing nor any middle content.

    Raised under ``on_divergence="fail"`` when an emission's
    ``(role, kline, significance)`` matches neither the closing nor any of the
    table's distinct middle contents. Carries the role, the emitted proposal,
    and the uncovered same-role contents at the moment of divergence. Peer
    divergence has no cursor — coverage is content-keyed, not positional.
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
    the bus delivered them. ``unmatched`` is populated only under
    ``on_divergence="accept"``; ``uncovered`` lists the distinct middle
    contents never seen (a coverage/efficiency diagnostic).
    """

    events: list[RationaliseEvent] = field(default_factory=list)
    complete: bool = False
    covered: bool = False
    unmatched: list[RationaliseEvent] = field(default_factory=list)
    uncovered: list[DecodedTurn] = field(default_factory=list)


# ── The bus-wired sink (enforces route-to-other, PDT-18) ──────────────────


class _BusEventSink:
    """An :class:`EventSink` that publishes each event to the other role.

    The route-to-other rule (PDT-18) is enforced structurally: the actor calls
    ``on_event(event)`` (the event carries the emitter's role on ``event.role``)
    and this sink addresses the bus ``Message`` to the *other* role. The actor
    cannot misroute — it merely publishes, as KAgent does.
    """

    def __init__(self, bus: MessageBus, other_role: str) -> None:
        self._bus = bus
        self._other = other_role

    def on_event(self, event: RationaliseEvent) -> None:
        self._bus.send(
            Message(role=self._other, action=_ACCEPT_ACTION, message=event)
        )


# ── The runner as a MessageBus subscriber (PDT-5..PDT-19) ─────────────────


class PeerRunner:
    """The peer-dialogue run: a bus subscriber + driver (spec §The Runner).

    The harness :class:`MessageBus` is the sink and the relay. The runner:

    - owns a ``MessageBus``;
    - builds a bus-wired :class:`EventSink` per actor (addressed to the other
      role) and constructs each actor with its sink via the actor factory —
      mirroring how a harness injects ``KAgentAdapter(bus)`` into ``KAgent``;
    - subscribes itself as a **wildcard handler** for coverage bookkeeping;
    - subscribes each actor's ``accept`` as its role's handler;
    - drives the run by seeding the opening (addressed to the trainer) and
      running ``bus.run()`` on a thread until the closing is seen or the idle
      timeout fires.

    Holds **coverage bookkeeping only**. No actor-coupling state. The relay
    lives in the bus; the runner only observes and records.

    Construct via :func:`run_peer`; call :meth:`run` to drive.
    """

    def __init__(
        self,
        decoded: Sequence[DecodedTurn],
        trainer_factory: ActorFactory,
        trainee_factory: ActorFactory,
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

        # Coverage bookkeeping (PDT-6). Fixed distinct middle set (duplicates
        # collapse to one entry); covered subset grows monotonically. The
        # opening (decoded[0]) is consumed positionally — neither coverage nor
        # divergence — so its key is tracked and skipped by the handler.
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
        self._thread_exc: BaseException | None = None

        # The bus is the sink and relay. Build the bus-wired sinks and construct
        # the actors with them (factories), then subscribe the actors' accept
        # handlers and the runner's wildcard coverage handler.
        self._bus = MessageBus()
        self._trainer = trainer_factory(_BusEventSink(self._bus, "K"))
        self._trainee = trainee_factory(_BusEventSink(self._bus, "T"))
        if self._trainer.role == self._trainee.role:
            raise ValueError(
                f"trainer and trainee must have different roles, got {self._trainer.role!r}"
            )
        self._bus.subscribe(WILDCARD_ROLE, self._on_emission)
        self._bus.subscribe(self._trainer.role, self._make_handler(self._trainer))
        self._bus.subscribe(self._trainee.role, self._make_handler(self._trainee))

    # -- the driver (PDT-18, PDT-19) -----------------------------------------

    def run(self) -> PeerRunResult:
        """Drive the peer run to completion (spec §The Runner).

        Seeds the opening by addressing an ``accept`` message to the trainer
        (``message=None`` signals "you open"), then runs ``bus.run()`` on a
        dedicated thread. Terminates when the coverage handler sees the closing
        (it calls ``bus.stop()``), or when the idle timeout fires with no
        closing (a stall: ``complete = False``).
        """
        self._bus.send(
            Message(role=self._trainer.role, action=_ACCEPT_ACTION, message=None)
        )
        bus_thread = threading.Thread(target=self._run_bus_bounded, daemon=True)
        bus_thread.start()
        bus_thread.join()
        if self._thread_exc is not None:
            raise self._thread_exc
        return self.result

    def _run_bus_bounded(self) -> None:
        """Run the bus, enforcing the idle timeout via a watchdog.

        The idle timeout is **skipped when a debugger is attached** (detected
        via :func:`sys.gettrace`, or forced off with the ``KALVIN_NO_IDLE_TIMEOUT``
        env var). Under a debugger the run pauses on breakpoints while wall
        clock keeps ticking; the watchdog would then mistake the pause for a
        stall and terminate the run early. The timeout still guards production
        (no debugger) and the unit tests (which pass ``idle_timeout`` short and
        run without a tracer attached).
        """
        if _idle_timeout_disabled():
            inner = threading.Thread(target=self._bus.run, daemon=True)
            inner.start()
            inner.join()
            return
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
        """Wildcard handler: record coverage on every observed emission."""
        event = msg.message
        if event is None:
            return  # the opening seed, not an emission
        assert isinstance(event, RationaliseEvent)
        self._events.append(event)
        key = self._event_key(event)

        # The opening is consumed positionally (PDT-3): neither coverage nor
        # divergence. The trainer publishes it as its first turn; skip it here.
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
        """Adapt an actor's ``accept`` to the bus's ``(msg) -> None`` handler."""

        def handler(msg: Message) -> None:
            actor.accept(msg.message)  # RationaliseEvent | None

        return handler

    # -- result + diagnostics -----------------------------------------------

    @property
    def complete(self) -> bool:
        """``closing-seen`` (PDT-13)."""
        return self._closing_seen

    @property
    def covered(self) -> bool:
        """True when every distinct middle content has been seen (efficiency)."""
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


# ── Helpers ───────────────────────────────────────────────────────────────


def _idle_timeout_disabled() -> bool:
    """True when the idle-timeout watchdog must not enforce its deadline.

    The timeout ends a stalled run as incomplete (PDT-19), but it is wall-clock
    based: while paused on a breakpoint the deadline still elapses, so the
    watchdog would terminate the run early. Disable it when a debugger tracer
    is attached (``sys.gettrace()`` returns one) or when the caller opts out via
    the ``KALVIN_NO_IDLE_TIMEOUT`` env var. Production and the unit tests run
    without a tracer, so PDT-19 is unaffected there.
    """
    if os.environ.get("KALVIN_NO_IDLE_TIMEOUT"):
        return True
    return sys.gettrace() is not None


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
    trainer_factory: ActorFactory,
    trainee_factory: ActorFactory,
    *,
    on_divergence: str = "fail",
    idle_timeout: float = 5.0,
) -> PeerRunner:
    """Construct a :class:`PeerRunner` for ``decoded`` (spec §The Runner).

    ``trainer_factory`` / ``trainee_factory`` are callables ``(sink) -> PeerActor``:
    the runner builds the bus-wired sink (it owns the bus) and constructs each
    actor with its sink — mirroring how a harness injects ``KAgentAdapter(bus)``
    into ``KAgent``. This makes any adapter-driven actor drop-in: the actor
    author writes only the publish-to-sink logic; the runner handles the bus.

    The caller calls :meth:`PeerRunner.run` to drive.
    """
    return PeerRunner(
        decoded,
        trainer_factory,
        trainee_factory,
        on_divergence=on_divergence,
        idle_timeout=idle_timeout,
    )
