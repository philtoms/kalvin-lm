"""Dialogue runner — a coverage-tracking subscriber over the harness
``MessageBus`` (spec ``@specs/dialogue-driven-training.md``).

The harness :class:`~training.harness.bus.MessageBus` is the **sink and the
relay**; this module's runner (:class:`Runner`) is a **coverage-tracking
wildcard subscriber** plus a thin driver that seeds the opening and runs the bus
until the closing is seen. The actors the runner drives live in
:mod:`training.dialogue.actors`.

An actor takes an :class:`EventSink` at construction and publishes events to it
via ``on_event``. The runner builds a bus-wired sink per actor (the sink bridges
``on_event`` to a bus ``Message`` addressed to the other role), so any actor is
drop-in: it publishes to its sink, the sink routes onto the bus.

The dialogue is messy and real: **no synchronised alternation** (an actor may
publish one-or-many per incoming), and **anticipation** and **interjection**
are first-class and unflagged. Agents must rationalise and cogitate to make
sense of the stream — the point of Kalvin.

Every ``accept`` yields at least one proposal (the ``burst >= 1`` contract,
§Actor contract): an actor with nothing substantive to say publishes a **PASS**
— a sentinel proposal (``PASS_SIGNATURE`` at S1). The runner intercepts PASS
before matching: it is neither coverage nor divergence, is routed to the other
role, and **two consecutive PASSes** (each side passing in turn) is a **stall**
that terminates the run incomplete. A run therefore never goes silent — a
compliant actor always emits — so the runner has no idle timeout: termination is
closing-seen (complete) or mutual-PASS (incomplete), both content-driven.

Spec mapping
------------
- DDT-5/DDT-6 — the runner is a bus subscriber holding coverage bookkeeping
  only; the bus is the sink/relay.
- DDT-7..DDT-10 — matching (content presence, idempotent coverage, divergence,
  closing).
- DDT-11/DDT-12 — anticipation + interjection (permitted, unflagged, middle-only).
- DDT-13 — completion (closing-seen; coverage is a diagnostic, not a gate).
- DDT-14/DDT-15 — :class:`Divergence` / :class:`RunResult`.
- DDT-17 — ``EventSink`` + sink-driven actors.
- DDT-18 — no synchronised alternation; route-to-other via the bus-wired sink.
- DDT-22 — ``burst >= 1``: an actor always emits at least one proposal; a
  PASS is the no-content proposal. Mutual PASS is the stall terminator.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, turn_content_key
from training.harness.bus import WILDCARD_ROLE, MessageBus
from training.harness.message import Message

# A content key: (role, kline_signature, kline_nodes_tuple, significance).
ContentKey = tuple[str, int, tuple[int, ...], int]

# The bus action used for actor emissions: the recipient's handler is the
# recipient actor's ``accept``.
_ACCEPT_ACTION = "accept"

# ── PASS — the no-content proposal (DDT-22) ──────────────────────────────
#
# An actor whose cogitation yields nothing substantive still owes the dialogue a
# proposal (``burst >= 1``). It publishes a PASS: a reserved sentinel signature
# at S1 (the top band). The meaning lives in the signature; S1 only sorts it
# highest. The runner intercepts PASS *before* content matching — it is neither
# coverage, nor divergence, nor a closing — routes it to the other role, and
# watches for two consecutive PASSes (each side passing) as the stall signal.
#
# A reserved bit pattern unlikely to collide with any compiled signature (the
# compiler hashes real content; this is an arbitrary fixed sentinel).
PASS_SIGNATURE: int = 0x504153535F504153  # "PASS_PAS" as bytes, a stable sentinel


def is_pass(event: RationaliseEvent) -> bool:
    """True when ``event`` is a PASS — the no-content proposal (DDT-22).

    Detected by the sentinel signature on the proposal's kline. The runner
    treats a PASS specially: it routes it to the other role but never matches,
    covers, or flags it as divergence. The actor base emits a PASS when its
    :meth:`~training.dialogue.actors.Actor.next_events` yields nothing, so the
    ``burst >= 1`` contract holds for every actor.
    """
    return event.proposal.kline.signature == PASS_SIGNATURE


def pass_event(role: str) -> RationaliseEvent:
    """Build a PASS :class:`RationaliseEvent` for ``role`` (DDT-22).

    A ``{PASS: []}`` kline at S1. ``query`` is the PASS itself (there is no
    inbound value to echo when the actor originates the PASS).
    """
    kv = KValue(KLine(PASS_SIGNATURE, []), SIG_S1)
    return RationaliseEvent(kind="frame", query=kv, proposal=kv, role=role)


# ── EventSink (DDT-17) — the actor's publish target ───────────────────────


@runtime_checkable
class EventSink(Protocol):
    """The publish target an actor holds.

    An actor is constructed with an ``EventSink`` and publishes events to it via
    ``on_event``. The bus-wired sink (:class:`_BusEventSink`) bridges each
    ``on_event`` to a bus ``Message`` addressed to the other role, so the
    actor's published events flow onto the relay without the actor knowing about
    the bus. This is least-coupling: the actor publishes, the sink routes.
    """

    def on_event(self, event: RationaliseEvent) -> None: ...


@runtime_checkable
class Actor(Protocol):
    """A dialogue actor.

    Holds an :class:`EventSink` (injected at construction) and publishes events
    to it. ``accept`` receives an incoming event (or ``None`` for the opening
    seed) and the actor decides whether/when/how-many events to publish via its
    sink — fire-and-forget, one-or-many (``burst >= 1``, DDT-22: an actor with
    nothing substantive publishes a PASS, never zero).

    This is the duck-typed contract the runner types :data:`ActorFactory`
    against; :class:`~training.dialogue.actors.Actor` is the base implementation
    that satisfies it. The runner depends on the contract, not the base class.
    """

    @property
    def role(self) -> str: ...

    def accept(self, event: RationaliseEvent | None) -> None: ...


# Actor factory: the runner builds the bus-wired sink (it owns the bus) and
# constructs the actor via this callable, passing the sink. Only the component
# that owns the bus can build the bus-wired sink, so it builds the actor too.
ActorFactory = Callable[[EventSink], Actor]


# ── Divergence (DDT-14) ───────────────────────────────────────────────────


class Divergence(Exception):  # noqa: N818 - spec names this type
    """A run emission matched neither the closing nor any middle content.

    Raised under ``on_divergence="fail"`` when an emission's
    ``(role, kline, significance)`` matches neither the closing nor any of the
    table's distinct middle contents. Carries the role, the emitted proposal,
    and the uncovered same-role contents at the moment of divergence. It has no
    cursor — coverage is content-keyed, not positional.
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


# ── Result (DDT-15) ───────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Outcome of a dialogue run (spec §Types).

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


# ── The bus-wired sink (enforces route-to-other, DDT-18) ──────────────────


class _BusEventSink:
    """An :class:`EventSink` that publishes each event to the other role.

    The route-to-other rule (DDT-18) is enforced structurally: the actor calls
    ``on_event(event)`` (the event carries the emitter's role on ``event.role``)
    and this sink addresses the bus ``Message`` to the *other* role. The actor
    cannot misroute — it merely publishes to its sink.
    """

    def __init__(self, bus: MessageBus, other_role: str) -> None:
        self._bus = bus
        self._other = other_role

    def on_event(self, event: RationaliseEvent) -> None:
        self._bus.send(
            Message(role=self._other, action=_ACCEPT_ACTION, message=event)
        )


# ── The runner as a MessageBus subscriber (DDT-5..DDT-22) ─────────────────


class Runner:
    """The dialogue run: a bus subscriber + driver (spec §The Runner).

    The harness :class:`MessageBus` is the sink and the relay. The runner:

    - owns a ``MessageBus``;
    - builds a bus-wired :class:`EventSink` per actor (addressed to the other
      role) and constructs each actor with its sink via the actor factory;
    - subscribes itself as a **wildcard handler** for coverage bookkeeping;
    - subscribes each actor's ``accept`` as its role's handler;
    - drives the run by seeding the opening (addressed to the trainer) and
      running ``bus.run()`` on a thread until the closing is seen (complete)
      or both actors pass consecutively (a stall: incomplete).

    Holds **coverage bookkeeping only**. No actor-coupling state. The relay
    lives in the bus; the runner only observes and records.

    Construct via :func:`run`; call :meth:`run` to drive.
    """

    def __init__(
        self,
        decoded: Sequence[DecodedTurn],
        trainer_factory: ActorFactory,
        trainee_factory: ActorFactory,
        *,
        on_divergence: str = "fail",
    ) -> None:
        if on_divergence not in ("fail", "accept"):
            raise ValueError(
                f"on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
            )
        if len(decoded) < 2:
            raise ValueError("a run needs at least an opening and a closing turn")
        self._on_divergence = on_divergence

        # Coverage bookkeeping (DDT-6). Fixed distinct middle set (duplicates
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

        # PASS-stall tracking (DDT-22). ``_last_pass_role`` records the role of
        # the most recent PASS (None when the last emission was substantive).
        # A stall is a PASS from one role followed by a PASS from the *other*
        # role — each side passing in turn — which ends the run incomplete. Two
        # PASSes from the same role are not a stall (that side is waiting).
        self._last_pass_role: str | None = None

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

    # -- the driver (DDT-18, DDT-22) -----------------------------------------

    def run(self) -> RunResult:
        """Drive the run to completion (spec §The Runner).

        Seeds the opening by addressing an ``accept`` message to the trainer
        (``message=None`` signals "you open"), then runs ``bus.run()`` on a
        dedicated thread. The bus exits when the coverage handler calls
        ``bus.stop()`` — which it does on the closing (complete) or on a
        mutual-PASS stall (incomplete). Every ``accept`` yields ≥1 proposal
        (the ``burst >= 1`` contract), so the bus never blocks indefinitely;
        there is no idle timeout.
        """
        self._bus.send(
            Message(role=self._trainer.role, action=_ACCEPT_ACTION, message=None)
        )
        bus_thread = threading.Thread(target=self._bus.run, daemon=True)
        bus_thread.start()
        bus_thread.join()
        if self._thread_exc is not None:
            raise self._thread_exc
        return self.result

    # -- coverage handler (the wildcard subscriber) (DDT-7..DDT-13) ---------

    def _on_emission(self, msg: Message) -> None:
        """Wildcard handler: record coverage on every observed emission."""
        event = msg.message
        if event is None:
            return  # the opening seed, not an emission
        assert isinstance(event, RationaliseEvent)

        # The closing is terminal (DDT-15): once seen, the run is over. The bus
        # dispatches role handlers before wildcards, so a role handler may react
        # to the closing (e.g. a synthesizing trainer ratifying it) and enqueue
        # an emission *before* this wildcard marks closing-seen and stops the
        # bus. Such trailing emissions are post-completion noise — drop them
        # entirely (no recording, no coverage, no divergence, no PASS logic).
        if self._closing_seen:
            return

        # DDT-22: a PASS is the no-content proposal. Intercept it *before* any
        # content matching: it is neither coverage, nor divergence, nor a
        # closing. It routes to the other role (the bus-wired sink already
        # addressed it). A stall is **each side passing in turn**: a PASS from
        # one role followed by a PASS from the *other* role. Two PASSes from
        # the same role are not a stall — that side is simply waiting (cogitation
        # repeatedly yields nothing while the other side still has content),
        # and the dialogue continues.
        if is_pass(event):
            self._events.append(event)
            role = event.role or "?"
            if self._last_pass_role is not None and self._last_pass_role != role:
                # Each side passed in turn: the dialogue has nothing more to
                # say. Stop the bus; ``complete`` stays False (closing unseen).
                self._bus.stop()
                return
            self._last_pass_role = role
            return
        self._last_pass_role = None

        self._events.append(event)
        key = self._event_key(event)

        # The opening is consumed positionally (DDT-3): neither coverage nor
        # divergence. The trainer publishes it as its first turn; skip it here.
        if key == self._opening_key and not self._closing_seen:
            return

        # DDT-10: closing seen → mark and stop the bus (terminal).
        if key == self._closing_key:
            self._closing_seen = True
            self._bus.stop()
            return

        # DDT-7/DDT-8: a content present in the distinct middle marks it covered
        # (idempotent — re-emitting covered content is not divergence).
        if key in self._distinct_middle:
            self._covered.add(key)
            return

        # DDT-9: divergence — present nowhere in the table.
        if self._on_divergence == "fail":
            exc = Divergence(
                role=event.role or "?",
                emitted=event.proposal,
                unconsumed=tuple(self._uncovered_rows_for_role(event.role)),
            )
            self._thread_exc = exc
            self._bus.stop()
            return
        self._unmatched.append(event)

    # -- actor handler adapter ----------------------------------------------

    def _make_handler(self, actor: Actor):
        """Adapt an actor's ``accept`` to the bus's ``(msg) -> None`` handler."""

        def handler(msg: Message) -> None:
            actor.accept(msg.message)  # RationaliseEvent | None

        return handler

    # -- result + diagnostics -----------------------------------------------

    @property
    def complete(self) -> bool:
        """``closing-seen`` (DDT-13)."""
        return self._closing_seen

    @property
    def covered(self) -> bool:
        """True when every distinct middle content has been seen (efficiency)."""
        return self._distinct_middle <= self._covered

    @property
    def result(self) -> RunResult:
        """The current :class:`RunResult` snapshot (DDT-15)."""
        return RunResult(
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


def run(
    decoded: Sequence[DecodedTurn],
    trainer_factory: ActorFactory,
    trainee_factory: ActorFactory,
    *,
    on_divergence: str = "fail",
) -> Runner:
    """Construct a :class:`Runner` for ``decoded`` (spec §The Runner).

    ``trainer_factory`` / ``trainee_factory`` are callables ``(sink) -> Actor``:
    the runner builds the bus-wired sink (it owns the bus) and constructs each
    actor with its sink. This makes any actor drop-in: the actor author writes
    only the publish-to-sink logic; the runner handles the bus.

    The caller calls :meth:`Runner.run` to drive.
    """
    return Runner(
        decoded,
        trainer_factory,
        trainee_factory,
        on_divergence=on_divergence,
    )
