"""Dialogue runner — a coverage-tracking subscriber over the harness
``MessageBus`` (spec ``@specs/dialogue-driven-training.md``).

The harness :class:`~training.harness.bus.MessageBus` is the **sink and the
relay**; this module's runner (:class:`Runner`) is a **coverage-tracking
wildcard subscriber** plus a thin driver that seeds the opening and runs the bus
until a terminal condition. The actors the runner drives live in
:mod:`training.dialogue.actors`.

An actor takes an :class:`EventSink` at construction and publishes a **burst**
of events to it via ``on_burst``. The runner builds a bus-wired sink per actor
(the sink bridges ``on_burst`` to a single bus ``Message`` — carrying the whole
burst as its payload — addressed to the other role), so any actor is drop-in:
it publishes to its sink, the sink routes onto the bus. Because the bus payload
is ``Any``, a burst rides as one ``Message``; it never serialises across many.

The dialogue is messy and real: **no synchronised alternation** (an actor may
publish one-or-many per incoming), and **anticipation** and **interjection**
are first-class and unflagged. Agents must rationalise and cogitate to make
sense of the stream — the point of Kalvin.

Every ``accept`` yields at least one proposal (the ``burst >= 1`` contract,
§Actor contract): an actor with nothing substantive to say publishes a **PASS**
— a sentinel proposal (``PASS_SIGNATURE`` at S1). The runner intercepts PASS
before matching: it routes to the other role, and **two consecutive PASSes**
(each side passing in turn) is a terminal condition (the actors have nothing
more to say). A run ends mechanically on the close content being seen, the
coverage set being exhausted, or mutual PASS. The **displacement** (uncovered
coverage rows) records how much of the authored exchange the actors traversed.

Spec mapping
------------
- The runner is a bus subscriber holding coverage bookkeeping; the bus is the
  sink/relay. It tracks coverage and immediate divergence.
- Matching (content presence, coverage budget, duplicate-key exhaustion,
  immediate divergence, close).
- Anticipation + interjection (permitted, unflagged).
- The close is de-positional: any agent may emit it at any time; a table has no
  start-middle-end structure, only a coverage set and one close turn.
- ``RunResult`` records the arrival log, immediate divergences, and displacement.
- ``EventSink`` + sink-driven actors.
- No synchronised alternation; route-to-other via the bus-wired sink.
- ``burst >= 1``: an actor always emits at least one proposal; a PASS is the
  no-content proposal. Mutual PASS is a terminal condition.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1

# A burst: the list of events one actor publishes in a single ``accept`` reply,
# and the list it receives as the other role's reply. A burst is the bus
# payload (``Message.message``) — the payload-blind bus carries it whole, so a
# reply never serialises across multiple messages. An empty burst is the
# opening seed (no prior turn).
Burst = list[RationaliseEvent]
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
# highest. The runner intercepts PASS *before* content matching — it routes
# it to the other role, and watches for two consecutive PASSes (each side
# passing) as the stall signal.
#
# A reserved bit pattern unlikely to collide with any compiled signature (the
# compiler hashes real content; this is an arbitrary fixed sentinel).
PASS_SIGNATURE: int = 0x504153535F504153  # "PASS_PAS" as bytes, a stable sentinel


def is_pass(event: RationaliseEvent) -> bool:
    """True when ``event`` is a PASS — the no-content proposal (DDT-22).

    Detected by the sentinel signature on the proposal's kline. The runner
    treats a PASS specially: it routes it to the other role but does not match,
    cover, or flag it as divergence. The actor base emits a PASS when its
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


# ── EventSink (DDT-21) — the actor's publish target ───────────────────────


@runtime_checkable
class EventSink(Protocol):
    """The publish target an actor holds.

    An actor is constructed with an ``EventSink`` and publishes a **burst** of
    events to it via ``on_burst``. The bus-wired sink (:class:`_BusEventSink`)
    bridges each ``on_burst`` to a single bus ``Message`` — carrying the whole
    burst as its payload — addressed to the other role, so the actor's
    published burst flows onto the relay as one message without the actor
    knowing about the bus. This is least-coupling: the actor publishes a
    burst, the sink routes it.
    """

    def on_burst(self, events: list[RationaliseEvent]) -> None: ...


@runtime_checkable
class Actor(Protocol):
    """A dialogue actor.

    Holds an :class:`EventSink` (injected at construction) and publishes a
    **burst** of events to it. ``accept`` receives the incoming burst — the
    other role's whole reply as one list (empty for the opening seed) — and
    the actor decides how-many events to publish via its sink: fire-and-forget,
    one-or-many (``burst >= 1``, DDT-22: an actor with nothing substantive
    publishes a PASS, never zero).

    This is the duck-typed contract the runner types :data:`ActorFactory`
    against; :class:`~training.dialogue.actors.Actor` is the base implementation
    that satisfies it. The runner depends on the contract, not the base class.
    """

    @property
    def role(self) -> str: ...

    def accept(self, incoming: list[RationaliseEvent]) -> None: ...


# Actor factory: the runner builds the bus-wired sink (it owns the bus) and
# constructs the actor via this callable, passing the sink. Only the component
# that owns the bus can build the bus-wired sink, so it builds the actor too.
ActorFactory = Callable[[EventSink], Actor]


# ── Divergence (DDT-14) ───────────────────────────────────────────────────


class Divergence(Exception):  # noqa: N818 - spec names this type
    """A run emission the authored table did not authorise.

    Raised under ``on_divergence="fail"``. Two reasons:

    - ``"unmatched"`` — the emission's ``(role, kline, significance)`` matches
      neither the close nor any coverage content (present nowhere in the
      table).
    - ``"exhausted"`` — **duplicate key exhaustion**: the content *is* in the
      coverage set, but every authored copy has already been consumed. The
      table authorises a fixed count of each content (its multiplicity in the
      coverage rows); emitting more copies than authored is divergence.

    Carries the role, the emitted proposal, the reason, the unconsumed
    same-role contents at the moment of divergence, and the last event that
    successfully consumed a coverage allowance (the last healthy point before
    divergence). It has no cursor — coverage is content-keyed, not positional.
    """

    #: ``"unmatched"`` (present nowhere) or ``"exhausted"`` (duplicate key
    #: exhaustion — more copies emitted than the table authored).
    reason: str

    def __init__(
        self,
        role: str,
        emitted: KValue,
        unconsumed: tuple[DecodedTurn, ...],
        *,
        reason: str = "unmatched",
        last_coverage_event: RationaliseEvent | None = None,
    ) -> None:
        self.role = role
        self.emitted = emitted
        self.unconsumed = unconsumed
        self.reason = reason
        self.last_coverage_event = last_coverage_event
        if reason == "exhausted":
            msg = (
                f"{role} divergence: emitted sig={emitted.significance:#x} "
                f"exhausts its coverage budget "
                f"(every authored copy already consumed; "
                f"{len(unconsumed)} same-role contents remain uncovered)"
            )
        else:
            msg = (
                f"{role} divergence: emitted sig={emitted.significance:#x} "
                f"matches no closing or middle content "
                f"({len(unconsumed)} uncovered same-role contents)"
            )
        super().__init__(msg)


# ── Result ───────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """The record of a dialogue run (spec §Types).

    Records what happened — the arrival-ordered emission log, the immediate
    divergences (emissions matching nothing, under ``on_divergence="accept"``),
    and the **displacement**: the coverage rows never emitted (how far the
    realized dialogue fell short of the authored whole-exchange coverage). A run
    ends mechanically (close content seen, coverage exhausted, or mutual PASS);
    the displacement is the signal of how much of the exchange the actors
    traversed.

    ``last_coverage_event`` is the most recent emission that successfully
    consumed a coverage allowance — the last healthy point before any
    divergence (or ``None`` when no coverage content was ever matched). It is
    the anchor for divergence diagnostics: it says where the run was still on
    the authored exchange.
    """

    events: list[RationaliseEvent] = field(default_factory=list)
    unmatched: list[RationaliseEvent] = field(default_factory=list)
    uncovered: list[DecodedTurn] = field(default_factory=list)
    last_coverage_event: RationaliseEvent | None = None


# ── The bus-wired sink (enforces route-to-other, DDT-18) ──────────────────


class _BusEventSink:
    """An :class:`EventSink` that publishes a whole burst to the other role.

    The route-to-other rule (DDT-18) is enforced structurally: the actor calls
    ``on_burst(events)`` and this sink addresses one bus ``Message`` to the
    *other* role, carrying the burst as its payload. The payload-blind bus
    delivers it whole — a reply never serialises across multiple messages.
    """

    def __init__(self, bus: MessageBus, other_role: str) -> None:
        self._bus = bus
        self._other = other_role

    def on_burst(self, events: list[RationaliseEvent]) -> None:
        self._bus.send(
            Message(role=self._other, action=_ACCEPT_ACTION, message=events)
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
    - drives the run by seeding the trainer (the opening seed) and running
      ``bus.run()`` on a thread until a terminal condition: the close content
      is seen, the coverage set is exhausted, or both actors pass in turn.

    The runner holds coverage bookkeeping only and tracks immediate divergence
    and displacement. No actor-coupling state. The relay lives in the bus; the
    runner observes and records.

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
            raise ValueError("a run needs at least two turns (a coverage set and a close)")
        self._on_divergence = on_divergence

        # The close is de-positional: the ``close:true`` turn if any, else the
        # last row. It may be emitted by any agent at any time to end the run.
        # Everything else is the coverage set. The coverage set is a **budget**
        # per content key: a content's multiplicity in the coverage rows is the
        # number of times the authored exchange permits it. Emitting more
        # copies than authored is duplicate-key exhaustion → divergence.
        close_idx = next((i for i, t in enumerate(decoded) if t.close), len(decoded) - 1)
        self._closing_key: ContentKey = turn_content_key(decoded[close_idx])
        coverage = [t for i, t in enumerate(decoded) if i != close_idx]
        self._coverage_budget: Counter[ContentKey] = Counter(
            turn_content_key(t) for t in coverage
        )
        self._consumed: Counter[ContentKey] = Counter()
        self._closed: bool = False
        self._events: list[RationaliseEvent] = []
        self._unmatched: list[RationaliseEvent] = []
        self._thread_exc: BaseException | None = None
        # The last emission that successfully consumed a coverage allowance —
        # the last healthy point before any divergence. Surfaced on
        # ``RunResult.last_coverage_event`` and on ``Divergence`` for diagnostics.
        self._last_coverage_event: RationaliseEvent | None = None

        # PASS tracking. ``_last_pass_role`` records the role of the most recent
        # PASS (None when the last emission was substantive). A PASS from one
        # role followed by a PASS from the other is a terminal condition (the
        # actors have nothing more to say). Two PASSes from the same role are
        # not (that side is waiting while the other still has content).
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

    # -- the driver ---------------------------------------------------------

    def run(self) -> RunResult:
        """Drive the run (spec §The Runner).

        Seeds the trainer (addressing an ``accept`` message with an empty
        burst — ``message=[]`` — the opening seed), then runs ``bus.run()`` on
        a dedicated thread. The bus exits when the coverage handler calls
        ``bus.stop()`` — on the close content being seen, the coverage set
        being exhausted, or a mutual PASS. Every ``accept`` yields ≥1 proposal
        (``burst >= 1``), so the bus never blocks indefinitely.
        """
        self._bus.send(
            Message(role=self._trainer.role, action=_ACCEPT_ACTION, message=[])
        )
        bus_thread = threading.Thread(target=self._bus.run, daemon=True)
        bus_thread.start()
        bus_thread.join()
        if self._thread_exc is not None:
            raise self._thread_exc
        return self.result

    # -- coverage handler (the wildcard subscriber) -------------------------

    def _on_emission(self, msg: Message) -> None:
        """Wildcard handler: track coverage and divergence on every emission.

        ``msg.message`` is a burst (a ``list[RationaliseEvent]``): the other
        role's whole reply carried as one bus payload. The bus delivers it
        whole, so this handler unpacks the burst and applies the per-event
        coverage / PASS / divergence logic to each event in arrival order.

        Coverage-exhaustion is checked at the **burst boundary** (after every
        event in the burst has been matched), not mid-burst: a single burst may
        legitimately exhaust a key's budget and then over-emit it (the third
        copy is duplicate-key exhaustion, §Matching), and that over-emission
        must be observable as divergence rather than swallowed by an early
        exhaustion-termination. The close, by contrast, terminates immediately
        (it may land mid-burst; later events in the same burst are dropped as
        trailing-after-terminal).
        """
        burst = msg.message
        if not burst:
            return  # the opening seed, not an emission
        for event in burst:
            self._observe(event)
            if self._closed:
                return
        # Entry exhaustion: every authored coverage copy has been consumed.
        # Checked at the burst boundary so an over-budget emission inside the
        # burst is surfaced as divergence first.
        if not self._closed and self._consumed == self._coverage_budget:
            self._closed = True
            self._bus.stop()

    def _observe(self, event: RationaliseEvent) -> None:
        """Apply coverage / PASS / divergence bookkeeping to one emission."""
        assert isinstance(event, RationaliseEvent)

        # The run is closed: drop everything after. The bus dispatches role
        # handlers before wildcards, so a role handler may react to a terminal
        # emission and enqueue another *before* this wildcard marks the run
        # closed and stops the bus. Such trailing emissions are noise.
        if self._closed:
            return

        # A PASS is the no-content proposal. Intercept it *before* any content
        # matching: it routes to the other role (the bus-wired sink already
        # addressed it). A PASS from one role followed by a PASS from the other
        # is terminal (the actors have nothing more to say). Two PASSes from the
        # same role are not (that side is waiting while the other still has
        # content).
        if is_pass(event):
            self._events.append(event)
            role = event.role or "?"
            if self._last_pass_role is not None and self._last_pass_role != role:
                self._closed = True
                self._bus.stop()
                return
            self._last_pass_role = role
            return
        self._last_pass_role = None

        self._events.append(event)
        key = self._event_key(event)

        # The close content, emitted by any agent at any time, ends the run.
        if key == self._closing_key:
            self._closed = True
            self._bus.stop()
            return

        # A content present in the coverage set consumes one authored copy.
        # The coverage set is a per-key budget (its multiplicity in the table):
        # emitting more copies than authored is duplicate-key exhaustion →
        # divergence. Coverage exhaustion (every budget spent) is checked at
        # the burst boundary by ``_on_emission``, not here, so an over-budget
        # emission arriving later in the same burst is surfaced as divergence.
        budget = self._coverage_budget.get(key, 0)
        if self._consumed[key] < budget:
            self._consumed[key] += 1
            self._last_coverage_event = event
            return

        # Immediate divergence. Two reasons, distinguished for diagnostics:
        # ``"exhausted"`` — the content is in the table but every authored
        # copy has been consumed (duplicate-key exhaustion); ``"unmatched"`` —
        # the content is present nowhere in the table. Either way the run stops
        # immediately: a divergent emission means the actor has left the
        # authored exchange, so there is no further authored progress to make.
        # ``on_divergence`` only governs how the divergence is reported — raise
        # (``fail``) or record (``accept``) — not whether to stop.
        reason = "exhausted" if budget > 0 else "unmatched"
        self._record_divergence(event, reason)
        self._closed = True
        self._bus.stop()

    def _record_divergence(self, event: RationaliseEvent, reason: str) -> None:
        """Record the :class:`Divergence` for ``event`` per the run's policy.

        Under ``on_divergence="fail"`` the exception is stashed on
        ``_thread_exc`` (re-raised by :meth:`run` on the caller's thread).
        Under ``"accept"`` the divergent emission is appended to
        ``_unmatched``. Either way the caller stops the run immediately after.
        """
        exc = Divergence(
            role=event.role or "?",
            emitted=event.proposal,
            unconsumed=tuple(self._uncovered_rows_for_role(event.role)),
            reason=reason,
            last_coverage_event=self._last_coverage_event,
        )
        if self._on_divergence == "fail":
            self._thread_exc = exc
        else:
            self._unmatched.append(event)

    # -- actor handler adapter ----------------------------------------------

    def _make_handler(self, actor: Actor):
        """Adapt an actor's ``accept`` to the bus's ``(msg) -> None`` handler."""

        def handler(msg: Message) -> None:
            actor.accept(msg.message)  # list[RationaliseEvent] (empty = seed)

        return handler

    # -- result + displacement --------------------------------------------

    @property
    def result(self) -> RunResult:
        """The current :class:`RunResult` snapshot (spec §Types).

        The arrival log, immediate divergences, the displacement (coverage
        rows never emitted), and the last event that consumed a coverage
        allowance.
        """
        return RunResult(
            events=list(self._events),
            unmatched=list(self._unmatched),
            uncovered=list(self._uncovered_rows()),
            last_coverage_event=self._last_coverage_event,
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
        return [
            turn
            for turn in self._uncovered_rows()
            if turn.role == r
        ]

    def _uncovered_rows(self) -> list[DecodedTurn]:
        # A coverage key is uncovered to the degree its budget is unconsumed:
        # one placeholder per remaining authored copy, so the displacement
        # count reflects authored rows not traversed (not just distinct keys).
        out: list[DecodedTurn] = []
        for k in sorted(self._coverage_budget, key=_key_sort):
            remaining = self._coverage_budget[k] - self._consumed[k]
            out.extend([_placeholder_turn(k)] * max(remaining, 0))
        return out


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
