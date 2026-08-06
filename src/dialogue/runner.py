"""Dialogue runner — a coverage-tracking wildcard subscriber over the harness ``MessageBus``.

A thin driver opens a run (delivers the first row to the opposite role)
and runs the bus until a terminal condition (close observed / coverage
exhausted / mutual PASS)."""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.significance import SIG_S1

# A burst: the events one actor publishes in a single ``accept`` reply, and
# the list it receives as the other role's reply.
Burst = list[RationaliseEvent]
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from dialogue.decoder import DecodedTurn, turn_content_key
from training.harness.bus import WILDCARD_ROLE, MessageBus
from training.harness.message import Message

ContentKey = tuple[str, int, tuple[int, ...], int]
GroundingKey = tuple[int, tuple[int, ...], int]

_ACCEPT_ACTION = "accept"

# PASS — the no-content proposal. A reserved bit pattern unlikely to collide
# with any compiled signature; two consecutive PASSes (one per role) terminate.
PASS_SIGNATURE: int = 0x504153535F504153


def is_pass(event: RationaliseEvent) -> bool:
    """True when ``event`` is a PASS — the no-content proposal."""
    return event.proposal.kline.signature == PASS_SIGNATURE


def pass_event(role: str) -> RationaliseEvent:
    """Build a PASS :class:`RationaliseEvent` for ``role``."""
    kv = KValue(KLine(PASS_SIGNATURE, []), SIG_S1)
    return RationaliseEvent(kind="frame", query=kv, proposal=kv, role=role)


# ── EventSink — the actor's publish target ───────────────────────────────


@runtime_checkable
class EventSink(Protocol):
    """The publish target an actor holds (bridged to the bus by ``_BusEventSink``)."""

    def on_burst(self, events: list[RationaliseEvent]) -> None: ...


@runtime_checkable
class Actor(Protocol):
    """A dialogue actor."""

    @property
    def role(self) -> str: ...

    def accept(self, incoming: list[RationaliseEvent]) -> None: ...


# ``(sink) -> Actor``. The runner builds the bus-wired sink and constructs the
# actor with it.
ActorFactory = Callable[[EventSink], Actor]


# ── Divergence ────────────────────────────────────────────────────────────


class Divergence(Exception):  # noqa: N818 - spec names this type
    """A run emission the authored table did not authorise.

    ``reason`` is ``"unmatched"`` (matches no close or coverage content) or
    ``"exhausted"`` (coverage content whose authored copies are all consumed).
    """

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


class GroundingDivergence(Exception):  # noqa: N818
    """A K grounding the script's ``events`` did not authorise (white-box
    counterpart to :class:`Divergence`)."""

    reason: str

    def __init__(
        self,
        grounded: KValue,
        unconsumed: tuple[DecodedTurn, ...],
        *,
        reason: str = "unmatched",
        last_coverage_event: RationaliseEvent | None = None,
    ) -> None:
        self.grounded = grounded
        self.unconsumed = unconsumed
        self.reason = reason
        self.last_coverage_event = last_coverage_event
        if reason == "exhausted":
            msg = (
                f"grounding divergence: grounded sig={grounded.significance:#x} "
                f"exhausts its expected budget "
                f"(every authored copy already consumed)"
            )
        elif reason == "missing":
            msg = (
                f"grounding divergence: asserted grounding "
                f"sig={grounded.significance:#x} was never observed "
                f"({len(unconsumed)} asserted groundings unobserved)"
            )
        else:
            msg = (
                f"grounding divergence: grounded sig={grounded.significance:#x} "
                f"matches no expected grounding "
                f"({len(unconsumed)} expected groundings remain unconsumed)"
            )
        super().__init__(msg)


# ── Result ───────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Arrival-ordered events, divergences (accept-mode), and displacement
    (``uncovered``/``uncovered_groundings``: rows/groundings never emitted)."""

    events: list[RationaliseEvent] = field(default_factory=list)
    unmatched: list[RationaliseEvent] = field(default_factory=list)
    uncovered: list[DecodedTurn] = field(default_factory=list)
    last_coverage_event: RationaliseEvent | None = None
    unmatched_groundings: list[KValue] = field(default_factory=list)
    uncovered_groundings: list[DecodedTurn] = field(default_factory=list)


# ── The bus-wired sink ───────────────────────────────────────────────────


class _BusEventSink:
    """An :class:`EventSink` that publishes a whole burst to the other role."""

    def __init__(self, bus: MessageBus, other_role: str) -> None:
        self._bus = bus
        self._other = other_role

    def on_burst(self, events: list[RationaliseEvent]) -> None:
        self._bus.send(
            Message(role=self._other, action=_ACCEPT_ACTION, message=events)
        )


# ── The runner as a MessageBus subscriber ─────────────────────────────────


class Runner:
    """The dialogue run: a bus subscriber + driver. Construct via :func:`run`."""

    def __init__(
        self,
        decoded: Sequence[DecodedTurn],
        trainer_factory: ActorFactory | None,
        trainee_factory: ActorFactory | None,
        *,
        expected_groundings: Sequence[DecodedTurn] = (),
        on_divergence: str = "fail",
        trainer: Actor | None = None,
        trainee: Actor | None = None,
    ) -> None:
        if on_divergence not in ("fail", "accept"):
            raise ValueError(
                f"on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
            )
        if len(decoded) < 2:
            raise ValueError("a run needs at least two turns (an opening and a close)")
        self._on_divergence = on_divergence

        # Close = the ``close:true`` turn (else the last row); everything else
        # is the coverage budget (a per-key multiplicity).
        close_idx = next((i for i, t in enumerate(decoded) if t.close), len(decoded) - 1)
        self._closing_key: ContentKey = turn_content_key(decoded[close_idx])
        coverage = [t for i, t in enumerate(decoded) if i != close_idx]
        self._coverage_budget: Counter[ContentKey] = Counter(
            turn_content_key(t) for t in coverage
        )
        # Expected groundings: a subset check (extra K groundings not policed).
        self._expected_groundings: dict[GroundingKey, DecodedTurn] = {
            _grounding_key(t): t for t in expected_groundings
        }
        self._observed_groundings: set[GroundingKey] = set()
        self._consumed: Counter[ContentKey] = Counter()
        self._closed: bool = False
        self._events: list[RationaliseEvent] = []
        self._unmatched: list[RationaliseEvent] = []
        self._thread_exc: BaseException | None = None
        self._last_coverage_event: RationaliseEvent | None = None
        self._last_pass_role: str | None = None
        self._unmatched_groundings: list[KValue] = []

        # Bus-wired sinks + actors. Pre-built actors (run sequencing: shared
        # instances across runs) are re-bound to this run's bus.
        self._bus = MessageBus()
        trainer_sink = _BusEventSink(self._bus, "K")
        trainee_sink = _BusEventSink(self._bus, "T")
        if trainer is not None:
            trainer._bind_sink(trainer_sink)  # noqa: SLF001
            self._trainer = trainer
        else:
            self._trainer = trainer_factory(trainer_sink)
        if trainee is not None:
            trainee._bind_sink(trainee_sink)  # noqa: SLF001
            self._trainee = trainee
        else:
            self._trainee = trainee_factory(trainee_sink)
        if self._trainer.role == self._trainee.role:
            raise ValueError(
                f"trainer and trainee must have different roles, got {self._trainer.role!r}"
            )
        self._trainee_observable = hasattr(self._trainee, "drain_observations")
        self._bus.subscribe(WILDCARD_ROLE, self._on_emission)
        self._bus.subscribe(self._trainer.role, self._make_handler(self._trainer))
        self._bus.subscribe(self._trainee.role, self._make_handler(self._trainee))

        # The opening: the maximal same-role prefix, delivered by the runner
        # to the opposite role.
        opener_role = decoded[0].role
        opening_turns: list[DecodedTurn] = []
        for t in decoded:
            if t.role != opener_role:
                break
            opening_turns.append(t)
        self._opening_events = [
            RationaliseEvent(
                kind="frame", query=t.value, proposal=t.value, role=t.role,
            )
            for t in opening_turns
        ]
        self._opening_recipient = (
            self._trainee.role if opener_role == self._trainer.role
            else self._trainer.role
        )

    # -- the driver ---------------------------------------------------------

    def run(self) -> RunResult:
        """Deliver the opening to the opposite role and drive the bus to a
        terminal condition."""
        self._bus.send(
            Message(
                role=self._opening_recipient,
                action=_ACCEPT_ACTION,
                message=list(self._opening_events),
            )
        )
        bus_thread = threading.Thread(target=self._bus.run, daemon=True)
        bus_thread.start()
        bus_thread.join()
        if self._thread_exc is not None:
            raise self._thread_exc
        if self._trainee_observable:
            self._check_grounding_assertions()
            if self._thread_exc is not None:
                raise self._thread_exc
        return self.result

    # -- coverage handler (the wildcard subscriber) -------------------------

    def _on_emission(self, msg: Message) -> None:
        """Wildcard handler: coverage/divergence/PASS bookkeeping per emission."""
        burst = msg.message
        for event in burst:
            self._observe(event)
            if self._closed:
                return
        # Coverage exhaustion (checked at the burst boundary so an over-budget
        # emission inside the burst surfaces as divergence first).
        if not self._closed and self._consumed == self._coverage_budget:
            self._closed = True
            self._bus.stop()

    def _observe(self, event: RationaliseEvent) -> None:
        """Apply coverage / PASS / divergence bookkeeping to one emission."""
        if self._closed:
            return  # drop trailing emissions

        # PASS: intercepted before matching. Two consecutive (one per role) terminate.
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

        # Coverage row with copies remaining: consume one.
        budget = self._coverage_budget.get(key, 0)
        if self._consumed[key] < budget:
            self._consumed[key] += 1
            self._last_coverage_event = event
            return

        # The close (excluded from the budget) terminates on first observation.
        if key == self._closing_key:
            self._closed = True
            self._bus.stop()
            return

        # Divergence: ``exhausted`` (budget spent) or ``unmatched`` (present nowhere).
        reason = "exhausted" if budget > 0 else "unmatched"
        self._record_divergence(event, reason)
        if self._on_divergence == "fail":
            self._closed = True
            self._bus.stop()

    def _record_divergence(self, event: RationaliseEvent, reason: str) -> None:
        """Record the :class:`Divergence` for ``event`` per the run's policy."""
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

    def _observe_grounding(self, grounded: KValue) -> None:
        """Record a K grounding observation (checked at run end)."""
        self._observed_groundings.add(_grounding_key_from_value(grounded))

    # -- actor handler adapter ----------------------------------------------

    def _make_handler(self, actor: Actor):
        """Adapt an actor's ``accept`` to the bus handler; drain a trainee's
        groundings afterwards."""

        def handler(msg: Message) -> None:
            actor.accept(msg.message)  # list[RationaliseEvent] (empty = seed)
            if hasattr(actor, "drain_observations"):
                for grounded in actor.drain_observations():
                    self._observe_grounding(grounded)
                    if self._closed:
                        return

        return handler

    # -- result + displacement --------------------------------------------

    @property
    def trainer(self) -> Actor:
        return self._trainer

    @property
    def trainee(self) -> Actor:
        """Exposed for post-run inspection (e.g. a rationalising trainee's state)."""
        return self._trainee

    @property
    def result(self) -> RunResult:
        """The current :class:`RunResult` snapshot."""
        return RunResult(
            events=list(self._events),
            unmatched=list(self._unmatched),
            uncovered=list(self._uncovered_rows()),
            last_coverage_event=self._last_coverage_event,
            unmatched_groundings=list(self._unmatched_groundings),
            uncovered_groundings=list(self._uncovered_groundings()),
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
        # One placeholder per remaining authored copy.
        out: list[DecodedTurn] = []
        for k in sorted(self._coverage_budget, key=_key_sort):
            remaining = self._coverage_budget[k] - self._consumed[k]
            out.extend([_placeholder_turn(k)] * max(remaining, 0))
        return out

    def _uncovered_groundings(self) -> list[DecodedTurn]:
        """Asserted groundings never observed; ``[]`` for a non-observable trainee."""
        if not self._trainee_observable:
            return []
        return [
            self._expected_groundings[k]
            for k in sorted(self._expected_groundings, key=_grounding_key_sort)
            if k not in self._observed_groundings
        ]

    def _check_grounding_assertions(self) -> None:
        """Raise/record a :class:`GroundingDivergence` for any asserted grounding
        never observed (subset check)."""
        missing = self._uncovered_groundings()
        if not missing:
            return
        if self._on_divergence == "fail":
            self._thread_exc = GroundingDivergence(
                grounded=missing[0].value,
                unconsumed=tuple(missing),
                reason="missing",
                last_coverage_event=self._last_coverage_event,
            )
        else:
            self._unmatched_groundings.extend(t.value for t in missing)


def _key_sort(k: ContentKey):
    return (k[0], k[1], k[2], k[3])


def _grounding_key(turn: DecodedTurn) -> GroundingKey:
    """The grounding identity of an expected grounding row."""
    return (
        turn.value.kline.signature,
        tuple(turn.value.kline.nodes),
        turn.value.significance,
    )


def _grounding_key_from_value(value: KValue) -> GroundingKey:
    """The grounding identity of an observed K grounding."""
    return (
        value.kline.signature,
        tuple(value.kline.nodes),
        value.significance,
    )


def _grounding_key_sort(k: GroundingKey):
    return (k[0], k[1], k[2])


def _placeholder_turn(k: ContentKey) -> DecodedTurn:
    """Reconstruct a minimal ``DecodedTurn`` from a content key for diagnostics."""
    from typing import cast

    from kalvin.kline import KLine
    from dialogue.decoder import Role

    return DecodedTurn(
        role=cast(Role, k[0]),
        op="?",
        value=KValue(KLine(k[1], list(k[2])), k[3]),
    )


def run(
    decoded: Sequence[DecodedTurn],
    trainer_factory: ActorFactory | None,
    trainee_factory: ActorFactory | None,
    *,
    expected_groundings: Sequence[DecodedTurn] = (),
    on_divergence: str = "fail",
    trainer: Actor | None = None,
    trainee: Actor | None = None,
) -> Runner:
    """Construct a :class:`Runner`. Pass factories, or pre-built ``trainer``/
    ``trainee`` for run sequencing (shared instances re-bound per run)."""
    return Runner(
        decoded,
        trainer_factory,
        trainee_factory,
        expected_groundings=expected_groundings,
        on_divergence=on_divergence,
        trainer=trainer,
        trainee=trainee,
    )
