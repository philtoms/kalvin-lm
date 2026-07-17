"""Dialogue runner — a coverage-tracking subscriber over the harness ``MessageBus``.

The :class:`Runner` is a coverage-tracking wildcard subscriber plus a thin
driver: it seeds the opening and runs the bus until a terminal condition. The
actors it drives live in :mod:`training.dialogue.actors`.

An actor takes an :class:`EventSink` at construction and publishes a **burst**
of events to it via ``on_burst``. The runner builds a bus-wired sink per actor
(the sink bridges ``on_burst`` to a single bus ``Message`` addressed to the
other role), so any actor is drop-in.

Every ``accept`` yields at least one proposal (``burst >= 1``): an actor with
nothing substantive publishes a **PASS** — a sentinel proposal. The runner
intercepts PASS before matching; two consecutive PASSes (each side passing) is
terminal. A run ends on the close content being seen, the coverage set being
exhausted, or mutual PASS. The **displacement** (uncovered coverage rows)
ecords how much of the authored exchange the actors traversed.
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
# and the list it receives as the other role's reply. An empty burst is the
# opening seed.
Burst = list[RationaliseEvent]
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, turn_content_key
from training.harness.bus import WILDCARD_ROLE, MessageBus
from training.harness.message import Message

# A content key: (role, kline_signature, kline_nodes_tuple, significance).
ContentKey = tuple[str, int, tuple[int, ...], int]

# A grounding key: (kline_signature, kline_nodes_tuple, significance). K's
# internal S1 grounding events are verified white-box against the script's
# ``events``; role is always K and not part of the key.
GroundingKey = tuple[int, tuple[int, ...], int]

# The bus action used for actor emissions: the recipient's handler is the
# recipient actor's ``accept``.
_ACCEPT_ACTION = "accept"

# ── PASS — the no-content proposal ────────────────────────────────────────
#
# An actor whose cogitation yields nothing substantive still owes the dialogue
# a proposal (``burst >= 1``). It publishes a PASS: a reserved sentinel
# signature at S1. The runner intercepts PASS *before* content matching and
# watches for two consecutive PASSes (each side passing) as the stall signal.
#
# A reserved bit pattern unlikely to collide with any compiled signature.
PASS_SIGNATURE: int = 0x504153535F504153  # "PASS_PAS" as bytes, a stable sentinel


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
    """The publish target an actor holds.

    The bus-wired sink (:class:`_BusEventSink`) bridges each ``on_burst`` to
    a single bus ``Message`` addressed to the other role, so the actor's
    published burst flows onto the relay without the actor knowing about the bus.
    """

    def on_burst(self, events: list[RationaliseEvent]) -> None: ...


@runtime_checkable
class Actor(Protocol):
    """A dialogue actor.

    Holds an :class:`EventSink` (injected at construction) and publishes a
    **burst** of events to it. ``accept`` receives the incoming burst — the
    other role's whole reply as one list (empty for the opening seed) — and
    the actor decides how-many events to publish via its sink: fire-and-forget,
    one-or-many (``burst >= 1``: an actor with nothing substantive publishes a
    PASS, never zero).
    """

    @property
    def role(self) -> str: ...

    def accept(self, incoming: list[RationaliseEvent]) -> None: ...


# Actor factory: the runner builds the bus-wired sink (it owns the bus) and
# constructs the actor via this callable, passing the sink. Only the component
# that owns the bus can build the bus-wired sink, so it builds the actor too.
ActorFactory = Callable[[EventSink], Actor]


# ── Divergence ────────────────────────────────────────────────────────────


class Divergence(Exception):  # noqa: N818 - spec names this type
    """A run emission the authored table did not authorise.

    Raised under ``on_divergence="fail"``. ``reason`` is ``"unmatched"`` (the
    emission matches neither the close nor any coverage content) or
    ``"exhausted"`` (the content is in the coverage set but every authored
    copy has already been consumed).
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


class GroundingDivergence(Exception):  # noqa: N818
    """A K grounding observation the script's ``events`` did not authorise.

    White-box counterpart to :class:`Divergence`. Raised under
    ``on_divergence="fail"`` when K grounds a kline not expected by the
    script (``reason="unmatched"``) or grounds more copies of an expected
    grounding than authored (``reason="exhausted"``).
    """

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
    """The record of a dialogue run.

    ``events`` is arrival-ordered; ``unmatched`` holds immediate divergences
    (accept-mode only); ``uncovered`` is the **displacement** — coverage rows
    never emitted (one placeholder per unconsumed authored copy).
    ``last_coverage_event`` is the last emission that consumed a coverage
    allowance — the last healthy point before any divergence.
    ``unmatched_groundings`` holds grounding divergences (accept-mode only);
    ``uncovered_groundings`` is the grounding displacement — expected
    groundings never observed.
    """

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
    """The dialogue run: a bus subscriber + driver.

    Owns a ``MessageBus``; builds a bus-wired :class:`EventSink` per actor and
    constructs each actor with its sink; subscribes itself as a wildcard
    handler for coverage bookkeeping and each actor's ``accept`` as its role's
    handler; then seeds the trainer and runs ``bus.run()`` on a thread until a
    terminal condition. The runner holds coverage bookkeeping only; the relay
    lives in the bus.

    Construct via :func:`run`; call :meth:`run` to drive.
    """

    def __init__(
        self,
        decoded: Sequence[DecodedTurn],
        trainer_factory: ActorFactory,
        trainee_factory: ActorFactory,
        *,
        expected_groundings: Sequence[DecodedTurn] = (),
        on_divergence: str = "fail",
    ) -> None:
        if on_divergence not in ("fail", "accept"):
            raise ValueError(
                f"on_divergence must be 'fail' or 'accept', got {on_divergence!r}"
            )
        if len(decoded) < 2:
            raise ValueError("a run needs at least two turns (a coverage set and a close)")
        self._on_divergence = on_divergence

        # The close is the ``close:true`` turn if any, else the last row;
        # everything else is the coverage set. The coverage set is a per-key
        # budget (a content's multiplicity in the coverage rows).
        close_idx = next((i for i, t in enumerate(decoded) if t.close), len(decoded) - 1)
        self._closing_key: ContentKey = turn_content_key(decoded[close_idx])
        coverage = [t for i, t in enumerate(decoded) if i != close_idx]
        self._coverage_budget: Counter[ContentKey] = Counter(
            turn_content_key(t) for t in coverage
        )
        # Expected groundings: a set of targeted assertions (subset check).
        # The runner verifies each asserted grounding is observed at least
        # once across the run; extra K groundings are not policed (model B).
        self._expected_groundings: dict[GroundingKey, DecodedTurn] = {
            _grounding_key(t): t for t in expected_groundings
        }
        self._observed_groundings: set[GroundingKey] = set()
        self._consumed: Counter[ContentKey] = Counter()
        self._closed: bool = False
        self._events: list[RationaliseEvent] = []
        self._unmatched: list[RationaliseEvent] = []
        self._thread_exc: BaseException | None = None
        # The last emission that consumed a coverage allowance — the last
        # healthy point before any divergence.
        self._last_coverage_event: RationaliseEvent | None = None

        # PASS tracking: the role of the most recent PASS (None when the last
        # emission was substantive). A PASS from one role followed by a PASS
        # from the other is terminal.
        self._last_pass_role: str | None = None

        # Grounding-divergence accumulations (accept-mode).
        self._unmatched_groundings: list[KValue] = []

        # Build the bus-wired sinks, construct the actors, and subscribe the
        # actors' accept handlers + the wildcard coverage handler.
        self._bus = MessageBus()
        self._trainer = trainer_factory(_BusEventSink(self._bus, "K"))
        self._trainee = trainee_factory(_BusEventSink(self._bus, "T"))
        if self._trainer.role == self._trainee.role:
            raise ValueError(
                f"trainer and trainee must have different roles, got {self._trainer.role!r}"
            )
        # Grounding assertions apply only to an observable trainee (one that
        # exposes ``drain_observations``); a table trainee has no groundings.
        self._trainee_observable = hasattr(self._trainee, "drain_observations")
        self._bus.subscribe(WILDCARD_ROLE, self._on_emission)
        self._bus.subscribe(self._trainer.role, self._make_handler(self._trainer))
        self._bus.subscribe(self._trainee.role, self._make_handler(self._trainee))

    # -- the driver ---------------------------------------------------------

    def run(self) -> RunResult:
        """Seed the trainer and run ``bus.run()`` on a dedicated thread until a
        terminal condition."""
        self._bus.send(
            Message(role=self._trainer.role, action=_ACCEPT_ACTION, message=[])
        )
        bus_thread = threading.Thread(target=self._bus.run, daemon=True)
        bus_thread.start()
        bus_thread.join()
        if self._thread_exc is not None:
            raise self._thread_exc
        # White-box: every asserted grounding must have been observed (model B),
        # and only when the trainee is observable.
        if self._trainee_observable:
            self._check_grounding_assertions()
            if self._thread_exc is not None:
                raise self._thread_exc
        return self.result

    # -- coverage handler (the wildcard subscriber) -------------------------

    def _on_emission(self, msg: Message) -> None:
        """Wildcard handler: track coverage and divergence on every emission."""
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

        # Closed: drop trailing emissions.
        if self._closed:
            return

        # A PASS is intercepted before content matching. A PASS from one role
        # followed by a PASS from the other is terminal.
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

        # In the coverage set with copies remaining: consume one. (A close
        # that recurs as coverage consumes its coverage copies first; the
        # close terminates only once its budget is exhausted.) Budget
        # exhaustion is checked at the burst boundary by ``_on_emission``.
        budget = self._coverage_budget.get(key, 0)
        if self._consumed[key] < budget:
            self._consumed[key] += 1
            self._last_coverage_event = event
            return

        # The close content ends the run (any agent, any time) — once its
        # coverage copies are consumed (a unique close has none, so fires now).
        if key == self._closing_key:
            self._closed = True
            self._bus.stop()
            return

        # Immediate divergence (exhausted: budget spent; unmatched: present
        # nowhere). Either stops the run immediately.
        reason = "exhausted" if budget > 0 else "unmatched"
        self._record_divergence(event, reason)
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
        """White-box: record a K grounding observation (model B).

        Observations accumulate across the run; the missing-assertion check
        runs at run end. Extra observations (not asserted) are ignored.
        """
        self._observed_groundings.add(_grounding_key_from_value(grounded))

    # -- actor handler adapter ----------------------------------------------

    def _make_handler(self, actor: Actor):
        """Adapt an actor's ``accept`` to the bus's ``(msg) -> None`` handler.

        For a trainee that exposes ``drain_observations`` (the
        :class:`~training.dialogue.actors.RationalisingTrainee`), observations
        are drained after ``accept`` and verified against the expected
        groundings budget (white-box).
        """

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
        """The trainer actor (built internally from ``trainer_factory``)."""
        return self._trainer

    @property
    def trainee(self) -> Actor:
        """The trainee actor (built internally from ``trainee_factory``).

        Exposed so callers (e.g. ``scripts/dialogue_run.py -v``) can inspect a
        real actor's post-run state — notably a
        :class:`~training.dialogue.actors.RationalisingTrainee`'s grounded
        model.
        """
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
        # One placeholder per remaining authored copy, so the displacement
        # count reflects authored rows not traversed.
        out: list[DecodedTurn] = []
        for k in sorted(self._coverage_budget, key=_key_sort):
            remaining = self._coverage_budget[k] - self._consumed[k]
            out.extend([_placeholder_turn(k)] * max(remaining, 0))
        return out

    def _uncovered_groundings(self) -> list[DecodedTurn]:
        """Asserted groundings never observed (grounding displacement)."""
        missing = [
            self._expected_groundings[k]
            for k in sorted(self._expected_groundings, key=_grounding_key_sort)
            if k not in self._observed_groundings
        ]
        return missing

    def _check_grounding_assertions(self) -> None:
        """Raise/record a :class:`GroundingDivergence` for any asserted
        grounding never observed (model B subset check)."""
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
    expected_groundings: Sequence[DecodedTurn] = (),
    on_divergence: str = "fail",
) -> Runner:
    """Construct a :class:`Runner` for ``decoded``.

    ``trainer_factory`` / ``trainee_factory`` are callables ``(sink) -> Actor``:
    the runner builds the bus-wired sink and constructs each actor with it.
    ``expected_groundings`` are the decoded ``events`` the runner verifies
    white-box against K's grounding observations. The caller calls
    :meth:`Runner.run` to drive.
    """
    return Runner(
        decoded,
        trainer_factory,
        trainee_factory,
        expected_groundings=expected_groundings,
        on_divergence=on_divergence,
    )
