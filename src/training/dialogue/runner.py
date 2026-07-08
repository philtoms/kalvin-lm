"""Bus-agnostic dialogue runner (spec ``@specs/dialogue-driven-training.md``).

A lesson is an authored dialogue between a Trainer (T) and a Trainee (K). The
runner loads and decodes the table, then drives the two actors through it. The
runner owns the table cursor; at each step it reads whose row is next and asks
that actor for its next row. Greediness is the runner's behaviour: while
consecutive table rows share an actor, the same actor is asked again.

This is the **ordered** (synchronous) regime. The **peer** regime — a sink that
receives out-of-order emissions after the trainer's opening — lives in
:mod:`training.dialogue.peer_runner` (spec ``@specs/peer-dialogue.md``).

Each actor holds its own cursor and yields its rows one at a time. Both default
actors (:class:`TableTrainer`, :class:`TableTrainee`) are structurally identical,
differing only by which ``actor`` they read — that symmetry is what makes either
replaceable by a real trainer or a real trainee. The runner does not defend
against replacement; it will evolve when real actors arrive (bringing the harness
bus with them).

The runner carries no notion of "learned" or "grounding" and emits no grounding
signal (spec §What Training Is). Training is the deterministic mechanism that
ensures the correct next response to an event is available; the validation of
both actors (§Validation) is how that is enforced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S2
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.rationalise import Rationaliser
from training.dialogue.synthesize import synthesize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier
    from kalvin.nlp_tokenizer import NLPTokenizer
    from training.dialogue.decoder import DialogueTable


class ActorDivergence(Exception):  # noqa: N818 - spec names this type
    """An actor response did not match the table at the cursor (spec §Validation).

    Raised by :func:`run` when an actor's emitted ``proposal`` (KLine +
    significance) differs from the decoded row at the cursor the actor returned.
    Under the table-reading actors this cannot occur (they read the same table);
    the check exists for the real actors.
    """

    def __init__(
        self, role: str, cursor: int, expected: KValue, emitted: KValue
    ) -> None:
        self.role = role
        self.cursor = cursor
        self.expected = expected
        self.emitted = emitted
        super().__init__(
            f"{role} divergence at cursor {cursor}: "
            f"expected sig={expected.significance:#x}, "
            f"emitted sig={emitted.significance:#x}"
        )


# ── Actor interface (spec §Actor) ─────────────────────────────────────────


@runtime_checkable
class Actor(Protocol):
    """One side of the dialogue (spec §Actor).

    Each actor yields its rows one at a time in order. ``respond`` returns the
    event for its next row, or ``None`` when the actor is exhausted. The actor
    holds no cursor it returns to the runner; it does not know the table's actor
    sequence, and the runner decides whose turn it is. The event's ``proposal``
    is the actor's turn; ``query`` is the ``incoming`` event's ``proposal`` (or
    the actor's own turn on the opening). The **runner owns the validation
    index** into the full decoded table; the actor returns only its event.
    """

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None: ...


# ── Table-reading actors (spec §The Runner, §Actor) ───────────────────────
#
# Each actor holds its own index into its own rows (the subsequence of the
# decoded table matching its ``role``) and yields them one at a time in order.
# It never inspects the incoming event to decide what to emit (DDT-10) —
# ``incoming`` only supplies the emitted event's ``query`` voice. The actor is
# dumb on purpose: it does not know about actor alternation or run boundaries,
# and it returns no cursor to the runner (the runner owns the validation index).
# The runner decides whose turn it is; greediness is the runner's behaviour.


class _TableActor:
    """Default actor: yields its rows in order, one per ``respond`` call.

    Holds the decoded table and a ``role`` label, plus an internal index that
    starts before the actor's first row. Each ``respond``/``accept`` advances
    the index by one and produces the event for that row (or nothing once
    exhausted). Each event's ``query`` is the incoming event's ``proposal`` (or
    the row's own turn on the opening).

    Adapter pattern (spec ``@specs/peer-dialogue.md`` §Actor contract): the
    actor optionally holds an :class:`~training.dialogue.peer_runner.EventSink`
    (injected at construction, as :class:`~kalvin.agent.KAgent` holds an
    adapter). ``accept`` publishes the next row via the sink (peer regime);
    ``respond`` returns it directly (ordered regime). When no sink is supplied
    the actor is ordered-only.
    """

    def __init__(
        self,
        table: Sequence[DecodedTurn],
        role: str,
        *,
        kind: str,
        sink=None,
    ) -> None:
        self._rows: tuple[DecodedTurn, ...] = tuple(t for t in table if t.role == role)
        self._index = -1
        self._kind = kind
        self._role = role
        self._sink = sink

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return self._role

    def _next_event(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None:
        nxt = self._index + 1
        if nxt >= len(self._rows):
            return None
        self._index = nxt
        turn = self._rows[nxt]
        query = incoming.proposal if incoming is not None else turn.value
        return RationaliseEvent(
            kind=self._kind, query=query, proposal=turn.value, role=self._role
        )

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None:
        # Ordered regime: return the next event directly.
        return self._next_event(incoming)

    def accept(self, incoming: RationaliseEvent | None) -> None:
        # Peer regime: publish the next event (if any) via the sink.
        event = self._next_event(incoming)
        if event is not None and self._sink is not None:
            self._sink.on_event(event)


class TableTrainer(_TableActor):
    """The default trainer: yields the table's T-rows in order.

    Replaceable by a real trainer that satisfies :class:`Actor`. Pass ``sink``
    for the peer regime (publishes via it); omit it for the ordered regime.
    """

    def __init__(self, table: Sequence[DecodedTurn], sink=None) -> None:
        super().__init__(table, role="T", kind="frame", sink=sink)


class TableTrainee(_TableActor):
    """The default trainee: yields the table's K-rows in order.

    Replaceable by a real trainee (Kalvin) that satisfies :class:`Actor`. Pass
    ``sink`` for the peer regime (publishes via it); omit it for the ordered
    regime.
    """

    def __init__(self, table: Sequence[DecodedTurn], sink=None) -> None:
        super().__init__(table, role="K", kind="frame", sink=sink)


# ── Synthesizing actor (spec §Actor; plan @plans/implement-synthesizing-trainer.md) ─
#
# A real trainer that derives each turn from the compiled script and the
# trainee's last KValue (via :func:`~training.dialogue.synthesize.synthesize`),
# never reading the decoded table. The table is only the validation oracle the
# runner checks the synthesised turn against (D1). Unlike the table-reading
# actors it is non-exhausting: R1–R3 always produce a KValue, so ``respond``
# never returns ``None``. Not produced by :func:`default_actors`; callers wire
# it explicitly (like :class:`~training.dialogue.rationalise.Rationaliser`).


# ── Synthesizing trainer ──────────────────────────────────────────────────
#
# A real trainer that derives each turn from the compiled script and the
# trainee's last KValue (via :func:`~training.dialogue.synthesize.synthesize`),
# never reading the decoded table. The table is only the validation oracle the
# runner checks the synthesised turn against (D1). Unlike the table-reading
# actors it is non-exhausting: R1–R3 always produce a KValue, so ``respond``
# never returns ``None`` on its own. Not produced by :func:`default_actors`;
# callers wire it explicitly (like :class:`~training.dialogue.rationalise.Rationaliser`).
#
# The trainer does **not** detect script closes. Close detection is the
# runner's job: it owns the table cursor and reads each turn's ``close``
# marker (spec §Dialogue Table). On a close the runner routes the *next* turn
# to the trainer with ``incoming=None`` (an open), so the trainer opens the
# next script rather than replying to the close. This keeps close-recognition
# in one place (the runner) instead of duplicated across trainer derivations.


class SynthesizingTrainer:
    """A trainer that synthesises each turn from the compiled script.

    Drop-in for :class:`TableTrainer` wherever an :class:`Actor` is accepted.
    Holds the compiled script (for structure — R2/R3 decompositions) and the
    ordered script ``primaries`` (for openings — R1), plus a signifier. The
    table is only the validation oracle; the constructor takes neither
    ``decoded`` nor the table. Pass ``sink`` for the peer regime.

    The trainer does not detect script closes. The runner owns the table
    cursor and the ``close`` markers; on a close it routes the next trainer
    turn as an open (``incoming=None``). On an open the trainer emits the
    current primary at S2 (R1) and advances to the next primary, so a
    multi-script file opens each script's own primary in turn. On a reply
    (``incoming`` is a KValue) it delegates to :func:`synthesize` (R2/R3).
    """

    def __init__(
        self,
        compiled: list[KValue],
        signifier: KSignifier,
        primaries: list[KLine],
        sink=None,
    ) -> None:
        if not primaries:
            raise ValueError("SynthesizingTrainer needs at least one primary")
        self._compiled = compiled
        self._signifier = signifier
        self._primaries = tuple(primaries)
        self._primary_index = 0
        self._sink = sink

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return "T"

    def _next_event(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None:
        if incoming is None:
            # R1 — open the current script's primary at S2, then advance so
            # the next open (after the next close) opens the following primary.
            primary = self._primaries[self._primary_index]
            self._primary_index = min(
                self._primary_index + 1, len(self._primaries) - 1
            )
            proposal = KValue(primary, SIG_S2)
            query = proposal
        else:
            incoming_value = incoming.proposal
            proposal = synthesize(self._compiled, incoming_value, self._signifier)
            query = incoming_value
        return RationaliseEvent(
            kind="frame", query=query, proposal=proposal, role="T"
        )

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None:
        return self._next_event(incoming)

    def accept(self, incoming: RationaliseEvent | None) -> None:
        event = self._next_event(incoming)
        if event is not None and self._sink is not None:
            self._sink.on_event(event)


# ── Rationalising actor (spec §Actor; plan @plans/implement-rationalising-trainee.md) ─
#
# A real trainee that rationalises each turn from its own state and the
# trainer's last KValue (via the :class:`~training.dialogue.rationalise.Rationaliser`
# engine), never reading the decoded table. Mirrors :class:`SynthesizingTrainer`:
# the engine returns a ``KValue``; this actor wraps it in a ``RationaliseEvent``.
# Unlike the table-reading actors it is non-exhausting only while work remains —
# the engine returns ``None`` when nothing is workable (D12), which this actor
# forwards. Not produced by :func:`default_actors`; callers wire it explicitly.


class RationalisingTrainee:
    """A trainee that rationalises each turn from its own model state.

    Drop-in for :class:`TableTrainee` wherever an :class:`Actor` is accepted.
    Holds a :class:`~training.dialogue.rationalise.Rationaliser` engine and a
    signifier (the engine is built at construction) and wraps each emitted
    ``KValue`` in a ``RationaliseEvent``. Constructor does **not** take
    ``decoded`` — the table is only the validation oracle. Pass ``sink`` for
    the peer regime (publishes via it); omit for ordered. Pass
    ``burst_mode=True`` for the peer regime so cogitation batches identity
    asks into one blast (the engine emits one-at-a-time otherwise, preserving
    the ordered regime's golden-master sequence).
    """

    def __init__(self, signifier: KSignifier, sink=None, *, burst_mode: bool = False) -> None:
        self._engine = Rationaliser(signifier, burst_mode=burst_mode)
        self._sink = sink
        self._burst_mode = burst_mode
        # FIFO of buffered events for the ordered regime: ``respond`` doles out
        # a batch one event at a time to preserve the one-event-per-call contract.
        # Every incoming is processed (fed to the engine) regardless of buffer
        # state; the buffer only smooths emission granularity.
        self._respond_list: list[RationaliseEvent] = []

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return "K"

    def _process_and_collect(
        self, incoming: RationaliseEvent | None
    ) -> list[RationaliseEvent]:
        """Feed ``incoming`` to the engine and collect its emitted batch as events.

        Every call processes its incoming (the engine's entry-rule bookkeeping
        runs each time), so the trainee's state stays in lockstep with the
        trainer's responses even while a burst is being drained from the buffer.
        """
        incoming_value = incoming.proposal if incoming is not None else None
        query = incoming_value
        events: list[RationaliseEvent] = []
        for proposal in self._engine.rationalise(incoming_value):
            q = query if query is not None else proposal
            events.append(
                RationaliseEvent(kind="frame", query=q, proposal=proposal, role="K")
            )
        return events

    def next_events(
        self, incoming: RationaliseEvent | None
    ) -> list[RationaliseEvent]:
        """Rationalise ``incoming`` into a batch of events (peer regime).

        The engine returns an identity blast or a single relationship emission
        (never mixed); each emitted ``KValue`` is wrapped in a
        ``RationaliseEvent``. Returns an empty list when nothing is workable.
        """
        return self._process_and_collect(incoming)

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> RationaliseEvent | None:
        # Ordered regime: process every incoming (append its batch to the FIFO),
        # then dole one event out per call. This keeps state lockstep with the
        # trainer's responses while preserving one-event-per-call emission.
        self._respond_list.extend(self._process_and_collect(incoming))
        if not self._respond_list:
            return None
        return self._respond_list.pop(0)

    def accept(self, incoming: RationaliseEvent | None) -> None:
        # Peer regime: publish the whole batch (the point of batching).
        if self._sink is None:
            return
        for event in self._process_and_collect(incoming):
            self._sink.on_event(event)


# ── The run (spec §The Runner) ────────────────────────────────────────────


@dataclass
class RunResult:
    """Outcome of a dialogue run.

    ``events`` is the ordered, interleaved sequence of emitted events (trainer
    and trainee, in exchange order). ``complete`` is True when the run ended
    with every decoded row emitted and validated (or, for a run with no
    reference table, when it ended without divergence).
    """

    events: list[RationaliseEvent] = field(default_factory=list)
    complete: bool = False


def run(decoded: Sequence[DecodedTurn], trainer: Actor, trainee: Actor) -> RunResult:
    """Drive a decoded dialogue to exhaustion.

    The runner owns the table cursor and acts as a router. Each step: read the
    actor of ``decoded[cursor]`` to decide whose turn it is, ask that actor for
    its next row, then **validate the response by the role the actor declared on
    its event** (§Validation) — not by the table's view of who responded.
    Greediness is automatic — while consecutive rows share an actor the same
    actor is asked again. The run ends when the cursor passes the end of the
    table.

    Keying validation on the event's self-declared ``role`` (rather than the
    table key) is what lets a real, possibly asynchronous actor announce itself:
    the runner routes on what the actor said, checking it against the table.

    Script boundaries come from the table: a turn carrying a ``close`` marker
    ends its script, and the next turn opens a fresh one (the runner hands the
    next actor ``incoming=None`` so it opens rather than replying to the close).
    The runner owns this boundary; the trainer does not detect closes.
    """
    actors: dict[str, Actor] = {"T": trainer, "K": trainee}
    result = RunResult()

    incoming: RationaliseEvent | None = None
    cursor = 0
    while cursor < len(decoded):
        role = decoded[cursor].role
        actor = actors[role]
        event = actor.respond(incoming)
        if event is None:
            break  # actor exhausted before the table — incomplete
        result.events.append(event)
        _validate(cursor, event, decoded)
        # A close ends a script (its ``close`` marker names which). The next
        # turn opens a new script, so the actor is handed ``incoming=None`` (an
        # open) rather than the close event (a reply). The trainer does not
        # detect closes; the runner owns the boundary.
        incoming = None if decoded[cursor].close else event
        cursor += 1

    result.complete = cursor == len(decoded)
    return result


def _validate(
    cursor: int, event: RationaliseEvent, decoded: Sequence[DecodedTurn]
) -> None:
    """Validate ``event`` against ``decoded[cursor]`` by self-declared ``role``.

    The actor announces its role on the event; the runner checks the event's
    ``role`` equals ``decoded[cursor].role`` and the emitted ``proposal`` equals
    ``decoded[cursor].value``. A role mismatch or a kline/significance mismatch
    raises :class:`ActorDivergence` (spec §Validation).

    Keying validation on the event's self-declared ``role`` (rather than the
    table key) is what lets a real, possibly asynchronous actor announce itself:
    the runner routes on what the actor said, checking it against the table.
    """
    if event.role is None:
        return  # internal emission — no decoded reference to validate against
    if event.role != decoded[cursor].role:
        raise ActorDivergence(
            role=event.role,
            cursor=cursor,
            expected=decoded[cursor].value,
            emitted=event.proposal,
        )
    expected = decoded[cursor].value
    if (
        event.proposal.kline != expected.kline
        or event.proposal.significance != expected.significance
    ):
        raise ActorDivergence(
            role=event.role,
            cursor=cursor,
            expected=expected,
            emitted=event.proposal,
        )


# ── Convenience: decode + run from a table ────────────────────────────────


def run_table(
    table: DialogueTable,
    *,
    tokenizer: NLPTokenizer | None = None,
    signifier: KSignifier | None = None,
    trainer: Actor | None = None,
    trainee: Actor | None = None,
) -> RunResult:
    """Decode ``table`` and run it through a fresh ``TableTrainer``/``TableTrainee``.

    A thin convenience over :func:`run` for the common case (default actors).
    Pass ``trainer``/``trainee`` to substitute real actors.
    """
    from training.dialogue.decoder import decode

    decoded = decode(table, tokenizer=tokenizer, signifier=signifier)
    return run(
        decoded,
        trainer=trainer or TableTrainer(decoded),
        trainee=trainee or TableTrainee(decoded),
    )


def default_actors(decoded: Sequence[DecodedTurn]) -> tuple[TableTrainer, TableTrainee]:
    """Construct the default ``TableTrainer`` and ``TableTrainee`` pair.

    Use this in callers that build the default pair; pass ``trainer``/``trainee``
    to :func:`run` to substitute real actors.
    """
    return TableTrainer(decoded), TableTrainee(decoded)
