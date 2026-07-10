"""Dialogue actors (spec ``@specs/dialogue-driven-training.md``).

A lesson is an authored dialogue between a Trainer (T) and a Trainee (K). The
actors here are the **dialogue participants**: each holds an
:class:`~training.dialogue.runner.EventSink` (injected at construction) and
publishes its turns to it via :meth:`Actor.accept` (fire-and-forget,
one-or-many per incoming). The runner (:mod:`training.dialogue.runner`) drives
the run over the harness :class:`~training.harness.bus.MessageBus` and
validates emissions against the authored table.

Every ``accept`` yields **at least one** proposal (the ``burst >= 1``
contract, DDT-22): an actor whose cogitation produces nothing substantive
still owes the dialogue a turn, and the :class:`Actor` base emits a **PASS** —
a sentinel no-content proposal
(:func:`~training.dialogue.runner.pass_event`) — when :meth:`next_events`
yields nothing. The runner intercepts PASS before matching; two consecutive
PASSes (each side passing) end the run as a stall.

The two default actors (:class:`TableTrainer`, :class:`TableTrainee`) are
table-reading and structurally identical, differing only by which ``role``
they read. That symmetry is what makes either replaceable by a real trainer or
a real trainee.

Two real actors are provided: :class:`SynthesizingTrainer` (a trainer that
derives each turn from the compiled script) and :class:`RationalisingTrainee` (a
stateful trainee that rationalises each turn from its own model). Both publish
via their injected sink, so they are drop-in for the default actors.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S2
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.rationalise import Rationaliser
from training.dialogue.runner import pass_event
from training.dialogue.synthesize import synthesize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier
    from training.dialogue.runner import EventSink


# ── The Actor base class ─────────────────────────────────────────────────
#
# Every dialogue participant shares one invariant: it holds an EventSink and
# publishes its turns to it via ``accept``, fire-and-forget, **one-or-many**
# per incoming. The base class owns that invariant — the sink, the role, and
# the publish loop. A subclass implements only ``next_events``: given an
# incoming event (or ``None`` for the opening seed), produce the turns to
# publish. The base publishes each, and — crucially — emits a PASS when
# ``next_events`` yields nothing, so the ``burst >= 1`` contract always holds.


class Actor:
    """A dialogue participant that publishes turns via an injected sink.

    Holds an :class:`~training.dialogue.runner.EventSink` and a ``role`` (the
    routing key this actor emits on its events). :meth:`accept` is the
    fire-and-forget entry point: it calls :meth:`next_events` and publishes
    each produced event to the sink. ``incoming`` is ``None`` on the opening
    seed (an actor may open), or a prior event otherwise.

    Subclasses override :meth:`next_events` to decide what to publish for a
    given incoming — one, or many events. They must not publish directly to
    the sink; they return events and the base publishes them. When
    :meth:`next_events` yields nothing, the base emits a single PASS
    (:func:`~training.dialogue.runner.pass_event`) so the ``burst >= 1``
    contract (DDT-22) holds: every ``accept`` publishes at least one proposal.
    """

    def __init__(self, *, role: str, sink: EventSink) -> None:
        self._role = role
        self._sink = sink

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return self._role

    def next_events(
        self, incoming: RationaliseEvent | None
    ) -> Iterable[RationaliseEvent]:
        """Produce the events to publish for ``incoming``.

        ``incoming`` is ``None`` on the opening seed; otherwise it is the prior
        event. Returns one, or many events; the base publishes each via the
        sink. Override in a subclass; the base raises
        :class:`NotImplementedError`. Returning nothing is permitted — the
        base then emits a PASS to satisfy ``burst >= 1`` (DDT-22).
        """
        raise NotImplementedError

    def accept(self, incoming: RationaliseEvent | None) -> None:
        """Publish every event produced for ``incoming`` via the sink.

        ``burst >= 1`` (DDT-22): when :meth:`next_events` yields nothing the
        base emits a single PASS so the dialogue always gets a turn. The
        runner intercepts PASS before matching; two consecutive PASSes end the
        run as a stall.
        """
        count = 0
        for event in self.next_events(incoming):
            self._sink.on_event(event)
            count += 1
        if count == 0:
            self._sink.on_event(pass_event(self._role))


# ── Table-reading actors (spec §The Runner, §Actor) ───────────────────────
#
# Each table actor holds its own index into its own rows (the subsequence of the
# decoded table matching its ``role``) and yields them one at a time in order,
# one per ``accept``. ``incoming`` only supplies the emitted event's ``query``
# voice (DDT-10). The actor is dumb on purpose: it does not know about run
# boundaries or the table cursor, and it publishes only via its sink.


class _TableActor(Actor):
    """Default actor: yields its rows in order, one per ``accept`` call.

    Holds the decoded table and a ``role`` label, plus an internal index that
    starts before the actor's first row. Each ``accept`` advances the index by
    one and produces the event for that row (or nothing once exhausted). Each
    event's ``query`` is the incoming event's ``proposal`` (or the row's own
    turn on the opening).
    """

    def __init__(
        self,
        table: Sequence[DecodedTurn],
        role: str,
        *,
        kind: str,
        sink: EventSink,
    ) -> None:
        super().__init__(role=role, sink=sink)
        self._rows: tuple[DecodedTurn, ...] = tuple(t for t in table if t.role == role)
        self._index = -1
        self._kind = kind

    def next_events(
        self, incoming: RationaliseEvent | None
    ) -> Iterable[RationaliseEvent]:
        nxt = self._index + 1
        if nxt >= len(self._rows):
            return
        self._index = nxt
        turn = self._rows[nxt]
        query = incoming.proposal if incoming is not None else turn.value
        yield RationaliseEvent(
            kind=self._kind, query=query, proposal=turn.value, role=self._role
        )


class TableTrainer(_TableActor):
    """The default trainer: yields the table's T-rows in order.

    Replaceable by a real trainer that publishes via its sink.
    """

    def __init__(self, table: Sequence[DecodedTurn], sink: EventSink) -> None:
        super().__init__(table, role="T", kind="frame", sink=sink)


class TableTrainee(_TableActor):
    """The default trainee: yields the table's K-rows in order.

    Replaceable by a real trainee that publishes via its sink.
    """

    def __init__(self, table: Sequence[DecodedTurn], sink: EventSink) -> None:
        super().__init__(table, role="K", kind="frame", sink=sink)


# ── Synthesizing trainer ──────────────────────────────────────────────────
#
# A real trainer that derives each turn from the compiled script and the
# trainee's last KValue (via :func:`~training.dialogue.synthesize.synthesize`).
# The runner checks the synthesised turn against the table (D1). Unlike the
# table-reading actors it is non-exhausting: R1–R3 always produce a KValue, so
# ``accept`` always publishes. Callers wire it explicitly (like
# :class:`RationalisingTrainee`).
#
# The trainer does **not** detect script closes. Close detection is the
# runner's job: it owns the table and reads each turn's ``close`` marker (spec
# §Dialogue Table). The opening seed (``incoming=None``) is what opens a script:
# on ``None`` the trainer emits the current primary at S2 (R1) and advances to
# the next primary, so a multi-script file opens each script's own primary in
# turn. On a reply (``incoming`` is a KValue) it delegates to
# :func:`synthesize` (R2/R3).


class SynthesizingTrainer(Actor):
    """A trainer that synthesises each turn from the compiled script.

    Drop-in for :class:`TableTrainer`. Holds the compiled script (for structure
    — R2/R3 decompositions) and the ordered script ``primaries`` (for openings
    — R1), plus a signifier and the injected sink it publishes to. The runner
    checks each turn against the table.

    On the opening seed (``incoming=None``) the trainer emits the current
    primary at S2 (R1) and advances to the next primary, so a multi-script file
    opens each script's own primary in turn. On a reply (``incoming`` is a
    KValue) it delegates to :func:`synthesize` (R2/R3).
    """

    def __init__(
        self,
        compiled: list[KValue],
        signifier: KSignifier,
        primaries: list[KLine],
        sink: EventSink,
    ) -> None:
        if not primaries:
            raise ValueError("SynthesizingTrainer needs at least one primary")
        super().__init__(role="T", sink=sink)
        self._compiled = compiled
        self._signifier = signifier
        self._primaries = tuple(primaries)
        self._primary_index = 0

    def next_events(
        self, incoming: RationaliseEvent | None
    ) -> Iterable[RationaliseEvent]:
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
        yield RationaliseEvent(
            kind="frame", query=query, proposal=proposal, role="T"
        )


# ── Rationalising trainee ─────────────────────────────────────────────────
#
# A real trainee that rationalises each turn from its own state and the
# trainer's last KValue (via the :class:`~training.dialogue.rationalise.Rationaliser`
# engine). The engine returns a batch of ``KValue``\ s; this actor yields each
# wrapped in a ``RationaliseEvent``. Callers wire it explicitly.


class RationalisingTrainee(Actor):
    """A trainee that rationalises each turn from its own model state.

    Drop-in for :class:`TableTrainee`. Holds a
    :class:`~training.dialogue.rationalise.Rationaliser` engine and a signifier
    (the engine is built at construction), and publishes each emitted ``KValue``
    (wrapped in a ``RationaliseEvent``) via the injected sink. The runner checks
    each turn against the table.

    A trainee opens via the trainer (on the ``None`` seed); its ``accept``
    always receives a real ``RationaliseEvent``. When the engine has nothing
    workable for a turn, :meth:`next_events` yields nothing and the
    :class:`Actor` base emits a PASS (``burst >= 1``, DDT-22) — the trainee is
    waiting, not silent.
    """

    def __init__(self, signifier: KSignifier, sink: EventSink) -> None:
        super().__init__(role="K", sink=sink)
        self._engine = Rationaliser(signifier)

    def next_events(
        self, incoming: RationaliseEvent | None
    ) -> Iterable[RationaliseEvent]:
        """Rationalise ``incoming`` into a batch of events.

        The engine returns an identity blast, a batch of S3 pairings, or the
        S1 countersignature reciprocal pair (relationship paths are never mixed
        with identities); each emitted ``KValue`` is wrapped in a
        ``RationaliseEvent``. Yields nothing when nothing is workable — the
        :class:`Actor` base then emits a PASS so the dialogue always gets a
        turn (DDT-22).

        Every call processes its incoming (the engine's entry-rule bookkeeping
        runs each time), so the trainee's state stays in lockstep with the
        trainer's responses.

        ``incoming`` is always a real event here (the trainer opens via the
        ``None`` seed). ``None`` is a caller bug and crashes loudly.
        """
        assert incoming is not None, "a trainee does not open; incoming must be set"
        incoming_value = incoming.proposal
        query = incoming_value
        for proposal in self._engine.rationalise(incoming_value):
            yield RationaliseEvent(
                kind="frame", query=query, proposal=proposal, role="K"
            )
