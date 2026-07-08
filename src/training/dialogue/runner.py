"""Dialogue actors (spec ``@specs/dialogue-driven-training.md``, ``@specs/peer-dialogue.md``).

A lesson is an authored dialogue between a Trainer (T) and a Trainee (K). The
actors here are the **dialogue participants**: each holds an
:class:`~training.dialogue.peer_runner.EventSink` (injected at construction, as
:class:`~kalvin.agent.KAgent` holds an adapter) and publishes its turns to it via
``accept`` (fire-and-forget, zero-or-many per incoming). The peer runner
(:mod:`training.dialogue.peer_runner`) drives the run over the harness
:class:`~training.harness.bus.MessageBus` and validates emissions against the
authored table.

The two default actors (:class:`TableTrainer`, :class:`TableTrainee`) are
table-reading doubles — structurally identical, differing only by which ``role``
they read. That symmetry is what makes either replaceable by a real trainer or a
real trainee. The default actors are scaffolding, not the design's point.

Two real actors are provided: :class:`SynthesizingTrainer` (a trainer that
derives each turn from the compiled script) and :class:`RationalisingTrainee` (a
stateful trainee that rationalises each turn from its own model). Both are
adapter-driven and publish via their injected sink, so they are drop-in for the
default actors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S2
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.rationalise import Rationaliser
from training.dialogue.synthesize import synthesize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier
    from training.dialogue.peer_runner import EventSink


# ── Table-reading actors (spec §The Runner, §Actor) ───────────────────────
#
# Each actor holds its own index into its own rows (the subsequence of the
# decoded table matching its ``role``) and yields them one at a time in order,
# one per ``accept``. It never inspects the incoming event to decide what to
# emit (DDT-10) — ``incoming`` only supplies the emitted event's ``query``
# voice. The actor is dumb on purpose: it does not know about run boundaries or
# the table cursor, and it publishes only via its sink.


class _TableActor:
    """Default actor: yields its rows in order, one per ``accept`` call.

    Holds the decoded table and a ``role`` label, plus an internal index that
    starts before the actor's first row. Each ``accept`` advances the index by
    one and publishes the event for that row (or nothing once exhausted) via
    the injected :class:`~training.dialogue.peer_runner.EventSink`. Each event's
    ``query`` is the incoming event's ``proposal`` (or the row's own turn on the
    opening).
    """

    def __init__(
        self,
        table: Sequence[DecodedTurn],
        role: str,
        *,
        kind: str,
        sink: EventSink,
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

    def accept(self, incoming: RationaliseEvent | None) -> None:
        event = self._next_event(incoming)
        if event is not None:
            self._sink.on_event(event)


class TableTrainer(_TableActor):
    """The default trainer: yields the table's T-rows in order.

    Replaceable by a real trainer that publishes via its sink.
    """

    def __init__(self, table: Sequence[DecodedTurn], sink: EventSink) -> None:
        super().__init__(table, role="T", kind="frame", sink=sink)


class TableTrainee(_TableActor):
    """The default trainee: yields the table's K-rows in order.

    Replaceable by a real trainee (Kalvin) that publishes via its sink.
    """

    def __init__(self, table: Sequence[DecodedTurn], sink: EventSink) -> None:
        super().__init__(table, role="K", kind="frame", sink=sink)


# ── Synthesizing trainer ──────────────────────────────────────────────────
#
# A real trainer that derives each turn from the compiled script and the
# trainee's last KValue (via :func:`~training.dialogue.synthesize.synthesize`),
# never reading the decoded table. The table is only the validation oracle the
# peer runner checks the synthesised turn against (D1). Unlike the table-reading
# actors it is non-exhausting: R1–R3 always produce a KValue, so ``accept``
# always publishes. Not produced by the peer runner; callers wire it explicitly
# (like :class:`RationalisingTrainee`).
#
# The trainer does **not** detect script closes. Close detection is the peer
# runner's job: it owns the table and reads each turn's ``close`` marker (spec
# §Dialogue Table). The opening seed (``incoming=None``) is what opens a script:
# on ``None`` the trainer emits the current primary at S2 (R1) and advances to
# the next primary, so a multi-script file opens each script's own primary in
# turn. On a reply (``incoming`` is a KValue) it delegates to
# :func:`synthesize` (R2/R3).


class SynthesizingTrainer:
    """A trainer that synthesises each turn from the compiled script.

    Drop-in for :class:`TableTrainer`. Holds the compiled script (for structure
    — R2/R3 decompositions) and the ordered script ``primaries`` (for openings
    — R1), plus a signifier and the injected sink it publishes to. The table is
    only the validation oracle; the constructor takes neither ``decoded`` nor
    the table.

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

    def accept(self, incoming: RationaliseEvent | None) -> None:
        event = self._next_event(incoming)
        if event is not None:
            self._sink.on_event(event)


# ── Rationalising trainee ─────────────────────────────────────────────────
#
# A real trainee that rationalises each turn from its own state and the
# trainer's last KValue (via the :class:`~training.dialogue.rationalise.Rationaliser`
# engine), never reading the decoded table. Mirrors :class:`SynthesizingTrainer`:
# the engine returns a batch of ``KValue``\ s; this actor wraps each in a
# ``RationaliseEvent`` and publishes the whole batch. Not produced by the peer
# runner; callers wire it explicitly.


class RationalisingTrainee:
    """A trainee that rationalises each turn from its own model state.

    Drop-in for :class:`TableTrainee`. Holds a
    :class:`~training.dialogue.rationalise.Rationaliser` engine and a signifier
    (the engine is built at construction), and publishes each emitted ``KValue``
    (wrapped in a ``RationaliseEvent``) via the injected sink. Constructor does
    **not** take ``decoded`` — the table is only the validation oracle.
    """

    def __init__(self, signifier: KSignifier, sink: EventSink) -> None:
        self._engine = Rationaliser(signifier)
        self._sink = sink

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return "K"

    def next_events(
        self, incoming: RationaliseEvent | None
    ) -> list[RationaliseEvent]:
        """Rationalise ``incoming`` into a batch of events.

        The engine returns an identity blast or a single relationship emission
        (never mixed); each emitted ``KValue`` is wrapped in a
        ``RationaliseEvent``. Returns an empty list when nothing is workable.
        """
        return self._process_and_collect(incoming)

    def _process_and_collect(
        self, incoming: RationaliseEvent | None
    ) -> list[RationaliseEvent]:
        """Feed ``incoming`` to the engine and collect its emitted batch as events.

        Every call processes its incoming (the engine's entry-rule bookkeeping
        runs each time), so the trainee's state stays in lockstep with the
        trainer's responses.
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

    def accept(self, incoming: RationaliseEvent | None) -> None:
        """Publish the whole batch for ``incoming`` (the point of batching)."""
        for event in self._process_and_collect(incoming):
            self._sink.on_event(event)
