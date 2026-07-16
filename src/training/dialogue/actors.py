"""Dialogue actors.

Each actor holds an :class:`~training.dialogue.runner.EventSink` (injected at
construction) and publishes its turns to it via :meth:`Actor.accept`
(fire-and-forget, one-or-many per incoming). The runner drives the run over the
harness ``MessageBus`` and validates emissions against the authored table.

The two defaults (:class:`TableTrainer`, :class:`TableTrainee`) are table-reading
and structurally symmetric, differing only by which ``role`` they read. Two
real actors are provided: :class:`SynthesizingTrainer` and
:class:`RationalisingTrainee`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S2
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.rationalise import Rationaliser, RationaliserState
from training.dialogue.runner import pass_event
from training.dialogue.synthesize import synthesize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier
    from training.dialogue.runner import EventSink


# ── The Actor base class ─────────────────────────────────────────────────


class Actor:
    """A dialogue participant that publishes turns via an injected sink.

    Holds an :class:`~training.dialogue.runner.EventSink` and a ``role``.
    :meth:`accept` calls :meth:`next_events`, collects the produced events into
    one burst, and publishes it via the sink. When :meth:`next_events` yields
    nothing, the base emits a single PASS so the ``burst >= 1`` contract holds.
    """

    def __init__(self, *, role: str, sink: EventSink) -> None:
        self._role = role
        self._sink = sink

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return self._role

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        """Produce the events to publish for the incoming burst.

        Override in a subclass; the base raises :class:`NotImplementedError`.
        Returning nothing is permitted — the base then emits a PASS.
        """
        raise NotImplementedError

    def accept(self, incoming: list[RationaliseEvent]) -> None:
        """Publish every event produced for ``incoming`` as one burst."""
        burst = list(self.next_events(incoming))
        if not burst:
            burst = [pass_event(self._role)]
        self._sink.on_burst(burst)


# ── Table-reading actors ─────────────────────────────────────────────────


class _TableActor(Actor):
    """Default actor: answers each incoming event with its next row.

    Holds the decoded table, a ``role`` label, and a cursor. On the opening
    seed (an empty burst) it emits its opening contiguous same-role run. On a
    reply burst of N events it emits one response row per incoming event.
    Each response row's ``query`` is the corresponding incoming event's
    ``proposal``; each seed row's ``query`` is its own turn.
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
        self._table: tuple[DecodedTurn, ...] = tuple(table)
        self._cursor = -1
        self._kind = kind

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        if not incoming:
            # Opening seed: emit the opening contiguous same-role run.
            i = self._cursor + 1
            while i < len(self._table) and self._table[i].role != self._role:
                i += 1
            start = i
            while i < len(self._table) and self._table[i].role == self._role:
                turn = self._table[i]
                yield RationaliseEvent(
                    kind=self._kind, query=turn.value, proposal=turn.value,
                    role=self._role,
                )
                i += 1
            self._cursor = i - 1 if i > start else self._cursor
            return
        # Reply: answer each incoming event with the next same-role row.
        for event in incoming:
            yield from self._emit_row(query=event.proposal)

    def _emit_row(self, query: KValue) -> Iterable[RationaliseEvent]:
        """Emit the next same-role row, advancing past other-role rows."""
        i = self._cursor + 1
        while i < len(self._table) and self._table[i].role != self._role:
            i += 1
        if i >= len(self._table):
            return  # no same-role row remains
        turn = self._table[i]
        self._cursor = i
        yield RationaliseEvent(
            kind=self._kind,
            query=query,
            proposal=turn.value,
            role=self._role,
        )


class ScriptTrainer(_TableActor):
    """The default trainer: yields the table's T-rows in order."""

    def __init__(self, table: Sequence[DecodedTurn], sink: EventSink) -> None:
        super().__init__(table, role="T", kind="frame", sink=sink)


class ScriptTrainee(_TableActor):
    """The default trainee: yields the table's K-rows in order."""

    def __init__(self, table: Sequence[DecodedTurn], sink: EventSink) -> None:
        super().__init__(table, role="K", kind="frame", sink=sink)


# ── Synthesizing trainer ──────────────────────────────────────────────────


class SynthesizingTrainer(Actor):
    """A trainer that synthesises each turn from the compiled script.

    Drop-in for :class:`TableTrainer`. On the opening seed (an empty incoming
    burst) it emits the current primary at S2 (R1) and advances, so a
    multi-script file opens each script's own primary in turn. On a reply (a
    non-empty burst) it delegates to :func:`synthesize` (R2/R3) against the
    burst's last proposal.
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
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        if not incoming:
            # R1 — open the current script's primary at S2, then advance.
            primary = self._primaries[self._primary_index]
            self._primary_index = min(
                self._primary_index + 1, len(self._primaries) - 1
            )
            proposal = KValue(primary, SIG_S2)
            query = proposal
        else:
            incoming_value = incoming[-1].proposal
            proposal = synthesize(self._compiled, incoming_value, self._signifier)
            query = incoming_value
        yield RationaliseEvent(
            kind="frame", query=query, proposal=proposal, role="T"
        )


# ── Rationalising trainee ─────────────────────────────────────────────────


class RationalisingTrainee(Actor):
    """A trainee that rationalises each turn from its own model state.

    Drop-in for :class:`TableTrainee`. Holds a
    :class:`~training.dialogue.rationalise.Rationaliser` engine and owns the
    :class:`~training.dialogue.rationalise.RationaliserState`. Each turn it
    runs the pure engine over the incoming batch, publishes the resulting
    dialogue emissions (wrapped in ``RationaliseEvent``\ s), and retains the
    S1 grounding observations for the runner to drain via
    :meth:`drain_observations`. A trainee opens via the trainer; when the
    engine has nothing workable, :meth:`next_events` yields nothing and the
    base emits a PASS.
    """

    def __init__(self, signifier: KSignifier, sink: EventSink) -> None:
        super().__init__(role="K", sink=sink)
        self._engine = Rationaliser(signifier)
        self._state = RationaliserState()
        self._observations: list[KValue] = []
        # Signatures of proposals already emitted into the dialogue (by
        # ``(signature, nodes)`` key). The engine is free to re-derive an
        # emission — it has no memory of what it has said — and the actor drops
        # any proposal it has already published, so K never repeats itself.
        # This is the single deduplication point; it lets the engine stay
        # stateless about its own emissions.
        self._emitted: set[tuple[int, tuple[int, ...]]] = set()

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        """Rationalise the incoming events into a batch of proposals.

        Drops any proposal the engine re-derives that K has already published
        (emission deduplication). If every proposal is a duplicate the batch is
        empty and the base emits a PASS — K waits for the trainer.
        """
        assert incoming, "a trainee does not open; incoming must be non-empty"
        query = incoming[-1].proposal
        batch, observations = self._engine.rationalise(
            self._state, [e.proposal for e in incoming]
        )
        self._observations.extend(observations)
        for proposal in batch:
            key = (proposal.kline.signature, tuple(proposal.kline.nodes))
            if key in self._emitted:
                continue
            self._emitted.add(key)
            yield RationaliseEvent(
                kind="frame", query=query, proposal=proposal, role="K"
            )

    def drain_observations(self) -> list[KValue]:
        """Return and clear the S1 grounding observations accumulated this turn.

        The runner calls this after each K turn to verify K's internal
        grounding against the script's ``events`` (white-box), separately from
        the dialogue coverage check (black-box).
        """
        drained, self._observations = self._observations, []
        return drained
