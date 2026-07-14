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
# the publish step. A subclass implements only ``next_events``: given the
# incoming burst (the other role's whole reply as one list; empty for the
# opening seed), produce the turns to publish. The base packs them into one
# burst and publishes it via the sink, and — crucially — emits a PASS when
# ``next_events`` yields nothing, so the ``burst >= 1`` contract always holds.


class Actor:
    """A dialogue participant that publishes turns via an injected sink.

    Holds an :class:`~training.dialogue.runner.EventSink` and a ``role`` (the
    routing key this actor emits on its events). :meth:`accept` is the
    fire-and-forget entry point: it calls :meth:`next_events`, collects the
    produced events into one burst, and publishes that burst to the sink.
    ``incoming`` is the other role's whole reply as one list — empty on the
    opening seed (an actor may open), or a prior burst otherwise.

    Subclasses override :meth:`next_events` to decide what to publish for a
    given incoming burst — one, or many events. They must not publish directly
    to the sink; they return events and the base publishes them as one burst.
    When :meth:`next_events` yields nothing, the base emits a single PASS
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
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        """Produce the events to publish for the incoming ``burst``.

        ``incoming`` is the other role's whole reply as one list — empty on
        the opening seed. Returns one, or many events; the base collects them
        into a burst and publishes it via the sink. Override in a subclass; the
        base raises :class:`NotImplementedError`. Returning nothing is
        permitted — the base then emits a PASS to satisfy ``burst >= 1``
        (DDT-22).
        """
        raise NotImplementedError

    def accept(self, incoming: list[RationaliseEvent]) -> None:
        """Publish every event produced for ``incoming`` as one burst.

        Collects :meth:`next_events`' output into a single burst and publishes
        it via ``on_burst`` — one bus message carrying the whole reply.
        ``burst >= 1`` (DDT-22): when :meth:`next_events` yields nothing the
        base emits a single-PASS burst so the dialogue always gets a turn. The
        runner intercepts PASS before matching; two consecutive PASSes end the
        run as a stall.
        """
        burst = list(self.next_events(incoming))
        if not burst:
            burst = [pass_event(self._role)]
        self._sink.on_burst(burst)


# ── Table-reading actors (spec §The Runner, §Actor) ───────────────────────
#
# Each table actor holds a cursor into the **full decoded table** (not just its
# own rows). On the opening seed (an empty incoming burst) it emits its opening
# **contiguous same-role run** — every consecutive row tagged with its
# ``role`` from the cursor, stopping at the first other-role row. On a reply (a
# non-empty burst of N events) it **responds to each entry**: it advances to its
# next same-role row per incoming event and emits it, with that event's
# proposal as the emitted row's ``query`` voice. So a trainee burst of N events
# gets N trainer responses, not one — the table actor matches and answers the
# whole batch. ``incoming``'s content is still not used to *realign* the cursor
# (that is R1); the cursor only advances forward through the table's own order
# (DDT-10 intact). The actor is dumb on purpose: it does not know about run
# boundaries or coverage, and it publishes only via its sink.


class _TableActor(Actor):
    """Default actor: answers each incoming event with its next row.

    Holds the decoded table and a ``role`` label, plus an internal cursor into
    the full table that starts before the first row. On the opening seed (an
    empty burst) it emits its opening contiguous same-role run (skipping any
    other-role rows to its first row, then every consecutive same-role row
    from there). On a reply burst of N events it emits one response row per
    incoming event — advancing past any other-role rows to its next same-role
    row for each — so the whole received batch is matched and answered. Each
    response row's ``query`` is the corresponding incoming event's
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
            # Opening seed: emit the opening contiguous same-role run. Skip any
            # other-role rows to reach this actor's first row, then emit every
            # consecutive same-role row from there. Each emitted row's query is
            # its own turn (there is no inbound voice).
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
        # Reply: answer each incoming event with the next same-role row, using
        # that event's proposal as the emitted row's query voice. Yields nothing
        # for an entry once this actor's rows are exhausted (the base may then
        # emit a PASS for the whole accept).
        for event in incoming:
            yield from self._emit_row(query=event.proposal)

    def _emit_row(self, query: KValue) -> Iterable[RationaliseEvent]:
        """Emit the next same-role row, advancing past other-role rows.

        Advances the cursor past any other-role rows to the next same-role row
        and emits it, stamped with ``query`` as the inbound voice. Yields
        nothing once this actor's rows are exhausted.
        """
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
# §Dialogue Table). The opening seed (an empty incoming burst) is what opens a
# script: on the empty burst the trainer emits the current primary at S2 (R1)
# and advances to the next primary, so a multi-script file opens each script's
# own primary in turn. On a reply (a non-empty burst) it delegates to
# :func:`synthesize` (R2/R3) against the burst's last proposal.


class SynthesizingTrainer(Actor):
    """A trainer that synthesises each turn from the compiled script.

    Drop-in for :class:`TableTrainer`. Holds the compiled script (for structure
    — R2/R3 decompositions) and the ordered script ``primaries`` (for openings
    — R1), plus a signifier and the injected sink it publishes to. The runner
    checks each turn against the table.

    On the opening seed (an empty incoming burst) the trainer emits the
    current primary at S2 (R1) and advances to the next primary, so a
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
            # R1 — open the current script's primary at S2, then advance so
            # the next open (after the next close) opens the following primary.
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

    A trainee opens via the trainer (on the empty-burst seed); its ``accept``
    always receives a non-empty burst. When the engine has nothing workable for
    a turn, :meth:`next_events` yields nothing and the :class:`Actor` base
    emits a PASS (``burst >= 1``, DDT-22) — the trainee is waiting, not silent.
    """

    def __init__(self, signifier: KSignifier, sink: EventSink) -> None:
        super().__init__(role="K", sink=sink)
        self._engine = Rationaliser(signifier)

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        """Rationalise the incoming events into a batch of proposals.

        The engine receives every incoming proposal and reacts to the lot;
        each emitted ``KValue`` is wrapped in a ``RationaliseEvent``. Yields
        nothing when nothing is workable — the :class:`Actor` base then emits
        a PASS so the dialogue always gets a turn (DDT-22).

        ``incoming`` is always non-empty here (the trainer opens via the
        empty-burst seed). An empty list is a caller bug and crashes loudly.
        """
        assert incoming, "a trainee does not open; incoming must be non-empty"
        query = incoming[-1].proposal
        for proposal in self._engine.rationalise([e.proposal for e in incoming]):
            yield RationaliseEvent(
                kind="frame", query=query, proposal=proposal, role="K"
            )
