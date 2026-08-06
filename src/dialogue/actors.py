"""Dialogue actors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from kalvin.events import RationaliseEvent
from kalvin.significance import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import is_canon
from kalvin.kvalue import KValue
from dialogue.decoder import DecodedTurn, turn_content_key
from dialogue.rationalise import Rationaliser, RationaliserState
from dialogue.runner import is_pass, pass_event
from dialogue.synthesize import synthesize

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kalvin.abstract import KSignifier
    from dialogue.runner import EventSink


# ── The Actor base class ─────────────────────────────────────────────────


class Actor:
    """A dialogue participant that publishes a burst per ``accept``.

    ``accept`` publishes whatever ``next_events`` yields; if it yields nothing,
    the base emits a PASS (the ``burst >= 1`` contract). ``_bind_sink`` re-wires
    the sink for run sequencing (shared instances across runs).
    """

    def __init__(self, *, role: str, sink: EventSink) -> None:
        self._role = role
        self._sink = sink

    @property
    def role(self) -> str:
        return self._role

    def _bind_sink(self, sink: EventSink) -> None:
        self._sink = sink

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        raise NotImplementedError

    def accept(self, incoming: list[RationaliseEvent]) -> None:
        burst = list(self.next_events(incoming))
        if not burst:
            burst = [pass_event(self._role)]
        self._sink.on_burst(burst)


# ── Table-reading actors ─────────────────────────────────────────────────


class _TableActor(Actor):
    """Answers each incoming event with the next same-role row from ``table``.

    Reactive only. If this actor plays the opening role, its cursor starts
    past the opening same-role prefix (the runner already delivered those).
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
        self._kind = kind
        # Skip the opening same-role prefix when this actor opens (runner-delivered).
        if table and table[0].role == role:
            i = 0
            while i < len(table) and table[i].role == role:
                i += 1
            self._cursor = i - 1
        else:
            self._cursor = -1

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        # One row per incoming event, but never mix S1 with other bands in a
        # single burst: a burst is all-S1 or all-non-S1. When the next row
        # would cross that boundary, stop — the held-back row emits next turn.
        # Mirrors how the rationaliser batches (S1 ratifications never share a
        # burst with S2/S3/S4 proposals).
        burst_class: bool | None = None
        for event in incoming:
            row = self._peek_row()
            if row is None:
                return
            row_class = row.value.significance == SIG_S1
            if burst_class is None:
                burst_class = row_class
            elif row_class != burst_class:
                return
            yield from self._emit_row(query=event.proposal)

    def _peek_row(self) -> DecodedTurn | None:
        """The next same-role row without advancing the cursor."""
        i = self._cursor + 1
        while i < len(self._table) and self._table[i].role != self._role:
            i += 1
        return self._table[i] if i < len(self._table) else None

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


# A trainer that owes scripted driving moves (closes, etc.) when cogitation or
# synthesis has nothing to reply with. Holds a per-content-key count of T rows
# already emitted; ``_next_scripted_t`` returns the earliest T row not yet
# emitted to its authored multiplicity.
class _ScriptedFallback:
    """Mixin: advance through the decoded table's T rows by emitted count."""

    _table: tuple[DecodedTurn, ...]
    _t_covered: Counter[tuple]

    def _init_fallback(self, table: Sequence[DecodedTurn] | None) -> None:
        self._table = tuple(table) if table else ()
        self._t_covered = Counter()
        # The runner delivers the opening same-role prefix (T opens most runs);
        # those rows are already covered, so the fallback skips them. Matches
        # ``_TableActor`` starting its cursor past the opening prefix.
        for turn in self._table:
            if turn.role != "T":
                break
            self._t_covered[turn_content_key(turn)] += 1

    def _next_scripted_t(self) -> KValue | None:
        budget: Counter[tuple] = Counter(
            turn_content_key(t) for t in self._table if t.role == "T"
        )
        for turn in self._table:
            if turn.role != "T":
                continue
            key = turn_content_key(turn)
            if self._t_covered[key] < budget[key]:
                return turn.value
        return None

    def _mark_covered(self, proposal: KValue) -> None:
        self._t_covered[(
            "T",
            proposal.kline.signature,
            tuple(proposal.kline.nodes),
            proposal.significance,
        )] += 1


class SynthesizingTrainer(_ScriptedFallback, Actor):
    """Replies to each incoming event via :func:`synthesize`; falls back to the
    scripted next T row when K PASSes. Reactive only (the runner opens)."""

    def __init__(
        self,
        compiled: list[KValue],
        signifier: KSignifier,
        sink: EventSink,
        *,
        table: Sequence[DecodedTurn] | None = None,
    ) -> None:
        super().__init__(role="T", sink=sink)
        self._compiled = compiled
        self._signifier = signifier
        # Signatures emitted at S1 by either side — the trainer's view of what
        # K has grounded, read by ``synthesize`` to pick a canon's significance.
        self._grounded: set[int] = set()
        self._init_fallback(table)

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        for event in incoming:
            if event.proposal.significance == SIG_S1:
                self._grounded.add(event.proposal.kline.signature)
                self._grounded.update(event.proposal.kline.nodes)
        # K PASSed: owe a driving move from the table, else PASS back.
        if all(is_pass(e) for e in incoming):
            fallback = self._next_scripted_t()
            if fallback is None:
                return
            self._ratify(fallback)
            self._mark_covered(fallback)
            yield RationaliseEvent(
                kind="frame", query=incoming[-1].proposal,
                proposal=fallback, role="T",
            )
            return
        for event in incoming:
            proposal = synthesize(
                self._compiled, event.proposal, self._signifier, self._grounded,
            )
            self._ratify(proposal)
            self._mark_covered(proposal)
            yield RationaliseEvent(
                kind="frame", query=event.proposal, proposal=proposal, role="T"
            )

    def _ratify(self, proposal: KValue) -> None:
        """Record an S1 proposal's signature (and a canon's nodes) as grounded."""
        if proposal.significance == SIG_S1:
            self._grounded.add(proposal.kline.signature)
            self._grounded.update(proposal.kline.nodes)


# ── Rationalising trainee ─────────────────────────────────────────────────


class RationalisingTrainee(Actor):
    """Replies from the shared rationaliser engine; dedups its own emissions and
    exposes S1 groundings via :meth:`drain_observations`."""

    def __init__(
        self, signifier: KSignifier, sink: EventSink,
        *, state: RationaliserState | None = None,
    ) -> None:
        super().__init__(role="K", sink=sink)
        self._engine = Rationaliser(signifier)
        self._state = state if state is not None else RationaliserState()
        self._observations: list[KValue] = []
        # ``(signature, nodes)`` keys this actor has published — the single
        # dedup point (the engine is stateless about its own emissions).
        self._emitted: set[tuple[int, tuple[int, ...]]] = set()

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        query = incoming[-1].proposal
        batch, observations = self._engine.rationalise(
            self._state, [e.proposal for e in incoming]
        )
        self._observations.extend(observations)
        for proposal in batch:
            if proposal.significance not in _TRAINEE_BANDS:
                continue
            key = (proposal.kline.signature, tuple(proposal.kline.nodes))
            if key in self._emitted:
                continue
            self._emitted.add(key)
            yield RationaliseEvent(
                kind="frame", query=query, proposal=proposal, role="K"
            )

    def drain_observations(self) -> list[KValue]:
        """Return and clear the S1 groundings accumulated since last call."""
        drained, self._observations = self._observations, []
        return drained


# The speech-act bands each role keeps from the role-neutral engine's batch.
# T: S1 (ratify) / S2 (propose). K: S2 (similar-fit) / S3 (connote) / S4 (ask).
_TRAINER_BANDS = frozenset({SIG_S1, SIG_S2})
_TRAINEE_BANDS = frozenset({SIG_S2, SIG_S3, SIG_S4})


class RationalisingTrainer(_ScriptedFallback, Actor):
    """Cogitates via the shared engine (keeping S1/S2); escalates to
    :func:`synthesize` when cogitation has nothing to say; falls back to the
    scripted next T row when K PASSes. Reactive only (the runner opens)."""

    def __init__(
        self,
        signifier: KSignifier,
        sink: EventSink,
        *,
        compiled: Sequence[KValue] | None = None,
        table: Sequence[DecodedTurn] | None = None,
        state: RationaliserState | None = None,
    ) -> None:
        super().__init__(role="T", sink=sink)
        self._signifier = signifier
        self._engine = Rationaliser(signifier)
        self._state = state if state is not None else RationaliserState()
        self._observations: list[KValue] = []
        self._emitted: set[tuple[int, tuple[int, ...]]] = set()
        self._compiled: tuple[KValue, ...] = tuple(compiled) if compiled else ()
        self._grounded: set[int] = set()
        self._init_fallback(table)
        # Supervisor-load baseline (escalation load).
        self._supervisor_asks: int = 0
        self._supervisor_emissions: list[KValue] = []

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        for event in incoming:
            if event.proposal.significance == SIG_S1:
                self._grounded.add(event.proposal.kline.signature)
                self._grounded.update(event.proposal.kline.nodes)

        # K PASSed: owe a driving move from the table, else PASS back.
        if all(is_pass(e) for e in incoming):
            fallback = self._next_scripted_t()
            if fallback is not None:
                event = self._emit(fallback, query=incoming[-1].proposal)
                if event is not None:
                    yield event
            return

        # Cogitate; keep the trainer's own bands.
        query = incoming[-1].proposal
        batch, observations = self._engine.rationalise(
            self._state, [e.proposal for e in incoming]
        )
        self._observations.extend(observations)
        emitted_any = False
        for proposal in batch:
            if proposal.significance not in _TRAINER_BANDS:
                continue
            event = self._emit(self._trainer_kline_override(proposal), query=query)
            if event is not None:
                emitted_any = True
                yield event
        if emitted_any:
            return

        # Cogitation had nothing for T: ask the supervisor.
        for incoming_event in incoming:
            if is_pass(incoming_event):
                continue
            self._supervisor_asks += 1
            supervised = synthesize(
                list(self._compiled),
                incoming_event.proposal,
                self._signifier,
                self._grounded,
            )
            event = self._emit(supervised, query=query)
            if event is not None:
                self._supervisor_emissions.append(supervised)
                yield event

    def _trainer_kline_override(self, proposal: KValue) -> KValue:
        """Propose canons at S2 even when the engine has grounded every node."""
        if proposal.significance == SIG_S1 and is_canon(proposal.kline, self._signifier):
            return KValue(proposal.kline, SIG_S2)
        return proposal

    def _emit(self, proposal: KValue, *, query: KValue) -> RationaliseEvent | None:
        """Build one T event for ``proposal``, deduping and bookkeeping. Returns
        ``None`` when ``proposal`` is a duplicate."""
        key = (proposal.kline.signature, tuple(proposal.kline.nodes))
        if key in self._emitted:
            return None
        self._emitted.add(key)
        if proposal.significance == SIG_S1:
            self._grounded.add(proposal.kline.signature)
            self._grounded.update(proposal.kline.nodes)
        self._mark_covered(proposal)
        return RationaliseEvent(kind="frame", query=query, proposal=proposal, role="T")

    def drain_observations(self) -> list[KValue]:
        """Return and clear the S1 groundings accumulated since last call."""
        drained, self._observations = self._observations, []
        return drained

    def supervisor_escalations(self) -> tuple[int, list[KValue]]:
        """The escalation-load baseline: ``(asks, published answers)``."""
        return self._supervisor_asks, list(self._supervisor_emissions)
