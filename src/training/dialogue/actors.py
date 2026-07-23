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

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from kalvin.events import RationaliseEvent
from kalvin.expand import SIG_S1, SIG_S2, SIG_S3, SIG_S4
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, turn_content_key
from training.dialogue.rationalise import Rationaliser, RationaliserState
from training.dialogue.runner import is_pass, pass_event
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
    non-empty burst) it delegates to :func:`synthesize` (R2/R3) once per
    incoming event — a real trainee may emit a burst of several asks, and the
    trainer answers each rather than only the last.

    The trainer keeps a lightweight view of what K has grounded: the set of
    signatures emitted at S1 in the dialogue (by either side). An S1 emission
    is a ratification — its signature is grounded knowledge. R2 uses this to
    pick a canon's pedagogical significance.

    **Scripted fallback.** A synthesizing trainer is reactive: it derives a
    reply from K's proposal. When K has no substantive proposal (it PASSed),
    the trainer has nothing to synthesise against — yet it may still owe the
    dialogue a driving move (a close, the next script's opening) that has no
    structural derivation from K's state. In that case the decoded ``table``
    supplies the next T proposal: the earliest T coverage row not yet emitted.
    This is a scoped exception to script-blindness — synthesis drives every
    real exchange; the script steps in only for the trainer's driving moves.
    Without a ``table`` the trainer PASSes back (the original behaviour).
    """

    def __init__(
        self,
        compiled: list[KValue],
        signifier: KSignifier,
        primaries: list[KLine],
        sink: EventSink,
        *,
        table: Sequence[DecodedTurn] | None = None,
    ) -> None:
        if not primaries:
            raise ValueError("SynthesizingTrainer needs at least one primary")
        super().__init__(role="T", sink=sink)
        self._compiled = compiled
        self._signifier = signifier
        self._primaries = tuple(primaries)
        self._primary_index = 0
        # Signatures emitted at S1 in the dialogue (by either side) — the
        # trainer's view of K's grounded knowledge.
        self._grounded: set[int] = set()
        # The decoded table (optional) for the scripted fallback, plus a
        # per-content-key count of T coverage rows already emitted, so the
        # fallback advances past covered rows — multiplicity-aware, mirroring
        # the runner's coverage budget (a close that recurs as coverage needs
        # as many emissions as its authored copies).
        self._table: tuple[DecodedTurn, ...] = tuple(table) if table else ()
        self._t_covered: Counter[tuple] = Counter()

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        # Track K's grounded knowledge: any S1 emission ratifies its
        # signature (and a canon's nodes) as grounded.
        for event in incoming:
            if event.proposal.significance == SIG_S1:
                self._grounded.add(event.proposal.kline.signature)
                self._grounded.update(event.proposal.kline.nodes)
        if not incoming:
            # R1 — open the current script's primary at S2, then advance.
            primary = self._primaries[self._primary_index]
            self._primary_index = min(
                self._primary_index + 1, len(self._primaries) - 1
            )
            proposal = KValue(primary, SIG_S2)
            self._mark_covered(proposal)
            yield RationaliseEvent(
                kind="frame", query=proposal, proposal=proposal, role="T"
            )
            return
        # If K PASSed (no substantive proposal to synthesise against), fall
        # back to the scripted next T proposal — the trainer's driving move.
        if all(is_pass(e) for e in incoming):
            fallback = self._next_scripted_t()
            if fallback is not None:
                if fallback.significance == SIG_S1:
                    self._grounded.add(fallback.kline.signature)
                    self._grounded.update(fallback.kline.nodes)
                self._mark_covered(fallback)
                yield RationaliseEvent(
                    kind="frame", query=incoming[-1].proposal,
                    proposal=fallback, role="T",
                )
            # No scripted row remains (or no table): PASS back.
            return
        # R2/R3 — reply to each incoming event (one synthesised proposal per
        # event), mirroring the table trainer. A real trainee may emit a burst
        # of several asks; replying to only the last would drop the others and
        # stall the exchange.
        for event in incoming:
            incoming_value = event.proposal
            proposal = synthesize(
                self._compiled,
                incoming_value,
                self._signifier,
                self._grounded,
            )
            # T's own S1 emissions ratify their signatures too.
            if proposal.significance == SIG_S1:
                self._grounded.add(proposal.kline.signature)
                self._grounded.update(proposal.kline.nodes)
            self._mark_covered(proposal)
            yield RationaliseEvent(
                kind="frame", query=incoming_value, proposal=proposal, role="T"
            )

    # -- scripted fallback --------------------------------------------------

    def _next_scripted_t(self) -> KValue | None:
        """The earliest T coverage row not yet emitted to its full multiplicity.

        The scripted fallback's notion of "the next proposal T owes": the
        first T row (coverage or close) whose emitted count is below its
        authored multiplicity in the table. Counting (not a cursor) keeps the
        fallback robust to the synthesis path having emitted rows out of
        order, and honours a close that recurs as coverage (it needs as many
        emissions as it has authored copies).
        """
        budget: Counter[tuple] = Counter()
        for turn in self._table:
            if turn.role != "T":
                continue
            budget[turn_content_key(turn)] += 1
        for turn in self._table:
            if turn.role != "T":
                continue
            key = turn_content_key(turn)
            if self._t_covered[key] < budget[key]:
                return turn.value
        return None

    def _mark_covered(self, proposal: KValue) -> None:
        """Record a T proposal's content key as covered (for the fallback)."""
        self._t_covered[(
            "T",
            proposal.kline.signature,
            tuple(proposal.kline.nodes),
            proposal.significance,
        )] += 1


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


# ── Rationalising trainer ─────────────────────────────────────────────────

#: The significance bands a trainer speaks in. The dialogue is band-asymmetric:
#: S1 (ratify) and S2 (propose/canon) are the trainer's speech acts; S3
#: (connote) and S4 (identity ask) are the trainee's. A rationalising trainer
#: shares the trainee's cogitation engine, whose emissions may land in either
#: side's bands — only an S1/S2 emission is something a trainer should say.
_TRAINER_BANDS = frozenset({SIG_S1, SIG_S2})


class RationalisingTrainer(Actor):
    """A trainer that rationalises each turn, with a synthesizer supervisor.

    A rationalising counterpart to :class:`RationalisingTrainee`: it shares the
    pure :class:`~training.dialogue.rationalise.Rationaliser` engine and owns a
    :class:`~training.dialogue.rationalise.RationaliserState` of its own. The
    difference is that the trainer **leads**: it handles the opening seed (an
    empty incoming burst) by emitting the current primary at S2 and advancing,
    then routes and cogitates over K's replies like the trainee.

    **Supervisor escalation.** The shared engine was written from the trainee's
    perspective, so the trainer's own cogitation will naturally PASS at moments
    where a trainer must act — answering K's S4 identity ask, ratifying an S3
    proposal. At those natural-PASS moments the trainer asks its **supervisor**,
    :func:`~training.dialogue.synthesize.synthesize`, which answers from the
    compiled source (R2 reply to an identity, R3 echo of a compiled kline). The
    supervisor's reply flows back through the trainer's normal emission path —
    deduped, recorded as ratified knowledge when at S1 — so cogitation and
    supervision share one bookkeeping loop.

    The two share a ``_grounded`` view (the signatures emitted at S1 by either
    side), exactly like :class:`SynthesizingTrainer`: it is what the supervisor
    reads to pick a canon's pedagogical significance. A scripted fallback (the
    decoded ``table``) remains as the **driving-move backstop** for moves that
    have no supervisor derivation — a close, the next script's opening after K
    PASSes.

    Drop-in for :class:`ScriptTrainer` / :class:`SynthesizingTrainer`. Like the
    trainee, it deduplicates its own emissions (the engine is stateless about
    what it has said, so the actor is the single dedup point) and exposes its
    internal S1 groundings via :meth:`drain_observations` for inspection.
    """

    def __init__(
        self,
        signifier: KSignifier,
        primaries: Sequence[KLine],
        sink: EventSink,
        *,
        compiled: Sequence[KValue] | None = None,
        table: Sequence[DecodedTurn] | None = None,
    ) -> None:
        if not primaries:
            raise ValueError("RationalisingTrainer needs at least one primary")
        super().__init__(role="T", sink=sink)
        self._signifier = signifier
        self._primaries: tuple[KLine, ...] = tuple(primaries)
        self._primary_index = 0
        self._engine = Rationaliser(signifier)
        self._state = RationaliserState()
        self._observations: list[KValue] = []
        # Signatures of proposals already emitted into the dialogue (by
        # ``(signature, nodes)`` key). The engine is free to re-derive an
        # emission — the actor drops any proposal it has already published,
        # so T never repeats itself. Mirrors the trainee's dedup point.
        self._emitted: set[tuple[int, tuple[int, ...]]] = set()
        # The compiled source (optional) — the body the **supervisor**
        # (:func:`~training.dialogue.synthesize.synthesize`) answers from
        # when the trainer's own cogitation has nothing to say. Without it
        # the trainer has no supervisor and a natural PASS stays a PASS.
        self._compiled: tuple[KValue, ...] = tuple(compiled) if compiled else ()
        # Signatures emitted at S1 in the dialogue (by either side) — the
        # trainer's view of ratified knowledge, fed to the supervisor so a
        # canon's pedagogical significance reflects what K has actually
        # grounded. Mirrors :class:`SynthesizingTrainer`.
        self._grounded: set[int] = set()
        # The decoded table (optional) for the scripted fallback — the
        # driving-move backstop (a close, the next script's opening) that has
        # no supervisor derivation. Multiplicity-aware, mirroring the runner's
        # coverage budget and the SynthesizingTrainer's fallback. Without a
        # ``table`` the trainer PASSes back.
        self._table: tuple[DecodedTurn, ...] = tuple(table) if table else ()
        self._t_covered: Counter[tuple] = Counter()
        # Supervisor-load baseline: how often cogitation naturally PASSed and
        # the trainer asked the supervisor, and which proposals the supervisor
        # actually supplied (dedup'd-in). The goal is for this to trend toward
        # zero as the rationaliser takes on more of the trainer's load — the
        # count is the baseline for that trajectory.
        self._supervisor_asks: int = 0
        self._supervisor_emissions: list[KValue] = []

    def next_events(
        self, incoming: list[RationaliseEvent]
    ) -> Iterable[RationaliseEvent]:
        # Track ratified knowledge: any S1 emission (by either side) grounds
        # its signature (and a canon's nodes). Fed to the supervisor so its
        # canon significance reflects what K has actually grounded.
        for event in incoming:
            if event.proposal.significance == SIG_S1:
                self._grounded.add(event.proposal.kline.signature)
                self._grounded.update(event.proposal.kline.nodes)

        # Opening seed — the trainer leads. Emit the current primary at S2,
        # advance, and seed the engine's work-list with the relationship that
        # primary declares so cogitation can drive from it on later turns.
        if not incoming:
            primary = self._primaries[self._primary_index]
            self._primary_index = min(
                self._primary_index + 1, len(self._primaries) - 1
            )
            proposal = KValue(primary, SIG_S2)
            self._seed_work_list(primary)
            event = self._emit(proposal, query=proposal)
            if event is not None:
                yield event
            return

        # If K PASSed, the trainer owes a driving move (a close, the next
        # script's opening) that has no supervisor derivation — the scripted
        # fallback is the backstop. Without a table the trainer PASSes back
        # (mutual PASS — terminal).
        if all(is_pass(e) for e in incoming):
            fallback = self._next_scripted_t()
            if fallback is not None:
                event = self._emit(fallback, query=incoming[-1].proposal)
                if event is not None:
                    yield event
            return

        # Route + cogitate over K's replies (shared engine), deduplicating
        # any proposal T has already published. The engine was written from
        # the trainee's perspective, so its cogitation may emit at the
        # **trainee's** bands (S3 proposals, S4 identity asks) — speech acts a
        # trainer does not make (T leads and ratifies: S1/S2 only). Only an
        # emission at the trainer's own bands counts as "cogitation had
        # something to say"; otherwise this is the trainer's natural PASS —
        # escalate to the supervisor (``synthesize``), which answers from the
        # compiled source (R2 reply to an identity, R3 echo of a kline).
        query = incoming[-1].proposal
        batch, observations = self._engine.rationalise(
            self._state, [e.proposal for e in incoming]
        )
        self._observations.extend(observations)
        emitted_any = False
        for proposal in batch:
            if proposal.significance not in _TRAINER_BANDS:
                continue  # a trainee speech act (S3/S4) — not for T to say
            event = self._emit(proposal, query=query)
            if event is not None:
                emitted_any = True
                yield event
        if emitted_any:
            return
        # Natural PASS: cogitation produced nothing for T to say. Ask the
        # supervisor.
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

    def _emit(self, proposal: KValue, *, query: KValue) -> RationaliseEvent | None:
        """Build one T event for ``proposal`` with shared bookkeeping.

        The single emission path for cogitated, supervised, and fallback
        proposals: dedup against what T has already published, record an S1
        proposal as ratified knowledge (feeding the supervisor next turn), and
        mark its content key covered (for the scripted fallback). Returns the
        event, or ``None`` when ``proposal`` is a duplicate — so the caller can
        tell cogitation produced nothing new and escalate to the supervisor.
        """
        key = (proposal.kline.signature, tuple(proposal.kline.nodes))
        if key in self._emitted:
            return None
        self._emitted.add(key)
        if proposal.significance == SIG_S1:
            self._grounded.add(proposal.kline.signature)
            self._grounded.update(proposal.kline.nodes)
        self._mark_covered(proposal)
        return RationaliseEvent(
            kind="frame", query=query, proposal=proposal, role="T"
        )

    def drain_observations(self) -> list[KValue]:
        """Return and clear the S1 grounding observations accumulated this turn.

        Surfaced for inspection (e.g. by the driver). The runner's grounding
        assertions apply only to the trainee, so these observations are not
        verified against the script; they are the trainer's internal bookkeeping.
        """
        drained, self._observations = self._observations, []
        return drained

    def supervisor_escalations(self) -> tuple[int, list[KValue]]:
        """The supervisor-load baseline: ``(asks, emissions)``.

        ``asks`` is how many times cogitation naturally PASSed and the trainer
        consulted the supervisor; ``emissions`` is the subset of those answers
        that were actually published (not dedup'd out), in emission order. The
        goal is for both to trend toward zero as the rationaliser takes on more
        of the trainer's load — this is the baseline for that trajectory.
        """
        return self._supervisor_asks, list(self._supervisor_emissions)

    # -- opening seed -------------------------------------------------------

    def _seed_work_list(self, primary: KLine) -> None:
        """Seed the engine's work-list with the relationship the primary declares.

        A primary opens as a proposal: the trainer has *asserted* it, not
        grounded it. Placing it on the work-list lets cogitation treat it as a
        pending entry the trainee will be asked to recognise — mirroring how a
        trainee's work-list accumulates incoming S2 proposals.
        """
        if primary not in self._state.work_list:
            self._state.work_list.append(primary)

    # -- scripted fallback --------------------------------------------------

    def _next_scripted_t(self) -> KValue | None:
        """The earliest T coverage row not yet emitted to its full multiplicity.

        The same notion of "the next proposal T owes" as the
        :class:`SynthesizingTrainer` fallback: the first T row (coverage or
        close) whose emitted count is below its authored multiplicity. Counting
        (not a cursor) keeps the fallback robust to the cogitation path having
        emitted rows out of order, and honours a close that recurs as coverage.
        """
        budget: Counter[tuple] = Counter()
        for turn in self._table:
            if turn.role != "T":
                continue
            budget[turn_content_key(turn)] += 1
        for turn in self._table:
            if turn.role != "T":
                continue
            key = turn_content_key(turn)
            if self._t_covered[key] < budget[key]:
                return turn.value
        return None

    def _mark_covered(self, proposal: KValue) -> None:
        """Record a T proposal's content key as covered (for the fallback)."""
        self._t_covered[(
            "T",
            proposal.kline.signature,
            tuple(proposal.kline.nodes),
            proposal.significance,
        )] += 1
