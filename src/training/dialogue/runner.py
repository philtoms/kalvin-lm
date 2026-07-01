"""Bus-agnostic dialogue runner (spec ``@specs/dialogue-driven-training.md``).

A lesson is an authored dialogue between a Trainer (T) and a Trainee (K). The
runner loads and decodes the table, then drives the two actors through it. The
runner owns the table cursor; at each step it reads whose row is next and asks
that actor for its next row. Greediness is the runner's behaviour: while
consecutive table rows share an actor, the same actor is asked again.

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
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn

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

    Each actor holds its own cursor (starting before its first row) and yields
    its rows one at a time in order. ``respond`` advances the cursor by one and
    returns ``(next_cursor, event)`` — the new cursor position and the event for
    that row — or ``None`` when the actor is exhausted. The actor does not know
    the table's actor sequence; the runner decides whose turn it is. The event's
    ``proposal`` is the actor's turn; ``query`` is the ``incoming`` event's
    ``proposal`` (or the actor's own turn on the opening).
    """

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> tuple[int, RationaliseEvent] | None: ...


# ── Table-reading actors (spec §The Runner, §Actor) ───────────────────────
#
# Each actor holds its own cursor into its own rows (the subsequence of the
# decoded table matching its ``actor`` label) and yields them one at a time in
# order. It never inspects the incoming event to decide what to emit (DDT-10) —
# ``incoming`` only supplies the emitted event's ``query`` voice. The actor is
# dumb on purpose: it does not know about actor alternation or run boundaries.
# The runner decides whose turn it is; greediness is the runner's behaviour.


class _TableActor:
    """Default actor: yields its rows in order, one per ``respond`` call.

    Holds the decoded table and an ``actor`` label, plus a cursor that starts
    before the actor's first row (-1). Each ``respond`` advances the cursor by
    one and returns ``(next_cursor, event)`` — the new cursor and the event for
    that row — or ``None`` once exhausted. Each event's ``query`` is the incoming
    event's ``proposal`` (or the row's own turn on the opening).
    """

    def __init__(self, table: Sequence[DecodedTurn], role: str, *, kind: str) -> None:
        self._rows: tuple[DecodedTurn, ...] = tuple(t for t in table if t.role == role)
        self._cursor = -1
        self._kind = kind
        self._role = role

    @property
    def cursor(self) -> int:
        """The last cursor position advanced to (-1 before any call)."""
        return self._cursor

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._rows) - 1

    @property
    def role(self) -> str:
        """The role this actor emits on its events (the routing key)."""
        return self._role

    def respond(
        self, incoming: RationaliseEvent | None
    ) -> tuple[int, RationaliseEvent] | None:
        nxt = self._cursor + 1
        if nxt >= len(self._rows):
            return None
        self._cursor = nxt
        turn = self._rows[nxt]
        query = incoming.proposal if incoming is not None else turn.value
        return nxt, RationaliseEvent(
            kind=self._kind, query=query, proposal=turn.value, role=self._role
        )


class TableTrainer(_TableActor):
    """The default trainer: yields the table's T-rows in order.

    Replaceable by a real trainer that satisfies :class:`Actor`.
    """

    def __init__(self, table: Sequence[DecodedTurn]) -> None:
        super().__init__(table, role="T", kind="frame")


class TableTrainee(_TableActor):
    """The default trainee: yields the table's K-rows in order.

    Replaceable by a real trainee (Kalvin) that satisfies :class:`Actor`.
    """

    def __init__(self, table: Sequence[DecodedTurn]) -> None:
        super().__init__(table, role="K", kind="frame")


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
    """
    actors: dict[str, Actor] = {"T": trainer, "K": trainee}
    rows_by_role = {
        "T": [t for t in decoded if t.role == "T"],
        "K": [t for t in decoded if t.role == "K"],
    }
    result = RunResult()

    incoming: RationaliseEvent | None = None
    cursor = 0
    while cursor < len(decoded):
        role = decoded[cursor].role
        actor = actors[role]
        nxt = actor.respond(incoming)
        if nxt is None:
            break  # actor exhausted before the table — incomplete
        next_cursor, event = nxt
        result.events.append(event)
        _validate(next_cursor, event, rows_by_role)
        incoming = event
        cursor += 1

    result.complete = cursor == len(decoded)
    return result


def _validate(
    cursor: int, event: RationaliseEvent, rows_by_role: dict[str, list[DecodedTurn]]
) -> None:
    """Validate ``event`` by its self-declared ``role`` (§Validation).

    The actor announces its role on the event; the runner uses that role (not the
    table key) to find the decoded rows for that actor and check the emitted
    ``proposal`` equals the row at ``cursor``. A role the table has no rows for,
    or a kline/significance mismatch, raises :class:`ActorDivergence`.

    ``cursor`` is the next-cursor the actor returned (the 0-based index of the
    row among that actor's own rows). Skipped when the event carries no role
    (a real actor substituted later may run without a decoded reference table).
    """
    role = event.role
    if role is None:
        return
    rows = rows_by_role.get(role)
    if not rows:
        raise ActorDivergence(
            role=role,
            cursor=max(cursor, 0),
            expected=event.proposal,
            emitted=event.proposal,
        )
    if cursor < 0 or cursor >= len(rows):
        raise ActorDivergence(
            role=role,
            cursor=max(cursor, 0),
            expected=rows[-1].value,
            emitted=event.proposal,
        )
    expected = rows[cursor].value
    if (
        event.proposal.kline != expected.kline
        or event.proposal.significance != expected.significance
    ):
        raise ActorDivergence(
            role=role,
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
