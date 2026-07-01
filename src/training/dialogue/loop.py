"""Dispatch-driven training loop (spec ``@specs/dialogue-driven-training.md``
§Training Loop; plan Phase 3).

The loop is **dispatch-driven**, not replay (DDT-22): the trainer-side computes
each T turn via the stateless supply function (:func:`training.dialogue.supply`)
in response to the stub's K emission; the table **validates** the trainer's
output and **scripts** the stub's. The table never drives the trainer.

Both actors are **self-cursored** (DDT-23): this loop owns the cursor over
T-rows; :class:`~training.dialogue.stub_kagent.StubKAgent` owns its cursor over
K-rows. Dispatch is **greedy per actor** (DDT-24): a run of consecutive
same-actor rows is consumed whole.

Each turn is **validated on both sides** (Model A, DDT-25): the received K is
checked against the table's K at the cursor (stub/table divergence), then the
computed T is checked against the table's T at the cursor (supply-function bug).
Failures fail fast with side attribution.

Termination is **dual-exhaustion** (DDT-26, DDT-27): the run ends only when
**both** cursors are empty. A non-empty stub cursor at trainer exhaustion fails
loudly (a truncated table that never emits its final K-run — the closing S1 —
can never pass). The closing S1 is therefore **verified by construction**, not
semantically detected (stateless-compatible, D1).

Execution is **synchronous** under the stub (§Training Loop): submit → drain →
validate, no event loop. The canonical 1:1 table makes multi-row K-run timing
moot (deferred per spec §Out of Scope).

The loop is bus-faithful: it submits T turns via the real
:class:`~training.harness.adapter.KAgentAdapter` (action ``rationalise``) and
observes the stub's K emissions via a ``supervisor``-role subscription, exactly
as a real trainer participant would. This proves the harness plumbing while
keeping the loop a synchronous iterator.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from kalvin.events import RationaliseEvent
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.stub_kagent import StubKAgent
from training.dialogue.supply import (
    SupplyMiss,
    TrainerResponse,
    build_held_index,
    opening,
    respond,
)


class LoopError(Exception):
    """A two-sided validation failure or a dual-exhaustion violation.

    Carries side attribution (DDT-25). ``kind`` is one of:

    - ``stub_divergence`` — the stub emitted a K that does not match the table's
      K at the cursor (the stub's view of the table diverges from the
      trainer-side's).
    - ``supply_bug`` — the trainer's supply function computed a T that does not
      match the table's T at the cursor.
    - ``k_unemitted`` — DDT-26 truncated table: the trainer cursor is exhausted
      but the stub cursor is non-empty (the final K-run, the closing S1, was
      never emitted — so it can never be verified).
    - ``t_unsubmitted`` — the stub cursor emptied while the trainer still has
      T-rows to submit (the table is malformed: more T-rows than K-rows drive).
    """

    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class LoopResult:
    """Outcome of a dispatch-driven run.

    ``k_emissions`` is the ordered sequence of K turns the stub emitted (each
    validated against the table). ``t_submissions`` is the ordered sequence of T
    turns the trainer computed and submitted (each validated against the table).
    ``closing`` is the final K turn (the primary's S1 countersign), verified by
    dual-exhaustion.
    """

    t_submissions: list[DecodedTurn] = field(default_factory=list)
    k_emissions: list[DecodedTurn] = field(default_factory=list)
    closing: DecodedTurn | None = None

    @property
    def dual_exhaustion(self) -> bool:
        """True when both cursors reached the end (the run's success gate)."""
        # ``closing`` is set only on a successful dual-exhaustion run.
        return self.closing is not None


def _turn_eq(a: DecodedTurn, b: DecodedTurn) -> bool:
    """KValue-aware turn equality (significance is a separate axis; KV-2).

    :class:`KValue` compares by kline only (KV-2), so a real turn-equality check
    must compare significance explicitly. Used by the two-sided Model A
    validation (DDT-25).
    """
    return (
        a.actor == b.actor
        and a.op == b.op
        and a.value.kline == b.value.kline
        and a.value.significance == b.value.significance
    )


def run_session(
    table,
    *,
    adapter,
    tokenizer=None,
    signifier=None,
) -> LoopResult:
    """Drive a dialogue table end-to-end through the stub + adapter, bus-faithfully.

    The public entry point. Compiles ``table.script`` once to build the held
    index, pre-decodes the table (:func:`~training.dialogue.decoder.decode`),
    wires the adapter-bound :class:`StubKAgent`, and drives the dispatch loop.

    Parameters
    ----------
    table:
        The :class:`~training.dialogue.decoder.DialogueTable` (``script`` +
        ``turns``).
    adapter:
        A :class:`~training.harness.adapter.KAgentAdapter` whose bound KAgent is
        a :class:`StubKAgent` constructed with the table's K-rows. The loop
        submits T turns via the bus (action ``rationalise``) and observes K
        emissions via a ``supervisor``-role subscription, synchronously.
    tokenizer, signifier:
        Forwarded to :func:`decode` and :func:`build_held_index`.

    The run is synchronous (§Training Loop): each T submission is drained before
    the next, so the stub's K emissions are observed in order. Returns a
    :class:`LoopResult`; raises :class:`LoopError` on any two-sided divergence or
    exhaustion failure.
    """
    from training.harness.bus import MessageBus  # local import: thin driver
    from training.dialogue.decoder import decode

    decoded = decode(table, tokenizer=tokenizer, signifier=signifier)
    from ks.compiler import compile_source
    entries = compile_source(
        table.script, tokenizer=tokenizer, signifier=signifier, dev=True
    )
    held = build_held_index(entries)

    bus: MessageBus = adapter._bus  # noqa: SLF001 — the adapter owns the bus
    from training.harness.constants import SUPERVISOR_ROLE, TRAINEE_ROLE
    from training.harness.message import Message

    # Observe K emissions via a supervisor-role subscription. The adapter routes
    # the stub's on_event emissions back to the sender (the trainer/supervisor);
    # we capture them in order between submissions (synchronous drain).
    captured: list[RationaliseEvent] = []

    def on_supervisor(msg: Message) -> None:
        if isinstance(msg.message, RationaliseEvent):
            captured.append(msg.message)

    bus.subscribe(SUPERVISOR_ROLE, on_supervisor)

    import queue

    def submit(t_value: KValue) -> list[RationaliseEvent]:
        """Post a T turn to the stub via the bus; return its K emissions in order."""
        before = len(captured)
        bus.send(
            Message(
                role=TRAINEE_ROLE,
                action="rationalise",
                message=t_value,
                sender=SUPERVISOR_ROLE,
            )
        )
        # Synchronous drain: the stub resolves everything inside rationalise(),
        # so its emissions are queued on the bus by the time send() returns.
        while True:
            try:
                msg = bus._queue.get_nowait()  # noqa: SLF001
            except queue.Empty:
                break
            bus._dispatch(msg)  # noqa: SLF001
        return captured[before:]

    return run_with_held(decoded, held, submit=submit)


def run_with_held(
    decoded: list[DecodedTurn],
    held,
    *,
    submit: Callable[[KValue], list[RationaliseEvent]],
) -> LoopResult:
    """Drive a pre-decoded dialogue through the stub via a submit callback.

    This is the bus-agnostic core of the loop (DDT-22..27), factored so the
    bus-faithful driver (:func:`run_session`) supplies the ``submit`` callback
    that posts a T-turn KValue and returns the stub's K emissions it produced.

    Parameters
    ----------
    decoded:
        The flat ordered ``list[DecodedTurn]`` (the pre-decoded dialogue table).
    held:
        The :class:`~training.dialogue.supply.HeldIndex` (compiled once).
    submit:
        A callback ``(T-turn KValue) -> list[RationaliseEvent]`` that submits the
        T turn to the stub (via the adapter/bus) and returns the K emissions the
        stub produced in response, in order. The bus-faithful driver drains the
        bus between submissions so these are observed synchronously.

    The loop owns the T-row cursor; the stub (behind ``submit``) owns the K-row
    cursor. Greedy per-actor dispatch (DDT-24): consecutive same-actor rows are
    consumed whole. Two-sided Model A validation (DDT-25) and dual-exhaustion
    termination (DDT-26, DDT-27) are enforced here.
    """

    # Partition by actor: each actor's self-held cursor (DDT-23).
    t_rows = [t for t in decoded if t.actor == "T"]
    k_rows = [t for t in decoded if t.actor == "K"]
    # The K-side cursor is owned by the stub; we mirror it here only to validate
    # the stub's emissions against the table (Model A K-side). It advances in
    # lockstep with the stub's real cursor as ``submit`` is called.
    k_expect_cursor = 0

    result = LoopResult()
    t_cursor = 0

    # The opening (turn 0) is computed, not read (DDT-7). Every T turn —
    # including the opening — flows through the supply function and is validated
    # against the table. The primary is the first T row's kline.
    primary = t_rows[0].value
    computed_opening = opening(primary, held)
    _validate_t(computed_opening, t_rows[0])  # DDT-25 T-side
    result.t_submissions.append(computed_opening)
    k_emissions = _submit_and_collect(computed_opening.value, submit)
    # The opening's K emissions must match the next K-rows at the cursor.
    for ev in k_emissions:
        _validate_k_event(ev, k_rows, k_expect_cursor)
        result.k_emissions.append(
            _event_to_turn(ev, actor="K", op=k_rows[k_expect_cursor].op)
        )
        k_expect_cursor += 1
    t_cursor = 1

    # Reactive phase (DDT-5): every subsequent T turn is a deterministic
    # response to a K turn. Drive until both cursors are empty (dual-exhaustion).
    while t_cursor < len(t_rows) or k_expect_cursor < len(k_rows):
        # If there are K emissions already collected (a multi-row K-run) that we
        # have not yet responded to, the next T is a response to the last of
        # those. For the 1:1 canonical table each submission yields exactly one
        # K-row; the greedy model handles multi-row runs by consuming them whole
        # before computing the next T.
        if k_expect_cursor >= len(k_rows):
            # The stub cursor emptied while the trainer still has T-rows. The
            # table is malformed (more T-rows than K-rows drive); fail loudly.
            raise LoopError(
                "t_unsubmitted: the stub cursor is empty but the trainer has "
                f"{len(t_rows) - t_cursor} T-row(s) left to submit",
                kind="t_unsubmitted",
            )

        # The K-row at the cursor is the turn the trainer responds to. Dispatch
        # via the stateless supply function (DDT-5/6/8/13/14).
        incoming_k = k_rows[k_expect_cursor - 1] if k_expect_cursor > 0 else k_rows[0]
        # Respond to the most recently observed K emission (the last one).
        last_k = result.k_emissions[-1]
        resp = respond(last_k, held)
        if resp.miss is not None:
            raise LoopError(
                f"supply miss: {resp.miss}", kind="supply_bug"
            ) from resp.miss
        if resp.turn is None:
            # S1 terminal (DDT-14): the trainer takes no action. This is the
            # closing-K case: the stub emitted the final S1 and the trainer has
            # nothing left to submit. Dual-exhaustion will confirm below.
            break

        # Validate the computed T against the table's T at the cursor (DDT-25).
        if t_cursor >= len(t_rows):
            raise LoopError(
                "supply_bug: trainer computed a T turn but the trainer cursor is "
                "exhausted (the supply function over-produced)",
                kind="supply_bug",
            )
        _validate_t(resp.turn, t_rows[t_cursor])
        result.t_submissions.append(resp.turn)
        t_cursor += 1

        # Submit T and collect the stub's K emissions; validate each (DDT-25).
        k_emissions = _submit_and_collect(resp.turn.value, submit)
        for ev in k_emissions:
            if k_expect_cursor >= len(k_rows):
                raise LoopError(
                    "stub_divergence: the stub emitted more K-rows than the table "
                    f"prescribes (cursor {k_expect_cursor} >= {len(k_rows)})",
                    kind="stub_divergence",
                )
            _validate_k_event(ev, k_rows, k_expect_cursor)
            result.k_emissions.append(
                _event_to_turn(ev, actor="K", op=k_rows[k_expect_cursor].op)
            )
            k_expect_cursor += 1

    # Dual-exhaustion gate (DDT-26, DDT-27).
    if k_expect_cursor != len(k_rows):
        # The trainer cursor is exhausted but the stub cursor is non-empty: a
        # truncated table whose final K-run (the closing S1) was never emitted
        # and therefore never verified (DDT-26).
        raise LoopError(
            "k_unemitted: trainer cursor exhausted but the stub cursor is "
            f"non-empty (K consumed {k_expect_cursor}/{len(k_rows)} — the final "
            "K-run, the closing S1, was never emitted/verified)",
            kind="k_unemitted",
        )
    if t_cursor != len(t_rows):
        raise LoopError(
            "t_unsubmitted: K-rows exhausted but T-rows remain "
            f"(T consumed {t_cursor}/{len(t_rows)})",
            kind="t_unsubmitted",
        )

    # The closing K (the primary's S1 countersign) is verified by dual-exhaustion
    # (DDT-27): it must be the last emitted K. Identify it structurally.
    if result.k_emissions:
        result.closing = result.k_emissions[-1]
    return result


# ── Model A validation helpers (DDT-25) ───────────────────────────────────


def _validate_t(computed: DecodedTurn, table_turn: DecodedTurn) -> None:
    """T-side of Model A: computed T must equal the table's T (DDT-25).

    A mismatch is a supply-function bug (the trainer computed a turn that
    diverges from the authored table). Fails fast with side attribution.
    """
    if not _turn_eq(computed, table_turn):
        raise LoopError(
            f"supply_bug: computed T {computed.op} (sig={computed.value.significance:#x}) "
            f"!= table T {table_turn.op} (sig={table_turn.value.significance:#x}) "
            f"at the cursor",
            kind="supply_bug",
        )


def _validate_k_event(
    event: RationaliseEvent,
    k_rows: list[DecodedTurn],
    cursor: int,
) -> None:
    """K-side of Model A: emitted K must equal the table's K at the cursor (DDT-25).

    A mismatch is a stub/table divergence (the stub emitted a K that does not
    match the table the trainer-side is validating against). Fails fast.
    """
    if cursor >= len(k_rows):
        raise LoopError(
            f"stub_divergence: stub emitted a K-row past the table's end "
            f"(cursor {cursor} >= {len(k_rows)})",
            kind="stub_divergence",
        )
    expected = k_rows[cursor]
    emitted = event.proposal
    if (
        emitted.kline != expected.value.kline
        or emitted.significance != expected.value.significance
    ):
        raise LoopError(
            f"stub_divergence: emitted K (sig={emitted.significance:#x}) != "
            f"table K {expected.op} (sig={expected.value.significance:#x}) "
            f"at cursor {cursor}",
            kind="stub_divergence",
        )


def _submit_and_collect(
    t_value: KValue,
    submit: Callable[[KValue], list[RationaliseEvent]],
) -> list[RationaliseEvent]:
    """Submit a T-turn KValue and return the stub's K emissions in order."""
    return submit(t_value)


def _event_to_turn(event: RationaliseEvent, *, actor: str, op: str) -> DecodedTurn:
    """Reconstruct a DecodedTurn from an observed, validated RationaliseEvent.

    The op is **not** recoverable from the event alone (the stub emits a
    ``RationaliseEvent`` with no op field, and the emitted kline's ``dbg.op`` is
    the compiled structural state, which differs from the dialogue op — e.g. an
    IDENTITY request whose label resolves to a COUNTERSIGNED compiled kline).
    The caller passes the op of the table row the event was just validated
    against, so :attr:`LoopResult.k_emissions` records the validated truth rather
    than a guess.
    """
    return DecodedTurn(actor=actor, op=op, value=event.proposal)
