"""Phase 3 — the Runner as a MessageBus subscriber.

Spec: ``@specs/dialogue-driven-training.md`` DDT-5..DDT-22. The runner is a
coverage-tracking wildcard subscriber over a ``MessageBus`` (the sink +
relay), plus a driver that seeds the opening and runs the bus until the
closing is seen (complete) or both actors pass consecutively (a stall:
incomplete). Actors reply fire-and-forget via the bus; no synchronised
alternation; anticipation and interjection are first-class.

Every ``accept`` yields at least one proposal (``burst >= 1``, DDT-22): the
actor base emits a PASS when ``next_events`` yields nothing, and two
consecutive PASSes end the run as a stall. Tests use a
:class:`_ScriptedActor` that emits scripted **bursts** of replies (one burst
per ``accept``), then PASSes once exhausted — modelling one-or-many replies
per accept with deterministic control over the messy relay.
"""

from __future__ import annotations

from typing import cast

import pytest

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, Role
from training.dialogue.runner import (
    Divergence,
    PASS_SIGNATURE,
    is_pass,
    pass_event,
    run,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _kv(sig: int, band: int) -> KValue:
    return KValue(KLine(sig), band)


def _turn(role: str, sig: int, band: int) -> DecodedTurn:
    return DecodedTurn(role=cast(Role, role), op="IDENTITY", value=_kv(sig, band))


def _ev(role: str, sig: int, band: int) -> RationaliseEvent:
    return RationaliseEvent(
        kind="frame", query=_kv(sig, band), proposal=_kv(sig, band), role=role
    )


def _decoded(
    opening: tuple[str, int, int],
    middles: list[tuple[str, int, int]],
    closing: tuple[str, int, int],
) -> list[DecodedTurn]:
    return [_turn(*opening)] + [_turn(*m) for m in middles] + [_turn(*closing)]


def _bursts(*events: RationaliseEvent) -> list[list[RationaliseEvent]]:
    """Wrap each event as its own single-event burst (one reply per accept)."""
    return [[e] for e in events]


class _ScriptedActor:
    """An actor that emits scripted bursts of replies across ``accept``.

    Holds a sink (injected at construction, as KAgent holds an adapter). Each
    ``accept`` consumes the next burst (a list of events) and publishes the
    whole burst to the sink via ``on_burst`` — modelling one-or-many replies
    per accept as a single bus payload. When the burst list is exhausted,
    ``accept`` publishes a single-PASS burst
    (:func:`~training.dialogue.runner.pass_event`) — the ``burst >= 1``
    contract (DDT-22): a compliant actor never replies zero. Two consecutive
    PASSes (both actors exhausted) end the run as a stall.
    """

    def __init__(self, role: str, bursts: list[list[RationaliseEvent]], sink=None):
        self._role = cast(Role, role)
        self._bursts = [list(b) for b in bursts]
        self._i = 0
        self._sink = sink

    @property
    def role(self) -> str:
        return self._role

    def accept(self, incoming) -> None:  # type: ignore[no-untyped-def]
        if self._sink is None:
            return
        if self._i >= len(self._bursts):
            # Exhausted: ``burst >= 1`` — emit a PASS rather than silence.
            self._sink.on_burst([pass_event(self._role)])
            return
        burst = self._bursts[self._i]
        self._i += 1
        if not burst:
            # An explicit empty burst is also a PASS (one-or-many per accept).
            self._sink.on_burst([pass_event(self._role)])
            return
        self._sink.on_burst(burst)


def _run(
    decoded: list[DecodedTurn],
    trainer_bursts: list[list[RationaliseEvent]],
    trainee_bursts: list[list[RationaliseEvent]],
    *,
    on_divergence: str = "fail",
):
    """Construct actors (via factories) + runner and drive to completion."""
    runner = run(
        decoded,
        lambda sink: _ScriptedActor("T", trainer_bursts, sink=sink),
        lambda sink: _ScriptedActor("K", trainee_bursts, sink=sink),
        on_divergence=on_divergence,
    )
    return runner, runner.run()


# ── The runner drives the bus and tracks coverage (not a judge) ────────


def test_runner_drives_the_exchange_via_bus():
    """The runner drives the bus: the seed fires, replies relay, and the run
    terminates. It is not a judge — assert coverage (displacement) and that the
    exchange ran, not a verdict."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert len(res.events) >= 2  # the exchange ran
    assert res.uncovered == []  # full coverage — zero displacement


def test_runner_holds_no_actor_coupling_state():
    """The runner has no per-actor cursors / turn tracking."""
    runner = run(
        _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)),
        lambda sink: _ScriptedActor("T", [], sink=sink),
        lambda sink: _ScriptedActor("K", [], sink=sink),
    )
    for attr in ("_cursor", "_turn", "_next_role", "_pacing", "_whos_turn"):
        assert not hasattr(runner, attr)


# ── Content matching + idempotent coverage ──────────────────────────────


def test_coverage_row_emission_marks_covered():
    """An emission matching a coverage row marks it covered; full coverage is
    zero displacement."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.uncovered == []


def test_duplicate_content_collapses_idempotently():
    """Duplicate table rows collapse to one distinct content; re-emitting
    covered content is not divergence."""
    decoded = _decoded(
        ("T", 1, 1), [("K", 2, 2), ("K", 2, 2), ("T", 3, 3)], ("T", 9, 9)
    )
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.unmatched == []


def test_role_mismatch_is_immediate_divergence():
    """Matching is same-role; a K emission whose content matches a T-only row
    has a different (role,kline,sig) key → immediate divergence."""
    decoded = _decoded(("T", 1, 1), [("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 3, 3)),
        on_divergence="accept",
    )
    assert len(res.unmatched) == 1


# ── DDT-9: divergence policy ──────────────────────────────────────────────


def test_divergence_fail_raises_on_caller_thread():
    """DDT-9: on_divergence='fail' raises Divergence on the caller's thread
    (captured from the bus dispatch thread and re-raised by run())."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner = run(
        decoded,
        lambda sink: _ScriptedActor("T", _bursts(_ev("T", 1, 1), _ev("T", 9, 9)), sink=sink),
        lambda sink: _ScriptedActor("K", _bursts(_ev("K", 99, 99)), sink=sink),  # nothing matches
        on_divergence="fail",
    )
    with pytest.raises(Divergence) as exc_info:
        runner.run()
    assert exc_info.value.role == "K"


def test_divergence_accept_records_and_continues():
    """on_divergence='accept' records an immediate divergence to unmatched.
    Under the stop-on-divergence rule the run halts at the divergent emission,
    so the later legit K(2,2) is never reached (it surfaces as displacement)."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K emits off-script K(99) then legit K(2,2).
        trainee_bursts=_bursts(_ev("K", 99, 99), _ev("K", 2, 2)),
        on_divergence="accept",
    )
    assert len(res.unmatched) == 1
    assert res.unmatched[0].proposal.kline.signature == 99
    # The run stopped at K(99,99); K(2,2) was never covered = displacement.
    assert {t.value.kline.signature for t in res.uncovered} == {2}


# ── Duplicate-key exhaustion (coverage as a per-key budget) ────────────────


def test_coverage_budget_counts_duplicate_rows():
    """The coverage set is a per-key budget: two authored copies of K(2,2) let
    the run emit K(2,2) twice without divergence."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        # K emits its budget of two K(2,2) copies — exactly authored.
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.unmatched == []
    assert res.uncovered == []


def test_over_budget_emission_is_exhaustion_divergence():
    """The headline case: the table authors two X keys, the run emits three →
    the third is immediate divergence (duplicate-key exhaustion), not a clean
    match. The three emissions ride one burst so all are matched before the
    burst-boundary exhaustion check."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K emits three K(2,2) in one burst — one more than the budget of two.
        trainee_bursts=[[_ev("K", 2, 2), _ev("K", 2, 2), _ev("K", 2, 2)]],
        on_divergence="accept",
    )
    assert len(res.unmatched) == 1
    assert res.unmatched[0].proposal.kline.signature == 2


def test_exhaustion_divergence_stops_immediately_under_accept():
    """Duplicate-key exhaustion is terminal under *both* policies: once the
    budget is depleted, further over-budget copies in the same burst are not
    recorded — the run stops at the first over-budget copy."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K emits FOUR K(2,2) in one burst — two over budget. Only the first
        # over-budget copy is recorded; the run halts there.
        trainee_bursts=[
            [_ev("K", 2, 2), _ev("K", 2, 2), _ev("K", 2, 2), _ev("K", 2, 2)]
        ],
        on_divergence="accept",
    )
    assert len(res.unmatched) == 1
    assert res.unmatched[0].proposal.kline.signature == 2


def test_unmatched_divergence_stops_immediately_under_accept():
    """An unmatched divergence also stops the run immediately under accept:
    a second off-table emission in the same burst is not recorded. The
    ``on_divergence`` policy governs raise-vs-record, not whether to stop."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K emits two off-table K(99) in one burst; only the first is recorded.
        trainee_bursts=[[_ev("K", 99, 99), _ev("K", 99, 99)]],
        on_divergence="accept",
    )
    assert len(res.unmatched) == 1
    assert res.unmatched[0].proposal.kline.signature == 99


def test_exhaustion_divergence_under_fail_carries_reason():
    """Under on_divergence='fail', over-budget emission raises a Divergence
    whose ``reason`` is 'exhausted' (distinguished from 'unmatched')."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner = run(
        decoded,
        lambda sink: _ScriptedActor(
            "T", _bursts(_ev("T", 1, 1), _ev("T", 9, 9)), sink=sink
        ),
        # K emits K(2,2) twice in one burst — budget is one.
        lambda sink: _ScriptedActor(
            "K", [[_ev("K", 2, 2), _ev("K", 2, 2)]], sink=sink
        ),
        on_divergence="fail",
    )
    with pytest.raises(Divergence) as exc_info:
        runner.run()
    assert exc_info.value.reason == "exhausted"


def test_unmatched_divergence_reason_is_unmatched():
    """A divergence for content present nowhere has reason 'unmatched'."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner = run(
        decoded,
        lambda sink: _ScriptedActor(
            "T", _bursts(_ev("T", 1, 1), _ev("T", 9, 9)), sink=sink
        ),
        lambda sink: _ScriptedActor("K", _bursts(_ev("K", 99, 99)), sink=sink),
        on_divergence="fail",
    )
    with pytest.raises(Divergence) as exc_info:
        runner.run()
    assert exc_info.value.reason == "unmatched"


def test_last_coverage_event_recorded_on_result():
    """RunResult.last_coverage_event is the last emission that consumed a
    coverage allowance — the last healthy point."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    # The last coverage-consuming emission is T(3,3) (the close T(9,9) is not
    # a coverage match).
    assert res.last_coverage_event is not None
    assert res.last_coverage_event.proposal.kline.signature == 3


def test_last_coverage_event_carried_on_exhaustion_divergence():
    """Divergence.last_coverage_event anchors where the run was healthy
    before the over-budget emission."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("K", 2, 2)], ("T", 9, 9))
    runner = run(
        decoded,
        lambda sink: _ScriptedActor(
            "T", _bursts(_ev("T", 1, 1), _ev("T", 9, 9)), sink=sink
        ),
        # K emits three K(2,2) in one burst; the third exhausts the budget.
        lambda sink: _ScriptedActor(
            "K", [[_ev("K", 2, 2), _ev("K", 2, 2), _ev("K", 2, 2)]], sink=sink
        ),
        on_divergence="fail",
    )
    with pytest.raises(Divergence) as exc_info:
        runner.run()
    assert exc_info.value.last_coverage_event is not None
    assert exc_info.value.last_coverage_event.proposal.kline.signature == 2


def test_last_coverage_event_none_when_nothing_covered():
    """When no coverage content was ever matched, last_coverage_event is None.
    The trainer diverges on its very first (opening) emission, so no coverage
    allowance is ever consumed."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner = run(
        decoded,
        # T opens with off-table content T(99,99) — diverges immediately.
        lambda sink: _ScriptedActor("T", _bursts(_ev("T", 99, 99)), sink=sink),
        lambda sink: _ScriptedActor("K", [], sink=sink),
        on_divergence="accept",
    )
    res = runner.run()
    assert res.last_coverage_event is None


# ── Close: de-positional, any agent, any time ─────────────────────────────


def test_close_emission_terminates_run():
    """The close content, emitted by any agent at any time, terminates the run."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert res.uncovered == []


def test_close_can_come_first():
    """The close is de-positional: emitting it first (anticipation) terminates
    the run immediately. The displacement (uncovered coverage rows) is the
    signal of how much of the exchange was skipped — not a verdict."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T emits opening and close in one burst.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 9, 9)]],
        trainee_bursts=[],
    )
    uncovered_sigs = {t.value.kline.signature for t in res.uncovered}
    assert {2, 3} <= uncovered_sigs  # the skipped coverage rows = displacement


def test_entry_exhaustion_terminates_run():
    """When the coverage set is fully covered, the run terminates (entry
    exhaustion) even without the close being emitted. The runner is not a
    judge: close-vs-exhaustion is not an important distinction."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens (covers T(1,1)); never emits the close.
        trainer_bursts=_bursts(_ev("T", 1, 1)),
        # K covers K(2,2). Now every coverage row is covered → exhaustion.
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert res.uncovered == []  # full coverage


def test_mutual_pass_terminates_with_displacement():
    """Two consecutive PASSes from the two roles is a terminal condition (the
    actors have nothing more to say). It is not a verdict: any coverage rows
    never emitted are reported as displacement."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens, then exhausts → PASS.
        trainer_bursts=_bursts(_ev("T", 1, 1)),
        # K exhausts immediately → PASS.
        trainee_bursts=[],
    )
    uncovered_sigs = {t.value.kline.signature for t in res.uncovered}
    assert {2, 3, 9} & uncovered_sigs  # displacement: coverage never traversed


def test_emission_after_terminal_is_not_divergence():
    """The run is closed mechanically on the first terminal emission. The bus
    dispatches role handlers before wildcards, so a role handler may react to a
    terminal emission and enqueue another *before* the wildcard marks the run
    closed. Such a trailing emission must NOT be treated as divergence.
    Regression for the SynthesizingTrainer-vs-TableTrainee run."""
    # A T coverage row (T 3,3) lets T's 2nd accept match; T's 3rd accept reacts
    # to the K close (9,9) by emitting an off-table T(7,7) "ratification".
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("K", 9, 9))
    runner = run(
        decoded,
        lambda sink: _ScriptedActor(
            "T", _bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 7, 7)), sink=sink
        ),
        lambda sink: _ScriptedActor(
            "K", _bursts(_ev("K", 2, 2), _ev("K", 9, 9)), sink=sink
        ),
        on_divergence="fail",
    )
    res = runner.run()
    assert res.unmatched == []  # the trailing T(7,7) was not divergence


# ── Anticipation + interjection (permitted, unflagged) ───────────────────


def test_anticipation_permitted_and_unflagged():
    """A trainer emits its coverage row before the authored causal order; a
    normal match, not divergence."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.unmatched == []


def test_interjection_within_budget_is_permitted():
    """An actor may interject covered content up to its authored budget.
    The table authors one K(2,2); emitting it exactly once is a normal match.
    (Emitting *more* copies than authored is duplicate-key exhaustion — see
    the exhaustion tests below.)"""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K interjects K(2,2) once — within its budget of one.
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert res.unmatched == []


# ── No synchronised alternation, route-to-other ───────────────────────────


def test_pass_then_burst_relayed_correctly():
    """An actor may reply one-or-many per accept; the bus relays each. Here the
    trainee PASSes (an empty accept yields a PASS under ``burst >= 1``), and the
    trainer emits the whole exchange in a single accept burst — terminating
    the run without the trainee ever emitting substance (extreme trainer
    autonomy)."""
    decoded = _decoded(("T", 1, 1), [("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T emits everything in its opening accept: open, coverage, close.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)]],
        trainee_bursts=[[]],
    )
    assert res.uncovered == []  # the coverage row was traversed


# ── PASS (burst >= 1) + mutual-PASS termination ─────────────────────────


def test_pass_is_not_coverage_or_divergence():
    """A PASS is intercepted before matching — neither coverage nor divergence.
    A trainee that PASSes (nothing workable) while the trainer drives the
    exchange runs cleanly with no unmatched emission."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K PASSes once (empty accept), then covers the coverage row.
        trainee_bursts=[[], _bursts(_ev("K", 2, 2))[0]],
    )
    assert res.unmatched == []
    # The PASS is recorded in arrival-ordered events but never matched.
    passes = [e for e in res.events if is_pass(e)]
    assert len(passes) == 1
    assert passes[0].proposal.kline.signature == PASS_SIGNATURE


def test_mutual_pass_terminates():
    """Two consecutive PASSes from the two roles is a terminal condition (the
    actors have nothing more to say). It is mechanical termination, not a
    verdict; the displacement is whatever coverage was never traversed."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens (sig 1), then exhausts → PASS.
        trainer_bursts=_bursts(_ev("T", 1, 1)),
        # K exhausts immediately → PASS.
        trainee_bursts=[],
    )
    # The two trailing emissions are both PASSes (the two roles passing).
    trailing = res.events[-2:]
    assert all(is_pass(e) for e in trailing)


def test_pass_event_builder_has_sentinel_signature_at_s1():
    """pass_event() builds {PASS:[]} at S1 — the sentinel signature on an
    identity kline, top-band significance."""
    from kalvin.expand import SIG_S1

    ev = pass_event("K")
    assert ev.role == "K"
    assert ev.proposal.kline.signature == PASS_SIGNATURE
    assert ev.proposal.kline.nodes == []
    assert ev.proposal.significance == SIG_S1
    assert is_pass(ev)


# ── Arrival-ordered log + displacement ──────────────────────────────────


def test_result_events_are_arrival_ordered():
    """events are in arrival order (bus delivery order); the seed emission first."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    sigs = [e.proposal.kline.signature for e in res.events]
    assert sigs[0] == 1  # the seed emission first


def test_result_uncovered_reports_displacement():
    """uncovered lists the coverage rows never emitted — the displacement."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens and closes immediately; coverage rows never traversed.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 9, 9)]],
        trainee_bursts=[],
    )
    uncovered_sigs = {t.value.kline.signature for t in res.uncovered}
    assert {2, 3} <= uncovered_sigs


# ── Construction validation ───────────────────────────────────────────────


def test_run_validates_on_divergence():
    with pytest.raises(ValueError):
        run(
            _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)),
            lambda sink: _ScriptedActor("T", [], sink=sink),
            lambda sink: _ScriptedActor("K", [], sink=sink),
            on_divergence="bogus",
        )


def test_run_rejects_too_few_turns():
    with pytest.raises(ValueError):
        run(
            [_turn("T", 1, 1)],
            lambda sink: _ScriptedActor("T", [], sink=sink),
            lambda sink: _ScriptedActor("K", [], sink=sink),
        )
