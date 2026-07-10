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
    ``accept`` consumes the next burst (a list of events) and publishes every
    event in it via the sink — modelling one-or-many replies per accept. When
    the burst list is exhausted, ``accept`` publishes a PASS
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

    def accept(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._sink is None:
            return
        if self._i >= len(self._bursts):
            # Exhausted: ``burst >= 1`` — emit a PASS rather than silence.
            self._sink.on_event(pass_event(self._role))
            return
        burst = self._bursts[self._i]
        self._i += 1
        if not burst:
            # An explicit empty burst is also a PASS (one-or-many per accept).
            self._sink.on_event(pass_event(self._role))
            return
        for reply in burst:
            self._sink.on_event(reply)


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


# ── DDT-5/DDT-6: bus subscriber, coverage-only state ─────────────────────


def test_runner_drives_to_completion_via_bus():
    """DDT-5: the runner drives the bus; opening seeds, replies relay, closing
    terminates. Trainer opens, trainee covers the middle, trainer closes."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens, then (after K's turn) closes.
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert res.complete
    assert res.covered


def test_runner_holds_no_actor_coupling_state():
    """DDT-6: the runner has no per-actor cursors / turn tracking."""
    runner = run(
        _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)),
        lambda sink: _ScriptedActor("T", [], sink=sink),
        lambda sink: _ScriptedActor("K", [], sink=sink),
    )
    for attr in ("_cursor", "_turn", "_next_role", "_pacing", "_whos_turn"):
        assert not hasattr(runner, attr)


# ── DDT-7/DDT-8: content matching + idempotent coverage ──────────────────


def test_middle_emission_marks_covered():
    """DDT-7: an emission matching a distinct middle content marks it covered."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens, covers its middle T(3,3), then closes (3 accept calls).
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        # K covers its middle K(2,2), then re-emits to give T its closing turn.
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.covered
    assert res.complete


def test_duplicate_middle_content_collapses_idempotently():
    """DDT-8: duplicate table rows collapse to one distinct content; re-emitting
    covered content is not divergence."""
    decoded = _decoded(
        ("T", 1, 1), [("K", 2, 2), ("K", 2, 2), ("T", 3, 3)], ("T", 9, 9)
    )
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        # K emits K(2,2) twice (table has it twice) — idempotent, not divergence.
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.complete
    assert res.unmatched == []


def test_role_mismatch_is_divergence():
    """DDT-7: matching is same-role; a K emission whose content matches a T-only
    middle has a different (role,kline,sig) key → divergence."""
    decoded = _decoded(("T", 1, 1), [("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K emits K(3,3): content is T-only → no same-role match → unmatched.
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
    """DDT-9: on_divergence='accept' records to unmatched and continues."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K emits off-script K(99) then legit K(2,2).
        trainee_bursts=_bursts(_ev("K", 99, 99), _ev("K", 2, 2)),
        on_divergence="accept",
    )
    assert res.complete
    assert len(res.unmatched) == 1
    assert res.unmatched[0].proposal.kline.signature == 99


# ── DDT-10/DDT-13: closing + completion ───────────────────────────────────


def test_closing_emission_completes_run():
    """DDT-10/DDT-13: the closing emission marks closing-seen = completion."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert res.complete


def test_extreme_anticipation_closing_first_is_complete():
    """DDT-13: closing-first is technically complete. The trainer emits opening
    AND closing in a single accept burst (anticipates past the whole middle)."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens and closes in one burst — one-or-many replies per accept.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 9, 9)]],
        trainee_bursts=[],
    )
    assert res.complete
    assert not res.covered  # middle unseen → coverage diagnostic False


def test_no_closing_means_incomplete():
    """DDT-13/DDT-22: without the closing, both actors exhaust and PASS in turn
    → mutual PASS stall → incomplete."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1)),  # opens but never closes
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert not res.complete
    assert res.covered  # middle covered, but closing never arrived


def test_emission_after_closing_is_not_divergence():
    """DDT-15: the closing is terminal. The bus dispatches role handlers before
    wildcards, so a role handler may react to the closing (e.g. a synthesizing
    trainer ratifying the trainee's closing countersign) and enqueue an
    emission *before* the wildcard marks closing-seen. Such a trailing emission
    matches nothing and must NOT be treated as divergence — the run is already
    complete. Regression for the SynthesizingTrainer-vs-TableTrainee run."""
    # A T middle row (T 3,3) lets T's 2nd accept match; T's 3rd accept reacts to
    # the K closing (9,9) by emitting an off-table T(7,7) "ratification".
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("K", 9, 9))
    runner = run(
        decoded,
        lambda sink: _ScriptedActor(
            "T", _bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 7, 7)), sink=sink
        ),
        # K covers the middle then delivers the closing.
        lambda sink: _ScriptedActor(
            "K", _bursts(_ev("K", 2, 2), _ev("K", 9, 9)), sink=sink
        ),
        on_divergence="fail",
    )
    res = runner.run()
    assert res.complete  # closing seen
    assert res.unmatched == []  # the trailing T(7,7) was not divergence


# ── DDT-11/DDT-12: anticipation + interjection ───────────────────────────


def test_anticipation_permitted_and_unflagged():
    """DDT-11: the trainer emits its middle before the authored causal order;
    a normal match, not divergence."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens, emits its middle T(3,3) early, then closes.
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    assert res.complete
    assert res.unmatched == []


def test_interjection_permitted():
    """DDT-11: an actor may emit extra covered content (interjection);
    idempotent coverage, not divergence."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K interjects K(2,2) twice (more than the table has).
        trainee_bursts=[[_ev("K", 2, 2), _ev("K", 2, 2)]],
    )
    assert res.complete
    assert res.unmatched == []


# ── DDT-18: no synchronised alternation, route-to-other ───────────────────


def test_zero_replies_then_burst_relayed_correctly():
    """DDT-18: an actor may reply one-or-many per accept; the bus relays each.
    Here the trainee PASSes (an empty accept yields a PASS under ``burst >= 1``),
    and the trainer emits opening + middle + closing in a single accept burst —
    terminating the run without the trainee ever emitting substance (extreme
    trainer autonomy)."""
    decoded = _decoded(("T", 1, 1), [("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T emits everything in its opening accept: open, middle, close.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)]],
        # K PASSes — but T's burst self-terminates via the closing.
        trainee_bursts=[[]],
    )
    assert res.complete
    assert res.covered


# ── DDT-22: PASS (burst >= 1) + mutual-PASS stall ─────────────────────────


def test_pass_is_not_recorded_as_coverage_or_divergence():
    """DDT-22: a PASS is intercepted before matching — neither coverage nor
    divergence. A trainee that PASSes (nothing workable) while the trainer
    drives opening + closing completes cleanly with no unmatched emission."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens, then closes.
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        # K PASSes once (empty accept), then covers the middle.
        trainee_bursts=[[], _bursts(_ev("K", 2, 2))[0]],
    )
    assert res.complete
    assert res.unmatched == []
    # The PASS is recorded in arrival-ordered events but never matched.
    passes = [e for e in res.events if is_pass(e)]
    assert len(passes) == 1
    assert passes[0].proposal.kline.signature == PASS_SIGNATURE


def test_mutual_pass_ends_run_incomplete():
    """DDT-22: two consecutive PASSes (each side passing) is the stall
    terminator → the run ends incomplete (closing unseen), with no idle
    timeout. Both actors exhaust after the opening."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens (sig 1), then exhausts → PASS.
        trainer_bursts=_bursts(_ev("T", 1, 1)),
        # K exhausts immediately → PASS.
        trainee_bursts=[],
    )
    assert not res.complete  # mutual-PASS stall, closing never arrived
    # The two trailing emissions are both PASSes (T then K, or K then T).
    trailing = res.events[-2:]
    assert all(is_pass(e) for e in trailing)


def test_pass_event_builder_has_sentinel_signature_at_s1():
    """DDT-22: pass_event() builds {PASS:[]} at S1 — the sentinel signature on
    an identity kline, top-band significance."""
    from kalvin.expand import SIG_S1

    ev = pass_event("K")
    assert ev.role == "K"
    assert ev.proposal.kline.signature == PASS_SIGNATURE
    assert ev.proposal.kline.nodes == []
    assert ev.proposal.significance == SIG_S1
    assert is_pass(ev)


# ── DDT-15: arrival-ordered events + diagnostics ──────────────────────────


def test_result_events_are_arrival_ordered():
    """DDT-15: events are in arrival order (bus delivery order); opening first."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2), _ev("K", 2, 2)),
    )
    sigs = [e.proposal.kline.signature for e in res.events]
    assert sigs[0] == 1  # opening first
    assert 9 in sigs  # closing present


def test_result_uncovered_reports_unseen_middle():
    """DDT-15: uncovered lists distinct middle contents never seen."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens and closes immediately; middle never covered.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 9, 9)]],
        trainee_bursts=[],
    )
    assert res.complete
    assert not res.covered
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
