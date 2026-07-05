"""Phase 3 — the PeerRunner as a MessageBus subscriber.

Spec: ``@specs/peer-dialogue.md`` PDT-5..PDT-19. The runner is a coverage-
tracking wildcard subscriber over a ``MessageBus`` (the sink + relay), plus a
driver that seeds the opening and runs the bus until the closing is seen.
Actors reply fire-and-forget via the bus; no synchronised alternation;
anticipation and interjection are first-class.

Tests use a :class:`_ScriptedActor` that emits scripted **bursts** of replies
(one burst per ``accept``, each burst a list of events), giving deterministic
control over the messy relay including zero-or-many replies per accept.
"""

from __future__ import annotations

from typing import cast

import pytest

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn, Role
from training.dialogue.peer_runner import (
    PeerDivergence,
    run_peer,
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
    event in it via the sink — modelling zero-or-many replies per accept. When
    the burst list is exhausted, ``accept`` replies zero times.
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
        if self._i >= len(self._bursts) or self._sink is None:
            return  # exhausted or no sink — reply zero times
        burst = self._bursts[self._i]
        self._i += 1
        for reply in burst:
            self._sink.on_event(reply)


def _run(
    decoded: list[DecodedTurn],
    trainer_bursts: list[list[RationaliseEvent]],
    trainee_bursts: list[list[RationaliseEvent]],
    *,
    on_divergence: str = "fail",
    idle_timeout: float = 1.0,
):
    """Construct actors (via factories) + runner and drive to completion."""
    runner = run_peer(
        decoded,
        lambda sink: _ScriptedActor("T", trainer_bursts, sink=sink),
        lambda sink: _ScriptedActor("K", trainee_bursts, sink=sink),
        on_divergence=on_divergence,
        idle_timeout=idle_timeout,
    )
    return runner, runner.run()


# ── PDT-5/PDT-6: bus subscriber, coverage-only state ─────────────────────


def test_runner_drives_to_completion_via_bus():
    """PDT-5: the runner drives the bus; opening seeds, replies relay, closing
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
    """PDT-6: the runner has no per-actor cursors / turn tracking."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)),
        lambda sink: _ScriptedActor("T", [], sink=sink),
        lambda sink: _ScriptedActor("K", [], sink=sink),
    )
    for attr in ("_cursor", "_turn", "_next_role", "_pacing", "_whos_turn"):
        assert not hasattr(runner, attr)


# ── PDT-7/PDT-8: content matching + idempotent coverage ──────────────────


def test_middle_emission_marks_covered():
    """PDT-7: an emission matching a distinct middle content marks it covered."""
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
    """PDT-8: duplicate table rows collapse to one distinct content; re-emitting
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
    """PDT-7: matching is same-role; a K emission whose content matches a T-only
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


# ── PDT-9: divergence policy ──────────────────────────────────────────────


def test_divergence_fail_raises_on_caller_thread():
    """PDT-9: on_divergence='fail' raises PeerDivergence on the caller's thread
    (captured from the bus dispatch thread and re-raised by run())."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner = run_peer(
        decoded,
        lambda sink: _ScriptedActor("T", _bursts(_ev("T", 1, 1), _ev("T", 9, 9)), sink=sink),
        lambda sink: _ScriptedActor("K", _bursts(_ev("K", 99, 99)), sink=sink),  # nothing matches
        on_divergence="fail",
        idle_timeout=1.0,
    )
    with pytest.raises(PeerDivergence) as exc_info:
        runner.run()
    assert exc_info.value.role == "K"


def test_divergence_accept_records_and_continues():
    """PDT-9: on_divergence='accept' records to unmatched and continues."""
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


# ── PDT-10/PDT-13: closing + completion ───────────────────────────────────


def test_closing_emission_completes_run():
    """PDT-10/PDT-13: the closing emission marks closing-seen = completion."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1), _ev("T", 9, 9)),
        trainee_bursts=_bursts(_ev("K", 2, 2)),
    )
    assert res.complete


def test_extreme_anticipation_closing_first_is_complete():
    """PDT-13: closing-first is technically complete. The trainer emits opening
    AND closing in a single accept burst (anticipates past the whole middle)."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T opens and closes in one burst — zero-or-many replies per accept.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 9, 9)]],
        trainee_bursts=[],
    )
    assert res.complete
    assert not res.covered  # middle unseen → coverage diagnostic False


def test_no_closing_means_incomplete():
    """PDT-13/PDT-19: without the closing, the run stalls → incomplete."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1)),  # opens but never closes
        trainee_bursts=_bursts(_ev("K", 2, 2)),
        idle_timeout=0.5,
    )
    assert not res.complete
    assert res.covered  # middle covered, but closing never arrived


# ── PDT-11/PDT-12: anticipation + interjection ───────────────────────────


def test_anticipation_permitted_and_unflagged():
    """PDT-11: the trainer emits its middle before the authored causal order;
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
    """PDT-11: an actor may emit extra covered content (interjection);
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


# ── PDT-18: no synchronised alternation, route-to-other ───────────────────


def test_zero_replies_then_burst_relayed_correctly():
    """PDT-18: an actor may reply zero-or-many per accept; the bus relays each.
    Here the trainee replies zero, and the trainer emits opening + middle +
    closing in a single accept burst — terminating the run without the trainee
    ever emitting (extreme trainer autonomy)."""
    decoded = _decoded(("T", 1, 1), [("T", 3, 3)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        # T emits everything in its opening accept: open, middle, close.
        trainer_bursts=[[_ev("T", 1, 1), _ev("T", 3, 3), _ev("T", 9, 9)]],
        # K never emits — but T's burst self-terminates via the closing.
        trainee_bursts=[[]],
    )
    assert res.complete
    assert res.covered


# ── PDT-19: idle timeout ──────────────────────────────────────────────────


def test_idle_timeout_ends_stalled_run_as_incomplete():
    """PDT-19: silence before the closing → idle timeout → complete=False
    (non-fatal, surfaced in result)."""
    decoded = _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9))
    runner, res = _run(
        decoded,
        trainer_bursts=_bursts(_ev("T", 1, 1)),  # opens, then both go silent
        trainee_bursts=_bursts(_ev("K", 2, 2)),
        idle_timeout=0.3,
    )
    assert not res.complete  # stalled
    assert res.covered


# ── PDT-15: arrival-ordered events + diagnostics ──────────────────────────


def test_result_events_are_arrival_ordered():
    """PDT-15: events are in arrival order (bus delivery order); opening first."""
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
    """PDT-15: uncovered lists distinct middle contents never seen."""
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


def test_run_peer_validates_on_divergence():
    with pytest.raises(ValueError):
        run_peer(
            _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)),
            lambda sink: _ScriptedActor("T", [], sink=sink),
            lambda sink: _ScriptedActor("K", [], sink=sink),
            on_divergence="bogus",
        )


def test_run_peer_rejects_too_few_turns():
    with pytest.raises(ValueError):
        run_peer(
            [_turn("T", 1, 1)],
            lambda sink: _ScriptedActor("T", [], sink=sink),
            lambda sink: _ScriptedActor("K", [], sink=sink),
        )
