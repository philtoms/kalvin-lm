"""Phase 3 — the PeerRunner sink: matching, completion, divergence, anticipation.

Spec: ``@specs/peer-dialogue.md`` PDT-5..PDT-13. The runner is a sink that
receives emissions, validates each against the unconsumed same-role middle
rows by content (duplicates collapse), watches for the closing, and reports
completion. It carries no actor-coupling state.

Tests build a small decoded peer table by hand (bypassing the script decoder)
and push emissions in various orders.
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

# ── Fixtures: hand-built decoded turns ────────────────────────────────────
#
# Each turn is a (role, sig, significance) triple. The kline is a bare
# identity (sig, no nodes) so two turns differ iff their (role, sig, sig-band)
# triple differs — exactly the content-key axes the matcher uses.


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
    rows = [_turn(*opening)] + [_turn(*m) for m in middles] + [_turn(*closing)]
    return rows


# ── PDT-5: sink contract ──────────────────────────────────────────────────


def test_runner_exposes_receive_complete_result():
    """PDT-5: the sink exposes receive(), complete, result; no driver methods."""
    runner = run_peer(_decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)))
    assert callable(runner.receive)
    assert hasattr(runner, "complete")
    assert hasattr(runner, "result")
    # No respond() / driver-style entry points.
    assert not hasattr(runner, "respond")


def test_run_peer_validates_on_divergence_value():
    with pytest.raises(ValueError):
        run_peer(_decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)), on_divergence="bogus")


def test_run_peer_rejects_too_few_turns():
    with pytest.raises(ValueError):
        run_peer([_turn("T", 1, 1)])


# ── PDT-6: coverage bookkeeping only ──────────────────────────────────────


def test_runner_tracks_coverage_not_actor_state():
    """PDT-6: the runner has no per-actor cursors / turn tracking attributes."""
    runner = run_peer(_decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)))
    # Only coverage bookkeeping should be present; no actor-coupling state.
    for attr in ("_cursor", "_turn", "_actor", "_next_role", "_pacing"):
        assert not hasattr(runner, attr), f"runner has actor-coupling attr {attr!r}"


# ── PDT-7/PDT-8: content matching + duplicate collapse ────────────────────


def test_emission_matching_middle_consumes_it():
    """PDT-7: an emission matching an unconsumed same-role middle row consumes it."""
    runner = run_peer(_decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9)))
    runner.receive(_ev("K", 2, 2))
    assert not runner.complete  # closing + one middle still outstanding
    runner.receive(_ev("T", 3, 3))
    assert not runner.complete  # closing still outstanding
    runner.receive(_ev("T", 9, 9))  # closing
    assert runner.complete


def test_duplicate_rows_collapse_in_one_step():
    """PDT-8: a single emission of X consumes all duplicate X middle rows."""
    # Two identical K middle rows (same role, sig, band).
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2), ("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    )
    runner.receive(_ev("K", 2, 2))  # one emission collapses both K(2,2) rows
    res = runner.result
    # Both K(2,2) middles are now consumed (covered over distinct contents; the
    # single distinct K(2,2) is gone from the unconsumed set).
    assert all(
        not (t.role == "K" and t.value.kline.signature == 2)
        for t in res.uncovered
    )


def test_role_mismatch_is_not_a_match():
    """PDT-7: matching is same-role; a K emission of T-content does not consume it."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("T", 3, 3)], ("T", 9, 9)), on_divergence="accept"
    )
    runner.receive(_ev("K", 3, 3))  # wrong role → divergence (accept-mode records)
    res = runner.result
    assert len(res.unmatched) == 1


# ── PDT-9: divergence policy ──────────────────────────────────────────────


def test_divergence_fail_raises_with_unconsumed_context():
    """PDT-9: on_divergence='fail' raises PeerDivergence naming role + unconsumed."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)), on_divergence="fail"
    )
    with pytest.raises(PeerDivergence) as exc_info:
        runner.receive(_ev("K", 99, 99))  # nothing outstanding for K(99,99)
    err = exc_info.value
    assert err.role == "K"
    # The unconsumed same-role (K) rows at the moment of divergence.
    assert any(t.role == "K" and t.value.kline.signature == 2 for t in err.unconsumed)


def test_divergence_accept_records_and_continues():
    """PDT-9: on_divergence='accept' records to unmatched and continues."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)), on_divergence="accept"
    )
    runner.receive(_ev("K", 99, 99))  # off-script
    runner.receive(_ev("K", 2, 2))     # legitimate middle
    runner.receive(_ev("T", 9, 9))     # closing
    res = runner.result
    assert res.complete is True
    assert len(res.unmatched) == 1
    assert res.unmatched[0].proposal.kline.signature == 99


# ── PDT-10: closing ───────────────────────────────────────────────────────


def test_closing_emission_marks_closing_seen():
    """PDT-10: an emission equal to closing content marks closing-seen."""
    runner = run_peer(_decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)))
    runner.receive(_ev("T", 9, 9))
    assert not runner.complete  # middle not yet covered
    runner.receive(_ev("K", 2, 2))
    assert runner.complete


def test_closing_recognized_regardless_of_middle_order():
    """PDT-10: the closing is recognized whenever its content arrives; the
    decode-time invariant guarantees the closing content cannot collide with a
    middle row, so no closing-vs-middle precedence question arises at the runner."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    )
    runner.receive(_ev("T", 9, 9))  # closing first
    runner.receive(_ev("T", 3, 3))
    runner.receive(_ev("K", 2, 2))
    assert runner.complete


# ── PDT-11/PDT-12: anticipation (permitted, unflagged, middle-only) ────────


def test_anticipation_in_middle_is_permitted_and_unflagged():
    """PDT-11: an actor may emit ahead of the authored causal order; it's a
    normal match, not divergence, not flagged. Here the trainer emits its
    second middle turn before the trainee emits its first."""
    runner = run_peer(
        _decoded(
            ("T", 1, 1),
            [("K", 2, 2), ("T", 3, 3)],  # authored order: K then T
            ("T", 9, 9),
        )
    )
    # Anticipate: T emits its middle before K does.
    runner.receive(_ev("T", 3, 3))
    runner.receive(_ev("K", 2, 2))
    runner.receive(_ev("T", 9, 9))
    res = runner.result
    assert res.complete
    assert res.unmatched == []  # anticipation is not recorded as unmatched


def test_out_of_order_completion_is_valid():
    """PDT-11/PDT-13: the middle can be consumed in any order; completion holds."""
    runner = run_peer(
        _decoded(
            ("T", 1, 1),
            [("K", 2, 2), ("T", 3, 3), ("K", 4, 4)],
            ("T", 9, 9),
        )
    )
    # Fully scrambled middle order.
    runner.receive(_ev("K", 4, 4))
    runner.receive(_ev("T", 3, 3))
    runner.receive(_ev("K", 2, 2))
    runner.receive(_ev("T", 9, 9))
    assert runner.complete


# ── PDT-13: completion ────────────────────────────────────────────────────


def test_completion_requires_closing_and_middle_coverage():
    """PDT-13: complete = closing-seen AND middle distinct-set exhausted."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    )
    # Middle covered, closing not seen.
    runner.receive(_ev("K", 2, 2))
    runner.receive(_ev("T", 3, 3))
    assert not runner.complete
    assert runner.result.covered is True  # covered is the middle-only diagnostic
    # Closing arrives.
    runner.receive(_ev("T", 9, 9))
    assert runner.complete


def test_coverage_alone_does_not_terminate():
    """PDT-13: a run may continue after the middle is covered if the closing
    has not arrived; a further middle emission is then a divergence."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2)], ("T", 9, 9)), on_divergence="accept"
    )
    runner.receive(_ev("K", 2, 2))  # middle covered
    assert runner.result.covered is True
    assert not runner.complete  # closing not seen → run continues
    runner.receive(_ev("K", 50, 50))  # nothing outstanding → unmatched
    res = runner.result
    assert not res.complete
    assert len(res.unmatched) == 1


def test_result_uncovered_reports_outstanding_middle():
    """PDT-15/PDT-13: uncovered lists distinct middle rows never consumed."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    )
    runner.receive(_ev("K", 2, 2))
    res = runner.result
    assert any(t.role == "T" and t.value.kline.signature == 3 for t in res.uncovered)
    assert not any(t.role == "K" and t.value.kline.signature == 2 for t in res.uncovered)


# ── PDT-15: arrival-ordered events ─────────────────────────────────────────


def test_result_events_are_arrival_ordered():
    """PDT-15: events are in arrival order, not table order."""
    runner = run_peer(
        _decoded(("T", 1, 1), [("K", 2, 2), ("T", 3, 3)], ("T", 9, 9))
    )
    runner.receive(_ev("T", 3, 3))
    runner.receive(_ev("K", 2, 2))
    runner.receive(_ev("T", 9, 9))
    res = runner.result
    assert [e.proposal.kline.signature for e in res.events] == [3, 2, 9]
