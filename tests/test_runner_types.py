"""Phase 2 — runner sink types: ``Divergence`` and ``RunResult``.

They carry run-shaped data (an unconsumed set; arrival-ordered events;
displacement) and pin the types' shape and defaults. The runner is not a
judge: ``RunResult`` carries no complete/covered verdict, only the log,
immediate divergences, and displacement.

The sink behaviour itself (Phases 3) is exercised in
``tests/test_runner.py``.
"""

from __future__ import annotations

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.runner import Divergence, RunResult


def _kv(sig: int = 1) -> KValue:
    return KValue(KLine(sig), 0)


def _turn(role: str, sig: int) -> DecodedTurn:
    from typing import cast

    from training.dialogue.decoder import Role

    return DecodedTurn(role=cast(Role, role), op="IDENTITY", value=_kv(sig))


def _ev(role: str, sig: int) -> RationaliseEvent:
    return RationaliseEvent(kind="frame", query=_kv(sig), proposal=_kv(sig), role=role)


# ── Divergence ────────────────────────────────────────────────────────


def test_divergence_carries_role_emitted_unconsumed():
    """carries (role, emitted, unconsumed) and a descriptive message."""
    emitted = _kv(7)
    unconsumed = (_turn("T", 1), _turn("T", 2))
    err = Divergence(role="T", emitted=emitted, unconsumed=unconsumed)
    assert err.role == "T"
    assert err.emitted is emitted
    assert err.unconsumed == unconsumed
    assert err.reason == "unmatched"  # default
    assert err.last_coverage_event is None  # default
    assert "T" in str(err) and "divergence" in str(err)


def test_divergence_exhausted_reason_message():
    """reason='exhausted' distinguishes duplicate-key exhaustion and shapes the message."""
    err = Divergence(
        role="K",
        emitted=_kv(2),
        unconsumed=(_turn("K", 3),),
        reason="exhausted",
    )
    assert err.reason == "exhausted"
    assert "exhausts its coverage budget" in str(err)


def test_divergence_has_no_cursor_field():
    """divergence carries no cursor — coverage is content-keyed."""
    err = Divergence(role="K", emitted=_kv(1), unconsumed=())
    assert not hasattr(err, "cursor")


# ── RunResult ─────────────────────────────────────────────────────────


def test_run_result_defaults_are_empty():
    """a fresh result is empty — no verdict fields, just the log/divergence/
    displacement records."""
    r = RunResult()
    assert r.events == []
    assert r.unmatched == []
    assert r.uncovered == []
    assert r.last_coverage_event is None
    assert not hasattr(r, "complete")
    assert not hasattr(r, "covered")


def test_run_result_events_append_in_arrival_order():
    """events are arrival-ordered — the order they were appended."""
    r = RunResult()
    e1, e2, e3 = _ev("K", 1), _ev("T", 2), _ev("K", 3)
    r.events.extend([e1, e2, e3])
    assert r.events == [e1, e2, e3]
