"""Phase 2 — runner sink types: ``Divergence`` and ``RunResult``.

Spec: ``@specs/dialogue-runner.md`` PDT-14, PDT-15. They carry run-shaped data
(an unconsumed set; arrival-ordered events) and pin the types' shape and
defaults.

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


# ── PDT-14: Divergence ────────────────────────────────────────────────


def test_divergence_carries_role_emitted_unconsumed():
    """PDT-14: carries (role, emitted, unconsumed) and a descriptive message."""
    emitted = _kv(7)
    unconsumed = (_turn("T", 1), _turn("T", 2))
    err = Divergence(role="T", emitted=emitted, unconsumed=unconsumed)
    assert err.role == "T"
    assert err.emitted is emitted
    assert err.unconsumed == unconsumed
    assert "T" in str(err) and "divergence" in str(err)


def test_divergence_has_no_cursor_field():
    """PDT-14: divergence carries no cursor — coverage is content-keyed."""
    err = Divergence(role="K", emitted=_kv(1), unconsumed=())
    assert not hasattr(err, "cursor")


# ── PDT-15: RunResult ─────────────────────────────────────────────────


def test_run_result_defaults_are_empty_and_incomplete():
    """PDT-15: a fresh result is empty and not complete/covered."""
    r = RunResult()
    assert r.events == []
    assert r.complete is False
    assert r.covered is False
    assert r.unmatched == []
    assert r.uncovered == []


def test_run_result_events_append_in_arrival_order():
    """PDT-15: events are arrival-ordered — the order they were appended."""
    r = RunResult()
    e1, e2, e3 = _ev("K", 1), _ev("T", 2), _ev("K", 3)
    r.events.extend([e1, e2, e3])
    assert r.events == [e1, e2, e3]
