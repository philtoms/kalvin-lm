"""Phase 2 — peer runner sink types: ``PeerDivergence`` and ``PeerRunResult``.

Spec: ``@specs/peer-dialogue.md`` PDT-14, PDT-15. These are the peer-regime
analogues of the synchronous ``ActorDivergence`` / ``RunResult``. They carry
peer-shaped data (an unconsumed set; arrival-ordered events) that the
synchronous types' cursor-shaped fields cannot express.

The sink behaviour itself (Phases 3) is exercised in
``tests/test_peer_runner.py``. These tests pin the types' shape and defaults.
"""

from __future__ import annotations

from kalvin.events import RationaliseEvent
from kalvin.kline import KLine
from kalvin.kvalue import KValue
from training.dialogue.decoder import DecodedTurn
from training.dialogue.peer_runner import PeerDivergence, PeerRunResult


def _kv(sig: int = 1) -> KValue:
    return KValue(KLine(sig), 0)


def _turn(role: str, sig: int) -> DecodedTurn:
    from typing import cast

    from training.dialogue.decoder import Role

    return DecodedTurn(role=cast(Role, role), op="IDENTITY", value=_kv(sig))


def _ev(role: str, sig: int) -> RationaliseEvent:
    return RationaliseEvent(kind="frame", query=_kv(sig), proposal=_kv(sig), role=role)


# ── PDT-14: PeerDivergence ────────────────────────────────────────────────


def test_peer_divergence_carries_role_emitted_unconsumed():
    """PDT-14: carries (role, emitted, unconsumed) and a descriptive message."""
    emitted = _kv(7)
    unconsumed = (_turn("T", 1), _turn("T", 2))
    err = PeerDivergence(role="T", emitted=emitted, unconsumed=unconsumed)
    assert err.role == "T"
    assert err.emitted is emitted
    assert err.unconsumed == unconsumed
    assert "T" in str(err) and "divergence" in str(err)


def test_peer_divergence_is_distinct_from_actor_divergence():
    """PDT-14: a separate type, not the synchronous ActorDivergence."""
    from training.dialogue.runner import ActorDivergence

    assert PeerDivergence is not ActorDivergence
    assert not issubclass(PeerDivergence, ActorDivergence)


def test_peer_divergence_has_no_cursor_field():
    """PDT-14: peer divergence carries no cursor (unlike ActorDivergence)."""
    err = PeerDivergence(role="K", emitted=_kv(1), unconsumed=())
    assert not hasattr(err, "cursor")


# ── PDT-15: PeerRunResult ─────────────────────────────────────────────────


def test_peer_run_result_defaults_are_empty_and_incomplete():
    """PDT-15: a fresh result is empty and not complete/covered."""
    r = PeerRunResult()
    assert r.events == []
    assert r.complete is False
    assert r.covered is False
    assert r.unmatched == []
    assert r.uncovered == []


def test_peer_run_result_is_distinct_from_run_result():
    """PDT-15: a separate type, not the synchronous RunResult."""
    from training.dialogue.runner import RunResult

    assert PeerRunResult is not RunResult


def test_peer_run_result_events_append_in_arrival_order():
    """PDT-15: events are arrival-ordered — the order they were appended."""
    r = PeerRunResult()
    e1, e2, e3 = _ev("K", 1), _ev("T", 2), _ev("K", 3)
    r.events.extend([e1, e2, e3])
    assert r.events == [e1, e2, e3]
