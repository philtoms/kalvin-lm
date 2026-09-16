"""Ratification: a `==` goal grades the engine's proposals, and the
S1-stamped answer grounds on receipt.

The harness answers a goal-paired proposal at γ of its content against
the goal's target (the goal's signature value) — the reached goal is
ratified at S1, never the S4 decline. The engine honours the stamp: a
stamped-S1 kline grounds on receipt (the stamp, not structure, is the
licence — the reached-goal answer is a misfit in the question's head);
an ask never grounds however stamped.
"""

from __future__ import annotations

from kalvin.bpe_tokenizer import BPETokenizer
from kalvin.kline import ASK_SIG, KLine, KNode
from kalvin.kvalue import KValue
from kalvin.significance import SIG_S1, gamma_to_byte
from kalvin.signifier import NLPSignifier
from dialogue.engine import Engine
from dialogue.engine_state import EngineState
from dialogue.harness import Harness


def bit(n: int) -> KNode:
    return KNode(1 << (32 + n))


M, H, A, L = bit(0), bit(1), bit(2), bit(3)
MHALL = int(M) | int(H) | int(A) | int(L)
WDMH = int(bit(4)) | int(M) | int(H)  # the question's head


def _harness() -> Harness:
    return Harness(BPETokenizer(), Engine(EngineState(NLPSignifier())))


def test_goal_reached_proposal_ratifies_at_s1():
    h = _harness()
    goal = KValue(KLine(MHALL, [M, H, A, L, L]), SIG_S1)
    prop = KValue(KLine(WDMH, [M, H, A, L, L]), 74)  # content == the target
    r = h._grade_proposal(prop, {WDMH: goal})
    assert r is not None and r.significance == SIG_S1


def test_off_goal_proposal_grades_below_s1():
    h = _harness()
    goal = KValue(KLine(MHALL, [M, H, A, L, L]), SIG_S1)
    prop = KValue(KLine(WDMH, [M, H]), 74)  # content mh — off the goal
    r = h._grade_proposal(prop, {WDMH: goal})
    assert r is not None and r.significance < SIG_S1


def test_unpaired_proposal_has_no_grade():
    h = _harness()
    assert h._grade_proposal(KValue(KLine(WDMH, [M, H]), 74), {}) is None


def test_stamped_s1_grounds_on_receipt():
    e = Engine(EngineState(NLPSignifier()))
    prop = KLine(WDMH, [M, H, A, L, L])  # misfit head — never structural
    e.rationalise([KValue(prop, SIG_S1)])
    assert e.state.is_grounded(prop)


def test_stamped_s1_ask_never_grounds():
    e = Engine(EngineState(NLPSignifier()))
    ask = KLine(WDMH | ASK_SIG, [M, H])
    e.rationalise([KValue(ask, SIG_S1)])
    assert not e.state.is_grounded(ask)
