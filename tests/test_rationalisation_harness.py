"""The dialogue path — the scripted-run hookup (dev/dialogue/harness.py)."""

from __future__ import annotations

from fractions import Fraction

from dev.dialogue.harness import Pending, Runner, _to_kline
from rationalisation import Domain, KLine, can, effort


SCRIPT = """
(Mary had a little lamb)
MHALL == SVO =>
   Subject < M
   Verb < H
   Object < ALL =>
     A = Det
     L = Mod
     L = O
"""


def test_scripts_run_and_ask():
    # Every run is a sequence of asks: one step per ask==goal scenario.
    runner = Runner()
    steps = runner.run(SCRIPT)
    assert len(steps) == 1


def test_the_one_entry_is_the_ask():
    runner = Runner()
    steps = runner.run(SCRIPT)
    assert len(steps[0].kline.nodes) == 5  # the MHALL ask: the full sentence
    # the primed klines are in the domain (they never rationalise as steps)
    assert any(can(k.nodes) == k.signature for k in runner.domain if len(k.nodes) == 1)


def test_a_script_without_an_ask_is_rejected():
    runner = Runner()
    try:
        runner.run("(nothing to ask)")
    except ValueError as exc:
        assert "ask==goal" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_proposal_discharge_is_by_a_later_entry():
    runner = Runner()
    runner.domain = Domain([KLine(1, [1])])  # seed the substrate
    runner._discharge(0, KLine(1, [2]))  # step 0 presents, raises
    p = Pending.__new__(Pending)  # a proposal raised at step 0, target 4
    p.prop, p.raised_at, p.discharged_at = (
        type("P", (), {"kline": KLine(4, []), "kind": "acquire", "effort": Fraction(1)})(),
        0,
        None,
    )
    runner.pending = [p]
    runner._discharge(0, KLine(4, [4]))  # same step: never answers itself
    assert p.discharged_at is None
    runner._discharge(1, KLine(4, [4]))  # a later entry touching 4
    assert p.discharged_at == 1


def test_compiled_entry_maps_to_a_kline():
    runner = Runner()
    steps = runner.run(SCRIPT)
    entry = steps[0]
    assert entry.kline.nodes  # the compiled nodes carried over
    assert can(entry.kline.nodes) != 0
    assert effort(entry.kline) >= 0
