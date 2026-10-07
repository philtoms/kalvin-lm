"""TRS dialogue path — the scripted-run hookup (dev/dialogue/trs_harness.py)."""

from __future__ import annotations

from fractions import Fraction

from dev.dialogue.trs_harness import Pending, TrsRunner, _to_kline
from trs import Domain, Kline, can, effort


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


def test_scripts_run_and_store():
    # Every run is an ask==goal run: two steps, everything else primes.
    runner = TrsRunner()
    steps = runner.run(SCRIPT)
    assert len(steps) == 2


def test_the_two_entries_are_goal_then_ask():
    runner = TrsRunner()
    steps = runner.run(SCRIPT)
    assert len(steps[0].kline.adj) == 3  # the goal canon: Subject, Verb, Object
    assert len(steps[1].kline.adj) == 5  # the MHALL ask: the full sentence
    # the primed klines are in the domain (they never rationalise as steps)
    assert any(can(k.adj) == k.sig for k in runner.domain if len(k.adj) == 1)


def test_a_script_without_an_ask_is_rejected():
    runner = TrsRunner()
    try:
        runner.run("(nothing to ask)")
    except ValueError as exc:
        assert "ask==goal" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_proposal_discharge_is_by_a_later_entry():
    runner = TrsRunner()
    runner.domain = Domain([Kline(1, frozenset([1]))])  # seed the substrate
    runner._discharge(0, Kline(1, frozenset([2])))  # step 0 presents, raises
    p = Pending.__new__(Pending)  # a proposal raised at step 0, target 4
    p.prop, p.raised_at, p.discharged_at = (
        type("P", (), {"kline": Kline(4, frozenset()), "kind": "acquire", "effort": Fraction(1)})(),
        0,
        None,
    )
    runner.pending = [p]
    runner._discharge(0, Kline(4, frozenset([4])))  # same step: never answers itself
    assert p.discharged_at is None
    runner._discharge(1, Kline(4, frozenset([4])))  # a later entry touching 4
    assert p.discharged_at == 1


def test_compiled_entry_maps_to_a_kline():
    runner = TrsRunner()
    steps = runner.run(SCRIPT)
    entry = steps[0]
    assert entry.kline.adj  # the compiled nodes carried over
    assert can(entry.kline.adj) != 0
    assert effort(entry.kline) >= 0
