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
    runner = TrsRunner()
    steps = runner.run(SCRIPT)
    assert len(steps) >= 5
    # Step 1 (the sentence) has no S1 candidate — nothing stores. The five
    # cross-reentry MHALL:[word] stores of the old FIFO runner are gone:
    # openers are per work entry now (amendment 18).
    assert steps[0].result.stores == []


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
    steps = runner.run("(Mary had a little lamb)")
    entry = steps[0]
    assert entry.kline.adj  # the compiled nodes carried over
    assert can(entry.kline.adj) != 0
    assert effort(entry.kline) >= 0
