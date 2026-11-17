"""The dialogue path — the scripted-run hookup (dev/dialogue/harness.py)."""

from __future__ import annotations
from pathlib import Path

from fractions import Fraction

from dev.dialogue.harness import Runner, _to_kline, _tlabel
from kalvin.kline import KNode
from ks.compiler import path_resolver
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

def test_compiled_entry_maps_to_a_kline():
    runner = Runner()
    steps = runner.run(SCRIPT)
    entry = steps[0]
    assert entry.kline.nodes  # the compiled nodes carried over
    assert can(entry.kline.nodes) != 0
    assert effort(entry.kline) >= 0


def test_tlabel_prefers_the_value_s_own_label():
    # A value that carries a label is named by it; the script registry
    # fills gaps (plain ints, unlabelled values).
    assert _tlabel(KNode(1 << 32, "M"), {1 << 32: "Mary"}, {}) == "M"
    assert _tlabel(1 << 32, {1 << 32: "Mary"}, {}) == "Mary"


def test_claimed_event_labels_decompose_mts():
    # wdmh.ks: A = WDMH against C = MHALL — fit Mary (had and have share
    # the initial but not the bit, so the fit is Mary alone; a single
    # component stands whole), underfit WDH, overfit HALL.
    script = Path("dev/dialogue/scripts/wdmh.ks").read_text()
    steps = Runner().run(script, resolver=path_resolver(Path("dev/dialogue/scripts")))
    claimed = [ev for s in steps for ev in s.result.events if ev.kind == "claimed"]
    assert claimed, "no claimed event"
    ev = claimed[-1]
    assert ev.F.label == "Mary"
    assert ev.U.label == "WDH"
    assert ev.O.label == "HALL"
