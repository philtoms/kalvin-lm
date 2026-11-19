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

_SCRIPTS = Path(__file__).resolve().parents[1] / "dev/dialogue/scripts"
MHALL = (_SCRIPTS / "mhall.ks").read_text()
WDMH = (_SCRIPTS / "wdmh.ks").read_text()


def test_scripts_run_and_ask():
    # Every run is a sequence of asks: one step per ask==goal scenario.
    runner = Runner()
    steps = runner.run(SCRIPT)
    assert len(steps) == 1


def test_the_one_entry_is_the_ask():
    runner = Runner()
    steps = runner.run(SCRIPT)
    assert len(steps[0].kline.nodes) == 5  # the MHALL ask: the full sentence
    # the other compiled klines are held as scaffolding, not primed — the
    # domain grows only from what a stall serves, and only from what is held
    assert all(k in set(runner.scaffolding) for k in runner.domain)


def test_a_served_stall_supplies_scaffolding_on_demand(tmp_path):
    # wdmh over mhall: a stall is served — the held klines that qualify as
    # candidates for the stalled signature join the domain mid-run — and
    # later steps read them as candidates; the unserved stalls stand
    (tmp_path / "mhall.ks").write_text(MHALL)
    (tmp_path / "wdmh.ks").write_text(WDMH)
    runner = Runner()
    steps = runner.run(WDMH, resolver=path_resolver(tmp_path))
    assert len(steps) == 2
    held = set(runner.scaffolding)
    assert runner.domain  # a stall was served — scaffolding landed
    assert all(k in held for k in runner.domain)  # only held klines ever join
    assert runner.stalls  # the unserved stalls stand


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
    # component stands whole), underfit WDH, overfit HALL. The claim no
    # longer grounds (scaffolding is held, not primed — WDH starves), but
    # the entry event carries the same decomposed labels.
    script = Path("dev/dialogue/scripts/wdmh.ks").read_text()
    steps = Runner().run(script, resolver=path_resolver(Path("dev/dialogue/scripts")))
    entered = [
        ev for s in steps for ev in s.result.events
        if ev.kind == "entry" and ev.entry.label == "WDMH"
    ]
    assert entered, "no entry event"
    ev = entered[0]
    assert ev.F.label == "Mary"
    assert ev.U.label == "WDH"
    assert ev.O.label == "HALL"
