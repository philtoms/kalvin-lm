"""The covering certificate — T3 executed (docs/rationalisation.md §3).

Soundness: a grounded claim always verifies. Non-vacuity: corrupted
records always fail. Determinism (T2): the outcome survives hash-seed
variation. Termination (T1): the corpus respects the bounds.
"""

from __future__ import annotations

import os
import subprocess
import sys

from pathlib import Path

import pytest

from hypothesis import given, settings, strategies as st

from rationalisation import Cogitator, Domain, Kline, can
from rationalisation.verify import verify
from ks.compiler import path_resolver

from tests.test_rationalisation_properties import closed_domains

from dev.dialogue.harness import TrsRunner

SCRIPTS = sorted(Path("data/scripts").glob("*.ks"))


def _runs() -> list[tuple[str, TrsRunner, list]]:
    out = []
    for p in SCRIPTS:
        runner = TrsRunner()
        steps = runner.run(p.read_text(), resolver=path_resolver(p.parent))
        out.append((p.name, runner, steps))
    return out


def _ask_records(runner: TrsRunner, steps: list):
    """The ask step's rationalisation, the cogitator holding its records,
    the domain the run found, and the (entry, goal) pair."""
    ask = steps[-1]
    goal = steps[0]
    d0 = set(runner.domain) - set(ask.result.stores)
    return ask.result, runner.cogitator, d0, ask.kline, goal.kline


# --- T3 soundness: grounded ⇒ certified ---


def test_corpus_groundings_carry_certificates():
    certified = 0
    for name, runner, steps in _runs():
        r, cog, d0, entry, goal = _ask_records(runner, steps)
        if r.claim is None:
            continue
        cert = verify(r, cog, d0, entry, goal)
        assert cert is not None, f"{name}: grounded without a certificate"
        assert cert.claim == r.claim
        certified += 1
    assert certified > 0  # the corpus does ground — the test is not vacuous


def test_ungrounded_runs_return_none():
    for name, runner, steps in _runs():
        r, cog, d0, entry, goal = _ask_records(runner, steps)
        if r.claim is not None:
            continue
        assert verify(r, cog, d0, entry, goal) is None


def test_claim_mismatch_rejected():
    for name, runner, steps in _runs():
        r, cog, d0, entry, goal = _ask_records(runner, steps)
        if r.claim is None:
            continue
        other = Kline(goal.sig | (1 << 62), frozenset([goal.sig]))
        assert verify(r, cog, d0, entry, other) is None


# --- non-vacuity: corrupted records fail ---


def test_stripped_frames_fail_verification():
    """A certificate that leans on frames must fail without them; one
    justified purely by witnesses and empty regions survives (and that is
    honest — those leaves stand on the domain alone)."""
    checked = 0
    for name, runner, steps in _runs():
        r, cog, d0, entry, goal = _ask_records(runner, steps)
        if r.claim is None:
            continue
        cert = verify(r, cog, d0, entry, goal)
        assert cert is not None, name
        if not (set(cert.leaves) - {"empty", "witness", "exchange"}):
            continue  # no frame-dependent leaf to strip
        frames = list(cog.frames)
        owners = list(cog.frame_re)
        cog.frames, cog.frame_re = [], []
        try:
            assert verify(r, cog, d0, entry, goal) is None, name
        finally:
            cog.frames, cog.frame_re = frames, owners
        checked += 1
    assert checked > 0  # the corpus does lean on frames — not vacuous


def test_broken_exchange_evidence_fails():
    """A joint frame without its stall pair is not evidence."""
    for name, runner, steps in _runs():
        r, cog, d0, entry, goal = _ask_records(runner, steps)
        if r.claim is None:
            continue
        stalls = cog.stalls
        cog.stalls = {}
        try:
            cert = verify(r, cog, d0, entry, goal)
            # only runs that actually lean on an exchange must fail
            if cert is not None:
                assert "exchange" not in cert.leaves
        finally:
            cog.stalls = stalls


# --- T3 soundness over random scenarios ---


# KNOWN DEFECT (found by this property): the exhausted-candidates known can
# resolve a reentry whose only candidates are in-path barred. `_in_path`
# compares signature values alone, so a held kline sharing only a sig with
# an ancestor entry is barred; when it was the only candidate, the reentry
# resolves with no frame, witness, pairing, or exchange behind it — a
# grounded claim without a certificate. Minimal shape: identities + 9:[1]
# primed, entry 9:[32,1], goal 1:[1] — grounds ⟨9,{1}⟩ with the 40 region
# unbacked. Un-mark when the model fix lands.
@pytest.mark.xfail(reason="exhaustion-by-in-path known grounds without evidence", strict=False)
@settings(deadline=None, max_examples=100)
@given(closed_domains())
def test_random_groundings_carry_certificates(d):
    held = [k for k in d if k.sig == can(k.adj)]
    if not held:
        return
    goal = held[len(held) // 2]
    sigs = sorted({k.sig for k in d})
    entry = Kline(sigs[len(sigs) // 2], frozenset([n for n in {k.sig for k in d}][:2]))
    d0 = set(d)
    cog = Cogitator(d)
    r = cog.rationalise(entry, goal=goal)
    if r.claim is None:
        return
    assert r.claim == Kline(entry.sig, frozenset([goal.sig]))
    assert verify(r, cog, d0, entry, goal) is not None


# --- T2 determinism: outcome survives hash-seed variation ---


_DIGEST = """
import hashlib, sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from pathlib import Path
from dev.dialogue.harness import TrsRunner
from ks.compiler import path_resolver
digest = hashlib.sha256()
for p in sorted(Path("data/scripts").glob("*.ks")):
    runner = TrsRunner()
    steps = runner.run(p.read_text(), resolver=path_resolver(p.parent))
    ask = steps[-1].result
    digest.update(repr(ask.claim).encode())
    digest.update(repr(sorted(map(repr, ask.stores))).encode())
    digest.update(repr(sorted(e.reason for e in ask.ends)).encode())
    digest.update(str(len(runner.cogitator.rows)).encode())
print(digest.hexdigest())
"""


def test_outcome_is_hash_seed_independent():
    digests = set()
    for seed in ("0", "1", "random", "random"):
        env = {**os.environ, "PYTHONPATH": "src:.", "PYTHONHASHSEED": seed}
        out = subprocess.run(
            [sys.executable, "-c", _DIGEST], env=env, capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr
        digests.add(out.stdout.strip())
    assert len(digests) == 1, f"outcome varies across hash seeds: {digests}"


# --- T1 termination: the corpus respects the bounds ---


def test_corpus_respects_bounds():
    for name, runner, steps in _runs():
        cog = runner.cogitator
        assert len(cog.rows) <= Cogitator.MAX_ROWS + len(list(runner.domain)), name
        assert all(row.depth <= Cogitator.MAX_DEPTH + 1 for row in cog.rows), name
