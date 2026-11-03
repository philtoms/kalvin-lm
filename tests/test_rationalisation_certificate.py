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

from hypothesis import given, settings

from rationalisation import ReferenceCogitator, Domain, Kline, can
from rationalisation.verify import Certificate, Records, verify


from tests.test_rationalisation_properties import closed_domains

from dev.dialogue.harness import Runner
from ks.compiler import path_resolver

pytestmark = pytest.mark.skip(
    reason="T1–T4 parked (ruling 2026-11-01): premature. The instruments stay "
           "dormant until the signature×signature semantics settles."
)


SCRIPTS = sorted(Path("data/scripts").glob("*.ks"))


class RecordingRunner(Runner):
    """Captures the reference's records after each step — rationalise()
    rebinds its arrays per run, so each run's records survive by
    reference. The certificate instruments the reference, not the
    default Cogitator."""

    def __init__(self, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.cogitator = ReferenceCogitator(self.domain)
        self.records: list[Records] = []

    def _step(self, steps, i, value, goal=None):
        super()._step(steps, i, value, goal=goal)
        cog = self.cogitator
        self.records.append(
            Records(cog.frames, cog.frame_re, cog.stalls, cog.rows, cog.goal_row)
        )


def _runs() -> list[tuple[str, RecordingRunner, list]]:
    out = []
    for p in SCRIPTS:
        runner = RecordingRunner()
        steps = runner.run(p.read_text(), resolver=path_resolver(p.parent))
        out.append((p.name, runner, steps))
    return out


def _scenarios(runner, steps):
    """Each ask==goal pair with the records its ask run left, and the
    domain as that run found it (primed entries plus every earlier
    scenario's stores)."""
    primed = set(runner.domain) - {
        s for st in steps for s in st.result.stores
    }
    out = []
    for k in range(0, len(steps), 2):
        goal_step, ask_step = steps[k], steps[k + 1]
        d0 = primed | {s for st in steps[: k + 1] for s in st.result.stores}
        out.append((ask_step.result, runner.records[k + 1], d0,
                    ask_step.kline, goal_step.kline))
    return out


def _all_scenarios():
    for name, runner, steps in _runs():
        for r, records, d0, entry, goal in _scenarios(runner, steps):
            yield name, r, records, d0, entry, goal


# --- T3 soundness: grounded ⇒ certified ---


def test_corpus_groundings_carry_certificates():
    certified = 0
    for name, r, records, d0, entry, goal in _all_scenarios():
        if r.claim is None:
            continue
        cert = verify(r, records, d0, entry, goal)
        assert cert is not None, f"{name}: grounded without a certificate"
        assert cert.claim == r.claim
        certified += 1
    assert certified > 0  # the corpus does ground — the test is not vacuous


def test_ungrounded_runs_return_none():
    for name, r, records, d0, entry, goal in _all_scenarios():
        if r.claim is not None:
            continue
        assert verify(r, records, d0, entry, goal) is None, name


def test_claim_mismatch_rejected():
    for name, r, records, d0, entry, goal in _all_scenarios():
        if r.claim is None:
            continue
        other = Kline(goal.sig | (1 << 62), frozenset([goal.sig]))
        assert verify(r, records, d0, entry, other) is None, name


# --- non-vacuity: corrupted records fail ---


def test_stripped_frames_fail_verification():
    """A certificate that leans on frames must fail without them; one
    justified purely by witnesses, empty regions, and exchanges survives
    (and that is honest — those leaves stand on the domain alone)."""
    checked = 0
    for name, r, records, d0, entry, goal in _all_scenarios():
        if r.claim is None:
            continue
        cert = verify(r, records, d0, entry, goal)
        assert cert is not None, name
        if not (set(cert.leaves) - {"empty", "witness", "exchange"}):
            continue  # no frame-dependent leaf to strip
        frames, owners = list(records.frames), list(records.frame_re)
        records.frames, records.frame_re = [], []
        try:
            assert verify(r, records, d0, entry, goal) is None, name
        finally:
            records.frames, records.frame_re = frames, owners
        checked += 1
    assert checked > 0  # the corpus does lean on frames — not vacuous


def test_broken_exchange_evidence_fails():
    """A joint frame without its stall pair is not evidence."""
    for name, r, records, d0, entry, goal in _all_scenarios():
        if r.claim is None:
            continue
        stalls = records.stalls
        records.stalls = {}
        try:
            cert = verify(r, records, d0, entry, goal)
            # only runs that actually lean on an exchange must fail
            if cert is not None:
                assert "exchange" not in cert.leaves
        finally:
            records.stalls = stalls


# --- T3 soundness over random scenarios ---


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
    cog = ReferenceCogitator(d)
    r = cog.rationalise(entry, goal=goal)
    if r.claim is None:
        return
    assert r.claim == Kline(entry.sig, frozenset([goal.sig]))
    records = Records(cog.frames, cog.frame_re, cog.stalls, cog.rows)
    assert verify(r, records, d0, entry, goal) is not None


# --- the regression: exhaustion is not evidence ---


def test_exhaustion_does_not_ground():
    """The shape the property test found: identities plus 9:[1] primed,
    entry 9:[32,1] against goal 1:[1]. The 40 region has no backing — the
    only pairing runs the u-chain into an identity that stalls — so the
    claim must not ground. Before the fix, the exhausted-candidates known
    resolved the region on the in-path bar alone."""
    d = Domain(
        [Kline(1 << i, frozenset([1 << i])) for i in range(6)]
        + [Kline(9, frozenset([1]))]
    )
    d0 = set(d)
    cog = ReferenceCogitator(d)
    entry, goal = Kline(9, frozenset([32, 1])), Kline(1, frozenset([1]))
    r = cog.rationalise(entry, goal=goal)
    assert r.claim is None
    records = Records(cog.frames, cog.frame_re, cog.stalls, cog.rows)
    assert verify(r, records, d0, entry, goal) is None


# --- T2 determinism: outcome survives hash-seed variation ---


_DIGEST = """
import hashlib, sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from pathlib import Path
from dev.dialogue.harness import Runner
from ks.compiler import path_resolver
digest = hashlib.sha256()
for p in sorted(Path("data/scripts").glob("*.ks")):
    runner = Runner()
    steps = runner.run(p.read_text(), resolver=path_resolver(p.parent))
    for st in steps:
        digest.update(repr(st.result.claim).encode())
        digest.update(repr(sorted(map(repr, st.result.stores))).encode())
        digest.update(repr(sorted(e.reason for e in st.result.ends)).encode())
        digest.update(str(len(st.result.events)).encode())
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
        for records in runner.records:
            assert len(records.rows) <= ReferenceCogitator.MAX_ROWS + len(list(runner.domain)), name
            assert all(row.depth <= ReferenceCogitator.MAX_DEPTH + 1 for row in records.rows), name
