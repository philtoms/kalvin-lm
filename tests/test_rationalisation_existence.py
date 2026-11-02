"""Certificate existence — the refusal half of T4 (docs/rationalisation.md §4).

exists_certificate decides, without running the search, whether a
covering certificate is available at all. With T3 (grounding ⇒
certificate) this gives refusal prediction: ``False`` means no schedule,
no order, no bounds can ground the scenario.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hypothesis import given, settings

from rationalisation import ReferenceCogitator, Domain, Kline, can
from rationalisation.existence import exists_certificate

from tests.test_rationalisation_properties import closed_domains

from dev.dialogue.harness import Runner
from ks.compiler import path_resolver

pytestmark = pytest.mark.skip(
    reason="T1–T4 parked (ruling 2026-11-01): premature. The instruments stay "
           "dormant until the signature×signature semantics settles."
)


SCRIPTS = sorted(Path("data/scripts").glob("*.ks"))


def _scenarios():
    for p in SCRIPTS:
        runner = Runner()
        steps = runner.run(p.read_text(), resolver=path_resolver(p.parent))
        primed = set(runner.domain) - {
            s for st in steps for s in st.result.stores
        }
        for k in range(0, len(steps), 2):
            goal, ask = steps[k], steps[k + 1]
            d0 = primed | {s for st in steps[: k + 1] for s in st.result.stores}
            yield p.name, ask.result, d0, ask.kline, goal.kline


def _scenario(d):
    held = [k for k in d if k.sig == can(k.adj)]
    if not held:
        return None
    goal = held[len(held) // 2]
    sigs = sorted({k.sig for k in d})
    entry = Kline(sigs[len(sigs) // 2], frozenset([n for n in {k.sig for k in d}][:2]))
    return d, goal, entry


# --- refusal soundness: grounding implies an available certificate ---


def test_corpus_groundings_have_certificates_available():
    for name, r, d0, entry, goal in _scenarios():
        if r.claim is None:
            continue
        assert exists_certificate(d0, entry, goal) is True, name


@settings(deadline=None, max_examples=60)
@given(closed_domains())
def test_grounding_implies_certificate_exists(d):
    sc = _scenario(d)
    if sc is None:
        return
    d1, goal, entry = sc
    d0 = set(d1)
    cog = ReferenceCogitator(d1)
    r = cog.rationalise(entry, goal=goal)
    if r.claim is None:
        return
    assert exists_certificate(d0, entry, goal) is True


# --- refusal prediction: no certificate, no grounding — ever ---


def test_the_exhaustion_regression_is_refused():
    """The shape the soundness defect was found on: no certificate
    exists, so no schedule could ground it — and the run does not."""
    d = Domain(
        [Kline(1 << i, frozenset([1 << i])) for i in range(6)]
        + [Kline(9, frozenset([1]))]
    )
    entry, goal = Kline(9, frozenset([32, 1])), Kline(1, frozenset([1]))
    assert exists_certificate(d, entry, goal) is False
    assert ReferenceCogitator(d).rationalise(entry, goal=goal).claim is None


def test_disjoint_content_is_refused():
    """An ask whose content nothing holds or covers: refused before
    running."""
    d = Domain([Kline(1, frozenset([1])), Kline(2, frozenset([2]))])
    entry, goal = Kline(1 | 16, frozenset([16, 1])), Kline(1, frozenset([1]))
    assert exists_certificate(d, entry, goal) is False
    assert ReferenceCogitator(d).rationalise(entry, goal=goal).claim is None


def test_covered_ask_is_available():
    d = Domain([Kline(1 << i, frozenset([1 << i])) for i in range(6)])
    assert (
        exists_certificate(
            d, Kline(3, frozenset([1, 2])), Kline(3, frozenset([1, 2]))
        )
        is True
    )


# --- the completeness probe: available ⇒ grounded? ---
# MEASURED (400-scenario sweep): ~20% of random scenarios carry a
# certificate the deterministic run misses — the in_path bar dominates
# the attribution (the run halts 'exhausted' on a pairing the checker
# rightly explores). The probe stays as the regression instrument for
# every future search change: each new xpass narrows the measured gap.
@pytest.mark.xfail(reason="the path bar prunes certificate-bearing pairings — measured, not yet theorem", strict=False)
@settings(deadline=None, max_examples=60)
@given(closed_domains())
def test_available_certificate_is_found_by_the_run(d):
    sc = _scenario(d)
    if sc is None:
        return
    d1, goal, entry = sc
    cog = ReferenceCogitator(d1)
    r = cog.rationalise(entry, goal=goal)
    if exists_certificate(set(d1), entry, goal) is True:
        assert r.claim is not None, "certificate available but the run missed it"
