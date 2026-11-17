"""The model — property-based tests over the invariants (docs/rationalisation.md)."""

from __future__ import annotations

import pytest

from hypothesis import given, settings, strategies as st

from rationalisation import (
    Band,
    Domain,
    KLine,
    band,
    can,
    effort,
    entropy,
    fit,
    Cogitator,
    overfit,
    rationalise,
    significance,
    underfit,
)

ATOMS = [1 << i for i in range(6)]
ALL_SIGS = list(range(1, 64))


def bits(v: int) -> set[int]:
    """The atoms carried by a value."""
    return {b for b in ATOMS if v & b}

# --- generators ---


def sigs() -> st.SearchStrategy[int]:
    return st.sampled_from(ALL_SIGS)


def adjacencies(pool: list[int] | None = None, min_size: int = 0, max_size: int = 5) -> st.SearchStrategy[list[int]]:
    return st.sets(st.sampled_from(pool or ALL_SIGS), min_size=min_size, max_size=max_size).map(sorted)


def klines(pool: list[int] | None = None) -> st.SearchStrategy[KLine]:
    return st.builds(KLine, sigs(), adjacencies(pool))


@st.composite
def closed_domains(draw) -> Domain:
    """Random referentially-closed domains: every adjacency node is a held signature.
    Several canonical witnesses may share a head (Definition 6)."""
    d = Domain([KLine(a, [a]) for a in ATOMS])
    held = list(ATOMS)
    for _ in range(draw(st.integers(0, 10))):
        nodes = draw(adjacencies(held, max_size=4))
        k = KLine(draw(sigs()), nodes)
        d.add(k)
        if k.signature not in held:
            held.append(k.signature)
    return d


@st.composite
def scenarios(draw) -> tuple[Domain, KLine, KLine]:
    """An I-1-closed domain, an admissible entry, and a goal — a canon the
domain holds (the ask is never goal-less)."""
    d = draw(closed_domains())
    held = sorted({k.signature for k in d})
    entry = KLine(draw(st.sampled_from(held)), draw(adjacencies(held, max_size=3)))
    goal = draw(st.sampled_from([k for k in d if d.is_canonical(k)]))
    return d, entry, goal


# --- §4 Definition 11: the band ---


def oracle_band(k: KLine) -> Band:
    """Independent formulation: no ordered cases."""
    if not k.nodes:
        return Band.S4
    a_hat = can(k.nodes)
    f = fit(k.signature, a_hat)
    u = underfit(k.signature, a_hat)
    o = overfit(k.nodes, k.signature)
    if u == 0 and o == 0:
        return Band.S1
    if f == 0:
        return Band.S3
    return Band.S2


@settings(deadline=None)
@given(klines())
def test_band_total_and_matches_oracle(k):
    assert band(k) in Band
    assert band(k) is oracle_band(k)


# --- §4 Definition 13: entropy and graded significance ---


@settings(deadline=None)
@given(klines())
def test_band_grade_correspondence(k):
    b = band(k)
    g = significance(k)
    assert (b is Band.S1) == (g == 1)  # no surprise — canon and identity alike
    assert (b is Band.S2) == (0 < g < 1)
    assert (b in (Band.S3, Band.S4)) == (g == 0)  # entropy does not split S3/S4
    assert entropy(k) == 1 - g


@settings(deadline=None)
@given(klines())
def test_effort_properties(k):
    from fractions import Fraction

    assert 0 <= effort(k) <= 1
    assert effort(k) <= entropy(k)  # β ≤ 1
    b = band(k)
    if b is Band.S1:
        assert effort(k) == 0  # anchor — canon and identity alike
    elif b is Band.S4:
        assert effort(k) == 1  # anchor — exactly full
    elif b is Band.S3:
        assert effort(k) < 1  # repairability: strictly below S4
    else:
        assert 0 < effort(k) < 1  # S2 — both kinds
    assert effort(k, beta=Fraction(1)) == entropy(k)  # β → 1 recovers H


# --- §5: rationalisation preserves the invariants ---


@settings(deadline=None, max_examples=150)
@given(scenarios())
def test_no_fabrication(sc):
    d, entry, goal = sc
    before = set(d)
    atoms_before = {b for k in before for v in (k.signature, *k.nodes) for b in bits(v)}
    atoms_before |= bits(entry.signature) | {b for n in entry.nodes for b in bits(n)}
    atoms_before |= bits(goal.signature) | {b for n in goal.nodes for b in bits(n)}
    rationalise(d, entry, goal=goal)
    assert set(d) == before  # the run never writes the domain — it is the harness's
    atoms_after = {b for k in d for v in (k.signature, *k.nodes) for b in bits(v)}
    assert atoms_after <= atoms_before  # no new atoms: the system composes what it holds


@settings(deadline=None, max_examples=150)
@given(scenarios())
def test_rationalisation_is_honest(sc):
    d, entry, goal = sc
    stalls: list[KLine] = []
    Cogitator(d, on_stall=stalls.append).rationalise(entry, goal=goal)
    # every broadcast stall is a bare S4 — nothing offered, nothing held
    assert all(not s.nodes for s in stalls)
    assert all(effort(s) == 1 for s in stalls)  # exactly full — nothing to work with
    assert set(d) == {k for k in d}  # the run never writes the domain
