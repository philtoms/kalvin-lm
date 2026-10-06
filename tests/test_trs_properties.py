"""TRS model — property-based tests over the invariants (docs/trs-significance.md)."""

from __future__ import annotations

from hypothesis import assume, given, settings, strategies as st

from trs import (
    Band,
    Domain,
    Kline,
    band,
    can,
    contained,
    decompose,
    effort,
    entropy,
    mu,
    proposals,
    rationalise,
    r2_products,
    r3_substitute,
    significance,
)

ATOMS = [1 << i for i in range(6)]
ALL_SIGS = list(range(1, 64))


def bits(v: int) -> set[int]:
    """The atoms carried by a value."""
    return {b for b in ATOMS if v & b}

# --- generators ---


def sigs() -> st.SearchStrategy[int]:
    return st.sampled_from(ALL_SIGS)


def adjacencies(pool: list[int] | None = None, min_size: int = 0, max_size: int = 5) -> st.SearchStrategy[frozenset[int]]:
    return st.sets(st.sampled_from(pool or ALL_SIGS), min_size=min_size, max_size=max_size).map(frozenset)


def klines(pool: list[int] | None = None) -> st.SearchStrategy[Kline]:
    return st.builds(Kline, sigs(), adjacencies(pool))


@st.composite
def closed_domains(draw) -> Domain:
    """Random I-1-closed domains: every adjacency node is a held signature.
    Several canonical witnesses may share a head (I-2 withdrawn, D15)."""
    d = Domain([Kline(a, frozenset([a])) for a in ATOMS])
    held = list(ATOMS)
    for _ in range(draw(st.integers(0, 10))):
        nodes = draw(adjacencies(held, max_size=4))
        k = Kline(draw(sigs()), nodes)
        d.add(k)
        if k.sig not in held:
            held.append(k.sig)
    return d


@st.composite
def scenarios(draw) -> tuple[Domain, Kline]:
    """An I-1-closed domain and an admissible entry (Definition 8 presentation)."""
    d = draw(closed_domains())
    held = sorted({k.sig for k in d})
    entry = Kline(draw(st.sampled_from(held)), draw(adjacencies(held, max_size=3)))
    return d, entry


@st.composite
def s2_scenarios(draw) -> tuple[Domain, Kline]:
    """An I-1-closed domain and an admissible S2 kline: F ≠ ⊥, short of canonical."""
    d = draw(closed_domains())
    held = sorted({k.sig for k in d})
    nodes = draw(adjacencies(held, max_size=4))
    a_hat = can(nodes)
    pool = [s for s in ALL_SIGS if s & a_hat and s != a_hat]
    assume(pool)
    return d, Kline(draw(st.sampled_from(pool)), nodes)


@st.composite
def s3_scenarios(draw) -> tuple[Domain, Kline]:
    """An I-1-closed domain and an admissible S3 kline: no correspondence."""
    d = draw(closed_domains())
    held = sorted({k.sig for k in d})
    nodes = draw(adjacencies(held, min_size=1, max_size=4))
    a_hat = can(nodes)
    pool = [s for s in ALL_SIGS if s & a_hat == 0]
    assume(pool)
    return d, Kline(draw(st.sampled_from(pool)), nodes)


# --- §4 Definition 11: the band ---


def oracle_band(k: Kline) -> Band:
    """Independent formulation: no ordered cases."""
    if not k.adj:
        return Band.S4
    t = decompose(k)
    if t.U == 0 and t.O == 0:
        return Band.S1
    if t.F == 0:
        return Band.S3
    return Band.S2


@settings(deadline=None)
@given(klines())
def test_band_total_and_matches_oracle(k):
    assert band(k) in Band
    assert band(k) is oracle_band(k)


# --- §4 Definition 12: decomposition ---


@settings(deadline=None)
@given(klines())
def test_decomposition_laws(k):
    t = decompose(k)
    a_hat = can(k.adj)
    assert k.sig == t.F | t.U
    assert a_hat == t.F | t.O
    assert t.F & t.U == 0 and t.F & t.O == 0 and t.U & t.O == 0
    # U is signature-side; O is adjacency-side
    assert contained(t.U, k.sig) and t.U & a_hat == 0
    assert contained(t.O, a_hat) and t.O & k.sig == 0
    assert (band(k) is Band.S2) == (t.F != 0 and (t.U != 0 or t.O != 0))


# --- §2 Definition 6: witness (first-stored; I-2 withdrawn, D15) ---


@settings(deadline=None)
@given(closed_domains())
def test_witness_law(d):
    canonical_sigs = {k.sig for k in d if d.is_canonical(k)}
    for s in canonical_sigs:
        assert can(d.witness(s)) == s  # can(witness(s)) = s
    for s in range(1, 64):
        if s not in canonical_sigs:
            assert d.witness(s) == frozenset()  # total: ∅ when not held


@settings(deadline=None)
@given(closed_domains())
def test_witness_returns_first_stored(d):
    """Several canonical witnesses may share a head; the lookup keeps the
    first-stored and the law holds."""
    seen: dict[int, frozenset[int]] = {}
    for k in d:
        if d.is_canonical(k):
            first = seen.setdefault(k.sig, k.adj)
            if k.adj != first:
                assert d.witness(k.sig) == first  # coexistence: first-stored wins
                assert can(d.witness(k.sig)) == k.sig  # the law holds regardless


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


@settings(deadline=None)
@given(s2_scenarios())
def test_r2_never_reduces_entropy(sc):
    """Weighted by head measure, R2 never reduces entropy; exact iff the fit head is held."""
    d, k = sc
    t = decompose(k)
    total = mu(k.sig | can(k.adj))
    carried = sum(mu(p.sig) * entropy(p) for p in r2_products(d, k))
    assert carried >= total * entropy(k)
    fit_held = t.F == 0 or d.witness(t.F) != frozenset()
    assert (carried == total * entropy(k)) == fit_held


# --- §5 R2/R3: product well-formedness ---


@settings(deadline=None)
@given(s2_scenarios())
def test_r2_products_well_formed(sc):
    d, k = sc
    assert band(k) is Band.S2
    t = decompose(k)
    products = r2_products(d, k)
    assert len(products) <= 3
    for p in products:
        assert p.sig in (t.F, t.U, t.O) and p.sig != 0  # non-empty heads from the decomposition
        assert p.adj == d.witness(t.F if p.sig == t.F else (t.O if p.sig == t.U else t.U))
        assert all(n in {kk.sig for kk in d} for n in p.adj) or not p.adj  # witnessed adjacencies are held


@settings(deadline=None)
@given(s3_scenarios())
def test_r3_product_well_formed(sc):
    d, k = sc
    assert band(k) is Band.S3
    held_sigs = {kk.sig for kk in d}
    assert all(n in held_sigs for n in k.adj)  # admissible: references held
    n_prime = r3_substitute(d, k)
    if n_prime:
        assert can(n_prime) != 0  # the product's head is legal (Definition 2)
    assert all(n in held_sigs for n in n_prime)  # I-1 survives substitution


# --- §5: rationalisation preserves the invariants ---


@settings(deadline=None, max_examples=150)
@given(scenarios())
def test_i1_maintained_by_construction(sc):
    d, entry = sc
    assert d.i1_closed()
    rationalise(d, entry)
    assert d.i1_closed()


@settings(deadline=None, max_examples=150)
@given(scenarios())
def test_no_fabrication(sc):
    d, entry = sc
    before = set(d)
    atoms_before = {b for k in before for v in (k.sig, *k.adj) for b in bits(v)}
    atoms_before |= bits(entry.sig) | {b for n in entry.adj for b in bits(n)}
    r = rationalise(d, entry)
    added = set(d) - before
    assert added == set(r.stores)  # the domain grows only through R1 stores
    for k in added:
        assert k.sig == can(entry.adj)  # every store is an opening candidate of this entry
        assert any(k.adj == kk.adj for kk in before)  # valued against a held adjacency
    atoms_after = {b for k in d for v in (k.sig, *k.adj) for b in bits(v)}
    assert atoms_after <= atoms_before  # no new atoms: the system composes what it holds


@settings(deadline=None, max_examples=150)
@given(scenarios(), st.integers(1, 24))
def test_rationalisation_bounded_and_honest(sc, bound):
    d, entry = sc
    r = rationalise(d, entry, max_rounds=bound)
    assert r.rounds <= bound
    assert all(
        e.reason.startswith(("R4", "no candidates", "bound")) for e in r.ends
    )
    # every halt records its kline and its halt measure
    assert all(e.kline is not None and e.effort is not None for e in r.ends)
    ps = proposals(r)
    assert len(ps) == len(r.ends)  # every halt generates exactly one proposal
    assert all(p.kind in ("acquire", "connect", "continue") for p in ps)
    assert all(p.effort == effort(p.kline) for p in ps)
    assert all(0 <= p.effort <= 1 for p in ps)
    assert ps == sorted(ps, key=lambda p: p.effort)  # graded significance: least entropy first


# --- the R1 rewrite rule: opener stored, endpoint constructable ---


def test_r1_rewrite_stores_opener_not_endpoint():
    """The trace opening ⟨2, {1}⟩ grounds out at S1 via the R3 expansion
    {1} → {1,2}. R1 stores the opening candidate; the S1 candidate ⟨3, {1,2}⟩
    is constructable from it and is not stored."""
    A, B = 1, 2
    d = Domain(
        [
            Kline(A, frozenset([A])),
            Kline(B, frozenset([B])),
            Kline(3, frozenset([3])),  # held canonical under head 3
            Kline(A, frozenset([A, B])),  # non-canonical: the substitutable source
        ]
    )
    assert d.i1_closed()
    r = rationalise(d, Kline(A, frozenset([B])))
    assert Kline(2, frozenset([A])) in r.stores  # the opening candidate, stored by proof
    assert Kline(2, frozenset([A])) in d
    assert Kline(3, frozenset([A, B])) not in d  # constructable — not minted
    assert can(d.witness(3)) == 3  # the held canonical keeps the key
