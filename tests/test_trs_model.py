"""TRS model — tests against docs/trs-significance.md."""

from __future__ import annotations

import pytest
from fractions import Fraction

from trs import (
    BETA,
    Band,
    Domain,
    Kline,
    band,
    can,
    decompose,
    effort,
    entropy,
    proposals,
    rationalise,
    significance,
    r2_products,
    r3_substitute,
)

A, B, C, D = 1, 2, 4, 8
AB, CD, ABC, ABD, ABCD = 3, 12, 7, 11, 15


def k(sig: int, *adj: int) -> Kline:
    return Kline(sig, frozenset(adj))


def base_domain(*extra: Kline) -> Domain:
    """Single atoms held canonically — a simple substrate."""
    return Domain([k(A, A), k(B, B), k(C, C), k(D, D), *extra])


# --- §2 Definitions 4–6: can, canonical, witness ---


def test_can_folds_composition():
    assert can(frozenset()) == 0
    assert can(frozenset({AB})) == AB
    assert can(frozenset({AB, C})) == ABC


def test_witness_held_and_unheld():
    d = base_domain(k(AB, A, B))
    assert d.witness(AB) == frozenset({A, B})
    assert can(d.witness(AB)) == AB  # the law: can(witness(s)) = s
    assert d.witness(ABCD) == frozenset()  # not held: no witness


def test_witnessed_adjacencies_are_held_signatures():
    """Stores copy held adjacencies, so witnessed adjacencies reference held
    signatures — a corollary of the rules, no longer an invariant."""
    d = base_domain(k(AB, A, B), k(ABCD, AB, C, D))
    sigs = {kk.sig for kk in d}
    for s in (AB, ABCD):
        assert all(n in sigs for n in d.witness(s))


def test_kline_rejects_empty_signature():
    with pytest.raises(ValueError):
        k(0, A)


# --- §4 Definition 11: the band ---


def test_bands():
    assert band(k(AB, A, B)) is Band.S1  # canonical
    assert band(k(A, A)) is Band.S1  # single-node canonical
    assert band(k(ABC, AB, C)) is Band.S1  # canonical — composition equals head
    assert band(k(ABCD, AB, C)) is Band.S2  # partial overlap, not canonical
    assert band(k(AB, CD)) is Band.S3  # disjoint
    assert band(k(AB)) is Band.S4  # no adjacency


def test_s2_requires_overlap_short_of_canonical():
    # σ ⪯ can(N) with excess: pure overfit — still S2
    assert band(k(AB, A, B, C)) is Band.S2
    # can(N) ⪯ σ with excess: pure underfit — still S2
    assert band(k(ABCD, AB)) is Band.S2


# --- §4 Definition 12: decomposition ---


def test_decomposition_partition_and_flow_sides():
    t = decompose(k(ABC, ABD))  # â = ABD
    assert (t.F, t.U, t.O) == (AB, C, D)
    assert t.F | t.U == ABC  # σ = F ∨ U
    assert t.F | t.O == ABD  # â = F ∨ O
    assert t.F & t.U == t.F & t.O == t.U & t.O == 0
    # U is signature-side (claimed, not supplied); O is adjacency-side
    t2 = decompose(k(ABCD, AB))
    assert t2.U == CD and t2.O == 0
    t3 = decompose(k(AB, A, B, C))
    assert t3.U == 0 and t3.O == C


# --- §5 R2: three klines, cross-wise pairing ---


def test_r2_cross_wise_pairing():
    d = base_domain(k(AB, A, B), k(C, C), k(D, D))
    products = r2_products(d, k(ABC, ABD))
    # F=AB carries its own witness; U=C carries witness(O)=witness(D); O=D carries witness(U)=witness(C)
    assert [(p.sig, p.adj) for p in products] == [
        (AB, frozenset({A, B})),
        (C, frozenset({D})),
        (D, frozenset({C})),
    ]


def test_r2_unheld_residual_carries_no_adjacency():
    d = Domain()  # nothing held: no witness for F, U, or O
    products = r2_products(d, k(ABC, ABD))
    assert [p.sig for p in products] == [AB, C, D]
    assert all(p.adj == frozenset() for p in products)  # R4 arrivals


def test_r2_pure_overfit_skips_empty_head():
    d = base_domain()
    products = r2_products(d, k(AB, A, B, C))  # U = ⊥
    assert [p.sig for p in products] == [AB, C]  # F-product and O-product only


# --- §5 R3: substitute, canonicalise, re-enter ---


def test_r3_substitution_expands_reference():
    d = base_domain(k(AB, A, B))
    n_prime = r3_substitute(d, k(CD, AB))
    assert n_prime == frozenset({A, B})


def test_r3_single_pass():
    # b's kline would substitute b → {b, c}; single pass forbids the second round
    d = base_domain(k(AB, A, B), k(B, B, C))
    n_prime = r3_substitute(d, k(CD, AB))
    assert n_prime == frozenset({A, B})


def test_r3_emptied_expansion():
    d = base_domain(k(AB))  # terminal kline headed AB
    n_prime = r3_substitute(d, k(CD, AB))
    assert n_prime == frozenset()  # routes to R4


# --- §4 Definition 13: entropy and graded significance ---


def test_entropy_endpoints():
    assert entropy(k(AB, A, B)) == 0  # canon: no surprise at S1
    assert entropy(k(AB, AB)) == 0  # identity: no surprise at S1 either
    assert significance(k(AB, A, B)) == 1
    assert entropy(k(C, A, B)) == 1  # S3: no correspondence, full surprise
    assert entropy(Kline(AB, frozenset())) == 1  # S4: 100% surprise


def test_significance_interior_for_s2():
    assert significance(k(AB, A)) == Fraction(1, 2)  # F = A, σ ∨ â = AB
    assert 0 < significance(k(ABC, AB)) < 1


def test_band_grade_correspondence():
    for kk in (k(AB, A, B), k(AB, AB), k(AB, A), k(C, A, B), Kline(AB, frozenset())):
        b = band(kk)
        g = significance(kk)
        assert (b is Band.S1) == (g == 1)
        assert (b is Band.S2) == (0 < g < 1)
        assert (b in (Band.S3, Band.S4)) == (g == 0)  # entropy does not split S3/S4


def test_multiple_canonical_witnesses_coexist():
    """D15 — I-2 withdrawn: C:[AB] and C:[A,B] coexist under one head;
    witness keeps the first-stored and the law holds."""
    d = base_domain(k(AB, AB), k(AB, A, B))
    assert k(AB, AB) in d and k(AB, A, B) in d
    assert d.is_canonical(k(AB, AB)) and d.is_canonical(k(AB, A, B))
    assert d.witness(AB) == frozenset([AB])  # first-stored
    assert can(d.witness(AB)) == AB  # the law


# --- §4 Definition 14: effort — entropy biased by repairability ---


def test_effort_anchors():
    assert effort(k(AB, A, B)) == 0  # S1 — canon
    assert effort(k(AB, AB)) == 0  # S1 — identity
    assert effort(Kline(AB, frozenset())) == 1  # S4 — exactly full, nothing to work with


def test_effort_s3_strictly_below_s4():
    assert effort(k(C, A, B)) < 1  # S3: adjacency present — repairable
    assert effort(k(C, A, B)) == Fraction(1 + BETA * 2, 3)  # (μ(σ) + β·μ(â))/μ(σ ∨ â) = (1+1)/3
    assert effort(Kline(C, frozenset())) == 1  # S4 with the same claim: strictly more


def test_effort_s2_interior_both_kinds():
    assert 0 < effort(k(AB, A)) < 1  # pure underfit: U carries it
    assert 0 < effort(k(A, AB)) < 1  # pure overfit: β·O carries it (β > 0)


def test_effort_beta_interpolation():
    for kk in (k(AB, A), k(C, A, B), k(ABC, AB), k(AB, A, B)):
        assert effort(kk, beta=Fraction(1)) == entropy(kk)  # β → 1 recovers H
        assert effort(kk) <= entropy(kk)  # E ≤ H for β ≤ 1


def test_ends_carry_the_halt_measure():
    d = base_domain(k(AB))  # terminal neighbour
    r = rationalise(d, k(A, A))
    halted = [e for e in r.ends if e.kline is not None]
    assert halted and all(e.effort == effort(e.kline) for e in halted)


# --- §5 Definition 15: proposals at halt ---


def test_proposal_acquire_from_unheld_witness():
    d = base_domain(k(ABD, A, B))  # the R2 trace pushes ⟨B, ∅⟩ (unheld B)
    r = rationalise(d, k(A, A, B))
    ps = [p for p in proposals(r) if p.kind == "acquire" and p.kline == k(B)]
    assert ps and ps[0].effort == 1  # nothing held explains B — full effort


def test_proposal_connect_from_empty_overlap():
    r = rationalise(Domain(), k(AB, A, B))
    ps = proposals(r)
    connect = [p for p in ps if p.kind == "connect"]
    acquires = [p for p in ps if p.kind == "acquire"]
    assert len(connect) == 1 and connect[0].kline == k(AB, A, B)
    # each unrecognised node routes through S4 — an acquire ask apiece (D16)
    assert {p.kline.sig for p in acquires} == {A, B}
    assert all(p.effort == 1 for p in acquires)


def test_proposal_continue_names_min_effort_frontier():
    d = base_domain(k(AB, A, B, C))  # the overfit cycle
    r = rationalise(d, k(A, A, B), max_rounds=1)
    ps = [p for p in proposals(r) if p.kind == "continue"]
    assert ps and ps[0].effort == effort(ps[0].kline)  # the frontier's cheapest member


def test_emptied_expansion_records_its_candidate():
    d = base_domain(k(AB), k(ABC, AB))  # k(AB) is terminal — the only substitutable
    r = rationalise(d, k(C, C))
    ends = [e for e in r.ends if e.reason == "R4: expansion emptied the adjacency"]
    assert ends and ends[0].kline == k(C, AB)  # the candidate whose expansion emptied
    assert ends[0].effort == effort(k(C, AB))


def test_d_is_live_during_the_run():
    """D14 — the run's own writes close routes: the entry's S1 store mints
    ⟨AB, {C}⟩, which enters covered(AB) and keeps the later expansion of
    ⟨C, {AB}⟩ from emptying — the run bound-loops instead of halting.
    Same D₀, entry ⟨C, {C}⟩ (no store intervenes): the emptied halt fires."""
    d0 = [k(A, A), k(B, B), k(C, C), k(AB), k(ABC, AB)]
    r = rationalise(Domain(d0), Kline(C, frozenset([AB])))
    assert k(AB, C) in r.stores  # minted mid-run — the rescuing witness
    assert not any(e.reason == "R4: expansion emptied the adjacency" for e in r.ends)
    r2 = rationalise(Domain(d0), k(C, C))
    assert any(e.reason == "R4: expansion emptied the adjacency" for e in r2.ends)


def test_proposals_cover_every_end():
    d = base_domain(k(AB), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert len(proposals(r)) == len(r.ends)  # one proposal per halt
    assert all(p.kind in ("acquire", "connect", "continue") for p in proposals(r))


def test_proposals_ordered_by_significance():
    # ends at E = 2/3 (emptied expansion ⟨C, {AB}⟩) and E = 1 (⟨C, ∅⟩, no adjacency)
    d = base_domain(k(AB), k(ABC, AB), k(CD))  # k(AB) and k(CD) are terminal
    r = rationalise(d, k(C, C))
    ps = proposals(r)
    assert len(ps) == 2
    assert ps == sorted(ps, key=lambda p: p.effort)  # ascending effort
    assert ps[0].kline == k(C, AB) and ps[0].effort == Fraction(2, 3)  # least entropy first
    assert ps[1].kline == k(C) and ps[1].effort == 1  # the full-surprise acquire sorts last


# --- §5 + D10: R1 stores; admission iff S1 ---


def test_r1_stores_the_work_entrys_first_candidate():
    d = base_domain(k(ABD, A, B))  # held but non-canonical (can = AB)
    r = rationalise(d, k(A, A, B))
    # the entry's first candidate ⟨AB, {A}⟩ is the opener, licensed by its
    # sibling S1 ⟨AB, {A, B}⟩; it does not change for this work entry
    assert r.stores == [k(AB, A)]
    assert k(A, A, B) not in d  # the entry itself is never stored (D10)
    assert not d.is_canonical(k(AB, A))  # a licensed claim, not a canon


def test_r4_candidate_with_no_adjacency():
    d = base_domain(k(AB))  # terminal neighbour
    r = rationalise(d, k(A, A))
    assert any(e.reason.startswith("R4") for e in r.ends)


def test_empty_lookup_terminates_entry():
    d = Domain()  # no neighbours overlap the entry signature
    r = rationalise(d, k(AB, A, B))
    assert any("no candidates" in e.reason for e in r.ends)
    assert r.stores == []


def test_rationalisation_reaches_s1_through_reentry():
    # S3 candidate → R3 product ⟨AB, {A,B}⟩ re-enters as a new work entry:
    # its own first candidate ⟨AB, {A}⟩ is the opener, licensed when a
    # sibling candidate reaches S1 (R1, Definition 9)
    d = base_domain(k(AB, A, B), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert k(AB, A) in r.stores  # the re-entered entry's first candidate
    assert k(AB, A) in d
    assert k(C, AB) not in d  # traces no longer span reentries (amendment 18)
    assert k(AB, AB) not in d  # the S1 endpoint is constructable — not minted


def test_rationalisation_is_bounded_under_cycles():
    d = base_domain(k(AB, A, B, C))  # pure-overfit source: fit product re-derives S2
    r = rationalise(d, k(A, A, B), max_rounds=16)
    assert r.rounds <= 16  # arrival is strategy (§7): the bound holds
