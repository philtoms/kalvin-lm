"""TRS model — tests against docs/trs-significance.md."""

from __future__ import annotations

import pytest

from trs import (
    Band,
    Domain,
    Kline,
    band,
    can,
    decompose,
    rationalise,
    r2_products,
    r3_substitute,
)

A, B, C, D = 1, 2, 4, 8
AB, CD, ABC, ABD, ABCD = 3, 12, 7, 11, 15


def k(sig: int, *adj: int) -> Kline:
    return Kline(sig, frozenset(adj))


def base_domain(*extra: Kline) -> Domain:
    """Single atoms held canonically — the I-1 substrate."""
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


def test_i1_corollary_witnessed_adjacencies_are_held_signatures():
    d = base_domain(k(AB, A, B), k(ABCD, AB, C, D))
    sigs = {kk.sig for kk in d}
    assert d.i1_closed()
    for s in (AB, ABCD):
        assert all(n in sigs for n in d.witness(s))


def test_i1_detects_violation():
    d = Domain([k(A, B)])  # B is not a signature in D
    assert not d.i1_closed()


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


# --- §5 + D10: R1 stores; admission iff S1 ---


def test_r1_stores_opening_candidate_first_round():
    d = base_domain(k(ABD, A, B))  # held but non-canonical (can = AB)
    r = rationalise(d, k(A, A, B))
    assert k(AB, A, B) in r.stores  # first round: opener == the S1 candidate (canonical)
    assert k(AB, A) in r.stores  # the S2 opener ⟨AB, {A}⟩, licensed by its trace's S1
    assert k(A, A, B) not in d  # the entry itself is never stored (D10)
    assert not any(d.is_canonical(s) for s in r.stores if s == k(AB, A))  # licensed, not canonical


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
    # S3 candidate ⟨C, {AB}⟩ → R3 product ⟨AB, {A,B}⟩ re-enters → S1:
    # the proof licenses the opening candidate, not the S1 candidate (R1)
    d = base_domain(k(AB, A, B), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert k(C, AB) in r.stores  # the opening candidate is stored
    assert k(C, AB) in d
    assert k(AB, AB) not in d  # constructable from the opener — not stored


def test_rationalisation_is_bounded_under_cycles():
    d = base_domain(k(AB, A, B, C))  # pure-overfit source: fit product re-derives S2
    r = rationalise(d, k(A, A, B), max_rounds=16)
    assert r.rounds <= 16  # arrival is strategy (§7): the bound holds
