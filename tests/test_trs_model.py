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
AB, BC, CD, ABC, ABD, ABCD = 3, 6, 12, 7, 11, 15


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


# --- §5 Rule R2: same-side products — the fit kline earns, the residual products re-enter ---


def test_r2_reconstruction_same_side_pairing():
    e = k(ABC, A, B, C)  # can(N_E) = ABC
    c = k(ABC, A, B, D)  # â = ABD; F=AB, U=C, O=D — node-clean
    products, k_f = r2_products(c, e)
    # each head carries its own side: U the entry's U-side, O the candidate's O-side
    assert [(p.sig, sorted(p.adj)) for p in products] == [(C, [C]), (D, [D])]
    assert (k_f.sig, sorted(k_f.adj)) == (AB, [A, B])


def test_r2_reconstruction_crossing_node_starves_the_sides():
    # B lives inside the crossing node BC: nothing in N is ⪯ B or ⪯ C —
    # the fit kline starves and O carries no O-side content; U re-pairs
    # with the entry's own A (Definition 12/13)
    e = k(AB, A, B)
    c = k(AB, BC)  # F=B, U=A, O=C
    products, k_f = r2_products(c, e)
    assert [(p.sig, sorted(p.adj)) for p in products] == [(A, [A]), (C, [])]
    assert (k_f.sig, sorted(k_f.adj)) == (B, [])


def test_r2_reconstruction_pure_overfit_skips_empty_head():
    e = k(AB, A, B)
    c = k(AB, A, B, C)  # U = ⊥
    products, k_f = r2_products(c, e)
    assert [(p.sig, sorted(p.adj)) for p in products] == [(C, [C])]
    assert (k_f.sig, sorted(k_f.adj)) == (AB, [A, B])


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
    """Multiple canonical witnesses coexist under one head;
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


def test_proposal_connect_from_empty_overlap():
    # nothing is checked at presentation — the no-neighbours halt is the ask
    r = rationalise(Domain(), k(AB, A, B))
    ps = proposals(r)
    assert len(ps) == 1
    assert ps[0].kind == "connect" and ps[0].kline == k(AB, A, B)


def test_crossing_candidate_stalls_at_the_underfit():
    # the first candidate ⟨ABC,{AB,16}⟩ fits AB, overfits 16, underfits C:
    # the 16 child grounds under its own claim, the AB child merges the
    # grounded 16 (both clear), and the base stalls proposing the
    # grounded-so-far kline with C still asking
    d = Domain([Kline(0b1011, frozenset({AB, D})), Kline(0b10011, frozenset({AB, 16}))])
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores == [Kline(16, frozenset({16})), Kline(AB, frozenset({AB, 16}))]
    assert [(e.reason, e.kline) for e in r.ends] == [
        ("no candidates — the entry terminates", k(C, C)),
        ("stalled at S2: underfit ungrounded", Kline(ABC, frozenset({AB, 16}))),
    ]


def test_disjoint_candidate_asks_s4():
    # ⟨C, {AB}⟩ shares no bit with the claim — F = ∅: the ask is the
    # empty kline ⟨C, ∅⟩, the full-surprise acquire
    d = base_domain(k(AB), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(C) and ends[0].effort == 1


def test_proposals_cover_every_end():
    d = base_domain(k(AB), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert len(proposals(r)) == len(r.ends)  # one proposal per halt
    assert all(p.kind in ("acquire", "connect", "continue") for p in proposals(r))


def test_proposals_ordered_by_significance():
    # ends at E = 1/3 (the stalled frame's grounded-so-far ⟨ABC,{A,B}⟩)
    # and E = 1 (the S4 ask ⟨C, ∅⟩): least entropy sorts first
    d = Domain([k(AB, A, B), k(C)])  # k(C) is terminal
    r = rationalise(d, k(ABC, A, B, C))
    ps = proposals(r)
    assert len(ps) == 2
    assert ps == sorted(ps, key=lambda p: p.effort)  # ascending effort
    assert ps[0].kind == "connect" and ps[0].kline == Kline(ABC, frozenset({A, B})) and ps[0].effort == Fraction(1, 3)
    assert ps[1].kind == "acquire" and ps[1].kline == k(C) and ps[1].effort == 1


# --- §5 R1: stores; admission iff S1 ---


def test_r1_stores_the_s1_candidate():
    d = base_domain(k(ABD, A, B))  # held but non-canonical (can = AB)
    r = rationalise(d, k(A, A, B))
    # the candidate ⟨AB, {A, B}⟩ is itself S1 — stored as presented (Rule R1)
    assert r.stores == [k(AB, A, B)]
    assert k(A, A, B) not in d  # the entry itself is never stored
    assert d.is_canonical(k(AB, A, B))  # a full canon (Definition 5)


def test_terminal_candidate_asks_s4():
    d = base_domain(k(AB))  # terminal neighbour
    r = rationalise(d, k(A, A))
    assert any(e.reason.startswith("S4") for e in r.ends)


def test_empty_lookup_terminates_entry():
    d = Domain()  # no neighbours overlap the entry signature
    r = rationalise(d, k(AB, A, B))
    assert any("no candidates" in e.reason for e in r.ends)
    assert r.stores == []


# --- §5 rationalise: the standardised reentry ---


def test_entry_with_no_adjacency_is_r4():
    d = base_domain()  # presentation admits any kline — checks nothing (Definition 8)
    r = rationalise(d, k(AB))
    assert r.stores == []
    assert [e.reason for e in r.ends] == ["R4: entry kline with no adjacency"]
    assert [p.kind for p in proposals(r)] == ["acquire"]


def test_canon_memo_is_per_run():
    """The canon memo bounds one run's recursion (§7) — a later run of the
    same entry starts fresh: both runs process fully and agree."""

    def fresh():
        return rationalise(base_domain(k(AB, A, B)), k(A, A))

    first, second = fresh(), fresh()
    # both runs ground ⟨A, {A,B}⟩: the B atom grounds the overfit side,
    # the merge folds it in, both sides clear
    assert first.stores == second.stores == [Kline(A, frozenset({A, B}))]
    assert first.ends == second.ends == []  # the memo leaves no residue across runs


def test_s3_grounds_nothing_asks_instead_of_claiming():
    # the ⟨ABC,{AB}⟩ witness shares no bit with the claim C — F = ∅: an
    # acquire ask on the empty kline, not a contentless licensed claim
    d = base_domain(k(AB, A, B), k(ABC, AB))
    r = rationalise(d, k(C, C))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(C)


# --- §5 candidate consolidation ---


def test_the_claim_grounds_when_both_sides_resolve():
    # SVO:[S], SVO:[V], SVO:[O] analog: the popped candidate leaves an
    # underfit region; its reentry chain grounds the atoms bottom-up, the
    # parent merges the grounded region, and the claim stores as the full
    # canon — children complete before the parent's revisit
    d = Domain([k(AB, A, B), k(BC, B, C)])
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores[-1] == k(ABC, A, B, C)
    assert r.ends == []
    assert d.is_canonical(k(ABC, A, B, C))


def test_identities_witness_nothing():
    # over a pure atom substrate every neighbour is an identity — the
    # entry cannot ground through its own reflexive decomposition; it asks
    d = base_domain()
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores == []
    assert [e.reason for e in r.ends] == ["no candidates — the entry terminates"]
    ps = proposals(r)
    assert ps[0].kind == "connect" and ps[0].kline == k(ABC, A, B, C)


def test_aggregation_unions_witnesses_from_many_klines():
    # ⟨A,{A}⟩ and ⟨ABD,{A,B}⟩ both fully covered — the aggregate carries
    # the union {A,B} under the canon AB: the whole is S1 in one step
    d = base_domain(k(ABD, A, B))
    r = rationalise(d, k(AB, A, B))
    assert r.stores == [k(AB, A, B)]


def test_partial_fit_never_stores_alone_the_frame_stalls():
    # only the AB canon witnessed: the frame's f = {A,B}, u = {C}; the C
    # reentry meets nobody and asks, then the frame stalls proposing the
    # grounded-so-far kline — the fit itself never stores
    d = Domain([k(AB, A, B)])
    r = rationalise(d, k(ABC, A, B, C))
    assert r.stores == []
    ps = proposals(r)
    assert [(p.kind, p.kline) for p in ps] == [
        ("connect", k(C, C)),
        ("connect", Kline(ABC, frozenset({A, B}))),
    ]


def test_the_unpopped_candidate_never_runs():
    # the ⟨ABC,{CD}⟩ witness would overfit — but only the list's first is
    # popped: the covered ⟨ABD,{A,B}⟩ candidate grounds the claim and the
    # rest of the list is never work
    d = base_domain(k(CD, CD), k(ABC, CD), k(ABD, A, B))
    r = rationalise(d, k(AB, A, B))
    assert r.stores == [k(AB, A, B)]
    assert r.ends == []
    assert d.is_canonical(k(AB, A, B))


def test_terminal_neighbours_consolidate_into_one_ask():
    # two terminal witnesses overlap the canon — one popped candidate, one
    # S4 ask on ⟨A, ∅⟩, not one per terminal neighbour
    d = base_domain(k(AB), k(A | C))
    r = rationalise(d, k(A, A))
    halts = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert len(halts) == 1 and halts[0].kline == k(A)


def test_disjoint_overfit_asks_s4():
    # the popped candidate ⟨AB, {D}⟩ shares no bit with the claim — F = ∅:
    # the empty kline asks; no substitution runs at reentry anymore
    d = base_domain(k(C, D), k(D, A, B))
    r = rationalise(d, k(C, A, B))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(AB)


def test_witness_node_never_a_head_asks_s4():
    # the witness node C is never a held head — the candidate ⟨AB,{C}⟩
    # fits nothing: the empty kline asks, no proof substitution
    d = Domain([Kline(A | C, frozenset({C}))])
    r = rationalise(d, k(AB, A, B))
    assert r.stores == []
    ends = [e for e in r.ends if e.reason == "S4: candidate with no fit"]
    assert ends and ends[0].kline == k(AB)


def test_crossing_node_witnesses_nothing():
    # the crossing node BC lies in neither the fit nor the overfit region
    # — n_f and n_o stay empty; the grounded A side merges, both clear,
    # and the claim grounds carrying only A
    d = base_domain(k(ABC, BC))
    r = rationalise(d, k(AB, A, B))
    assert r.stores == [Kline(AB, frozenset({A}))]
    assert r.ends == []

